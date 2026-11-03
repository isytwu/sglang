"""Unit tests for the UMBP direct linker's rank-replicated key handling."""

import ctypes
import unittest
from queue import Queue
from types import SimpleNamespace
from unittest import mock

import torch

import sglang.srt.mem_cache.unified_cache.unified_cache_linker as unified_cache_linker
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache import linker_pool_assembler
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import DevicePoolEntry
from sglang.srt.mem_cache.storage.umbp.umbp_direct_linker import (
    UMBPDirectLinker,
    _ordered_layers,
    _PoolRangePlan,
    _storage_suffix,
)
from sglang.srt.mem_cache.storage.umbp.umbp_store import UMBPStore
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestStorageSuffix(CustomTestCase):
    def test_replicated_group_drops_the_tp_term(self):
        # MLA/DSA pages are byte-identical on every attention TP rank, so every
        # rank has to name the same object or the store holds tp_size copies.
        suffixes = {
            _storage_suffix(
                rank_replicated=True, tp_rank=rank, attn_cp_rank=0, pp_rank=0
            )
            for rank in range(8)
        }
        self.assertEqual(suffixes, {"cp0_pp0"})

    def test_sharded_group_keeps_one_keyspace_per_rank(self):
        suffixes = [
            _storage_suffix(
                rank_replicated=False, tp_rank=rank, attn_cp_rank=0, pp_rank=0
            )
            for rank in range(8)
        ]
        self.assertEqual(len(set(suffixes)), 8)
        self.assertEqual(suffixes[3], "tp3_cp0_pp0")

    def test_cp_and_pp_shard_even_when_replicated(self):
        # CP shards the sequence and PP shards layers, so neither may collapse.
        self.assertEqual(
            _storage_suffix(rank_replicated=True, tp_rank=0, attn_cp_rank=1, pp_rank=2),
            "cp1_pp2",
        )
        self.assertNotEqual(
            _storage_suffix(rank_replicated=True, tp_rank=0, attn_cp_rank=0, pp_rank=0),
            _storage_suffix(rank_replicated=True, tp_rank=0, attn_cp_rank=1, pp_rank=0),
        )


class TestPackedDraftRanges(CustomTestCase):
    def test_target_and_draft_share_a_transfer_layer(self):
        buffers = [torch.zeros((4, width), dtype=torch.uint8) for width in (3, 5, 7)]
        entry = DevicePoolEntry(
            name=PoolName.SWA,
            indices_from_pool=PoolName.SWA,
            device_pool=None,
            components=[buffers],
            layer_mapping={3: (0, 2), 5: 1},
            page_size=2,
            rows_are_pages=True,
        )
        self.assertEqual(_ordered_layers(entry), [3, 5])
        linker = UMBPDirectLinker.__new__(UMBPDirectLinker)
        linker.pools = {PoolName.SWA: entry}
        plan = _PoolRangePlan(PoolName.SWA, ["page"], [1], 1)
        self.assertEqual(
            linker._layer_group_ranges(plan, [3]),
            (
                [[buffers[0][1].data_ptr(), buffers[2][1].data_ptr()]],
                [[3, 7]],
                [[0, 8]],
            ),
        )
        self.assertEqual(
            linker._layer_group_ranges(plan, [3, 5]),
            (
                [
                    [
                        buffers[0][1].data_ptr(),
                        buffers[2][1].data_ptr(),
                        buffers[1][1].data_ptr(),
                    ]
                ],
                [[3, 7, 5]],
                [[0, 8, 3]],
            ),
        )
        entry.layer_mapping = {3: (0, 1), 5: 1}
        with self.assertRaisesRegex(ValueError, "maps buffer 1 more than once"):
            _ordered_layers(entry)


class TestEveryRankWrites(CustomTestCase):
    """Collapsing the key must not also elide the write on non-zero ranks.

    A Local-mode UMBP tier is in-process and private to its rank, and a
    standalone server is per node, so a page written only by rank 0 is not
    reachable from the other ranks' stores. They then miss on a key nobody
    wrote for them, and the attention group's MIN over the restorable mask
    drives the hit to zero.
    """

    def test_offload_queues_the_write_on_every_rank(self):
        linker = UMBPDirectLinker.__new__(UMBPDirectLinker)
        linker._offload_results = Queue()
        linker._offload_queue = Queue()
        linker._gc_frozen = True  # short-circuits _freeze_gc_once
        linker.pool_group = SimpleNamespace(
            resolve_transfers=lambda _transfers, allow_partial: ["transfer"]
        )
        fake_device = SimpleNamespace(
            Event=lambda: SimpleNamespace(record=lambda: None)
        )
        with mock.patch.object(unified_cache_linker, "device_module", fake_device):
            self.assertTrue(linker.offload(["t"]))
        self.assertEqual(linker._offload_queue.qsize(), 1)
        # The result comes from the offload thread once the write lands, so
        # nothing may pre-post one here.
        self.assertEqual(linker._offload_results.qsize(), 0)

    def test_empty_transfer_set_produces_no_result(self):
        linker = UMBPDirectLinker.__new__(UMBPDirectLinker)
        linker._offload_results = Queue()
        linker._offload_queue = Queue()
        linker._gc_frozen = True
        linker.pool_group = SimpleNamespace(
            resolve_transfers=lambda _transfers, allow_partial: []
        )
        self.assertFalse(linker.offload([]))
        self.assertEqual(linker._offload_results.qsize(), 0)
        self.assertEqual(linker._offload_queue.qsize(), 0)


class TestLoadReadyEvents(CustomTestCase):
    def test_load_thread_waits_on_every_event(self):
        linker = UMBPDirectLinker.__new__(UMBPDirectLinker)
        order = mock.MagicMock()
        linker.record_transfer_ready = order.record_transfer_ready
        order.record_transfer_ready.return_value = unified_cache_linker.TransferReady(
            [order.indices_ready, order.forward_done]
        )
        linker._gc_frozen = True
        linker._pending = {"rid": ["transfer"]}
        linker._build_load_plans = lambda _transfers: []
        linker.layer_done_counter = mock.MagicMock()
        linker.layer_done_counter.update_producer.return_value = 3
        linker._stats = {"load": 0}
        linker._load_queue = Queue()
        # No layer group to read, so only the waits run.
        linker.num_layers = 0
        linker.layer_group = 1
        linker.pool_layers = {}
        self.assertEqual(linker.start_layer_wise_loading(), 3)
        linker._load_queue.put(None)
        linker._load_thread_func()
        self.assertEqual(
            order.mock_calls,
            [
                mock.call.record_transfer_ready(),
                mock.call.indices_ready.synchronize(),
                mock.call.forward_done.synchronize(),
            ],
        )
        linker.layer_done_counter.fail.assert_not_called()


class _RangedStore:
    """In-memory UMBP client. Like mori, a put has to tile its object exactly."""

    def __init__(self):
        self.objects: dict[str, bytearray] = {}

    def batch_put_ranges_from_ptr(self, keys, object_sizes, ptrs, sizes, offsets):
        results = []
        for key, object_size, range_ptrs, range_sizes, range_offsets in zip(
            keys, object_sizes, ptrs, sizes, offsets
        ):
            cursor = 0
            for offset, length in sorted(zip(range_offsets, range_sizes)):
                if offset != cursor:
                    break
                cursor += length
            results.append(cursor == object_size)
            if cursor != object_size:
                continue
            data = self.objects.setdefault(key, bytearray(object_size))
            for ptr, length, offset in zip(range_ptrs, range_sizes, range_offsets):
                view = (ctypes.c_char * length).from_buffer(data, offset)
                ctypes.memmove(view, ptr, length)
        return results

    def batch_get_ranges_into_ptr(self, keys, ptrs, sizes, offsets):
        results = []
        for key, range_ptrs, range_sizes, range_offsets in zip(
            keys, ptrs, sizes, offsets
        ):
            data = self.objects.get(key)
            results.append(data is not None)
            if data is None:
                continue
            for ptr, length, offset in zip(range_ptrs, range_sizes, range_offsets):
                view = (ctypes.c_char * length).from_buffer(data, offset)
                ctypes.memmove(ptr, view, length)
        return results

    def batch_exists(self, keys):
        return [key in self.objects for key in keys]


class TestFullSwaRoundTrip(CustomTestCase):
    """Offload, look up and load a FULL + SWA (Muse Spark) page group.

    Full-attention K/V cover every prefix page; the SWA pool only keeps the
    trailing window, so a restore needs just those pages from it.
    """

    def _group(self, page_size, full_pages, swa_pages):
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        def pool(layers, pages):
            mha = MHATokenToKVPool.__new__(MHATokenToKVPool)
            mha.page_size, mha.kv_cache_layout = page_size, "nhd"
            mha.k_scale_buffer = mha.native_k_scale_buffer = None
            mha.k_buffer = [
                torch.randint(0, 256, (pages * page_size, 2, 3), dtype=torch.uint8)
                for _ in range(layers)
            ]
            mha.v_buffer = [
                torch.randint(0, 256, (pages * page_size, 2, 5), dtype=torch.uint8)
                for _ in range(layers)
            ]
            return mha

        kvcache = SWAKVPool.__new__(SWAKVPool)
        kvcache.start_layer = 0
        kvcache.full_kv_pool = pool(2, full_pages)
        kvcache.swa_kv_pool = pool(6, swa_pages)
        kvcache.layers_mapping = {
            **{layer: (index, False) for index, layer in enumerate([1, 5])},
            **{layer: (index, True) for index, layer in enumerate([0, 2, 3, 4, 6, 7])},
        }
        return linker_pool_assembler._build_swa_device_pool_group(
            kvcache, SimpleNamespace(mtp_draft_device_pools=()), page_size
        )

    def _linker(self, group):
        storage = UMBPStore.__new__(UMBPStore)
        storage.client = _RangedStore()
        storage.registered_pools = group.entry_map
        storage.mla_suffix = storage.mha_suffix = "tp3_cp0_pp0"
        storage.is_mla_backend = True
        storage.config_prefix = None
        linker = UMBPDirectLinker.__new__(UMBPDirectLinker)
        linker.storage = storage
        linker.pool_group = group
        linker.pools = group.entry_map
        linker.num_layers = group.num_layers
        linker.pool_layers = {
            name: _ordered_layers(entry) for name, entry in group.entry_map.items()
        }
        linker.layer_group = 2
        linker.layer_done_counter = mock.Mock()
        linker._stats = {"lookup": 0, "load": 0, "offload": 0, "offload_batches": 0}
        linker._pending = {}
        linker._async_offload_index_snapshot = False
        return linker

    def test_full_kv_and_swa_window_round_trip(self):
        group = self._group(page_size=2, full_pages=8, swa_pages=4)
        linker = self._linker(group)
        ready = SimpleNamespace(synchronize=lambda: None)
        keys = [f"page{i}" for i in range(3)]
        # Full pages 1-3 hold the 3-page prefix; SWA pages 0 and 2 hold its
        # 2-page window.
        offload = [
            PoolTransfer(
                name=PoolName.KV,
                keys=keys,
                device_indices=torch.tensor([2, 3, 4, 5, 6, 7]),
            ),
            PoolTransfer(
                name=PoolName.SWA,
                keys=keys[1:],
                device_indices=torch.tensor([0, 1, 4, 5]),
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            ),
        ]
        expanded = group.resolve_transfers(offload, allow_partial=True)
        self.assertTrue(linker._run_offload([(expanded, ready)]))
        self.assertEqual(
            sorted(linker.storage.client.objects),
            sorted(
                [f"{key}_tp3_cp0_pp0_{PoolName.KV}" for key in keys]
                + [f"{key}_tp3_cp0_pp0_{PoolName.SWA}" for key in keys[1:]]
            ),
        )

        def lookup(num_pages, window):
            return linker.lookup(
                "rid",
                [
                    PoolTransfer(name=PoolName.KV, keys=keys[:num_pages]),
                    PoolTransfer(
                        name=PoolName.SWA,
                        keys=keys[max(0, num_pages - window) : num_pages],
                        hit_policy=PoolHitPolicy.TRAILING_PAGES,
                    ),
                ],
            )

        self.assertEqual(lookup(3, 2), [3])
        # A 2-page prefix's window starts at page 0, whose SWA was never stored.
        self.assertEqual(lookup(2, 2), [])

        kv, swa = group.entry_map[PoolName.KV], group.entry_map[PoolName.SWA]
        before_kv = [buffer[2:8].clone() for buffer in kv.components[0]]
        before_swa = [buffer[[0, 1, 4, 5]].clone() for buffer in swa.components[0]]
        load = [
            PoolTransfer(
                name=PoolName.KV,
                keys=keys,
                device_indices=torch.tensor([10, 11, 12, 13, 14, 15]),
            ),
            PoolTransfer(
                name=PoolName.SWA,
                keys=keys[1:],
                device_indices=torch.tensor([2, 3, 6, 7]),
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            ),
        ]
        self.assertTrue(linker.load("rid", load))
        plans = linker._build_load_plans(list(linker._pending.values()))
        linker._run_layer_wise_batch(0, plans, ready)
        linker.layer_done_counter.fail.assert_not_called()
        self.assertEqual(
            sorted(
                c.args[1] for c in linker.layer_done_counter.complete.call_args_list
            ),
            list(range(8)),
        )

        for got, want in zip([buffer[10:16] for buffer in kv.components[0]], before_kv):
            self.assertTrue(torch.equal(got, want))
        for got, want in zip(
            [buffer[[2, 3, 6, 7]] for buffer in swa.components[0]], before_swa
        ):
            self.assertTrue(torch.equal(got, want))


if __name__ == "__main__":
    unittest.main()
