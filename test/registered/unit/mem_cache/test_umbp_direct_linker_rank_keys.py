"""Unit tests for the UMBP direct linker's rank-replicated key handling."""

import ctypes
import unittest
from queue import Queue
from types import SimpleNamespace
from unittest import mock

import torch

import sglang.srt.mem_cache.storage.umbp.umbp_direct_linker as umbp_direct_linker
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache import linker_pool_assembler
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import DevicePoolEntry
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, MLATokenToKVPool
from sglang.srt.mem_cache.storage.mooncake_store import mooncake_direct_linker
from sglang.srt.mem_cache.storage.umbp.umbp_direct_linker import (
    UMBPDirectLinker,
    _ordered_layers,
    _PoolRangePlan,
    _storage_suffix,
)
from sglang.srt.mem_cache.storage.umbp.umbp_store import UMBPStore
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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

    def test_dcp_ranks_hold_their_own_objects(self):
        # A DCP rank holds every dcp_size-th token of a page, so replicas share
        # an object only within one DCP rank and one DCP size.
        replicated = {
            _storage_suffix(
                rank_replicated=True,
                tp_rank=rank,
                attn_cp_rank=0,
                pp_rank=0,
                dcp_rank=rank % 4,
                dcp_size=4,
            )
            for rank in range(8)
        }
        self.assertEqual(replicated, {f"dcp{rank}of4_cp0_pp0" for rank in range(4)})
        self.assertEqual(
            _storage_suffix(
                rank_replicated=False,
                tp_rank=3,
                attn_cp_rank=0,
                pp_rank=0,
                dcp_rank=3,
                dcp_size=8,
            ),
            "tp3_dcp3of8_cp0_pp0",
        )

    def test_mooncake_names_the_same_objects(self):
        for replicated in (False, True):
            for dcp_rank, dcp_size in ((0, 1), (1, 2)):
                kwargs = dict(
                    rank_replicated=replicated,
                    tp_rank=1,
                    attn_cp_rank=0,
                    pp_rank=0,
                    dcp_rank=dcp_rank,
                    dcp_size=dcp_size,
                )
                self.assertEqual(
                    mooncake_direct_linker._storage_suffix(**kwargs),
                    _storage_suffix(**kwargs),
                )


class _RangedStore:
    """In-memory UMBP client: like mori, a put has to tile its object exactly,
    and a put to a key that is already stored succeeds without a write."""

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
            if cursor != object_size or key in self.objects:
                continue
            data = self.objects[key] = bytearray(object_size)
            for ptr, length, offset in zip(range_ptrs, range_sizes, range_offsets):
                ctypes.memmove(
                    (ctypes.c_char * length).from_buffer(data, offset), ptr, length
                )
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
                ctypes.memmove(
                    ptr, (ctypes.c_char * length).from_buffer(data, offset), length
                )
        return results

    def batch_exists(self, keys):
        return [key in self.objects for key in keys]


class TestDcpRoundTrip(CustomTestCase):
    """Each DCP rank stores and restores its own shard of every tree page."""

    LAYERS, PAGES = 2, 3

    def _rank_linker(self, store, page_size, dcp_size, dcp_rank):
        kvcache = HybridLinearKVPool.__new__(HybridLinearKVPool)
        kvcache.start_layer = 0
        kvcache.full_attention_layer_id_mapping = {
            layer: layer for layer in range(self.LAYERS)
        }
        kvcache.full_kv_pool = MLATokenToKVPool.__new__(MLATokenToKVPool)
        kvcache.full_kv_pool.page_size = page_size
        kvcache.full_kv_pool.kv_buffer = [
            torch.randint(0, 256, (self.PAGES * page_size, 1, 6), dtype=torch.uint8)
            for _ in range(self.LAYERS)
        ]
        mamba_cache = SimpleNamespace(
            temporal=torch.zeros((1, 2, 3), dtype=torch.uint8), conv=[]
        )
        params = SimpleNamespace(
            mtp_draft_device_pools=(),
            sidecar_draft_device_pools=(),
            req_to_token_pool=SimpleNamespace(
                mamba_pool=SimpleNamespace(mamba_cache=mamba_cache, num_mamba_layers=1),
                mamba_map={self.LAYERS: 0},
                mamba_ckpt_pool=None,
                short_conv_pool=SimpleNamespace(enabled=False),
                ngram_pool=SimpleNamespace(enabled=False),
            ),
        )
        with (
            mock.patch.object(
                linker_pool_assembler,
                "get_memory",
                return_value=SimpleNamespace(enable_unified_memory=False),
            ),
            mock.patch.object(
                linker_pool_assembler,
                "get_parallel",
                return_value=SimpleNamespace(
                    attn_dcp_size=dcp_size, attn_dcp_rank=dcp_rank
                ),
            ),
        ):
            group = linker_pool_assembler._build_mamba_device_pool_group(
                kvcache, params, page_size * dcp_size
            )

        storage = UMBPStore.__new__(UMBPStore)
        storage.client = store
        storage.registered_pools = group.entry_map
        storage.mla_suffix = storage.mha_suffix = _storage_suffix(
            rank_replicated=False,
            tp_rank=dcp_rank,
            attn_cp_rank=0,
            pp_rank=0,
            dcp_rank=group.dcp_rank,
            dcp_size=group.dcp_size,
        )
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
        linker.layer_group = 1
        linker.layer_done_counter = mock.Mock()
        linker._stats = dict.fromkeys(
            ("lookup", "load", "offload", "offload_batches"), 0
        )
        linker._pending = {}
        linker._async_offload_index_snapshot = False
        return linker, kvcache.full_kv_pool.kv_buffer

    def test_every_dcp_rank_restores_its_own_shard(self):
        ready = SimpleNamespace(synchronize=lambda: None)
        # One store, and the same tree page (so the same page keys) at every
        # DCP size: only the DCP part of the suffix keeps the shards apart.
        store = _RangedStore()
        for page_size, dcp_size in ((1, 4), (2, 2), (4, 1)):
            transfers = [
                PoolTransfer(
                    name=PoolName.KV,
                    keys=[f"page{page}" for page in range(self.PAGES)],
                    device_indices=torch.arange(self.PAGES * page_size * dcp_size),
                )
            ]
            ranks = [
                self._rank_linker(store, page_size, dcp_size, dcp_rank)
                for dcp_rank in range(dcp_size)
            ]
            for linker, _ in ranks:
                expanded = linker.pool_group.resolve_transfers(
                    transfers, allow_partial=True
                )
                self.assertTrue(linker._run_offload([(expanded, ready)]))
            for linker, buffers in ranks:
                stored = [buffer.clone() for buffer in buffers]
                for buffer in buffers:
                    buffer.zero_()
                self.assertTrue(linker.load("rid", transfers))
                plans = linker._build_load_plans(list(linker._pending.values()))
                linker._pending.clear()
                linker._run_layer_wise_batch(0, plans, ready)
                linker.layer_done_counter.fail.assert_not_called()
                for buffer, want in zip(buffers, stored):
                    self.assertTrue(torch.equal(buffer, want))


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
        with mock.patch.object(umbp_direct_linker, "device_module", fake_device):
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


if __name__ == "__main__":
    unittest.main()
