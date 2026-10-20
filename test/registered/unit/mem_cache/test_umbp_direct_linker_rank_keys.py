"""Unit tests for the UMBP direct linker's rank-replicated key handling."""

import ctypes
import unittest
from queue import Queue
from types import SimpleNamespace
from unittest import mock

import torch

import sglang.srt.mem_cache.storage.umbp.umbp_direct_linker as umbp_direct_linker
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolEntry,
    _build_dsa_device_pool_group,
)
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
        # The draft buffer follows the group's target buffers, as in the object.
        self.assertEqual(
            linker._layer_group_ranges(plan, [3, 5]),
            (
                [
                    [
                        buffers[0][1].data_ptr(),
                        buffers[1][1].data_ptr(),
                        buffers[2][1].data_ptr(),
                    ]
                ],
                [[3, 5, 7]],
                [[0, 3, 8]],
            ),
        )
        entry.layer_mapping = {3: (0, 1), 5: 1}
        with self.assertRaisesRegex(ValueError, "maps buffer 1 more than once"):
            _ordered_layers(entry)

    def test_write_ranges_tile_every_object_in_order(self):
        """The store takes a put only if its ranges tile the object exactly."""
        k = [torch.zeros((4, 3), dtype=torch.uint8) for _ in range(3)]
        v = [torch.zeros((4, 5), dtype=torch.uint8) for _ in range(3)]
        for components, layer_mapping, packed in (
            # As _mha_device_pool_entry builds it: one packed component [*k, *v].
            ([[*k[:2], *v[:2]]], {0: (0, 2), 1: (1, 3)}, True),
            # One object per component, with a draft packed onto layer 0.
            ([k, v], {0: (0, 2), 1: 1}, False),
        ):
            entry = DevicePoolEntry(
                name=PoolName.DRAFT,
                indices_from_pool=PoolName.KV,
                device_pool=None,
                components=components,
                layer_mapping=layer_mapping,
                page_size=2,
                rows_are_pages=False,
                packed=packed,
            )
            linker = UMBPDirectLinker.__new__(UMBPDirectLinker)
            linker.pools = {entry.name: entry}
            linker.pool_layers = {entry.name: _ordered_layers(entry)}
            per_page = 1 if packed else len(components)
            plan = _PoolRangePlan(entry.name, ["key"] * 2 * per_page, [0, 2], per_page)
            ptrs, sizes, offsets = linker._all_layer_ranges(plan)
            object_sizes = umbp_direct_linker._object_sizes_per_page(entry)
            with self.subTest(packed=packed):
                self.assertEqual(len(ptrs), 2 * per_page)
                # The first object walks its component's buffers in object order.
                self.assertEqual(
                    ptrs[0], [buffer[0].data_ptr() for buffer in components[0]]
                )
                for index, (size, offset) in enumerate(zip(sizes, offsets)):
                    self.assertEqual(offset, [sum(size[:i]) for i in range(len(size))])
                    self.assertEqual(sum(size), object_sizes[index % per_page])


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


class TestPackedLayerRoundTrip(CustomTestCase):
    """Offload, look up and load packed layers through the linker's own paths."""

    def _round_trip(self, group, pages):
        storage = UMBPStore.__new__(UMBPStore)
        storage.client = _RangedStore()
        storage.registered_pools = group.entry_map
        storage.mla_suffix = storage.mha_suffix = "tp0_cp0_pp0"
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

        tokens = torch.arange(pages * group.page_size)
        keys = [f"page{i}" for i in range(pages)]
        transfers = [PoolTransfer(name=PoolName.KV, keys=keys, device_indices=tokens)]
        ready = SimpleNamespace(synchronize=lambda: None)
        expanded = group.resolve_transfers(transfers, allow_partial=True)
        self.assertTrue(linker._run_offload([(expanded, ready)]))
        self.assertEqual(linker.lookup("rid", transfers), list(range(1, pages + 1)))

        buffers = [
            buffer
            for entry in group.entries
            for component in entry.components
            for buffer in component
        ]
        stored = [buffer.clone() for buffer in buffers]
        for buffer in buffers:
            buffer.zero_()
        self.assertTrue(linker.load("rid", transfers))
        plans = linker._build_load_plans(list(linker._pending.values()))
        linker._run_layer_wise_batch(0, plans, ready)
        linker.layer_done_counter.fail.assert_not_called()
        for buffer, want in zip(buffers, stored):
            self.assertTrue(torch.equal(buffer, want))

    def test_dsa_target_and_mtp_draft(self):
        page_size, layers, pages = 2, 3, 4

        def pool(layer_count, rows, width):
            return [
                torch.randint(0, 256, (rows, width), dtype=torch.uint8)
                for _ in range(layer_count)
            ]

        def kvcache(layer_count):
            return SimpleNamespace(
                page_size=page_size,
                layer_num=layers,
                kv_buffer=pool(layer_count, pages * page_size, 6),
                index_k_with_scale_buffer=pool(layer_count, pages, 5),
            )

        group = _build_dsa_device_pool_group(kvcache(layers), page_size, (kvcache(1),))
        self._round_trip(group, pages)


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
