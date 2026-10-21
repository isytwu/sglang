"""Unit tests for the UMBP direct linker's rank-replicated key handling."""

import unittest
from queue import Queue
from types import SimpleNamespace
from unittest import mock

import torch

import sglang.srt.mem_cache.storage.umbp.umbp_direct_linker as umbp_direct_linker
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
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


class TestTouch(CustomTestCase):
    def _linker(self, batch_touch):
        pools = {
            name: SimpleNamespace(packed=True, components=[[]])
            for name in (PoolName.KV, PoolName.MAMBA)
        }
        storage = UMBPStore.__new__(UMBPStore)
        storage.registered_pools = pools
        storage.mla_suffix = storage.mha_suffix = "tp0_cp0_pp0"
        storage.config_prefix = None
        storage.client = SimpleNamespace(batch_touch=batch_touch)
        linker = UMBPDirectLinker.__new__(UMBPDirectLinker)
        linker.storage = storage
        linker.pools = pools
        linker.num_layers = 1
        linker.pool_group = SimpleNamespace(
            resolve_transfers=lambda transfers, allow_partial: transfers
        )
        linker._stats = dict.fromkeys(
            ("touch_keys", "touched", "touch_failures", "touch_dropped"), 0
        )
        linker._touch_queue = Queue()
        return linker

    @staticmethod
    def _touch(pages):
        return [
            PoolTransfer(name=PoolName.KV, keys=pages),
            PoolTransfer(
                name=PoolName.MAMBA,
                keys=pages[-1:],
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            ),
        ]

    def test_queued_touches_go_out_once_tail_first(self):
        sent = []
        linker = self._linker(lambda keys: sent.append(list(keys)) or len(keys))
        # Queued behind a slow RPC: coalesced, and the shared head goes last.
        linker.touch(self._touch(["p0", "p1"]))
        linker.touch(self._touch(["p0", "p1", "p2"]))
        linker._touch_queue.put(None)
        linker._touch_thread_func()

        self.assertEqual(
            sent,
            [
                [
                    "p1_tp0_cp0_pp0_temporal",
                    "p2_tp0_cp0_pp0_kv",
                    "p1_tp0_cp0_pp0_kv",
                    "p0_tp0_cp0_pp0_kv",
                    "p2_tp0_cp0_pp0_temporal",
                ]
            ],
        )
        self.assertEqual(linker._stats["touched"], 5)
        self.assertEqual(linker._touch_queue.unfinished_tasks, 0)

    def test_a_failing_store_is_counted_not_fatal(self):
        def batch_touch(keys):
            raise RuntimeError("store down")

        linker = self._linker(batch_touch)
        linker.touch(self._touch(["p0"]))
        linker._touch_queue.put(None)
        linker._touch_thread_func()
        self.assertEqual(linker._stats["touch_failures"], 1)
        self.assertEqual(linker._touch_queue.unfinished_tasks, 0)


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
