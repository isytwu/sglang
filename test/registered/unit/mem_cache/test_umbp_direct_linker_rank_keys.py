"""Unit tests for the UMBP direct linker's rank-replicated key handling."""

import unittest
from queue import Queue
from types import SimpleNamespace
from unittest import mock

import torch

import sglang.srt.mem_cache.unified_cache.unified_cache_linker as unified_cache_linker
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import DevicePoolEntry
from sglang.srt.mem_cache.storage.umbp.umbp_direct_linker import (
    UMBPDirectLinker,
    _ordered_layers,
    _PoolRangePlan,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


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


if __name__ == "__main__":
    unittest.main()
