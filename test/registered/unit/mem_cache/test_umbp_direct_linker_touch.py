"""Unit tests for the UMBP direct linker's recency hints (on_device_hit)."""

import collections
import threading
import time
import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage.umbp.umbp_direct_linker import UMBPDirectLinker
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE = 4


class _FakeClient:
    def __init__(self):
        self.calls = []
        self.fail = False

    def batch_touch(self, keys):
        if self.fail:
            raise RuntimeError("rpc down")
        self.calls.append(list(keys))
        return len(keys)


class _InlineExecutor:
    def submit(self, fn, *args):
        fn(*args)


def _linker(client, interval=30.0):
    linker = UMBPDirectLinker.__new__(UMBPDirectLinker)
    linker._touch_interval = interval
    linker._closed = False
    linker._touched_at = collections.OrderedDict()
    linker._touch_busy = False
    linker._touch_lock = threading.Lock()
    linker._touch_diag = collections.Counter()
    linker._touch_diag_last = time.monotonic()
    linker._touch_executor = _InlineExecutor()
    linker.page_size = PAGE
    linker.storage = SimpleNamespace(client=client)
    linker.pool_group = SimpleNamespace(resolve_transfers=lambda transfers: transfers)
    linker._object_keys_for_pages = lambda pages, transfer: (
        [f"{page}_{transfer.name.value}" for page in pages],
        1,
    )
    return linker


def _lookup_transfers(hashes):
    return [
        SimpleNamespace(name=PoolName.KV, keys=list(hashes)),
        SimpleNamespace(name=PoolName.MAMBA, keys=list(hashes[-1:])),
    ]


class TestRecencyHints(CustomTestCase):
    def setUp(self):
        self.client = _FakeClient()
        self.linker = _linker(self.client)
        self.key = RadixKey(list(range(4 * PAGE + 3)))

    def test_a_hint_names_every_kv_page_and_only_the_mamba_boundary(self):
        self.linker.on_device_hit("r", self.key, 4 * PAGE + 3, 7, _lookup_transfers)

        (sent,) = self.client.calls
        kv = [key for key in sent if key.endswith("_kv")]
        mamba = [key for key in sent if key.endswith("_mamba")]
        # The partial page is not stored, so it is not hinted either.
        self.assertEqual(len(kv), 4)
        self.assertEqual(mamba, [kv[-1].replace("_kv", "_mamba")])

    def test_a_shorter_hit_hints_the_same_chain_from_the_root(self):
        self.linker.on_device_hit("r1", self.key, 4 * PAGE, 7, _lookup_transfers)
        self.linker.on_device_hit("r2", self.key, 2 * PAGE, 8, _lookup_transfers)

        first, second = self.client.calls
        self.assertEqual(second[:2], first[:2])

    def test_a_node_is_hinted_at_most_once_per_interval(self):
        self.linker.on_device_hit("r1", self.key, 4 * PAGE, 7, _lookup_transfers)
        self.linker.on_device_hit("r2", self.key, 4 * PAGE, 7, _lookup_transfers)

        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(self.linker._touch_diag["skipped_recent"], 1)

    def test_a_hint_is_dropped_while_the_previous_one_is_in_flight(self):
        self.linker._touch_busy = True

        self.linker.on_device_hit("r", self.key, 2 * PAGE, 9, _lookup_transfers)

        self.assertEqual(self.client.calls, [])
        self.assertEqual(self.linker._touch_diag["dropped_busy"], 1)
        # The node was not marked, so its next hit is hinted.
        self.linker._touch_busy = False
        self.linker.on_device_hit("r", self.key, 2 * PAGE, 9, _lookup_transfers)
        self.assertEqual(len(self.client.calls), 1)

    def test_a_failing_rpc_is_counted_and_releases_the_sender(self):
        self.client.fail = True

        self.linker.on_device_hit("r", self.key, 2 * PAGE, 10, _lookup_transfers)

        self.assertEqual(self.linker._touch_diag["rpc_errors"], 1)
        self.assertFalse(self.linker._touch_busy)
        self.assertEqual(self.linker._touch_interval, 30.0)

    def test_a_failure_on_the_scheduler_path_turns_hints_off(self):
        def broken(hashes):
            raise ValueError("no layout")

        self.linker.on_device_hit("r", self.key, 2 * PAGE, 11, broken)

        self.assertEqual(self.linker._touch_interval, 0.0)
        self.linker.on_device_hit("r", self.key, 2 * PAGE, 12, _lookup_transfers)
        self.assertEqual(self.client.calls, [])

    def test_hints_are_off_at_a_zero_interval(self):
        linker = _linker(self.client, interval=0.0)

        linker.on_device_hit("r", self.key, 4 * PAGE, 7, _lookup_transfers)

        self.assertEqual(self.client.calls, [])


if __name__ == "__main__":
    unittest.main()
