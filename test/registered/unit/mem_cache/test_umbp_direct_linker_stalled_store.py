"""Unit tests for how the UMBP direct linker behaves when its store stalls."""

import threading
import time
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Queue
from types import SimpleNamespace
from unittest import mock

import sglang.srt.mem_cache.storage.umbp.umbp_direct_linker as umbp_direct_linker
from sglang.srt.mem_cache.storage.umbp.umbp_direct_linker import UMBPDirectLinker
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _linker(timeout_s=1.0, store_lookup=None):
    linker = UMBPDirectLinker.__new__(UMBPDirectLinker)
    linker._lookup_timeout_s = timeout_s
    linker._lookup_pool = ThreadPoolExecutor(max_workers=1)
    linker._stuck_lookup = None
    linker._offload_stall_s = 5.0
    linker._offload_busy_since = None
    linker._ranks_see_stall = False
    linker._stats = {"lookup_timeout": 0, "lookup_skipped": 0, "offload_skipped": 0}
    linker._lookup = store_lookup
    linker._offload_results = Queue()
    linker._offload_queue = Queue()
    linker._gc_frozen = True
    linker.pool_group = SimpleNamespace(
        resolve_transfers=lambda _transfers, allow_partial: ["transfer"]
    )
    return linker


_FAKE_DEVICE = SimpleNamespace(Event=lambda: SimpleNamespace(record=lambda: None))


class TestLookupDeadline(CustomTestCase):
    def test_answer_within_the_deadline_is_returned(self):
        linker = _linker(1.0, lambda rid, transfers: [1, 2])
        self.assertEqual(linker.lookup("r", []), [1, 2])
        self.assertEqual(linker._stats["lookup_timeout"], 0)
        linker._lookup_pool.shutdown(wait=True)

    def test_stalled_store_is_a_miss_and_gets_no_new_rpc_until_it_answers(self):
        release = threading.Event()
        calls = []

        def store_lookup(rid, transfers):
            calls.append(rid)
            if rid == "first":
                release.wait(10)
            return [3]

        linker = _linker(0.05, store_lookup)
        start = time.monotonic()
        self.assertEqual(linker.lookup("first", []), [])
        self.assertLess(time.monotonic() - start, 5)
        # The stalled call is still out: later lookups miss without reaching the store.
        self.assertEqual(linker.lookup("second", []), [])
        self.assertEqual(calls, ["first"])
        self.assertEqual(linker._stats["lookup_timeout"], 1)
        self.assertEqual(linker._stats["lookup_skipped"], 1)

        release.set()
        linker._stuck_lookup.result(timeout=5)
        self.assertEqual(linker.lookup("third", []), [3])
        self.assertEqual(calls, ["first", "third"])
        self.assertIsNone(linker._stuck_lookup)
        linker._lookup_pool.shutdown(wait=True)

    def test_zero_timeout_looks_up_on_the_calling_thread(self):
        threads = []

        def store_lookup(rid, transfers):
            threads.append(threading.current_thread())
            return [5]

        linker = _linker(0.0, store_lookup)
        self.assertEqual(linker.lookup("r", []), [5])
        self.assertIs(threads[0], threading.current_thread())
        linker._lookup_pool.shutdown(wait=True)


class TestOffloadBackPressure(CustomTestCase):
    """A queued offload pins its tree node until it lands; a stalled store must
    not be fed more, or the pinned nodes exhaust the Mamba slots. The decision
    follows the ranks' agreed view, so every rank pins the same nodes."""

    def test_store_reports_unresponsive_while_a_lookup_is_stuck(self):
        linker = _linker()
        self.assertTrue(linker.store_responsive())
        linker._stuck_lookup = Future()  # never completes
        self.assertFalse(linker.store_responsive())

    def test_store_reports_unresponsive_while_a_batch_is_busy_too_long(self):
        linker = _linker()
        linker._offload_busy_since = time.monotonic()
        self.assertTrue(linker.store_responsive())
        linker._offload_busy_since = time.monotonic() - 10
        self.assertFalse(linker.store_responsive())

    def test_offload_follows_the_agreed_view_not_this_ranks_own(self):
        linker = _linker()
        linker._stuck_lookup = Future()  # this rank alone sees a stall
        with mock.patch.object(umbp_direct_linker, "device_module", _FAKE_DEVICE):
            self.assertTrue(linker.offload(["t"]))
            linker.set_store_responsive(False)
            self.assertFalse(linker.offload(["t"]))
            self.assertEqual(linker._stats["offload_skipped"], 1)
            linker.set_store_responsive(True)
            self.assertTrue(linker.offload(["t"]))
        self.assertEqual(linker._offload_queue.qsize(), 2)


if __name__ == "__main__":
    unittest.main()
