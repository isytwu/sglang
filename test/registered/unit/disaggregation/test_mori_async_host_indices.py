"""Mori prefill: page and state indices copied to the host asynchronously."""

import importlib
import sys
import threading
import types
import unittest
from collections import defaultdict
from enum import IntEnum

import numpy as np
import torch


def _install_mori_stubs() -> None:
    try:
        importlib.import_module("mori.cpp")
        importlib.import_module("mori.io")
        return
    except (ImportError, OSError):
        pass

    mori = types.ModuleType("mori")
    mori.__path__ = []
    mori_cpp = types.ModuleType("mori.cpp")
    mori_io = types.ModuleType("mori.io")

    class FakeStatusCode(IntEnum):
        SUCCESS = 0
        IN_PROGRESS = 1
        FAILED = 2

    mori_cpp.TransferStatus = object
    for name in (
        "BackendType",
        "EngineDesc",
        "IOEngine",
        "IOEngineConfig",
        "MemoryDesc",
        "MemoryLocationType",
        "PollCqMode",
        "RdmaBackendConfig",
    ):
        setattr(mori_io, name, object)
    mori_io.StatusCode = FakeStatusCode
    mori.cpp = mori_cpp
    mori.io = mori_io
    sys.modules["mori"] = mori
    sys.modules["mori.cpp"] = mori_cpp
    sys.modules["mori.io"] = mori_io


_install_mori_stubs()

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.utils import FastQueue, TransferKVChunk
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.disaggregation.mori.conn import (
    MoriKVManager,
    MoriKVSender,
    _SubmissionLocal,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOM = 5


class _Event:
    def __init__(self, log):
        self.log = log

    def synchronize(self):
        self.log.append("indices_event")


def _manager(async_host_indices: bool) -> MoriKVManager:
    m = MoriKVManager.__new__(MoriKVManager)
    m.disaggregation_mode = DisaggregationMode.PREFILL
    m.async_host_indices = async_host_indices
    m.enable_deferred_decode_kv_release = False
    m.request_status = {ROOM: KVPoll.WaitingForInput}
    m.transfer_infos = {ROOM: {}}
    m.transfer_lock = threading.Lock()
    m._submission_local = _SubmissionLocal()
    m._staging_outstanding = defaultdict(int)
    m._num_shards = 1
    m._transfer_queues = [FastQueue()]
    m.attn_cp_size = 1
    m.attn_cp_rank = 0
    m.failure_lock = threading.Lock()
    m.failure_records = {}
    return m


def _host_chunk(log):
    # What _send_kv_chunk queues in host mode: a pinned-tensor page snapshot
    # and int64 state views, all valid only after indices_event.
    return TransferKVChunk(
        room=ROOM,
        prefill_kv_indices=torch.tensor([3, 4, 9], dtype=torch.int64),
        index_slice=slice(0, 3),
        is_last_chunk=True,
        prefill_aux_index=1,
        state_indices=[
            np.array([[7]], dtype=np.int64),
            None,
            np.array([1, 2, 3], dtype=np.int64),
        ],
        indices_event=_Event(log),
    )


class TestMoriAsyncHostIndices(CustomTestCase):
    def test_worker_waits_for_the_copy_before_reading_indices(self):
        log = []
        m = _manager(async_host_indices=True)
        seen = {}

        def submit(chunk):
            log.append("submit")
            seen["kv"] = chunk.prefill_kv_indices
            seen["state"] = chunk.state_indices
            return []

        m._submit_transfer_chunk = submit
        m.conclude_transfer = lambda **kw: log.append("conclude")
        m._process_transfer_chunk(_host_chunk(log))
        self.assertEqual(log, ["indices_event", "submit", "conclude"])
        self.assertIsInstance(seen["kv"], np.ndarray)
        np.testing.assert_array_equal(seen["kv"], [3, 4, 9])
        self.assertEqual(seen["state"][0].dtype, np.int32)
        np.testing.assert_array_equal(seen["state"][0], [7])
        self.assertIsNone(seen["state"][1])
        self.assertEqual(seen["state"][2].dtype, np.int32)

    def test_scheduler_side_prepare_reads_no_element(self):
        m = _manager(async_host_indices=True)
        mamba = np.array([[7]], dtype=np.int64)
        pages = np.array([1, 2, 3], dtype=np.int64)
        out = m.prepare_send_state_indices([mamba, None, pages])
        # Views of the host buffers, not converted copies.
        self.assertTrue(np.shares_memory(out[0], mamba))
        self.assertTrue(np.shares_memory(out[2], pages))
        self.assertEqual(out[0].shape, (1,))
        self.assertIsNone(out[1])

    def test_sync_mode_keeps_casting_on_the_scheduler(self):
        m = _manager(async_host_indices=False)
        out = m.prepare_send_state_indices([np.array([[7]], dtype=np.int64)])
        self.assertEqual(out[0].dtype, np.int32)
        log = []
        chunk = _host_chunk(log)
        chunk.indices_event = None
        m._resolve_host_indices(chunk)
        self.assertIsInstance(chunk.prefill_kv_indices, torch.Tensor)

    def test_add_transfer_request_queues_the_indices_event(self):
        m = _manager(async_host_indices=True)
        event = _Event([])
        m.add_transfer_request(
            ROOM,
            torch.tensor([1, 2]),
            slice(0, 2),
            False,
            indices_event=event,
        )
        self.assertIs(m._transfer_queues[0].get().indices_event, event)

    def test_sender_advertises_host_indices_from_the_manager(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                sender = MoriKVSender.__new__(MoriKVSender)
                sender.kv_mgr = _manager(async_host_indices=enabled)
                self.assertIs(sender.supports_host_tensor_indices, enabled)


if __name__ == "__main__":
    unittest.main()
