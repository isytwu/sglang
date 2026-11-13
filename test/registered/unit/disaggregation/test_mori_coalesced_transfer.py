"""Mori prefill: a chunk's KV and state writes in one engine.batch_write."""

import importlib
import sys
import threading
import types
import unittest
from collections import Counter, defaultdict
from enum import IntEnum

import numpy as np


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

from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.srt.disaggregation.common.utils import TransferKVChunk
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.disaggregation.mori.conn import (
    KVArgsRegisterInfo,
    MoriKVManager,
    TransferInfo,
    _MoriTransferSubmissionError,
    _SubmissionLocal,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MLA_LAYERS = 3
DRAFT_KV = 1
KDA_LAYERS = 4
DSA_LAYERS = 3
PAGE = 1024
ROOM = 7


class _Status:
    def InProgress(self):
        return False

    def Failed(self):
        return False


class _Engine:
    def __init__(self, batch_uids: bool):
        self.writes = []
        self.uid_calls = 0
        self.next_uid = 0
        self.lock = threading.Lock()
        if not batch_uids:
            self.allocate_transfer_uids = None

    def allocate_transfer_uid(self):
        self.uid_calls += 1
        self.next_uid += 1
        return self.next_uid - 1

    def allocate_transfer_uids(self, n):
        self.uid_calls += 1
        self.next_uid += n
        return list(range(self.next_uid - n, self.next_uid))

    def batch_write(self, srcs, local_offsets, dsts, remote_offsets, sizes, uids):
        assert len(srcs) == len(local_offsets) == len(dsts) == len(sizes) == len(uids)
        self.writes.append(
            [
                (s, tuple(lo), d, tuple(ro), tuple(sz), uid)
                for s, lo, d, ro, sz, uid in zip(
                    srcs, local_offsets, dsts, remote_offsets, sizes, uids
                )
            ]
        )
        return [_Status() for _ in sizes]

    def entries(self):
        """Posted writes without their transfer ids."""
        return Counter(entry[:5] for call in self.writes for entry in call)


def _descs(prefix, n):
    return [f"{prefix}{i}" for i in range(n)]


def _manager(engine, coalesce: bool, events=None) -> MoriKVManager:
    m = MoriKVManager.__new__(MoriKVManager)
    m.disaggregation_mode = DisaggregationMode.PREFILL
    m.engine = engine
    m._coalesce_transfers = coalesce
    m.is_mla_backend = True
    m.is_hybrid_mla_backend = True
    m.attn_tp_size = 8
    m.state_tp_size = 8
    m.pp_size = 1
    m.enable_deferred_decode_kv_release = True
    m.request_status = {ROOM: KVPoll.WaitingForInput}
    m.transfer_lock = threading.Lock()
    m._abort_ack_lock = threading.Lock()
    m._staging_outstanding = defaultdict(int)
    m._deferred_ack_targets = {}
    m._deferred_ack_poisoned_rooms = set()
    m._submission_local = _SubmissionLocal()
    m._send_aux_rdma = False
    m.failure_lock = threading.Lock()
    m.failure_records = {}
    m.pp_group = types.SimpleNamespace(is_last_rank=True)

    n_kv = MLA_LAYERS + DRAFT_KV
    mamba_ids = list(range(KDA_LAYERS)) * 2
    m.kv_args = types.SimpleNamespace(
        engine_rank=0,
        page_size=64,
        kv_item_lens=[PAGE] * MLA_LAYERS + [PAGE // 4] * DRAFT_KV,
        num_draft_entries=DRAFT_KV,
        kv_layer_ids=list(range(n_kv)),
        prefill_start_layer=0,
        state_types=[StateType.MAMBA, StateType.DSA, StateType.REQUEST_ROW],
        state_item_lens=[
            [4096] * KDA_LAYERS + [256] * KDA_LAYERS,
            [128] * DSA_LAYERS,
            [8],
        ],
        state_dim_per_tensor=[[12] * (2 * KDA_LAYERS), [], []],
        state_layer_ids=[mamba_ids, list(range(DSA_LAYERS)), [0]],
        state_conv_shard_groups=[[None] * (2 * KDA_LAYERS), [], []],
        state_slice_outer_counts=[[1] * (2 * KDA_LAYERS), [], []],
        aux_item_lens=[16],
    )
    m._aux = np.zeros(16 * 8, dtype=np.uint8)
    m.kv_args.aux_data_ptrs = [m._aux.ctypes.data]
    m.kv_mem_descs = _descs("kv", n_kv)
    m.aux_mem_descs = _descs("aux", 1)
    comp_sizes = [2 * KDA_LAYERS, DSA_LAYERS, 1]
    m.state_mem_descs = [_descs(f"st{c}.", n) for c, n in enumerate(comp_sizes)]
    m.decode_kv_args_table = {
        "decode": KVArgsRegisterInfo(
            endpoint="10.0.0.2",
            dst_port=7000,
            engine_desc=None,
            dst_kv_mem_descs=_descs("dkv", n_kv),
            dst_aux_mem_descs=_descs("daux", 1),
            dst_state_mem_descs=[
                _descs(f"dst{c}.", n) for c, n in enumerate(comp_sizes)
            ],
            gpu_id=0,
            decode_tp_size=8,
            decode_tp_rank=0,
            dst_kv_item_len=PAGE,
            dst_state_item_lens=[list(x) for x in m.kv_args.state_item_lens],
            dst_state_dim_per_tensor=[list(x) for x in m.kv_args.state_dim_per_tensor],
            dst_state_layer_ids=[list(x) for x in m.kv_args.state_layer_ids],
            dst_kv_layer_ids=list(m.kv_args.kv_layer_ids),
        )
    }
    m.transfer_infos = {
        ROOM: {
            "decode": TransferInfo(
                room=ROOM,
                endpoint="10.0.0.2",
                dst_port=7000,
                engine_key="decode",
                dst_kv_indices=np.array([40, 41, 42, 50, 51, 60], dtype=np.int32),
                dst_aux_index=3,
                dst_state_indices=[
                    np.array([9], dtype=np.int32),
                    np.array([70, 71, 72, 80, 81, 90], dtype=np.int32),
                    np.array([9], dtype=np.int32),
                ],
                required_dst_info_num=1,
                is_dummy=False,
            )
        }
    }
    events = events if events is not None else []

    def send_aux_frame(*args, **kwargs):
        events.append(("aux", len(engine.writes)))

    m._send_aux_data_to_endpoint = send_aux_frame
    return m


def _chunk(is_last: bool) -> TransferKVChunk:
    return TransferKVChunk(
        room=ROOM,
        prefill_kv_indices=np.array([10, 11, 12, 20, 21, 30], dtype=np.int32),
        index_slice=slice(0, 6),
        is_last_chunk=is_last,
        prefill_aux_index=2 if is_last else None,
        state_indices=(
            [
                np.array([5], dtype=np.int32),
                np.array([100, 101, 102, 110, 111, 120], dtype=np.int32),
                np.array([5], dtype=np.int32),
            ]
            if is_last
            else None
        ),
        num_kv_tokens=6 * 64,
    )


class TestMoriCoalescedTransfer(CustomTestCase):
    def _submit(self, coalesce, is_last=True, batch_uids=True, events=None):
        engine = _Engine(batch_uids)
        m = _manager(engine, coalesce, events)
        statuses = m._submit_transfer_chunk(_chunk(is_last))
        return engine, m, statuses

    def test_coalesced_posts_the_same_writes_in_one_call(self):
        per_layer, _, per_layer_statuses = self._submit(coalesce=False)
        coalesced, _, coalesced_statuses = self._submit(coalesce=True)
        # KV (3 MLA + 1 draft), KDA state (2 tensors x 4 layers), index-K (3), row.
        n_entries = MLA_LAYERS + DRAFT_KV + 2 * KDA_LAYERS + DSA_LAYERS + 1
        self.assertEqual(len(per_layer.writes), n_entries)
        self.assertEqual(len(coalesced.writes), 1)
        self.assertEqual(coalesced.entries(), per_layer.entries())
        self.assertEqual(len(coalesced_statuses), n_entries)
        self.assertEqual(len(per_layer_statuses), n_entries)

    def test_non_last_chunk_coalesces_kv_only(self):
        per_layer, _, _ = self._submit(coalesce=False, is_last=False)
        coalesced, _, _ = self._submit(coalesce=True, is_last=False)
        self.assertEqual(len(coalesced.writes), 1)
        self.assertEqual(len(coalesced.writes[0]), MLA_LAYERS + DRAFT_KV)
        self.assertEqual(coalesced.entries(), per_layer.entries())

    def test_one_uid_call_per_chunk_when_the_engine_batches_ids(self):
        engine, _, _ = self._submit(coalesce=True)
        uids = [entry[5] for entry in engine.writes[0]]
        self.assertEqual(engine.uid_calls, 1)
        self.assertEqual(len(set(uids)), len(uids))

    def test_per_entry_uids_on_an_engine_without_batch_ids(self):
        engine, _, _ = self._submit(coalesce=True, batch_uids=False)
        uids = [entry[5] for entry in engine.writes[0]]
        self.assertEqual(engine.uid_calls, len(uids))
        self.assertEqual(len(set(uids)), len(uids))

    def test_aux_follows_the_posted_writes(self):
        for coalesce in (False, True):
            with self.subTest(coalesce=coalesce):
                events = []
                engine, _, _ = self._submit(coalesce=coalesce, events=events)
                self.assertEqual(events, [("aux", len(engine.writes))])

    def test_statuses_are_recorded_for_the_drainer(self):
        engine = _Engine(batch_uids=True)
        m = _manager(engine, coalesce=True)
        recorded = []
        m._submission_local.statuses = recorded
        statuses = m._submit_kv_transfer(
            ROOM,
            _chunk(True).prefill_kv_indices,
            slice(0, 6),
            True,
            aux_index=2,
            state_indices=_chunk(True).state_indices,
            num_kv_tokens=6 * 64,
        )
        self.assertEqual(recorded, statuses)

    def test_failure_before_posting_leaves_nothing_in_flight(self):
        engine = _Engine(batch_uids=True)
        m = _manager(engine, coalesce=True)

        def broken_state(*args, **kwargs):
            raise RuntimeError("state layout mismatch")

        m.send_state = broken_state
        with self.assertRaises(_MoriTransferSubmissionError) as ctx:
            m._submit_transfer_chunk(_chunk(True))
        self.assertEqual(ctx.exception.statuses, [])
        self.assertEqual(engine.writes, [])
        self.assertIsNone(m._submission_local.batch)
        # The thread's next submission posts normally.
        m.send_state = MoriKVManager.send_state.__get__(m)
        m.request_status[ROOM] = KVPoll.WaitingForInput
        m._submit_transfer_chunk(_chunk(True))
        self.assertEqual(len(engine.writes), 1)

    def test_identical_item_lens_materialize_one_plan(self):
        engine = _Engine(batch_uids=True)
        m = _manager(engine, coalesce=True)
        calls = []
        original = m._build_contiguous_transfer_plan

        def counting(grouped_plan, item_len):
            calls.append(item_len)
            return original(grouped_plan, item_len)

        m._build_contiguous_transfer_plan = counting
        m._submit_transfer_chunk(_chunk(False))
        self.assertEqual(sorted(calls), sorted({PAGE, PAGE // 4}))


if __name__ == "__main__":
    unittest.main()
