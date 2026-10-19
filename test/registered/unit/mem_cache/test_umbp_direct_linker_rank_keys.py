"""Unit tests for the UMBP direct linker's rank-replicated key handling."""

import unittest
from queue import Queue
from types import SimpleNamespace
from unittest import mock

import sglang.srt.mem_cache.storage.umbp.umbp_direct_linker as umbp_direct_linker
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import DevicePoolEntry
from sglang.srt.mem_cache.storage.umbp.umbp_direct_linker import (
    UMBPDirectLinker,
    _pool_suffix_overrides,
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


class TestSharedKvKeys(CustomTestCase):
    def test_only_a_kv_replicated_group_shares_the_kv_key(self):
        for rank_replicated, kv_rank_replicated, expected in (
            (False, True, {PoolName.KV: "cp0_pp0"}),
            (False, False, {}),
            (True, True, {}),
        ):
            group = SimpleNamespace(
                rank_replicated=rank_replicated, kv_rank_replicated=kv_rank_replicated
            )
            self.assertEqual(
                _pool_suffix_overrides(
                    pool_group=group, tp_rank=3, attn_cp_rank=0, pp_rank=0
                ),
                expected,
            )

    def test_the_override_renames_only_the_kv_objects(self):
        store = UMBPStore.__new__(UMBPStore)
        store.registered_pools = {
            PoolName.KV: SimpleNamespace(components=[object()]),
            PoolName.MAMBA: DevicePoolEntry.__new__(DevicePoolEntry),
        }
        store.mla_suffix = store.mha_suffix = "tp3_cp0_pp0"
        store.is_mla_backend = True
        store.config_prefix = ""
        store.pool_suffix_overrides = {PoolName.KV: "cp0_pp0"}

        def key(pool):
            return store._get_hybrid_page_component_keys(
                ["page"], SimpleNamespace(name=pool)
            )[0][0]

        self.assertNotIn("tp3", key(PoolName.KV))
        self.assertIn("tp3", key(PoolName.MAMBA))


class TestSharedKeyWriteRace(CustomTestCase):
    def _run_offload(self, refused, held):
        linker = UMBPDirectLinker.__new__(UMBPDirectLinker)
        linker.pool_group = SimpleNamespace(rank_replicated=False)
        exists = mock.Mock(side_effect=lambda keys: [held] * len(keys))
        linker.storage = SimpleNamespace(
            pool_suffix_overrides={PoolName.KV: "cp0_pp0"},
            client=SimpleNamespace(batch_exists=exists),
        )
        linker._stats = {"offload": 0, "offload_batches": 0}
        plans = [
            SimpleNamespace(name=PoolName.KV, keys=["kv0", "kv1"]),
            SimpleNamespace(name=PoolName.MAMBA, keys=["m0"]),
        ]
        written = []

        def put_plan(plan, *, log_failures):
            written.append(plan.name)
            return refused.get(plan.name, [])

        with (
            mock.patch.object(linker, "_build_load_plans", return_value=plans),
            mock.patch.object(linker, "_put_plan", side_effect=put_plan),
        ):
            ok = linker._run_offload([([], SimpleNamespace(synchronize=lambda: None))])
        return ok, written, exists

    def test_a_refused_key_fails_unless_another_rank_stored_it(self):
        for refused, held, stored in (
            ({PoolName.KV: ["kv1"]}, True, True),
            ({PoolName.KV: ["kv1"]}, False, False),
            # A per-rank key has no other writer.
            ({PoolName.MAMBA: ["m0"]}, True, False),
        ):
            with self.subTest(refused=refused, held=held):
                ok, written, exists = self._run_offload(refused, held)
                self.assertEqual(ok, stored)
                self.assertEqual(written, [PoolName.KV, PoolName.MAMBA])
                self.assertEqual(exists.called, PoolName.KV in refused)

    def test_a_run_of_dry_batches_is_reported_once(self):
        linker = UMBPDirectLinker.__new__(UMBPDirectLinker)
        clock = iter(range(100000))
        with (
            mock.patch.object(
                umbp_direct_linker.time, "monotonic", side_effect=lambda: next(clock)
            ),
            mock.patch.object(umbp_direct_linker.logger, "error") as error,
        ):
            for _ in range(umbp_direct_linker.OFFLOAD_DRY_ALERT_BATCHES * 2):
                linker._note_offload_outcome(False)
        self.assertEqual(error.call_count, 1)


if __name__ == "__main__":
    unittest.main()
