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
    def test_every_rank_shares_the_kv_key_of_a_kv_replicated_group(self):
        group = SimpleNamespace(rank_replicated=False, kv_rank_replicated=True)
        overrides = [
            _pool_suffix_overrides(
                pool_group=group, tp_rank=rank, attn_cp_rank=0, pp_rank=0
            )
            for rank in range(8)
        ]
        self.assertEqual(overrides, [{PoolName.KV: "cp0_pp0"}] * 8)

    def test_sharded_and_fully_replicated_groups_keep_their_suffix(self):
        for group in (
            SimpleNamespace(rank_replicated=False, kv_rank_replicated=False),
            SimpleNamespace(rank_replicated=True, kv_rank_replicated=True),
        ):
            self.assertEqual(
                _pool_suffix_overrides(
                    pool_group=group, tp_rank=3, attn_cp_rank=0, pp_rank=0
                ),
                {},
            )

    def test_the_override_renames_only_the_kv_objects(self):
        store = UMBPStore.__new__(UMBPStore)
        store.registered_pools = {
            PoolName.KV: SimpleNamespace(components=[object()]),
            PoolName.DRAFT: SimpleNamespace(components=[object()]),
            PoolName.MAMBA: DevicePoolEntry.__new__(DevicePoolEntry),
        }
        store.mla_suffix = store.mha_suffix = "tp3_cp0_pp0"
        store.is_mla_backend = True
        store.config_prefix = ""

        def keys(pool):
            return store._get_hybrid_page_component_keys(
                ["page"], SimpleNamespace(name=pool)
            )[0]

        before = {pool: keys(pool) for pool in store.registered_pools}
        store.pool_suffix_overrides = {PoolName.KV: "cp0_pp0"}
        after = {pool: keys(pool) for pool in store.registered_pools}

        self.assertNotEqual(after[PoolName.KV], before[PoolName.KV])
        self.assertNotIn("tp3", after[PoolName.KV][0])
        self.assertIn("_cp0_pp0_", after[PoolName.KV][0])
        # The draft and the KDA state differ per rank and keep their keyspace.
        self.assertEqual(after[PoolName.DRAFT], before[PoolName.DRAFT])
        self.assertEqual(after[PoolName.MAMBA], before[PoolName.MAMBA])
        self.assertIn("tp3", after[PoolName.MAMBA][0])


if __name__ == "__main__":
    unittest.main()
