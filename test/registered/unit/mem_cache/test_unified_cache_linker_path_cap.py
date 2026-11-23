"""The external store follows the device tree's per-path Mamba state cap: a state
the cap takes off the device because a deeper one on its path replaced it is
dropped from the store too, while fork states, which the cap keeps, stay."""

import os
import sys
import unittest
from collections import defaultdict
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_unified_cache_linker as base  # noqa: E402

from sglang.srt.mem_cache.hicache_storage import PoolName  # noqa: E402
from sglang.srt.mem_cache.unified_cache.component_type import (  # noqa: E402
    ComponentType,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.test.kits.unified_radix_cache.config import CacheConfig  # noqa: E402
from sglang.test.kits.unified_radix_cache.fixtures import build_fixture  # noqa: E402
from sglang.test.kits.unified_radix_cache.mixins import (  # noqa: E402
    UnifiedRadixCacheFixtures,
)
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _RemovingLinker(base._InMemoryUnifiedCacheLinker):
    supports_remove = True

    def __init__(self, fail=False):
        super().__init__(defaultdict(set))
        self.removed = []
        self.fail = fail

    def remove(self, transfers):
        if self.fail:
            raise RuntimeError("store is gone")
        for transfer in transfers:
            self.removed.extend(transfer.keys)
            self.stored_keys[transfer.name] -= set(transfer.keys)


class TestPathCapDropsStoredStates(UnifiedRadixCacheFixtures, CustomTestCase):
    def _cache(self, store):
        # Level 2 caps the tree at one Mamba state per path.
        self.cfg = CacheConfig(
            page_size=1,
            components=(ComponentType.FULL, ComponentType.MAMBA),
            kv_size=128,
            max_context_len=64,
            mamba_cache_size=16,
            hybrid_cache_checkpoints_level=2,
        )
        with patch(
            "sglang.test.kits.unified_radix_cache.fixtures.get_device",
            return_value="cpu",
        ):
            self.cache, self.allocator, self.req_pool = build_fixture(self.cfg)
        self.store = store
        self.cache.init_cache_linker(store)

    def _turn(self, tokens):
        """Insert ``tokens`` with a state at their end and let it land."""
        result = self._insert(self.cache, self.allocator, self.req_pool, tokens)
        self.cache.linker.offload_new_state(result.last_device_node)
        while self.store.pending_offloads:
            self.store.complete_next_offload(True)
        self.cache.check_hicache_events()
        return self.cache.tree_core.get_hash_values(result.last_device_node)[-1]

    def test_cap_drops_the_replaced_state_from_the_store(self):
        self._cache(_RemovingLinker())
        first = self._turn(list(range(1, 5)))
        self.assertIn(first, self.store.stored_keys[PoolName.MAMBA])
        second = self._turn(list(range(1, 9)))
        self.assertEqual(self.store.removed, [first])
        self.assertEqual(self.store.stored_keys[PoolName.MAMBA], {second})
        # The replaced state's KV stays stored.
        self.assertIn(first, self.store.stored_keys[PoolName.KV])
        self.cache.sanity_check()

    def test_fork_state_stays_in_the_store(self):
        self._cache(_RemovingLinker())
        middle = self._turn(list(range(1, 9)))
        self._turn([1, 2, 3, 4, 9, 10, 11, 12])
        # 1..4 is now a fork; a state there stays when a deeper one lands.
        fork = self._turn([1, 2, 3, 4])
        self._turn(list(range(1, 9)) + [13, 14, 15, 16])
        self.assertEqual(self.store.removed, [middle])
        self.assertIn(fork, self.store.stored_keys[PoolName.MAMBA])
        self.cache.sanity_check()

    def test_store_without_remove_keeps_every_state(self):
        store = base._InMemoryUnifiedCacheLinker(defaultdict(set))
        self._cache(store)
        first = self._turn(list(range(1, 5)))
        self._turn(list(range(1, 9)))
        self.assertIn(first, store.stored_keys[PoolName.MAMBA])
        self.cache.sanity_check()

    def test_failed_remove_stops_removing_and_spares_the_insert(self):
        self._cache(_RemovingLinker(fail=True))
        self._turn(list(range(1, 5)))
        with self.assertLogs(
            "sglang.srt.mem_cache.unified_cache.unified_cache_linker", "ERROR"
        ):
            self._turn(list(range(1, 9)))
        self.assertFalse(self.cache.linker._remove_states)
        self._turn(list(range(1, 13)))
        self.cache.sanity_check()


if __name__ == "__main__":
    unittest.main()
