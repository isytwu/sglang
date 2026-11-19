"""The external linker offloads a Mamba state an insert attaches to a node that is
already stored: the write-through chain offloads new leaves only, so without that a
checkpoint inside a stored path never reaches the store."""

import os
import sys
import unittest
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_unified_cache_linker as base  # noqa: E402

from sglang.srt.mem_cache.hicache_storage import PoolName  # noqa: E402
from sglang.srt.mem_cache.unified_cache.component_type import (  # noqa: E402
    ComponentType,
)
from sglang.test.ci.ci_register import register_cuda_ci  # noqa: E402
from sglang.test.kits.unified_radix_cache.config import CacheConfig  # noqa: E402
from sglang.test.kits.unified_radix_cache.fixtures import build_fixture  # noqa: E402

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class _StateCoverage:
    def _cache(self):
        self.cfg = CacheConfig(
            page_size=1,
            components=(ComponentType.FULL, ComponentType.MAMBA),
            kv_size=128,
            max_context_len=64,
            mamba_cache_size=16,
        )
        self.cache, self.allocator, self.req_pool = build_fixture(self.cfg)
        self.store = base._InMemoryUnifiedCacheLinker(defaultdict(set))
        self.cache.init_cache_linker(self.store)

    def _turn(self, tokens, lands=True, offload_new_state=True):
        """Insert ``tokens`` as a request's cache insert does; return the state keys
        it added to the store."""
        before = set(self.store.stored_keys[PoolName.MAMBA])
        result = self._insert(self.cache, self.allocator, self.req_pool, tokens)
        if offload_new_state:
            self.cache.linker.offload_new_state(result.last_device_node)
        while self.store.pending_offloads:
            self.store.complete_next_offload(lands)
        self.cache.check_hicache_events()
        return set(self.store.stored_keys[PoolName.MAMBA]) - before

    def test_chain_alone_misses_a_state_inside_a_stored_path(self):
        self._cache()
        self._turn(list(range(1, 9)))
        # 1..4 is stored as part of 1..8; a state at 4 splits that node.
        self.assertEqual(self._turn(list(range(1, 5)), offload_new_state=False), set())

    def test_state_inside_a_stored_path_reaches_the_store(self):
        self._cache()
        self.assertEqual(len(self._turn(list(range(1, 9)))), 1)
        self.assertEqual(len(self._turn(list(range(1, 5)))), 1)
        self.assertEqual(len(self.store.stored_keys[PoolName.MAMBA]), 2)
        self.cache.sanity_check()

    def test_same_state_goes_out_once(self):
        self._cache()
        self._turn(list(range(1, 9)))
        self._turn(list(range(1, 5)))
        self.assertEqual(self._turn(list(range(1, 5))), set())
        self.cache.sanity_check()

    def test_failed_state_offload_retries_on_the_next_insert(self):
        self._cache()
        self._turn(list(range(1, 9)))
        self.assertEqual(self._turn(list(range(1, 5)), lands=False), set())
        self.assertEqual(len(self._turn(list(range(1, 5)))), 1)
        self.cache.sanity_check()


class TestStateCoveragePythonBackend(
    _StateCoverage, base.TestUnifiedCacheLinkerPythonBackend
):
    pass


if __name__ == "__main__":
    unittest.main()
