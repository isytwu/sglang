"""SGLANG_LINKER_SKIP_OUTPUT_OFFLOAD: a finished request's insert offloads its
prompt only. Its output reaches the store once a later prompt reuses it, as that
insert writes the reused nodes through."""

import os
import sys
import unittest
from array import array
from collections import defaultdict
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_unified_cache_linker as base  # noqa: E402

from sglang.srt.environ import envs  # noqa: E402
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


class TestSkipOutputOffload(UnifiedRadixCacheFixtures, CustomTestCase):
    def _cache(self, skip_output):
        self.cfg = CacheConfig(
            page_size=1,
            components=(ComponentType.FULL, ComponentType.MAMBA),
            kv_size=128,
            max_context_len=64,
            mamba_cache_size=16,
        )
        self.cache, self.allocator, self.req_pool = build_fixture(self.cfg)
        self.store = base._InMemoryUnifiedCacheLinker(defaultdict(set))
        with envs.SGLANG_LINKER_SKIP_OUTPUT_OFFLOAD.override(skip_output):
            self.cache.init_cache_linker(self.store)

    def _cache_req(self, tokens, prompt_len=None):
        """Cache ``tokens`` as a request; finished, with a ``prompt_len``-token
        prompt, if ``prompt_len`` is given."""
        req = SimpleNamespace(
            linker_offload_skip_prefix_len=0,
            extra_key=None,
            cache_salt=None,
            origin_input_ids=array("q", tokens[:prompt_len]),
            finished=lambda: prompt_len is not None,
        )
        with self.cache._linker_offload_skip_prefix(req, tokens):
            result = self._insert(self.cache, self.allocator, self.req_pool, tokens)
            self.cache.linker.offload_new_state(result.last_device_node)
        while self.store.pending_offloads:
            self.store.complete_next_offload(True)
        self.cache.check_hicache_events()
        return result.last_device_node

    def _stored(self, pool):
        return len(self.store.stored_keys[pool])

    def test_finished_request_offloads_its_prompt_only(self):
        self._cache(skip_output=True)
        self._cache_req(list(range(1, 5)))
        output = self._cache_req(list(range(1, 9)), prompt_len=4)
        self.assertEqual(self._stored(PoolName.KV), 4)
        self.assertEqual(self._stored(PoolName.MAMBA), 1)
        self.assertFalse(self.cache.tree_core.is_external_cache_stored(output))
        self.assertEqual(self.cache.linker.offload_end_limit, 0)
        self.cache.sanity_check()

    def test_later_prompt_writes_the_output_through(self):
        self._cache(skip_output=True)
        self._cache_req(list(range(1, 5)))
        output = self._cache_req(list(range(1, 9)), prompt_len=4)
        self._cache_req(list(range(1, 13)))
        self.assertTrue(self.cache.tree_core.is_external_cache_stored(output))
        self.assertEqual(self._stored(PoolName.KV), 12)
        self.cache.sanity_check()

    def test_unfinished_request_offloads_everything(self):
        self._cache(skip_output=True)
        self._cache_req(list(range(1, 9)))
        self.assertEqual(self._stored(PoolName.KV), 8)
        self.assertEqual(self._stored(PoolName.MAMBA), 1)

    def test_default_offloads_outputs(self):
        self._cache(skip_output=False)
        self._cache_req(list(range(1, 5)))
        output = self._cache_req(list(range(1, 9)), prompt_len=4)
        self.assertTrue(self.cache.tree_core.is_external_cache_stored(output))
        self.assertEqual(self._stored(PoolName.KV), 8)
        self.assertEqual(self._stored(PoolName.MAMBA), 2)


if __name__ == "__main__":
    unittest.main()
