"""Unit tests for the Mooncake direct linker's rank-replicated key handling."""

import unittest

from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    _storage_suffix,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestStorageSuffix(CustomTestCase):
    def test_replicated_group_is_one_keyspace_across_tp_and_cp(self):
        # Prefill CP gathers the whole sequence into every CP rank's pool.
        suffixes = {
            _storage_suffix(
                rank_replicated=True, tp_rank=rank, attn_cp_rank=cp, pp_rank=1
            )
            for rank in range(4)
            for cp in range(2)
        }
        self.assertEqual(suffixes, {"pp1"})

    def test_sharded_group_keeps_its_tp_and_cp_terms(self):
        self.assertEqual(
            _storage_suffix(
                rank_replicated=False, tp_rank=3, attn_cp_rank=1, pp_rank=0
            ),
            "tp3_cp1_pp0",
        )


if __name__ == "__main__":
    unittest.main()
