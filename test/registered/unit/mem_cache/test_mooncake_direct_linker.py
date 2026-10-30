"""Unit tests for the Mooncake direct linker's writer election."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.storage.mooncake_store import mooncake_direct_linker
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestOffloadOwner(CustomTestCase):
    def test_one_rank_writes_a_rank_replicated_group(self):
        group = SimpleNamespace(rank_replicated=True, entry_map={}, num_layers=1)
        for tp_rank, attn_cp_rank, owner in (
            (0, 0, True),
            (0, 1, False),
            (1, 0, False),
        ):
            params = SimpleNamespace(
                page_size=1,
                token_to_kv_pool_allocator=MagicMock(),
                attn_tp_cache_group=None,
                tp_cache_group=None,
                pp_rank=0,
                pp_size=1,
                attn_cp_rank=attn_cp_rank,
                attn_cp_size=2,
            )
            with (
                patch.object(
                    mooncake_direct_linker,
                    "resolve_hybrid_device_pool_group",
                    return_value=group,
                ),
                patch.object(
                    mooncake_direct_linker,
                    "get_parallel",
                    return_value=SimpleNamespace(tp_size=2),
                ),
                patch.object(
                    mooncake_direct_linker,
                    "get_model",
                    return_value=SimpleNamespace(model_path="model"),
                ),
                patch.object(torch.distributed, "is_initialized", return_value=True),
                patch.object(torch.distributed, "get_rank", return_value=tp_rank),
                patch.object(torch.distributed, "get_world_size", return_value=2),
            ):
                linker = mooncake_direct_linker.MooncakeDirectLinker(
                    SimpleNamespace(enable_linker_mla_dedup=False),
                    params,
                    components=set(),
                    storage=MagicMock(),
                )
            try:
                self.assertIs(linker.offload_owner, owner)
                self.assertEqual(linker.storage.mla_suffix, "pp0")
            finally:
                linker.close()


if __name__ == "__main__":
    unittest.main()
