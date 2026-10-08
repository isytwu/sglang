# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""`--dcp-size > 1` refuses hybrid sliding-window models.

BUG REGRESSION. The SWA allocator family never widens its loc space, so a
widened loc is never collapsed back to a row on this path. Unlike the MLA
pools nothing range-checks it either: the server boots, serves, and reports
plausible numbers computed over the wrong KV.

    python -m pytest test/registered/unit/server_args/test_hybrid_swa_dcp_gate.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import msgspec

from sglang.srt.arg_groups.kv_cache_hook import handle_cache_compatibility
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _run_handler(*, dcp_size, is_hybrid_swa):
    """Run just `handle_cache_compatibility` over a minimal stand-in."""
    sa = ServerArgs(model_path="dummy")
    for name, value in {
        "dcp_size": dcp_size,
        "disaggregation_decode_retraction_backup": None,
        "disaggregation_mode": "null",
        "disaggregation_decode_enable_offload_kvcache": False,
        "enable_priority_scheduling": False,
        "disable_priority_preemption": True,
        "enable_hierarchical_cache": False,
        "disable_radix_cache": False,
        "hicache_storage_backend": None,
        "swa_full_tokens_ratio": 1.0,
    }.items():
        msgspec.Struct.__setattr__(sa, name, value)
    with patch(
        "sglang.srt.arg_groups.kv_cache_hook.model_config_of",
        return_value=SimpleNamespace(is_hybrid_swa=is_hybrid_swa),
    ):
        handle_cache_compatibility(sa)


class TestHybridSwaDcpGate(unittest.TestCase):
    def test_hybrid_swa_with_dcp_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            _run_handler(dcp_size=8, is_hybrid_swa=True)
        self.assertIn("hybrid sliding-window", str(ctx.exception))

    def test_gate_fires_only_on_the_pair(self):
        """An inverted condition here would reject every DCP launch."""
        # Kimi-K3 is hybrid mamba, not hybrid SWA: DCP stays available to it.
        _run_handler(dcp_size=8, is_hybrid_swa=False)
        _run_handler(dcp_size=1, is_hybrid_swa=True)


if __name__ == "__main__":
    unittest.main()
