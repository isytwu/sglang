"""KL for the UMBP direct linker on Kimi Linear (FULL + Mamba: MLA + KDA).

Mirrors test_unified_cache_linker_kl_inkling.py with a local UMBP standalone
server. Turn 2 is scored right after the restore boundary: further out, the MLA
layers mask a wrong KDA state.
"""

import json
import os
import subprocess
import tempfile
import time
import unittest

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kits.unified_radix_cache_kit import _random_suffixes
from sglang.test.kl_multiturn_utils import (
    _extract_output_logprobs,
    _flush_cache,
    _generate,
    _replay_and_compare_kl,
)
from sglang.test.kl_test_utils import get_input_ids
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    find_available_port,
    popen_launch_server,
    terminate_and_kill_process_tree,
    unified_radix_tree_server_env,
)

register_amd_ci(est_time=1800, suite="nightly-amd-8-gpu-mi35x", nightly=True)

_MODEL_PATH = os.environ.get(
    "KIMI_LINEAR_MODEL_PATH", "moonshotai/Kimi-Linear-48B-A3B-Instruct"
)

# Not bit-exact on ROCm: on MI355X a correct restore measured 0.8e-3 to 3e-3,
# one that left the KDA state zeroed about 5e-2.
KL_DIV_THRESHOLD = 1e-2
PAGE_SIZE = 64
# A multiple of the DCP4-widened tree page.
TRACK_INTERVAL = 512
# Turn 1's continuation and turn 2's suffix.
TAIL_TOKENS = 16


def _start_umbp_standalone_server(socket_path):
    import mori

    binary = os.path.join(os.path.dirname(mori.__file__), "umbp_standalone_server")
    process = subprocess.Popen(
        [binary, f"unix://{socket_path}"],
        env={
            **os.environ,
            "UMBP_DRAM_CAPACITY": str(64 << 30),
            "UMBP_SSD_ENABLED": "0",
            "HSA_USERPTR_FOR_PAGED_MEM": "0",
            "HSA_ENABLE_IPC_MODE_LEGACY": "1",
        },
    )
    deadline = time.monotonic() + 300
    while not os.path.exists(socket_path):
        if process.poll() is not None or time.monotonic() > deadline:
            raise RuntimeError("The UMBP standalone server did not start.")
        time.sleep(1)
    return process


class _KimiLinearUMBPLinkerKLMixin:
    dcp_size = 1
    extra_args: tuple[str, ...] = ()

    @classmethod
    def setUpClass(cls):
        cls.model = _MODEL_PATH
        cls.base_url = f"http://127.0.0.1:{find_available_port(30000)}"
        socket_path = os.path.join(tempfile.mkdtemp(prefix="umbp-sa-"), "sa.sock")
        cls.umbp = _start_umbp_standalone_server(socket_path)
        cls.addClassCleanup(terminate_and_kill_process_tree, cls.umbp, wait_timeout=30)
        other_args = [
            "--trust-remote-code",
            "--tp-size",
            "4",
            *(["--dcp-size", str(cls.dcp_size)] if cls.dcp_size > 1 else []),
            "--page-size",
            str(PAGE_SIZE),
            "--attention-backend",
            "aiter",
            "--mamba-track-interval",
            str(TRACK_INTERVAL),
            "--mem-fraction-static",
            "0.7",
            "--max-running-requests",
            "1",
            "--enable-cache-report",
            "--enable-unified-cache-external-linker",
            "--unified-cache-external-linker-backend",
            "mori",
            "--hicache-storage-backend-extra-config",
            json.dumps({"standalone_address": f"unix://{socket_path}"}),
            *cls.extra_args,
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=unified_radix_tree_server_env(
                "python", HSA_ENABLE_IPC_MODE_LEGACY="1", SGLANG_USE_AITER="1"
            ),
        )
        cls.addClassCleanup(
            terminate_and_kill_process_tree, cls.process, wait_timeout=60
        )
        cls.input_ids = get_input_ids(
            tokenizer_path=cls.model, num_samples=4, trust_remote_code=True
        )

    def test_load_back_after_flush(self):
        """Every second-turn prefix comes back from UMBP alone."""
        first_turn = [list(ids[:1024]) for ids in self.input_ids]
        histories = [
            ids + _generate(self.base_url, [ids], TAIL_TOKENS)[0]["output_ids"]
            for ids in first_turn
        ]
        # Offloads land asynchronously; give the last checkpoint time to arrive.
        time.sleep(10)

        _flush_cache(self.base_url)
        sequences, logprobs = [], []
        for history, suffix in zip(
            histories, _random_suffixes(len(histories), TAIL_TOKENS, seed=100)
        ):
            ids = history + suffix
            (result,) = _generate(self.base_url, [ids], 512, return_logprob=True)
            expected = (len(history) - 1) // TRACK_INTERVAL * TRACK_INTERVAL
            cached_tokens = result["meta_info"]["cached_tokens"]
            details = result["meta_info"].get("cached_tokens_details") or {}
            self.assertGreaterEqual(cached_tokens, expected)
            self.assertEqual(int(details.get("host", 0)), cached_tokens)
            sequences.append(ids + result["output_ids"])
            logprobs.append(_extract_output_logprobs(result))

        _replay_and_compare_kl(
            self.base_url,
            self.model,
            KL_DIV_THRESHOLD,
            sequences,
            logprobs,
            label=f"umbp_linker_load_back_dcp{self.dcp_size}",
            sampling_temperature=0,
        )


@unittest.skipUnless(is_hip(), "The UMBP standalone server needs ROCm.")
class TestKimiLinearUMBPLinkerKLTP4(_KimiLinearUMBPLinkerKLMixin, CustomTestCase):
    pass


@unittest.skipUnless(is_hip(), "The UMBP standalone server needs ROCm.")
class TestKimiLinearUMBPLinkerKLDCP4(_KimiLinearUMBPLinkerKLMixin, CustomTestCase):
    dcp_size = 4
    # AITER's DCP decode kernel reads an fp8 KV cache.
    extra_args = ("--kv-cache-dtype", "fp8_e4m3")


if __name__ == "__main__":
    unittest.main()
