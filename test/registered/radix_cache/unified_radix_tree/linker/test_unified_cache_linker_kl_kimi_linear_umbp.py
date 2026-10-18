"""KL for the UMBP direct linker on Kimi Linear (FULL + Mamba: MLA + KDA).

Turn 1 offloads to a local UMBP standalone server, /flush_cache drops the
device tree, and turn 2 restores every prefix from UMBP alone: the MLA KV pages
and the KDA state at the restore boundary. Turn 2's outputs are then replayed
against a full recompute. Under TP every rank shares one key per MLA KV page;
under DCP each rank keeps its own keys and the DCP-widened KV rows collapse.

Turn 2 is scored a few tokens past the restore boundary: the MLA layers see the
whole prefix, so further out they mask a wrong KDA state.
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
# A multiple of the DCP4-widened tree page (4 * PAGE_SIZE).
TRACK_INTERVAL = 512
# Turn 1's continuation and turn 2's suffix.
TAIL_TOKENS = 16


class _UMBPStandaloneServer:
    """A node-local UMBP standalone server on a Unix socket."""

    def __init__(self, dram_gib: int = 64):
        import mori

        self.binary = os.path.join(
            os.path.dirname(mori.__file__), "umbp_standalone_server"
        )
        self.dir = tempfile.mkdtemp(prefix="umbp-sa-")
        self.socket_path = os.path.join(self.dir, "sa.sock")
        self.address = f"unix://{self.socket_path}"
        self.log_path = os.path.join(self.dir, "sa.log")
        self.dram_gib = dram_gib
        self.process = None

    def start(self):
        env = {
            **os.environ,
            "UMBP_DRAM_CAPACITY": str(self.dram_gib << 30),
            "UMBP_SSD_ENABLED": "0",
            "HSA_USERPTR_FOR_PAGED_MEM": "0",
            "HSA_ENABLE_IPC_MODE_LEGACY": "1",
            "MORI_UMBP_LOG_LEVEL": "info",
        }
        with open(self.log_path, "w") as log:
            self.process = subprocess.Popen(
                [self.binary, self.address],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        # The server binds its socket once it serves.
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if os.path.exists(self.socket_path):
                return
            if self.process.poll() is not None:
                with open(self.log_path) as f:
                    raise RuntimeError(
                        f"UMBP standalone server exited:\n{f.read()[-2000:]}"
                    )
            time.sleep(1)
        raise RuntimeError("UMBP standalone server did not start in 300 s")

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()


class _KimiLinearUMBPLinkerKLMixin:
    tp_size = 4
    dcp_size = 1
    extra_args: tuple[str, ...] = ()

    @classmethod
    def setUpClass(cls):
        cls.model = _MODEL_PATH
        cls.base_url = f"http://127.0.0.1:{find_available_port(30000)}"
        cls.umbp = _UMBPStandaloneServer()
        cls.addClassCleanup(cls.umbp.stop)
        cls.umbp.start()
        other_args = [
            "--trust-remote-code",
            "--tp-size",
            str(cls.tp_size),
            *(["--dcp-size", str(cls.dcp_size)] if cls.dcp_size > 1 else []),
            "--page-size",
            str(PAGE_SIZE),
            # The Triton backend's DCP decode does not run MLA on ROCm.
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
            json.dumps({"standalone_address": cls.umbp.address}),
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
            label=f"umbp_linker_load_back_tp{self.tp_size}_dcp{self.dcp_size}",
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
