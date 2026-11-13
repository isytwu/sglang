"""Finished requests' output states under UMBPDirectLinker's keep_states_per_path.

The linker tells a store that tracks output states which landed Mamba state ends a
finished request's output (past its prompt end). The store keeps such a state below
its prompt until a later state passes that prompt: continued, it counts as any state;
left, it is dropped. It never makes the states above it forks, so a conversation that
leaves every output keeps two prompt states per path like one whose outputs are not
offloaded at all."""

import os
import sys
import unittest
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_unified_cache_linker as base  # noqa: E402

from sglang.srt.mem_cache.storage.umbp.umbp_direct_linker import (  # noqa: E402
    UMBPDirectLinker,
)
from sglang.srt.mem_cache.unified_cache.component_type import (  # noqa: E402
    ComponentType,
)
from sglang.test.kits.unified_radix_cache.config import CacheConfig  # noqa: E402
from sglang.test.kits.unified_radix_cache.fixtures import build_fixture  # noqa: E402


class _Policy:
    """UMBPDirectLinker's state policy, without a store."""

    note_state_path = UMBPDirectLinker.note_state_path
    _state_landed = UMBPDirectLinker._state_landed
    _open_output = UMBPDirectLinker._open_output
    _settle_outputs = UMBPDirectLinker._settle_outputs

    def __init__(self, keep_states):
        self._keep_states = keep_states
        self._stats = defaultdict(int)
        self._state_paths = {}
        self._kept_states = {}
        self._state_branches = {}
        self._open_outputs = {}
        self._open_output_count = 0
        self.removed = []

    def remove(self, transfers):
        for transfer in transfers:
            self.removed.extend(transfer.keys)

    def land(self, key, path, output=False, success=True):
        self.note_state_path(key, list(path), True, output=output)
        self._state_landed(key, success)


class TestOutputStatePolicy(unittest.TestCase):
    def test_left_outputs_drop_and_prompts_keep_two(self):
        p = _Policy(keep_states=2)
        p.land("p1", [])
        p.land("o1", ["p1"], output=True)
        p.land("p2", ["p1"])
        self.assertEqual(p.removed, ["o1"])
        p.land("o2", ["p1", "p2"], output=True)
        p.land("p3", ["p1", "p2"])
        self.assertEqual(p.removed, ["o1", "o2", "p1"])
        self.assertEqual(p._stats["output_state_dropped"], 2)
        self.assertEqual(p._stats["state_dropped"], 1)
        self.assertEqual(p._open_outputs, {})

    def test_continued_output_counts_as_a_state(self):
        p = _Policy(keep_states=2)
        p.land("p1", [])
        p.land("o1", ["p1"], output=True)
        p.land("p2", ["p1", "o1"])
        self.assertEqual(p.removed, ["p1"])
        self.assertIn("o1", p._kept_states)
        p.land("p3", ["p1", "o1", "p2"])
        self.assertEqual(p.removed, ["p1", "o1"])

    def test_an_output_alone_makes_no_fork(self):
        p = _Policy(keep_states=2)
        p.land("p1", [])
        p.land("o1", ["p1"], output=True)
        p.land("p2", ["p1"])
        self.assertEqual(p._state_branches.get("p1"), {"p2"})

    def test_two_prompt_branches_keep_the_fork(self):
        p = _Policy(keep_states=2)
        p.land("p1", [])
        p.land("o1", ["p1"], output=True)
        p.land("a2", ["p1"])
        p.land("b2", ["p1"])
        p.land("a3", ["p1", "a2"])
        p.land("b3", ["p1", "b2"])
        self.assertEqual(p.removed, ["o1"])
        self.assertIn("p1", p._kept_states)

    def test_failed_landings_change_nothing(self):
        p = _Policy(keep_states=2)
        p.land("p1", [])
        p.land("o1", ["p1"], output=True, success=False)
        self.assertEqual(p._open_outputs, {})
        p.land("o1b", ["p1"], output=True)
        p.land("p2", ["p1"], success=False)
        self.assertEqual(p.removed, [])
        self.assertEqual(p._open_outputs, {"p1": {"o1b": None}})

    def test_output_waits_for_a_state_below_its_prompt(self):
        p = _Policy(keep_states=2)
        p.land("p1", [])
        p.land("o1", ["p1"], output=True)
        p.land("q1", [])
        self.assertEqual(p.removed, [])
        self.assertEqual(p._open_outputs, {"p1": {"o1": None}})


class _RecordingLinker(base._InMemoryUnifiedCacheLinker):
    tracks_state_paths = True
    tracks_output_states = True

    def __init__(self):
        super().__init__(defaultdict(set))
        self.noted = []

    def note_state_path(self, state_key, path_keys, replaceable, output=False):
        self.noted.append((len(path_keys), replaceable, output))


class _LegacyLinker(base._InMemoryUnifiedCacheLinker):
    """A store without tracks_output_states keeps the three-argument call."""

    tracks_state_paths = True

    def __init__(self):
        super().__init__(defaultdict(set))
        self.noted = []

    def note_state_path(self, state_key, path_keys, replaceable):
        self.noted.append((len(path_keys), replaceable))


class _OutputStatesCore:
    def _cache(self, store):
        self.cfg = CacheConfig(
            page_size=1,
            components=(ComponentType.FULL, ComponentType.MAMBA),
            kv_size=128,
            max_context_len=64,
            mamba_cache_size=16,
        )
        self.cache, self.allocator, self.req_pool = build_fixture(self.cfg)
        self.store = store
        self.cache.init_cache_linker(store)

    def _turn(self, tokens, prompt_end=0):
        with self.cache.linker.offload_window(0, 0, prompt_end):
            self._insert(self.cache, self.allocator, self.req_pool, tokens)
        while self.store.pending_offloads:
            self.store.complete_next_offload(True)
        self.cache.check_hicache_events()

    def test_state_past_the_prompt_end_is_an_output(self):
        self._cache(_RecordingLinker())
        self._turn(list(range(1, 5)))
        self._turn(list(range(1, 9)), prompt_end=4)
        self.assertEqual(self.store.noted, [(0, True, False), (1, True, True)])
        self.cache.sanity_check()

    def test_state_at_the_prompt_end_is_not(self):
        self._cache(_RecordingLinker())
        self._turn(list(range(1, 9)), prompt_end=8)
        self.assertEqual(self.store.noted, [(0, True, False)])
        self.assertEqual(self.cache.linker.offload_prompt_end, 0)

    def test_legacy_store_gets_three_arguments(self):
        self._cache(_LegacyLinker())
        self._turn(list(range(1, 5)))
        self._turn(list(range(1, 9)), prompt_end=4)
        self.assertEqual(self.store.noted, [(0, True), (1, True)])


class TestOutputStatesCorePythonBackend(
    _OutputStatesCore, base.TestUnifiedCacheLinkerPythonBackend
):
    pass


if __name__ == "__main__":
    unittest.main()
