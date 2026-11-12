"""opt4: the linker hands the store each Mamba state's node-boundary path, and
UMBPDirectLinker's keep_states_per_path policy keeps that many states per path:
a deeper landing drops the oldest, a fork's state stays, a failed landing drops
nothing."""

import os
import sys
from collections import defaultdict, deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_unified_cache_linker as base  # noqa: E402

from sglang.srt.mem_cache.hicache_storage import PoolName  # noqa: E402
from sglang.srt.mem_cache.storage.umbp.umbp_direct_linker import (  # noqa: E402
    UMBPDirectLinker,
)
from sglang.srt.mem_cache.unified_cache.component_type import (  # noqa: E402
    ComponentType,
)
from sglang.test.kits.unified_radix_cache.config import CacheConfig  # noqa: E402
from sglang.test.kits.unified_radix_cache.fixtures import build_fixture  # noqa: E402


class _KeepingLinker(base._InMemoryUnifiedCacheLinker):
    """The in-memory store under UMBPDirectLinker's keep_states_per_path."""

    tracks_state_paths = True
    note_state_path = UMBPDirectLinker.note_state_path
    _state_landed = UMBPDirectLinker._state_landed

    def __init__(self, keep_states):
        super().__init__(defaultdict(set))
        self._keep_states = keep_states
        self._stats = defaultdict(int)
        self._offload_state_keys = deque()
        self._state_paths = {}
        self._kept_states = {}
        self._state_branches = {}
        self.removed = []

    # As UMBPDirectLinker.offload / pop_completed_offload around the store.
    def offload(self, transfers, session_id=None):
        queued = super().offload(transfers, session_id)
        state = next((t for t in transfers if t.name == PoolName.MAMBA), None)
        self._offload_state_keys.append(
            state.keys[-1] if state is not None and state.keys else None
        )
        return queued

    def pop_completed_offload(self):
        success = super().pop_completed_offload()
        key = self._offload_state_keys.popleft()
        if key is not None:
            self._state_landed(key, success)
        return success

    def remove(self, transfers):
        for transfer in transfers:
            self.removed.append(sorted(transfer.keys))
            self.stored_keys[transfer.name] -= set(transfer.keys)


class _KeepStatesPerPath:
    def _cache(self, keep_states):
        self.cfg = CacheConfig(
            page_size=1,
            components=(ComponentType.FULL, ComponentType.MAMBA),
            kv_size=128,
            max_context_len=64,
            mamba_cache_size=16,
        )
        self.cache, self.allocator, self.req_pool = build_fixture(self.cfg)
        self.store = _KeepingLinker(keep_states)
        self.cache.init_cache_linker(self.store)

    def _turn(self, tokens, lands=True):
        """Insert ``tokens``; return the state keys it added to the store."""
        before = set(self.store.stored_keys[PoolName.MAMBA])
        self._insert(self.cache, self.allocator, self.req_pool, tokens)
        while self.store.pending_offloads:
            self.store.complete_next_offload(lands)
        self.cache.check_hicache_events()
        return set(self.store.stored_keys[PoolName.MAMBA]) - before

    def test_path_names_the_states_above(self):
        self._cache(keep_states=2)
        (first,) = self._turn(list(range(1, 5)))
        (second,) = self._turn(list(range(1, 9)))
        self._insert(self.cache, self.allocator, self.req_pool, list(range(1, 13)))
        ((path, replaceable),) = self.store._state_paths.values()
        self.assertEqual(path, [first, second])
        self.assertTrue(replaceable)

    def test_third_state_drops_the_first(self):
        self._cache(keep_states=2)
        first = self._turn(list(range(1, 5)))
        self._turn(list(range(1, 9)))
        self.assertEqual(self.store.removed, [])
        self._turn(list(range(1, 13)))
        self.assertEqual(self.store.removed, [sorted(first)])
        self.assertEqual(len(self.store.stored_keys[PoolName.MAMBA]), 2)
        self.cache.sanity_check()

    def test_keep_one_drops_the_previous(self):
        self._cache(keep_states=1)
        first = self._turn(list(range(1, 5)))
        self._turn(list(range(1, 9)))
        self.assertEqual(self.store.removed, [sorted(first)])
        self.cache.sanity_check()

    def test_fork_keeps_its_state(self):
        self._cache(keep_states=2)
        fork = self._turn(list(range(1, 5)))
        a, b = list(range(1, 5)) + [5, 6, 7, 8], list(range(1, 5)) + [20, 21, 22, 23]
        self._turn(a)
        self._turn(b)
        self._turn(a + [9, 10, 11, 12])
        self._turn(b + [24, 25, 26, 27])
        self.assertTrue(fork <= self.store.stored_keys[PoolName.MAMBA])
        self.assertNotIn(sorted(fork), self.store.removed)
        self.cache.sanity_check()

    def test_failed_landing_drops_nothing(self):
        self._cache(keep_states=2)
        first = self._turn(list(range(1, 5)))
        self._turn(list(range(1, 9)))
        self._turn(list(range(1, 13)), lands=False)
        self.assertEqual(self.store.removed, [])
        self.assertTrue(first <= self.store.stored_keys[PoolName.MAMBA])
        self.cache.sanity_check()


class TestKeepStatesPerPathPythonBackend(
    _KeepStatesPerPath, base.TestUnifiedCacheLinkerPythonBackend
):
    pass
