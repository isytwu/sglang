"""With SGLANG_LINKER_STATE_WRITE_BACK=1 the linker writes a Mamba state to the
store when the device evicts it, never at insert, and never one the path cap
replaced."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.unified_cache.cache_action import (
    FreeComponentDeviceSlot,
    MambaEvictExcessPathStates,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    UnifiedCacheLinker,
    UnifiedCacheLinkerWrapper,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Backend(UnifiedCacheLinker):
    def __init__(self):
        self.layer_done_counter = object()
        self.pools = {}
        self.offloads = []
        self.completed = []

    def lookup(self, rid, transfers, resident_pages=0):
        return []

    def load(self, rid, transfers):
        return True

    def start_layer_wise_loading(self):
        return -1

    def finish_layer_wise_loading(self, counter_index):
        return True

    def cancel_queued_load(self, rid):
        return False

    def offload(self, transfers, session_id=None):
        self.offloads.append(list(transfers))
        return True

    def num_completed_offloads(self):
        return len(self.completed)

    def pop_completed_offload(self):
        return self.completed.pop(0)

    def reset(self):
        self.completed.clear()

    def close(self):
        pass


class _Storage:
    """State slots 10..(10 + free) are free; records copies and frees."""

    def __init__(self, free):
        self.free = list(range(10, 10 + free))
        self.copies = []
        self.freed = []

    def take_state_slots(self, sub_pool, count):
        if len(self.free) < count:
            return None
        taken, self.free = self.free[:count], self.free[count:]
        return torch.tensor(taken)

    def free_state_slots(self, sub_pool, slots):
        self.freed.extend(slots.reshape(-1).tolist())
        self.free.extend(slots.reshape(-1).tolist())

    def copy_state(self, sub_pool, src, dst):
        self.copies.append((src.tolist(), dst.tolist()))

    def quantizes_checkpoints(self, sub_pool):
        return False

    def tree_sub_pool(self, name):
        return name


class _Mamba:
    component_type = ComponentType.MAMBA

    def __init__(self, storage):
        self.storage = storage

    def _state(self):
        return self.storage, "MAMBA"

    def build_external_linker_transfer(self, phase, node, keys):
        return PoolTransfer(
            name=PoolName.MAMBA,
            keys=[keys[-1]],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )


class _Full:
    component_type = ComponentType.FULL

    def build_external_linker_transfer(self, phase, node, keys):
        return PoolTransfer(name=PoolName.KV, keys=list(keys))


class _TreeCore:
    """Nodes with a KV page each and a device state in ``states``."""

    is_eagle = False
    enable_external_cache_linker = False

    def __init__(self, states):
        self.states = states
        self.pending = {}

    def build_external_linker_offload_transfers(self, node_id):
        transfers = [
            PoolTransfer(
                name=PoolName.KV,
                device_indices=torch.arange(4),
                keys=[f"{node_id}-last"],
            )
        ]
        if node_id in self.states:
            transfers.append(
                PoolTransfer(
                    name=PoolName.MAMBA,
                    device_indices=torch.tensor([self.states[node_id]]),
                    keys=[f"{node_id}-last"],
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            )
        return transfers

    def mark_external_linker_offload_pending(self, node_id):
        self.pending[node_id] = True

    def finish_external_linker_offload(self, node_ids, ack_id, success):
        self.pending.pop(ack_id, None)

    def get_component_device_value(self, node_id, component_type):
        slot = self.states.get(node_id)
        return None if slot is None else torch.tensor([slot])

    def get_hash_values(self, node_id):
        return [f"{node_id}-0", f"{node_id}-last"]


@pytest.fixture
def write_back(monkeypatch):
    monkeypatch.setenv("SGLANG_LINKER_STATE_WRITE_BACK", "1")


def _setup(states, free_slots=4):
    storage = _Storage(free_slots)
    mamba, full = _Mamba(storage), _Full()
    locks = []
    cache = SimpleNamespace(
        page_size=4,
        tree_components=(ComponentType.FULL, ComponentType.MAMBA),
        components={ComponentType.FULL: full, ComponentType.MAMBA: mamba},
        _components_tuple=(full, mamba),
        tree_core=_TreeCore(states),
        write_through_threshold=256,
        inc_lock_ref=lambda node: (
            locks.append(("inc", node)) or SimpleNamespace(to_dec_params=lambda: node)
        ),
        dec_lock_ref=lambda node, params: locks.append(("dec", node)),
        _all_reduce_attn_groups=lambda tensor, op: None,
    )
    backend = _Backend()
    backend.supports_remove = True
    wrapper = UnifiedCacheLinkerWrapper(cache, backend)
    return wrapper, backend, storage, locks


def _offloaded_pools(backend):
    return [[transfer.name for transfer in transfers] for transfers in backend.offloads]


def test_insert_writes_the_kv_and_keeps_the_state_on_the_device(write_back):
    wrapper, backend, _, _ = _setup({"a": 3, "b": 5})

    wrapper.offload_nodes(["a"])
    wrapper.offload_new_state("b")

    assert _offloaded_pools(backend) == [[PoolName.KV]]
    assert wrapper._unwritten_states == {3: ("a", "a-last"), 5: ("b", "b-last")}


def test_an_evicted_state_is_written_from_a_copy_and_its_slot_freed_at_the_ack(
    write_back,
):
    wrapper, backend, storage, _ = _setup({"a": 3})
    wrapper.offload_nodes(["a"])
    backend.offloads.clear()

    wrapper.before_state_free(torch.tensor([3]))

    # Copied into a free slot before the device frees slot 3.
    assert storage.copies == [([3], [10])]
    ((state,),) = backend.offloads
    assert state.name == PoolName.MAMBA
    assert state.keys == ["a-last"]
    assert state.device_indices.tolist() == [10]
    assert wrapper.offload_stats["states_written_back"] == 1
    assert storage.freed == []

    backend.completed.extend([True, False])  # the KV offload, then the state's
    wrapper.commit_completed_offloads(wrapper.take_completed_offloads(2))
    # A failed write-back only loses the copy: its slot still comes back.
    assert storage.freed == [10]
    assert wrapper.pending_offloads == []
    assert wrapper._staged_states == 0


def test_a_state_the_path_cap_replaced_is_never_written(write_back):
    wrapper, backend, storage, _ = _setup({"a": 3, "b": 5})
    wrapper.offload_nodes(["a"])
    backend.offloads.clear()

    with wrapper.superseded_states():
        wrapper.before_state_free(torch.tensor([3]))
    # Neither is a later free of a slot it no longer remembers.
    wrapper.before_state_free(torch.tensor([3]))

    assert backend.offloads == []
    assert storage.copies == []
    assert 3 not in wrapper._unwritten_states
    # Nothing was written, so nothing is removed when the cap replaces it.
    assert wrapper.path_state_holders("b") == []


def test_without_a_free_slot_or_past_the_limit_the_state_is_dropped(write_back):
    wrapper, backend, storage, _ = _setup({"a": 3, "b": 4, "c": 5}, free_slots=1)
    for node in ("a", "b", "c"):
        wrapper.offload_new_state(node)

    wrapper.before_state_free(torch.tensor([3]))
    wrapper.before_state_free(torch.tensor([4]))  # no free slot left
    wrapper.state_write_back_slots = 1
    storage.free.append(11)
    wrapper.before_state_free(torch.tensor([5]))  # one copy already in flight

    assert len(backend.offloads) == 1
    assert wrapper.offload_stats["states_dropped"] == 2
    assert wrapper._unwritten_states == {}


def test_reset_returns_the_copies_in_flight(write_back):
    wrapper, backend, storage, _ = _setup({"a": 3})
    wrapper.offload_new_state("a")
    wrapper.before_state_free(torch.tensor([3]))

    wrapper.reset()

    assert storage.freed == [10]
    assert wrapper.pending_offloads == []
    assert wrapper._unwritten_states == {}


def test_without_the_flag_states_are_written_through(monkeypatch):
    monkeypatch.delenv("SGLANG_LINKER_STATE_WRITE_BACK", raising=False)
    wrapper, backend, storage, _ = _setup({"a": 3, "b": 5})

    wrapper.offload_nodes(["a"])
    wrapper.offload_new_state("b")
    wrapper.before_state_free(torch.tensor([3]))

    assert _offloaded_pools(backend) == [
        [PoolName.KV, PoolName.MAMBA],
        [PoolName.MAMBA],
    ]
    assert storage.copies == []


# ---- the Mamba component's hooks ----


def _component(linker, storage):
    component = MambaComponent.__new__(MambaComponent)
    component.component_type = ComponentType.MAMBA
    component.cache = SimpleNamespace(
        linker=linker,
        kv_storage=lambda: storage,
        _apply_eviction_result=lambda result: [
            component.apply_component_action(
                FreeComponentDeviceSlot(result.device_frees.pop(ct), component_type=ct)
            )
            for ct in list(result.device_frees)
        ],
    )
    return component


def test_mamba_frees_offer_states_to_the_linker_first():
    seen = []
    storage = _Storage(free=0)
    linker = SimpleNamespace(
        before_state_free=lambda slots: seen.append(
            (slots.tolist(), list(storage.freed))
        )
    )
    component = _component(linker, storage)

    component.apply_component_action(
        FreeComponentDeviceSlot([torch.tensor([7])], component_type=ComponentType.MAMBA)
    )

    # Offered while the slot is still held.
    assert seen == [([7], [])]
    assert storage.freed == [7]


def test_the_path_cap_frees_its_states_as_superseded():
    flags = []
    storage = _Storage(free=0)

    class _Linker:
        superseding = False

        def path_state_holders(self, tail):
            return []

        def superseded_states(self):
            linker = self

            class _Scope:
                def __enter__(self):
                    linker.superseding = True

                def __exit__(self, *exc):
                    linker.superseding = False

            return _Scope()

        def before_state_free(self, slots):
            flags.append((slots.tolist(), self.superseding))

    component = _component(_Linker(), storage)

    def evict_excess_path_states(tail, device_frees, host_frees, cache_actions):
        device_frees[ComponentType.MAMBA].append(torch.tensor([9]))

    component.tree_core = SimpleNamespace(
        evict_excess_path_states=evict_excess_path_states
    )

    component.apply_component_action(MambaEvictExcessPathStates("tail"))

    assert flags == [([9], True)]
    assert storage.freed == [9]
