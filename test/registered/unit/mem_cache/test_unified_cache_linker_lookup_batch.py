"""Admissions of one scheduling round share their external-store lookups."""

from array import array
from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.mem_cache.common import admission_radix_key
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage.umbp.umbp_direct_linker import UMBPDirectLinker
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components import LinkerTransferPhase
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    UnifiedCacheLinker,
    UnifiedCacheLinkerWrapper,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PAGE = 4


class _Linker(UnifiedCacheLinker):
    """Restores the longest run of a tail's KV keys found in ``stored``."""

    def __init__(self, stored):
        self.layer_done_counter = object()
        self.pools = {}
        self.stored = set(stored)
        self.batches = []

    def lookup(self, rid, transfers, resident_pages=0):
        return self.lookup_batch([(rid, transfers, resident_pages)])[0][0]

    def lookup_batch(self, requests):
        self.batches.append([rid for rid, _, _ in requests])
        results = []
        for _, transfers, _ in requests:
            keys = next(t for t in transfers if t.name == PoolName.KV).keys
            pages = 0
            while pages < len(keys) and keys[pages] in self.stored:
                pages += 1
            results.append((list(range(1, pages + 1)), None))
        return results

    def load(self, rid, transfers):
        return True

    def start_layer_wise_loading(self):
        return -1

    def finish_layer_wise_loading(self, counter_index):
        return True

    def cancel_queued_load(self, rid):
        return False

    def offload(self, transfers, session_id=None):
        return True

    def num_completed_offloads(self):
        return 0

    def pop_completed_offload(self):
        return True

    def reset(self):
        pass

    def close(self):
        pass


class _Full:
    component_type = ComponentType.FULL

    def build_external_linker_transfer(self, phase, node, keys):
        assert phase == LinkerTransferPhase.LOOKUP
        return PoolTransfer(name=PoolName.KV, keys=list(keys))


def _setup(reduce=None):
    reductions = []

    def all_reduce(tensor, op):
        assert op == torch.distributed.ReduceOp.MIN
        reductions.append(len(tensor))
        if reduce is not None:
            reduce(tensor)

    full = _Full()
    cache = SimpleNamespace(
        page_size=PAGE,
        tree_components=(ComponentType.FULL,),
        components={ComponentType.FULL: full},
        _components_tuple=(full,),
        tree_core=SimpleNamespace(enable_external_cache_linker=False, is_eagle=False),
        write_through_threshold=256,
        get_last_hash_value=lambda node: "ab" * 32,
        _all_reduce_attn_groups=all_reduce,
    )
    linker = _Linker(stored=())
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)
    return wrapper, linker, reductions


def _store_tail(wrapper, linker, key, device_hit_len, pages):
    """Store the first ``pages`` pages of ``key``'s tail past the device hit."""
    hashes = wrapper._tail_hashes(key, "node", device_hit_len)
    linker.stored.update(hashes[:pages])


def _key(tokens, offset=0):
    return RadixKey(array("q", range(offset, offset + tokens)))


def _result(device_hit_len):
    return MatchResult(
        device_prefix_len=device_hit_len,
        last_device_node="node",
        last_host_node="node",
        best_match_node="node",
    )


def test_admission_reuses_a_probe_lookup_of_the_same_round():
    wrapper, linker, reductions = _setup()
    req = SimpleNamespace(rid="a")
    key = _key(26)
    _store_tail(wrapper, linker, key, 8, 4)

    wrapper.begin_lookup_round()
    probe = wrapper.match(key, req, _result(8), probe=True)
    # Admission matches one token less, which keeps the same tail pages.
    admission = wrapper.match(key[:25], req, _result(8))

    assert linker.batches == [["a"]]
    assert len(reductions) == 1
    assert probe.host_hit_length == admission.host_hit_length == 4 * PAGE
    assert wrapper.hit_markers["a"].device_hit_len == 8
    assert wrapper.lookup_stats["round_reuses"] == 1


def test_admission_looks_up_again_in_a_later_round():
    wrapper, linker, _ = _setup()
    req = SimpleNamespace(rid="a")
    key = _key(24)
    _store_tail(wrapper, linker, key, 8, 4)

    wrapper.begin_lookup_round()
    wrapper.match(key, req, _result(8), probe=True)
    assert wrapper.has_hit("a")
    # The store evicts the tail before the next round admits the request.
    linker.stored.clear()
    wrapper.begin_lookup_round()
    result = wrapper.match(key, req, _result(8))

    assert linker.batches == [["a"], ["a"]]
    assert result.host_hit_length == 0
    assert not wrapper.has_hit("a")


def test_without_a_round_admission_always_looks_up():
    wrapper, linker, _ = _setup()
    req = SimpleNamespace(rid="a")
    key = _key(24)

    wrapper.match(key, req, _result(8), probe=True)
    wrapper.match(key, req, _result(8))
    wrapper.prefetch_lookups([(req, key)])

    assert linker.batches == [["a"], ["a"]]


def test_prefetched_admissions_share_one_lookup_and_one_reduction():
    wrapper, linker, reductions = _setup()
    reqs = [SimpleNamespace(rid=rid) for rid in ("a", "b", "c")]
    keys = [_key(40, 0), _key(32, 100), _key(20, 200)]
    device_hits = [8, 4, 0]
    stored_pages = [5, 0, 2]

    wrapper.begin_lookup_round()
    for req, key, device_hit, pages in zip(reqs, keys, device_hits, stored_pages):
        _store_tail(wrapper, linker, key, device_hit, pages)
        # Probes of earlier rounds, reused while the queue is ordered.
        wrapper._round_device_hits[req.rid] = (device_hit, "node")
    reductions.clear()

    wrapper.prefetch_lookups(list(zip(reqs, keys)))

    assert linker.batches == [["a", "b", "c"]]
    # One mask per tail, laid end to end in a single reduction.
    assert reductions == [(40 - 8) // PAGE + 1 + (32 - 4) // PAGE + 1 + 20 // PAGE + 1]
    results = [
        wrapper.match(key, req, _result(device_hit))
        for req, key, device_hit in zip(reqs, keys, device_hits)
    ]
    assert len(linker.batches) == 1
    assert [result.host_hit_length for result in results] == [20, 0, 8]
    assert wrapper.has_hit("a") and not wrapper.has_hit("b") and wrapper.has_hit("c")
    assert wrapper.lookup_stats["batched"] == 3
    assert wrapper.lookup_stats["round_reuses"] == 3


def test_prefetched_hits_are_intersected_across_ranks_per_tail():
    def peer_lacks_second_tail(mask):
        # Masks: tail "a" at [0, 9), tail "b" at [9, 15); the peer restores
        # nothing of "b".
        peer = torch.ones_like(mask)
        peer[9:] = 0
        mask.copy_(torch.minimum(mask, peer))

    wrapper, linker, _ = _setup(reduce=peer_lacks_second_tail)
    reqs = [SimpleNamespace(rid="a"), SimpleNamespace(rid="b")]
    keys = [_key(40, 0), _key(28, 100)]
    wrapper.begin_lookup_round()
    for req, key in zip(reqs, keys):
        _store_tail(wrapper, linker, key, 8, 8)
        wrapper._round_device_hits[req.rid] = (8, "node")

    wrapper.prefetch_lookups(list(zip(reqs, keys)))

    assert wrapper.match(keys[0], reqs[0], _result(8)).host_hit_length == 32
    assert wrapper.match(keys[1], reqs[1], _result(8)).host_hit_length == 0
    assert not wrapper.has_hit("b")


def test_prefetch_skips_tails_this_round_already_looked_up():
    wrapper, linker, _ = _setup()
    req = SimpleNamespace(rid="a")
    key = _key(26)

    wrapper.begin_lookup_round()
    wrapper.match(key, req, _result(8), probe=True)
    wrapper.prefetch_lookups([(req, key[:25])])

    assert linker.batches == [["a"]]


def test_admission_whose_device_hit_moved_looks_up_on_its_own():
    wrapper, linker, _ = _setup()
    req = SimpleNamespace(rid="a")
    key = _key(32)
    wrapper.begin_lookup_round()
    wrapper._round_device_hits["a"] = (8, "node")
    wrapper.prefetch_lookups([(req, key)])

    # An earlier admission's allocation evicted part of the device prefix.
    wrapper.match(key, req, _result(4))

    assert linker.batches == [["a"], ["a"]]


def test_prefetch_needs_this_rounds_probe():
    wrapper, linker, _ = _setup()
    req = SimpleNamespace(rid="a")
    wrapper.begin_lookup_round()
    wrapper._round_device_hits["a"] = (8, "node")
    wrapper.begin_lookup_round()

    wrapper.prefetch_lookups([(req, _key(32))])

    assert linker.batches == []


# ---- UMBP: one existence RPC per pool for every request ----


class _Client:
    def __init__(self, stored):
        self.stored = set(stored)
        self.calls = []

    def batch_exists(self, keys):
        self.calls.append(list(keys))
        return [key in self.stored for key in keys]


def _umbp_linker(stored, num_layers=2):
    linker = UMBPDirectLinker.__new__(UMBPDirectLinker)
    linker.num_layers = num_layers
    linker._stats = {"lookup": 0}
    linker.pools = {
        PoolName.KV: SimpleNamespace(name=PoolName.KV, packed=True, components=[[0]]),
        PoolName.MAMBA: SimpleNamespace(
            name=PoolName.MAMBA, packed=False, components=[[0], [1]]
        ),
    }
    linker.pool_group = SimpleNamespace(
        resolve_transfers=lambda transfers, **flags: list(transfers)
    )

    def component_keys(page_keys, transfer):
        suffixes = ["k"] if transfer.name == PoolName.KV else ["t", "c"]
        return [f"{key}/{suffix}" for key in page_keys for suffix in suffixes], len(
            suffixes
        )

    client = _Client(stored)
    linker.storage = SimpleNamespace(
        client=client, _get_hybrid_page_component_keys=component_keys
    )
    return linker, client


def _umbp_transfers(prefix, pages):
    keys = [f"{prefix}{page}" for page in range(pages)]
    return [
        PoolTransfer(name=PoolName.KV, keys=keys),
        PoolTransfer(
            name=PoolName.MAMBA,
            keys=[keys[-1]],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        ),
    ]


def _umbp_store(prefix, kv_pages, state_pages):
    stored = {f"{prefix}{page}/k" for page in range(kv_pages)}
    for page in state_pages:
        stored |= {f"{prefix}{page}/t", f"{prefix}{page}/c"}
    return stored


def test_umbp_lookup_batch_matches_one_lookup_each_in_fewer_rpcs():
    stored = (
        _umbp_store("a", 6, [2, 5])
        | _umbp_store("b", 3, [1])
        | _umbp_store("c", 0, [])
        | _umbp_store("d", 300, [99, 299])
    )
    requests = [
        ("a", _umbp_transfers("a", 8), 0),
        ("b", _umbp_transfers("b", 4), 0),
        ("c", _umbp_transfers("c", 4), 0),
        # The first 100 pages are resident: the KV pool does not probe them.
        ("d", _umbp_transfers("d", 300), 100),
    ]

    single, single_client = _umbp_linker(stored)
    expected = [
        single.lookup(rid, transfers, resident_pages=resident)
        for rid, transfers, resident in requests
    ]
    batched, batched_client = _umbp_linker(stored)
    answers = batched.lookup_batch(requests)

    assert [restorable for restorable, _ in answers] == expected
    assert expected[:3] == [[3, 6], [2], []]
    assert expected[3] == [100, 300]
    # Per pool, the budget is 64 pages x 2 layers of objects per RPC: the KV
    # pool fits its 216 probed pages in two calls, the state pool (two objects
    # a page) its 309 pages in five; one lookup each takes twelve.
    assert len(batched_client.calls) == 2 + 5
    assert len(single_client.calls) == 12
    assert batched._stats["lookup"] == 4


def test_umbp_lookup_batch_never_splits_a_page_across_calls():
    stored = _umbp_store("a", 200, [199])
    linker, client = _umbp_linker(stored, num_layers=1)

    ((restorable, _),) = linker.lookup_batch([("a", _umbp_transfers("a", 200), 0)])

    assert restorable == [200]
    state_calls = [call for call in client.calls if call[0].endswith(("/t", "/c"))]
    assert all(len(call) % 2 == 0 for call in state_calls)


# ---- scheduler: which requests a round looks up together ----


def _waiting(rid, tokens, cached):
    req = Req(rid, "", array("q", range(tokens)), SamplingParams())
    req.prefix_len = cached
    req.host_hit_length = 0
    return req


def test_admission_candidates_cover_the_round_budget():
    queue = [
        _waiting("a", 4001, 1000),
        _waiting("b", 5001, 0),
        _waiting("c", 9001, 0),
        _waiting("d", 2001, 0),
    ]
    cache = SimpleNamespace(
        swa_reprefill_tail_tokens=lambda req: 0,
        linker_lookup_batch_size=lambda: 8,
    )
    scheduler = SimpleNamespace(tree_cache=cache, waiting_queue=queue)
    adder = SimpleNamespace(rem_input_tokens=16384, rem_chunk_tokens=16384)

    picked = Scheduler._linker_admission_candidates(scheduler, adder)

    # 3000 + 5000 leave room; the 9000 that overruns the budget is matched too.
    assert [req.rid for req, _ in picked] == ["a", "b", "c"]
    assert [len(key) for _, key in picked] == [4000, 5000, 9000]

    cache.linker_lookup_batch_size = lambda: 2
    assert len(Scheduler._linker_admission_candidates(scheduler, adder)) == 2
    adder.rem_chunk_tokens = 0
    assert Scheduler._linker_admission_candidates(scheduler, adder) == []


def test_admission_key_is_the_key_admission_matches():
    matched = []

    class _Cache:
        def supports_mamba(self):
            return False

        def swa_reprefill_tail_tokens(self, req):
            return 0

        def match_prefix(self, params):
            matched.append(params.key)
            return MatchResult(
                device_prefix_len=0,
                last_device_node=None,
                last_host_node=None,
                best_match_node=None,
            )

    req = _waiting("a", 130, 0)
    req.output_ids.extend([7, 8])
    key = admission_radix_key(req, _Cache())
    req.init_next_round_input(_Cache())

    assert len(key) == len(matched[0]) == 131
    assert key.token_ids is matched[0].token_ids
