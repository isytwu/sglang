"""Direct L3 support for :class:`UnifiedRadixCache`.

Links the cache's device pools straight to an external KV store, with no host
tier in between. The transport contract and tree-side wrapper live here, while
each backend owns its device-pool layout and physical I/O.

* :class:`UnifiedCacheLinker` -- the transport interface a backend implements.
* :class:`UnifiedCacheLinkerWrapper` -- the tree-side flow that drives it. The
  cache owns one as a plain attribute, keeping the whole external-cache path out
  of the main tree file.

The tree only needs a handful of guarded hooks:

* ``match_prefix``      -> :meth:`UnifiedCacheLinkerWrapper.match`
* ``init_load_back``    -> :meth:`UnifiedCacheLinkerWrapper.load_back`
* ``BackupKV`` actions  -> :meth:`UnifiedCacheLinkerWrapper.offload_nodes`

A load lands in device slots private to the loading request and joins the tree
only when that request inserts it like any computed KV. A failed load therefore
reaches nothing but its own requests, which the scheduler aborts.
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, NamedTuple, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import (
    ComponentType,
    ExternalLinkerLoadPhase,
    LinkerTransferPhase,
    TreeComponent,
)
from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.srt.utils import get_device_module

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import NodeId
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

logger = logging.getLogger(__name__)
device_module = get_device_module()

_EXTERNAL_LINKER_SUPPORTED_COMPONENTS = frozenset(
    {
        ComponentType.FULL,
        ComponentType.SWA,
        ComponentType.MAMBA,
    }
)

# Offload results a rank takes from its backend ahead of the ranks' agreement.
OFFLOAD_REPORT_WINDOW = 32


class LayerWiseLoadCounter:
    """CPU completion counter compatible with KV pools' layer wait hook.

    A failed layer does not raise into the forward that waits on it: the
    forward runs on whatever the slots hold, and :meth:`finish` reports the
    failure afterwards so the scheduler can abort the loading requests before
    their output is used.
    """

    def __init__(
        self,
        num_layers: int,
        on_layer_ready: Optional[Callable[[int, int], None]] = None,
    ):
        self.num_layers = num_layers
        self.on_layer_ready = on_layer_ready
        self.producer_index = -1
        self.consumer_index = -1
        self.futures: dict[int, list[Future]] = {}

    def update_producer(self) -> int:
        self.producer_index += 1
        self.futures[self.producer_index] = [Future() for _ in range(self.num_layers)]
        return self.producer_index

    def set_consumer(self, index: int) -> None:
        self.consumer_index = index

    def complete(self, index: int, layer: int) -> None:
        self.futures[index][layer].set_result(None)

    def fail(self, index: int, error: BaseException) -> None:
        for future in self.futures.get(index, ()):
            if not future.done():
                future.set_exception(error)

    def wait_until(self, threshold: int) -> None:
        index = self.consumer_index
        futures = self.futures.get(index)
        if futures is None:
            return
        # Blocks until the layer resolves; a failure is reported by finish().
        futures[threshold].exception()
        # Runs on failure too: an MLA dedup source must still issue the layer
        # broadcast its peers are waiting for.
        if self.on_layer_ready is not None:
            self.on_layer_ready(index, threshold)

    def finish(self, index: int) -> bool:
        """Wait for every layer of batch ``index``; True when all of them landed."""
        futures = self.futures.get(index)
        if futures is None:
            return True
        # Wait before dropping the entry: the load thread still completes the
        # layers the forward never waited on through it.
        # Do not short-circuit after an error: later layers may still be writing.
        errors = [future.exception() for future in futures]
        landed = all(error is None for error in errors)
        del self.futures[index]
        return landed

    def reset(self) -> None:
        self.producer_index = -1
        self.consumer_index = -1
        self.futures.clear()


class TransferReady(NamedTuple):
    events: list[torch.cuda.Event]

    def synchronize(self) -> None:
        for event in self.events:
            event.synchronize()

    def wait(self, stream: torch.cuda.Stream) -> None:
        for event in self.events:
            stream.wait_event(event)


class UnifiedCacheLinker(ABC):
    """External KV store reached directly from the device pools.

    Out-of-tree linkers are constructed per worker as
    ``Linker(server_args, params, components=..., extra_config=..., forward_stream=...)``.
    Store ``forward_stream`` and use ``record_transfer_ready()`` to gate loads
    and offloads. ``layer_done_counter`` provides ``set_consumer(index)`` and
    ``wait_until(layer)``. Report completion only after device copies finish.
    A store whose eviction should hear about device hits sets
    ``touch_interval_s`` and implements ``touch()``.
    """

    layer_done_counter: object
    pools: dict[PoolName, object]
    # None when the current stream is the only one using the pages.
    forward_stream: Optional[torch.cuda.Stream] = None
    # Log each request's load time and bytes.
    log_load_timing: bool = False
    # Seconds between touches of one device node, set in __init__ and read once
    # when the cache attaches the linker; None never calls touch().
    touch_interval_s: Optional[float] = None
    # Whether remove() drops stored objects; set in __init__.
    supports_remove: bool = False

    def record_transfer_ready(self) -> TransferReady:
        """Record what a transfer waits on before it touches device pages.

        The current stream computed the page indices; under read-done the
        forward in flight may still use those pages, including freed ones.
        """
        indices_ready = device_module.Event()
        indices_ready.record()
        events = [indices_ready]
        if self.forward_stream is not None:
            forward_done = device_module.Event()
            forward_done.record(self.forward_stream)
            events.append(forward_done)
        return TransferReady(events)

    def touch(self, transfers: list[PoolTransfer]) -> None:
        """Best-effort hint that these pages were just served from the device
        cache, for a store whose eviction cannot see device hits.

        LOOKUP-phase transfers, one per linked component, whose keys run from
        the sequence head to the device boundary; each transfer's own keys
        name the pages it holds (TRAILING_PAGES: the window ending at the
        boundary). Called inside the rank-lockstep match, so it must not
        block, raise, or communicate across ranks.
        """

    def remove(self, transfers: list[PoolTransfer]) -> None:
        """Best-effort drop of stored objects no lookup should restore again:
        a Mamba state the device cache replaced deeper along its path, which
        would otherwise hold store space until eviction reached it.

        LOOKUP-phase transfers naming the objects' pages. Called inside the
        rank-lockstep insert, so it must not block, raise, or communicate
        across ranks.
        """

    @abstractmethod
    def lookup(
        self, rid: str, transfers: list[PoolTransfer], resident_pages: int = 0
    ) -> list[int]:
        """Return every prefix length (in pages) that is fully restorable.

        A length is included only when *all* pools satisfy their hit policy at
        that exact boundary (contiguous prefix pools, plus each trailing-window
        pool's window ending there). Trailing-window state (SWA / compress
        state) only exists at offloaded node boundaries, so the set is sparse
        and generally non-contiguous -- returning just the local maximum would
        let the tree pick a length that is invalid on another rank.

        The first ``resident_pages`` pages are on the device already, so the
        contiguous prefix pools count them as present without a stored copy.

        Local to this rank; the tree intersects the sets across ranks.
        """

    def lookup_batch(
        self, requests: Sequence[tuple[str, list[PoolTransfer], int]]
    ) -> list[tuple[list[int], object]]:
        """``lookup`` for several ``(rid, transfers, resident_pages)`` at once,
        as ``(restorable lengths, detail)`` per request, where ``detail`` is
        what the store reports about a short lookup. A store that can should
        answer them with fewer round trips than one lookup each."""
        results = []
        for rid, transfers, resident_pages in requests:
            restorable = (
                self.lookup(rid, transfers, resident_pages=resident_pages)
                if resident_pages
                else self.lookup(rid, transfers)
            )
            results.append((restorable, getattr(self, "last_lookup_detail", None)))
        return results

    @abstractmethod
    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        """Queue a load into the given device indices.

        The transfer is executed by the next ``start_layer_wise_loading`` call,
        not here.
        """

    @abstractmethod
    def start_layer_wise_loading(self) -> int:
        """Start queued loads and return the layer-counter consumer index."""

    @abstractmethod
    def finish_layer_wise_loading(self, counter_index: int) -> bool:
        """Wait until the batch started as ``counter_index`` stops writing to
        device memory; True when every layer landed."""

    @abstractmethod
    def cancel_queued_load(self, rid: str) -> bool:
        """Cancel a load that has not started yet."""

    @abstractmethod
    def offload(
        self, transfers: list[PoolTransfer], session_id: Optional[str] = None
    ) -> bool:
        """Queue every transfer for atomic persistence.

        ``session_id`` names the session whose insert produced the transfers,
        when known; backends may attribute the stored objects to it.
        """

    @abstractmethod
    def num_completed_offloads(self) -> int:
        """Return the number of completed offloads waiting to be consumed."""

    @abstractmethod
    def pop_completed_offload(self) -> bool:
        """Consume the oldest completed offload and return its result."""

    @abstractmethod
    def reset(self) -> None:
        """Quiesce all transfers and reset backend state before returning."""

    @abstractmethod
    def close(self) -> None:
        """Quiesce all transfers and release backend resources."""


class ExternalCacheHitMarker(NamedTuple):
    """What ``match`` found in the external store, consumed by ``load_back``.

    ``tail_hashes`` are the per-page storage hashes of the restorable tail,
    starting at ``device_hit_len``. ``resident_len`` ends the device-resident
    Full KV the load reuses instead of loading (a node end within the hit), or
    is 0; ``resident_key`` is the radix key up to it.
    """

    tail_hashes: list[str]
    device_hit_len: int
    resident_len: int = 0
    resident_key: Optional[RadixKey] = None


class _LookupRecord(NamedTuple):
    """A request's last store lookup.

    ``shape`` is (device hit, tail pages), which fixes the tail hashes for a
    request's tokens; ``round`` is the scheduling round that looked it up.
    """

    shape: tuple[int, int]
    hit: Optional[tuple[int, int, int]]
    marker: Optional[ExternalCacheHitMarker]
    round: int


class _TailLookup(NamedTuple):
    """One tail ready for the store: what ``_lookup_tails`` asks and reduces."""

    key: RadixKey
    device_hit_len: int
    tail_hashes: list[str]
    transfers: list[PoolTransfer]
    resident_ends: list[int]


def _slot_of(indices: torch.Tensor) -> int:
    return int(indices.reshape(-1)[0].item())


class _PendingOffload(NamedTuple):
    # None for a written-back state, whose node may be gone by its ack.
    lock_node_id: Optional[NodeId]
    lock_params: Optional[DecLockRefParams]
    publish_node_ids: list[NodeId]
    # A Mamba state added to a node already stored or pending: the tree tracks
    # no external state for it, so its ack only releases the lock.
    state_only: bool = False
    # The device slot a written-back state was copied into, freed at the ack.
    staging: Optional[torch.Tensor] = None


class UnifiedCacheLinkerWrapper:
    """Drives an external KV store on behalf of one :class:`UnifiedRadixCache`."""

    def __init__(
        self,
        cache: UnifiedRadixCache,
        cache_linker: UnifiedCacheLinker,
    ):
        unsupported = set(cache.tree_components) - _EXTERNAL_LINKER_SUPPORTED_COMPONENTS
        if unsupported:
            names = ", ".join(
                component.name for component in sorted(unsupported, key=int)
            )
            raise ValueError(
                "External cache linker supports only Full, SWA and Mamba tree "
                f"components; unsupported: {names}"
            )

        self.cache = cache
        self.cache_linker = cache_linker
        swa = cache.components.get(ComponentType.SWA)
        # A store without SWA pages tombstones restored tails instead.
        self._skip_swa = swa is not None and PoolName.SWA not in cache_linker.pools
        self._components = tuple(
            component
            for component in cache._components_tuple
            if not (self._skip_swa and component is swa)
        )
        # rid -> what match found, consumed by the next init_load_back.
        self.hit_markers: dict[str, ExternalCacheHitMarker] = {}
        # rid -> its last lookup. The scheduler matches every waiting request
        # on each pass to order the queue; such a probe reuses the hit while
        # the device hit and the tail are unchanged. Admission reuses only a
        # lookup of the same scheduling round, as fresh as one made then. A
        # lookup is a collective, so the reuse must not depend on time: every
        # rank has to make the same call.
        self._probe_hits: dict[str, _LookupRecord] = {}
        # Scheduling rounds, counted alike on every rank (0 until the scheduler
        # starts one, which reuses nothing at admission), and the device hit
        # each request's probe found in the current one.
        self._lookup_round = 0
        self._round_device_hits: dict[str, tuple[int, NodeId]] = {}
        # Requests admitted in a round are looked up together, at most this
        # many; 0 looks each one up on its own.
        self.lookup_batch_size = int(os.environ.get("SGLANG_LINKER_LOOKUP_BATCH", "8"))
        # Requests loading in the batch being built, then its counter index once
        # the batch starts; finish_loads consumes both after its forward.
        self.inflight_load_rids: list[str] = []
        # Loaded tails stay outside the tree until their forward completes.
        self.private_load_indices: dict[str, torch.Tensor] = {}
        self.inflight_load_index = -1
        # Offloads in flight, each holding a lock on its node until it lands.
        self.pending_offloads: list[_PendingOffload] = []
        # Results taken from the backend that the ranks have not agreed on yet.
        self._completed_offloads: deque[bool] = deque()
        # Page-aligned radix-key prefix excluded from offload during an insert.
        self.offload_skip_prefix_len = 0
        self._touch_interval_s = cache_linker.touch_interval_s
        if self._touch_interval_s is not None and not self._touch_interval_s > 0:
            raise ValueError(
                f"touch_interval_s must be None or > 0, got {self._touch_interval_s}."
            )
        # Device nodes touched in the current interval bucket; NodeIds are
        # never reused.
        self._touch_bucket = -1
        self._touched: set[NodeId] = set()
        # A Mamba hybrid drops a node's device state but keeps its Full KV, so
        # the device match stops short of KV that is still resident. Reuse it
        # and load only the state and the pages past it; the store need not
        # keep the resident pages, which it may have evicted unread. Full must
        # be the only paged component, and resident lengths must be in
        # radix-key units.
        self._reuse_resident_kv = (
            ComponentType.MAMBA in cache.components
            and ComponentType.SWA not in cache.components
            and not cache.tree_core.is_eagle
        )
        self.resident_reuse_stats = {
            "loads": 0,
            "state_only": 0,
            "partial": 0,
            "stale": 0,
            "reused_tokens": 0,
            "loaded_tokens": 0,
        }
        # Node -> the device Mamba slot whose state went to the store. The
        # write-through chain offloads new leaves only, so a state attached to
        # a node already stored (a checkpoint inside a resident path, a branch
        # point, a recompute of an evicted state) would never get there.
        self._offload_states = ComponentType.MAMBA in cache.components
        self._state_offloaded: dict[NodeId, int] = {}
        # Write a Mamba state back when the device evicts it instead of through
        # at insert: --mamba-max-states-per-path replaces most states with a
        # deeper one first, and a replaced state never reaches the store.
        # Device slot -> (node, page hash) of each tree state not yet written.
        self._state_write_back = (
            self._offload_states
            and os.environ.get("SGLANG_LINKER_STATE_WRITE_BACK") == "1"
        )
        self._unwritten_states: dict[int, tuple[NodeId, str]] = {}
        # Slots freed while set were replaced along their path, not evicted.
        self._states_superseded = False
        # Written-back states copied out and not acked yet, at most this many.
        self.state_write_back_slots = int(
            os.environ.get("SGLANG_LINKER_STATE_WRITE_BACK_SLOTS", "8")
        )
        self._staged_states = 0
        # Radix-key end past which an insert offloads nothing; 0 is no limit.
        # Set for finished requests when their output is never reused.
        self.offload_end_limit = 0
        self.skip_output_offload = (
            os.environ.get("SGLANG_LINKER_SKIP_OUTPUT_OFFLOAD") == "1"
        )
        self.offload_stats = {
            "state_only": 0,
            "state_only_failed": 0,
            "output_trimmed": 0,
            "output_skipped": 0,
            "states_removed": 0,
            "states_written_back": 0,
            "states_dropped": 0,
        }
        self._remove_states = cache_linker.supports_remove
        # Lookups restoring under half of a long tail are logged once per
        # request and shape, with the store's per-pool verdict.
        self.lookup_stats = {
            "lookups": 0,
            "tail_tokens": 0,
            "hit_tokens": 0,
            "short": 0,
            "probe_reuses": 0,
            "round_reuses": 0,
            "batched": 0,
            "batches": 0,
        }
        self._logged_short: dict[str, tuple[int, int]] = {}

        cache.tree_core.enable_external_cache_linker = True
        cache.write_through_threshold = 1

    def _lookup_transfers(self, hashes: list[str]) -> list[PoolTransfer]:
        transfers = []
        for component in self._components:
            transfer = component.build_external_linker_transfer(
                LinkerTransferPhase.LOOKUP, None, hashes
            )
            if transfer is None:
                return []
            transfers.append(transfer)
        return transfers

    @property
    def layer_done_counter(self) -> object:
        return self.cache_linker.layer_done_counter

    def has_hit(self, rid: str) -> bool:
        return rid in self.hit_markers

    # ---- match: probe the remote store and report host_hit_length ----

    def match(
        self, key: RadixKey, req: Req, result: MatchResult, *, probe: bool = False
    ) -> MatchResult:
        cache = self.cache
        key, _ = key.maybe_to_bigram_view(cache.tree_core.is_eagle)
        page = cache.page_size
        device_hit_len = result.device_prefix_len
        if device_hit_len >= page and self._touch_interval_s is not None:
            self._touch(result.last_device_node, device_hit_len // page)
        if device_hit_len >= len(key):
            return result

        shape = (device_hit_len, (len(key) - device_hit_len) // page)
        known = self._probe_hits.get(req.rid)
        if probe:
            self._round_device_hits[req.rid] = (
                device_hit_len,
                result.last_device_node,
            )
            if known is not None and known.shape == shape:
                self.lookup_stats["probe_reuses"] += 1
                return self._with_host_hit(result, known.hit)
        elif (
            known is not None
            and known.shape == shape
            and known.round == self._lookup_round > 0
        ):
            # This round looked the tail up already (its probe, or the batch
            # ahead of admission), so the hit is as fresh as a new lookup.
            self.lookup_stats["round_reuses"] += 1
            self._set_hit_marker(req.rid, known.marker)
            return self._with_host_hit(result, known.hit)
        else:
            # init_load_back follows admission, so it must load what this finds.
            self.hit_markers.pop(req.rid, None)
        lookup = self._prepare_tail(key, result.last_device_node, device_hit_len)
        if lookup is None:
            hit, marker = None, None
        else:
            hit, marker = self._finish_tail(
                req, lookup, *self._lookup_tails([req.rid], [lookup])[0]
            )
        if marker is not None:
            self.hit_markers[req.rid] = marker
        self._probe_hits[req.rid] = _LookupRecord(
            shape, hit, marker, self._lookup_round
        )
        return self._with_host_hit(result, hit)

    def _set_hit_marker(
        self, rid: str, marker: Optional[ExternalCacheHitMarker]
    ) -> None:
        if marker is None:
            self.hit_markers.pop(rid, None)
        else:
            self.hit_markers[rid] = marker

    def begin_lookup_round(self) -> None:
        """Start a scheduling round; admission reuses only this round's lookups."""
        self._lookup_round += 1
        self._round_device_hits.clear()

    def prefetch_lookups(self, entries: Sequence[tuple[Req, RadixKey]]) -> None:
        """Look up, together, the admission keys of requests this round will
        likely admit: one hash pass, one store round trip per pool and one
        cross-rank reduction for all of them, which their admission matches
        then reuse.

        ``entries`` must be the same on every rank. A request whose probe did
        not run this round, whose admission hit cannot be told from its probe,
        or whose tail this round already looked up, is left to its admission.
        """
        if self._lookup_round == 0:
            return
        page = self.cache.page_size
        reqs, lookups, shapes = [], [], []
        for req, key in entries:
            key, _ = key.maybe_to_bigram_view(self.cache.tree_core.is_eagle)
            probed = self._round_device_hits.get(req.rid)
            if probed is None:
                continue
            device_hit_len, last_device_node = probed
            # A shorter key keeps the probe's device hit while that ends
            # within it; past that, admission matches a shallower state.
            tail_pages = (len(key) - device_hit_len) // page
            if tail_pages <= 0:
                continue
            shape = (device_hit_len, tail_pages)
            known = self._probe_hits.get(req.rid)
            if (
                known is not None
                and known.shape == shape
                and known.round == self._lookup_round
            ):
                continue
            lookup = self._prepare_tail(key, last_device_node, device_hit_len)
            if lookup is None:
                continue
            reqs.append(req)
            lookups.append(lookup)
            shapes.append(shape)
        if not lookups:
            return
        self.lookup_stats["batches"] += 1
        self.lookup_stats["batched"] += len(lookups)
        found = self._lookup_tails([req.rid for req in reqs], lookups)
        for req, lookup, shape, (hit_pages, detail) in zip(
            reqs, lookups, shapes, found
        ):
            hit, marker = self._finish_tail(req, lookup, hit_pages, detail)
            self._probe_hits[req.rid] = _LookupRecord(
                shape, hit, marker, self._lookup_round
            )

    @staticmethod
    def _with_host_hit(
        result: MatchResult, hit: Optional[tuple[int, int, int]]
    ) -> MatchResult:
        if hit is None:
            return result
        hit_tokens, swa_host_hit_length, mamba_host_hit_length = hit
        return result._replace(
            last_host_node=result.best_match_node,
            host_hit_length=hit_tokens,
            swa_host_hit_length=max(result.swa_host_hit_length, swa_host_hit_length),
            mamba_host_hit_length=max(
                result.mamba_host_hit_length, mamba_host_hit_length
            ),
        )

    def _prepare_tail(
        self, key: RadixKey, last_device_node: NodeId, device_hit_len: int
    ) -> Optional[_TailLookup]:
        """The store lookup for the tail past the device prefix, or None when
        there is nothing to ask."""
        tail_hashes = self._tail_hashes(key, last_device_node, device_hit_len)
        if not tail_hashes:
            return None
        transfers = self._lookup_transfers(tail_hashes)
        if not transfers:
            return None
        resident_ends = self._resident_ends(
            key,
            device_hit_len,
            device_hit_len + len(tail_hashes) * self.cache.page_size,
        )
        return _TailLookup(key, device_hit_len, tail_hashes, transfers, resident_ends)

    def _lookup_tails(
        self, rids: list[str], lookups: list[_TailLookup]
    ) -> list[tuple[int, object]]:
        """Ask the store for every tail at once and intersect the results
        across ranks in one reduction: per tail, the restorable pages all
        ranks share and the store's detail."""
        page = self.cache.page_size
        requests = []
        for rid, lookup in zip(rids, lookups):
            resident_pages = (
                (lookup.resident_ends[-1] - lookup.device_hit_len) // page
                if lookup.resident_ends
                else 0
            )
            requests.append((rid, lookup.transfers, resident_pages))
        answers = self.cache_linker.lookup_batch(requests)
        restorables = []
        for lookup, (_, _, resident_pages), (restorable, _) in zip(
            lookups, requests, answers
        ):
            if lookup.resident_ends:
                # Resident KV needs no stored copy, only a state where the hit
                # ends, and a hit ends inside it only at a node end.
                node_ends = {
                    (end - lookup.device_hit_len) // page
                    for end in lookup.resident_ends
                }
                restorable = [
                    pages
                    for pages in restorable
                    if pages > resident_pages or pages in node_ends
                ]
            restorables.append(restorable)
        hit_pages = self._sync_restorable_prefixes(
            restorables, [len(lookup.tail_hashes) for lookup in lookups]
        )
        return [(pages, detail) for pages, (_, detail) in zip(hit_pages, answers)]

    def _finish_tail(
        self, req: Req, lookup: _TailLookup, hit_pages: int, detail: object
    ) -> tuple[Optional[tuple[int, int, int]], Optional[ExternalCacheHitMarker]]:
        """The store's hit past the device prefix as (tokens, SWA tokens, Mamba
        slots) and the marker init_load_back consumes, or (None, None)."""
        page = self.cache.page_size
        device_hit_len = lookup.device_hit_len
        resident_ends = lookup.resident_ends
        self._record_lookup(
            req,
            len(lookup.key),
            device_hit_len,
            len(lookup.tail_hashes),
            resident_ends,
            hit_pages,
            detail,
        )
        if hit_pages == 0:
            return None, None
        by_pool = {transfer.name: transfer for transfer in lookup.transfers}
        hit_tokens = hit_pages * page

        swa_transfer = by_pool.get(PoolName.SWA)
        swa_host_hit_length = (
            min(len(swa_transfer.keys), hit_pages) * page
            if swa_transfer is not None
            else 0
        )
        # Mamba keeps a single state slot per node, so a hit is worth one slot.
        mamba_host_hit_length = 1 if PoolName.MAMBA in by_pool else 0

        resident_len = (
            min(device_hit_len + hit_tokens, resident_ends[-1]) if resident_ends else 0
        )
        marker = ExternalCacheHitMarker(
            tail_hashes=list(lookup.tail_hashes[:hit_pages]),
            device_hit_len=device_hit_len,
            resident_len=resident_len,
            resident_key=lookup.key[:resident_len] if resident_len else None,
        )
        return (hit_tokens, swa_host_hit_length, mamba_host_hit_length), marker

    def _touch(self, node_id: NodeId, pages: Optional[int]) -> None:
        """Touch the pages up to ``node_id``, once per node and interval;
        ``pages``, when given, is the path length the node must have.

        Decided from the node and the wall-clock interval, so the ranks of a
        group touch the same pages; a match that straddles an interval edge
        can move one rank's touch to its next match.
        """
        try:
            bucket = int(time.time() // self._touch_interval_s)
            if bucket != self._touch_bucket:
                self._touch_bucket = bucket
                self._touched.clear()
            elif node_id in self._touched:
                return
            self._touched.add(node_id)
            tree_core = self.cache.tree_core
            hashes = tree_core.get_prefix_hash_values(
                node_id
            ) + tree_core.get_hash_values(node_id)
            if not hashes or (pages is not None and len(hashes) != pages):
                return
            transfers = []
            for component in self._components:
                transfer = component.build_external_linker_transfer(
                    LinkerTransferPhase.LOOKUP, None, hashes
                )
                if transfer is None:
                    return
                transfers.append(transfer)
            self.cache_linker.touch(transfers)
        except Exception:
            # A hint must never break the rank-lockstep match it runs in.
            logger.exception("External cache touch failed; touches stop on this rank.")
            self._touch_interval_s = None

    def touch_running(self, reqs: Sequence[Req]) -> None:
        """Touch the cached prefix of each request in a batch. A decoding
        request's prompt-end state may leave the device long before its next
        turn, and the store's copy ages from when it was written."""
        if self._touch_interval_s is None:
            return
        cache = self.cache
        for req in reqs:
            node = req.last_node
            if not isinstance(node, int) or node == cache.root_node_handle(
                req.extra_key
            ):
                continue
            self._touch(node, None)
            if self._touch_interval_s is None:
                return

    def _record_lookup(
        self,
        req: Req,
        key_len: int,
        device_hit_len: int,
        tail_pages: int,
        resident_ends: list[int],
        hit_pages: int,
        detail: object = None,
    ) -> None:
        page = self.cache.page_size
        if detail is None:
            detail = getattr(self.cache_linker, "last_lookup_detail", None)
        stats = self.lookup_stats
        stats["lookups"] += 1
        stats["tail_tokens"] += tail_pages * page
        stats["hit_tokens"] += hit_pages * page
        if tail_pages * page >= 32768 and hit_pages * 2 < tail_pages:
            stats["short"] += 1
            shape = (device_hit_len, hit_pages)
            if self._logged_short.get(req.rid) != shape:
                self._logged_short[req.rid] = shape
                logger.info(
                    "External linker short lookup: rid=%s key=%d device_hit=%d "
                    "resident_end=%d tail_pages=%d hit_pages=%d pools=%s",
                    req.rid,
                    key_len,
                    device_hit_len,
                    resident_ends[-1] if resident_ends else 0,
                    tail_pages,
                    hit_pages,
                    detail,
                )
        if stats["lookups"] % 1024 == 0:
            logger.info(
                "External linker lookup stats: %s offload: %s",
                stats,
                self.offload_stats,
            )

    def _sync_restorable_prefix(
        self, restorable: list[int], *, num_pages: int, device_hit_pages: int
    ) -> int:
        """Intersect the per-rank sets of restorable prefix lengths and return the
        longest one, or 0 when the ranks share none beyond the device prefix.

        A rank's set is sparse, so reducing per-rank maxima could land on a
        length that only some ranks can restore. On a 0/1 mask MIN is AND, which
        makes the reduction an intersection.
        """
        restorable = [pages for pages in restorable if pages > device_hit_pages]
        return self._sync_restorable_prefixes([restorable], [num_pages])[0]

    def _sync_restorable_prefixes(
        self, restorables: list[list[int]], num_pages: list[int]
    ) -> list[int]:
        """``_sync_restorable_prefix`` for several tails in one reduction: their
        masks laid end to end."""
        offsets = [0]
        for pages in num_pages:
            offsets.append(offsets[-1] + pages + 1)
        values = [0] * offsets[-1]
        for restorable, start, limit in zip(restorables, offsets, num_pages):
            for pages in restorable:
                if 0 < pages <= limit:
                    values[start + pages] = 1
        mask = torch.tensor(values, dtype=torch.int)
        self.cache._all_reduce_attn_groups(mask, torch.distributed.ReduceOp.MIN)
        values = mask.tolist()
        common = []
        for start, end in zip(offsets, offsets[1:]):
            segment = values[start:end]
            common.append(
                next(
                    (pages for pages in reversed(range(end - start)) if segment[pages]),
                    0,
                )
            )
        return common

    def _resident_ends(self, key: RadixKey, start: int, end: int) -> list[int]:
        """Node ends past ``start`` on the device-resident Full path of
        ``key[:end]``, ascending. Reused KV must stop at a node end, so the
        device path to that node is exactly the reused prefix."""
        if not self._reuse_resident_kv:
            return []
        tree_core = self.cache.tree_core
        matched, node, _ = tree_core.match_full_device_prefix(key[:end])
        if matched <= start:
            return []
        (path,) = tree_core.collect_path_info([node])
        return [info.path_end for info in path if start < info.path_end <= matched]

    def _tail_hashes(
        self, key: RadixKey, last_device_node: NodeId, device_hit_len: int
    ) -> list[str]:
        """Per-page storage hashes for the device-uncached tail of the prefix."""
        last_hash = None
        if device_hit_len > 0:
            last_hash = self.cache.get_last_hash_value(last_device_node)
            if last_hash is None:
                # Without the anchor the tail would hash as if it started at the
                # sequence head, yielding keys that can never match.
                return []
        page = self.cache.page_size
        tail_len = (len(key) - device_hit_len) // page * page
        if tail_len == 0:
            return []
        return get_storage_hash_str(
            key[device_hit_len : device_hit_len + tail_len],
            last_hash,
            page_size=page,
        )

    # ---- init_load_back: remote -> request-private device slots ----

    def load_back(self, req: Req) -> tuple[int, NodeId]:
        """Queue the external hit into device slots owned by ``req``.

        The loaded tail extends the request's prefix but stays out of the
        tree: the request inserts it like computed KV after its forward, once
        :meth:`finish_loads` has confirmed the load. Device-resident Full KV in
        the hit is reused rather than reloaded; ``req.last_node`` then moves to
        the node that ends it, and the tree keeps owning that prefix.
        """
        cache = self.cache
        hit = self.hit_markers.pop(req.rid, None)
        if hit is None:
            return 0, req.last_node

        device_hit_len = hit.device_hit_len
        tail_hashes = hit.tail_hashes
        prefix_len = device_hit_len + len(tail_hashes) * cache.page_size

        resident = self._lock_resident_kv(hit)
        if resident is None:
            return 0, req.last_node
        resident_len, resident_node, resident_lock = resident
        resident_pages = (resident_len - device_hit_len) // cache.page_size
        try:
            # Build per-component linker transfers; Full skips the resident pages.
            component_transfers: list[tuple[TreeComponent, PoolTransfer]] = []
            for component in self._components:
                keys = tail_hashes
                if component.component_type == ComponentType.FULL:
                    keys = tail_hashes[resident_pages:]
                    if not keys:
                        # Every page is resident: the load moves the state alone.
                        continue
                transfer = component.build_external_linker_transfer(
                    LinkerTransferPhase.LOAD, None, keys
                )
                if transfer is None:
                    self._update_load(
                        ExternalLinkerLoadPhase.ABORT,
                        req,
                        component_transfers,
                        prefix_len,
                    )
                    return 0, req.last_node
                component_transfers.append((component, transfer))

            full_transfer = next(
                (t for _, t in component_transfers if t.name == PoolName.KV), None
            )
            # Queue before PREPARE touches the request, so a refused load only has
            # the allocations to undo before falling back to recompute. The result
            # depends on the transfers alone, so every rank takes the same branch.
            try:
                queued = self.cache_linker.load(
                    req.rid, [transfer for _, transfer in component_transfers]
                )
            except BaseException:
                self._update_load(
                    ExternalLinkerLoadPhase.ABORT, req, component_transfers, prefix_len
                )
                raise
            if not queued:
                self._update_load(
                    ExternalLinkerLoadPhase.ABORT, req, component_transfers, prefix_len
                )
                return 0, req.last_node
            self._update_load(
                ExternalLinkerLoadPhase.PREPARE,
                req,
                component_transfers,
                prefix_len,
            )
        finally:
            if resident_lock is not None:
                # The request locks req.last_node before anything else allocates.
                cache.dec_lock_ref(resident_node, resident_lock)

        # Components omitted from the linker do not run their PREPARE hook.
        # Keep a non-restorable SWA range as tombstones instead of rebuilding
        # it from an uninitialized FULL-to-SWA mapping when the tail is inserted.
        if self._skip_swa:
            req.kv.set_evicted_seqlen(
                ComponentType.SWA,
                max(req.kv.get_evicted_seqlen(ComponentType.SWA), prefix_len),
            )

        # match placed the SWA branch point before this tail was known; a point
        # the tail covers is gone, since the request inserts the whole tail.
        if (
            req.swa_branching_seqlen is not None
            and req.swa_branching_seqlen <= prefix_len
        ):
            req.swa_branching_seqlen = None
        loaded_tokens = 0
        if full_transfer is not None:
            self.private_load_indices[req.rid] = full_transfer.device_indices
            loaded_tokens = len(full_transfer.device_indices)
        self.inflight_load_rids.append(req.rid)
        reused_tokens = resident_len - device_hit_len
        self._record_load(reused_tokens, loaded_tokens)
        if resident_node is None:
            return loaded_tokens, req.last_node
        # The tree owns the reused KV, so a failed load frees only the tail.
        req.kv.protect_prefix(resident_len)
        if self._touch_interval_s is not None:
            # The store never reads the reused KV, so its copy would age out.
            self._touch(resident_node, resident_len // cache.page_size)
        # The scheduler ranked and sized this request by the full hit; from here
        # on the hit only feeds the tier split, which counts reused KV as device.
        req.host_hit_length -= reused_tokens
        return reused_tokens + loaded_tokens, resident_node

    def _lock_resident_kv(
        self, hit: ExternalCacheHitMarker
    ) -> Optional[tuple[int, Optional[NodeId], Optional[DecLockRefParams]]]:
        """Re-check and pin the resident KV ``match`` found. Allocations since
        then may have evicted it, and the store was never asked for a copy, so
        that request recomputes instead: None."""
        if not hit.resident_len:
            return hit.device_hit_len, None, None
        matched, node, pinned = self.cache.tree_core.match_full_device_prefix(
            hit.resident_key
        )
        if matched != hit.resident_len or pinned != matched:
            self.resident_reuse_stats["stale"] += 1
            return None
        return matched, node, self.cache.inc_lock_ref(node).to_dec_params()

    def _record_load(self, reused_tokens: int, loaded_tokens: int) -> None:
        stats = self.resident_reuse_stats
        stats["loads"] += 1
        if reused_tokens:
            stats["state_only" if loaded_tokens == 0 else "partial"] += 1
        stats["reused_tokens"] += reused_tokens
        stats["loaded_tokens"] += loaded_tokens
        if stats["loads"] % 128 == 0:
            logger.info("External linker resident KV reuse: %s", stats)

    def _update_load(
        self,
        phase: ExternalLinkerLoadPhase,
        req: Req,
        component_transfers: list[tuple[TreeComponent, PoolTransfer]],
        prefix_len: int,
    ) -> list[PoolTransfer]:
        if not component_transfers:
            return []
        full = component_transfers[0][1]
        result = []
        transfers = (
            reversed(component_transfers)
            if phase == ExternalLinkerLoadPhase.ABORT
            else component_transfers
        )
        for component, transfer in transfers:
            transfer = component.update_external_linker_load(
                phase, req, full, transfer, prefix_len
            )
            if transfer is not None:
                result.append(transfer)
        return result

    # ---- offload: device -> remote, driven by the write-through chain ----

    def offload_nodes(self, node_ids: Sequence[NodeId]) -> None:
        """Persist a write-through chain, skipping nodes already in the store."""
        for node_id in node_ids:
            transfers = self.cache.tree_core.build_external_linker_offload_transfers(
                node_id
            )
            if transfers is None:
                continue
            if self._skip_swa:
                transfers = [t for t in transfers if t.name != PoolName.SWA]
            if self.offload_skip_prefix_len:
                transfers = self._drop_offload_skip_prefix(node_id, transfers)
                if transfers is None:
                    continue
            if self.offload_end_limit:
                transfers = self._drop_offload_past_end(node_id, transfers)
                if transfers is None:
                    continue
            state = next((t for t in transfers if t.name == PoolName.MAMBA), None)
            if state is not None and self._state_write_back:
                self._remember_state(
                    node_id, _slot_of(state.device_indices), state.keys[-1]
                )
                transfers = [t for t in transfers if t.name != PoolName.MAMBA]
                state = None
            if self._offload_node(node_id, transfers) and state is not None:
                self._state_offloaded[node_id] = _slot_of(state.device_indices)

    def offload_new_state(self, node_id: NodeId) -> None:
        """Offload the Mamba state an insert left on ``node_id`` when the
        write-through chain did not: the node was already stored or pending,
        or its walk-time offload ran before the state was attached."""
        if not self._offload_states:
            return
        tree_core = self.cache.tree_core
        value = tree_core.get_component_device_value(node_id, ComponentType.MAMBA)
        if value is None:
            return
        slot = _slot_of(value)
        if self._state_offloaded.get(node_id) == slot:
            return
        if (
            self._state_write_back
            and self._unwritten_states.get(slot, (None,))[0] == node_id
        ):
            return
        if self.offload_end_limit:
            (path,) = tree_core.collect_path_info([node_id])
            if not path or path[-1].path_end > self.offload_end_limit:
                return
        hashes = tree_core.get_hash_values(node_id)
        if not hashes:
            return
        if self._state_write_back:
            self._remember_state(node_id, slot, hashes[-1])
            return
        # The component's own OFFLOAD transfer: one state, keyed by the node end.
        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            device_indices=value,
            keys=[hashes[-1]],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        cache = self.cache
        lock_params = cache.inc_lock_ref(node_id).to_dec_params()
        try:
            queued = self.cache_linker.offload(
                [transfer], session_id=getattr(cache, "insert_session_id", None)
            )
        except BaseException:
            cache.dec_lock_ref(node_id, lock_params)
            raise
        if not queued:
            cache.dec_lock_ref(node_id, lock_params)
            return
        self._state_offloaded[node_id] = slot
        self.offload_stats["state_only"] += 1
        self.pending_offloads.append(
            _PendingOffload(node_id, lock_params, [], state_only=True)
        )

    # ---- state write-back: a Mamba state reaches the store when evicted ----

    def _remember_state(self, node_id: NodeId, slot: int, page_hash: str) -> None:
        # A slot holds one tree state at a time: a reused slot names its new one.
        self._unwritten_states[slot] = (node_id, page_hash)

    @contextmanager
    def superseded_states(self):
        """Mark device frees as the path cap's: a deeper state on the same
        path replaced each one, so none is written back."""
        self._states_superseded = True
        try:
            yield
        finally:
            self._states_superseded = False

    def before_state_free(self, slots: torch.Tensor) -> None:
        """Write back the tree states in ``slots`` before the device frees them.

        Each state is copied into a free slot on the current stream, which every
        later forward waits on before it can write the freed slot, and the copy
        is offloaded from there; its slot returns at the offload's ack. Without
        a free slot, or past ``state_write_back_slots`` copies in flight, the
        state is dropped, as a store without it would have.

        The tree tracks no external state for a state, and every rank evicts
        the same states and holds the same free slots, so the ranks agree on
        every pending offload this queues.
        """
        if not self._unwritten_states:
            return
        for index, slot in enumerate(slots.reshape(-1).tolist()):
            entry = self._unwritten_states.pop(slot, None)
            if entry is None or self._states_superseded:
                continue
            node_id, page_hash = entry
            source = slots.reshape(-1)[index : index + 1]
            if not self._write_back_state(source, page_hash):
                self.offload_stats["states_dropped"] += 1

    def _write_back_state(self, source: torch.Tensor, page_hash: str) -> bool:
        if self._staged_states >= self.state_write_back_slots:
            return False
        storage, sub_pool = self.cache.components[ComponentType.MAMBA]._state()
        staging = storage.take_state_slots(sub_pool, 1)
        if staging is None:
            return False
        storage.copy_state(sub_pool, source, staging)
        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            device_indices=staging,
            keys=[page_hash],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        try:
            queued = self.cache_linker.offload([transfer])
        except BaseException:
            storage.free_state_slots(sub_pool, staging)
            raise
        if not queued:
            storage.free_state_slots(sub_pool, staging)
            return False
        self._staged_states += 1
        self.offload_stats["states_written_back"] += 1
        self.pending_offloads.append(
            _PendingOffload(None, None, [], state_only=True, staging=staging)
        )
        return True

    def _release_offload(self, pending: _PendingOffload) -> None:
        if pending.lock_node_id is not None:
            self.cache.dec_lock_ref(pending.lock_node_id, pending.lock_params)
        if pending.staging is not None:
            storage, sub_pool = self.cache.components[ComponentType.MAMBA]._state()
            storage.free_state_slots(sub_pool, pending.staging)
            self._staged_states -= 1

    @contextmanager
    def skip_offload_prefix(self, length: int):
        """Skip offloading the first ``length`` radix-key units during an insert."""
        self.offload_skip_prefix_len = length
        try:
            yield
        finally:
            self.offload_skip_prefix_len = 0

    @contextmanager
    def offload_window(self, skip_prefix_len: int, end_limit: int):
        """During an insert, skip offloading the first ``skip_prefix_len``
        radix-key units and everything past ``end_limit`` (0: no limit)."""
        self.offload_skip_prefix_len = skip_prefix_len
        self.offload_end_limit = end_limit
        try:
            yield
        finally:
            self.offload_skip_prefix_len = 0
            self.offload_end_limit = 0

    def _drop_offload_past_end(
        self, node_id: NodeId, transfers: list[PoolTransfer]
    ) -> Optional[list[PoolTransfer]]:
        info = self.cache.tree_core.collect_path_info([node_id])[0][-1]
        limit = self.offload_end_limit
        if info.path_end <= limit:
            return transfers
        if info.path_start >= limit:
            self.offload_stats["output_skipped"] += 1
            return None
        page = self.cache.page_size
        kept = []
        for transfer in transfers:
            # The state sits at the node end, past the limit.
            if transfer.name == PoolName.MAMBA:
                continue
            first = info.path_end - len(transfer.keys) * page
            keep = max(0, (limit - first) // page)
            if keep:
                kept.append(
                    replace(
                        transfer,
                        keys=transfer.keys[:keep],
                        device_indices=transfer.device_indices[: keep * page],
                    )
                )
        self.offload_stats["output_trimmed"] += 1
        return kept or None

    def _drop_offload_skip_prefix(
        self, node_id: NodeId, transfers: list[PoolTransfer]
    ) -> Optional[list[PoolTransfer]]:
        info = self.cache.tree_core.collect_path_info([node_id])[0][-1]
        page = self.cache.page_size
        kept = []
        for transfer in transfers:
            first = info.path_end - len(transfer.keys) * page
            drop = min(
                len(transfer.keys),
                max(0, (self.offload_skip_prefix_len - first) // page),
            )
            if drop == len(transfer.keys):
                if transfer.name == PoolName.KV:
                    # The caller delegates storage of this prefix to another writer.
                    # Mark it stored locally to suppress uploads, without a persistence ACK.
                    self.cache.tree_core.mark_external_cache_stored_path(
                        node_id, info.parent_id
                    )
                    return None
                continue
            if drop:
                transfer = replace(
                    transfer,
                    keys=transfer.keys[drop:],
                    device_indices=transfer.device_indices[drop * page :],
                )
            kept.append(transfer)
        return kept

    def _offload_node(self, node_id: NodeId, transfers: list[PoolTransfer]) -> bool:
        cache = self.cache
        lock_params = cache.inc_lock_ref(node_id).to_dec_params()
        try:
            queued = self.cache_linker.offload(
                transfers, session_id=getattr(cache, "insert_session_id", None)
            )
        except BaseException:
            cache.dec_lock_ref(node_id, lock_params)
            raise
        if not queued:
            cache.dec_lock_ref(node_id, lock_params)
            return False

        cache.tree_core.mark_external_linker_offload_pending(node_id)
        self.pending_offloads.append(_PendingOffload(node_id, lock_params, [node_id]))
        return True

    def replace_pending_offload_node(
        self, ack_id: NodeId, old_node_id: NodeId, new_node_ids: list[NodeId]
    ) -> None:
        for index, pending in enumerate(self.pending_offloads):
            if pending.lock_node_id != ack_id or pending.state_only:
                continue
            publish_node_ids = []
            for node_id in pending.publish_node_ids:
                if node_id == old_node_id:
                    publish_node_ids.extend(new_node_ids)
                else:
                    publish_node_ids.append(node_id)
            self.pending_offloads[index] = pending._replace(
                publish_node_ids=publish_node_ids
            )
            return

    def num_completed_offloads(self) -> int:
        return min(
            len(self._completed_offloads) + self.cache_linker.num_completed_offloads(),
            len(self.pending_offloads),
        )

    def take_completed_offloads(self, finish_count: int) -> list[bool]:
        assert finish_count <= len(self.pending_offloads)
        taken = []
        for _ in range(finish_count):
            if self._completed_offloads:
                taken.append(self._completed_offloads.popleft())
            else:
                taken.append(self.cache_linker.pop_completed_offload())
        return taken

    def offload_completion_report(self) -> Optional[torch.Tensor]:
        """This rank's completed offloads for the per-step MIN reduction, as
        ``[count, first failure]``, the failure index being the count when
        none failed. Two elements, because a gloo reduction of one or two
        costs half of one of four or more.

        None when no offload is pending: inserts queue offloads in rank
        lockstep, so every rank skips the reduction together.
        """
        if not self.pending_offloads:
            return None
        window = min(len(self.pending_offloads), OFFLOAD_REPORT_WINDOW)
        while (
            len(self._completed_offloads) < window
            and self.cache_linker.num_completed_offloads()
        ):
            self._completed_offloads.append(self.cache_linker.pop_completed_offload())
        count = len(self._completed_offloads)
        first_failure = next(
            (index for index, ok in enumerate(self._completed_offloads) if not ok),
            count,
        )
        return torch.tensor([count, first_failure], dtype=torch.int)

    def commit_offload_completion_report(self, report: torch.Tensor) -> None:
        """Commit what a reduced ``offload_completion_report`` agreed on: every
        rank's offloads below the earliest failure landed, and that one failed
        on some rank. Later results wait for the next step."""
        count, first_failure = report.tolist()
        committed = min(count, first_failure + 1)
        assert committed <= len(self._completed_offloads)
        for _ in range(committed):
            self._completed_offloads.popleft()
        self.commit_completed_offloads(
            [index != first_failure for index in range(committed)]
        )

    def commit_completed_offloads(self, successes: Sequence[bool]) -> None:
        assert len(successes) <= len(self.pending_offloads)
        for success in successes:
            pending = self.pending_offloads.pop(0)
            if not pending.state_only:
                self.cache.tree_core.finish_external_linker_offload(
                    pending.publish_node_ids, pending.lock_node_id, success
                )
            if not success:
                # The state never landed; a later insert of the node retries it.
                self._state_offloaded.pop(pending.lock_node_id, None)
                if pending.state_only:
                    self.offload_stats["state_only_failed"] += 1
            self._release_offload(pending)

    def start_layer_wise_loading(self) -> int:
        self.inflight_load_index = self.cache_linker.start_layer_wise_loading()
        return self.inflight_load_index

    def finish_loads(self) -> list[str]:
        """Wait for the loads of the batch that just ran; return the rids whose
        KV did not land, identically on every rank.

        Runs after that batch's forward, so no slot it wrote to has been freed
        or inserted yet. Every rank holds the same ``inflight_load_rids``
        because ``match`` reduces the hit across ranks, so either all ranks or
        none enter the reduction. The verdict is reduced because an MLA dedup
        peer only receives the broadcast and never sees the source's error.
        """
        rids, self.inflight_load_rids = self.inflight_load_rids, []
        if not rids:
            return []
        index, self.inflight_load_index = self.inflight_load_index, -1
        landed = torch.tensor(
            [int(self.cache_linker.finish_layer_wise_loading(index))],
            dtype=torch.int,
        )
        self.cache._all_reduce_attn_groups(landed, torch.distributed.ReduceOp.MIN)
        for rid in rids:
            self.private_load_indices.pop(rid, None)
        if landed.item():
            logger.info(
                "External linker stage load completed: requests=%d rids=%s",
                len(rids),
                rids,
            )
            return []
        logger.error(
            "External linker load failed; aborting %d request(s): %s", len(rids), rids
        )
        return rids

    # ---- path cap: drop the store's copies of the states it replaced ----

    def path_state_holders(self, tail: NodeId) -> list[NodeId]:
        """Nodes above ``tail`` holding a device Mamba state, read before the
        path cap runs, when the store can drop what it evicts."""
        # Written back only when evicted, a replaced state is not in the store.
        if not self._remove_states or self._state_write_back:
            return []
        try:
            tree_core = self.cache.tree_core
            (path,) = tree_core.collect_path_info([tail])
            return [
                info.node_id
                for info in path[:-1]
                if tree_core.get_component_device_value(
                    info.node_id, ComponentType.MAMBA
                )
                is not None
            ]
        except Exception:
            self._stop_removing_states()
            return []

    def remove_replaced_states(self, holders: list[NodeId]) -> None:
        """Drop the stored copies of the states the path cap took off the
        device. A deeper state on the same path replaced each one, and the
        store would hold it until eviction reached it."""
        try:
            tree_core = self.cache.tree_core
            hashes = []
            for node_id in holders:
                if (
                    tree_core.get_component_device_value(node_id, ComponentType.MAMBA)
                    is not None
                ):
                    continue
                # A state attached to this node again must be offloaded again.
                self._state_offloaded.pop(node_id, None)
                node_hashes = tree_core.get_hash_values(node_id)
                if node_hashes:
                    # A node's state is stored under the hash of its last page.
                    hashes.append(node_hashes[-1])
            mamba = self.cache.components[ComponentType.MAMBA]
            for page_hash in hashes:
                # A Mamba LOOKUP transfer names only its trailing page.
                transfer = mamba.build_external_linker_transfer(
                    LinkerTransferPhase.LOOKUP, None, [page_hash]
                )
                if transfer is not None:
                    self.cache_linker.remove([transfer])
                    self.offload_stats["states_removed"] += 1
        except Exception:
            self._stop_removing_states()

    def _stop_removing_states(self) -> None:
        # A store cleanup must never break the insert it runs in.
        logger.exception(
            "Dropping replaced Mamba states from the store failed; they now stay "
            "until eviction on this rank."
        )
        self._remove_states = False

    # ---- lifecycle ----

    def reset(self) -> None:
        self.cache_linker.reset()
        self._completed_offloads.clear()
        self._touched.clear()
        self._state_offloaded.clear()
        self._unwritten_states.clear()
        self._logged_short.clear()
        self.hit_markers.clear()
        self._probe_hits.clear()
        self._round_device_hits.clear()
        self.inflight_load_rids.clear()
        self.private_load_indices.clear()
        self.inflight_load_index = -1
        self._release_pending_offloads()

    def _release_pending_offloads(self) -> None:
        for pending in self.pending_offloads:
            if not pending.state_only:
                self.cache.tree_core.finish_external_linker_offload(
                    pending.publish_node_ids, pending.lock_node_id, False
                )
            self._release_offload(pending)
        self.pending_offloads.clear()

    def release_request(self, rid: str) -> None:
        self.hit_markers.pop(rid, None)
        self._probe_hits.pop(rid, None)
        self._round_device_hits.pop(rid, None)
        self._logged_short.pop(rid, None)
        # A queued load targets slots the request owns and is about to free.
        # A started one is finished by finish_loads before anything is freed.
        if rid in self.inflight_load_rids and self.cache_linker.cancel_queued_load(rid):
            self.inflight_load_rids.remove(rid)
            self.private_load_indices.pop(rid, None)

    def close(self) -> None:
        self.cache_linker.close()
        self._completed_offloads.clear()
        self.inflight_load_rids.clear()
        self.private_load_indices.clear()
        self._release_pending_offloads()
