"""Timing probes for the EP x KV-transfer (MoRI IO) contention study. All no-ops unless
EPIO_PROBE_DIR is set. Output: JSON lines under $EPIO_PROBE_DIR, one file per process.

  step_<role>_tp<r>.jsonl  one line per forward: GPU time of the forward (HIP events on the
                           stream the forward is enqueued on), its wall-clock end (observed by a
                           poller thread, ~0.3 ms resolution), batch shape, in-flight transfers
  xfer_tp<r>.jsonl         prefill: one line per MoRI transfer chunk (enqueue / dequeue /
                           wait-event done / submitted / completed, bytes, segments)
  kvarr_tp<r>.jsonl        decode: one line per request whose KV arrived (attn TP rank 0)
  epprof_tp<r>_<n>.npz     on-demand snapshots of the MoRI EP kernel profiler (EPIO_EPPROF=1,
                           mori built with ENABLE_PROFILER): written when $EPIO_PROBE_DIR/epprof.trigger
                           holds a new sequence number
"""

import json
import os
import threading
import time
from collections import deque

_DIR = os.environ.get("EPIO_PROBE_DIR")
_RANKS = os.environ.get("EPIO_PROBE_RANKS", "all")


def enabled() -> bool:
    return bool(_DIR)


def tp_rank() -> int:
    """Global TP rank of this scheduler process (the Scheduler object has no tp_rank here)."""
    try:
        from sglang.srt.runtime_context import get_parallel

        return int(get_parallel().tp_rank)
    except Exception:  # noqa: BLE001
        return 0


def rank_enabled(rank: int) -> bool:
    return enabled() and (_RANKS == "all" or str(rank) in _RANKS.split(","))


class _Writer:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._f = open(path, "a", buffering=1 << 16)
        self._lock = threading.Lock()
        self._last = time.time()

    def write(self, rec):
        line = json.dumps(rec, separators=(",", ":"))
        with self._lock:
            self._f.write(line + "\n")
            now = time.time()
            if now - self._last > 1.0:
                self._f.flush()
                self._last = now

    def flush(self):
        with self._lock:
            self._f.flush()


_writers = {}
_wlock = threading.Lock()


def writer(name):
    with _wlock:
        w = _writers.get(name)
        if w is None:
            w = _Writer(os.path.join(_DIR, name))
            _writers[name] = w
        return w


# --------------------------------------------------------------------------- steps
class StepProbe:
    """Wraps model_worker.forward_batch_generation. Events are recorded on the current stream
    (the forward stream under the overlap scheduler) around the call; a poller thread turns
    the end event into a wall-clock time without blocking the scheduler."""

    def __init__(self, scheduler):
        import torch

        self.torch = torch
        self.sched = scheduler
        self.rank = tp_rank()
        role = getattr(getattr(scheduler, "disaggregation_mode", None), "value", "null")
        self.role = role
        self.w = writer(f"step_{role}_tp{self.rank}.jsonl")
        self.pending = deque()
        self.step = 0
        self.prev_end_wall = None
        self.lock = threading.Lock()
        t = threading.Thread(target=self._poll, daemon=True, name="epio-step-poll")
        t.start()

    def _meta(self, batch):
        fm = getattr(batch, "forward_mode", None)
        mode = getattr(fm, "name", str(fm)).lower() if fm is not None else "none"
        reqs = getattr(batch, "reqs", None) or []
        m = {"mode": mode, "bs": len(reqs)}
        ext = getattr(batch, "extend_num_tokens", None)
        if ext is not None:
            m["ext_tok"] = int(ext)
        try:
            if mode.startswith("extend") or mode.startswith("mixed"):
                m["prefix_tok"] = int(sum(len(getattr(r, "prefix_indices", ()) or ()) for r in reqs))
        except Exception:  # noqa: BLE001
            pass
        s = self.sched
        try:
            if self.role == "prefill":
                m["inflight"] = len(getattr(s, "disagg_prefill_inflight_queue", ()) or ())
            elif self.role == "decode":
                tq = getattr(s, "disagg_decode_transfer_queue", None)
                pq = getattr(s, "disagg_decode_prealloc_queue", None)
                m["xferq"] = len(getattr(tq, "queue", ()) or ())
                m["preallocq"] = len(getattr(pq, "queue", ()) or ())
                m["running"] = len(getattr(getattr(s, "running_batch", None), "reqs", ()) or ())
        except Exception:  # noqa: BLE001
            pass
        return m

    def wrap(self, fn):
        torch = self.torch

        def wrapped(batch, *a, **k):
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            t_launch = time.time()
            ev0.record()
            out = fn(batch, *a, **k)
            ev1.record()
            t_ret = time.time()
            rec = {"step": self.step, "t_launch": t_launch, "t_ret": t_ret}
            rec.update(self._meta(batch))
            self.step += 1
            with self.lock:
                self.pending.append((ev0, ev1, rec))
            return out

        return wrapped

    def _poll(self):
        while True:
            item = None
            with self.lock:
                if self.pending:
                    item = self.pending[0]
            if item is None:
                time.sleep(0.0005)
                continue
            ev0, ev1, rec = item
            try:
                done = ev1.query()
            except Exception:  # noqa: BLE001
                with self.lock:
                    self.pending.popleft()
                continue
            if not done:
                time.sleep(0.0003)
                continue
            t_obs = time.time()
            with self.lock:
                self.pending.popleft()
            try:
                gpu_ms = ev0.elapsed_time(ev1)
            except Exception:  # noqa: BLE001
                gpu_ms = -1.0
            rec["t_end"] = t_obs
            rec["gpu_ms"] = round(gpu_ms, 4)
            rec["t_start"] = t_obs - gpu_ms / 1e3
            self.w.write(rec)


def maybe_install_scheduler(scheduler):
    if not rank_enabled(tp_rank()):
        return
    try:
        mw = scheduler.model_worker
        probe = StepProbe(scheduler)
        mw.forward_batch_generation = probe.wrap(mw.forward_batch_generation)
        scheduler._epio_step_probe = probe
        if os.environ.get("EPIO_EPPROF") == "1":
            EpProfSnapshotter(scheduler)
    except Exception as e:  # noqa: BLE001
        print(f"[epio_probe] step probe not installed: {e!r}", flush=True)


# --------------------------------------------------------------------------- transfers
_tls = threading.local()


class CountingEngine:
    """Forwards to a MoRI IOEngine; counts bytes / segments of batch_write per thread."""

    def __init__(self, engine):
        object.__setattr__(self, "_e", engine)

    def __getattr__(self, name):
        return getattr(self._e, name)

    def batch_write(self, local_descs, local_offsets, remote_descs, remote_offsets, sizes, uids):
        try:
            nb = 0
            ns = 0
            for s in sizes:
                ns += len(s)
                nb += int(sum(s))
            _tls.bytes = getattr(_tls, "bytes", 0) + nb
            _tls.segs = getattr(_tls, "segs", 0) + ns
            _tls.calls = getattr(_tls, "calls", 0) + 1
        except Exception:  # noqa: BLE001
            pass
        return self._e.batch_write(local_descs, local_offsets, remote_descs, remote_offsets, sizes, uids)


def reset_counts():
    _tls.bytes = 0
    _tls.segs = 0
    _tls.calls = 0


def take_counts():
    return getattr(_tls, "bytes", 0), getattr(_tls, "segs", 0), getattr(_tls, "calls", 0)


def xfer_record(rank, rec):
    if rank_enabled(rank):
        writer(f"xfer_tp{rank}.jsonl").write(rec)


def kv_arrival(rank, rec):
    if rank_enabled(rank):
        writer(f"kvarr_tp{rank}.jsonl").write(rec)


# --------------------------------------------------------------------------- EP profiler
def _find_mori_ops():
    import gc

    import mori

    ops = []
    for o in gc.get_objects():
        try:
            if isinstance(o, mori.ops.EpDispatchCombineOp):
                ops.append(o)
        except Exception:  # noqa: BLE001
            continue
    return ops


class EpProfSnapshotter:
    """On-demand snapshots of the MoRI kernel-profiler ring buffers of this rank's EP ops.
    Trigger: write a new integer into $EPIO_PROBE_DIR/epprof.trigger. Each snapshot also
    records a host/GPU clock anchor taken by launching a 1-thread wall_clock64 kernel."""

    def __init__(self, scheduler):
        self.rank = tp_rank()
        self.trig = os.path.join(_DIR, "epprof.trigger")
        self.seen = None
        self.n = 0
        t = threading.Thread(target=self._loop, daemon=True, name="epio-epprof")
        t.start()

    def _anchor(self):
        try:
            from sglang.srt.epio_clock import gpu_wall_clock_anchor

            return gpu_wall_clock_anchor()
        except Exception as e:  # noqa: BLE001
            return {"error": repr(e)}

    def _loop(self):
        import numpy as np
        import torch

        dev = torch.cuda.current_device()
        while True:
            time.sleep(0.5)
            try:
                seq = open(self.trig).read().strip()
            except OSError:
                continue
            if seq == self.seen:
                continue
            if self.seen is None:
                self.seen = seq  # do not fire on a trigger left from before startup
                continue
            self.seen = seq
            try:
                torch.cuda.set_device(dev)
                a0 = self._anchor()
                ops = _find_mori_ops()
                out = {}
                for i, op in enumerate(ops):
                    buf = op.get_debug_time_buf()
                    s = torch.cuda.Stream()
                    with torch.cuda.stream(s):
                        h = buf.to("cpu", non_blocking=False)
                    raw = h.numpy().reshape(-1, 2)
                    out[f"op{i}"] = raw[raw[:, 0] != 0]
                a1 = self._anchor()
                np.savez(os.path.join(_DIR, f"epprof_tp{self.rank}_{seq}.npz"),
                         anchor=json.dumps({"before": a0, "after": a1, "t": time.time(), "seq": seq}), **out)
                self.n += 1
            except Exception as e:  # noqa: BLE001
                print(f"[epio_probe] epprof snapshot failed: {e!r}", flush=True)
