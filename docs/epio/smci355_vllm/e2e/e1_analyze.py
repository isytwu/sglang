#!/usr/bin/env python3
"""Summarise one e2e/e1.sh run.

  e1_analyze.py results/e2e/<RUN> [--no-traces]   ->  markdown on stdout, <RUN>/summary.json

Per window (events.tsv; IO windows from io_on to io_end, base windows from start + 20 s, the profiler captures and
the 30 s after each cut out):
  step   the gap between consecutive streamed chunks of a request (= one decode step at stream-interval 1): median,
         p90, p99, and the change against the mean of the base windows
  NIC    RDMA bytes of every node's 8 NICs (fab_<node>.tsv), tx / rx GB/s
  vLLM   the engines' logged generation throughput (sum over the DP ranks) and running requests
  EP     from each capture's torch-profiler traces: MoRI dispatch / combine kernel durations (InterNodeV1LL), per rank
         file, then averaged over the ranks of each node
"""
import argparse
import datetime as dt
import glob
import gzip
import json
import multiprocessing
import os
import re
import statistics

import numpy as np

KERNELS = {"dispatch": re.compile(r"EpDispatchInterNodeV1Kernel"), "combine": re.compile(r"EpCombineInterNodeV1Kernel")}


def load_events(path):
    ev = []
    for line in open(path):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2:
            ev.append((float(p[0]), p[1], p[2] if len(p) > 2 else ""))
    return ev


def windows(ev):
    wins, cuts, cur = [], [], {}
    seen = {}
    for t, name, arg in ev:
        if name.startswith("prof_"):
            if arg == "start":
                cur[name] = t
            elif name in cur:
                cuts.append((cur.pop(name), t + 30))
            continue
        base = re.match(r"^(base|SPIN|I[WR]\w+?)(_io_on|_io_end)?$", name)
        if not base:
            continue
        w, kind = base.group(1), base.group(2) or ""
        if kind == "" and arg == "start":
            k = seen.get(w, 0) + 1
            seen[w] = k
            wins.append({"name": f"{w}_{k}", "kind": w, "start": t, "end": None, "on": None, "off": None})
        elif kind == "" and arg == "end":
            for x in reversed(wins):
                if x["kind"] == w and x["end"] is None:
                    x["end"] = t
                    break
        elif kind == "_io_on":
            wins[-1]["on"] = t
        elif kind == "_io_end":
            wins[-1]["off"] = t
    for x in wins:
        x["t0"] = (x["on"] + 5) if x["on"] else x["start"] + 20
        x["t1"] = (x["off"] - 5) if x["off"] else (x["end"] or x["start"]) - 5
    return wins, cuts


def load_gaps(path):
    ts, gs = [], []
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        c = r.get("chunks") or []
        t0 = r["t0"]
        for i in range(1, len(c)):
            ts.append(t0 + c[i])
            gs.append(c[i] - c[i - 1])
    return np.array(ts), np.array(gs)


def load_live(path):
    """loadgen's 10 s records of the in-flight streams' chunk gaps: (t, gap median ms, gap p90 ms)"""
    out = []
    if not os.path.exists(path):
        return np.zeros((0, 3))
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("gap_med_ms") is not None:
            out.append((r["t"], r["gap_med_ms"], r.get("gap_p90_ms") or float("nan")))
    return np.array(out).reshape(-1, 3)


def load_fab(path):
    rows = []
    hdr = None
    for line in open(path):
        if line.startswith("#"):
            continue
        p = line.split("\t")
        if hdr is None:
            hdr = p
            continue
        try:
            rows.append([float(x) for x in p])
        except ValueError:
            pass
    a = np.array(rows)
    # the 8 ports' byte counters; mb/fabmon.sh appends cnp_rx / ecn_rx after them
    tx = a[:, 7:23:2].sum(1)
    rx = a[:, 8:23:2].sum(1)
    return a[:, 0], tx, rx


def fab_rate(fab, t0, t1):
    t, tx, rx = fab
    i, j = np.searchsorted(t, t0), np.searchsorted(t, t1) - 1
    if j <= i:
        return float("nan"), float("nan")
    dtm = t[j] - t[i]
    return (tx[j] - tx[i]) / dtm / 1e9, (rx[j] - rx[i]) / dtm / 1e9


LOGLINE = re.compile(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d).*?Engine (\d+): .*?Avg generation throughput: ([\d.]+) tokens/s, "
                     r"Running: (\d+) reqs, Waiting: (\d+) reqs")


def load_vllm_logs(paths, year):
    out = []
    for p in paths:
        for line in open(p, errors="replace"):
            m = LOGLINE.search(line)
            if m:
                t = dt.datetime.strptime(f"{year}-{m.group(1)}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp()
                out.append((t, int(m.group(2)), float(m.group(3)), int(m.group(4)), int(m.group(5))))
    return out


def trace_kernels(path):
    with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as f:
        j = json.load(f)
    res = {k: [] for k in KERNELS}
    for e in j.get("traceEvents", []):
        if e.get("ph") != "X" or e.get("cat") not in ("kernel", "gpu_op", "Kernel"):
            continue
        n = e.get("name", "")
        for k, rx in KERNELS.items():
            if rx.search(n):
                res[k].append(float(e.get("dur", 0)))
    return res


def _trace_kernels_safe(path):
    try:
        return trace_kernels(path)
    except Exception as e:  # noqa: BLE001
        return str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--no-traces", action="store_true")
    args = ap.parse_args()
    R = args.run.rstrip("/")
    ev = load_events(os.path.join(R, "events.tsv"))
    wins, cuts = windows(ev)
    ts, gs = load_gaps(os.path.join(R, "load", "requests.jsonl"))
    live = load_live(os.path.join(R, "load", "live.jsonl"))
    fabs = {os.path.basename(p)[4:-4]: load_fab(p) for p in sorted(glob.glob(os.path.join(R, "fab_*.tsv")))}
    year = dt.datetime.fromtimestamp(ev[0][0], dt.timezone.utc).year if ev else 2026
    # E1: serve_<node>.log; E2 (srtctl): logs/<host>_decode_*.out
    logs = load_vllm_logs(sorted(glob.glob(os.path.join(R, "serve_*.log")) + glob.glob(os.path.join(R, "logs", "*_decode_*.out"))), year)
    cut_mask = np.zeros(len(ts), bool)
    for a, b in cuts:
        cut_mask |= (ts >= a) & (ts <= b)
    for x in wins:
        sel = (ts >= x["t0"]) & (ts <= x["t1"]) & ~cut_mask
        g = gs[sel] * 1e3
        x["n_gaps"] = int(len(g))
        x["step_med"] = float(np.median(g)) if len(g) else float("nan")
        x["step_p90"] = float(np.percentile(g, 90)) if len(g) else float("nan")
        x["step_p99"] = float(np.percentile(g, 99)) if len(g) else float("nan")
        # requests longer than the run (EP32: ~1400 s for OSL 8000) never complete inside a window; fall back to the
        # in-flight gaps loadgen logs every 10 s (median of the 10 s medians / p90s; n = number of 10 s records)
        if len(g) < 200 and len(live):
            s = (live[:, 0] >= x["t0"] + 10) & (live[:, 0] <= x["t1"])
            for a, b in cuts:
                s &= ~((live[:, 0] >= a) & (live[:, 0] <= b + 10))
            if s.any():
                x["n_gaps"] = int(s.sum())
                x["step_med"] = float(np.median(live[s, 1]))
                x["step_p90"] = float(np.median(live[s, 2]))
                x["step_p99"] = float(np.max(live[s, 2]))
                x["step_src"] = "live"
        x["nic"] = {n: fab_rate(f, x["t0"], x["t1"]) for n, f in fabs.items()}
        L = [l for l in logs if x["t0"] <= l[0] <= x["t1"] and not any(a <= l[0] <= b for a, b in cuts)]
        by_t = {}
        for t, eng, thr, run, wait in L:
            by_t.setdefault(eng, []).append((thr, run, wait))
        x["gen_tps"] = sum(statistics.mean(v[0] for v in vals) for vals in by_t.values()) if by_t else float("nan")
        x["running"] = sum(statistics.mean(v[1] for v in vals) for vals in by_t.values()) if by_t else float("nan")
    base = [x for x in wins if x["kind"] == "base" and x["n_gaps"]]
    b_med = statistics.mean(x["step_med"] for x in base) if base else float("nan")
    b_p90 = statistics.mean(x["step_p90"] for x in base) if base else float("nan")

    # profiler captures: trace files written during or after each capture (export happens at stop_profile)
    caps = []
    cur = {}
    for t, name, arg in ev:
        if name.startswith("prof_"):
            if arg == "start":
                cur[name] = t
            elif name in cur:
                caps.append((name[5:], cur.pop(name), t))
    traces = sorted(glob.glob(os.path.join(R, "prof", "**", "*.json*"), recursive=True)
                    + glob.glob(os.path.join(R, "logs", "prof", "**", "*.json*"), recursive=True), key=os.path.getmtime)
    cap_res = {}
    if not args.no_traces:
        # an EP32 run leaves ~32 traces per capture (2 GB in all): parse them in parallel
        todo = {name: [p for p in traces if a <= os.path.getmtime(p) <= b + 5] for name, a, b in caps}
        flat = sorted({p for v in todo.values() for p in v})
        with multiprocessing.Pool(min(24, max(1, len(flat)))) as pool:
            parsed = dict(zip(flat, pool.map(_trace_kernels_safe, flat)))
        for name, files in todo.items():
            per = []
            for p in files:
                k = parsed[p]
                if isinstance(k, str):
                    print(f"skip {p}: {k}")
                    continue
                if k["dispatch"]:
                    per.append({"file": os.path.basename(p), **{f"{kk}_med": float(np.median(v)) for kk, v in k.items() if v},
                                **{f"{kk}_mean": float(np.mean(v)) for kk, v in k.items() if v},
                                **{f"{kk}_n": len(v) for kk, v in k.items()}})
            cap_res[name] = per

    print(f"# E1 {os.path.basename(R)}\n")
    print("| 窗口 | 步时中位数 ms | 相对 base | p90 ms | 相对 base | p99 ms | 样本数 | 生成吞吐 tok/s | 运行请求 | " +
          " | ".join(f"{n} tx↑/rx↓ GB/s" for n in fabs) + " |")
    print("|---|---|---|---|---|---|---|---|---|" + "---|" * len(fabs))
    for x in wins:
        nic = " | ".join(f"{x['nic'][n][0]:.0f}↑ {x['nic'][n][1]:.0f}↓" for n in fabs)
        print(f"| {x['name']} | {x['step_med']:.2f} | {(x['step_med'] / b_med - 1) * 100:+.1f}% | {x['step_p90']:.2f} | "
              f"{(x['step_p90'] / b_p90 - 1) * 100:+.1f}% | {x['step_p99']:.1f} | {x['n_gaps']} | {x['gen_tps']:.0f} | "
              f"{x['running']:.0f} | {nic} |")
    if cap_res:
        print("\n### profiler 里的 MoRI EP 内核（µs，各 rank 文件的中位数再取平均）\n")
        print("| 采样 | 文件数 | dispatch 中位 | combine 中位 | dispatch 均值 | combine 均值 | 每文件 dispatch 次数 |")
        print("|---|---|---|---|---|---|---|")
        for name, per in cap_res.items():
            if not per:
                print(f"| {name} | 0 | - | - | - | - | - |")
                continue
            f = lambda k: statistics.mean(p[k] for p in per if k in p)  # noqa: E731
            print(f"| {name} | {len(per)} | {f('dispatch_med'):.1f} | {f('combine_med'):.1f} | {f('dispatch_mean'):.1f} | "
                  f"{f('combine_mean'):.1f} | {int(f('dispatch_n'))} |")
    json.dump({"windows": wins, "captures": cap_res, "base_step_med": b_med}, open(os.path.join(R, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
