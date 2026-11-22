#!/usr/bin/env python3
"""Natural-traffic analysis of P/D runs (e2e/pd.sh):  pd_analyze.py <run dir> [<run dir> ...]

For every second of a run's "nat" period (natural KV traffic, no injected IO) and every decode node: the KV coming
in (the node's received minus sent bytes: EP is symmetric between the two decode nodes, the KV only comes in), the
CNPs it received (each one throttles one of its sending QPs, and a decode node sends only EP data and READ requests),
the ECN-marked packets it received, and the decode's token gaps in that second (e2e/pd_proxy.py's series.jsonl,
one gap per decode step of a stream). Prints the step time against the KV inflow and against CNPs, per run, so that
a TC 104 run and a TC 0 run (KV away from EP's class) can be compared second by second.
"""
import datetime as dt
import glob
import json
import os
import re
import sys

import numpy as np


def events(run):
    ev = {}
    for line in open(os.path.join(run, "events.tsv")):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3:
            ev.setdefault(f"{p[1]} {p[2]}", float(p[0]))
    return ev


def fab(path):
    rows = []
    hdr = None
    for line in open(path):
        if line.startswith("#"):
            continue
        p = line.rstrip("\n").split("\t")
        if hdr is None:
            hdr = p
            continue
        try:
            rows.append([float(x) for x in p])
        except ValueError:
            pass
    a = np.array(rows)
    t = a[:, 0]
    tx, rx = a[:, 7:23:2].sum(1), a[:, 8:23:2].sum(1)
    cnp = a[:, hdr.index("cnp_rx")] if "cnp_rx" in hdr else np.zeros(len(t))
    ecn = a[:, hdr.index("ecn_rx")] if "ecn_rx" in hdr else np.zeros(len(t))
    pfc = a[:, 3]
    # per-second rates on the samples' own clock
    dt = np.diff(t)
    d = lambda x: np.diff(x) / dt  # noqa: E731
    return {"t": t[1:], "tx": d(tx), "rx": d(rx), "cnp": d(cnp), "ecn": d(ecn), "pfc_us": d(pfc)}


def series(run):
    out = {}
    for line in open(os.path.join(run, "load", "series.jsonl")):
        r = json.loads(line)
        out[r["t"]] = r
    return out


def at(f, k, t):
    i = np.searchsorted(f["t"], t)
    return f[k][i] if 0 <= i < len(f["t"]) else np.nan


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def one(run):
    ev = events(run)
    ser = series(run)
    # a run still in its nat period: up to the last second the proxy wrote
    t0, t1 = ev.get("nat start"), ev.get("nat end") or (max(ser) if ser else None)
    if t0 is None or t1 is None:
        print(f"{run}: no nat period")
        return None
    head = open(os.path.join(run, "driver.log")).readline()
    dec = head.split(" decode ")[1].split(" (")[0].split()
    fabs = {n: fab(os.path.join(run, f"fab_{n}.tsv")) for n in dec}
    urls = sorted({u for r in ser.values() for u in r["per"]})
    rows = []
    for s in range(int(t0) + 1, int(t1)):
        r = ser.get(s)
        if not r:
            continue
        # the decode is one DP group: a step takes as long on every rank, so one gap figure per second
        meds = [r["per"][u]["med"] for u in urls if r["per"][u]["med"] is not None]
        if not meds:
            continue
        kv = sum(max(0.0, at(fabs[n], "rx", s + 0.5) - at(fabs[n], "tx", s + 0.5)) for n in dec) / 1e9
        cnp = sum(at(fabs[n], "cnp", s + 0.5) for n in dec)
        ecn = sum(at(fabs[n], "ecn", s + 0.5) for n in dec)
        legs = sum(r["per"][u]["dstart"] for u in urls)
        rows.append((s, float(np.mean(meds)), kv, cnp, ecn, legs))
    a = np.array(rows)
    name = os.path.basename(run.rstrip("/"))
    print(f"\n## {name}: {len(a)} s of natural traffic ({(t1 - t0) / 60:.0f} min), decode {' '.join(dec)}\n")
    step, kv, cnp, ecn = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    print(f"step time per second: median {np.median(step):.1f} ms, p90 {pct(step, 90):.1f}, p99 {pct(step, 99):.1f}; "
          f"KV in (both decode nodes) mean {kv.mean():.2f} GB/s, p99 {pct(kv, 99):.1f}; "
          f"CNPs received {cnp.mean():.0f}/s (seconds with any: {100 * (cnp > 0).mean():.1f}%); "
          f"ECN marks {ecn.mean():.0f}/s\n")
    print("| KV in that second (GB/s) | seconds | share | step median (ms) | step p90 | CNPs/s at the decode nodes |")
    print("|---|---|---|---|---|---|")
    for lo, hi in ((0, 1), (1, 5), (5, 20), (20, 60), (60, 1e9)):
        m = (kv >= lo) & (kv < hi)
        if m.sum():
            print(f"| {lo}–{hi if hi < 1e9 else '∞'} | {m.sum()} | {100 * m.mean():.1f}% | {np.median(step[m]):.1f} | "
                  f"{pct(step[m], 90):.1f} | {cnp[m].mean():.0f} |")
    print("\n| CNPs received by the decode nodes in that second | seconds | share | step median (ms) | step p90 |")
    print("|---|---|---|---|---|")
    for lo, hi in ((0, 1), (1, 1000), (1000, 10000), (10000, 1e12)):
        m = (cnp >= lo) & (cnp < hi)
        if m.sum():
            print(f"| {lo}–{hi if hi < 1e12 else '∞'} | {m.sum()} | {100 * m.mean():.1f}% | {np.median(step[m]):.1f} | "
                  f"{pct(step[m], 90):.1f} |")
    return {"run": name, "step_med": float(np.median(step)), "step_p90": pct(step, 90), "step_p99": pct(step, 99),
            "kv_mean": float(kv.mean()), "cnp_mean": float(cnp.mean()), "cnp_share": float((cnp > 0).mean())}


LOGLINE = re.compile(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d).*?Engine (\d+): .*?Avg generation throughput: ([\d.]+) tokens/s, "
                     r"Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%")


def windows_normalized(run):
    """AgentX's batch changes all the time, so a window's raw step time says little. Fit the step time on the
    decode's running requests and KV cache usage (the batch's context; the conversations grow over a run) over the
    nat period (10 s points: the proxy's live.jsonl against the decode servers' 10 s log lines), then give every IO
    window its mean step time against that fit."""
    ev = [line.rstrip("\n").split("\t") for line in open(os.path.join(run, "events.tsv"))]
    year = dt.datetime.fromtimestamp(float(ev[0][0]), dt.timezone.utc).year
    running = {}
    for p in glob.glob(os.path.join(run, "serve_*.log")):
        for line in open(p, errors="replace"):
            m = LOGLINE.search(line)
            if m:
                t = dt.datetime.strptime(f"{year}-{m.group(1)}", "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=dt.timezone.utc).timestamp()
                running.setdefault(os.path.basename(p), []).append((t, int(m.group(4)), float(m.group(6))))
    live = [json.loads(line) for line in open(os.path.join(run, "load", "live.jsonl"))]

    def run_at(t):
        tot, kv = 0, 0.0
        for v in running.values():
            near = [(r, k) for tt, r, k in v if t - 10 <= tt <= t]
            if not near:
                return None
            tot += near[-1][0]
            kv += near[-1][1]
        return tot, kv

    marks = {}
    for t, name, arg in ((float(e[0]), e[1], e[2]) for e in ev if len(e) >= 3):
        marks.setdefault(name, {}).setdefault(arg, t)
    # a torch-profiler capture stalls the engine for seconds when it exports: leave out the 10 s points around it
    cuts = [(w["start"] - 10, w["end"] + 30) for n, w in marks.items() if n.startswith("prof_") and "end" in w]
    pts = [(x["t"], x["gap_med_ms"], run_at(x["t"])) for x in live if x.get("gap_med_ms") and x["gap_med_ms"] > 5
           and not any(a <= x["t"] <= b for a, b in cuts)]
    pts = [(t, s, r[0], r[1]) for t, s, r in pts if r is not None]
    nat = marks.get("nat", {})
    if "start" not in nat:
        return
    t1 = nat.get("end", max(p[0] for p in pts))
    fit = np.array([(r, k, s) for t, s, r, k in pts if nat["start"] + 60 <= t <= t1])
    if len(fit) < 10:
        return
    X = np.column_stack([np.ones(len(fit)), fit[:, 0], fit[:, 1]])
    coef = np.linalg.lstsq(X, fit[:, 2], rcond=None)[0]
    res = fit[:, 2] - X @ coef
    print(f"\n### {os.path.basename(run.rstrip('/'))}: step time against the decode's batch\n")
    print(f"nat fit over {len(fit)} 10-s points: step = {coef[0]:.1f} ms + {coef[1]:.2f} ms per running request + "
          f"{coef[2]:.2f} ms per % of KV cache in use (running {fit[:, 0].min():.0f}–{fit[:, 0].max():.0f}, "
          f"KV {fit[:, 1].min():.0f}–{fit[:, 1].max():.0f}% summed over the ranks); residual sd {res.std():.1f} ms\n")
    print("| window | 10-s points | running (mean) | KV in use (%, summed) | step (median, ms) | fit (ms) | step against the fit |")
    print("|---|---|---|---|---|---|---|")
    for name, w in marks.items():
        if "start" not in w or "end" not in w or not re.match(r"^(base|SPIN|I[WR]\w+)$", name):
            continue
        sel = np.array([(r, k, s) for t, s, r, k in pts if w["start"] + 30 <= t <= w["end"]])
        if not len(sel):
            continue
        pred = np.column_stack([np.ones(len(sel)), sel[:, 0], sel[:, 1]]) @ coef
        ratio = np.median(sel[:, 2] / pred)
        print(f"| {name} | {len(sel)} | {sel[:, 0].mean():.0f} | {sel[:, 1].mean():.0f} | {np.median(sel[:, 2]):.1f} | "
              f"{np.median(pred):.1f} | {(ratio - 1) * 100:+.1f}% |")


def main():
    for run in sys.argv[1:]:
        windows_normalized(run)
    res = [r for r in (one(run) for run in sys.argv[1:]) if r]
    if len(res) > 1:
        print("\n| run | step median | p90 | p99 | KV in (GB/s) | CNPs/s | seconds with CNPs |")
        print("|---|---|---|---|---|---|---|")
        for r in res:
            print(f"| {r['run']} | {r['step_med']:.1f} | {r['step_p90']:.1f} | {r['step_p99']:.1f} | {r['kv_mean']:.2f} | "
                  f"{r['cnp_mean']:.0f} | {100 * r['cnp_share']:.1f}% |")


if __name__ == "__main__":
    main()
