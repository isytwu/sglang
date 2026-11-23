#!/usr/bin/env python3
"""Per-window IO effect for an E1 run whose step time drifts (EP32: requests start finishing ~13 min into the load,
new ones prefill on the decode, and the p90 of every later 10 s record rises to ~720 ms with or without IO).

For each IO window: the median of loadgen's 10 s gap medians while the IO is on ([io_on - 15, io_end]) against the same
for the IO-off stretches right before and after it (the gap before the window and the gap after it), profiler
captures excluded. Plus each decode node's CNPs per second (rx_rdma_cnp_pkts) over the same stretches.

  python3 tools/ep32_win.py results/e2e/<run>
"""
import json
import os
import statistics
import sys
import time

import numpy as np

R = sys.argv[1].rstrip("/")
ev = []
for line in open(os.path.join(R, "events.tsv")):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        try:
            ev.append((float(p[0]), p[1], p[2] if len(p) > 2 else ""))
        except ValueError:
            pass

wins, cuts, start = [], [], {}
for t, k, a in ev:
    if k.startswith("prof_"):
        start[k] = t if a == "start" else start.get(k)
        if a == "end" and start.get(k):
            cuts.append((start[k] - 10, t + 30))
    elif a == "start" and not k.endswith(("_io_on", "_io_end")) and k not in ("serve_start",):
        wins.append({"name": k, "start": t, "on": None, "off": None, "end": None})
    elif k.endswith("_io_on") and wins:
        wins[-1]["on"] = t
    elif k.endswith("_io_end") and wins:
        wins[-1]["off"] = t
    elif a == "end" and wins and wins[-1]["name"] == k:
        wins[-1]["end"] = t

live = []
for line in open(os.path.join(R, "load", "live.jsonl")):
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    if r.get("gap_med_ms"):
        live.append((r["t"], r["gap_med_ms"]))
live = np.array(live)


def clean(t0, t1):
    s = (live[:, 0] >= t0 + 10) & (live[:, 0] <= t1)
    for a, b in cuts:
        s &= ~((live[:, 0] >= a) & (live[:, 0] <= b + 10))
    return live[s, 1]


def fab(path):
    rows, hdr = [], None
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
    return a[:, 0], a[:, -2]


fabs = {}
for f in sorted(os.listdir(R)):
    if f.startswith("fab_") and f.endswith(".tsv"):
        n = f[4:-4]
        try:
            fabs[n] = fab(os.path.join(R, f))
        except Exception:
            pass


def cnp(n, t0, t1):
    t, c = fabs[n]
    i, j = np.searchsorted(t, t0), np.searchsorted(t, t1) - 1
    return (c[j] - c[i]) / (t[j] - t[i]) if j > i else float("nan")


dec = [n for n in fabs if any(n in w for w in open(os.path.join(R, "driver.log")).readline().split())] or list(fabs)
print(f"# {os.path.basename(R)}: IO-on vs the IO-off stretches around each window (median of 10 s gap medians, ms)\n")
print("| window | IO on | n | IO off around | n | change | CNP/s per decode node, IO on |")
print("|---|---|---|---|---|---|---|")
for i, w in enumerate(wins):
    if w["name"] == "base":
        b = clean(w["start"], w["end"] or w["start"] + 120)
        if len(b):
            print(f"| base ({time.strftime('%H:%M', time.gmtime(w['start']))}) | {statistics.median(b):.1f} | {len(b)} | | | | "
                  + " ".join(f"{cnp(n, w['start'], w['end'] or w['start'] + 120):.0f}" for n in dec) + " |")
        continue
    if not (w["on"] and w["off"]):
        print(f"| {w['name']} | (incomplete) | | | | | |")
        continue
    # io_on is logged ~20 s after the senders came up (33 s after the window start with pre-started receivers, 50 s
    # without), so the IO has been on for a while by then
    on = clean(w["on"] - 15, w["off"])
    prev_end = wins[i - 1]["end"] if i else None
    nxt = wins[i + 1]["start"] if i + 1 < len(wins) else None
    off = np.concatenate([clean(prev_end, w["start"]) if prev_end else np.array([]),
                          clean(w["end"], nxt) if (w["end"] and nxt) else np.array([])])
    if not len(on):
        continue
    m_on = statistics.median(on)
    m_off = statistics.median(off) if len(off) else float("nan")
    print(f"| {w['name']} | {m_on:.1f} | {len(on)} | {m_off:.1f} | {len(off)} | {(m_on / m_off - 1) * 100:+.1f}% | "
          + " ".join(f"{cnp(n, w['on'], w['off']):.0f}" for n in dec) + " |")
