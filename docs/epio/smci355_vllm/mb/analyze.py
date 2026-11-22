#!/usr/bin/env python3
"""Summarise one mb/campaign.sh run (EP over the EP nodes next to MoRI IO).

  analyze.py results/mb/<CAMP>   ->  markdown tables on stdout, <CAMP>/summary.json

Per scenario and EP shape: each run is reduced to per-rank statistics (dispatch mean / median / p99, combine mean /
median / p99; combine without the local GEMMs when the duty cycle is on), averaged over ranks (all; node A = the
first EP node, ranks 0-7; node B = the second, ranks 8-15, the one that receives the IO; at EP32 also the other two
nodes) and over the reps; the change is against the mean of the no-IO scenarios (base*).
  v2 back to back : the per-round, per-rank series in the rank-0 log ("dispatch duration [..] avg")
  v2 duty cycle   : the LOCAL line every rank prints (dispatch mean / median, combine_ex median)
  v1              : the per-rank JSON of ep_v1_bench.py
"""
import glob
import json
import os
import re
import sys

import numpy as np

ROUND = re.compile(r"^\s+(dispatch|combine) duration \[([^\]]*)\] avg")
WHAT = {
    "IW8": "C 写 B（8 对）", "IR8": "B 读 C（8 对）", "IWk": "C 写 B（KV 形状）", "IRk": "B 读 C（KV 形状）",
    "OW8": "A 写 C（8 对）", "OR8": "C 读 A（8 对）", "D2W1": "A 写 B（1 对）", "D2W8": "A 写 B（8 对）",
    "IW1": "C 写 B（1 对）", "IR1": "B 读 C（1 对）", "OWk": "A 写 C（KV 形状）", "ORk": "C 读 A（KV 形状）",
    "IWkT0": "C 写 B（KV 形状，IO 在 TC 0）", "IR8T0": "B 读 C（8 对，IO 在 TC 0）",
    "IWhL": "C 写 B 的主机内存（同 socket）", "IWhR": "C 写 B 的主机内存（跨 socket）",
    "IRhR": "B 读 C 到自己的主机内存（跨 socket）", "I2Wk": "C 和 A 同时写 B（2:1 汇入）",
    "IWkA": "C 写所有 EP 节点", "IWkQ": "C 写 B（16 QP × 8 worker）", "IWk64": "C 写 B（64 MB 一批）",
    "IWkA1g": "C 写所有 EP 节点（每卡 1 GB/s）", "IWkA2": "C 写所有 EP 节点（每卡 2 GB/s）",
    "IWkA5": "C 写所有 EP 节点（每卡 5 GB/s）", "IWkA10": "C 写所有 EP 节点（每卡 10 GB/s）",
    "IWkAT0": "C 写所有 EP 节点（IO 在 TC 0）",
    "IWk5": "C 写 B（每卡 5 GB/s）", "IWk10": "C 写 B（每卡 10 GB/s）", "IWk20": "C 写 B（每卡 20 GB/s）",
    "IRk5": "B 读 C（每卡 5 GB/s）", "IRk10": "B 读 C（每卡 10 GB/s）", "IRk20": "B 读 C（每卡 20 GB/s）",
    "IR8s": "B 读 C（8 对，每批 64 个）", "OR8": "C 读 A（8 对）", "ORk": "C 读 A（KV 形状）",
    "IR8m": "B 读 C（8 对，每批 1024 个）", "IR8L": "B 读 C（8 对，128 KB × 2048）",
    "IR8b256": "B 读 C（每批 256 个，在途 8 MB）", "IR8b1024": "B 读 C（每批 1024 个，在途 32 MB）",
    "IR8b2048": "B 读 C（每批 2048 个，在途 64 MB）", "IR8b4096": "B 读 C（每批 4096 个，在途 128 MB）",
}
KEYS = ("d_mean", "d_med", "d_p99", "c_mean", "c_med", "c_p99")


def rank_stats(d, c):
    d, c = np.asarray(d, float), np.asarray(c, float)
    return {"d_mean": d.mean(), "d_med": np.median(d), "d_p99": np.percentile(d, 99),
            "c_mean": c.mean(), "c_med": np.median(c), "c_p99": np.percentile(c, 99)}


def parse_v2(log):
    """-> {rank: stats} or None"""
    rounds = {"dispatch": [], "combine": []}
    local = {}
    width = None
    for line in open(log, errors="replace"):
        m = ROUND.match(line)
        if m:
            v = [float(x) for x in m.group(2).split(",") if x.strip()]
            width = width or len(v)
            if len(v) == width:
                rounds[m.group(1)].append(v)
        elif line.startswith("LOCAL "):
            kv = dict(x.split("=", 1) for x in line.split()[1:] if "=" in x)
            local[int(kv["rank"])] = kv
    if local:
        out = {}
        for r, kv in local.items():
            cm = float(kv["combine_ex_us_med"])
            out[r] = {"d_mean": float(kv["dispatch_us_mean"]), "d_med": float(kv["dispatch_us_med"]),
                      "d_p99": float("nan"), "c_mean": cm, "c_med": cm, "c_p99": float("nan")}
        return out
    if not rounds["dispatch"] or not rounds["combine"]:
        return None
    d, c = np.array(rounds["dispatch"]), np.array(rounds["combine"])
    return {r: rank_stats(d[:, r], c[:, r]) for r in range(d.shape[1])}


def parse_v1(jdir):
    out = {}
    for f in glob.glob(os.path.join(jdir, "rank*.json")):
        j = json.load(open(f))
        for res in j["results"]:
            out[res["rank"]] = rank_stats(res["dispatch_us"], res["combine_ex_us"])
            out[res["rank"]]["rel_err"] = j["rel_err"]
    return out or None


def group(st, ranks):
    rows = [st[r] for r in ranks if r in st]
    return {k: float(np.nanmean([x[k] for x in rows])) if rows else float("nan") for k in KEYS}


def parse_rate(path):
    res = {}
    if os.path.exists(path):
        for line in open(path):
            m = re.match(r"^(\w)\s+tx(.*)\|\s*rx(.*)\|\s*sum tx ([\d.]+) rx ([\d.]+)", line)
            if m:
                res[m.group(1)] = (float(m.group(4)), float(m.group(5)))
    return res


def snap_delta(path):
    pre, post, out = {}, {}, {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        try:
            s = json.loads(line)
        except json.JSONDecodeError:
            continue
        (pre if s["tag"] == "pre" else post)[s["host"].replace("smci355-ccs-aus-", "")] = s
    for h in pre:
        if h in post:
            a, b = pre[h], post[h]
            d = {c: sum(b["dev"][k][c] - a["dev"][k][c] for k in a["dev"])
                 for c in ("tx_rdma_retx_pkts", "tx_rdma_ack_timeout", "rx_rdma_cnp_pkts", "rx_rdma_ecn_pkts")}
            for c in ("tx_us_p3", "rx_us_p3"):
                d["pfc_" + c] = b["pfc"][c] - a["pfc"][c]
            d["secs"] = b["ts"] - a["ts"]
            out[h] = d
    return out


def main():
    root = sys.argv[1].rstrip("/")
    log = os.path.join(root, "campaign.log")
    head = open(log).readline() if os.path.exists(log) else ""
    m = re.search(r"A=(\S+) .* B=(\S+) .* C=(\S+) .*EP nodes: (.*?) IOTC", head) or re.search(r"A=(\S+) .* B=(\S+) .* C=(\S+) ", head)
    A, B, C = m.groups()[:3] if m else ("A", "B", "C")
    ep_nodes = m.group(4).split() if m and m.lastindex and m.lastindex >= 4 else [A, B]
    hosts = ep_nodes + [C]
    camp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "campaign.sh")
    order = [l.split("|")[0].strip(' "') for l in open(camp) if re.match(r'^\s+"\w+\|', l)]
    scen = [s for s in order if os.path.isdir(os.path.join(root, s))]
    S = {}
    world = 0
    for s in scen:
        d = os.path.join(root, s)
        runs = {}
        for f in sorted(os.listdir(d)):
            mm = re.match(r"ep_(\w+?)_(\d+)_r0\.log$", f)
            if not mm:
                continue
            sh, rep = mm.groups()
            st = parse_v1(os.path.join(d, f"ep_{sh}_{rep}_json")) if sh.startswith("v1") else parse_v2(os.path.join(d, f))
            if st:
                runs.setdefault(sh, []).append(st)
                world = max(world, max(st) + 1)
        ep = {}
        for sh, rs in runs.items():
            w = max(max(st) + 1 for st in rs)
            grp = [("all", range(w)), ("A", range(0, 8)), ("B", range(8, 16))]
            if w > 16:
                grp.append(("rest", range(16, w)))
            g = {name: [group(st, rk) for st in rs] for name, rk in grp}
            ep[sh] = {name: {k: float(np.mean([x[k] for x in v])) for k in KEYS} for name, v in g.items()}
            ep[sh]["reps_d_mean"] = [round(x["d_mean"], 1) for x in g["all"]]
            errs = [st[r].get("rel_err") for st in rs for r in st if "rel_err" in st[r]]
            if errs:
                ep[sh]["max_rel_err"] = max(errs)
        S[s] = {"ep": ep, "rate": parse_rate(os.path.join(d, "rate_before.txt")), "cnt": snap_delta(os.path.join(d, "snap.jsonl"))}
    shapes = sorted({sh for s in S for sh in S[s]["ep"]})
    base = {}
    for sh in shapes:
        bs = [S[s]["ep"][sh] for s in S if s.startswith("base") and sh in S[s]["ep"]]
        if bs:
            base[sh] = {g: {k: float(np.mean([b[g][k] for b in bs])) for k in KEYS} for g in bs[0] if g in ("all", "A", "B", "rest")}
    S["_baseline"] = base
    json.dump(S, open(os.path.join(root, "summary.json"), "w"), indent=1)

    def pc(v, b):
        return f"{(v / b - 1) * 100:+.0f}%" if b and b == b and v == v else "-"

    print(f"# {os.path.basename(root)}: EP{world} on {' + '.join(ep_nodes)} (A={A}, B={B} receives the IO), IO node C={C}\n")
    labels = ["A", "B", "D", "E", "F", "G", "H"][:len(ep_nodes)] + ["C"]
    for sh in shapes:
        if sh not in base:
            continue
        b = base[sh]
        ck = "c_mean"
        rest = "rest" in b
        print(f"\n### EP 形状 {sh}（µs，各 rank 均值；括号里是相对无 IO 的变化）\n")
        print("| 场景 | IO 怎么走 | IO 速率 " + " / ".join(labels) + "（tx↑ rx↓, GB/s） | dispatch | A 侧 | B 侧 | " +
              ("其余节点 | " if rest else "") + "combine | A 侧 | B 侧 | " + ("其余节点 | " if rest else "") + "dispatch p99 | 各遍 dispatch |")
        print("|---|---|---|---|---|---|" + ("---|" if rest else "") + "---|---|---|" + ("---|" if rest else "") + "---|---|")
        for s in scen:
            e = S[s]["ep"].get(sh)
            if not e:
                continue
            r = S[s]["rate"]
            rate = " / ".join(f"{r[n][0]:.0f}↑{r[n][1]:.0f}↓" if n in r else "-" for n in labels)

            def cell(g, k):
                return f"{e[g][k]:.1f} ({pc(e[g][k], b[g][k])})" if g in e and g in b else "-"

            dcells = [cell("all", "d_mean"), cell("A", "d_mean"), cell("B", "d_mean")] + ([cell("rest", "d_mean")] if rest else [])
            ccells = [cell("all", ck), cell("A", ck), cell("B", ck)] + ([cell("rest", ck)] if rest else [])
            p99 = f"{e['all']['d_p99']:.0f} ({pc(e['all']['d_p99'], b['all']['d_p99'])})"
            print(f"| {s} | {WHAT.get(s, '无 IO')} | {rate} | " + " | ".join(dcells) + " | " + " | ".join(ccells) +
                  f" | {p99} | {' / '.join(str(x) for x in e['reps_d_mean'])} |")
        errs = [S[s]["ep"][sh].get("max_rel_err") for s in scen if sh in S[s]["ep"] and "max_rel_err" in S[s]["ep"][sh]]
        if errs:
            print(f"\n（{sh} combine 闭式校验最大相对误差 {max(errs):.1e}）")
    print(f"\n### 计数器（整个场景期间的增量，节点 {' / '.join(hosts)}）\n")
    print("| 场景 | 重传包 | ACK 超时 | 收到 CNP | 发出 PFC µs（p3） | 收到 PFC µs（p3） |")
    print("|---|---|---|---|---|---|")
    for s in scen:
        c = S[s]["cnt"]
        g = lambda k: "/".join(str(int(c.get(n, {}).get(k, 0))) for n in hosts)  # noqa: E731
        print(f"| {s} | {g('tx_rdma_retx_pkts')} | {g('tx_rdma_ack_timeout')} | {g('rx_rdma_cnp_pkts')} | "
              f"{g('pfc_tx_us_p3')} | {g('pfc_rx_us_p3')} |")


if __name__ == "__main__":
    main()
