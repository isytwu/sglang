#!/usr/bin/env python3
"""Figures for the K3.1 EP x KV write-up (docs/epio), one question per figure, Chinese labels.

  k31_doc_figs.py <out_dir>
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k31_analyze import extract, load_steps, netmon_node, offsets  # noqa: E402

T = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R1 = os.path.join(T, "results/k31_2p1d_tc104/c256_base")
R2 = os.path.join(T, "results/k31_2p1d_tc104/c256_inject")
R3 = os.path.join(T, "results/k31_2p1d_tc0/c256_base")
font_manager.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
plt.rcParams.update({"font.sans-serif": ["WenQuanYi Zen Hei"], "axes.unicode_minus": False, "font.size": 11})
C_KV, C_EP, C_STEP = "#d62728", "#1f77b4", "#2ca02c"
GB = 1e9


def kv_ep_5ms(sub, t_lo, t_hi):
    """KV into node02 and EP into node02 (sent by node04) per 5 ms bin, GB/s."""
    name = os.path.basename(sub)
    off = offsets(sub, name, ["01", "03", "02", "04"], "02")
    edges = np.arange(t_lo, t_hi, 0.005)
    n02 = netmon_node(extract(sub, "02"), off.get("02", 0.0), edges)
    n04 = netmon_node(extract(sub, "04"), off.get("04", 0.0), edges)
    rails = sorted(n02)
    ep_in = sum(np.nan_to_num(n04[r]["tx"]) for r in rails) / 0.005 / GB
    kv_in = sum(np.nan_to_num(n02[r]["rx"]) for r in rails) / 0.005 / GB - ep_in
    return edges[:-1], kv_in, ep_in, off


def fig_zoom(out):
    a = json.load(open(os.path.join(R1, "analysis.json")))
    t_lo, t_hi = a["window"]
    t, kv, ep, off = kv_ep_5ms(R1, t_lo, t_hi)
    # the minute with the most 5 ms bins above 40 GB/s
    m = (t - t[0]) // 60
    counts = np.bincount(m.astype(int), weights=(kv > 40).astype(float))
    k = int(np.argmax(counts[1:-1]) + 1)
    sel = m == k
    t0 = t[sel][0]
    st = load_steps(os.path.join(R1, "_x", "node02"), "decode", off.get("02", 0.0))[0]
    ms = (st["mode"] == "target_verify") & (st["t0"] >= t0) & (st["t0"] < t0 + 60) & (st["ms"] > 0)
    fig, ax = plt.subplots(4, 1, figsize=(12, 9.5), sharex=True, gridspec_kw={"height_ratios": [2, 1.2, 1.2, 1]})
    ax[0].plot(t[sel] - t0, kv[sel], color=C_KV, lw=0.8)
    ax[0].set_ylabel("KV 流入 (GB/s)")
    ax[0].set_title("真实负载中的一分钟（decode 节点 node02，5 ms 粒度）")
    ax[1].plot(t[sel] - t0, ep[sel], color=C_EP, lw=0.8)
    ax[1].set_ylabel("EP 流入 (GB/s)")
    ax[1].set_ylim(0, max(6, np.nanmax(ep[sel]) * 1.2))
    ax[2].scatter(st["t0"][ms] - t0, st["ms"][ms], s=8, color=C_STEP)
    ax[2].set_ylabel("decode 步 (ms)")
    ax[2].set_ylim(150, 280)
    ax[3].plot(st["t0"][ms] - t0, st["bs"][ms], color="#7f7f7f", lw=1)
    ax[3].set_ylabel("batch（请求数）")
    ax[3].set_xlabel("秒")
    for x in ax:
        x.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_one_minute.png"), dpi=120)
    plt.close(fig)


def fig_bins(out):
    labels = ["0.5–5", "5–20", "20–50", "50–100"]
    names = {"<0.5 GB/s": "base", "0.5-5": "0.5–5", "5-20": "5–20", "20-50": "20–50", "50-100": "50–100"}

    def rel(sub):
        a = json.load(open(os.path.join(sub, "analysis.json")))
        vals = {lab: [] for lab in labels}
        counts = {lab: 0 for lab in labels}
        for v in a["decode_steps"].values():
            bins = {names.get(b["kv_GBps"]): b for b in v["kv_bins"] if names.get(b["kv_GBps"])}
            if "base" not in bins:
                continue
            for lab in labels:
                if lab in bins:
                    vals[lab].append(bins[lab]["resid_pct_mean"] - bins["base"]["resid_pct_mean"])
                    counts[lab] += bins[lab]["steps"]
        return [np.mean(vals[lab]) if vals[lab] else np.nan for lab in labels], [counts[lab] for lab in labels]

    v104, c104 = rel(R1)
    v0, c0 = rel(R3)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    b1 = ax.bar(x - 0.2, v104, 0.4, label="TC 104（无损，实际配置）", color=C_EP)
    b2 = ax.bar(x + 0.2, v0, 0.4, label="TC 0（有损，对照）", color="#ff7f0e")
    for bars, cs in ((b1, c104), (b2, c0)):
        for b, c in zip(bars, cs):
            if np.isfinite(b.get_height()):
                ax.annotate(f"{c} 步", (b.get_x() + b.get_width() / 2, max(b.get_height(), 0)), ha="center",
                            va="bottom", fontsize=8, color="#555")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s} GB/s" for s in labels])
    ax.set_xlabel("这个 decode 步执行期间，流入 decode 节点的 KV 速率")
    ax.set_ylabel("比没有 KV 的步慢多少 (%)")
    ax.set_title("decode 步时间 vs 同时到达的 KV（已扣除 batch 大小的影响）")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_step_vs_kv.png"), dpi=120)
    plt.close(fig)


def fig_burst(out):
    a = json.load(open(os.path.join(R1, "analysis.json")))
    t, kv, ep, _ = kv_ep_5ms(R1, *a["window"])
    quiet = 20
    lag = np.arange(-40, 80)
    on = np.where(kv > 20)[0]
    onsets = np.array([i for i in on if i > quiet and np.all(kv[i - quiet:i] < 1) and i + lag[-1] < kv.size])
    idx = onsets[:, None] + lag[None, :]
    ms = lag * 5
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(ms, kv[idx].mean(0), color=C_KV, lw=1.5)
    ax[0].set_ylabel("KV 流入 (GB/s)")
    ax[0].set_title(f"{len(onsets)} 次 KV 突发对齐后取平均（node02，5 ms 粒度）")
    ax[1].plot(ms, ep[idx].mean(0), color=C_EP, lw=1.5)
    base = ep[idx][:, lag < -10].mean()
    ax[1].axhline(base, color=C_EP, ls="--", lw=0.8)
    ax[1].set_ylim(base * 0.8, base * 1.2)
    ax[1].set_ylabel("EP 流入 (GB/s)")
    ax[1].set_xlabel("相对 KV 突发开始的时间 (ms)")
    for x in ax:
        x.axvline(0, color="k", lw=0.6)
        x.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_burst_aligned.png"), dpi=120)
    plt.close(fig)


def fig_inject(out):
    inj = json.load(open(os.path.join(R2, "inject.json")))
    kinds = [("写 decode 的 GPU 显存（KV 形状）", ("A", "A1", "A2")), ("写 decode 的主机内存，同 NUMA", ("C1",)),
             ("写 decode 的主机内存，跨 NUMA", ("C2",))]
    rates = [("≈5 GB/s", 4, 6), ("≈21 GB/s", 19, 22), ("36–38 GB/s", 35, 39)]
    colors = [C_EP, "#9467bd", C_KV]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(rates))
    for j, ((name, phases), col) in enumerate(zip(kinds, colors)):
        mean, lo, hi = [], [], []
        for _, a, b in rates:
            v = [w[r]["slowdown_pct"] for w in inj["windows"] if w["phase"] in phases
                 and a <= (w.get("achieved_GBps_per_nic") or w["rate"]) <= b for r in ("r0", "r8") if r in w]
            mean.append(np.mean(v) if v else np.nan)
            lo.append(np.mean(v) - np.min(v) if v else 0)
            hi.append(np.max(v) - np.mean(v) if v else 0)
        bars = ax.bar(x + (j - 1) * 0.27, mean, 0.27, yerr=[lo, hi], capsize=3, label=name, color=col)
        for bar, m, h in zip(bars, mean, hi):
            ax.annotate(f"{m:+.1f}%", (bar.get_x() + bar.get_width() / 2, max(m + h, 0) + 0.25), ha="center",
                        fontsize=9)
    ax.axhspan(-2, 2, color="#999", alpha=0.15, label="单个窗口的噪声范围（±2%）")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rates])
    ax.set_xlabel("每张网卡的注入速率（网卡线速 50 GB/s；38 GB/s 档为背靠背，每次 EP 都和注入撞上）")
    ax.set_ylabel("decode 步变慢 (%)")
    ax.set_title("在真实负载上额外注入 RDMA 写：decode 步变慢多少")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_injection.png"), dpi=120)
    plt.close(fig)


def fig_inject_timeline(out):
    inj = json.load(open(os.path.join(R2, "inject.json")))
    with open(os.path.join(R2, "series_1s.tsv")) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        s = np.loadtxt(f, delimiter="\t")
    col = {h: i for i, h in enumerate(hdr)}
    t = s[:, col["t"]]
    t0 = t[0]
    rx = np.nan_to_num(s[:, col["kv_in_02_GBps"]]) + np.nan_to_num(s[:, col["kv_in_04_GBps"]])
    step = s[:, col["step_ms_tp0"]]
    k = np.ones(10) / 10
    good = np.isfinite(step)
    step_s = np.convolve(np.where(good, step, 0), k, "same") / np.maximum(np.convolve(good.astype(float), k, "same"), 1e-9)
    style = {"A": ("GPU 显存 1:1", C_EP), "A1": ("GPU 显存 1:1", C_EP), "A2": ("GPU 显存 1:1", C_EP),
             "B": ("GPU 显存 2:1 汇聚", "#17becf"), "C1": ("主机内存 同 NUMA", "#9467bd"),
             "C2": ("主机内存 跨 NUMA", C_KV)}
    fig, ax = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
    ax[0].plot((t - t0) / 60, rx, color="#444", lw=0.8)
    ax[0].set_ylabel("流入两台 decode 的\nKV + 注入 (GB/s)")
    ax[1].plot((t - t0) / 60, step_s, color=C_STEP, lw=1)
    ax[1].set_ylabel("decode 步 (ms, 10 s 平均)")
    ax[1].set_xlabel("分钟（profiling 阶段）")
    seen = set()
    for w in inj["windows"]:
        lab, c = style[w["phase"]]
        for x in ax:
            x.axvspan((w["t0"] - t0) / 60, (w["t1"] - t0) / 60, color=c, alpha=0.18,
                      label=lab if (x is ax[0] and lab not in seen) else None)
        seen.add(lab)
        ax[0].annotate(f"{w.get('achieved_GBps_per_nic') or w['rate']:.0f}", ((w["t0"] + w["t1"]) / 2 / 60 - t0 / 60,
                       rx.max() * 1.03), fontsize=9, color=c, ha="center")
    ax[0].set_title("注入实验的全过程：色块是注入窗口（上方数字是每张网卡 GB/s）")
    ax[0].set_ylim(0, rx.max() * 1.12)
    ax[0].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=4, frameon=False)
    for x in ax:
        x.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig5_injection_timeline.png"), dpi=120)
    plt.close(fig)


def fig_breakdown(out):
    parts = [("MoE 专家计算（GEMM、排序）", 530.2), ("fp8 量化", 165.4), ("其他", 135.4),
             ("MoRI EP（dispatch + combine）", 88.1), ("TP / DCP 集合通信", 38.4)]
    total = sum(v for _, v in parts)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    y = np.arange(len(parts))[::-1]
    for yy, (name, v) in zip(y, parts):
        c = C_EP if "EP" in name else "#bbbbbb"
        ax.barh(yy, v / total * 100, color=c)
        ax.text(v / total * 100 + 0.8, yy, f"{v / total * 100:.1f}%（每步约 {v / 5:.0f} ms）", va="center", fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels([n for n, _ in parts])
    ax.set_xlim(0, 75)
    ax.set_xlabel("占 decode 一步 GPU 时间的比例 (%)（负载下 5 步的 profile，每步约 229 ms）")
    ax.set_title("decode 一步里 GPU 在做什么：EP 只占 9%")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig6_step_breakdown.png"), dpi=120)
    plt.close(fig)


def fig_retx(out):
    a104 = json.load(open(os.path.join(R1, "analysis.json")))["retx"]
    a0 = json.load(open(os.path.join(R3, "analysis.json")))["retx"]
    groups = [("prefill 节点\n（KV 发送方）", ("01", "03")), ("decode 节点\n（EP 发送方）", ("02", "04"))]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for j, (key, ttl, fmt) in enumerate((("retx_frac", "重传包占发出包的比例", "{:.1e}"),
                                         ("tx_rdma_ack_timeout", "ACK 超时次数（约 35–55 分钟）", "{:.0f}"))):
        x = np.arange(len(groups))
        for i, (a, lab, c) in enumerate(((a104, "TC 104", C_EP), (a0, "TC 0", "#ff7f0e"))):
            v = [max(np.mean([a[n][key] for n in ns]), 1e-9 if key == "retx_frac" else 0.5) for _, ns in groups]
            raw = [np.mean([a[n][key] for n in ns]) for _, ns in groups]
            bars = ax[j].bar(x + (i - 0.5) * 0.38, v, 0.38, label=lab, color=c)
            for b, r in zip(bars, raw):
                ax[j].annotate("0" if r == 0 else fmt.format(r), (b.get_x() + b.get_width() / 2, b.get_height()),
                               ha="center", va="bottom", fontsize=8)
        ax[j].set_yscale("log")
        ax[j].set_xticks(x)
        ax[j].set_xticklabels([g for g, _ in groups])
        ax[j].set_title(ttl)
        ax[j].grid(alpha=0.3, axis="y")
        ax[j].legend(fontsize=9)
    fig.suptitle("无损 TC 104 vs 有损 TC 0：丢包重传（两台节点的平均）")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig7_retransmissions.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for fn in (fig_zoom, fig_bins, fig_burst, fig_inject, fig_inject_timeline, fig_breakdown):
        fn(out)
        print("done", fn.__name__)
