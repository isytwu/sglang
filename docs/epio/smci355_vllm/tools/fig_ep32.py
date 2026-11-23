#!/usr/bin/env python3
"""Figure 2 of docs/epio/k31_smci355_vllm_ep_io_zh.md: the same injected IO on EP16 and EP32 decode, and EP32's
response to paced bursts into one decode node and to the bytes per batch.

  python3 tools/fig_ep32.py <out.png>

EP16: e1ra_6866 / e1t1_6866 / e1_6881 (fig 1); EP32: e1m2_7005 (output length 64k, no request finishes during the
run), step change = median of the 10 s gap medians with the IO on against the IO-off stretches around the window
(tools/ep32_win.py).
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = "DejaVu Sans"

# (label, EP16 %, EP32 %); None = not measured
SAME = [
    ("1 GB/s\nper flow", 20.8, 13.4),
    ("2 GB/s", 35.8, 19.1),
    ("5 GB/s", 57.9, 27.0),
    ("10 GB/s", 73.1, 24.9),
    ("full rate,\ncontinuous", 66.0, 10.5),
    ("one decode,\nline rate\nwrite", 44.0, 18.4),
    ("one decode,\nline rate\nread", None, 27.8),
    ("1 MB per\nbatch, full", 0.4, 3.3),
    ("full rate,\nTC 0", 0.6, 0.3),
]
# EP32, 16 MB bursts into one decode node: rate per flow (GB/s) -> %, and continuous
PACED = [("2", 9.3), ("5", 14.4), ("10", 17.3), ("20", 18.7), ("continuous\n(~48)", 18.4)]
# EP32, 10 GB/s per flow into every decode node: bytes per batch -> %
BATCH = [("2 MB", 6.7), ("4 MB", 11.0), ("16 MB", 24.9)]


def main(out):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw={"width_ratios": [2.2, 1.1, 0.8]})
    x = range(len(SAME))
    w = 0.38
    for off, idx, lab, col in ((-w / 2, 1, "EP16 (2 decode nodes, base 302 ms)", "tab:blue"),
                               (w / 2, 2, "EP32 (4 decode nodes, base 528 ms)", "tab:red")):
        xs = [i + off for i, r in zip(x, SAME) if r[idx] is not None]
        vs = [r[idx] for r in SAME if r[idx] is not None]
        b = ax[0].bar(xs, vs, w, label=lab, color=col)
        ax[0].bar_label(b, labels=[f"{v:+.1f}%" if abs(v) < 5 else f"{v:+.0f}%" for v in vs], fontsize=7)
    ax[0].set_xticks(list(x))
    ax[0].set_xticklabels([r[0] for r in SAME], fontsize=7)
    ax[0].set_title("(a) same injected IO (16 MB per batch, into every decode node unless marked)", fontsize=9)
    ax[0].set_ylabel("decode step time vs no IO (%)")
    ax[0].legend(fontsize=8)
    ax[0].set_ylim(-5, 85)

    lab = [p[0] for p in PACED if p[1] is not None]
    v = [p[1] for p in PACED if p[1] is not None]
    b = ax[1].bar(lab, v, color=["tab:red"] * (len(v) - 1) + ["tab:gray"])
    ax[1].bar_label(b, fmt="%+.1f%%", fontsize=7)
    ax[1].set_title("(b) EP32: 16 MB bursts into one decode node\nrate per flow (GB/s)", fontsize=9)
    ax[1].set_ylim(-5, 85)

    lab = [p[0] for p in BATCH if p[1] is not None]
    v = [p[1] for p in BATCH if p[1] is not None]
    b = ax[2].bar(lab, v, color=["tab:blue", "tab:orange", "tab:red"][: len(v)])
    ax[2].bar_label(b, fmt="%+.1f%%", fontsize=7)
    ax[2].set_title("(c) EP32: 10 GB/s per flow\nby bytes per batch", fontsize=9)
    ax[2].set_ylim(-5, 85)
    for a in ax:
        a.axhline(0, color="k", lw=0.6)
        a.grid(axis="y", alpha=0.3)
    fig.suptitle("K3.1 vLLM decode, 128 sequences per DP rank, qlen 8: EP16 vs EP32 (global batch 512), smci355", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "smci355_fig2_ep32.png")
