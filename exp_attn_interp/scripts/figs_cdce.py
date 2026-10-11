from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(".")
BLUE, RED, GREY, GREEN, ORANGE, PURPLE = "#2563eb", "#dc2626", "#6b7280", "#059669", "#f59e0b", "#7c3aed"


if __name__ == "__main__":
    d = json.loads((ROOT / "work/tmp/topiclane.json").read_text())
    g, s, e, L = d["gain_current"], d["share_final22"], d["erasure_rows1to35"], d["lens"]
    B = 36
    x = np.arange(B)
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    ax = axes[0, 0]
    for key, col, lab in (
        ("mlp_topic", RED, "topic lane (pair mean − population)"),
        ("mlp_lean", BLUE, "lean lane (concept − antagonist)"),
    ):
        p50, p10, p90 = (np.array(g[key][k]) for k in ("p50", "p10", "p90"))
        ax.plot(x, p50, "-o", ms=3, color=col, label=lab)
        ax.fill_between(x, p10, p90, color=col, alpha=0.15)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvspan(18.5, 22.5, color=ORANGE, alpha=0.2, label="steering band 19–22")
    ax.set_ylim(-0.2, 0.15)
    ax.set_title("MLP write along the lane's own direction, per block (median, p10–p90 over 1,036 pairs)")
    ax.set_xlabel("block")
    ax.set_ylabel("fraction of the lane's content added (+) or removed (−)")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    for key, col, lab in (("attn_topic", RED, "attention, topic lane"), ("attn_lean", BLUE, "attention, lean lane")):
        ax.plot(x, np.array(s[key]["p50"]), "-o", ms=3, color=col, label=lab)
    for key, col, lab in (("mlp_topic", RED, "MLP, topic lane"), ("mlp_lean", BLUE, "MLP, lean lane")):
        ax.plot(x, np.array(s[key]["p50"]), "--", color=col, label=lab)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(19, color=GREY, ls=":", lw=1.2)
    ax.annotate(
        "block 19: the stroke is lean-only\n(0.176 vs 0.105)",
        xy=(19, 0.176),
        xytext=(8, 0.25),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color=GREY),
    )
    ax.set_title("share of the lane's row-22 direction written per block (median)")
    ax.set_xlabel("block")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    rows = np.arange(1, 36)
    for key, col, lab in (("topic_cos", RED, "topic: cos(ΣA, ΣM)"), ("lean_cos", BLUE, "lean: cos(ΣA, ΣM)")):
        ax.plot(rows, [e[key][i]["p50"] for i in range(35)], "-", color=col, label=lab)
    for key, col, lab in (("topic_surv", RED, "topic: |ΣA+ΣM|/|ΣA|"), ("lean_surv", BLUE, "lean: |ΣA+ΣM|/|ΣA|")):
        ax.plot(rows, [e[key][i]["p50"] for i in range(35)], "--", color=col, label=lab)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvspan(18.5, 22.5, color=ORANGE, alpha=0.2)
    ax.set_title("cancellation between the cumulative attention and MLP writes, by row (whole lane)")
    ax.set_xlabel("row")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    lr = L["rows"]
    ax.plot(
        lr,
        [v["p50"] for v in L["topic_out_guidance_pct"]],
        "-o",
        ms=3,
        color=RED,
        label="topic direction t(r) → pair's own situation words",
    )
    ax.plot(
        lr,
        [v["p50"] for v in L["lean_out_guidance_pct"]],
        "-o",
        ms=3,
        color=BLUE,
        label="lean v(r) → situation words (control)",
    )
    ax.plot(
        lr,
        [v["p50"] for v in L["lean_out_markergap_pct"]],
        "-s",
        ms=3,
        color=BLUE,
        alpha=0.5,
        label="lean v(r) → own pole words",
    )
    ax.plot(
        lr[1:],
        [v["p50"] for v in L["topic_negmlp_cum_pct"][1:]],
        "--",
        color=RED,
        label="−(cumulative MLP topic write) → situation words",
    )
    ax.plot(
        lr[1:],
        [v["p50"] for v in L["topic_mlp_cum_pct"][1:]],
        ":",
        color=RED,
        label="cumulative MLP topic write → situation words",
    )
    ax.plot(
        lr[1:],
        [v["p50"] for v in L["lean_negmlp_cum_pct"][1:]],
        "--",
        color=BLUE,
        alpha=0.6,
        label="−(cumulative MLP lean write) → pole words",
    )
    ax.axvspan(18.5, 22.5, color=ORANGE, alpha=0.2)
    ax.set_ylim(0, 1.02)
    ax.set_title("logit lens, output form: rank of the pair's own word set among 1,036 pairs (median)")
    ax.set_xlabel("row")
    ax.set_ylabel("percentile")
    ax.legend(fontsize=7, loc="center right")
    fig.suptitle(
        "§CD — the two lanes of the same decomposition: the pair's shared situation is stripped like the pole contrast",
        fontsize=13,
    )
    fig.tight_layout()
    (ROOT / "figures/lanes").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "figures/lanes/L1-lanes.png", dpi=160)
    d = json.loads((ROOT / "work/tmp/tokan.json").read_text())
    w = d["window"]["pol_u"]
    R = 37
    rows = np.arange(R)
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    ax = axes[0, 0]
    ax.plot(rows, w["corr_at_0"], "-o", ms=3, color="k", label="the word under the pen (lag 0)")
    ax.plot(rows, w["corr_sum_past"], "-o", ms=3, color=BLUE, label="the past (Σ corr, lags −32…−1)")
    ax.plot(rows, w["corr_sum_future"], "-o", ms=3, color=RED, label="the future (Σ corr, lags +1…+32)")
    ax.plot(rows, w["corr_at_m1"], "--", color=BLUE, alpha=0.7, label="previous token (lag −1)")
    ax.plot(rows, w["corr_at_p1"], "--", color=RED, alpha=0.7, label="next token (lag +1)")
    ax.axvspan(18.5, 22.5, color=ORANGE, alpha=0.2, label="band 19–22")
    ax.axvline(22, color=GREY, ls=":", lw=1)
    ax.set_title("the token's concept coordinate: correlation with word polarity at lag h, by row (3.3 M tokens)")
    ax.set_xlabel("row")
    ax.set_ylabel("pooled within-story correlation")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    S = np.array(d["classes_snr_median"])
    cls = d["classes"]
    for ci, col, lab in ((0, PURPLE, "punctuation"), (1, GREEN, "function words"), (2, BLUE, "content words")):
        ax.plot(rows, S[ci], "-o", ms=3, color=col, label=lab)
    ax.plot(rows, d["snr_all_median"], "-", color="k", alpha=0.5, label="all tokens")
    ax.axvline(7, color=GREY, ls=":", lw=1)
    ax.axvspan(18.5, 22.5, color=ORANGE, alpha=0.2)
    ax.set_title("token SNR by token class (median over 1,036 pairs): punctuation > function > content from block 7")
    ax.set_xlabel("row")
    ax.set_ylabel("pole gap / within-story token spread")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    A = np.array(d["latency"]["auroc_raw_mean"])
    P = d["prefix"]
    for pi, col in zip((3, 4, 5, 6), (GREY, BLUE, RED, GREEN)):
        ax.plot(rows, A[pi], "-o", ms=3, color=col, label=f"first {P[pi]} tokens")
    ax.plot(rows, A[-1], "-", color="k", alpha=0.5, label="whole story")
    ax.axhline(0.95, color="k", lw=0.6, ls="--")
    ax.axvspan(18.5, 22.5, color=ORANGE, alpha=0.2)
    ax.set_ylim(0.6, 1.0)
    ax.set_title("detection latency: mean 8×8 story AUROC of the prefix mean, by row")
    ax.set_xlabel("row")
    ax.set_ylabel("AUROC")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    ac = np.array(d["autocorr"])
    for l, col in ((0, "k"), (1, BLUE), (3, RED), (7, GREEN)):
        ax.plot(rows, ac[:, l], "-o", ms=3, color=col, label=f"lag {l + 1}")
    ax.axvspan(18.5, 22.5, color=ORANGE, alpha=0.2)
    ax.axvline(23.5, color=GREY, ls=":", lw=1)
    ax.set_title("within-story autocorrelation of the concept coordinate, by row (peak 23–24)")
    ax.set_xlabel("row")
    ax.legend(fontsize=8)
    fig.suptitle("§CE — the token's clock, the token's class, and how soon a monitor can decide", fontsize=13)
    fig.tight_layout()
    (ROOT / "figures/token").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "figures/token/T1-token.png", dpi=160)
    d = json.loads((ROOT / "work/tmp/xdecomp_summary.json").read_text())
    order = [
        "Qwen1.5-0.5B",
        "Qwen2.5-3B-Instruct",
        "Qwen3-4B",
        "Qwen2.5-7B",
        "deepseek-llm-7b-chat",
        "Llama-3.1-8B-Instruct",
        "Qwen3-8B",
        "gemma-2-9b-it",
    ]
    names = [n for n in order if n in d] + [n for n in d if n not in order]
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    for ax, name in zip(axes.flat, names):
        r = d[name]
        B = r["n_blocks"]
        m = np.array(r["mlp_share_last_median"])
        snr = np.array(r["story_snr_median"])
        bstar = r["crossing_last"]["first_positive"]
        ax.bar(np.arange(B), m, color=GREY, label="MLP write along v(last), per block (median share)")
        ax.axhline(0, color="k", lw=0.6)
        if bstar is not None:
            ax.axvline(bstar - 0.5, color="k", ls=":", lw=1.5, label=f"b* = {bstar} ({bstar / B:.2f} of depth)")
        ax2 = ax.twinx()
        ax2.plot(
            np.arange(B + 1), snr, "-o", ms=3, color=RED, label=f"held-out story SNR (max at row {r['snr_argmax_row']})"
        )
        ax2.set_ylabel("story SNR", color=RED)
        ax.set_title(f"{name}: {B} blocks, {r['n_pairs']} pairs")
        ax.set_xlabel("block / row")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    fig.suptitle(
        "§CF — the MLP sign crossing and the probe optimum in eight checkpoints of five families (one code path)",
        fontsize=13,
    )
    fig.tight_layout()
    (ROOT / "figures/xmodel").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "figures/xmodel/X1-crossing.png", dpi=150)
    print("figures written")
