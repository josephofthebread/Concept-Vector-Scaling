from __future__ import annotations

import collections
import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
import numpy as np
from safetensors import safe_open

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger("gainan")


def crossing_first_positive(c: np.ndarray, thr: float = 0.01) -> int:
    for b in range(len(c) - 2):
        if c[b] > thr and c[b + 1] > thr and c[b + 2] > thr:
            return b
    return -1


def band(ax, M, color, label, xs=None, ls="-"):
    xs = np.arange(M.shape[1]) if xs is None else xs
    ax.plot(xs, np.median(M, 0), color=color, lw=2, label=label, ls=ls)
    ax.fill_between(xs, np.percentile(M, 25, 0), np.percentile(M, 75, 0), color=color, alpha=0.15)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--mlpgain", type=Path, default=Path("work/mlpgain"))
    parser.add_argument("--out", type=Path, default=Path("work/gainan"))
    parser.add_argument("--figs", type=Path, default=Path("figures/gain"))
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--band", default="19,22")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    args.figs.mkdir(parents=True, exist_ok=True)
    b0, b1 = (int(x) for x in args.band.split(","))
    files = sorted(args.mlpgain.glob("pair-*.safetensors"))
    G, P, QU, QG, pairs = [], [], [], [], []
    top1, top10, half, negfrac = [], [], [], []
    top_neg: list[collections.Counter] = []
    top_pos: list[collections.Counter] = []
    for i, f in enumerate(files):
        with safe_open(str(f), framework="np") as h:
            g = h.get_tensor("gain")
            p = h.get_tensor("path")
            qu, qg = h.get_tensor("q_up"), h.get_tensor("q_gate")
            neur = h.get_tensor("neuron").astype(np.float32)
        B, nI = neur.shape[1], neur.shape[2]
        if not top_neg:
            top_neg = [collections.Counter() for _ in range(B)]
            top_pos = [collections.Counter() for _ in range(B)]
        G.append(g)
        P.append(p)
        QU.append(qu)
        QG.append(qg)
        pairs.append(int(f.stem.split("-")[1]))
        ns = neur[0]
        tot = ns.sum(1)
        a = np.abs(ns)
        srt = -np.sort(-a, axis=1)
        csum = np.cumsum(srt, 1)
        top1.append(csum[:, max(1, nI // 100) - 1] / a.sum(1).clip(1e-9))
        top10.append(csum[:, nI // 10 - 1] / a.sum(1).clip(1e-9))
        half.append((csum < 0.5 * a.sum(1, keepdims=True)).sum(1) + 1)
        negfrac.append(np.where(ns < 0, -ns, 0).sum(1) / a.sum(1).clip(1e-9))
        for b in range(B):
            order = np.argsort(ns[b])
            top_neg[b].update(order[: args.topk].tolist())
            top_pos[b].update(order[-args.topk :].tolist())
        if i % 50 == 0:
            log.info(
                "%d/%d: gain along v_k(b) (linear, stories) at blocks 6/12/18/22/28: %s; total from neurons %s",
                i,
                len(files),
                [round(float(g[:2, b, 0, :2].mean()), 3) for b in (6, 12, 18, 22, 28)],
                [round(float(tot[b]), 3) for b in (6, 12, 18, 22, 28)],
            )
    Ga = np.stack(G)
    Pa = np.stack(P)
    QUa, QGa = np.stack(QU), np.stack(QG)
    T1, T10, HF, NF = (np.stack(x) for x in (top1, top10, half, negfrac))
    K, _, B, _, _ = Ga.shape
    lin = Ga[..., 0]
    mid = Ga[..., 1]
    big = Ga[..., 2]
    st = lin[:, :2].mean(1)
    ne = lin[:, 2]
    np.savez(
        str(args.out / "gainan.npz"),
        pairs=np.array(pairs),
        lin=lin,
        mid=mid,
        big=big,
        path=Pa,
        q_up=QUa,
        q_gate=QGa,
        top1=T1,
        top10=T10,
        half=HF,
        negfrac=NF,
    )
    names = (
        "v_k(b)",
        "v_k(35)",
        "random",
        "v_other(b)",
        "e_gap (input form)",
        "u_gap (output form)",
        "mean residual",
        "shared mode",
        "concept-specific part",
    )[: Ga.shape[3]]
    for j, nm in enumerate(names):
        log.info(
            "median linear gain along %s, stories, by block: %s",
            nm,
            [round(float(np.median(st[:, b, j])), 3) for b in range(0, B, 2)],
        )
    log.info(
        "median linear gain along v_k(b), NEUTRAL text: %s",
        [round(float(np.median(ne[:, b, 0])), 3) for b in range(0, B, 2)],
    )
    log.info(
        "median gain at the injection size (eps 0.25) along v_k(b), stories: %s",
        [round(float(np.median(mid[:, :2, b, 0].mean(1))), 3) for b in range(0, B, 2)],
    )
    log.info(
        "median full-norm (eps 1) gain along v_k(b), stories: %s",
        [round(float(np.median(big[:, :2, b, 0].mean(1))), 3) for b in range(0, B, 2)],
    )
    pu, pg = Pa[:, :2, :, 0, 0].mean(1), Pa[:, :2, :, 0, 1].mean(1)
    log.info("up-path median: %s", [round(float(np.median(pu[:, b])), 3) for b in range(0, B, 2)])
    log.info("gate-path median: %s", [round(float(np.median(pg[:, b])), 3) for b in range(0, B, 2)])
    log.info("weight-only Q_up along v_k(b): %s", [round(float(np.median(QUa[:, b, 0])), 3) for b in range(0, B, 2)])
    log.info("weight-only Q_up along random: %s", [round(float(np.median(QUa[:, b, 2])), 3) for b in range(0, B, 2)])
    log.info("weight-only Q_gate along v_k(b): %s", [round(float(np.median(QGa[:, b, 0])), 3) for b in range(0, B, 2)])
    log.info(
        "neurons: top-1%% share %s; neurons for half %s; negative share %s",
        [round(float(np.median(T1[:, b])), 2) for b in range(0, B, 3)],
        [int(np.median(HF[:, b])) for b in range(0, B, 3)],
        [round(float(np.median(NF[:, b])), 2) for b in range(0, B, 3)],
    )
    share_neg = np.array([max(top_neg[b].values()) / K if top_neg[b] else 0 for b in range(B)])
    share_pos = np.array([max(top_pos[b].values()) / K if top_pos[b] else 0 for b in range(B)])
    n_generic_neg = np.array([sum(1 for _, c in top_neg[b].items() if c >= 0.25 * K) for b in range(B)])
    n_generic_pos = np.array([sum(1 for _, c in top_pos[b].items() if c >= 0.25 * K) for b in range(B)])
    log.info("most shared negative neuron (fraction of concepts) by block: %s", [round(float(x), 2) for x in share_neg])
    log.info("most shared positive neuron by block: %s", [round(float(x), 2) for x in share_pos])
    log.info("neurons in >=25%% of concepts' top-%d negative: %s", args.topk, n_generic_neg.tolist())
    log.info("neurons in >=25%% of concepts' top-%d positive: %s", args.topk, n_generic_pos.tolist())
    out = {
        "K": K,
        "topk": args.topk,
        "neg": {str(b): [[int(n), c] for n, c in top_neg[b].most_common(12)] for b in range(B)},
        "pos": {str(b): [[int(n), c] for n, c in top_pos[b].most_common(12)] for b in range(B)},
    }
    (args.out / "top_neurons.json").write_text(json.dumps(out))

    fig, axes = plt.subplots(2, 3, figsize=(26, 13))
    ax = axes[0, 0]
    for j, (nm, col) in enumerate(
        zip(names, ("#111827", "#7c3aed", "#9ca3af", "#2563eb", "#16a34a", "#dc2626", "#f59e0b", "#0891b2", "#be185d"))
    ):
        band(ax, st[:, :, j], col, f"stories, along {nm}", ls="-" if j < 4 else ":")
    band(ax, ne[:, :, 0], "#dc2626", "neutral text, along v_k(b)", ls="--")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvspan(b0, b1, color="#f59e0b", alpha=0.12)
    ax.set_title("MLP gain along a direction (linear response per unit push, median and IQR over concepts)")
    ax.set_xlabel("block b")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    band(ax, st[:, :, 0], "#111827", "linear (eps 0.1)")
    band(ax, mid[:, :2, :, 0].mean(1), "#2563eb", "the paper's injection size (eps 0.25 x median norm)")
    band(ax, big[:, :2, :, 0].mean(1), "#f59e0b", "full-norm push (eps 1.0)")
    band(ax, big[:, 2, :, 0], "#dc2626", "steering-sized, neutral text", ls="--")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvspan(b0, b1, color="#f59e0b", alpha=0.12)
    ax.set_title("gain along v_k(b): small vs steering-sized push")
    ax.set_xlabel("block b")
    ax.legend(fontsize=8)
    ax = axes[0, 2]
    order = np.argsort(np.argmax(np.cumsum(st[:, :, 0], 1) > 0, 1))
    im = ax.imshow(st[order][:, :, 0], aspect="auto", cmap="RdBu_r", vmin=-0.3, vmax=0.3, interpolation="nearest")
    ax.set_title("linear gain along v_k(b), per concept (stories)")
    ax.set_xlabel("block b")
    ax.set_ylabel("concept")
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax = axes[1, 0]
    band(ax, pu, "#2563eb", "up path: silu(g) * <W_up, d~> <W_down, d>")
    band(ax, pg, "#dc2626", "gate path: silu'(g) u * <W_gate, d~> <W_down, d>")
    band(ax, pu + pg, "#111827", "sum")
    band(ax, st[:, :, 0], "#9ca3af", "finite-difference gain", ls="--")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvspan(b0, b1, color="#f59e0b", alpha=0.12)
    ax.set_title("linear gain split into the two MLP paths (along v_k(b), stories)")
    ax.set_xlabel("block b")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    for j, (nm, col) in enumerate(zip(names[:4], ("#111827", "#7c3aed", "#9ca3af", "#2563eb"))):
        band(ax, QUa[:, :, j], col, f"Q_up along {nm}")
    band(ax, QGa[:, :, 0], "#dc2626", "Q_gate along v_k(b)", ls="--")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvspan(b0, b1, color="#f59e0b", alpha=0.12)
    ax.set_title("weight-only forms sum_n <W_up[n], gamma d> <W_down[:,n], d> (no activations)")
    ax.set_xlabel("block b")
    ax.legend(fontsize=8)
    ax = axes[1, 2]
    band(ax, T1, "#111827", "share of |gain| in top 1 % neurons")
    band(ax, T10, "#2563eb", "share in top 10 %")
    band(ax, NF, "#dc2626", "share of |gain| that is negative")
    ax.plot(np.arange(B), share_neg, color="#7c3aed", lw=2, label="most shared top-neg neuron (fraction of concepts)")
    ax.plot(np.arange(B), share_pos, color="#f59e0b", lw=2, label="most shared top-pos neuron")
    ax.axvspan(b0, b1, color="#f59e0b", alpha=0.12)
    ax.set_ylim(0, 1)
    ax.set_title("neuron-level structure of the gain along v_k(b)")
    ax.set_xlabel("block b")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.figs / "G1-gain.png", dpi=170)
    plt.close(fig)
    nd = Pa.shape[3]
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    cols = ("#111827", "#7c3aed", "#9ca3af", "#2563eb", "#16a34a", "#dc2626", "#f59e0b", "#0891b2", "#be185d")
    for j in range(nd):
        band(axes[0], Pa[:, :2, :, j, 0].mean(1), cols[j], names[j])
        band(axes[1], Pa[:, :2, :, j, 1].mean(1), cols[j], names[j])
        gate_on = np.array([crossing_first_positive(np.median(Pa[:, :2, :, j, 1].mean(1), 0)) for _ in (0,)])[0]
        log.info(
            "gate path along %-22s: median by block %s; first block with gate path > 0.01 and staying: %s",
            names[j],
            [round(float(np.median(Pa[:, :2, b, j, 1].mean(1))), 3) for b in range(0, B, 2)],
            gate_on,
        )
    for ax, ttl in zip(
        axes[:2],
        (
            "up path silu(g) <W_up, d~> <W_down, d> (the linear part through every neuron)",
            "gate path silu'(g) u <W_gate, d~> <W_down, d> (neurons whose gate the direction opens)",
        ),
    ):
        ax.axhline(0, color="k", lw=0.5)
        ax.axvspan(b0, b1, color="#f59e0b", alpha=0.12)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("block b")
        ax.legend(fontsize=7)
    ax = axes[2]
    for j in (0, 2, 5, 7, 8):
        if j < nd:
            band(ax, st[:, :, j], cols[j], names[j])
    ax.axhline(0, color="k", lw=0.5)
    ax.axvspan(b0, b1, color="#f59e0b", alpha=0.12)
    ax.set_title("total linear gain (stories)")
    ax.set_xlabel("block b")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.figs / "G3-paths.png", dpi=170)
    plt.close(fig)
    log.info("wrote %s", args.figs)


if __name__ == "__main__":
    main()
