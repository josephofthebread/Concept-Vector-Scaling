from __future__ import annotations

import logging
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
import numpy as np
from safetensors import safe_open

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger("anatomy")
REGIMES = ((0, 9), (9, 19), (19, 23), (23, 36))


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--decomp", type=Path, default=Path("work/decomp"))
    parser.add_argument("--out", type=Path, default=Path("work/anatomy"))
    parser.add_argument("--figs", type=Path, default=Path("figures/formation"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    args.figs.mkdir(parents=True, exist_ok=True)
    per: dict[int, dict[int, list]] = {}
    read_layers = None
    for f in sorted(args.decomp.glob("item-*.safetensors")):
        with safe_open(str(f), framework="np") as h:
            ca, cm, ce, tot = (h.get_tensor(x) for x in ("c_attn", "c_mlp", "c_embed", "total"))
            k, side = int(h.get_tensor("pair_id")), int(h.get_tensor("side"))
            if read_layers is None:
                read_layers = [int(x) for x in h.metadata()["read_layers"].split(",")]
        per.setdefault(k, {0: [], 1: []})[side].append((ca, cm, ce, tot))
    assert read_layers is not None
    ks = sorted(k for k in per if per[k][0] and per[k][1])
    A = np.stack([np.mean([x[0] for x in per[k][0]], 0) - np.mean([x[0] for x in per[k][1]], 0) for k in ks])
    M = np.stack([np.mean([x[1] for x in per[k][0]], 0) - np.mean([x[1] for x in per[k][1]], 0) for k in ks])
    E = np.stack([np.mean([x[2] for x in per[k][0]], 0) - np.mean([x[2] for x in per[k][1]], 0) for k in ks])
    T = np.stack([np.mean([x[3] for x in per[k][0]], 0) - np.mean([x[3] for x in per[k][1]], 0) for k in ks])
    K, _, B, H = A.shape
    recon = E + A.sum((2, 3)) + M.sum(2)
    log.info(
        "%d concepts; exactness: median rel.err %.1e (rows %s)",
        K,
        float(np.median(np.abs(recon - T) / np.abs(T))),
        read_layers,
    )
    out = {}
    for i, L in enumerate(read_layers):
        if L < 24:
            continue
        tot = T[:, i]
        sh_e = E[:, i] / tot
        sh_a = np.stack([A[:, i, lo : min(hi, L)].sum((1, 2)) / tot for lo, hi in REGIMES], 1)
        sh_m = np.stack([M[:, i, lo : min(hi, L)].sum(1) / tot for lo, hi in REGIMES], 1)
        pos = (
            np.where(A[:, i] > 0, A[:, i], 0).sum((1, 2))
            + np.where(M[:, i] > 0, M[:, i], 0).sum(1)
            + np.clip(E[:, i], 0, None)
        )
        neg = -(
            np.where(A[:, i] < 0, A[:, i], 0).sum((1, 2))
            + np.where(M[:, i] < 0, M[:, i], 0).sum(1)
            + np.clip(E[:, i], None, 0)
        )
        cancel = neg / pos
        allc = np.concatenate([A[:, i].reshape(K, -1), M[:, i]], 1)
        srt = -np.sort(-allc, 1)
        n_half = (np.cumsum(srt, 1) < 0.5 * tot[:, None]).sum(1) + 1
        heads_half = (np.cumsum(-np.sort(-A[:, i].reshape(K, -1), 1), 1) < 0.5 * A[:, i].sum((1, 2))[:, None]).sum(
            1
        ) + 1
        out[L] = {
            "sh_e": sh_e,
            "sh_a": sh_a,
            "sh_m": sh_m,
            "cancel": cancel,
            "n_half": n_half,
            "heads_half": heads_half,
            "attn_total": A[:, i].sum((1, 2)) / tot,
            "mlp_total": M[:, i].sum(1) / tot,
        }
        log.info(
            "row %d: median shares of |v|: embed %.3f; attention by regime %s (total %.3f); MLP by regime %s (total %.3f); cancelled/positive %.2f; components for half %d (heads alone %d)",
            L,
            np.median(sh_e),
            [round(float(np.median(sh_a[:, r])), 3) for r in range(4)],
            float(np.median(A[:, i].sum((1, 2)) / tot)),
            [round(float(np.median(sh_m[:, r])), 3) for r in range(4)],
            float(np.median(M[:, i].sum(1) / tot)),
            float(np.median(cancel)),
            int(np.median(n_half)),
            int(np.median(heads_half)),
        )
        log.info(
            "row %d: per-block median share, attention: %s",
            L,
            [round(float(np.median(A[:, i, b].sum(1) / tot)), 3) for b in range(min(B, L))],
        )
        log.info(
            "row %d: per-block median share, MLP:       %s",
            L,
            [round(float(np.median(M[:, i, b] / tot)), 3) for b in range(min(B, L))],
        )
    np.savez(str(args.out / "anatomy.npz"), pairs=np.array(ks), read_layers=np.array(read_layers), A=A, M=M, E=E, T=T)
    fig, axes = plt.subplots(1, 3, figsize=(26, 7))
    x = np.arange(B)
    i35 = read_layers.index(35)
    ax = axes[0]
    tot = T[:, i35]
    ax.bar(
        x - 0.2,
        np.median(A[:, i35].sum(2) / tot[:, None], 0),
        width=0.4,
        color="#2563eb",
        label="attention (all heads of the block)",
    )
    ax.bar(x + 0.2, np.median(M[:, i35] / tot[:, None], 0), width=0.4, color="#9ca3af", label="MLP")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvspan(18.5, 22.5, color="#f59e0b", alpha=0.12)
    ax.set_title("share of |v(35)| written by each block's attention and MLP (median over concepts)")
    ax.set_xlabel("block b")
    ax.legend(fontsize=8)
    ax = axes[1]
    labels = [f"attn {lo}-{hi - 1}" for lo, hi in REGIMES] + [f"MLP {lo}-{hi - 1}" for lo, hi in REGIMES] + ["embed"]
    vals = (
        [np.median(out[35]["sh_a"][:, r]) for r in range(4)]
        + [np.median(out[35]["sh_m"][:, r]) for r in range(4)]
        + [np.median(out[35]["sh_e"])]
    )
    cols = ["#93c5fd", "#60a5fa", "#2563eb", "#1e3a8a", "#e5e7eb", "#9ca3af", "#6b7280", "#374151", "#f59e0b"]
    ax.bar(range(len(vals)), vals, color=cols)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("anatomy of v(35): share by component and regime (median over concepts)")
    ax = axes[2]
    ax.hist(np.clip(out[35]["cancel"], 0, 3), bins=40, color="#7c3aed", alpha=0.8, label="cancelled / positive, row 35")
    ax.hist(np.clip(out[24]["cancel"], 0, 3), bins=40, color="#f59e0b", alpha=0.5, label="row 24")
    ax.set_title("how much of the positive writing is cancelled by negative writes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.figs / "A11-anatomy.png", dpi=170)
    log.info("wrote %s", args.figs)


if __name__ == "__main__":
    main()
