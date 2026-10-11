from __future__ import annotations

import collections
import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
import numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from exp_attn_interp.config import INSTRUCT_MODEL  # noqa: E402

log = logging.getLogger("attnsrcan")
CATS = ("template", "marker_own", "marker_other", "self", "delim", "story_other")
REGIMES = ((0, 9), (9, 19), (19, 23), (23, 36))


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    parser.add_argument("--attnsrc", type=Path, default=Path("work/attnsrc"))
    parser.add_argument("--out", type=Path, default=Path("work/attnsrcan"))
    parser.add_argument("--figs", type=Path, default=Path("figures/attnsrc"))
    parser.add_argument("--band", default="19,22")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    args.figs.mkdir(parents=True, exist_ok=True)
    b0, b1 = (int(x) for x in args.band.split(","))
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    per: dict[int, dict[int, list]] = collections.defaultdict(lambda: {0: [], 1: []})
    tokc: list[dict[int, dict[int, float]]] = [
        collections.defaultdict(lambda: collections.defaultdict(float)) for _ in REGIMES
    ]
    for f in sorted(args.attnsrc.glob("item-*.safetensors")):
        with safe_open(str(f), framework="np") as h:
            c = h.get_tensor("contrib")
            m = h.get_tensor("mass")
            src = h.get_tensor("src").astype(np.float32)
            ids = h.get_tensor("ids")
            k = int(h.get_tensor("pair_id"))
            side = int(h.get_tensor("side"))
        per[k][side].append((c, m))
        sign = 1.0 if side == 0 else -1.0
        for r, (lo, hi) in enumerate(REGIMES):
            s = src[1, lo:hi].sum(0)
            for t, val in zip(ids.tolist(), s.tolist()):
                tokc[r][k][t] += sign * val
    ks = sorted(k for k in per if per[k][0] and per[k][1])
    D = np.stack([np.mean([c for c, _ in per[k][0]], 0) - np.mean([c for c, _ in per[k][1]], 0) for k in ks])
    Mass = np.stack([np.mean([m for _, m in per[k][0] + per[k][1]], 0) for k in ks])
    K, _, B, H, C = D.shape
    blk = D.sum(3)
    tot = blk.sum(-1)
    np.savez(str(args.out / "sources.npz"), pairs=np.array(ks), D=D, mass=Mass)
    for i, nm in enumerate(("v_k(b)", "v_k(35)")):
        log.info(
            "diff-write along %s by block, total: %s",
            nm,
            [round(float(np.median(tot[:, i, b])), 3) for b in range(0, B, 2)],
        )
        for c, cat in enumerate(CATS):
            log.info("  %-13s %s", cat, [round(float(np.median(blk[:, i, b, c])), 3) for b in range(0, B, 2)])
    for c, cat in enumerate(CATS):
        log.info(
            "attention mass on %-13s by block: %s",
            cat,
            [round(float(np.median(Mass[:, b, :, c].mean(1))), 3) for b in range(0, B, 3)],
        )
    words = {}
    for r, (lo, hi) in enumerate(REGIMES):
        agg: dict[int, float] = collections.defaultdict(float)
        cnt: dict[int, int] = collections.defaultdict(int)
        for k in ks:
            for t, v in tokc[r][k].items():
                agg[t] += v
                cnt[t] += 1
        top = sorted(agg.items(), key=lambda kv: -kv[1])[:25]
        bot = sorted(agg.items(), key=lambda kv: kv[1])[:15]
        words[f"{lo}-{hi - 1}"] = {
            "top": [[tok.decode([t]), round(v, 2), cnt[t]] for t, v in top],
            "bottom": [[tok.decode([t]), round(v, 2), cnt[t]] for t, v in bot],
        }
        log.info(
            "blocks %d-%d, source tokens carrying the most contrast (summed over concepts): %s",
            lo,
            hi - 1,
            [(tok.decode([t]), round(v, 1)) for t, v in top[:15]],
        )
    (args.out / "source_words.json").write_text(json.dumps(words, ensure_ascii=False, indent=1))
    fig, axes = plt.subplots(2, 3, figsize=(26, 12))
    x = np.arange(B)
    cols = ("#9ca3af", "#dc2626", "#2563eb", "#111827", "#f59e0b", "#7c3aed")
    for i, nm in enumerate(("v_k(b)", "v_k(35)")):
        ax = axes[0, i]
        base = np.zeros(B)
        med = np.stack([np.median(blk[:, i, :, c], 0) for c in range(C)])
        for c in range(C):
            ax.plot(x, med[c], color=cols[c], lw=2, label=CATS[c])
        ax.plot(x, np.median(tot[:, i], 0), color="k", lw=1, ls="--", label="total attention diff-write")
        ax.axhline(0, color="k", lw=0.5)
        ax.axvspan(b0, b1, color="#f59e0b", alpha=0.12)
        ax.set_title(f"attention diff-write along {nm} by source category (median over concepts, sum over heads)")
        ax.set_xlabel("block b")
        ax.legend(fontsize=8)
        _ = base
    ax = axes[0, 2]
    for c in range(C):
        ax.plot(x, np.median(Mass[:, :, :, c].mean(2), 0), color=cols[c], lw=2, label=CATS[c])
    ax.set_title("attention mass by source category (mean over heads, median over concepts)")
    ax.set_xlabel("block b")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    share = blk[:, 1, :, 1] / np.abs(tot[:, 1]).clip(1e-6)
    im = ax.imshow(
        np.clip(share, -1.5, 1.5), aspect="auto", cmap="RdBu_r", vmin=-1.5, vmax=1.5, interpolation="nearest"
    )
    ax.set_title("share of the diff-write along v_k(35) from own-pole marker sources, per concept")
    ax.set_xlabel("block b")
    ax.set_ylabel("concept")
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax = axes[1, 1]
    hm = np.median(D[:, 1, :, :, 1], 0).T
    im = ax.imshow(
        hm, aspect="auto", cmap="RdBu_r", vmin=-np.abs(hm).max(), vmax=np.abs(hm).max(), interpolation="nearest"
    )
    ax.set_title("per-head diff-write along v_k(35) from own-pole markers (median over concepts)")
    ax.set_xlabel("block b")
    ax.set_ylabel("head")
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax = axes[1, 2]
    hm2 = np.median(D[:, 1, :, :, 0], 0).T
    im = ax.imshow(
        hm2, aspect="auto", cmap="RdBu_r", vmin=-np.abs(hm2).max(), vmax=np.abs(hm2).max(), interpolation="nearest"
    )
    ax.set_title("per-head diff-write along v_k(35) from template sources (median over concepts)")
    ax.set_xlabel("block b")
    ax.set_ylabel("head")
    plt.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(args.figs / "I1-sources.png", dpi=170)
    plt.close(fig)
    log.info("wrote %s", args.figs)


if __name__ == "__main__":
    main()
