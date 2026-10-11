from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from exp_attn_interp.config import STEER_BAND_START  # noqa: E402

log = logging.getLogger("figs")
FAMILIES = [
    ("Qwen3-8B", "work", "#2563eb"),
    ("Meta-Llama-3.1-8B-Instruct", "work-Meta-Llama-3.1-8B-Instruct", "#dc2626"),
    ("gemma-2-9b-it", "work-gemma-2-9b-it", "#059669"),
]
BAND_LEN = 4


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.svg")
    fig.savefig(out / f"{name}.png", dpi=150)
    plt.close(fig)
    log.info("wrote %s/%s.{svg,png}", out, name)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if Path("work/decompan.json").exists():
        d = json.loads(Path("work/decompan.json").read_text())
        nb = d["n_blocks"]
        band = d.get("band_start", 19)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        ax = axes[0]
        ax.plot(range(nb), d["formation"]["cum_curve_median"], "-o", ms=3, color="#2563eb", lw=2)
        ax.axvspan(band, band + BAND_LEN - 1, color="#f59e0b", alpha=0.2, label="steering optimum band")
        ax.axhline(1.0, color="gray", ls=":", lw=0.8)
        ax.set_xlabel("block")
        ax.set_ylabel("fraction of v_k assembled (median over concepts)")
        ax.set_title("Formation of the concept direction through the stack")
        ax.legend(fontsize=8)
        ax = axes[1]
        pa, pm = d["per_block_attention_share_median"], d["per_block_mlp_share_median"]
        ax.bar(range(nb), pa, color="#2563eb", label="attention heads")
        ax.bar(range(nb), pm, bottom=pa, color="#9ca3af", label="MLP")
        ax.axvspan(band, band + BAND_LEN - 1, color="#f59e0b", alpha=0.2)
        ax.set_xlabel("block")
        ax.set_ylabel("share of v_k written in this block")
        ax.set_title("Who writes it: attention vs MLP per block")
        ax.legend(fontsize=8)
        fig.tight_layout()
        save(fig, args.out, "fig-formation")
    if Path("work/roles.json").exists():
        r = json.loads(Path("work/roles.json").read_text())
        prof = r["depth_profile"]
        nb = len(next(iter(prof.values())))
        x = np.arange(nb)
        fig, ax = plt.subplots(figsize=(9, 3.6))
        for role, color in (("builder", "#2563eb"), ("detector", "#dc2626"), ("propagator", "#f59e0b")):
            if role in prof:
                ax.plot(x, prof[role], "-o", ms=3, lw=1.8, color=color, label=role)
        ax.set_xlabel("block")
        ax.set_ylabel("heads with role")
        ax.set_title("Concept-role census over depth")
        ax.legend(fontsize=8)
        fig.tight_layout()
        save(fig, args.out, "fig-roles")
    if Path("work/stories_rw/edgemap.json").exists():
        em = json.loads(Path("work/stories_rw/edgemap.json").read_text())
        pb = em["per_block"]
        b0 = STEER_BAND_START["Qwen/Qwen3-8B"]
        fig, ax = plt.subplots(figsize=(9, 3.4))
        ax.bar(range(len(pb)), pb, color="#2563eb")
        ax.axvspan(b0, b0 + BAND_LEN - 1, color="#f59e0b", alpha=0.2, label="steering optimum band")
        ax.set_yscale("symlog")
        ax.set_xlabel("block")
        ax.set_ylabel("concepts with a marker-local head")
        ax.set_title("Where concept vectors couple to heads vs where steering acts")
        ax.legend(fontsize=8)
        fig.tight_layout()
        save(fig, args.out, "fig-locus")
    fams = [(n, Path(w) / "decompan.json", c) for n, w, c in FAMILIES if (Path(w) / "decompan.json").exists()]
    if len(fams) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for name, p, color in fams:
            dd = json.loads(p.read_text())
            x = np.arange(dd["n_blocks"]) / dd["n_blocks"]
            axes[0].plot(x, dd["formation"]["cum_curve_median"], "-", lw=2, color=color, label=name)
            axes[1].plot(x, dd["per_block_attention_share_median"], "-", lw=1.5, color=color, label=name)
        axes[0].set_xlabel("relative depth")
        axes[0].set_ylabel("fraction of v_k assembled")
        axes[0].set_title("Formation curve per family")
        axes[0].legend(fontsize=8)
        axes[1].set_xlabel("relative depth")
        axes[1].set_ylabel("attention share per block")
        axes[1].set_title("Attention-written share per family")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        save(fig, args.out, "fig-models")
    fams = [(n, Path(w) / "rotan.json", c) for n, w, c in FAMILIES if (Path(w) / "rotan.json").exists()]
    if fams:
        dose = (
            Path("work/steerdose2/summary.json")
            if Path("work/steerdose2/summary.json").exists()
            else Path("work/steerdose/summary.json")
        )
        ncol = 2 + (1 if dose.exists() else 0)
        fig, axes = plt.subplots(1, ncol, figsize=(5.2 * ncol, 4))
        ax = axes[0]
        x = np.arange(len(fams))
        for i, (name, pth, color) in enumerate(fams):
            d = json.loads(pth.read_text())
            ax.bar(i - 0.2, d["v_mid"]["attn_share_median"], 0.4, color=color, label="attention" if i == 0 else None)
            ax.bar(
                i - 0.2,
                d["v_mid"]["mlp_share_median"],
                0.4,
                bottom=d["v_mid"]["attn_share_median"],
                color="#9ca3af",
                label="MLP" if i == 0 else None,
            )
            ax.bar(i + 0.2, d["u"]["attn_share_median"], 0.4, color=color, hatch="//")
            ax.bar(
                i + 0.2,
                d["u"]["mlp_share_median"],
                0.4,
                bottom=d["u"]["attn_share_median"],
                color="#9ca3af",
                hatch="//",
            )
        ax.set_xticks(x)
        ax.set_xticklabels([n.split("-")[0] for n, _, _ in fams])
        ax.set_ylabel("share written")
        ax.set_title("Preserved v(mid) (solid) vs new component u (hatched)")
        ax.legend(fontsize=8)
        ax = axes[1]
        for name, pth, color in fams:
            d = json.loads(pth.read_text())
            nb = d["n_blocks"]
            ax.plot(np.arange(nb) / nb, d["v_mid"]["cum_median"], "-", color=color, lw=2, label=f"{name}: preserved")
            ax.plot(np.arange(nb) / nb, d["u"]["cum_median"], "--", color=color, lw=2, label=f"{name}: new")
        ax.axhline(0.5, color="gray", ls=":", lw=0.8)
        ax.set_xlabel("relative depth")
        ax.set_ylabel("cumulative share")
        ax.set_title("Where each component is written")
        ax.legend(fontsize=6)
        if dose.exists():
            ax = axes[2]
            dd = json.loads(dose.read_text())["dose"]["marker_rate"]["k"]
            ks = sorted(int(k) for k in dd)
            ax.plot(
                ks, [100 * dd[str(k)]["top_kept"] for k in ks], "-o", color="#2563eb", label="top-k builders ablated"
            )
            ax.plot(
                ks, [100 * dd[str(k)]["rand_kept"] for k in ks], "-o", color="#9ca3af", label="k random heads ablated"
            )
            ax.set_xscale("log")
            ax.set_xlabel("k heads zeroed (of 448 after the injection row)")
            ax.set_ylabel("% of steering effect kept (marker rate)")
            ax.set_title("Steering rides on the ranked builders")
            ax.legend(fontsize=8)
        fig.tight_layout()
        save(fig, args.out, "fig-rotation-dose")
    runs = [
        ("Qwen3-8B", Path("work/gentraj_rows2"), "#2563eb"),
        ("Meta-Llama-3.1-8B-Instruct", Path("work/gentraj_rows_llama"), "#dc2626"),
        ("gemma-2-9b-it", Path("work/gentraj_rows_gemma"), "#059669"),
    ]
    runs = [r for r in runs if (r[1] / "gentraj_rows.json").exists()]
    if not runs:
        return
    fig, axes = plt.subplots(1, len(runs) + 1, figsize=(4.6 * (len(runs) + 1), 3.8))
    for ax, (name, d, color) in zip(axes[:-1], runs):
        meta = json.loads((d / "gentraj_rows.json").read_text())
        nb = {"Qwen3-8B": 36, "Meta-Llama-3.1-8B-Instruct": 32, "gemma-2-9b-it": 42}[name]
        for i, L in enumerate(meta["rows"]):
            eff = np.load(d / f"effect_L{L}.npy")
            c = np.nanmedian(eff, 0)
            ax.plot(
                np.arange(len(c)),
                c / max(c[0], 1e-6),
                lw=1.8,
                color=plt.cm.viridis(i / max(1, len(meta["rows"]) - 1)),
                label=f"row {L} ({L / nb:.2f} depth)",
            )
        ax.axhline(1.0, color="gray", ls=":", lw=0.8)
        ax.set_xlabel("generated token")
        ax.set_ylabel("effect / first-token effect")
        ax.set_title(f"{name.split('-')[0]}: compounding by injection row")
        ax.legend(fontsize=7)
    ax = axes[-1]
    for name, d, color in runs:
        meta = json.loads((d / "gentraj_rows.json").read_text())
        nb = {"Qwen3-8B": 36, "Meta-Llama-3.1-8B-Instruct": 32, "gemma-2-9b-it": 42}[name]
        rows = meta["rows"]
        ax.plot(
            [L / nb for L in rows],
            [meta["per_row"][str(L)]["rho_effect_vs_judged"] for L in rows],
            "-o",
            color=color,
            label=f"{name}: vs judge",
        )
        ax.plot(
            [L / nb for L in rows],
            [meta["per_row"][str(L)]["rho_effect_vs_align_L_last"] for L in rows],
            "--s",
            color=color,
            alpha=0.6,
            label=f"{name}: vs align(L,last)",
        )
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("injection row / depth")
    ax.set_ylabel("Spearman with per-concept effect")
    ax.set_title("What the projection effect tracks")
    ax.legend(fontsize=6)
    fig.tight_layout()
    save(fig, args.out, "fig-compound")


if __name__ == "__main__":
    main()
