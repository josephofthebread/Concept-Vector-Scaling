from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger("crossing")
MODELS = [
    ("Qwen3-8B", "work", (19, 22), 22, [22, 20], "#2563eb"),
    ("Qwen3-8B-Base", "work-Qwen3-8B-Base", None, None, [], "#60a5fa"),
    ("Llama-3.1-8B", "work-Meta-Llama-3.1-8B-Instruct", (13, 17), 14, [15, 15], "#dc2626"),
    ("gemma-2-9b", "work-gemma-2-9b-it", (21, 24), 22, [22, 24], "#059669"),
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fig, axes = plt.subplots(1, 4, figsize=(28, 6))
    out = {}
    for ax, (name, w, band, best, probe, color) in zip(axes, MODELS):
        d = json.loads((Path(w) / "decompan.json").read_text())
        m = np.array(d["per_block_mlp_share_median"])
        a = np.array(d["per_block_attention_share_median"])
        B = len(m)
        neg = [b for b in range(B) if m[b] < 0]
        cross = next((b for b in range(max(neg) if neg else 0, B) if m[b] > 0), -1)
        out[name] = {
            "last_negative": max(neg) if neg else None,
            "first_positive_after": cross,
            "band": band,
            "best": best,
            "probe": probe,
        }
        ax.bar(np.arange(B), m, color="#9ca3af", label="MLP write along v(last), per block (median share)")
        ax.plot(np.arange(B), a, "-o", ms=3, color=color, label="attention write along v(last)")
        ax.axhline(0, color="k", lw=0.6)
        if band:
            ax.axvspan(band[0], band[1], color="#f59e0b", alpha=0.25, label=f"judged steering band (best row {best})")
        for p in probe:
            ax.axvline(p, color="#7c3aed", ls="--", lw=1.2)
        ax.axvline(cross - 0.5, color="k", ls=":", lw=1.5, label=f"MLP sign crossing: block {cross}")
        ax.set_title(f"{name}: who writes the final concept direction, by block")
        ax.set_xlabel("block  (dashed purple: paper's best probing layers)")
        ax.legend(fontsize=7, loc="upper left")
        ax.set_ylim(-0.02, max(0.08, float(m.max()) * 1.05))
        log.info(
            "%s: MLP negative through block %s, first positive %d; band %s best %s; probe layers %s",
            name,
            out[name]["last_negative"],
            cross,
            band,
            best,
            probe,
        )
    fig.tight_layout()
    Path("figures/formation").mkdir(parents=True, exist_ok=True)
    fig.savefig("figures/formation/A8-crossing.png", dpi=170)
    Path("work/crossing.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
