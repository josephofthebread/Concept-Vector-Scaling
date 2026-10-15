from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODELS = [
    ("Qwen3-8B", "Qwen3-8B", "#2563eb"),
    ("Meta-Llama-3.1-8B-Instruct", "Llama-3.1-8B", "#dc2626"),
    ("gemma-2-9b-it", "gemma-2-9b", "#059669"),
]
FIELD_LABEL = {"lang": "language", "genre": "genre", "generator": "generator"}


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--field", default="lang")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for tag, label, color in MODELS:
        p = Path(f"work/oodcurve/oodcurve-{tag}.json")
        if not p.exists():
            continue
        s = json.loads(p.read_text()).get(args.field)
        if not s:
            continue
        band = s["band"]
        N = [b["N"] for b in band]
        med = [b["median"] for b in band]
        lo = [b["p2_5"] for b in band]
        hi = [b["p97_5"] for b in band]
        ns = s["nstar"]
        ax.plot(
            N,
            med,
            "-o",
            color=color,
            lw=2,
            ms=4,
            label=f"{label}  (N*={ns['median']}, p10–p90 {ns['p10']}–{ns['p90']})",
        )
        ax.fill_between(N, lo, hi, color=color, alpha=0.15)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("stories per class used to build the vector (N)")
    ax.set_ylabel("held-out ROC-AUC")
    ax.set_title(
        f"How many stories to stabilize a concept vector — per-{FIELD_LABEL[args.field]} cell\n"
        "median with 2.5–97.5 percentile band across concepts"
    )
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out = args.out or Path(f"work/oodcurve/oodcurve-{args.field}.svg")
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    print(f"wrote {out} (+ .png)")


if __name__ == "__main__":
    main()
