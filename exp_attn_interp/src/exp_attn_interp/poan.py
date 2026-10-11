from __future__ import annotations

import collections
import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger("poan")
METRICS = ("concept", "instruct", "fluency")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--judge", type=Path, default=Path("work/pojudge"))
    parser.add_argument("--figs", type=Path, default=Path("figures/posteer"))
    parser.add_argument("--reference", default="orig")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.figs.mkdir(parents=True, exist_ok=True)
    per_arm: dict[str, dict[int, dict[str, list[float]]]] = {}
    per_gen: dict[str, dict[str, dict[str, float]]] = {}
    for d in sorted(p for p in args.judge.iterdir() if p.is_dir()):
        scores: dict[str, dict[str, float]] = collections.defaultdict(dict)
        for m in METRICS:
            for f in d.glob(f"{m}-shard-*.jsonl"):
                for line in f.open():
                    r = json.loads(line)
                    scores[r["key"]][m] = float(r["score"])
        per_gen[d.name] = scores
        agg: dict[int, dict[str, list[float]]] = collections.defaultdict(lambda: collections.defaultdict(list))
        for key, s in scores.items():
            if len(s) < 3:
                continue
            k = int(key.split(":")[0])
            for m in METRICS:
                agg[k][m].append(s[m])
            agg[k]["overall"].append(
                0.0
                if min(s["concept"], s["instruct"], s["fluency"]) <= 0
                else 3.0 / (1 / s["concept"] + 1 / s["instruct"] + 1 / s["fluency"])
            )
        per_arm[d.name] = agg
        log.info(
            "arm %-8s: %d concepts, %d judged generations",
            d.name,
            len(agg),
            sum(1 for s in scores.values() if len(s) == 3),
        )
    ref = per_arm[args.reference]
    ks = sorted(k for k in ref if all(k in per_arm[a] for a in per_arm))
    rows = []
    for a, agg in per_arm.items():
        means = {m: float(np.mean([np.mean(agg[k][m]) for k in ks])) for m in METRICS + ("overall",)}
        dc = np.array([np.mean(agg[k]["concept"]) - np.mean(ref[k]["concept"]) for k in ks])
        do = np.array([np.mean(agg[k]["overall"]) - np.mean(ref[k]["overall"]) for k in ks])
        rows.append((a, means, dc, do))
        log.info(
            "%-8s concept %.3f instruct %.3f fluency %.3f overall %.3f | vs %s: d concept %+.3f (improved %.0f%%, worsened %.0f%%), d overall %+.3f",
            a,
            means["concept"],
            means["instruct"],
            means["fluency"],
            means["overall"],
            args.reference,
            dc.mean(),
            100 * (dc > 0).mean(),
            100 * (dc < 0).mean(),
            do.mean(),
        )
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    names = [r[0] for r in rows]
    x = np.arange(len(rows))
    for j, m in enumerate(METRICS + ("overall",)):
        axes[0].bar(x + (j - 1.5) * 0.2, [r[1][m] for r in rows], width=0.2, label=m)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].set_ylim(0, 2)
    axes[0].set_title(f"judge scores on the bottom-quartile concepts (n={len(ks)}), row 22, alpha 0.25")
    axes[0].legend(fontsize=8)
    axes[1].boxplot([r[2] for r in rows], labels=names, showmeans=True)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_title(f"per-concept change of the concept score vs {args.reference}")
    fig.tight_layout()
    fig.savefig(args.figs / "P1-arms.png", dpi=170)
    log.info("wrote %s", args.figs)


if __name__ == "__main__":
    main()
