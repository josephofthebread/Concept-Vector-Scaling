from __future__ import annotations

import collections
import glob
import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger("dosecollapse")
TRIGGER = Path("../exp_trigger_prompts")
ALPHAS = (-0.5, -0.25, -0.1, 0.1, 0.25, 0.5)
METRICS = ("concept", "instruct", "fluency")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--gainan", type=Path, default=Path("work/gainan/gainan.npz"))
    parser.add_argument("--curves", type=Path, default=Path("work/formation/curves.npz"))
    parser.add_argument("--out", type=Path, default=Path("work/dosecollapse.npz"))
    parser.add_argument("--figs", type=Path, default=Path("figures/dose"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.figs.mkdir(parents=True, exist_ok=True)
    norms: dict[int, float] = {}
    for f in ("norms-Qwen_Qwen3-8B.json", "norms-Qwen_Qwen3-8B-1217.json"):
        norms.update(
            {int(k): float(v) for k, v in json.loads((TRIGGER / "work" / "steer" / f).read_text())["norms"].items()}
        )
    layers = sorted(norms)
    sub: dict[str, dict[tuple[int, int, float], list[float]]] = {m: collections.defaultdict(list) for m in METRICS}
    for d in ("judge", "judge-1217"):
        for m in METRICS:
            for f in glob.glob(str(TRIGGER / "work" / d / f"{m}-shard-*.jsonl")):
                for line in open(f):
                    r = json.loads(line)
                    p, _slot, lay, al = r["key"].split(":")
                    sub[m][(int(p), int(lay[1:]), float(al[1:]))].append(float(r["score"]))
    log.info("judge cells: %d", len(sub["concept"]))
    G = np.load(args.gainan)

    lin, mid, big = G["lin"][:, :2, :, 0].mean(1), G["mid"][:, :2, :, 0].mean(1), G["big"][:, :2, :, 0].mean(1)
    B = lin.shape[1]
    pop = {0.1: np.median(lin, 0), 0.25: np.median(mid, 0), 0.5: np.median(big, 0)}
    carry = {a: {L: float(np.prod(1 + pop[a][L : B - 1])) for L in layers} for a in pop}
    for a in pop:
        log.info("carry_pred at |alpha| %.2f by row: %s", a, {L: round(carry[a][L], 2) for L in layers})
    C = np.load(args.curves)
    cp = np.load(Path(args.curves).parent / "pairs.npy")
    natural = {int(k): float(v) for k, v in zip(cp, C["norm_v"][:, B - 1])}
    nat_med = float(np.median(list(natural.values())))
    log.info("natural pole gap |v(35)|: median %.1f", nat_med)
    rows = []
    percell: dict[tuple[int, float], dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for (p, L, al), vals in sub["concept"].items():
        if L not in norms:
            continue
        key = (L, al)
        percell[key]["concept"].append(float(np.mean(vals)))
        percell[key]["instruct"].append(float(np.mean(sub["instruct"].get((p, L, al), [np.nan]))))
        percell[key]["fluency"].append(float(np.mean(sub["fluency"].get((p, L, al), [np.nan]))))
    for (L, al), sc in sorted(percell.items()):
        a = abs(al)
        a_eff = a * norms[L] * carry[a][L] / nat_med
        rows.append(
            (
                L,
                al,
                a_eff,
                np.nanmean(sc["concept"]),
                np.nanmean(sc["instruct"]),
                np.nanmean(sc["fluency"]),
                len(sc["concept"]),
            )
        )
    rows.sort(key=lambda r: r[2])
    for r in rows:
        log.info("L%2d alpha %+.2f  a_eff %.3f  concept %.3f instruct %.3f fluency %.3f (n %d)", *r)
    R = np.array([r[:6] for r in rows])
    np.savez(str(args.out), rows=R, layers=np.array(layers), nat_med=nat_med)
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    cmap = plt.get_cmap("viridis")
    for i, m in enumerate(METRICS):
        for sign, ax in ((1, axes[0, i]), (-1, axes[1, i])):
            for L in layers:
                sel = [r for r in rows if r[0] == L and np.sign(r[1]) == sign]
                sel.sort(key=lambda r: r[2])
                ax.plot(
                    [r[2] for r in sel],
                    [r[3 + i] for r in sel],
                    marker="o",
                    color=cmap((L - layers[0]) / (layers[-1] - layers[0])),
                    label=f"row {L}",
                )
            ax.set_xscale("log")
            ax.set_xlabel(
                "a_eff = |alpha| · norm_L · prod(1+gain) / |v(35)|  (carried amplitude in units of the natural pole gap)"
            )
            ax.set_ylabel(m)
            ax.set_title(
                f"{m} vs carried amplitude, alpha {'> 0 (concept pole)' if sign > 0 else '< 0 (antagonist pole)'}; one line per row, 3 alphas each"
            )
            ax.set_ylim(0, 2.05)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(args.figs / "S1-collapse.png", dpi=170)
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    for i, m in enumerate(METRICS):
        ax = axes[i]
        for L in layers:
            sel = [r for r in rows if r[0] == L and r[1] > 0]
            sel.sort(key=lambda r: r[1])
            ax.plot(
                [abs(r[1]) * norms[L] / nat_med for r in sel],
                [r[3 + i] for r in sel],
                marker="o",
                color=cmap((L - layers[0]) / (layers[-1] - layers[0])),
                label=f"row {L}",
            )
        ax.set_xscale("log")
        ax.set_xlabel("injected amplitude |alpha| · norm_L / |v(35)| (no carry)")
        ax.set_title(f"{m} vs injected amplitude (alpha > 0)")
        ax.set_ylim(0, 2.05)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(args.figs / "S2-raw.png", dpi=170)
    log.info("wrote %s", args.figs)


if __name__ == "__main__":
    main()
