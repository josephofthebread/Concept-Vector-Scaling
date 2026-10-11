from __future__ import annotations

import collections
import glob
import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
import numpy as np
from safetensors import safe_open
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger("steereff")
TRIGGER = Path("../exp_trigger_prompts")
ALPHAS = (0.1, 0.25, 0.5)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--gainan", type=Path, default=Path("work/gainan/gainan.npz"))
    parser.add_argument("--attnsrc", type=Path, default=Path("work/attnsrcan/sources.npz"))
    parser.add_argument("--family", default="qwen", choices=("qwen", "llama"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--figs", type=Path, default=Path("figures/dose"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.figs.mkdir(parents=True, exist_ok=True)
    fam = {
        "qwen": (("norms-Qwen_Qwen3-8B.json", "norms-Qwen_Qwen3-8B-1217.json"), ("judge", "judge-1217"), 22, (19, 22)),
        "llama": (("norms-unsloth_Meta-Llama-3.1-8B-Instruct.json",), ("judge-llama",), 15, (13, 17)),
    }[args.family]
    norm_files, judge_dirs, ref_row, band = fam
    out_path = args.out or Path(f"work/steereff-{args.family}.json")
    norms: dict[int, float] = {}
    for fname in norm_files:
        norms.update(
            {int(k): float(v) for k, v in json.loads((TRIGGER / "work" / "steer" / fname).read_text())["norms"].items()}
        )
    rows = sorted(norms)
    cell: dict[tuple[int, float], dict[int, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    flu: dict[tuple[int, float], dict[int, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for d in judge_dirs:
        for m, store in (("concept", cell), ("fluency", flu)):
            for f in glob.glob(str(TRIGGER / "work" / d / f"{m}-shard-*.jsonl")):
                for line in open(f):
                    r = json.loads(line)
                    p, _s, lay, al = r["key"].split(":")
                    store[(int(lay[1:]), float(al[1:]))][int(p)].append(float(r["score"]))
    score = {k: float(np.mean([np.mean(v) for v in d.values()])) for k, d in cell.items()}
    fscore = {k: float(np.mean([np.mean(v) for v in d.values()])) for k, d in flu.items()}
    profiles = {}
    for sign, name in ((1.0, "concept pole"), (-1.0, "antagonist pole")):
        ref = rows.index(ref_row)
        L_idx = {L: i for i, L in enumerate(rows)}
        obs = [(L, a, score[(L, sign * a)]) for L in rows for a in ALPHAS if (L, sign * a) in score]

        def unpack(theta):
            y0, y1, logh, p = theta[:4]
            logc = np.insert(theta[4:], ref, 0.0)
            return y0, y1, np.exp(logh), p, np.exp(logc)

        def resid(theta):
            y0, y1, h, p, c = unpack(theta)
            out = []
            for L, a, s in obs:
                xp = np.power(np.clip(a * c[L_idx[L]], 1e-9, None), p)
                out.append(y0 + (y1 - y0) * xp / (xp + h**p) - s)
            return np.array(out)

        theta0 = np.concatenate(
            [[min(s for _, _, s in obs), max(s for _, _, s in obs) + 0.2, np.log(0.3), 1.0], np.zeros(len(rows) - 1)]
        )
        sol = least_squares(
            resid, theta0, bounds=([-1, 0, -6, 0.2] + [-4] * (len(rows) - 1), [2, 3, 3, 6] + [4] * (len(rows) - 1))
        )
        y0, y1, h, p, c = unpack(sol.x)
        c, params, rms = {L: float(c[L_idx[L]]) for L in rows}, (y0, y1, h, p), float(np.sqrt(np.mean(sol.fun**2)))
        profiles[name] = c
        log.info(
            "%s: master y0 %.2f y1 %.2f half-point %.3f power %.2f, rms residual %.3f; efficacy c_L (c_22 = 1): %s",
            name,
            *params,
            rms,
            {L: round(v, 3) for L, v in c.items()},
        )
    if args.family != "qwen" or not args.gainan.exists():
        out_path.write_text(
            json.dumps(
                {
                    "efficacy": profiles,
                    "score": {f"{L}:{a}": v for (L, a), v in score.items()},
                    "fluency": {f"{L}:{a}": v for (L, a), v in fscore.items()},
                },
                indent=1,
            )
        )
        fig, ax = plt.subplots(1, 1, figsize=(9, 7))
        for name, c in profiles.items():
            ax.plot(rows, [c[L] for L in rows], marker="o", lw=2, label=f"fitted efficacy c_L, {name}")
        ax.axvspan(band[0], band[1], color="#f59e0b", alpha=0.12)
        ax.set_xlabel("injection row L")
        ax.set_title(f"{args.family}: steering efficacy per unit alpha by row (c_{ref_row} = 1)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.figs / f"S3-efficacy-{args.family}.png", dpi=170)
        log.info("wrote %s", args.figs)
        return
    G = np.load(args.gainan)
    B = G["lin"].shape[2]
    g = np.median(G["mid"][:, :2, :, 0].mean(1), 0)
    gate = np.median(G["path"][:, :2, :, 0, 1].mean(1), 0)
    carry = {L: float(np.prod(1 + g[L : B - 1])) for L in rows}
    gates = {L: float(np.clip(gate[L : B - 1], 0, None).sum()) for L in rows}
    surv_meas = {}
    for L in (12, 16, 20, 24, 28):
        vals = []
        for fp in Path(f"work/inject_L{L}").glob("pair-*.safetensors"):
            with safe_open(str(fp), framework="np") as h:
                vals.append(float(h.get_tensor("survival")))
        if vals:
            surv_meas[L] = float(np.median(vals))
    xs, ys = zip(*sorted(surv_meas.items()))
    surv = {L: float(np.interp(L, xs, ys)) for L in rows}
    log.info("measured survival by row: %s; interpolated: %s", surv_meas, {L: round(v, 2) for L, v in surv.items()})
    S = np.load(args.attnsrc)
    Dsrc = np.median(S["D"][:, 1], 0)
    per_block_all = Dsrc.sum((1, 2))
    per_block_ctx = Dsrc[:, :, [0, 1, 2, 4, 5]].sum((1, 2))
    vis_all = {L: float(per_block_all[L:].sum() / per_block_all.sum()) for L in rows}
    vis_ctx = {L: float(per_block_ctx[L:].sum() / per_block_ctx.sum()) for L in rows}
    log.info(
        "attention visibility above L (all / context sources): %s",
        {L: (round(vis_all[L], 2), round(vis_ctx[L], 2)) for L in rows},
    )
    predictor_map = {
        "raw norm_L": {L: norms[L] for L in rows},
        "norm_L * carry(L)": {L: norms[L] * carry[L] for L in rows},
        "norm_L * survival(L)": {L: norms[L] * surv[L] for L in rows},
        "norm_L * carry * gate capacity": {L: norms[L] * carry[L] * gates[L] for L in rows},
        "norm_L * survival * attention visibility (context)": {L: norms[L] * surv[L] * vis_ctx[L] for L in rows},
        "norm_L * survival * attention visibility (all)": {L: norms[L] * surv[L] * vis_all[L] for L in rows},
    }
    out = {
        "efficacy": profiles,
        "score": {f"{L}:{a}": v for (L, a), v in score.items()},
        "fluency": {f"{L}:{a}": v for (L, a), v in fscore.items()},
        "predictors": predictor_map,
        "carry": carry,
        "gates": gates,
        "survival": surv,
    }
    out_path.write_text(json.dumps(out, indent=1))
    fig, axes = plt.subplots(1, 3, figsize=(26, 7))
    ax = axes[0]
    for name, c in profiles.items():
        ax.plot(rows, [c[L] for L in rows], marker="o", lw=2, label=f"fitted efficacy c_L, {name}")
    for name, pr in predictor_map.items():
        ax.plot(rows, [pr[L] / pr[22] for L in rows], ls="--", marker=".", label=name + " (÷ value at 22)")
    ax.axvspan(band[0], band[1], color="#f59e0b", alpha=0.12)
    ax.set_yscale("log")
    ax.set_xlabel("injection row L")
    ax.set_title("efficacy per unit alpha, fitted from the judge grid, vs mechanism predictors")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax = axes[1]
    c = profiles["concept pole"]
    for L in rows:
        ax.plot(
            [a * c[L] for a in ALPHAS],
            [score[(L, a)] for a in ALPHAS],
            marker="o",
            color=plt.get_cmap("viridis")((L - 12) / 12),
            label=f"row {L}",
        )
    xx = np.logspace(-2, 0.5, 200)
    ax.set_xscale("log")
    ax.set_xlabel("alpha · c_L")
    ax.set_title("concept score vs the fitted effective push (all rows on one curve if the model holds)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    ax = axes[2]
    for L in rows:
        ax.plot(
            ALPHAS,
            [fscore[(L, a)] for a in ALPHAS],
            marker="o",
            color=plt.get_cmap("viridis")((L - 12) / 12),
            label=f"row {L}",
        )
        ax.plot(
            ALPHAS, [fscore[(L, -a)] for a in ALPHAS], marker="x", ls=":", color=plt.get_cmap("viridis")((L - 12) / 12)
        )
    ax.set_xscale("log")
    ax.set_xlabel("alpha (o concept pole, x antagonist pole)")
    ax.set_title("fluency depends on alpha, not on the row")
    ax.grid(alpha=0.3)
    _ = xx
    fig.tight_layout()
    fig.savefig(args.figs / "S3-efficacy.png", dpi=170)
    log.info("wrote %s", args.figs)


if __name__ == "__main__":
    main()
