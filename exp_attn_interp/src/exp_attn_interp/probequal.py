from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
import numpy as np
from safetensors.torch import load_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger("probequal")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--storyproj", type=Path, default=Path("work/storyproj"))
    parser.add_argument("--out", type=Path, default=Path("work/probequal"))
    parser.add_argument("--figs", type=Path, default=Path("figures/probequal"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    args.figs.mkdir(parents=True, exist_ok=True)
    steer = json.loads(Path("work/steerability.json").read_text())["qwen"]
    rows: dict[str, list[np.ndarray]] = {
        "auroc_pooled": [],
        "auroc_cos": [],
        "d_between": [],
        "token_snr": [],
        "cos_snr": [],
    }
    pairs = []
    for f in sorted(args.storyproj.glob("pair-*.safetensors")):
        t = {k: v.numpy() for k, v in load_file(str(f)).items()}
        k = int(f.stem.split("-")[1])
        pooled, mean, std, cmean, cstd = t["pooled"], t["mean"], t["std"], t["cmean"], t["cstd"]
        rows["auroc_pooled"].append(
            (
                (pooled[0][:, None, :] > pooled[1][None, :, :]).mean((0, 1))
                + 0.5 * (pooled[0][:, None, :] == pooled[1][None, :, :]).mean((0, 1))
            )
        )
        rows["auroc_cos"].append(
            (
                (cmean[0][:, None, :] > cmean[1][None, :, :]).mean((0, 1))
                + 0.5 * (cmean[0][:, None, :] == cmean[1][None, :, :]).mean((0, 1))
            )
        )
        gap = mean[0].mean(0) - mean[1].mean(0)
        within = np.sqrt(0.5 * (mean[0].var(0) + mean[1].var(0))).clip(1e-6)
        rows["d_between"].append(gap / within)
        rows["token_snr"].append(gap / np.median(np.concatenate([std[0], std[1]]), 0).clip(1e-6))
        cgap = cmean[0].mean(0) - cmean[1].mean(0)
        rows["cos_snr"].append(cgap / np.median(np.concatenate([cstd[0], cstd[1]]), 0).clip(1e-6))
        pairs.append(k)
    R = {key: np.stack(v) for key, v in rows.items()}
    np.savez(str(args.out / "probequal.npz"), pairs=np.array(pairs), **R)  # type: ignore[arg-type]
    st = np.array([steer.get(str(k), np.nan) for k in pairs])
    order = np.argsort(np.nan_to_num(st, nan=-1))
    L = R["auroc_pooled"].shape[1]
    for key in R:
        med = np.median(R[key], 0)
        log.info(
            "%-13s median by layer: %s  | argmax layer %d",
            key,
            [round(float(med[i]), 3) for i in range(0, L, 3)],
            int(np.argmax(med)),
        )
    best = {key: np.argmax(R[key], 1) for key in R}
    log.info("per-concept best layer (median over concepts): %s", {key: int(np.median(v)) for key, v in best.items()})

    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    xs = np.arange(L)
    for ax, key, title in zip(
        axes[0],
        ["auroc_pooled", "token_snr", "cos_snr"],
        [
            "pole-vs-antagonist AUROC of the pooled projection",
            "token SNR: pole gap / within-story token std of the projection",
            "token SNR on the cosine (the paper's z-scored readout)",
        ],
    ):
        M = R[key]
        ax.plot(xs, np.median(M, 0), color="#7c3aed", lw=2, label="median")
        ax.fill_between(xs, np.percentile(M, 25, 0), np.percentile(M, 75, 0), color="#7c3aed", alpha=0.2)
        ax.plot(xs, np.percentile(M, 10, 0), color="#7c3aed", lw=1, ls="--", label="p10")
        ax.axvspan(19, 22, color="#f59e0b", alpha=0.12)
        ax.set_title(title)
        ax.set_xlabel("layer L")
        ax.legend(fontsize=8)
    for ax, key, (vmin, vmax) in zip(
        axes[1], ["auroc_pooled", "token_snr", "cos_snr"], [(0.5, 1.0), (0, None), (0, None)]
    ):
        M = R[key][order]
        im = ax.imshow(
            M,
            aspect="auto",
            cmap="magma",
            vmin=vmin,
            vmax=vmax if vmax else np.percentile(M, 98),
            interpolation="nearest",
        )
        ax.set_title(key + " per concept (sorted by steerability)")
        ax.set_xlabel("layer L")
        plt.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(args.figs / "D1-probe-quality.png", dpi=170)
    plt.close(fig)
    log.info("wrote %s", args.figs)


if __name__ == "__main__":
    main()
