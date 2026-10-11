from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
import numpy as np
import torch
from safetensors.torch import load_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from exp_attn_interp.vectors import load_diff_vectors  # noqa: E402

log = logging.getLogger("manifold")


def unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--vdecomp", type=Path, default=Path("work/vdecomp"))
    parser.add_argument("--out", type=Path, default=Path("work/manifold"))
    parser.add_argument("--figs", type=Path, default=Path("figures/manifold"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rank", type=int, default=32, help="dimension of the concept subspace S")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    args.figs.mkdir(parents=True, exist_ok=True)
    dev = args.device
    files = sorted(args.vdecomp.glob("pair-*.safetensors"))
    if args.limit:
        files = files[: args.limit]
    G = load_diff_vectors(device=dev)
    steer = json.loads(Path("work/steerability.json").read_text())["qwen"]
    Gl = G[35]
    mean_dir = unit(Gl.mean(0))
    Gc = Gl - (Gl @ mean_dir)[:, None] * mean_dir[None]
    U, Sv, Vt = torch.linalg.svd(Gc, full_matrices=False)
    S_basis = Vt[: args.rank]
    log.info(
        "concept subspace: top-%d modes carry %.2f of the centered energy",
        args.rank,
        float((Sv[: args.rank] ** 2).sum() / (Sv**2).sum()),
    )
    cow = None
    n_ok = 0
    spec = []
    frac = []
    pole_same = []
    top_vs_v = []
    signed = []
    pairs = []
    for i, f in enumerate(files):
        t = load_file(str(f))
        k = int(f.stem.split("-")[1])
        ph = t["pole_head"].to(dev, torch.float32)
        pm = t["pole_mlp"].to(dev, torch.float32)
        B, H, hid = ph.shape[1:]
        Dh = ph[0] - ph[1]
        Dm = pm[0] - pm[1]
        C = Dh.reshape(B * H, hid)
        Cu = unit(C)
        cow = Cu @ Cu.T if cow is None else cow + Cu @ Cu.T
        sv = torch.linalg.svdvals(C)
        e = sv**2
        pr = float(e.sum() ** 2 / (e**2).sum())
        top = torch.linalg.svd(C, full_matrices=False)[2][0]
        vk = unit(G[35, k])
        spec.append(
            [
                pr,
                float(e[0] / e.sum()),
                float(e[:3].sum() / e.sum()),
                float(e[:10].sum() / e.sum()),
                abs(float(top @ vk)),
            ]
        )

        def split(X: torch.Tensor) -> torch.Tensor:
            tot = (X**2).sum(-1).clamp_min(1e-12)
            m = (X @ mean_dir) ** 2
            s = ((X @ S_basis.T) ** 2).sum(-1)
            return torch.stack([m / tot, s / tot, 1 - (m + s) / tot], -1)

        frac.append(torch.stack([split(Dh.sum(1)), split(Dm)], 1).cpu())
        De = t["pole_embed"].to(dev, torch.float32)[0] - t["pole_embed"].to(dev, torch.float32)[1]
        Vrun = torch.cat([De[None], De[None] + torch.cumsum(Dh.sum(1) + Dm, 0)], 0)[:B]
        Vh = unit(Vrun)
        spec_dir = unit(Vh - (Vh @ mean_dir)[:, None] * mean_dir[None])
        Aw, Mw = Dh.sum(1), Dm
        signed.append(
            torch.stack(
                [
                    (unit(Aw) @ mean_dir),
                    (unit(Mw) @ mean_dir),
                    (unit(Aw) * spec_dir).sum(-1),
                    (unit(Mw) * spec_dir).sum(-1),
                    (Vh @ mean_dir),
                ],
                1,
            ).cpu()
        )
        pole_same.append((unit(ph[0]) * unit(ph[1])).sum(-1).cpu())
        top_vs_v.append(abs(float(top @ vk)))
        pairs.append(k)
        n_ok += 1
        if i % 100 == 0:
            log.info("%d/%d pair %d: PR %.0f top1 %.2f top3 %.2f |cos(top, v)| %.2f", i, len(files), k, *spec[-1])
    assert cow is not None
    cow_np = (cow / n_ok).cpu().numpy()
    spec_a = np.array(spec)
    frac_a = torch.stack(frac).numpy()
    ps = torch.stack(pole_same).numpy()
    np.save(args.out / "cowrite.npy", cow_np)
    np.save(args.out / "spectrum.npy", spec_a)
    np.save(args.out / "energy_fractions.npy", frac_a)
    np.save(args.out / "pole_same.npy", ps)
    sg = torch.stack(signed).numpy()
    np.save(args.out / "signed.npy", sg)
    np.save(args.out / "pairs.npy", np.array(pairs))
    st = np.array([steer.get(str(k), np.nan) for k in pairs])
    log.info(
        "contribution spectrum per concept: PR median %.1f; top-1 %.2f, top-3 %.2f, top-10 %.2f of energy; |cos(top dir, v(35))| median %.2f",
        np.median(spec_a[:, 0]),
        np.median(spec_a[:, 1]),
        np.median(spec_a[:, 2]),
        np.median(spec_a[:, 3]),
        np.median(spec_a[:, 4]),
    )
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    im = axes[0].imshow(cow_np, cmap="RdBu_r", vmin=-0.3, vmax=0.3, interpolation="nearest")
    axes[0].set_title(
        "mean over concepts of cos(D head_i, D head_j): who writes the same direction (index = block*32 + head)"
    )
    ticks = np.arange(0, B * H, H * 3)
    axes[0].set_xticks(ticks)
    axes[0].set_xticklabels([str(t // H) for t in ticks])
    axes[0].set_yticks(ticks)
    axes[0].set_yticklabels([str(t // H) for t in ticks])
    axes[0].set_xlabel("block")
    axes[0].set_ylabel("block")
    plt.colorbar(im, ax=axes[0], fraction=0.03)
    blk = cow_np.reshape(B, H, B, H).mean((1, 3))
    im = axes[1].imshow(blk, cmap="RdBu_r", vmin=-0.1, vmax=0.1, interpolation="nearest")
    axes[1].set_title("block-level mean of the same (mean cosine between the heads of block i and of block j)")
    axes[1].set_xlabel("block")
    axes[1].set_ylabel("block")
    plt.colorbar(im, ax=axes[1], fraction=0.03)
    fig.tight_layout()
    fig.savefig(args.figs / "B1-cowrite.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(24, 6.5))
    names = [
        "along the shared mean direction",
        "inside the concept subspace S (top-%d, mean removed)" % args.rank,
        "outside both",
    ]
    med = np.median(frac_a, 0)
    for j, ax in enumerate(axes):
        ax.plot(range(B), med[:, 0, j], "-o", ms=3, color="#2563eb", label="attention write")
        ax.plot(range(B), med[:, 1, j], "-s", ms=3, color="#9ca3af", label="MLP write")
        ax.set_title("energy fraction of the block's write " + names[j])
        ax.set_xlabel("block")
        ax.axvspan(19, 22, color="#f59e0b", alpha=0.12)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(args.figs / "B2-energy-fractions.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(24, 6.5))
    xs = np.arange(B)
    for j, (ax, title) in enumerate(
        zip(axes[:2], ["cos(write, shared mean direction)", "cos(write, concept-specific part of v̂(b))"])
    ):
        for c, color, lab in ((0, "#2563eb", "attention"), (1, "#9ca3af", "MLP")):
            Mx = sg[:, :, 2 * j + c]
            ax.plot(xs, np.median(Mx, 0), color=color, lw=2, label=lab)
            ax.fill_between(xs, np.percentile(Mx, 25, 0), np.percentile(Mx, 75, 0), color=color, alpha=0.2)
        ax.axhline(0, color="k", lw=0.5)
        ax.axvspan(19, 22, color="#f59e0b", alpha=0.12)
        ax.set_title(title)
        ax.set_xlabel("block")
        ax.legend(fontsize=8)
    ax = axes[2]
    Mx = sg[:, :, 4]
    ax.plot(xs, np.median(Mx, 0), color="#7c3aed", lw=2)
    ax.fill_between(xs, np.percentile(Mx, 25, 0), np.percentile(Mx, 75, 0), color="#7c3aed", alpha=0.2)
    ax.set_title("cos(v̂(b), shared mean direction): how much of the running vector is the shared component")
    ax.set_xlabel("block")
    ax.axvspan(19, 22, color="#f59e0b", alpha=0.12)
    fig.tight_layout()
    fig.savefig(args.figs / "B4-signed.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    axes[0].scatter(spec_a[:, 0], st, s=6, alpha=0.5)
    axes[0].set_xlabel("participation ratio of the concept's head-contribution set")
    axes[0].set_ylabel("judged steerability")
    axes[0].set_title("how many directions the heads write in, vs steerability")
    axes[1].hist(spec_a[:, 4], bins=40, color="#2563eb")
    axes[1].set_title("|cos(top singular direction of the contributions, v(35))|")
    m = np.median(ps, 0)
    im = axes[2].imshow(m.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    axes[2].set_title(
        "median over concepts of cos(head write on concept pole, head write on antagonist pole)\n(+1: same direction on both poles = gain/bias; −1: opposite = contrast)"
    )
    axes[2].set_xlabel("block")
    axes[2].set_ylabel("head")
    plt.colorbar(im, ax=axes[2], fraction=0.03)
    fig.tight_layout()
    fig.savefig(args.figs / "B3-spectrum-poles.png", dpi=170)
    plt.close(fig)
    log.info("wrote %s and %s", args.out, args.figs)


if __name__ == "__main__":
    main()
