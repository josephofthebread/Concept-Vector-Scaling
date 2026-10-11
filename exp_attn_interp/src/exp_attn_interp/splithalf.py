from __future__ import annotations

import logging
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
import numpy as np
import torch
from safetensors.torch import load_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger("splithalf")


def unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--a", type=Path, default=Path("work/vdecomp"))
    parser.add_argument("--b", type=Path, default=Path("work/vdecomp2"))
    parser.add_argument("--out", type=Path, default=Path("work/splithalf"))
    parser.add_argument("--figs", type=Path, default=Path("figures/manifold"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    args.figs.mkdir(parents=True, exist_ok=True)
    dev = args.device
    files = sorted(f for f in args.a.glob("pair-*.safetensors") if (args.b / f.name).exists())
    rel_head, rel_mlp, rel_v, cross = [], [], [], None
    n = 0
    for i, f in enumerate(files):
        ta, tb = load_file(str(f)), load_file(str(args.b / f.name))
        Dh_a = (ta["pole_head"][0] - ta["pole_head"][1]).to(dev, torch.float32)
        Dh_b = (tb["pole_head"][0] - tb["pole_head"][1]).to(dev, torch.float32)
        Dm_a = (ta["pole_mlp"][0] - ta["pole_mlp"][1]).to(dev, torch.float32)
        Dm_b = (tb["pole_mlp"][0] - tb["pole_mlp"][1]).to(dev, torch.float32)
        De_a = (ta["pole_embed"][0] - ta["pole_embed"][1]).to(dev, torch.float32)
        De_b = (tb["pole_embed"][0] - tb["pole_embed"][1]).to(dev, torch.float32)
        B, H, hid = Dh_a.shape
        rel_head.append((unit(Dh_a) * unit(Dh_b)).sum(-1).cpu())
        rel_mlp.append((unit(Dm_a) * unit(Dm_b)).sum(-1).cpu())
        Va = torch.cat([De_a[None], De_a[None] + torch.cumsum(Dh_a.sum(1) + Dm_a, 0)], 0)
        Vb = torch.cat([De_b[None], De_b[None] + torch.cumsum(Dh_b.sum(1) + Dm_b, 0)], 0)
        rel_v.append((unit(Va) * unit(Vb)).sum(-1).cpu())
        Ca, Cb = unit(Dh_a.reshape(B * H, hid)), unit(Dh_b.reshape(B * H, hid))
        cross = Ca @ Cb.T if cross is None else cross + Ca @ Cb.T
        n += 1
        if i % 200 == 0:
            log.info(
                "%d/%d: head reliability median %.2f, vector reliability at L18 %.2f",
                i,
                len(files),
                float(rel_head[-1].median()),
                float(rel_v[-1][18]),
            )
    assert cross is not None
    RH = torch.stack(rel_head).numpy()
    RM = torch.stack(rel_mlp).numpy()
    RV = torch.stack(rel_v).numpy()
    X = (cross / n).cpu().numpy()
    X = 0.5 * (X + X.T)
    np.save(args.out / "rel_head.npy", RH)
    np.save(args.out / "rel_mlp.npy", RM)
    np.save(args.out / "rel_v.npy", RV)
    np.save(args.out / "cross_cowrite.npy", X)
    log.info(
        "vector split-half reliability by layer (median): %s",
        [round(float(np.median(RV[:, L])), 2) for L in range(0, B + 1, 3)],
    )
    log.info(
        "head reliability: median %.3f; heads with median > 0.3: %d of %d; MLP reliability by block: %s",
        float(np.median(RH)),
        int((np.median(RH, 0) > 0.3).sum()),
        B * H,
        [round(float(np.median(RM[:, b])), 2) for b in range(0, B, 3)],
    )
    diag = np.diag(X)
    log.info(
        "cross-half self-cosine of heads (mean over concepts): median %.3f, max %.3f; off-diagonal |mean| %.4f",
        float(np.median(diag)),
        float(diag.max()),
        float(np.abs(X - np.diag(diag)).mean()),
    )
    fig, axes = plt.subplots(1, 3, figsize=(26, 7))
    im = axes[0].imshow(np.median(RH, 0).T, aspect="auto", cmap="viridis", vmin=0, vmax=0.8, interpolation="nearest")
    axes[0].set_title("split-half reliability of each head's diff-write (median over concepts)")
    axes[0].set_xlabel("block")
    axes[0].set_ylabel("head")
    plt.colorbar(im, ax=axes[0], fraction=0.03)
    axes[1].plot(range(B + 1), np.median(RV, 0), color="#7c3aed", lw=2, label="whole vector v(L)")
    axes[1].fill_between(range(B + 1), np.percentile(RV, 25, 0), np.percentile(RV, 75, 0), color="#7c3aed", alpha=0.2)
    axes[1].plot(range(B), np.median(RM, 0), color="#9ca3af", lw=2, label="MLP write of block b")
    axes[1].plot(range(B), np.median(RH, (0, 2)), color="#2563eb", lw=2, label="median head write of block b")
    axes[1].axvspan(19, 22, color="#f59e0b", alpha=0.12)
    axes[1].set_title("split-half reliability (cos between the two 8-story halves) by depth")
    axes[1].set_xlabel("layer / block")
    axes[1].legend(fontsize=8)
    Xo = X - np.diag(diag)
    blk = Xo.reshape(B, H, B, H).mean((1, 3))
    im = axes[2].imshow(blk, cmap="RdBu_r", vmin=-0.02, vmax=0.02, interpolation="nearest")
    axes[2].set_title(
        "cross-half co-writing between blocks (mean cos of block i's heads in half A with block j's in half B, diagonal heads removed)"
    )
    axes[2].set_xlabel("block")
    axes[2].set_ylabel("block")
    plt.colorbar(im, ax=axes[2], fraction=0.03)
    fig.tight_layout()
    fig.savefig(args.figs / "B5-splithalf.png", dpi=170)
    plt.close(fig)
    log.info("wrote %s", args.figs)


if __name__ == "__main__":
    main()
