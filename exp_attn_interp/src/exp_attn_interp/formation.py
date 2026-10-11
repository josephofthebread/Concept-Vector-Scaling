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

log = logging.getLogger("formation")


def unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--vdecomp", type=Path, default=Path("work/vdecomp"))
    parser.add_argument("--out", type=Path, default=Path("work/formation"))
    parser.add_argument("--figs", type=Path, default=Path("figures/formation"))
    parser.add_argument("--limit", type=int, default=0)
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
    pairs, rows = [], {}
    V_all = []
    for i, f in enumerate(files):
        t = load_file(str(f))
        k = int(f.stem.split("-")[1])
        ph = t["pole_head"].to(dev, torch.float32)
        pm = t["pole_mlp"].to(dev, torch.float32)
        pe = t["pole_embed"].to(dev, torch.float32)
        pr = t["pole_resid"].to(dev, torch.float32)
        B, H, hid = ph.shape[1:]
        Dh, Dm, De, Dr = ph[0] - ph[1], pm[0] - pm[1], pe[0] - pe[1], pr[0] - pr[1]
        A = Dh.sum(1)
        W = A + Dm
        V = torch.cat([De[None], De[None] + torch.cumsum(W, 0)], 0)
        exact = (V[:B] - Dr[:B]).norm(dim=-1) / Dr[:B].norm(dim=-1).clamp_min(1e-8)
        Vh = unit(V)
        vL = Vh[B - 1]
        cur = Vh[:B]
        along = (W * cur).sum(-1)
        perp = (W - along[:, None] * cur).norm(dim=-1)
        along_a = (A * cur).sum(-1)
        perp_a = (A - along_a[:, None] * cur).norm(dim=-1)
        along_m = (Dm * cur).sum(-1)
        perp_m = (Dm - along_m[:, None] * cur).norm(dim=-1)
        rot = (Vh[1:] * Vh[:-1]).sum(-1)
        to_last = (Vh * vL).sum(-1)
        to_gen = (Vh * G[:, k]).sum(-1)
        wlast = (unit(W) * vL).sum(-1)
        pole_m = torch.stack([(pm[s_] * cur).sum(-1) for s_ in (0, 1)])
        pole_a = torch.stack([(ph[s_].sum(1) * cur).sum(-1) for s_ in (0, 1)])
        pole_mn = pm.norm(dim=-1)
        pole_an = ph.sum(2).norm(dim=-1)
        res_dir = unit(pr[:, :B])
        ctrl_resid = torch.stack([(unit(pm[s_]) * res_dir[s_]).sum(-1) for s_ in (0, 1)]).mean(0)
        ctrl_resid_a = torch.stack([(unit(ph[s_].sum(1)) * res_dir[s_]).sum(-1) for s_ in (0, 1)]).mean(0)
        kj = (k + 517) % G.shape[1]
        other = G[:B, kj]
        ctrl_other_m = (unit(Dm) * other).sum(-1)
        ctrl_other_a = (unit(A) * other).sum(-1)
        own_paper_m = (unit(Dm) * G[:B, k]).sum(-1)
        rows[k] = {
            "exact": exact.cpu(),
            "norm_v": V.norm(dim=-1).cpu(),
            "norm_w": W.norm(dim=-1).cpu(),
            "norm_a": A.norm(dim=-1).cpu(),
            "norm_m": Dm.norm(dim=-1).cpu(),
            "along": along.cpu(),
            "perp": perp.cpu(),
            "along_a": along_a.cpu(),
            "perp_a": perp_a.cpu(),
            "along_m": along_m.cpu(),
            "perp_m": perp_m.cpu(),
            "rot": rot.cpu(),
            "to_last": to_last.cpu(),
            "to_gen": to_gen.cpu(),
            "wlast": wlast.cpu(),
            "pole_m": pole_m.cpu(),
            "pole_a": pole_a.cpu(),
            "pole_mn": pole_mn.cpu(),
            "pole_an": pole_an.cpu(),
            "ctrl_resid": ctrl_resid.cpu(),
            "ctrl_resid_a": ctrl_resid_a.cpu(),
            "ctrl_other_m": ctrl_other_m.cpu(),
            "ctrl_other_a": ctrl_other_a.cpu(),
            "own_paper_m": own_paper_m.cpu(),
            "head_norm": Dh.norm(dim=-1).cpu(),
            "head_along": (Dh * cur[:, None, :]).sum(-1).cpu(),
        }
        pairs.append(k)
        V_all.append(V.to(torch.float16).cpu())
        if i % 100 == 0:
            log.info(
                "%d/%d pair %d exact %.1e..%.1e  rot(min over L) %.3f  to_gen(L18) %.3f",
                i,
                len(files),
                k,
                float(exact.min()),
                float(exact.max()),
                float(rot.min()),
                float(to_gen[18]),
            )
    K = len(pairs)
    np.save(args.out / "pairs.npy", np.array(pairs))
    np.save(args.out / "V.npy", torch.stack(V_all).numpy())
    stack = {key: torch.stack([rows[k][key] for k in pairs]).numpy() for key in rows[pairs[0]]}
    np.savez(str(args.out / "curves.npz"), **stack)  # type: ignore[arg-type]
    st = np.array([steer.get(str(k), np.nan) for k in pairs])
    order = np.argsort(np.nan_to_num(st, nan=-1))
    log.info(
        "exactness: median %.2e p95 %.2e (all raw rows)", np.median(stack["exact"]), np.percentile(stack["exact"], 95)
    )
    log.info(
        "agreement with 2,000-story vectors, median cos by layer: %s",
        [round(float(np.median(stack["to_gen"][:, L])), 2) for L in range(0, B + 1, 3)],
    )

    def heat(ax, M, title, cmap="viridis", vmin=None, vmax=None, xlabel="block"):
        im = ax.imshow(M[order], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("concept (sorted by judged steerability, low → high)")
        plt.colorbar(im, ax=ax, fraction=0.03)

    nw = stack["norm_w"] / stack["norm_v"][:, B - 1 : B]
    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    heat(axes[0, 0], nw, "|w_b| / |v(35)|: how much each block writes", vmin=0, vmax=np.percentile(nw, 99))
    heat(
        axes[0, 1],
        stack["along"] / stack["norm_v"][:, B - 1 : B],
        "⟨w_b, v̂(b)⟩ / |v(35)|: push along the current direction",
        cmap="RdBu_r",
        vmin=-0.15,
        vmax=0.15,
    )
    heat(
        axes[0, 2],
        stack["perp"] / stack["norm_v"][:, B - 1 : B],
        "|w_b ⟂ v̂(b)| / |v(35)|: turning",
        vmin=0,
        vmax=np.percentile(stack["perp"] / stack["norm_v"][:, B - 1 : B], 99),
    )
    heat(
        axes[1, 0], stack["rot"], "cos(v̂(L+1), v̂(L)): per-block rotation", cmap="magma", vmin=0.6, vmax=1.0, xlabel="L"
    )
    heat(axes[1, 1], stack["to_last"], "cos(v̂(L), v̂(35))", cmap="magma", vmin=0, vmax=1, xlabel="L")
    heat(axes[1, 2], stack["to_gen"], "cos(v̂(L), 2,000-story v̂(L))", cmap="magma", vmin=0, vmax=1, xlabel="L")
    fig.suptitle(
        f"Formation of {K} concept vectors through Qwen3-8B (16 stories each): per-concept curves", fontsize=13
    )
    fig.tight_layout()
    fig.savefig(args.figs / "A1-heatmaps.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(22, 11))
    x = np.arange(B)
    q = lambda M, p: np.percentile(M, p, axis=0)  # noqa: E731

    def band(ax, M, color, label, xs=x):
        ax.plot(xs, np.median(M, 0), color=color, lw=2, label=label)
        ax.fill_between(xs, q(M, 25), q(M, 75), color=color, alpha=0.2)

    band(axes[0, 0], stack["norm_a"] / stack["norm_v"][:, B - 1 : B], "#2563eb", "attention |a_b|")
    band(axes[0, 0], stack["norm_m"] / stack["norm_v"][:, B - 1 : B], "#9ca3af", "MLP |m_b|")
    axes[0, 0].set_title("norm of each block's write, / |v(35)| (median, IQR)")
    band(axes[0, 1], stack["along_a"] / stack["norm_v"][:, B - 1 : B], "#2563eb", "attention along v̂(b)")
    band(axes[0, 1], stack["along_m"] / stack["norm_v"][:, B - 1 : B], "#9ca3af", "MLP along v̂(b)")
    axes[0, 1].axhline(0, color="k", lw=0.5)
    axes[0, 1].set_title("signed push along the current direction v̂(b)")
    band(axes[0, 2], stack["perp_a"] / stack["norm_v"][:, B - 1 : B], "#2563eb", "attention ⟂")
    band(axes[0, 2], stack["perp_m"] / stack["norm_v"][:, B - 1 : B], "#9ca3af", "MLP ⟂")
    axes[0, 2].set_title("orthogonal (turning) part")
    ratio_a = stack["along_a"] / np.sqrt(stack["along_a"] ** 2 + stack["perp_a"] ** 2 + 1e-12)
    ratio_m = stack["along_m"] / np.sqrt(stack["along_m"] ** 2 + stack["perp_m"] ** 2 + 1e-12)
    band(axes[1, 0], ratio_a, "#2563eb", "attention")
    band(axes[1, 0], ratio_m, "#9ca3af", "MLP")
    axes[1, 0].axhline(0, color="k", lw=0.5)
    axes[1, 0].set_title("cos(write_b, v̂(b)): does the block write along the direction it receives?")
    band(axes[1, 1], stack["rot"], "#dc2626", "cos(v̂(L+1), v̂(L))", xs=np.arange(B))
    band(axes[1, 1], stack["to_last"][:, :B], "#7c3aed", "cos(v̂(L), v̂(35))", xs=np.arange(B))
    axes[1, 1].set_title("rotation per block and alignment with the last raw layer")
    band(axes[1, 2], stack["to_gen"][:, :B], "#059669", "cos(v̂(L), 2,000-story v̂(L))", xs=np.arange(B))
    band(
        axes[1, 2],
        stack["norm_v"][:, :B] / stack["norm_v"][:, B - 1 : B],
        "#f59e0b",
        "|v(L)| / |v(35)|",
        xs=np.arange(B),
    )
    axes[1, 2].set_title("agreement with the paper's vectors; growth of |v(L)|")
    for ax in axes.ravel():
        ax.axvspan(19, 22, color="#f59e0b", alpha=0.12)
        ax.set_xlabel("block b (write into row b+1)  |  shaded: steering band")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.figs / "A2-curves.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    heat(axes[0, 0], ratio_m, "cos(MLP write m_b, v̂(b)) per concept", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    heat(axes[0, 1], ratio_a, "cos(attention write a_b, v̂(b)) per concept", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    nv = stack["norm_v"][:, B - 1 : B]
    band(axes[1, 0], stack["pole_m"][:, 0] / nv, "#dc2626", "concept-pole MLP write along v̂(b)")
    band(axes[1, 0], stack["pole_m"][:, 1] / nv, "#2563eb", "antagonist-pole MLP write along v̂(b)")
    band(axes[1, 0], stack["along_m"] / nv, "#111827", "difference (= D mlp along v̂(b))")
    axes[1, 0].axhline(0, color="k", lw=0.5)
    axes[1, 0].set_title("MLP write of each pole projected on the concept direction, / |v(35)|")
    band(axes[1, 1], stack["pole_a"][:, 0] / nv, "#dc2626", "concept-pole attention write along v̂(b)")
    band(axes[1, 1], stack["pole_a"][:, 1] / nv, "#2563eb", "antagonist-pole attention write along v̂(b)")
    band(axes[1, 1], stack["along_a"] / nv, "#111827", "difference")
    axes[1, 1].axhline(0, color="k", lw=0.5)
    axes[1, 1].set_title("attention write of each pole projected on the concept direction, / |v(35)|")
    for ax in axes[1]:
        ax.axvspan(19, 22, color="#f59e0b", alpha=0.12)
        ax.set_xlabel("block")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.figs / "A4-mlp-flip.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    band(axes[0], ratio_m, "#111827", "cos(MLP diff-write, own v̂(b), 16-story)")
    band(axes[0], stack["own_paper_m"], "#7c3aed", "cos(MLP diff-write, own paper vector v̂(b))")
    band(axes[0], stack["ctrl_other_m"], "#dc2626", "cos(MLP diff-write, another concept's paper vector)")
    band(axes[0], stack["ctrl_resid"], "#059669", "cos(MLP pole write, the pole's whole residual direction)")
    axes[0].set_title("MLP: concept-specific opposition vs generic controls")
    band(axes[1], ratio_a, "#111827", "cos(attention diff-write, own v̂(b))")
    band(axes[1], stack["ctrl_other_a"], "#dc2626", "cos(attention diff-write, another concept's vector)")
    band(axes[1], stack["ctrl_resid_a"], "#059669", "cos(attention pole write, the pole's whole residual direction)")
    axes[1].set_title("attention: same controls")
    for ax in axes:
        ax.axhline(0, color="k", lw=0.5)
        ax.axvspan(19, 22, color="#f59e0b", alpha=0.12)
        ax.set_xlabel("block")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.figs / "A6-controls.png", dpi=170)
    plt.close(fig)

    ha = stack["head_along"]
    hn = stack["head_norm"]
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    m = np.median(ha / stack["norm_v"][:, B - 1 : B, None], 0)
    im = axes[0].imshow(m.T, aspect="auto", cmap="RdBu_r", vmin=-0.02, vmax=0.02, interpolation="nearest")
    axes[0].set_title("median over concepts of ⟨D head[b,h], v̂(b)⟩ / |v(35)|")
    axes[0].set_xlabel("block")
    axes[0].set_ylabel("head")
    plt.colorbar(im, ax=axes[0], fraction=0.03)
    pos = (ha > 0).mean(0)
    im = axes[1].imshow(pos.T, aspect="auto", cmap="coolwarm", vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title("fraction of concepts for which the head pushes along v̂(b) (0.5 = agnostic)")
    axes[1].set_xlabel("block")
    plt.colorbar(im, ax=axes[1], fraction=0.03)
    coh = np.abs(ha).mean(0) / hn.mean(0).clip(1e-8)
    im = axes[2].imshow(
        coh.T, aspect="auto", cmap="viridis", vmin=0, vmax=np.percentile(coh, 99), interpolation="nearest"
    )
    axes[2].set_title(
        "mean |⟨D head, v̂(b)⟩| / mean |D head|: share of the head's write that is along the current direction"
    )
    axes[2].set_xlabel("block")
    plt.colorbar(im, ax=axes[2], fraction=0.03)
    fig.tight_layout()
    fig.savefig(args.figs / "A3-heads.png", dpi=170)
    plt.close(fig)
    log.info("wrote %s and %s", args.out, args.figs)


if __name__ == "__main__":
    main()
