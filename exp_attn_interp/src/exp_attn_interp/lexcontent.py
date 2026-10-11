from __future__ import annotations

import json
import logging
import unicodedata
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from exp_attn_interp.common import load_model  # noqa: E402
from exp_attn_interp.config import INSTRUCT_MODEL  # noqa: E402

log = logging.getLogger("lexcontent")


@torch.inference_mode()
def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--vdecomp", type=Path, default=Path("work/vdecomp"))
    parser.add_argument("--markers", type=Path, default=Path("data/markers_presence.json"))
    parser.add_argument("--out", type=Path, default=Path("work/lexcontent"))
    parser.add_argument("--figs", type=Path, default=Path("figures/lexcontent"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    args.figs.mkdir(parents=True, exist_ok=True)
    model, tok = load_model(offline=True, model_name=INSTRUCT_MODEL)
    dev = model.device
    W_U = model.lm_head.weight.to(torch.float32)
    gamma = model.model.norm.weight.to(torch.float32)
    Wg = W_U * gamma[None, :]
    vocab = W_U.shape[0]
    han_tok = AutoTokenizer.from_pretrained(INSTRUCT_MODEL)
    han = torch.zeros(vocab, dtype=torch.bool)
    for i in range(vocab):
        s = han_tok.decode([i])
        han[i] = any("CJK" in unicodedata.name(ch, "") for ch in s)
    han = han.to(dev)
    mk = json.loads(args.markers.read_text())
    steer = json.loads(Path("work/steerability.json").read_text())["qwen"]
    files = sorted(args.vdecomp.glob("pair-*.safetensors"))
    if args.limit:
        files = files[: args.limit]

    def lens_stats(X: torch.Tensor, ids_c, ids_a):
        """X [n, hid] -> marker gap z [n], han share of top-60 [n]"""
        P = (X / X.norm(dim=-1, keepdim=True).clamp_min(1e-8)) @ Wg.T
        z = (P - P.mean(-1, keepdim=True)) / P.std(-1, keepdim=True).clamp_min(1e-6)
        gap = z[:, ids_c].mean(-1) - z[:, ids_a].mean(-1)
        top = P.topk(60, dim=-1).indices
        hs = han[top].float().mean(-1)
        return gap.cpu(), hs.cpu()

    rows: dict[str, list[torch.Tensor]] = {
        "gap_a": [],
        "gap_m": [],
        "gap_v": [],
        "han_a": [],
        "han_m": [],
        "han_v": [],
        "gap_w": [],
    }
    pairs = []
    for i, f in enumerate(files):
        t = load_file(str(f))
        k = int(f.stem.split("-")[1])
        ids_c, ids_a = mk["concept"][str(k)], mk["antagonist"][str(k)]
        if len(ids_c) < 2 or len(ids_a) < 2:
            continue
        ph = t["pole_head"].to(dev, torch.float32)
        pm = t["pole_mlp"].to(dev, torch.float32)
        pe = t["pole_embed"].to(dev, torch.float32)
        B = ph.shape[1]
        A = (ph[0] - ph[1]).sum(1)
        M = pm[0] - pm[1]
        De = pe[0] - pe[1]
        W = A + M
        V = torch.cat([De[None], De[None] + torch.cumsum(W, 0)], 0)
        ga, ha = lens_stats(A, ids_c, ids_a)
        gm, hm = lens_stats(M, ids_c, ids_a)
        gv, hv = lens_stats(V, ids_c, ids_a)
        gw, _ = lens_stats(W, ids_c, ids_a)
        for key, val in (
            ("gap_a", ga),
            ("gap_m", gm),
            ("gap_v", gv),
            ("han_a", ha),
            ("han_m", hm),
            ("han_v", hv),
            ("gap_w", gw),
        ):
            rows[key].append(val)
        pairs.append(k)
        if i % 100 == 0:
            log.info(
                "%d/%d pair %d: gap of v(L) at L=6/18/35: %.2f / %.2f / %.2f; han share of v(18) top-60 %.2f",
                i,
                len(files),
                k,
                float(gv[6]),
                float(gv[18]),
                float(gv[35]),
                float(hv[18]),
            )
    R = {key: torch.stack(val).numpy() for key, val in rows.items()}
    np.savez(str(args.out / "lens.npz"), pairs=np.array(pairs), **R)  # type: ignore[arg-type]
    st = np.array([steer.get(str(k), np.nan) for k in pairs])
    order = np.argsort(np.nan_to_num(st, nan=-1))
    B = R["gap_a"].shape[1]
    log.info(
        "median marker gap (z) of v(L): %s", [round(float(np.median(R["gap_v"][:, L])), 2) for L in range(0, B + 1, 3)]
    )
    log.info(
        "median marker gap of block writes, attention: %s",
        [round(float(np.median(R["gap_a"][:, b])), 2) for b in range(0, B, 3)],
    )
    log.info(
        "median marker gap of block writes, MLP: %s",
        [round(float(np.median(R["gap_m"][:, b])), 2) for b in range(0, B, 3)],
    )

    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    x = np.arange(B)

    def band(ax, Mx, color, label, xs):
        ax.plot(xs, np.median(Mx, 0), color=color, lw=2, label=label)
        ax.fill_between(xs, np.percentile(Mx, 25, 0), np.percentile(Mx, 75, 0), color=color, alpha=0.2)

    band(axes[0, 0], R["gap_a"], "#2563eb", "attention write a_b", x)
    band(axes[0, 0], R["gap_m"], "#9ca3af", "MLP write m_b", x)
    band(axes[0, 0], R["gap_w"], "#111827", "block write w_b", x)
    axes[0, 0].axhline(0, color="k", lw=0.5)
    axes[0, 0].set_title("logit-lens marker gap (z over vocabulary) of each block's write direction")
    band(axes[0, 1], R["gap_v"], "#7c3aed", "v(L)", np.arange(B + 1))
    axes[0, 1].set_title("marker gap of the running vector v(L)")
    band(axes[0, 2], R["han_a"], "#2563eb", "attention write", x)
    band(axes[0, 2], R["han_m"], "#9ca3af", "MLP write", x)
    band(axes[0, 2], R["han_v"][:, :B], "#7c3aed", "v(L)", x)
    axes[0, 2].set_title("Han-script share of the top-60 promoted tokens")
    for ax in axes[0]:
        ax.axvspan(19, 22, color="#f59e0b", alpha=0.12)
        ax.set_xlabel("block / layer")
        ax.legend(fontsize=8)

    def heat(ax, Mx, title, cmap="RdBu_r", vmin=-3, vmax=3):
        im = ax.imshow(Mx[order], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("block")
        ax.set_ylabel("concept (sorted by steerability)")
        plt.colorbar(im, ax=ax, fraction=0.03)

    heat(axes[1, 0], R["gap_a"], "marker gap of attention writes, per concept")
    heat(axes[1, 1], R["gap_m"], "marker gap of MLP writes, per concept")
    heat(axes[1, 2], R["gap_v"], "marker gap of v(L), per concept", vmin=-6, vmax=6)
    fig.tight_layout()
    fig.savefig(args.figs / "C1-lexical.png", dpi=170)
    plt.close(fig)
    log.info("wrote %s", args.figs)


if __name__ == "__main__":
    main()
