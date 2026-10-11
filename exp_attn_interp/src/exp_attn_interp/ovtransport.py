from __future__ import annotations

import json
import logging
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from exp_attn_interp.common import load_model  # noqa: E402
from exp_attn_interp.config import INSTRUCT_MODEL, vector_tag_for  # noqa: E402
from exp_attn_interp.vectors import load_diff_vectors  # noqa: E402

log = logging.getLogger("ovtransport")
TRIGGER = Path("../exp_trigger_prompts")
ROWS = (12, 16, 18, 19, 20, 22, 23, 24, 26, 28)


@torch.inference_mode()
def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    parser.add_argument("--every", type=int, default=4)
    parser.add_argument("--rows", default=",".join(map(str, ROWS)))
    parser.add_argument("--out", type=Path, default=Path("work/ovtransport.npz"))
    parser.add_argument("--figs", type=Path, default=Path("figures/dose"))
    parser.add_argument("--offline", action=BooleanOptionalAction, default=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.figs.mkdir(parents=True, exist_ok=True)
    rows = [int(x) for x in args.rows.split(",")]
    model, _tok = load_model(offline=args.offline, model_name=args.model)
    vectors = load_diff_vectors(device=str(model.device), tag=vector_tag_for(args.model))
    B = len(model.model.layers)
    last = B - 1
    K = vectors.shape[1]
    ks = list(range(0, K, args.every))
    cfg = model.config
    H, kv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", cfg.hidden_size // H)
    group = H // kv
    norms: dict[int, float] = {}
    tag = args.model.replace("/", "_")
    for f in (f"norms-{tag}.json", f"norms-{tag}-1217.json"):
        p = TRIGGER / "work" / "steer" / f
        if p.exists():
            norms.update({int(k): float(v) for k, v in json.loads(p.read_text())["norms"].items()})
    xs = sorted(norms)
    rms_b = {
        b: float(np.exp(np.interp(b, xs, [np.log(norms[x]) for x in xs]))) / np.sqrt(cfg.hidden_size) for b in range(B)
    }
    T = np.zeros((len(rows), len(ks), B, H), dtype=np.float32)
    for i, L in enumerate(rows):
        vL = vectors[L, ks]
        v35 = vectors[last, ks]
        for b in range(L, B):
            blk = model.model.layers[b]
            gamma = blk.input_layernorm.weight.to(torch.float32)
            Wv = blk.self_attn.v_proj.weight.to(torch.float32)
            Wo = blk.self_attn.o_proj.weight.to(torch.float32).view(-1, H, hd)
            x = (vL * gamma) / rms_b[b]
            vals = (x @ Wv.T).view(len(ks), kv, hd)
            vh = vals[:, torch.arange(H, device=vals.device) // group]
            out = torch.einsum("khd,ohd->kho", vh, Wo)
            T[i, :, b] = torch.einsum("kho,ko->kh", out, v35).cpu().numpy()
        log.info(
            "row %d: OV-transport of v(L) to v(35), summed over heads above L: median %.3f; top heads %s",
            L,
            float(np.median(T[i].sum((1, 2)))),
            [
                (int(b), int(h), round(float(v), 3))
                for b, h, v in sorted(
                    ((b, h, np.median(T[i, :, b, h])) for b in range(B) for h in range(H)), key=lambda t: -t[2]
                )[:6]
            ],
        )
    np.savez(str(args.out), T=T, rows=np.array(rows), pairs=np.array(ks))
    tot = np.median(T.sum((2, 3)), 1)
    pos = np.median(np.clip(T, 0, None).sum((2, 3)), 1)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    axes[0].plot(
        rows, tot, marker="o", lw=2, color="#111827", label="sum over heads above L of the OV-transport (signed)"
    )
    axes[0].plot(rows, pos, marker="o", lw=2, color="#2563eb", ls="--", label="positive part only")
    axes[0].axvspan(19, 22, color="#f59e0b", alpha=0.12)
    axes[0].set_xlabel("injection row L")
    axes[0].set_title(
        "weight-level capacity of the context path: how much of v(L) the heads above L can carry to v(35)"
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    i22 = rows.index(22)
    hm = np.median(T[i22], 0).T
    im = axes[1].imshow(
        hm, aspect="auto", cmap="RdBu_r", vmin=-np.abs(hm).max(), vmax=np.abs(hm).max(), interpolation="nearest"
    )
    axes[1].set_title("per-head OV-transport of v(22) onto v(35) (median over concepts)")
    axes[1].set_xlabel("block")
    axes[1].set_ylabel("head")
    plt.colorbar(im, ax=axes[1], fraction=0.03)
    fig.tight_layout()
    fig.savefig(args.figs / "S5-ovtransport.png", dpi=170)
    log.info("wrote %s", args.figs)


if __name__ == "__main__":
    main()
