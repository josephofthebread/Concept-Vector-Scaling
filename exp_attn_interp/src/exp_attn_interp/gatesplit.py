from __future__ import annotations

import logging
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import matplotlib
import numpy as np
import pyarrow.parquet as pq
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from exp_attn_interp.common import load_model, story_span  # noqa: E402
from exp_attn_interp.config import GENDATA_ROOT, INSTRUCT_MODEL, vector_tag_for  # noqa: E402
from exp_attn_interp.decomp import POOL_MIN, POOL_SKIP  # noqa: E402
from exp_attn_interp.mlpgain import PreMlpCatcher, dsilu, silu  # noqa: E402
from exp_attn_interp.vectors import load_diff_vectors  # noqa: E402
from exp_attn_interp.write import story_ids  # noqa: E402

log = logging.getLogger("gatesplit")
PARTS = ("open_up", "open_gate", "closed_up", "closed_gate")


@torch.inference_mode()
def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    parser.add_argument("--every", type=int, default=32)
    parser.add_argument("--per-pole", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("work/gatesplit"))
    parser.add_argument("--figs", type=Path, default=Path("figures/gain"))
    parser.add_argument("--band", default="19,22")
    parser.add_argument("--offline", action=BooleanOptionalAction, default=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    args.figs.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(offline=args.offline, model_name=args.model)
    pre = PreMlpCatcher(model.model)
    vectors = load_diff_vectors(device=str(model.device), tag=vector_tag_for(args.model))
    B = len(model.model.layers)
    hid = model.config.hidden_size
    dev = model.device
    tbl = pq.read_table(
        GENDATA_ROOT / "data" / "semantic.parquet",
        columns=["pair_id", "concept_text", "antagonist_text"],
        filters=[("language_id", "==", 0)],
    )
    by_pair: dict[int, list[tuple[str, str]]] = {}
    for p, c, a in zip(tbl["pair_id"].to_pylist(), tbl["concept_text"].to_pylist(), tbl["antagonist_text"].to_pylist()):
        by_pair.setdefault(int(p), []).append((c, a))
    pairs = sorted(by_pair)[:: args.every]
    gen = torch.Generator(device="cpu").manual_seed(0)
    R = torch.zeros(len(pairs), 2, B, 4, device=dev)
    frac_open = torch.zeros(len(pairs), B, device=dev)
    for i, k in enumerate(pairs):
        rnd = torch.randn(hid, generator=gen).to(dev)
        rnd = rnd / rnd.norm()
        n = 0
        for c, a in by_pair[k][: args.per_pole]:
            for text in (c, a):
                ids = story_ids(tokenizer, text)
                s0, s1 = story_span(tokenizer, ids, text)
                if (s1 - s0 - POOL_SKIP) >= POOL_MIN:
                    s0 += POOL_SKIP
                model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
                for b in range(B):
                    blk = model.model.layers[b]
                    xb = pre.x[b]
                    assert xb is not None
                    x = xb[0, s0:s1].to(torch.float32)
                    gamma = blk.post_attention_layernorm.weight.to(torch.float32)
                    eps = float(getattr(blk.post_attention_layernorm, "variance_epsilon", 1e-6))
                    Wg = blk.mlp.gate_proj.weight.to(torch.float32)
                    Wu = blk.mlp.up_proj.weight.to(torch.float32)
                    Wd = blk.mlp.down_proj.weight.to(torch.float32)
                    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
                    xn = x * rms * gamma
                    g, u = xn @ Wg.T, xn @ Wu.T
                    open_ = (g > 0).float()
                    frac_open[i, b] += open_.mean()
                    dirs = torch.stack([vectors[b, k], rnd])
                    dt = (gamma * dirs)[:, None, :] * rms[None]
                    pg = torch.einsum("dsh,ih->dsi", dt, Wg)
                    pu = torch.einsum("dsh,ih->dsi", dt, Wu)
                    wd = dirs @ Wd
                    up = silu(g)[None] * pu * wd[:, None, :]
                    gate = (dsilu(g) * u)[None] * pg * wd[:, None, :]
                    R[i, :, b, 0] += (up * open_[None]).sum(-1).mean(-1)
                    R[i, :, b, 1] += (gate * open_[None]).sum(-1).mean(-1)
                    R[i, :, b, 2] += (up * (1 - open_)[None]).sum(-1).mean(-1)
                    R[i, :, b, 3] += (gate * (1 - open_)[None]).sum(-1).mean(-1)
                n += 1
        R[i] /= n
        frac_open[i] /= n
        if i % 8 == 0:
            log.info(
                "%d/%d pair %d: along v_k(b) at b=10: open-up %.3f open-gate %.3f closed-up %.3f closed-gate %.3f; at b=24: %s",
                i,
                len(pairs),
                k,
                *R[i, 0, 10].tolist(),
                [round(v, 3) for v in R[i, 0, 24].tolist()],
            )
    Rn = R.cpu().numpy()
    fo = frac_open.cpu().numpy()
    np.savez(str(args.out / "gatesplit.npz"), R=Rn, frac_open=fo, pairs=np.array(pairs))
    for d, nm in enumerate(("v_k(b)", "random")):
        for p, part in enumerate(PARTS):
            log.info(
                "%-7s %-11s median by block: %s",
                nm,
                part,
                [round(float(np.median(Rn[:, d, b, p])), 3) for b in range(0, B, 2)],
            )
    log.info("fraction of open neurons by block: %s", [round(float(np.median(fo[:, b])), 3) for b in range(0, B, 2)])
    b0, b1 = (int(x) for x in args.band.split(","))
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    xs = np.arange(B)
    for d, nm in enumerate(("v_k(b)", "random")):
        ax = axes[d]
        for p, (part, col) in enumerate(zip(PARTS, ("#dc2626", "#f59e0b", "#2563eb", "#7c3aed"))):
            ax.plot(xs, np.median(Rn[:, d, :, p], 0), color=col, lw=2, label=part)
            ax.fill_between(
                xs, np.percentile(Rn[:, d, :, p], 25, 0), np.percentile(Rn[:, d, :, p], 75, 0), color=col, alpha=0.15
            )
        ax.plot(xs, np.median(Rn[:, d].sum(-1), 0), color="k", lw=1, ls="--", label="total linear gain")
        ax.axhline(0, color="k", lw=0.5)
        ax.axvspan(b0, b1, color="#f59e0b", alpha=0.12)
        ax.set_title(f"linear MLP gain along {nm} by neuron state and path (median over {len(pairs)} concepts)")
        ax.set_xlabel("block b")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.figs / "G4-gatesplit.png", dpi=170)
    log.info("wrote %s", args.figs)


if __name__ == "__main__":
    main()
