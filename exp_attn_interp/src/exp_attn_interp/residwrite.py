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

from exp_attn_interp.common import load_model  # noqa: E402
from exp_attn_interp.config import INSTRUCT_MODEL, vector_tag_for  # noqa: E402
from exp_attn_interp.vectors import load_diff_vectors  # noqa: E402

log = logging.getLogger("residwrite")


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def split(X, r=16):
    m = X.mean(0)
    mh = m / m.norm()
    Xc = X - m
    _, _, Wt = torch.linalg.svd(Xc, full_matrices=False)
    P = Wt[:r]
    st = (Xc @ P.T) @ P + (X @ mh)[:, None] * mh[None]
    return st, X - st


@torch.inference_mode()
def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    parser.add_argument("--vdecomp", type=Path, default=Path("work/vdecomp"))
    parser.add_argument("--every", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("work/residwrite.npz"))
    parser.add_argument("--figs", type=Path, default=Path("figures/field"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.figs.mkdir(parents=True, exist_ok=True)
    model, _tok = load_model(offline=True, model_name=args.model)
    dev = model.device
    U = model.lm_head.weight.to(torch.float32) * model.model.norm.weight.to(torch.float32)[None, :]
    E = model.model.embed_tokens.weight.to(torch.float32)
    mk = json.loads(Path("data/markers_presence.json").read_text())
    V = load_diff_vectors(device=str(dev), tag=vector_tag_for(args.model))
    K = V.shape[1]
    ug = torch.zeros(K, V.shape[2], device=dev)
    eg = torch.zeros_like(ug)
    ok = torch.zeros(K, dtype=torch.bool)
    for k in range(K):
        ic, ia = mk["concept"].get(str(k), []), mk["antagonist"].get(str(k), [])
        if len(ic) >= 2 and len(ia) >= 2:
            ug[k] = unit(U[ic].mean(0) - U[ia].mean(0))
            eg[k] = unit(E[ic].mean(0) - E[ia].mean(0))
            ok[k] = True
    del U, E
    _, ru = split(ug)
    _, re_ = split(eg)
    _, r35 = split(V[35])
    ru, re_, r35 = unit(ru), unit(re_), unit(r35)
    files = sorted(args.vdecomp.glob("pair-*.safetensors"))[:: args.every]
    rows: dict[str, list[np.ndarray]] = {
        k: [] for k in ("a_r35", "m_r35", "a_ru", "m_ru", "a_re", "m_re", "a_ru_other", "m_ru_other", "a_st", "m_st")
    }
    _, s35 = split(V[35])
    s35 = unit(s35)
    for i, f in enumerate(files):
        k = int(f.stem.split("-")[1])
        if not ok[k]:
            continue
        t = load_file(str(f))
        A = unit((t["pole_head"][0] - t["pole_head"][1]).to(dev, torch.float32).sum(1))
        M = unit((t["pole_mlp"][0] - t["pole_mlp"][1]).to(dev, torch.float32))
        other = (k + K // 2) % K
        for comp, X in (("a", A), ("m", M)):
            rows[f"{comp}_r35"].append((X @ r35[k]).cpu().numpy())
            rows[f"{comp}_ru"].append((X @ ru[k]).cpu().numpy())
            rows[f"{comp}_re"].append((X @ re_[k]).cpu().numpy())
            rows[f"{comp}_ru_other"].append((X @ ru[other]).cpu().numpy())
            rows[f"{comp}_st"].append((X @ s35[k]).cpu().numpy())
        if i % 50 == 0:
            log.info("%d/%d", i, len(files))
    R = {key: np.stack(v) for key, v in rows.items()}
    np.savez(str(args.out), **R)  # type: ignore[arg-type]
    B = R["a_r35"].shape[1]
    for key in R:
        log.info("%-11s median by block: %s", key, [round(float(np.median(R[key][:, b])), 3) for b in range(0, B, 2)])
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    x = np.arange(B)
    for ax, (ka, km, ttl) in zip(
        axes,
        (
            ("a_r35", "m_r35", "cos with the specific layer of v(35)"),
            ("a_ru", "m_ru", "cos with the concept's own output-form word residual (u_gap)"),
            ("a_re", "m_re", "cos with the concept's own input-form word residual (e_gap)"),
        ),
    ):
        ax.plot(x, np.median(R[ka], 0), color="#2563eb", lw=2, label="attention write")
        ax.plot(x, np.median(R[km], 0), color="#9ca3af", lw=2, label="MLP write")
        if ka == "a_ru":
            ax.plot(
                x,
                np.median(R["a_ru_other"], 0),
                color="#2563eb",
                lw=1,
                ls=":",
                label="attention, other concept's words",
            )
            ax.plot(x, np.median(R["m_ru_other"], 0), color="#9ca3af", lw=1, ls=":", label="MLP, other concept's words")
        ax.axhline(0, color="k", lw=0.5)
        ax.axvspan(19, 22, color="#f59e0b", alpha=0.12)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("block")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.figs / "F2-residwrite.png", dpi=170)
    log.info("wrote %s", args.figs)


if __name__ == "__main__":
    main()
