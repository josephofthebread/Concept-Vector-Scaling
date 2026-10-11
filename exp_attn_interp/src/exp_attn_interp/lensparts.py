from __future__ import annotations

import json
import logging
import unicodedata
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from exp_attn_interp.common import load_model
from exp_attn_interp.config import INSTRUCT_MODEL, vector_tag_for
from exp_attn_interp.vectors import load_diff_vectors

log = logging.getLogger("lensparts")


@torch.inference_mode()
def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--out", type=Path, default=Path("work/lensparts.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model, tok = load_model(offline=True, model_name=args.model)
    dev = model.device
    U = model.lm_head.weight.to(torch.float32) * model.model.norm.weight.to(torch.float32)[None, :]
    vocab = U.shape[0]
    han = torch.tensor(
        [any("CJK" in unicodedata.name(ch, "") for ch in tok.decode([i])) for i in range(vocab)], device=dev
    )
    mk = json.loads(Path("data/markers_presence.json").read_text())
    V = load_diff_vectors(device=str(dev), tag=vector_tag_for(args.model))
    K = V.shape[1]

    def parts(L):
        X = V[L]
        m = X.mean(0)
        mh = m / m.norm()
        Xc = X - m
        _, _, Wt = torch.linalg.svd(Xc, full_matrices=False)
        P = Wt[: args.rank]
        stance = (Xc @ P.T) @ P + (X @ mh)[:, None] * mh[None]
        return stance, X - stance

    def lens(Xs, ic, ia):
        Xu = Xs / Xs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        Pz = Xu @ U.T
        z = (Pz - Pz.mean(-1, keepdim=True)) / Pz.std(-1, keepdim=True).clamp_min(1e-6)
        gap = torch.stack([z[i, ic[i]].mean() - z[i, ia[i]].mean() for i in range(len(ic))])
        top = Pz.topk(60, dim=-1).indices
        return gap.cpu().numpy(), han[top].float().mean(-1).cpu().numpy(), Pz

    ks = [k for k in range(K) if len(mk["concept"].get(str(k), [])) >= 2 and len(mk["antagonist"].get(str(k), [])) >= 2]
    ic = [mk["concept"][str(k)] for k in ks]
    ia = [mk["antagonist"][str(k)] for k in ks]
    out: dict[str, dict] = {}
    X22, X35 = V[22], V[35]
    Xc = X22 - X22.mean(0)
    _, _, W = torch.linalg.svd(Xc, full_matrices=False)
    P64 = W[:64]
    Z = X22 @ P64.T
    lam = 1e-3 * (Z**2).sum() / 64
    M = torch.linalg.solve(Z.T @ Z + lam * torch.eye(64, device=dev), Z.T @ X35)
    T = lambda Q: (Q @ P64.T) @ M  # noqa: E731
    for L in (22, 35):
        st, rs = parts(L)
        for name, Xs in (("whole", V[L]), ("stance", st), ("residual", rs)):
            g, h, Pz = lens(Xs[ks], ic, ia)
            out[f"L{L}_{name}"] = {
                "marker_gap_median": float(np.median(g)),
                "marker_gap_p25": float(np.percentile(g, 25)),
                "han_share": float(np.median(h)),
                "energy_share": float(((Xs[ks] ** 2).sum(-1) / (V[L][ks] ** 2).sum(-1)).median()),
                "per_concept": {
                    "pairs": [int(k) for k in ks],
                    "gap": [float(x) for x in g],
                    "han": [float(x) for x in h],
                    "energy": [float(x) for x in ((Xs[ks] ** 2).sum(-1) / (V[L][ks] ** 2).sum(-1)).tolist()],
                },
            }
            log.info(
                "row %d %-8s: marker z-gap median %.2f (p25 %.2f); han share %.2f; energy share %.2f",
                L,
                name,
                np.median(g),
                np.percentile(g, 25),
                np.median(h),
                out[f"L{L}_{name}"]["energy_share"],
            )
            if L == 35:
                ex = {}
                for i in (0, 100, 300):
                    ex[int(ks[i])] = {
                        "top": [tok.decode([t]) for t in Pz[i].topk(12).indices.tolist()],
                        "bottom": [tok.decode([t]) for t in (-Pz[i]).topk(8).indices.tolist()],
                    }
                out[f"L{L}_{name}"]["examples"] = ex
    st22, rs22 = parts(22)
    for name, Xs in (("stance22->35", T(st22)), ("residual22->35", T(rs22))):
        g, h, _ = lens(Xs[ks], ic, ia)
        out[name] = {
            "marker_gap_median": float(np.median(g)),
            "han_share": float(np.median(h)),
            "cos_with_v35": float(torch.nn.functional.cosine_similarity(Xs[ks], V[35][ks], dim=-1).median()),
        }
        log.info(
            "%s: marker z-gap median %.2f; han %.2f; cos with v(35) %.3f",
            name,
            np.median(g),
            np.median(h),
            out[name]["cos_with_v35"],
        )
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
