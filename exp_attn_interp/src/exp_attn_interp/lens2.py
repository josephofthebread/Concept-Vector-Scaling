from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from exp_attn_interp.common import load_model
from exp_attn_interp.config import INSTRUCT_MODEL

log = logging.getLogger("lens2")


def unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


@torch.inference_mode()
def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    parser.add_argument("--vdecomp", type=Path, default=Path("work/vdecomp"))
    parser.add_argument("--markers", type=Path, default=Path("data/markers_presence.json"))
    parser.add_argument(
        "--mlpgain", type=Path, default=Path("work/mlpgain"), help="optional: lens the MLP response vectors too"
    )
    parser.add_argument("--out", type=Path, default=Path("work/lens2"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    model, _tok = load_model(offline=True, model_name=args.model)
    dev = model.device
    U = model.lm_head.weight.to(torch.float32) * model.model.norm.weight.to(torch.float32)[None, :]
    E = model.model.embed_tokens.weight.to(torch.float32)
    assert not model.config.tie_word_embeddings
    mk = json.loads(args.markers.read_text())
    files = sorted(args.vdecomp.glob("pair-*.safetensors"))
    if args.limit:
        files = files[: args.limit]

    def zgap(P: torch.Tensor, ic, ia) -> torch.Tensor:
        z = (P - P.mean(-1, keepdim=True)) / P.std(-1, keepdim=True).clamp_min(1e-6)
        return z[:, ic].mean(-1) - z[:, ia].mean(-1)

    keys = ("gapU", "gapE", "cosU", "cosE", "enU", "enE")
    comps = ("a", "m", "v", "w")
    rows: dict[str, list[torch.Tensor]] = {f"{key}_{c}": [] for key in keys for c in comps}
    rows.update({"cos_eu": [], "resp_cosE": [], "resp_cosU": [], "resp_gain": []})
    pairs = []
    for i, f in enumerate(files):
        k = int(f.stem.split("-")[1])
        ic, ia = mk["concept"].get(str(k), []), mk["antagonist"].get(str(k), [])
        if len(ic) < 2 or len(ia) < 2:
            continue
        t = load_file(str(f))
        ph = t["pole_head"].to(dev, torch.float32)
        pm = t["pole_mlp"].to(dev, torch.float32)
        pe = t["pole_embed"].to(dev, torch.float32)
        A = (ph[0] - ph[1]).sum(1)
        M = pm[0] - pm[1]
        De = pe[0] - pe[1]
        W = A + M
        V = torch.cat([De[None], De[None] + torch.cumsum(W, 0)], 0)
        e_gap = unit(E[ic].mean(0) - E[ia].mean(0))
        u_gap = unit(U[ic].mean(0) - U[ia].mean(0))
        both = sorted(set(ic) | set(ia))
        QE, _ = torch.linalg.qr(E[both].T)
        QU, _ = torch.linalg.qr(U[both].T)
        for c, X in (("a", A), ("m", M), ("v", V), ("w", W)):
            Xu = unit(X)
            rows[f"gapU_{c}"].append(zgap(Xu @ U.T, ic, ia).cpu())
            rows[f"gapE_{c}"].append(zgap(Xu @ E.T, ic, ia).cpu())
            rows[f"cosU_{c}"].append((Xu @ u_gap).cpu())
            rows[f"cosE_{c}"].append((Xu @ e_gap).cpu())
            rows[f"enU_{c}"].append(((Xu @ QU) ** 2).sum(-1).cpu())
            rows[f"enE_{c}"].append(((Xu @ QE) ** 2).sum(-1).cpu())
        rows["cos_eu"].append((e_gap @ u_gap).cpu()[None])
        g = args.mlpgain / f.name
        if g.exists():
            r = load_file(str(g))
            resp = r["resp"][0, :, 0].to(dev, torch.float32)
            ru = unit(resp)
            rows["resp_cosE"].append((ru @ e_gap).cpu())
            rows["resp_cosU"].append((ru @ u_gap).cpu())
            rows["resp_gain"].append(r["gain"][:2, :, 0, 0].mean(0))
        else:
            for key in ("resp_cosE", "resp_cosU", "resp_gain"):
                rows[key].append(torch.full((M.shape[0],), float("nan")))
        pairs.append(k)
        if i % 100 == 0:
            log.info(
                "%d/%d pair %d: cos(e_gap,u_gap) %.2f; MLP write cosE by block %s; cosU %s",
                i,
                len(files),
                k,
                float(e_gap @ u_gap),
                [round(float(x), 2) for x in rows["cosE_m"][-1][::4]],
                [round(float(x), 2) for x in rows["cosU_m"][-1][::4]],
            )
    R = {key: torch.stack(val).numpy() for key, val in rows.items()}
    np.savez(str(args.out / "lens2.npz"), pairs=np.array(pairs), **R)  # type: ignore[arg-type]
    B = R["cosE_m"].shape[1]
    for key in ("cosE_a", "cosE_m", "cosU_a", "cosU_m"):
        log.info("median %s by block: %s", key, [round(float(np.median(R[key][:, b])), 3) for b in range(0, B, 2)])
    log.info("median cos(e_gap, u_gap): %.3f", float(np.median(R["cos_eu"])))
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
