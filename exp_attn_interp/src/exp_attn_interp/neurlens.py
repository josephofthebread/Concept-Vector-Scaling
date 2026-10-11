from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from exp_attn_interp.common import load_model
from exp_attn_interp.config import INSTRUCT_MODEL, vector_tag_for
from exp_attn_interp.vectors import load_diff_vectors

log = logging.getLogger("neurlens")


@torch.inference_mode()
def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    parser.add_argument("--top", type=Path, default=Path("work/gainan/top_neurons.json"))
    parser.add_argument("--mlpgain", type=Path, default=Path("work/mlpgain"))
    parser.add_argument("--markers", type=Path, default=Path("data/markers_presence.json"))
    parser.add_argument("--out", type=Path, default=Path("work/neurlens"))
    parser.add_argument("--n-per-block", type=int, default=6)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    model, tok = load_model(offline=True, model_name=args.model)
    dev = model.device
    vectors = load_diff_vectors(device=str(dev), tag=vector_tag_for(args.model))
    U = model.lm_head.weight.to(torch.float32) * model.model.norm.weight.to(torch.float32)[None, :]
    E = model.model.embed_tokens.weight.to(torch.float32)
    top = json.loads(args.top.read_text())
    mk = json.loads(args.markers.read_text())
    files = sorted(args.mlpgain.glob("pair-*.safetensors"))
    pairs = [int(f.stem.split("-")[1]) for f in files]
    Kc = torch.tensor(pairs, device=dev)
    B = len(model.model.layers)
    neur = np.stack([safe_open(str(f), framework="np").get_tensor("neuron")[0] for f in files]).astype(np.float32)
    e_gap = torch.zeros(len(pairs), E.shape[1], device=dev)
    u_gap = torch.zeros_like(e_gap)
    for i, k in enumerate(pairs):
        ic, ia = mk["concept"].get(str(k), []), mk["antagonist"].get(str(k), [])
        if len(ic) >= 2 and len(ia) >= 2:
            e_gap[i] = torch.nn.functional.normalize(E[ic].mean(0) - E[ia].mean(0), dim=0)
            u_gap[i] = torch.nn.functional.normalize(U[ic].mean(0) - U[ia].mean(0), dim=0)
    report: dict[str, list] = {}
    stats: dict[str, list[float]] = {
        "rw_up_neg": [],
        "rw_gate_neg": [],
        "rw_up_pos": [],
        "rw_gate_pos": [],
        "read_up_abs_all": [],
        "cosE_neg": [],
        "cosU_neg": [],
        "cosE_pos": [],
        "cosU_pos": [],
        "concept_share_neg": [],
        "concept_share_pos": [],
    }
    for b in range(B):
        blk = model.model.layers[b]
        gamma = blk.post_attention_layernorm.weight.to(torch.float32)
        Wg = blk.mlp.gate_proj.weight.to(torch.float32)
        Wu = blk.mlp.up_proj.weight.to(torch.float32)
        Wd = blk.mlp.down_proj.weight.to(torch.float32)
        d = vectors[b, Kc]
        dg = d * gamma
        write = d @ Wd
        read_u = dg @ Wu.T
        read_g = dg @ Wg.T
        rw_u, rw_g = (read_u * write), (read_g * write)
        contrib = torch.tensor(neur[:, b], device=dev)
        neg = contrib < 0
        stats["rw_up_neg"].append(float((rw_u * neg).sum(1).mean()))
        stats["rw_up_pos"].append(float((rw_u * ~neg).sum(1).mean()))
        stats["rw_gate_neg"].append(float((rw_g * neg).sum(1).mean()))
        stats["rw_gate_pos"].append(float((rw_g * ~neg).sum(1).mean()))
        stats["read_up_abs_all"].append(float(read_u.abs().mean()))
        wdn = torch.nn.functional.normalize(Wd.T, dim=-1)
        cE = wdn @ e_gap.T
        cU = wdn @ u_gap.T
        kidx = torch.arange(len(pairs), device=dev)
        order = torch.argsort(contrib, dim=1)
        tn, tp = order[:, :10], order[:, -10:]
        sgn_n = torch.sign(write[kidx[:, None], tn])
        sgn_p = torch.sign(write[kidx[:, None], tp])
        stats["cosE_neg"].append(float((cE[tn, kidx[:, None]] * sgn_n).mean()))
        stats["cosU_neg"].append(float((cU[tn, kidx[:, None]] * sgn_n).mean()))
        stats["cosE_pos"].append(float((cE[tp, kidx[:, None]] * sgn_p).mean()))
        stats["cosU_pos"].append(float((cU[tp, kidx[:, None]] * sgn_p).mean()))
        stats["concept_share_neg"].append(
            float((contrib.gather(1, tn).sum(1) / contrib.clamp(max=0).sum(1).clamp(max=-1e-9)).mean())
        )
        stats["concept_share_pos"].append(
            float((contrib.gather(1, tp).sum(1) / contrib.clamp(min=0).sum(1).clamp(min=1e-9)).mean())
        )
        rows = []
        for kind in ("neg", "pos"):
            for n, cnt in top[kind][str(b)][: args.n_per_block]:
                w = Wd[:, n]
                pu = U @ w
                pe = E @ w
                wn = torch.nn.functional.normalize(w, dim=0)
                rows.append(
                    {
                        "kind": kind,
                        "neuron": n,
                        "shared_by": cnt,
                        "share_of_concepts": round(cnt / top["K"], 2),
                        "mean_contrib": round(float(contrib[:, n].mean()), 4),
                        "read_up_x_write_mean": round(float(rw_u[:, n].mean()), 4),
                        "read_gate_x_write_mean": round(float(rw_g[:, n].mean()), 4),
                        "write_mean": round(float(write[:, n].mean()), 4),
                        "cosE_gap": round(float((wn @ e_gap.T).mean()), 3),
                        "cosU_gap": round(float((wn @ u_gap.T).mean()), 3),
                        "U_top": [tok.decode([t]) for t in pu.topk(12).indices.tolist()],
                        "U_bottom": [tok.decode([t]) for t in (-pu).topk(12).indices.tolist()],
                        "E_top": [tok.decode([t]) for t in pe.topk(12).indices.tolist()],
                        "E_bottom": [tok.decode([t]) for t in (-pe).topk(12).indices.tolist()],
                    }
                )
        report[str(b)] = rows
        if b % 3 == 0:
            log.info(
                "block %d: read_up*write over neg/pos neurons %.3f/%.3f; top-neg neurons' output cosE %.3f cosU %.3f; top-pos cosE %.3f cosU %.3f; top-10 share of neg %.2f",
                b,
                stats["rw_up_neg"][-1],
                stats["rw_up_pos"][-1],
                stats["cosE_neg"][-1],
                stats["cosU_neg"][-1],
                stats["cosE_pos"][-1],
                stats["cosU_pos"][-1],
                stats["concept_share_neg"][-1],
            )
    (args.out / "neurons.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    np.savez(str(args.out / "neurons.npz"), **{k: np.array(v) for k, v in stats.items()})  # type: ignore[arg-type]
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
