from __future__ import annotations

import json
import logging
import time
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from exp_attn_interp.amp import AMPS, np_ranks
from exp_attn_interp.common import ValueCatcher, load_model, story_span
from exp_attn_interp.config import GENDATA_ROOT, INSTRUCT_MODEL, vector_tag_for
from exp_attn_interp.decomp import POOL_MIN, POOL_SKIP, MlpCatcher, read_layers_for, story_ids
from exp_attn_interp.vectors import load_diff_vectors

log = logging.getLogger("rot")


@torch.inference_mode()
def process(model, tokenizer, catcher, mlps, T, pair_id, side, story, out_dir, last, story_index, cos_ml):
    item_id = 300000 + pair_id * 20 + story_index * 2 + (0 if side == "concept" else 1)
    path = out_dir / f"item-{item_id:06d}.safetensors"
    if path.exists():
        return {"item": item_id, "skipped": True}
    ids = story_ids(tokenizer, story)
    n = len(ids)
    s0, s1 = story_span(tokenizer, ids, story)
    span = list(range(s0 + POOL_SKIP, s1)) if (s1 - s0 - POOL_SKIP) >= POOL_MIN else list(range(s0, s1))
    out = model(
        input_ids=torch.tensor([ids], device=model.device),
        output_attentions=True,
        output_hidden_states=True,
        use_cache=False,
    )
    hidden = out.hidden_states
    device = model.device
    S = torch.tensor(span, device=device)
    heads, kv, hd = catcher.heads, catcher.kv_heads, catcher.head_dim
    group = heads // kv
    c_attn = torch.zeros(3, last, heads)
    c_mlp = torch.zeros(3, last)
    c_embed = (T @ hidden[0][0][S].to(torch.float32).mean(0)).cpu()
    total = (T @ hidden[last][0][S].to(torch.float32).mean(0)).cpu()
    for b in range(last):
        attn = out.attentions[b][0].to(torch.float32)
        vals = catcher.values[b].view(n, kv, hd).to(torch.float32)
        Vv = vals[:, torch.arange(heads, device=device) // group]
        ctx = torch.bmm(attn[:, S, :], Vv.permute(1, 0, 2))
        scale, gain = mlps.attn_scale[b], mlps.attn_gain[b]
        if scale is not None:
            ctx = ctx * scale[S][None, :, None]
        W_O = model.model.layers[b].self_attn.o_proj.weight.to(torch.float32).view(-1, heads, hd)
        head_pool = torch.einsum("hsd,ohd->ho", ctx, W_O) / len(span)
        if gain is not None:
            head_pool = head_pool * gain[None, :]
        mo = mlps.out[b]
        assert mo is not None
        c_attn[:, b] = (head_pool @ T.T).T.cpu()
        c_mlp[:, b] = (T @ mo[0][S].to(torch.float32).mean(0)).cpu()
        del attn, ctx
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "c_attn": c_attn,
            "c_mlp": c_mlp,
            "c_embed": c_embed,
            "total": total,
            "pair_id": torch.tensor(pair_id),
            "side": torch.tensor(0 if side == "concept" else 1),
            "cos_mid_last": torch.tensor(cos_ml),
        },
        str(path),
        metadata={"last": str(last)},
    )
    recon = c_embed + c_attn.sum((1, 2)) + c_mlp.sum(1)
    relerr = ((recon - total).abs() / total.abs().clamp_min(1e-6)).tolist()
    return {"item": item_id, "relerr": [round(e, 4) for e in relerr]}


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    parser.add_argument("--per-pole", type=int, default=2)
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--band-start", type=int, default=19)
    parser.add_argument("--steer-family", default="qwen")
    parser.add_argument("--offline", action=BooleanOptionalAction, default=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    short = args.model.split("/")[-1]
    work = Path("work") if args.model == INSTRUCT_MODEL else Path(f"work-{short}")
    if args.analyze:
        steer = None
        sp = Path("work/steerability.json")
        if sp.exists():
            steer = json.loads(sp.read_text()).get(args.steer_family)
        amps = AMPS[args.model]
        band = args.band_start
        out = work / "rotan.json"
        files = sorted((work / "rot").glob("item-*.safetensors"))
        acc: dict[tuple[int, int], dict] = {}
        cos: dict[int, float] = {}
        for f in files:
            with safe_open(str(f), framework="np") as h:
                k, side = int(h.get_tensor("pair_id")), int(h.get_tensor("side"))
                ca, cm = h.get_tensor("c_attn").astype(np.float64), h.get_tensor("c_mlp").astype(np.float64)
                ce, tot = h.get_tensor("c_embed").astype(np.float64), h.get_tensor("total").astype(np.float64)
                cos[k] = float(h.get_tensor("cos_mid_last"))
            a = acc.setdefault(
                (k, side),
                {
                    "ca": np.zeros_like(ca),
                    "cm": np.zeros_like(cm),
                    "ce": np.zeros_like(ce),
                    "tot": np.zeros_like(tot),
                    "n": 0,
                },
            )
            for key, val in (("ca", ca), ("cm", cm), ("ce", ce), ("tot", tot)):
                a[key] += val
            a["n"] += 1
        pairs = sorted(k for k in {k for k, _ in acc} if (k, 0) in acc and (k, 1) in acc)
        B, H = acc[(pairs[0], 0)]["ca"].shape[1:]
        names = ["v_mid", "v_last", "u"]
        rows = []
        DA = np.zeros((len(pairs), 3, B, H))
        DM = np.zeros((len(pairs), 3, B))
        DT = np.zeros((len(pairs), 3))
        for i, k in enumerate(pairs):
            c, a = acc[(k, 0)], acc[(k, 1)]
            DA[i] = c["ca"] / c["n"] - a["ca"] / a["n"]
            DM[i] = c["cm"] / c["n"] - a["cm"] / a["n"]
            DT[i] = c["tot"] / c["n"] - a["tot"] / a["n"]
            rows.append(k)
        res: dict = {
            "n_concepts": len(pairs),
            "n_blocks": B,
            "targets": names,
            "cos_mid_last_median": float(np.median([cos[k] for k in pairs])),
        }
        for t, nm in enumerate(names):
            att = DA[:, t].sum((1, 2)) / DT[:, t]
            mlp = DM[:, t].sum(1) / DT[:, t]
            per_block = (DA[:, t].sum(2) + DM[:, t]) / DT[:, t][:, None]
            cum = np.cumsum(np.median(per_block, 0))
            res[nm] = {
                "attn_share_median": float(np.median(att)),
                "mlp_share_median": float(np.median(mlp)),
                "cum_median": cum.tolist(),
                "before_band": float(cum[band - 1]),
                "f50_block": int(np.argmax(cum >= 0.5)),
                "signal_median": float(np.median(DT[:, t])),
            }
            log.info(
                "%-6s at last layer: attention %.3f  MLP %.3f  | assembled before band %.3f, 50%% by block %d",
                nm,
                res[nm]["attn_share_median"],
                res[nm]["mlp_share_median"],
                res[nm]["before_band"],
                res[nm]["f50_block"],
            )
        for nm in ("u", "v_mid"):
            ti = names.index(nm)
            share = DA[:, ti] / DT[:, ti][:, None, None]
            mean, pos = share.mean(0), (share > 0).mean(0)
            top = np.argsort(-mean.ravel())[:10]
            res[f"{nm}_top_heads"] = [
                [int(i // H), int(i % H), float(mean.ravel()[i]), float(pos.ravel()[i])] for i in top
            ]
            res[f"{nm}_amplifiers"] = {
                f"{b}:{h}": {
                    "mean_share": float(mean[b, h]),
                    "frac_positive": float(pos[b, h]),
                    "rank": int((mean > mean[b, h]).sum()) + 1,
                }
                for b, h in amps
            }
            w = np.abs(share).reshape(len(pairs), -1)
            res[f"{nm}_pr_median"] = float(np.median(w.sum(1) ** 2 / (w**2).sum(1)))
            log.info(
                "%s writers: top heads (b,h,mean share,frac positive) %s",
                nm,
                [(a, b, round(c, 4), round(d, 2)) for a, b, c, d in res[f"{nm}_top_heads"]],
            )
            log.info("%s: amplifiers %s | PR %.0f", nm, res[f"{nm}_amplifiers"], res[f"{nm}_pr_median"])
        su = (DA[:, 2] / DT[:, 2][:, None, None]).mean(0)[B // 2 :].ravel()
        sm = (DA[:, 0] / DT[:, 0][:, None, None]).mean(0)[B // 2 :].ravel()
        sl = (DA[:, 1] / DT[:, 1][:, None, None]).mean(0)[B // 2 :].ravel()
        res["rho_heads_u_vs_vmid_late_half"] = float(np.corrcoef(np_ranks(su), np_ranks(sm))[0, 1])
        res["rho_heads_u_vs_vlast_late_half"] = float(np.corrcoef(np_ranks(su), np_ranks(sl))[0, 1])
        log.info(
            "head ranking, late half: rho(u-writers, v_mid-writers) %+.3f; rho(u-writers, v_last-writers) %+.3f",
            res["rho_heads_u_vs_vmid_late_half"],
            res["rho_heads_u_vs_vlast_late_half"],
        )
        u_attn = DA[:, 2].sum((1, 2)) / DT[:, 2]
        u_amp = np.array([sum(DA[i, 2, b, h] for b, h in amps) / DT[i, 2] for i in range(len(pairs))])
        m_amp = np.array([sum(DA[i, 0, b, h] for b, h in amps) / DT[i, 0] for i in range(len(pairs))])
        rotv = np.array([cos[k] for k in pairs])
        res["per_concept"] = {
            "rho_uattn_vs_cos": float(np.corrcoef(np_ranks(u_attn), np_ranks(rotv))[0, 1]),
            "rho_uamp_vs_cos": float(np.corrcoef(np_ranks(u_amp), np_ranks(rotv))[0, 1]),
        }
        log.info(
            "per concept: attention share of u vs cos(mid,last) rho %+.3f; amplifier share of u vs cos rho %+.3f",
            res["per_concept"]["rho_uattn_vs_cos"],
            res["per_concept"]["rho_uamp_vs_cos"],
        )
        if steer:
            ks = [i for i, k in enumerate(pairs) if str(k) in steer]
            s = np.array([steer[str(pairs[i])] for i in ks])
            for nm, x in (
                ("u_attn_share", u_attn[ks]),
                ("u_amp_share", u_amp[ks]),
                ("vmid_amp_share", m_amp[ks]),
                ("u_signal", DT[ks, 2]),
                ("u_signal_over_vlast", DT[ks, 2] / DT[ks, 1]),
                ("cos", rotv[ks]),
            ):
                r = float(np.corrcoef(np_ranks(x), np_ranks(s))[0, 1])
                res["per_concept"][f"rho_{nm}_vs_steer"] = r
                log.info("per concept: %s vs steerability rho %+.3f", nm, r)
        out.write_text(json.dumps(res, indent=1))
        log.info("wrote %s", out)
        return
    model, tokenizer = load_model(offline=args.offline, model_name=args.model)
    catcher = ValueCatcher(model.model)
    mlps = MlpCatcher(model.model)
    vectors = load_diff_vectors(device=str(model.device), tag=vector_tag_for(args.model))
    rl = read_layers_for(len(model.model.layers))
    mid, last = rl[len(rl) // 2], rl[-1]
    tbl = pq.read_table(
        GENDATA_ROOT / "data" / "semantic.parquet",
        columns=["pair_id", "concept_text", "antagonist_text"],
        filters=[("language_id", "==", 0)],
    )
    by_pair: dict[int, list[tuple[str, str]]] = {}
    for p, c, a in zip(tbl["pair_id"].to_pylist(), tbl["concept_text"].to_pylist(), tbl["antagonist_text"].to_pylist()):
        by_pair.setdefault(int(p), []).append((c, a))
    pairs = sorted(by_pair)[args.chunk * 64 : (args.chunk + 1) * 64]
    log.info("%s: mid %d last %d; chunk %d: %d pairs", short, mid, last, args.chunk, len(pairs))
    started = time.monotonic()
    n = 0
    for p in pairs:
        vm, vl = vectors[mid][p], vectors[last][p]
        cos_ml = float(vm @ vl)
        u = vl - cos_ml * vm
        u = u / u.norm().clamp_min(1e-6)
        T = torch.stack([vm, vl, u])
        for i, (c, a) in enumerate(by_pair[p][: args.per_pole]):
            for side, text in (("concept", c), ("antagonist", a)):
                r = process(model, tokenizer, catcher, mlps, T, p, side, text, work / "rot", last, i, cos_ml)
                n += 1
                if n <= 3 or n % 50 == 0:
                    log.info("%d %s", n, r)
    log.info("chunk %d done: %.1f min", args.chunk, (time.monotonic() - started) / 60)


if __name__ == "__main__":
    main()
