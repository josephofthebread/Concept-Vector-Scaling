from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(".")
GEN = Path("../exp_genvectors")
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))
K = 1036
B = 36
R = 37
LENS_ROWS = list(range(0, 36, 2)) + [35]


def q(x: np.ndarray, axis=0):
    return {
        "p50": np.nanmedian(x, axis).tolist(),
        "p10": np.nanpercentile(x, 10, axis).tolist(),
        "p90": np.nanpercentile(x, 90, axis).tolist(),
    }


def set_matrix(sets: dict[int, list[int]], vocab_index: dict[int, int], n_sub: int) -> torch.Tensor:
    """Row-normalized [K, n_sub] indicator of each pair's word set over the sub-vocabulary."""
    G = torch.zeros(K, n_sub)
    for kk, ids in sets.items():
        for t in ids:
            G[kk, vocab_index[t]] = 1.0
        G[kk] /= max(1, len(ids))
    return G


def percentile_of_own(scores: torch.Tensor) -> torch.Tensor:
    """Per column j, the fraction of other pairs' word sets scoring below pair j's own set."""
    own = scores.diagonal()
    below = (scores < own[None, :]).float().sum(0) - 0.0
    return below / (scores.shape[0] - 1)


def main() -> None:
    t0 = time.monotonic()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", local_files_only=True)
    resid = torch.zeros(K, 2, R, 4096)
    mlp = torch.zeros(K, 2, B, 4096)
    h35 = torch.zeros(K, 2, 4096)
    for k in range(K):
        acc_r = acc_m = acc_h = None
        tot = torch.zeros(2)
        for d in ("work/vdecomp", "work/vdecomp2"):
            with safe_open(str(ROOT / d / f"pair-{k:04d}.safetensors"), "pt") as f:
                n = f.get_tensor("n_stories").float()
                r = f.get_tensor("pole_resid").float() * n[:, None, None]
                m = f.get_tensor("pole_mlp").float() * n[:, None, None]
                h = f.get_slice("pole_head")[:, 35].float().sum(1) * n[:, None]
            acc_r = r if acc_r is None else acc_r + r
            acc_m = m if acc_m is None else acc_m + m
            acc_h = h if acc_h is None else acc_h + h
            tot += n
        r, m, h = acc_r / tot[:, None, None], acc_m / tot[:, None, None], acc_h / tot[:, None]
        resid[k], mlp[k], h35[k] = r, m, h
        if k % 200 == 0:
            print(f"loaded {k} pairs, {(time.monotonic() - t0) / 60:.1f} min", flush=True)
    A = torch.zeros(K, 2, B, 4096)
    A[:, :, :35] = resid[:, :, 1:36] - resid[:, :, 0:35] - mlp[:, :, :35]
    A[:, :, 35] = h35
    mu_pop = resid.mean((0, 1))
    t = resid.mean(1) - mu_pop[None]
    v = resid[:, 0] - resid[:, 1]
    At = A.mean(1) - A.mean((0, 1))[None]
    Mt = mlp.mean(1) - mlp.mean((0, 1))[None]
    Av = A[:, 0] - A[:, 1]
    Mv = mlp[:, 0] - mlp[:, 1]
    out: dict = {"n_pairs": K}
    rec_t = t[:, 0] + At[:, :35].sum(1) + Mt[:, :35].sum(1)
    rec_v = v[:, 0] + Av[:, :35].sum(1) + Mv[:, :35].sum(1)
    out["recon_relerr_row35"] = {
        "topic": float(((rec_t - t[:, 35]).norm(dim=-1) / t[:, 35].norm(dim=-1)).median()),
        "lean": float(((rec_v - v[:, 35]).norm(dim=-1) / v[:, 35].norm(dim=-1)).median()),
    }
    nrm = lambda x: x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    out["norm_topic_over_lean"] = q((t.norm(dim=-1) / v.norm(dim=-1)).numpy())
    out["cos_topic_lean"] = q((nrm(t) * nrm(v)).sum(-1).numpy())
    for Rf in (22, 35):
        th, vh = nrm(t[:, Rf]), nrm(v[:, Rf])
        tn, vn = t[:, Rf].norm(dim=-1), v[:, Rf].norm(dim=-1)
        out[f"share_final{Rf}"] = {
            "mlp_topic": q(((Mt * th[:, None]).sum(-1) / tn[:, None]).numpy()),
            "attn_topic": q(((At * th[:, None]).sum(-1) / tn[:, None]).numpy()),
            "mlp_lean": q(((Mv * vh[:, None]).sum(-1) / vn[:, None]).numpy()),
            "attn_lean": q(((Av * vh[:, None]).sum(-1) / vn[:, None]).numpy()),
        }
        cum_t = torch.cumsum((Mt * th[:, None]).sum(-1), 1)
        cum_v = torch.cumsum((Mv * vh[:, None]).sum(-1), 1)
        out[f"cummlp_final{Rf}"] = {"topic": q((cum_t / tn[:, None]).numpy()), "lean": q((cum_v / vn[:, None]).numpy())}

        def crossing(cum):
            ok = cum >= 0
            suffix = torch.flip(torch.cumprod(torch.flip(ok.float(), [1]), 1), [1]) > 0
            first = torch.where(suffix.any(1), suffix.float().argmax(1), torch.full((K,), B))
            return first

        out[f"crossing_final{Rf}"] = {
            "topic": np.bincount(crossing(cum_t).numpy(), minlength=B + 1).tolist(),
            "lean": np.bincount(crossing(cum_v).numpy(), minlength=B + 1).tolist(),
        }
    th22, vh22 = nrm(t[:, 22]), nrm(v[:, 22])
    eras = {
        "topic_cos": [],
        "topic_surv": [],
        "lean_cos": [],
        "lean_surv": [],
        "topic_cos_off": [],
        "topic_surv_off": [],
        "lean_cos_off": [],
        "lean_surv_off": [],
    }
    SAt = torch.zeros(K, 4096)
    SMt = torch.zeros(K, 4096)
    SAv = torch.zeros(K, 4096)
    SMv = torch.zeros(K, 4096)
    for r in range(1, 36):
        SAt += At[:, r - 1]
        SMt += Mt[:, r - 1]
        SAv += Av[:, r - 1]
        SMv += Mv[:, r - 1]
        for name, SA, SM, d in (("topic", SAt, SMt, th22), ("lean", SAv, SMv, vh22)):
            c = (nrm(SA) * nrm(SM)).sum(-1)
            s = (SA + SM).norm(dim=-1) / SA.norm(dim=-1).clamp_min(1e-8)
            eras[f"{name}_cos"].append(q(c.numpy()))
            eras[f"{name}_surv"].append(q(s.numpy()))
            XA = SA - (SA * d).sum(-1, keepdim=True) * d
            XM = SM - (SM * d).sum(-1, keepdim=True) * d
            c2 = (nrm(XA) * nrm(XM)).sum(-1)
            s2 = (XA + XM).norm(dim=-1) / XA.norm(dim=-1).clamp_min(1e-8)
            eras[f"{name}_cos_off"].append(q(c2.numpy()))
            eras[f"{name}_surv_off"].append(q(s2.numpy()))
    out["erasure_rows1to35"] = eras
    gt = (Mt * nrm(t[:, :B])).sum(-1) / t[:, :B].norm(dim=-1).clamp_min(1e-8)
    gv = (Mv * nrm(v[:, :B])).sum(-1) / v[:, :B].norm(dim=-1).clamp_min(1e-8)
    gat = (At * nrm(t[:, :B])).sum(-1) / t[:, :B].norm(dim=-1).clamp_min(1e-8)
    gav = (Av * nrm(v[:, :B])).sum(-1) / v[:, :B].norm(dim=-1).clamp_min(1e-8)
    out["gain_current"] = {
        "mlp_topic": q(gt.numpy()),
        "mlp_lean": q(gv.numpy()),
        "attn_topic": q(gat.numpy()),
        "attn_lean": q(gav.numpy()),
    }
    print(f"lanes built, {(time.monotonic() - t0) / 60:.1f} min", flush=True)
    snap = sorted((Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots").glob("*"))[-1]
    index = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    want = {"model.embed_tokens.weight": None, "lm_head.weight": None, "model.norm.weight": None}
    for name in want:
        with safe_open(str(snap / index[name]), "pt") as f:
            want[name] = f.get_tensor(name).float()
    E = want["model.embed_tokens.weight"]
    U = want["lm_head.weight"] * want["model.norm.weight"][None, :]
    ont = json.loads((GEN / "data" / "semantic.ontology.json").read_text())
    items = ont["pairs"] if isinstance(ont, dict) and "pairs" in ont else ont
    stop = set(
        "the a an of and or to in on at for with is are was were be been being by as it its this that these those i you he she "
        "they we them their our your my me him her his one two either side may apply about whether while another not no but so "
        "if than then there here when which who whom what how do does did have has had someone person people".split()
    )
    gsets: dict[int, list[int]] = {}
    for it in items:
        kk = int(it["pair_id"])
        ids = [
            int(t)
            for t in tok(it["narrative_guidance"], add_special_tokens=False)["input_ids"]
            if str(tok.decode([t])).strip().lower().strip('".,:;()') not in stop
            and str(tok.decode([t])).strip().isalpha()
        ]
        if ids:
            gsets[kk] = sorted(set(ids))
    markers = json.loads((ROOT / "data" / "markers_presence.json").read_text())
    csets = {int(k): v for k, v in markers["concept"].items()}
    asets = {int(k): v for k, v in markers["antagonist"].items()}
    vocab = sorted({tt for s in list(gsets.values()) + list(csets.values()) + list(asets.values()) for tt in s})
    vidx = {tt: i for i, tt in enumerate(vocab)}
    Gg = set_matrix(gsets, vidx, len(vocab))
    Gc = set_matrix(csets, vidx, len(vocab))
    Ga = set_matrix(asets, vidx, len(vocab))
    Us, Es = U[vocab], E[vocab]
    have_g = torch.tensor(sorted(gsets))
    lens: dict = {"rows": LENS_ROWS, "n_guidance_pairs": int(len(gsets))}
    for lane_name, lane in (("topic", t), ("lean", v)):
        for form_name, W in (("out", Us), ("in", Es)):
            pg, pm = [], []
            for r in LENS_ROWS:
                X = nrm(lane[:, r]).T
                sc = W @ X
                sg = Gg @ sc
                pct = percentile_of_own(sg[have_g][:, have_g])
                pg.append(q(pct.numpy()))
                gap = (Gc - Ga) @ sc
                pm.append(q(percentile_of_own(gap).numpy()))
            lens[f"{lane_name}_{form_name}_guidance_pct"] = pg
            lens[f"{lane_name}_{form_name}_markergap_pct"] = pm
    for lane_name, Acum, Mcum in (("topic", At, Mt), ("lean", Av, Mv)):
        pa, pm2 = [], []
        for r in LENS_ROWS:
            if r == 0:
                pa.append(None)
                pm2.append(None)
                continue
            SA = Acum[:, :r].sum(1)
            SM = Mcum[:, :r].sum(1)
            for dst, X in ((pa, nrm(SA)), (pm2, nrm(SM))):
                sc = Us @ X.T
                if lane_name == "topic":
                    sg = Gg @ sc
                    dst.append(q(percentile_of_own(sg[have_g][:, have_g]).numpy()))
                else:
                    dst.append(q(percentile_of_own((Gc - Ga) @ sc).numpy()))
        lens[f"{lane_name}_attn_cum_pct"] = pa
        lens[f"{lane_name}_mlp_cum_pct"] = pm2
        pn = []
        for r in LENS_ROWS:
            if r == 0:
                pn.append(None)
                continue
            SM = Mcum[:, :r].sum(1)
            sc = Us @ (-nrm(SM)).T
            if lane_name == "topic":
                pn.append(q(percentile_of_own((Gg @ sc)[have_g][:, have_g]).numpy()))
            else:
                pn.append(q(percentile_of_own((Gc - Ga) @ sc).numpy()))
        lens[f"{lane_name}_negmlp_cum_pct"] = pn
    out["lens"] = lens
    Path(ROOT / "work/tmp/topiclane.json").write_text(json.dumps(out))
    print(f"written, {(time.monotonic() - t0) / 60:.1f} min", flush=True)
    print("\nrecon rel.err row 35:", out["recon_relerr_row35"])
    print(
        "block | gain mlp topic | gain mlp lean | share22 mlp topic | share22 mlp lean | share22 attn topic | share22 attn lean"
    )
    for b in range(B):
        print(
            f"{b:5d} | {out['gain_current']['mlp_topic']['p50'][b]:+.3f} | {out['gain_current']['mlp_lean']['p50'][b]:+.3f} | "
            f"{out['share_final22']['mlp_topic']['p50'][b]:+.4f} | {out['share_final22']['mlp_lean']['p50'][b]:+.4f} | "
            f"{out['share_final22']['attn_topic']['p50'][b]:+.4f} | {out['share_final22']['attn_lean']['p50'][b]:+.4f}"
        )
    print(
        "\nrow | topic cos(SA,SM) | topic surv | lean cos | lean surv | topic cos off | topic surv off | lean cos off | lean surv off"
    )
    for i, r in enumerate(range(1, 36)):
        e = eras
        print(
            f"{r:3d} | {e['topic_cos'][i]['p50']:+.3f} | {e['topic_surv'][i]['p50']:.3f} | {e['lean_cos'][i]['p50']:+.3f} | {e['lean_surv'][i]['p50']:.3f} | "
            f"{e['topic_cos_off'][i]['p50']:+.3f} | {e['topic_surv_off'][i]['p50']:.3f} | {e['lean_cos_off'][i]['p50']:+.3f} | {e['lean_surv_off'][i]['p50']:.3f}"
        )
    print(
        "\nrow | topic->own guidance pct (out/in) | lean->own guidance pct (out/in) | topic->own markergap pct (out) | lean->own markergap pct (out/in)"
    )
    for i, r in enumerate(LENS_ROWS):
        L = lens
        print(
            f"{r:3d} | {L['topic_out_guidance_pct'][i]['p50']:.3f} / {L['topic_in_guidance_pct'][i]['p50']:.3f} | "
            f"{L['lean_out_guidance_pct'][i]['p50']:.3f} / {L['lean_in_guidance_pct'][i]['p50']:.3f} | {L['topic_out_markergap_pct'][i]['p50']:.3f} | "
            f"{L['lean_out_markergap_pct'][i]['p50']:.3f} / {L['lean_in_markergap_pct'][i]['p50']:.3f}"
        )
    print(
        "\nrow | topic: attn-cum pct | mlp-cum pct | -mlp-cum pct || lean: attn-cum | mlp-cum | -mlp-cum (own set, output form)"
    )
    for i, r in enumerate(LENS_ROWS):
        if r == 0:
            continue
        L = lens
        print(
            f"{r:3d} | {L['topic_attn_cum_pct'][i]['p50']:.3f} | {L['topic_mlp_cum_pct'][i]['p50']:.3f} | {L['topic_negmlp_cum_pct'][i]['p50']:.3f} || "
            f"{L['lean_attn_cum_pct'][i]['p50']:.3f} | {L['lean_mlp_cum_pct'][i]['p50']:.3f} | {L['lean_negmlp_cum_pct'][i]['p50']:.3f}"
        )
    print("\ncrossing (final 22) histogram topic:", out["crossing_final22"]["topic"])
    print("crossing (final 22) histogram lean :", out["crossing_final22"]["lean"])
    print(f"total {(time.monotonic() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
