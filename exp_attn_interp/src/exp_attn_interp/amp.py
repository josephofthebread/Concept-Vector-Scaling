from __future__ import annotations

import json
import logging
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from safetensors import safe_open

from exp_attn_interp.common import ValueCatcher, load_model, story_span
from exp_attn_interp.config import GENDATA_ROOT, INSTRUCT_MODEL, vector_tag_for
from exp_attn_interp.decomp import read_layers_for, story_ids
from exp_attn_interp.vectors import load_diff_vectors

log = logging.getLogger("amp")
AMPS = {
    "Qwen/Qwen3-8B": [(30, 0), (27, 8), (34, 19), (34, 15), (30, 20), (26, 27)],
    "unsloth/Meta-Llama-3.1-8B-Instruct": [(25, 18), (17, 20), (14, 24), (29, 31), (21, 19), (28, 20)],
    "google/gemma-2-9b-it": [(40, 3), (26, 9), (31, 3), (23, 14), (29, 11), (40, 6)],
}


def np_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    r = np.empty_like(order, dtype=np.float64)
    r[order] = np.arange(len(x))
    return r


def empirical_builder_share(decomp_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Mean over concepts of D_head / D_total at the last read layer: [B, H]; plus frac positive."""
    S, _ = empirical_shares(decomp_dir)
    return S.mean(0), (S > 0).mean(0)


def empirical_shares(decomp_dir: Path) -> tuple[np.ndarray, list[int]]:
    """Per-concept share D_head / D_total at the last read layer: [K, B, H], with the pair ids."""
    files = sorted(decomp_dir.glob("item-*.safetensors"))
    acc: dict[tuple[int, int], dict] = {}
    for f in files:
        with safe_open(str(f), framework="np") as h:
            k, side = int(h.get_tensor("pair_id")), int(h.get_tensor("side"))
            ca = h.get_tensor("c_attn").astype(np.float64)[-1]
            tot = float(h.get_tensor("total").astype(np.float64)[-1])
        a = acc.setdefault((k, side), {"ca": np.zeros_like(ca), "tot": 0.0, "n": 0})
        a["ca"] += ca
        a["tot"] += tot
        a["n"] += 1
    shares = []
    for k in sorted({k for k, _ in acc}):
        if (k, 0) in acc and (k, 1) in acc:
            c, a = acc[(k, 0)], acc[(k, 1)]
            D = c["ca"] / c["n"] - a["ca"] / a["n"]
            Dt = c["tot"] / c["n"] - a["tot"] / a["n"]
            shares.append(D / Dt)
    S = np.stack(shares)
    return S, [k for k in sorted({k for k, _ in acc}) if (k, 0) in acc and (k, 1) in acc]


@torch.inference_mode()
def weight_scores(model, catcher: ValueCatcher, vectors: torch.Tensor, read_last: int, postnorm: bool):
    layers = model.model.layers
    B, H, hd, kv = len(layers), catcher.heads, catcher.head_dim, catcher.kv_heads
    group = H // kv
    K = vectors.shape[1]
    v_last = vectors[read_last]
    copy_last = torch.zeros(B, H, K)
    copy_own = torch.zeros(B, H, K)
    eig_pos = torch.zeros(B, H)
    for b, blk in enumerate(layers):
        w_in = blk.input_layernorm.weight.to(torch.float32)
        if postnorm:
            w_in = 1.0 + w_in
        WV = blk.self_attn.v_proj.weight.to(torch.float32).view(kv, hd, -1)
        WO = blk.self_attn.o_proj.weight.to(torch.float32).view(-1, H, hd)
        if postnorm:
            WO = WO * (1.0 + blk.post_attention_layernorm.weight.to(torch.float32))[:, None, None]
        v_own = vectors[b]
        out_side = torch.einsum("ohd,ko->khd", WO, v_last)
        in_last = torch.einsum("gdi,ki->kgd", WV, v_last * w_in)
        in_own = torch.einsum("gdi,ki->kgd", WV, v_own * w_in)
        for h in range(H):
            g = h // group
            copy_last[b, h] = (out_side[:, h] * in_last[:, g]).sum(-1).cpu()
            copy_own[b, h] = (out_side[:, h] * in_own[:, g]).sum(-1).cpu()
            M = WV[g] @ WO[:, h, :]
            ev = torch.linalg.eigvals(M)
            eig_pos[b, h] = (ev.real.sum() / ev.abs().sum()).cpu()
    return copy_last, copy_own, eig_pos


@torch.inference_mode()
def attention_stats(model, tokenizer, stories: list[str]) -> np.ndarray:
    """[B, H, 4]: mean entropy, self mass, sink mass (pos 0-1), mean attended distance (over story tokens)."""
    B = len(model.model.layers)
    H = model.config.num_attention_heads
    acc = np.zeros((B, H, 4))
    n = 0
    for text in stories:
        ids = story_ids(tokenizer, text)
        s0, s1 = story_span(tokenizer, ids, text)
        out = model(input_ids=torch.tensor([ids], device=model.device), output_attentions=True, use_cache=False)
        T = len(ids)
        pos = torch.arange(T, device=model.device)
        dist = (pos[:, None] - pos[None, :]).clamp_min(0).to(torch.float32)
        for b in range(B):
            A = out.attentions[b][0].to(torch.float32)[:, s0:s1, :]
            ent = -(A.clamp_min(1e-12) * A.clamp_min(1e-12).log()).sum(-1).mean(-1)
            selfm = A[:, torch.arange(s1 - s0), torch.arange(s0, s1)].mean(-1)
            sink = A[:, :, :2].sum(-1).mean(-1)
            d = (A * dist[s0:s1][None]).sum(-1).mean(-1)
            acc[b, :, 0] += ent.cpu().numpy()
            acc[b, :, 1] += selfm.cpu().numpy()
            acc[b, :, 2] += sink.cpu().numpy()
            acc[b, :, 3] += d.cpu().numpy()
        n += 1
    return acc / n


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    parser.add_argument("--stories", type=int, default=64)
    parser.add_argument("--presentation", default="assistant", choices=["assistant", "raw"])
    parser.add_argument(
        "--amps-of",
        default="",
        help="model whose amplifier list to rank (default: --model; use the instruct model for its base)",
    )
    parser.add_argument("--offline", action=BooleanOptionalAction, default=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    short = args.model.split("/")[-1]
    work = Path("work") if args.model == INSTRUCT_MODEL else Path(f"work-{short}")
    amps = AMPS[args.amps_of or args.model]

    log.info("empirical builder shares from %s", work / "decomp")
    S_emp, emp_pairs = empirical_shares(work / "decomp")
    emp_mean, emp_pos = S_emp.mean(0), (S_emp > 0).mean(0)

    model, tokenizer = load_model(offline=args.offline, model_name=args.model)
    catcher = ValueCatcher(model.model)
    vectors = load_diff_vectors(device=str(model.device), tag=vector_tag_for(args.model, args.presentation))
    n_blocks = len(model.model.layers)
    read_last = read_layers_for(n_blocks)[-1]
    postnorm = hasattr(model.model.layers[0], "pre_feedforward_layernorm")
    copy_last, copy_own, eig_pos = weight_scores(model, catcher, vectors, read_last, postnorm)
    B, H = emp_mean.shape
    res: dict = {"model": args.model, "read_last": read_last, "amplifiers": amps}

    def head_rank(score: np.ndarray, b: int, h: int) -> int:
        return int((score > score[b, h]).sum()) + 1

    for name, sc in (("copy_last", copy_last), ("copy_own", copy_own)):
        mean = sc.mean(-1).numpy()
        pos = (sc > 0).float().mean(-1).numpy()
        late = slice(n_blocks // 2, read_last)
        r_all = float(np.corrcoef(np_ranks(mean[:read_last].ravel()), np_ranks(emp_mean[:read_last].ravel()))[0, 1])
        r_late = float(np.corrcoef(np_ranks(mean[late].ravel()), np_ranks(emp_mean[late].ravel()))[0, 1])
        r_pos = float(np.corrcoef(np_ranks(pos[late].ravel()), np_ranks(emp_pos[late].ravel()))[0, 1])
        top10 = np.argsort(-mean[:read_last].ravel())[:10]
        res[name] = {
            "rho_vs_empirical_share_all_heads": r_all,
            "rho_vs_empirical_share_late_half": r_late,
            "rho_fracpos_vs_empirical_fracpos_late_half": r_pos,
            "amplifier_ranks_of_%d" % (read_last * H): {f"{b}:{h}": head_rank(mean[:read_last], b, h) for b, h in amps},
            "amplifier_frac_positive": {f"{b}:{h}": float(pos[b, h]) for b, h in amps},
            "top10_weight_only": [[int(i // H), int(i % H), float(mean.ravel()[i])] for i in top10],
            "empirical_rank_of_top10_weight_only": [
                head_rank(emp_mean[:read_last], int(i // H), int(i % H)) for i in top10
            ],
        }
        log.info(
            "%s: rho(weight-only score, empirical builder share) all heads %+.3f, late half %+.3f; "
            "frac-positive rho %+.3f",
            name,
            r_all,
            r_late,
            r_pos,
        )
        log.info(
            "%s: amplifier ranks (weight-only, of %d): %s",
            name,
            read_last * H,
            res[name]["amplifier_ranks_of_%d" % (read_last * H)],
        )
        log.info(
            "%s: weight-only top10 %s -> empirical ranks %s",
            name,
            res[name]["top10_weight_only"],
            res[name]["empirical_rank_of_top10_weight_only"],
        )
    late_b = slice(n_blocks // 2, read_last)
    co = copy_own.numpy()
    per_concept = [
        float(np.corrcoef(np_ranks(co[late_b, :, k].ravel()), np_ranks(S_emp[i, late_b].ravel()))[0, 1])
        for i, k in enumerate(emp_pairs)
    ]
    per_head = {
        f"{b}:{h}": float(np.corrcoef(np_ranks(co[b, h, emp_pairs]), np_ranks(S_emp[:, b, h]))[0, 1]) for b, h in amps
    }
    all_heads_within = [
        float(np.corrcoef(np_ranks(co[b, h, emp_pairs]), np_ranks(S_emp[:, b, h]))[0, 1])
        for b in range(n_blocks // 2, read_last)
        for h in range(H)
    ]
    res["per_concept_prediction"] = {
        "rho_over_late_heads_median": float(np.median(per_concept)),
        "rho_over_late_heads_p10": float(np.percentile(per_concept, 10)),
        "rho_over_late_heads_p90": float(np.percentile(per_concept, 90)),
        "rho_over_concepts_within_amplifier": per_head,
        "rho_over_concepts_within_head_late_median": float(np.median(all_heads_within)),
    }
    log.info(
        "per-concept prediction: within a concept, weight-only vs empirical over late heads rho median %.3f "
        "(p10 %.3f p90 %.3f); within an amplifier head over concepts %s; within any late head median %.3f",
        res["per_concept_prediction"]["rho_over_late_heads_median"],
        res["per_concept_prediction"]["rho_over_late_heads_p10"],
        res["per_concept_prediction"]["rho_over_late_heads_p90"],
        {k: round(v, 3) for k, v in per_head.items()},
        res["per_concept_prediction"]["rho_over_concepts_within_head_late_median"],
    )
    np.save(work / "copy_own.npy", co)
    ep = eig_pos.numpy()
    res["eig_positivity"] = {
        "amplifiers": {f"{b}:{h}": float(ep[b, h]) for b, h in amps},
        "late_half_median": float(np.median(ep[n_blocks // 2 : read_last])),
        "late_half_p90": float(np.percentile(ep[n_blocks // 2 : read_last], 90)),
        "amplifier_ranks": {f"{b}:{h}": head_rank(ep[:read_last], b, h) for b, h in amps},
    }
    log.info(
        "OV eigen-positivity: amplifiers %s | late-half median %.3f p90 %.3f | ranks %s",
        res["eig_positivity"]["amplifiers"],
        res["eig_positivity"]["late_half_median"],
        res["eig_positivity"]["late_half_p90"],
        res["eig_positivity"]["amplifier_ranks"],
    )

    tbl = pq.read_table(
        GENDATA_ROOT / "data" / "semantic.parquet",
        columns=["pair_id", "concept_text"],
        filters=[("language_id", "==", 0)],
    )
    seen: dict[int, str] = {}
    for p, c in zip(tbl["pair_id"].to_pylist(), tbl["concept_text"].to_pylist()):
        seen.setdefault(int(p), c)
    stories = [seen[k] for k in sorted(seen)[:: max(1, len(seen) // args.stories)]][: args.stories]
    st = attention_stats(model, tokenizer, stories)
    names = ["entropy", "self_mass", "sink_mass", "mean_distance"]
    late_st = st[n_blocks // 2 : read_last].reshape(-1, 4)
    res["attention"] = {
        "late_half_median": dict(zip(names, np.median(late_st, 0).tolist())),
        "late_half_p10": dict(zip(names, np.percentile(late_st, 10, axis=0).tolist())),
        "late_half_p90": dict(zip(names, np.percentile(late_st, 90, axis=0).tolist())),
        "amplifiers": {f"{b}:{h}": dict(zip(names, st[b, h].tolist())) for b, h in amps},
    }
    for b, h in amps:
        log.info("L%d H%d attention: entropy %.2f self %.3f sink %.3f dist %.1f", b, h, *st[b, h])
    log.info("late-half heads median: %s", res["attention"]["late_half_median"])
    (work / "amp.json").write_text(json.dumps(res, indent=1))
    log.info("wrote %s", work / "amp.json")


if __name__ == "__main__":
    main()
