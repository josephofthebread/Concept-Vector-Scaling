from __future__ import annotations

import json
import logging
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import Any

import numpy as np
import torch

from exp_attn_interp.common import load_model
from exp_attn_interp.config import INSTRUCT_MODEL, vector_tag_for
from exp_attn_interp.decomp import read_layers_for
from exp_attn_interp.vectors import load_diff_vectors

log = logging.getLogger("lex2vec")


@torch.inference_mode()
def block_operator(model, b: int, postnorm: bool) -> torch.Tensor:
    blk = model.model.layers[b]
    H = model.config.num_attention_heads
    kv = model.config.num_key_value_heads
    hd = int(getattr(model.config, "head_dim", model.config.hidden_size // H))
    w_in = blk.input_layernorm.weight.to(torch.float32)
    if postnorm:
        w_in = 1.0 + w_in
    WV = blk.self_attn.v_proj.weight.to(torch.float32).view(kv, hd, -1)
    WO = blk.self_attn.o_proj.weight.to(torch.float32).view(-1, H, hd)
    if postnorm:
        WO = WO * (1.0 + blk.post_attention_layernorm.weight.to(torch.float32))[:, None, None]
    M = torch.zeros(WO.shape[0], WO.shape[0], device=WO.device)
    for h in range(H):
        M += WO[:, h, :] @ WV[h // (H // kv)]
    return M * w_in[None, :]


@torch.inference_mode()
def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    parser.add_argument("--presentation", default="assistant")
    parser.add_argument("--markers", type=Path, default=Path("data/markers_presence.json"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--offline", action=BooleanOptionalAction, default=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    short = args.model.split("/")[-1]
    work = Path("work") if args.model == INSTRUCT_MODEL else Path(f"work-{short}")
    out = args.out or (work / "lex2vec.json")
    model, _ = load_model(offline=args.offline, model_name=args.model)
    dev = model.device
    V = load_diff_vectors(device=str(dev), tag=vector_tag_for(args.model, args.presentation))
    n_blocks = len(model.model.layers)
    rl = read_layers_for(n_blocks)
    mid, last = rl[len(rl) // 2], rl[-1]
    postnorm = hasattr(model.model.layers[0], "pre_feedforward_layernorm")
    E = model.model.embed_tokens.weight.to(torch.float32)
    if postnorm:
        E = E * (model.config.hidden_size**0.5)
    mk = json.loads(args.markers.read_text())
    K = V.shape[1]
    rng = np.random.default_rng(0)
    vocab = E.shape[0]

    e = torch.zeros(K, E.shape[1], device=dev)
    e_rand = torch.zeros_like(e)
    for k in range(K):
        c, a = mk["concept"][str(k)], mk["antagonist"][str(k)]
        e[k] = E[c].mean(0) - E[a].mean(0)
        e_rand[k] = E[rng.integers(0, vocab, len(c)).tolist()].mean(0) - E[
            rng.integers(0, vocab, len(a)).tolist()
        ].mean(0)
    ops_lo = [(b, block_operator(model, b, postnorm)) for b in range(0, mid)]
    ops_hi = [(b, block_operator(model, b, postnorm)) for b in range(mid, last)]
    res: dict[str, Any] = {"model": args.model, "mid": mid, "last": last, "n": K}

    for name, x, ops, tgt in (
        ("lexicon->v_mid", e, ops_lo, V[mid]),
        ("random_lexicon->v_mid", e_rand, ops_lo, V[mid]),
        ("lexicon->v_last", e, ops_lo + ops_hi, V[last]),
        ("v_mid->v_last", V[mid], ops_hi, V[last]),
    ):
        target = tgt
        res_c: dict[str, Any] = {}
        predictions_mix_best = [-1.0, 0.0]
        S = torch.zeros_like(x)
        for _, M in ops:
            S += x @ M.T
        predictions = {
            "raw": torch.nn.functional.normalize(x, dim=-1),
            "hop1": torch.nn.functional.normalize(S, dim=-1),
        }
        best = None
        for c in [10**ee for ee in np.linspace(-4, 1, 26)]:
            v = x.clone()
            for _, M in ops:
                v = torch.nn.functional.normalize(v + c * (v @ M.T), dim=-1)
            cos = float((torch.nn.functional.normalize(v, dim=-1) * target).sum(-1).median())
            if best is None or cos > best[0]:
                best = (cos, c, v.clone())
            mix = torch.nn.functional.normalize(x + c * S, dim=-1)
            cm = float((mix * target).sum(-1).median())
            if "mix" not in predictions or cm > predictions_mix_best[0]:
                predictions["mix"] = mix
                predictions_mix_best[0] = cm
                predictions_mix_best[1] = c
        assert best is not None
        predictions["iter"] = best[2]
        res_c["iter_c"] = float(best[1])
        res_c["mix_c"] = predictions_mix_best[1]
        for pn, p in predictions.items():
            cos = (p * target).sum(-1)
            res_c[pn] = {
                "cos_median": float(cos.median()),
                "cos_p25": float(cos.quantile(0.25)),
                "cos_p75": float(cos.quantile(0.75)),
                "retrieval_top1": float(
                    (
                        (
                            torch.nn.functional.normalize(p, dim=-1) @ torch.nn.functional.normalize(target, dim=-1).T
                        ).argmax(1)
                        == torch.arange(len(p), device=p.device)
                    )
                    .float()
                    .mean()
                ),
            }
        res[name] = res_c
        log.info(
            "%-24s "
            + "  ".join(
                f"{pn}: cos {res_c[pn]['cos_median']:.3f} (retrieval {100 * res_c[pn]['retrieval_top1']:.0f}%%)"
                for pn in predictions
            ),
            name,
        )
    W_U = model.lm_head.weight.to(torch.float32)
    lens = torch.nn.functional.normalize(V[mid], dim=-1) @ W_U.T
    hit = []
    for k in range(K):
        c, a = mk["concept"][str(k)], mk["antagonist"][str(k)]
        hit.append(float(lens[k, c].mean() - lens[k, a].mean()))
    res["logit_lens_marker_gap_median"] = float(np.median(hit))
    res["logit_lens_marker_gap_frac_positive"] = float(np.mean(np.array(hit) > 0))
    log.info(
        "logit lens of v_mid: concept-marker minus antagonist-marker logit median %.3f, positive for %.0f%% of concepts",
        res["logit_lens_marker_gap_median"],
        100 * res["logit_lens_marker_gap_frac_positive"],
    )
    out.write_text(json.dumps(res, indent=1))
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
