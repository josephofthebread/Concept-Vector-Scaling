from __future__ import annotations

import collections
import json
import logging
import re
from argparse import ArgumentParser, BooleanOptionalAction
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

from exp_attn_interp.ablate import HeadAblator
from exp_attn_interp.amp import AMPS
from exp_attn_interp.common import load_model
from exp_attn_interp.config import INSTRUCT_MODEL, vector_tag_for
from exp_attn_interp.decomp import read_layers_for
from exp_attn_interp.vectors import load_diff_vectors

log = logging.getLogger("steerabl")
TRIGGER = Path("../exp_trigger_prompts")


@contextmanager
def steer(model: Any, layer: int | None, delta: torch.Tensor | None):
    """Add per-row delta [B, hidden] to the current position at the output of block layer-1
    (= hidden state row `layer`), as in exp_trigger_prompts.steer."""
    if layer is None or delta is None:
        yield
        return

    def hook(_m: Any, _i: Any, output: Any) -> Any:
        residual = output[0] if isinstance(output, tuple) else output
        residual = residual.clone()
        residual[:, -1, :] += delta.to(residual.dtype)
        return (residual, *output[1:]) if isinstance(output, tuple) else residual

    handle = model.model.layers[layer - 1].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.inference_mode()
def text_projection(
    model, tokenizer, prompts_rendered: list[str], responses: list[str], vecs: torch.Tensor, read_last: int, batch: int
) -> list[float]:
    """Mean over response tokens of <h_read_last, v> from a clean pass (no hooks)."""
    out: list[float] = []
    for s in range(0, len(responses), batch):
        pr, rs, vv = prompts_rendered[s : s + batch], responses[s : s + batch], vecs[s : s + batch]
        full = [p + r for p, r in zip(pr, rs)]
        enc = tokenizer(full, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        plen = [len(tokenizer(p, add_special_tokens=False).input_ids) for p in pr]
        tot = enc.attention_mask.sum(1).tolist()
        o = model(**enc, output_hidden_states=True, use_cache=False)
        h = o.hidden_states[read_last][..., :].to(torch.float32)
        T = h.shape[1]
        for i in range(len(full)):
            n_resp = tot[i] - plen[i]
            if n_resp <= 0:
                out.append(float("nan"))
                continue
            hi = h[i, T - n_resp :]
            out.append(float((hi @ vv[i]).mean()))
    return out


def law_report(args, records: list[dict[str, Any]], pairs: list[int], rot: np.ndarray) -> None:
    from exp_attn_interp.amp import np_ranks

    steer_j = json.loads(Path("work/steerability.json").read_text())["qwen"]
    per_pair = {}
    for k in pairs:
        gg: dict[str, float] = {}
        for key in ("marker_rate", "projection"):
            st_ = [r[key] for r in records if r["cond"] == "steer" and r["pair"] == k]
            ba_ = [r[key] for r in records if r["cond"] == "base" and r["pair"] == k]
            gg[key] = float(np.nanmean(st_) - np.nanmean(ba_))
        per_pair[k] = gg
    ksj = [k for k in pairs if str(k) in steer_j]
    rotv = np.array([rot[k] for k in pairs])
    law: dict[str, Any] = {"n": len(pairs), "per_pair": {str(k): v for k, v in per_pair.items()}}
    for key in ("marker_rate", "projection"):
        g = np.array([per_pair[k][key] for k in pairs])
        law[f"rho_rotation_vs_{key}_gain"] = float(np.corrcoef(np_ranks(rotv), np_ranks(g))[0, 1])
        gj = np.array([per_pair[k][key] for k in ksj])
        sj = np.array([steer_j[str(k)] for k in ksj])
        law[f"rho_judge_vs_{key}_gain"] = float(np.corrcoef(np_ranks(gj), np_ranks(sj))[0, 1])
        terc = np.array_split(np.argsort(rotv), 3)
        law[f"{key}_gain_by_rotation_tercile"] = [float(np.mean(g[t])) for t in terc]
        log.info(
            "LAW %s: rho(cos(mid,last), steering gain) %+.3f over %d concepts; rho(judged steerability, gain) %+.3f; "
            "gain by rotation tercile (most rotating -> most stable) %s",
            key,
            law[f"rho_rotation_vs_{key}_gain"],
            len(pairs),
            law[f"rho_judge_vs_{key}_gain"],
            [round(x, 3) for x in law[f"{key}_gain_by_rotation_tercile"]],
        )
    for r in records:
        r.pop("response", None)
    (args.out / "law.json").write_text(json.dumps(law, indent=1))
    return


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    parser.add_argument("--layer", type=int, default=21, help="injection row (hook on block layer-1)")
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--n-pairs", type=int, default=48, help="stratified by rotation tercile")
    parser.add_argument("--prompts-per-pair", type=int, default=4)
    parser.add_argument("--new-tokens", type=int, default=96)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--rand-seeds", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("work/steerabl"))
    parser.add_argument(
        "--dose",
        default="",
        help="comma list of k: ablate the top-k empirical builders after the "
        "injection row vs random-k (dose curve) instead of the amplifier arms",
    )
    parser.add_argument("--reanalyze", action="store_true", help="load <out>/generations.jsonl instead of generating")
    parser.add_argument(
        "--law",
        action="store_true",
        help="only base+steer arms on many concepts: per-concept "
        "lexical steering gain vs rotation (the steering law on a judge-free readout)",
    )
    parser.add_argument(
        "--dose-source",
        default="empirical",
        choices=["empirical", "weights"],
        help="rank heads by the measured decomposition or by the weight-only OV copy score (work/copy_own.npy)",
    )
    parser.add_argument("--offline", action=BooleanOptionalAction, default=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    tag = args.model.replace("/", "_")
    norms = {
        int(k): float(v)
        for k, v in json.loads((TRIGGER / "work" / "steer" / f"norms-{tag}.json").read_text())["norms"].items()
    }
    by_pair: dict[int, list[str]] = collections.defaultdict(list)
    with (TRIGGER / "work" / "prompts.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            if r["kind"] == "neutral" and r.get("ok") and r.get("prompt"):
                by_pair[int(r["pair_id"])].append(r["prompt"])
    markers = json.loads(Path("data/markers_unique.json").read_text())["concept_words"]
    rot = np.load("work/rotation_per_concept.npy")[0]
    ok = [
        k
        for k in sorted(by_pair)
        if len(by_pair[k]) >= args.prompts_per_pair and str(k) in markers and len(markers[str(k)]) >= 2
    ]
    order = sorted(ok, key=lambda k: rot[k])
    n3 = args.n_pairs // 3
    third = len(order) // 3
    pairs = (
        [order[i] for i in np.linspace(0, third - 1, n3).astype(int)]
        + [order[third + i] for i in np.linspace(0, third - 1, n3).astype(int)]
        + [order[2 * third + i] for i in np.linspace(0, len(order) - 2 * third - 1, n3).astype(int)]
    )
    tercile = {k: t for t, ks in enumerate((pairs[:n3], pairs[n3 : 2 * n3], pairs[2 * n3 :])) for k in ks}
    log.info(
        "%d pairs (rotation cos terciles: low %.2f–%.2f, mid, high %.2f–%.2f)",
        len(pairs),
        rot[pairs[0]],
        rot[pairs[n3 - 1]],
        rot[pairs[2 * n3]],
        rot[pairs[-1]],
    )

    if args.reanalyze:
        saved: list[dict[str, Any]] = [json.loads(line) for line in (args.out / "generations.jsonl").open()]
        return law_report(args, saved, sorted({r["pair"] for r in saved}), rot)
    model, tokenizer = load_model(offline=args.offline, model_name=args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    hd = int(getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads))
    ablator = HeadAblator(model.model, hd)
    vectors = load_diff_vectors(device=str(model.device), tag=vector_tag_for(args.model))
    n_blocks = len(model.model.layers)
    read_last = read_layers_for(n_blocks)[-1]
    amps = AMPS[args.model]
    H = model.config.num_attention_heads
    rng = np.random.default_rng(0)
    rand_sets = [
        [(int(b), int(h)) for b, h in zip(rng.integers(args.layer + 1, read_last, 6), rng.integers(0, H, 6))]
        for _ in range(args.rand_seeds)
    ]
    L, a = args.layer, args.alpha
    steer_conditions: list[tuple[str, float, int, list[tuple[int, int]]]] = [
        ("base", 0.0, L, []),
        ("steer", a, L, []),
        ("steer_x2", 2 * a, L, []),
        ("steer_amp", a, L, amps),
        ("amp_only", 0.0, L, amps),
        ("steer_vlast", a, read_last, []),
    ]
    steer_conditions += [(f"steer_rand{s}", a, L, rs) for s, rs in enumerate(rand_sets)]
    ks: list[int] = []
    if args.law:
        steer_conditions = [("base", 0.0, L, []), ("steer", a, L, [])]
    if args.dose:
        from exp_attn_interp.amp import empirical_builder_share

        if args.dose_source == "weights":
            emp = np.load("work/copy_own.npy").mean(-1)
        else:
            emp, _ = empirical_builder_share(Path("work/decomp"))
        emp[:L] = -np.inf
        emp[read_last:] = -np.inf
        ranking = [(int(i // H), int(i % H)) for i in np.argsort(-emp.ravel())]
        ks = [int(x) for x in args.dose.split(",")]
        steer_conditions = [("base", 0.0, L, []), ("steer", a, L, [])]
        for k in ks:
            top = ranking[:k]
            steer_conditions += [(f"top{k}", 0.0, L, top), (f"steer_top{k}", a, L, top)]
            for sd in range(args.rand_seeds):
                pool = [(b, h) for b in range(L, read_last) for h in range(H)]
                rr = np.random.default_rng(100 * k + sd)
                pick = [pool[i] for i in rr.choice(len(pool), k, replace=False)]
                steer_conditions += [(f"rand{k}_{sd}", 0.0, L, pick), (f"steer_rand{k}_{sd}", a, L, pick)]
        log.info("dose mode: k=%s, %d conditions; top-6 after row %d: %s", ks, len(steer_conditions), L, ranking[:6])
    kw = {"enable_thinking": False} if "Qwen3" in args.model else {}
    stim = [(k, j, by_pair[k][j]) for k in pairs for j in range(args.prompts_per_pair)]
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, **kw
        )
        for _, _, p in stim
    ]
    records: list[dict[str, Any]] = []
    for name, alpha, vrow, abl in steer_conditions:
        ablator.set(abl)
        for s in range(0, len(stim), args.batch):
            chunk, rend = stim[s : s + args.batch], rendered[s : s + args.batch]
            enc = tokenizer(rend, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            plen = enc.input_ids.shape[1]
            delta = None
            if alpha != 0.0:
                delta = torch.stack([vectors[vrow][k] for k, _, _ in chunk]) * (alpha * norms[L])
            with torch.inference_mode(), steer(model, L if alpha != 0.0 else None, delta):
                gen = model.generate(
                    **enc, max_new_tokens=args.new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id
                )
            for i, (k, j, p) in enumerate(chunk):
                g = gen[i, plen:]
                n_tok = int((g != tokenizer.pad_token_id).sum())
                text = tokenizer.decode(g, skip_special_tokens=True)
                words = markers[str(k)]
                if not words or n_tok == 0:
                    rate = 0.0
                else:
                    pat = re.compile(
                        r"\b(" + "|".join(re.escape(x.strip().lower()) for x in words if x.strip()) + r")\w*", re.I
                    )
                    rate = 100.0 * len(pat.findall(text)) / n_tok
                w = text.split()
                grams = [tuple(w[i : i + 4]) for i in range(len(w) - 3)]
                records.append(
                    {
                        "cond": name,
                        "pair": k,
                        "slot": j,
                        "tercile": tercile[k],
                        "rotation": float(rot[k]),
                        "n_tokens": n_tok,
                        "response": text,
                        "rendered": rend[i],
                        "marker_rate": rate,
                        "repetition": 0.0 if not grams else 1.0 - len(set(grams)) / len(grams),
                    }
                )
        ablator.set([])
        log.info("%s: %d generations", name, len(stim))
    ablator.set([])
    vecs = torch.stack([vectors[read_last][r["pair"]] for r in records])
    proj = text_projection(
        model,
        tokenizer,
        [r["rendered"] for r in records],
        [r["response"] for r in records],
        vecs,
        read_last,
        args.batch,
    )
    for rec, pv in zip(records, proj):
        rec["projection"] = pv
        rec.pop("rendered")
    with (args.out / "generations.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    def mean(cond: str, key: str, terc: int | None = None) -> float:
        xs = [r[key] for r in records if r["cond"] == cond and (terc is None or r["tercile"] == terc)]
        return float(np.nanmean(xs))

    summary: dict = {
        "layer": L,
        "alpha": a,
        "n_pairs": len(pairs),
        "amplifiers": amps,
        "random_sets": rand_sets,
        "conditions": {},
        "dose": {},
    }
    if args.law:
        return law_report(args, records, pairs, rot)
    if ks:
        for key in ("marker_rate", "projection"):
            gain = mean("steer", key) - mean("base", key)
            summary["dose"][key] = {"gain": gain, "k": {}}
            for k in ks:
                gt = mean(f"steer_top{k}", key) - mean(f"top{k}", key)
                gr = float(
                    np.mean(
                        [mean(f"steer_rand{k}_{sd}", key) - mean(f"rand{k}_{sd}", key) for sd in range(args.rand_seeds)]
                    )
                )
                summary["dose"][key]["k"][k] = {
                    "top_gain": gt,
                    "rand_gain": gr,
                    "top_kept": gt / gain if gain else float("nan"),
                    "rand_kept": gr / gain if gain else float("nan"),
                    "unsteered_top_level": mean(f"top{k}", key),
                    "unsteered_base_level": mean("base", key),
                }
                log.info(
                    "%s k=%d: steering gain with top-k builders ablated %.3f (%.0f%% kept) vs random-k %.3f (%.0f%% kept)",
                    key,
                    k,
                    gt,
                    100 * gt / gain if gain else 0,
                    gr,
                    100 * gr / gain if gain else 0,
                )
        for r in records:
            r.pop("response", None)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
        log.info("wrote %s", args.out)
        return
    for name, *_ in steer_conditions:
        summary["conditions"][name] = {key: mean(name, key) for key in ("marker_rate", "projection", "repetition")}
        summary["conditions"][name]["by_tercile"] = {
            t: {key: mean(name, key, t) for key in ("marker_rate", "projection")} for t in range(3)
        }
    for key in ("marker_rate", "projection"):
        base, st = mean("base", key), mean("steer", key)
        gain = st - base
        amp_gain = mean("steer_amp", key) - mean("amp_only", key)
        rand_gain = float(np.mean([mean(f"steer_rand{s}", key) for s in range(args.rand_seeds)])) - base
        summary[f"{key}_gain"] = {
            "steer": gain,
            "steer_amp_ablated": amp_gain,
            "steer_random_ablated": rand_gain,
            "amp_kept_fraction": amp_gain / gain if gain else float("nan"),
            "random_kept_fraction": rand_gain / gain if gain else float("nan"),
            "steer_x2": mean("steer_x2", key) - base,
            "steer_vlast": mean("steer_vlast", key) - base,
        }
        log.info(
            "%s: steering gain %.3f | amplifiers ablated %.3f (%.0f%% kept) | random ablated %.3f (%.0f%% kept) | "
            "x2 alpha %.3f | v(last) injected %.3f",
            key,
            gain,
            amp_gain,
            100 * amp_gain / gain if gain else 0,
            rand_gain,
            100 * rand_gain / gain if gain else 0,
            summary[f"{key}_gain"]["steer_x2"],
            summary[f"{key}_gain"]["steer_vlast"],
        )
        for t in range(3):
            g = mean("steer", key, t) - mean("base", key, t)
            ga = mean("steer_amp", key, t) - mean("amp_only", key, t)
            gv = mean("steer_vlast", key, t) - mean("base", key, t)
            log.info(
                "  tercile %d (rotation %s): gain %.3f | amp-ablated %.3f | v(last) %.3f",
                t,
                ["most rotating", "mid", "most stable"][t],
                g,
                ga,
                gv,
            )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
