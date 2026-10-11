from __future__ import annotations

import collections
import json
import logging
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import Any

import numpy as np
import torch

from exp_attn_interp.amp import np_ranks
from exp_attn_interp.common import load_model
from exp_attn_interp.config import INSTRUCT_MODEL, vector_tag_for
from exp_attn_interp.decomp import read_layers_for
from exp_attn_interp.steerabl import steer
from exp_attn_interp.vectors import load_diff_vectors

log = logging.getLogger("gentraj")
TRIGGER = Path("../exp_trigger_prompts")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    parser.add_argument("--layer", type=int, default=21)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--n-pairs", type=int, default=300)
    parser.add_argument("--prompts-per-pair", type=int, default=2)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--out", type=Path, default=Path("work/gentraj"))
    parser.add_argument(
        "--family",
        default="qwen",
        choices=["qwen", "llama", "gemma"],
        help="key into work/steerability.json and row of work/rotation_per_concept.npy",
    )
    parser.add_argument(
        "--layers",
        default="",
        help="comma list of injection rows; norms computed from the prompts' "
        "mean residual norm at each row; one gentraj_L<row>.json per row (base run shared)",
    )
    parser.add_argument(
        "--switch-off-at",
        type=int,
        default=0,
        help="add a third run: hook active only for the "
        "first S generated tokens, then removed (does the effect persist through the context?)",
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
    rot = np.load("work/rotation_per_concept.npy")[["qwen", "llama", "gemma"].index(args.family)]
    steer_j = json.loads(Path("work/steerability.json").read_text())[args.family]
    ok = [k for k in sorted(by_pair) if len(by_pair[k]) >= args.prompts_per_pair and str(k) in steer_j]
    order = sorted(ok, key=lambda k: rot[k])
    pairs = [order[i] for i in np.linspace(0, len(order) - 1, args.n_pairs).astype(int)]
    model, tokenizer = load_model(offline=args.offline, model_name=args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    vectors = load_diff_vectors(device=str(model.device), tag=vector_tag_for(args.model))
    n_blocks = len(model.model.layers)
    last = read_layers_for(n_blocks)[-1]
    L, a = args.layer, args.alpha
    kw = {"enable_thinking": False} if "Qwen3" in args.model else {}
    stim = [(k, j) for k in pairs for j in range(args.prompts_per_pair)]
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": by_pair[k][j]}], tokenize=False, add_generation_prompt=True, **kw
        )
        for k, j in stim
    ]
    T = args.new_tokens
    if args.layers:
        rows = [int(x) for x in args.layers.split(",")]
        norms = {}
        with torch.inference_mode():
            enc = tokenizer(rendered[:64], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            o = model(**enc, output_hidden_states=True, use_cache=False)
            m = enc.attention_mask.bool()
            for L in rows:
                h = o.hidden_states[L].to(torch.float32)
                norms[L] = float(h.norm(dim=-1)[m].median())
        log.info("norms from prompts: %s", {k: round(v, 1) for k, v in norms.items()})
        base = generate_traj(model, tokenizer, rendered, stim, vectors, last, T, args.batch, rows[0], None)
        log.info("base done")
        summary: dict[str, Any] = {"rows": rows, "norms": norms, "alpha": a, "per_row": {}}
        for L in rows:
            st = generate_traj(model, tokenizer, rendered, stim, vectors, last, T, args.batch, L, a * norms[L])
            eff = st - base
            np.save(args.out / f"effect_L{L}.npy", eff)
            per = {}
            for k in pairs:
                idx = [i for i, (kk, _) in enumerate(stim) if kk == k]
                per[k] = float(np.nanmean(eff[idx]))
            ks = [k for k in pairs if not np.isnan(per[k])]
            x = np.array([per[k] for k in ks])
            align = np.array([float(vectors[L][k] @ vectors[last][k]) for k in ks])
            rr = np.array([rot[k] for k in ks])
            sj = np.array([steer_j[str(k)] for k in ks])
            c = np.nanmedian(eff, 0)
            r: dict[str, Any] = {
                "effect_median_mean": float(np.nanmedian(x)),
                "curve_t0_t16_t32_t63": [float(c[0]), float(c[16]), float(c[32]), float(c[-1])],
                "align_median": float(np.median(align)),
                "rho_effect_vs_align_L_last": float(np.corrcoef(np_ranks(x), np_ranks(align))[0, 1]),
                "rho_effect_vs_rotation_mid_last": float(np.corrcoef(np_ranks(x), np_ranks(rr))[0, 1]),
                "rho_effect_vs_judged": float(np.corrcoef(np_ranks(x), np_ranks(sj))[0, 1]),
            }
            summary["per_row"][L] = r
            log.info(
                "ROW %d (norm %.0f, median align %.2f): mean effect %.1f, curve %s | rho effect vs align(L,last) %+.3f, "
                "vs rotation(mid,last) %+.3f, vs judged %+.3f",
                L,
                norms[L],
                r["align_median"],
                r["effect_median_mean"],
                [round(v, 1) for v in r["curve_t0_t16_t32_t63"]],
                r["rho_effect_vs_align_L_last"],
                r["rho_effect_vs_rotation_mid_last"],
                r["rho_effect_vs_judged"],
            )
        (args.out / "gentraj_rows.json").write_text(json.dumps(summary, indent=1))
        return
    conditions = ["base", "steer"] + (["switch"] if args.switch_off_at else [])
    traj: dict[str, np.ndarray] = {c: np.full((len(stim), T), np.nan) for c in conditions}
    counter: dict[str, Any] = {"n": 0, "delta": None}

    def switch_hook(_m: Any, _i: Any, output: Any) -> Any:
        d = counter.get("delta")
        if d is None or counter["n"] >= args.switch_off_at:
            counter["n"] += 1
            return None
        counter["n"] += 1
        residual = output[0] if isinstance(output, tuple) else output
        residual = residual.clone()
        residual[:, -1, :] += d.to(residual.dtype)
        return (residual, *output[1:]) if isinstance(output, tuple) else residual

    for cond in conditions:
        for s in range(0, len(stim), args.batch):
            chunk, rend = stim[s : s + args.batch], rendered[s : s + args.batch]
            enc = tokenizer(rend, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            delta = None
            if cond in ("steer", "switch"):
                delta = torch.stack([vectors[L][k] for k, _ in chunk]) * (a * norms[L])
            if cond == "switch":
                counter["n"], counter["delta"] = 0, delta
                handle = model.model.layers[L - 1].register_forward_hook(switch_hook)
                try:
                    with torch.inference_mode():
                        gen = model.generate(
                            **enc,
                            max_new_tokens=T,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                            output_hidden_states=True,
                            return_dict_in_generate=True,
                        )
                finally:
                    handle.remove()
                    counter["delta"] = None
            else:
                with torch.inference_mode(), steer(model, L if delta is not None else None, delta):
                    gen = model.generate(
                        **enc,
                        max_new_tokens=T,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        output_hidden_states=True,
                        return_dict_in_generate=True,
                    )
            vl = torch.stack([vectors[last][k] for k, _ in chunk])
            seq = gen.sequences[:, enc.input_ids.shape[1] :]
            for t, hs in enumerate(gen.hidden_states):
                h = hs[last][:, -1, :].to(torch.float32)
                proj = (h * vl).sum(-1).cpu().numpy()
                alive = (
                    (seq[:, t] != tokenizer.pad_token_id).cpu().numpy()
                    if t < seq.shape[1]
                    else np.zeros(len(chunk), bool)
                )
                for i in range(len(chunk)):
                    if alive[i] and t < T:
                        traj[cond][s + i, t] = proj[i]
            log.info("%s %d/%d", cond, min(s + args.batch, len(stim)), len(stim))
    eff = traj["steer"] - traj["base"]
    np.save(args.out / "effect.npy", eff)
    if "switch" in traj:
        sw = traj["switch"] - traj["base"]
        np.save(args.out / "effect_switch.npy", sw)
        S = args.switch_off_at
        c_on, c_sw = np.nanmedian(eff, 0), np.nanmedian(sw, 0)
        log.info(
            "SWITCH-OFF at token %d: effect with hook on / after hook removed: t%d %.1f / %.1f, t%d %.1f / %.1f, "
            "t%d %.1f / %.1f, t%d %.1f / %.1f",
            S,
            S - 1,
            c_on[S - 1],
            c_sw[S - 1],
            S,
            c_on[S],
            c_sw[S],
            S + 8,
            c_on[S + 8],
            c_sw[S + 8],
            T - 1,
            c_on[-1],
            c_sw[-1],
        )
        persist = np.nanmean(sw[:, S + 4 :], 1) / np.nanmean(eff[:, S + 4 :], 1)
        log.info(
            "SWITCH-OFF persistence (effect after removal / effect with hook, mean over t>=%d): median %.3f",
            S + 4,
            float(np.nanmedian(persist)),
        )
    np.save(args.out / "pairs.npy", np.array([k for k, _ in stim]))
    per: dict[int, dict[str, float]] = {}
    for k in pairs:
        rows = eff[[i for i, (kk, _) in enumerate(stim) if kk == k]]
        m = np.nanmean(rows, 0)
        q = T // 4
        x = np.arange(T)
        valid = ~np.isnan(m)
        slope = float(np.polyfit(x[valid], m[valid], 1)[0]) if valid.sum() > 4 else float("nan")
        per[k] = {
            "mean": float(np.nanmean(m)),
            "first_q": float(np.nanmean(m[:q])),
            "last_q": float(np.nanmean(m[-q:])),
            "slope": slope,
            "first_token": float(m[0]),
        }
    ks = [k for k in pairs if not np.isnan(per[k]["mean"])]
    rr = np.array([rot[k] for k in ks])
    sj = np.array([steer_j[str(k)] for k in ks])
    res: dict[str, Any] = {
        "n": len(ks),
        "layer": L,
        "alpha": a,
        "new_tokens": T,
        "effect_curve_median": np.nanmedian(eff, 0).tolist(),
        "effect_curve_by_rotation_tercile": {},
    }
    terc = np.array_split(np.argsort(rr), 3)
    for ti, tt in enumerate(terc):
        idx = [i for i, (kk, _) in enumerate(stim) if kk in {ks[j] for j in tt}]
        res["effect_curve_by_rotation_tercile"][ti] = np.nanmedian(eff[idx], 0).tolist()
    if "switch" in traj:
        sw_pp = {}
        for k in pairs:
            idx = [i for i, (kk, _) in enumerate(stim) if kk == k]
            sw_pp[k] = float(np.nanmean(sw[idx][:, args.switch_off_at + 4 :])) / max(
                float(np.nanmean(eff[idx][:, args.switch_off_at + 4 :])), 1e-6
            )
        for k in ks:
            per[k]["persistence"] = sw_pp[k]
    for name in ("mean", "first_q", "last_q", "slope", "first_token") + (("persistence",) if "switch" in traj else ()):
        x = np.array([per[k][name] for k in ks])
        good = ~np.isnan(x)
        res[f"rho_{name}_vs_rotation"] = float(np.corrcoef(np_ranks(x[good]), np_ranks(rr[good]))[0, 1])
        res[f"rho_{name}_vs_judged"] = float(np.corrcoef(np_ranks(x[good]), np_ranks(sj[good]))[0, 1])
        res[f"{name}_by_tercile"] = [float(np.nanmean(x[tt])) for tt in terc]
        log.info(
            "%-11s effect: vs rotation rho %+.3f | vs judged rho %+.3f | by rotation tercile (rotating->stable) %s",
            name,
            res[f"rho_{name}_vs_rotation"],
            res[f"rho_{name}_vs_judged"],
            [round(v, 2) for v in res[f"{name}_by_tercile"]],
        )
    c = res["effect_curve_median"]
    log.info(
        "median effect curve: t0 %.2f t%d %.2f t%d %.2f t%d %.2f",
        c[0],
        T // 4,
        c[T // 4],
        T // 2,
        c[T // 2],
        T - 1,
        c[-1],
    )
    (args.out / "gentraj.json").write_text(json.dumps(res, indent=1))


def generate_traj(model, tokenizer, rendered, stim, vectors, last, T, batch, L, delta_scale):
    """Per-position projection onto v(last) of the row-`last` residual; delta_scale None = unsteered."""
    out = np.full((len(stim), T), np.nan)
    for s in range(0, len(stim), batch):
        chunk, rend = stim[s : s + batch], rendered[s : s + batch]
        enc = tokenizer(rend, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        delta = None if delta_scale is None else torch.stack([vectors[L][k] for k, _ in chunk]) * delta_scale
        with torch.inference_mode(), steer(model, L if delta is not None else None, delta):
            gen = model.generate(
                **enc,
                max_new_tokens=T,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
        vl = torch.stack([vectors[last][k] for k, _ in chunk])
        seq = gen.sequences[:, enc.input_ids.shape[1] :]
        for t, hs in enumerate(gen.hidden_states):
            if t >= T:
                break
            proj = (hs[last][:, -1, :].to(torch.float32) * vl).sum(-1).cpu().numpy()
            alive = (seq[:, t] != tokenizer.pad_token_id).cpu().numpy()
            out[s : s + len(chunk), t] = np.where(alive, proj, np.nan)
    return out


if __name__ == "__main__":
    main()
