from __future__ import annotations

import collections
import json
import logging
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from exp_attn_interp.amp import AMPS, np_ranks
from exp_attn_interp.common import ValueCatcher, load_model
from exp_attn_interp.config import INSTRUCT_MODEL, vector_tag_for
from exp_attn_interp.decomp import MlpCatcher, read_layers_for
from exp_attn_interp.vectors import load_diff_vectors

log = logging.getLogger("inject")
TRIGGER = Path("../exp_trigger_prompts")


@torch.inference_mode()
def last_pos_writes(model, catcher, mlps, out, L: int, last: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head [last-L, H, hidden] and per-MLP [last-L, hidden] writes at the LAST position for blocks L..last-1."""
    heads, kv, hd = catcher.heads, catcher.kv_heads, catcher.head_dim
    group = heads // kv
    device = model.device
    hw, mw = [], []
    for b in range(L, last):
        attn = out.attentions[b][0].to(torch.float32)[:, -1, :]
        vals = catcher.values[b][0].view(-1, kv, hd).to(torch.float32)
        Vv = vals[:, torch.arange(heads, device=device) // group]
        ctx = torch.einsum("ht,thd->hd", attn, Vv)
        scale, gain = mlps.attn_scale[b], mlps.attn_gain[b]
        if scale is not None:
            ctx = ctx * scale[-1]
        W_O = model.model.layers[b].self_attn.o_proj.weight.to(torch.float32).view(-1, heads, hd)
        hwrite = torch.einsum("hd,ohd->ho", ctx, W_O)
        if gain is not None:
            hwrite = hwrite * gain[None, :]
        mo = mlps.out[b]
        assert mo is not None
        hw.append(hwrite)
        mw.append(mo[0, -1].to(torch.float32))
    return torch.stack(hw), torch.stack(mw)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    parser.add_argument("--layer", type=int, default=21)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--prompts-per-pair", type=int, default=2)
    parser.add_argument("--chunk", type=int, default=-1, help="64 pairs per chunk; -1 = all")
    parser.add_argument("--every", type=int, default=1, help="use every k-th pair (a spread subset)")
    parser.add_argument("--out", type=Path, default=Path("work/inject"))
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--offline", action=BooleanOptionalAction, default=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.analyze:
        files = sorted(args.out.glob("pair-*.safetensors"))
        rows: dict[int, dict] = {}
        HP = {}
        for f in files:
            with safe_open(str(f), framework="np") as h:
                k = int(h.get_tensor("pair_id"))
                rows[k] = {
                    n: float(h.get_tensor(n)) for n in ("survival", "carry", "gain", "cos_L_last", "delta_proj_last")
                }
                hp, mp = h.get_tensor("head_proj_last"), h.get_tensor("mlp_proj_last")
                rows[k]["attn"] = float(hp.sum())
                rows[k]["mlp"] = float(mp.sum())
                rows[k]["recon"] = rows[k]["delta_proj_last"] + rows[k]["attn"] + rows[k]["mlp"]
                HP[k] = hp
                L, last = int(h.metadata()["L"]), int(h.metadata()["last"])
        ks = sorted(rows)
        S = np.array([rows[k]["survival"] for k in ks])
        C = np.array([rows[k]["carry"] for k in ks])
        G = np.array([rows[k]["gain"] for k in ks])
        cos = np.array([rows[k]["cos_L_last"] for k in ks])
        att = np.array([rows[k]["attn"] for k in ks])
        mlp = np.array([rows[k]["mlp"] for k in ks])
        dl = np.array([rows[k]["delta_proj_last"] for k in ks])
        rec = np.array([rows[k]["recon"] for k in ks])
        res: dict[str, Any] = {
            "n": len(ks),
            "L": L,
            "last": last,
            "exactness_median_relerr": float(np.median(np.abs(rec - S) / np.abs(S).clip(1e-6))),
            "survival_median": float(np.median(S)),
            "carry_median": float(np.median(C)),
            "gain_median": float(np.median(G)),
            "share_of_survival_median": {
                "injected_delta": float(np.median(dl / S)),
                "attention": float(np.median(att / S)),
                "mlp": float(np.median(mlp / S)),
            },
        }
        log.info(
            "n=%d exact (median rel.err %.1e). survival <D,v_last>/|delta| median %.3f; carry <D,v_L>/|delta| %.3f; gain |D|/|delta| %.2f",
            len(ks),
            res["exactness_median_relerr"],
            res["survival_median"],
            res["carry_median"],
            res["gain_median"],
        )
        log.info(
            "who transports the injection to the output direction (share of survival, median): delta itself %.3f, attention %.3f, MLP %.3f",
            *res["share_of_survival_median"].values(),
        )
        sp = lambda a, b: float(np.corrcoef(np_ranks(a), np_ranks(b))[0, 1])  # noqa: E731
        res["rho_survival_vs_cos"] = sp(S, cos)
        res["rho_carry_vs_cos"] = sp(C, cos)
        res["rho_gain_vs_cos"] = sp(G, cos)
        log.info(
            "survival vs cos(v_L, v_last): rho %+.3f; carry vs cos %+.3f; gain vs cos %+.3f",
            res["rho_survival_vs_cos"],
            res["rho_carry_vs_cos"],
            res["rho_gain_vs_cos"],
        )
        steer = json.loads(Path("work/steerability.json").read_text())["qwen"]
        kj = [i for i, k in enumerate(ks) if str(k) in steer]
        sj = np.array([steer[str(ks[i])] for i in kj])
        for name, x in (
            ("survival", S),
            ("carry", C),
            ("gain", G),
            ("cos", cos),
            ("attn_transport", att),
            ("mlp_transport", mlp),
            ("attn_share_of_survival", att / S),
            ("survival_minus_delta", S - dl),
        ):
            res[f"rho_{name}_vs_judged"] = sp(x[kj], sj)
            log.info("judged steerability vs %-24s rho %+.3f", name, res[f"rho_{name}_vs_judged"])
        X = np.stack([np.ones(len(kj)), np_ranks(cos[kj])], 1)
        rs = np_ranks(S[kj]) - X @ np.linalg.lstsq(X, np_ranks(S[kj]), rcond=None)[0]
        rj = np_ranks(sj) - X @ np.linalg.lstsq(X, np_ranks(sj), rcond=None)[0]
        res["rho_survival_vs_judged_given_cos"] = float(np.corrcoef(rs, rj)[0, 1])
        log.info("judged steerability vs survival with cos removed: rho %+.3f", res["rho_survival_vs_judged_given_cos"])
        Hm = np.mean([HP[k] for k in ks], 0)
        with safe_open(str(sorted(Path("work/decomp").glob("item-*.safetensors"))[0]), framework="np") as h:
            pass
        from exp_attn_interp.amp import empirical_builder_share

        emp, _ = empirical_builder_share(Path("work/decomp"))
        res["rho_transport_vs_builder_share_heads"] = sp(Hm.ravel(), emp[L:last].ravel())
        top = np.argsort(-Hm.ravel())[:10]
        Hh = Hm.shape[1]
        res["top_transport_heads"] = [[int(L + i // Hh), int(i % Hh), float(Hm.ravel()[i])] for i in top]
        res["amplifier_transport_rank"] = {
            f"{b}:{h}": int((Hm.ravel() > Hm[b - L, h]).sum()) + 1 for b, h in AMPS[INSTRUCT_MODEL] if b >= L
        }
        log.info(
            "heads transporting the injection vs builder share (blocks %d..%d): rho %+.3f; top transport heads %s; amplifier ranks %s",
            L,
            last - 1,
            res["rho_transport_vs_builder_share_heads"],
            [(a, b, round(c, 4)) for a, b, c in res["top_transport_heads"]],
            res["amplifier_transport_rank"],
        )
        (args.out / "injectan.json").write_text(json.dumps(res, indent=1))
        return
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
    model, tokenizer = load_model(offline=args.offline, model_name=args.model)
    catcher = ValueCatcher(model.model)
    mlps = MlpCatcher(model.model)
    vectors = load_diff_vectors(device=str(model.device), tag=vector_tag_for(args.model))
    n_blocks = len(model.model.layers)
    last = read_layers_for(n_blocks)[-1]
    L, a = args.layer, args.alpha
    kw = {"enable_thinking": False} if "Qwen3" in args.model else {}
    pairs = sorted(k for k in by_pair if len(by_pair[k]) >= args.prompts_per_pair)
    if args.chunk >= 0:
        pairs = pairs[args.chunk * 64 : (args.chunk + 1) * 64]
    pairs = pairs[:: args.every]
    delta_holder: dict[str, torch.Tensor | None] = {"d": None}

    def hook(_m: Any, _i: Any, output: Any) -> Any:
        d = delta_holder["d"]
        if d is None:
            return None
        residual = output[0] if isinstance(output, tuple) else output
        residual = residual.clone()
        residual[:, -1, :] += d.to(residual.dtype)
        return (residual, *output[1:]) if isinstance(output, tuple) else residual

    if L not in norms:
        with torch.inference_mode():
            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": by_pair[k][0]}], tokenize=False, add_generation_prompt=True, **kw
                )
                for k in pairs[:64]
            ]
            tokenizer.padding_side = "left"
            enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            o = model(**enc, output_hidden_states=True, use_cache=False)
            m = enc.attention_mask.bool()
            norms[L] = float(o.hidden_states[L].to(torch.float32).norm(dim=-1)[m].median())
        log.info("row %d not in the calibration file; median prompt-token norm %.1f", L, norms[L])
    model.model.layers[L - 1].register_forward_hook(hook)
    for n, k in enumerate(pairs):
        path = args.out / f"pair-{k:04d}.safetensors"
        if path.exists():
            continue
        v_L, v_last = vectors[L][k], vectors[last][k]
        delta = v_L * (a * norms[L])
        D_all, hw_d, mw_d = [], [], []
        for j in range(args.prompts_per_pair):
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": by_pair[k][j]}], tokenize=False, add_generation_prompt=True, **kw
            )
            enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.inference_mode():
                delta_holder["d"] = None
                o0 = model(**enc, output_attentions=True, output_hidden_states=True, use_cache=False)
                h0 = o0.hidden_states[last][0, -1].to(torch.float32)
                hw0, mw0 = last_pos_writes(model, catcher, mlps, o0, L, last)
                delta_holder["d"] = delta
                o1 = model(**enc, output_attentions=True, output_hidden_states=True, use_cache=False)
                h1 = o1.hidden_states[last][0, -1].to(torch.float32)
                hw1, mw1 = last_pos_writes(model, catcher, mlps, o1, L, last)
                delta_holder["d"] = None
            D_all.append(h1 - h0)
            hw_d.append(hw1 - hw0)
            mw_d.append(mw1 - mw0)
        D = torch.stack(D_all).mean(0)
        HW = torch.stack(hw_d).mean(0)
        MW = torch.stack(mw_d).mean(0)
        dn = float(delta.norm())
        rec = {
            "pair_id": torch.tensor(k),
            "survival": (D @ v_last / dn).cpu(),
            "carry": (D @ v_L / dn).cpu(),
            "gain": (D.norm() / dn).cpu(),
            "cos_L_last": (v_L @ v_last).cpu(),
            "delta_proj_last": (delta @ v_last / dn).cpu(),
            "head_proj_last": (HW @ v_last / dn).cpu(),
            "mlp_proj_last": (MW @ v_last / dn).cpu(),
            "head_proj_L": (HW @ v_L / dn).cpu(),
            "mlp_proj_L": (MW @ v_L / dn).cpu(),
        }
        save_file(rec, str(path), metadata={"L": str(L), "last": str(last)})
        recon = float(rec["delta_proj_last"] + rec["head_proj_last"].sum() + rec["mlp_proj_last"].sum())
        if n < 3 or n % 50 == 0:
            log.info(
                "%d/%d pair %d: survival %.3f (recon %.3f) carry %.3f gain %.2f cos %.2f | attn %.3f mlp %.3f delta %.3f",
                n,
                len(pairs),
                k,
                float(rec["survival"]),
                recon,
                float(rec["carry"]),
                float(rec["gain"]),
                float(rec["cos_L_last"]),
                float(rec["head_proj_last"].sum()),
                float(rec["mlp_proj_last"].sum()),
                float(rec["delta_proj_last"]),
            )


if __name__ == "__main__":
    main()
