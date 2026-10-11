from __future__ import annotations

import json
import logging
import time
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from exp_attn_interp.config import INSTRUCT_MODEL, vector_tag_for
from exp_attn_interp.decomp import MlpCatcher
from exp_attn_interp.posteer import unit
from exp_attn_interp.reach import steer_from
from exp_attn_interp.vectors import load_diff_vectors

log = logging.getLogger("contextmech")
TRIGGER = Path("../exp_trigger_prompts")
CARRIERS = [(22, 19), (22, 7), (27, 8), (30, 20), (30, 0), (30, 22), (34, 15)]
STOP = set(
    "the a an of and or to in on at for with is are was were be been being by as it its this that these those i you he she they we "
    "them their our your my me him her his one two either side may apply setting situation described below involves contrast between "
    "about whether while another not no but so if than then there here when which who whom what how do does did have has had".split()
)


def mlp_out(mlps: MlpCatcher, b: int) -> torch.Tensor:
    t = mlps.out[b]
    assert t is not None
    return t


@torch.inference_mode()
def main() -> None:
    p = ArgumentParser()
    p.add_argument("--model", default=INSTRUCT_MODEL)
    p.add_argument("--inp", type=Path, default=Path("work/judge-in/pole.jsonl"))
    p.add_argument("--ctx", type=Path, default=Path("work/posteer-ctx"), help="posteer-ctx dir (system messages)")
    p.add_argument("--hook", action=BooleanOptionalAction, default=False, help="re-apply the arm's injection at row 22")
    p.add_argument("--out", type=Path, default=Path("work/contextmech"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--offline", action=BooleanOptionalAction, default=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.offline)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager", local_files_only=args.offline
    ).eval()
    dev = model.device
    kw: dict[str, Any] = {"enable_thinking": False}
    mlps = MlpCatcher(model.model)
    V = load_diff_vectors(device="cpu", tag=vector_tag_for(args.model))
    K = V.shape[1]
    V22 = V[22]
    m22 = unit(V22.mean(0))
    Xc = V22 - V22.mean(0)
    _, _, Wt = torch.linalg.svd(Xc, full_matrices=False)
    Q, _ = torch.linalg.qr(torch.cat([m22[None], Wt[:16]], 0).T)
    Q = Q.to(dev)
    U = model.lm_head.weight.float() * model.model.norm.weight.float()[None, :]
    markers = json.loads(Path("data/markers_presence.json").read_text())
    ugap = torch.zeros(K, U.shape[1], device=dev)
    for k in range(K):
        ic, ia = markers["concept"].get(str(k), []), markers["antagonist"].get(str(k), [])
        if len(ic) >= 2 and len(ia) >= 2:
            ugap[k] = unit(U[ic].mean(0) - U[ia].mean(0))
    norms = {
        int(a): float(b)
        for a, b in json.loads((TRIGGER / "work" / "steer" / "norms-Qwen_Qwen3-8B.json").read_text())["norms"].items()
    }
    system: dict[int, str] = {}
    for a in ("0.5", "-0.5", "0"):
        f = args.ctx / f"a{a}" / "orig.jsonl"
        if f.exists():
            for line in f.open():
                r = json.loads(line)
                if r.get("system"):
                    system[int(r["pair_id"])] = r["system"]
    sit: dict[int, torch.Tensor] = {}
    for k, s in system.items():
        toks = [
            int(t)
            for t in tokenizer(s, add_special_tokens=False)["input_ids"]
            if str(tokenizer.decode([t])).strip().lower().strip('".,:;()') not in STOP
            and str(tokenizer.decode([t])).strip().isalpha()
        ]
        if toks:
            sit[k] = unit(U[toks].mean(0))
    rows = [json.loads(line) for line in args.inp.open()]
    if args.limit:
        rows = rows[: args.limit]
    out_path = args.out / ("hook.jsonl" if args.hook else "nohook.jsonl")
    done = set()
    if out_path.exists():
        done = {json.loads(line)["key"] for line in out_path.open()}
    rows = [r for r in rows if r["key"] not in done]
    log.info("%d texts to run (hook %s)", len(rows), args.hook)
    started = time.monotonic()
    with out_path.open("a") as sink:
        for n, r in enumerate(rows):
            k = int(r["pair_id"])
            alpha = float(r["alpha"])
            msgs = ([{"role": "system", "content": system[k]}] if r["arm"] == "ctx" and k in system else []) + [
                {"role": "user", "content": r["prompt"]}
            ]
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
            p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            r_ids = tokenizer(r["response"], add_special_tokens=False)["input_ids"][:600]
            inp = torch.tensor([p_ids + r_ids], device=dev)
            start = len(p_ids) - 1
            sys_len = 0
            if msgs[0]["role"] == "system":
                sys_len = len(
                    tokenizer(tokenizer.apply_chat_template(msgs[:1], tokenize=False, **kw), add_special_tokens=False)[
                        "input_ids"
                    ]
                )
            delta = None
            if args.hook and alpha != 0.0:
                delta = (V[22, k].to(dev) * (alpha * norms[22]))[None]
            with steer_from(model, 22, delta, [start]):
                out = model(input_ids=inp, output_hidden_states=True, output_attentions=True, use_cache=False)
            pos = slice(start + 1, inp.shape[1])
            h22 = out.hidden_states[22][0, pos].float()
            h35 = out.hidden_states[35][0, pos].float()
            v22 = V[22, k].to(dev)
            v35 = V[35, k].to(dev)
            mean22, mean35 = h22.mean(0), h35.mean(0)
            rec: dict[str, Any] = {
                "key": r["key"],
                "pair_id": k,
                "arm": r["arm"],
                "alpha": alpha,
                "n_resp": int(h22.shape[0]),
                "hook": args.hook,
            }
            rec["proj22"] = float(mean22 @ v22)
            rec["cos22"] = float(unit(mean22) @ v22)
            rec["proj35"] = float(mean35 @ v35)
            rec["cos35"] = float(unit(mean35) @ v35)
            rec["stance22"] = float(((mean22 @ Q) ** 2).sum() / (mean22 @ mean22))
            rec["ugap35"] = float(unit(mean35) @ ugap[k]) if ugap[k].abs().sum() > 0 else None
            rec["sit35"] = float(unit(mean35) @ sit[k]) if k in sit else None
            rec["sit22"] = float(unit(mean22) @ sit[k]) if k in sit else None
            late = torch.stack([mlp_out(mlps, b)[0, pos].float().mean(0) for b in range(24, 36)]).sum(0)
            rec["late_ugap"] = float(unit(late) @ ugap[k]) if ugap[k].abs().sum() > 0 else None
            rec["late_sit"] = float(unit(late) @ sit[k]) if k in sit else None
            rec["late_v35"] = float(unit(late) @ v35)
            mid = torch.stack([mlp_out(mlps, b)[0, pos].float().mean(0) for b in range(12, 22)]).sum(0)
            rec["mid_v22"] = float(unit(mid) @ v22)
            mass = {"sys": 0.0, "user": 0.0, "resp": 0.0}
            for b, hidx in CARRIERS:
                att = out.attentions[b][0, hidx, pos, :].float()
                mass["sys"] += float(att[:, :sys_len].sum(1).mean()) if sys_len else 0.0
                mass["user"] += float(att[:, sys_len : start + 1].sum(1).mean())
                mass["resp"] += float(att[:, start + 1 :].sum(1).mean())
            rec["carrier_mass"] = {a: round(b / len(CARRIERS), 4) for a, b in mass.items()}
            sink.write(json.dumps(rec) + "\n")
            if n % 200 == 0:
                sink.flush()
                log.info("%d/%d, %.1f min", n, len(rows), (time.monotonic() - started) / 60)
    log.info("done %.1f min", (time.monotonic() - started) / 60)


if __name__ == "__main__":
    main()
