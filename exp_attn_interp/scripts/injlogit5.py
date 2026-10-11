import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from exp_attn_interp.config import INSTRUCT_MODEL, vector_tag_for
from exp_attn_interp.vectors import load_diff_vectors

p = argparse.ArgumentParser()
p.add_argument("--records", default="work/posteer-foot/orig.jsonl")
p.add_argument("--markers", type=Path, default=Path("data/markers_presence.json"))
p.add_argument("--inj", default="22,24,28,32")
p.add_argument("--alpha", type=float, default=0.5)
p.add_argument("--lens-row", type=int, default=33)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--limit", type=int, default=0)
args = p.parse_args()
inj_rows = sorted({int(x) for x in args.inj.split(",")})
mk = json.load(args.markers.open())
allrecs = [json.loads(l) for l in open(args.records)]
has_mk = lambda k: len(mk["concept"].get(str(k), [])) >= 2 and len(mk["antagonist"].get(str(k), [])) >= 2
same = [x for x in allrecs if abs(float(x["alpha"])) < 1e-6 and has_mk(x["pair_id"])]
norm_by_row = {}
for x in allrecs:
    if abs(abs(float(x["alpha"])) - args.alpha) < 1e-9:
        norm_by_row.setdefault(int(x["layer"]), float(x["norm"]))
if args.limit:
    same = same[: args.limit]
rkey = lambda x, L, a: f"same:{x['key']}:{L}:{a:+.2f}"
done = set()
args.out.parent.mkdir(parents=True, exist_ok=True)
if args.out.exists():
    done = {json.loads(l)["key"] for l in args.out.open()}
units = [
    (x, [(L, s * args.alpha) for L in inj_rows for s in (1.0, -1.0) if rkey(x, L, s * args.alpha) not in done])
    for x in same
]
units = [u for u in units if u[1]]
print(
    f"{len(units)} texts, {sum(len(u[1]) for u in units)} injections ({len(done)} done); rows {inj_rows}; lens row {args.lens_row}"
)
tok = AutoTokenizer.from_pretrained(INSTRUCT_MODEL, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    INSTRUCT_MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa", local_files_only=True
).eval()
dev = model.device
V = load_diff_vectors(device="cpu", tag=vector_tag_for(INSTRUCT_MODEL))
WuF = model.lm_head.weight.float()
gn = model.model.norm.weight.float()
eps = float(model.model.norm.variance_epsilon)
Ug = WuF * gn[None, :]
Umean = Ug.mean(0)
NB = len(model.model.layers)
LATE = list(range(args.lens_row, NB))
cap = {}
for b in LATE:

    def attn_hook(_m, _i, out, b=b):
        cap[("attn", b)] = (out[0] if isinstance(out, tuple) else out).detach()

    def mlp_hook(_m, _i, out, b=b):
        cap[("mlp", b)] = out.detach()

    model.model.layers[b].self_attn.register_forward_hook(attn_hook)
    model.model.layers[b].mlp.register_forward_hook(mlp_hook)


def final_hook(_m, _i, out):
    cap["final"] = (out[0] if isinstance(out, tuple) else out).detach()


model.model.layers[NB - 1].register_forward_hook(final_hook)


class Inject:
    def __init__(self, L, delta, start):
        self.L, self.delta, self.start, self.h = L, delta, start, None

    def __enter__(self):
        def hook(_m, _i, out):
            res = out[0] if isinstance(out, tuple) else out
            res = res.clone()
            res[:, self.start :, :] += self.delta.to(res.dtype)
            return (res, *out[1:]) if isinstance(out, tuple) else res

        self.h = model.model.layers[self.L - 1].register_forward_hook(hook)
        return self

    def __exit__(self, *a):
        self.h.remove()


def lens(h):
    rms = torch.sqrt((h * h).mean(-1) + eps)
    return ((h / rms[:, None]) * gn) @ WuF.T, rms


def run(ids, P, N, L=None, delta=None):
    cap.clear()
    if delta is not None:
        with Inject(L, delta, P - 1):
            out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    else:
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    sl = slice(P, N - 1)
    h33 = out.hidden_states[args.lens_row][0, sl].float()
    l33, _ = lens(h33)
    hf = cap["final"][0, sl].float()
    lf, rms = lens(hf)
    return {
        "lead33": l33.argmax(-1),
        "top": lf.argmax(-1),
        "lp": torch.log_softmax(lf, -1),
        "rms": rms,
        "comps": {f"{kind}{b}": cap[(kind, b)][0, sl].float() for b in LATE for kind in ("attn", "mlp")},
        "margin33": (l33.max(-1).values - l33.topk(2, -1).values[:, 1]),
    }


t0 = time.monotonic()
n_written = 0
with torch.inference_mode(), args.out.open("a") as sink:
    for ui, (rec, la) in enumerate(units):
        k = int(rec["pair_id"])
        msgs = ([{"role": "system", "content": rec["system"]}] if rec.get("system") else []) + [
            {"role": "user", "content": rec["prompt"]}
        ]
        prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        pids = tok(prefix, add_special_tokens=False)["input_ids"]
        rids = tok(rec["response"], add_special_tokens=False)["input_ids"]
        if len(rids) < 2:
            continue
        ids = torch.tensor([pids + rids], device=dev)
        P = len(pids)
        N = ids.shape[1]
        T = N - 1 - P
        nxt = ids[0, P + 1 :]
        F = torch.tensor(sorted(set(mk["concept"][str(k)]) | set(mk["antagonist"][str(k)])), device=dev)
        off = run(ids, P, N)
        inc = off["top"]
        d_inc = Ug[inc] - Umean[None, :]
        for L, a in la:
            delta = ((a * norm_by_row[L]) * (V[L][k] / V[L][k].norm())).to(dev)
            on = run(ids, P, N, L, delta)
            lead = on["lead33"]
            d_lead = Ug[lead] - Umean[None, :]
            terms_inc = {c: ((on["comps"][c] - off["comps"][c]) * d_inc).sum(-1) / on["rms"] for c in on["comps"]}
            terms_lead = {c: ((on["comps"][c] - off["comps"][c]) * d_lead).sum(-1) / on["rms"] for c in on["comps"]}
            klp = (on["lp"].exp() * (on["lp"] - off["lp"])).sum(-1)
            res = {
                "key": rkey(rec, L, a),
                "pair_id": k,
                "slot": rec["slot"],
                "layer": L,
                "alpha": a,
                "n_pos": int(T),
                "delta_norm": float(delta.norm()),
                "pos": {
                    "inc_is_text": [int(v) for v in (inc == nxt)],
                    "lead33_off_is_inc": [int(v) for v in (off["lead33"] == inc)],
                    "disp33": [int(v) for v in (lead != inc)],
                    "disp_final": [int(v) for v in (on["top"] != inc)],
                    "lead_is_marker": [int(v) for v in torch.isin(lead, F)],
                    "lead_is_text": [int(v) for v in (lead == nxt)],
                    "margin33_on": [round(float(v), 3) for v in on["margin33"]],
                    "margin33_off": [round(float(v), 3) for v in off["margin33"]],
                    "kl": [round(float(v), 4) for v in klp],
                    "mlp_inc": [round(float(v), 4) for v in sum(terms_inc[f"mlp{b}"] for b in LATE)],
                    "attn_inc": [round(float(v), 4) for v in sum(terms_inc[f"attn{b}"] for b in LATE)],
                    "mlp_lead": [round(float(v), 4) for v in sum(terms_lead[f"mlp{b}"] for b in LATE)],
                    **{f"mlp{b}_inc": [round(float(v), 4) for v in terms_inc[f"mlp{b}"]] for b in LATE},
                },
            }
            sink.write(json.dumps(res) + "\n")
            n_written += 1
        if ui % 50 == 0:
            sink.flush()
            print(f"{ui}/{len(units)} texts, {n_written} records, {(time.monotonic() - t0) / 60:.1f} min", flush=True)
print("done", n_written, f"{(time.monotonic() - t0) / 60:.1f} min")
