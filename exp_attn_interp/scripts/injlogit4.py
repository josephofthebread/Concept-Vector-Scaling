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
p.add_argument("--inj", default="22,28")
p.add_argument("--alpha", type=float, default=0.5)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--limit", type=int, default=0)
p.add_argument("--designs", default="own,same")
args = p.parse_args()
inj_rows = sorted({int(x) for x in args.inj.split(",")})
mk = json.load(args.markers.open())
allrecs = [json.loads(l) for l in open(args.records)]
has_mk = lambda k: len(mk["concept"].get(str(k), [])) >= 2 and len(mk["antagonist"].get(str(k), [])) >= 2
own = [
    x
    for x in allrecs
    if int(x["layer"]) in inj_rows and abs(abs(float(x["alpha"])) - args.alpha) < 1e-9 and has_mk(x["pair_id"])
]
same = [x for x in allrecs if abs(float(x["alpha"])) < 1e-6 and has_mk(x["pair_id"])]
norm_by_row = {}
for x in allrecs:
    if abs(abs(float(x["alpha"])) - args.alpha) < 1e-9:
        norm_by_row.setdefault(int(x["layer"]), float(x["norm"]))
if args.limit:
    own = own[: args.limit]
    same = same[: max(1, args.limit // 4)]
designs = set(args.designs.split(","))
units = [("own", x, [(int(x["layer"]), float(x["alpha"]))]) for x in own] if "own" in designs else []
units += (
    [("same", x, [(L, s * args.alpha) for L in inj_rows for s in (1.0, -1.0)]) for x in same]
    if "same" in designs
    else []
)
rkey = lambda d, x, L, a: f"{d}:{x['key']}:{L}:{a:+.2f}"
done = set()
args.out.parent.mkdir(parents=True, exist_ok=True)
if args.out.exists():
    done = {json.loads(l)["key"] for l in args.out.open()}
units = [(d, x, [(L, a) for L, a in la if rkey(d, x, L, a) not in done]) for d, x, la in units]
units = [u for u in units if u[2]]
print(
    f"{len(units)} texts, {sum(len(u[2]) for u in units)} injections to read ({len(done)} done); rows {inj_rows}, norms {norm_by_row}"
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
cap = {}
for b in range(NB):

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


def lp_of(h):
    rms = torch.sqrt((h * h).mean(-1) + eps)
    return torch.log_softmax(((h / rms[:, None]) * gn) @ WuF.T, -1), rms


def run(ids, P, N, L=None, delta=None):
    cap.clear()
    if delta is not None:
        with Inject(L, delta, P - 1):
            out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    else:
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    sl = slice(P, N - 1)
    comps = {"E": out.hidden_states[0][0, sl].float()}
    for b in range(NB):
        comps[f"attn{b}"] = cap[("attn", b)][0, sl].float()
        comps[f"mlp{b}"] = cap[("mlp", b)][0, sl].float()
    return {"h": cap["final"][0, sl].float(), "comps": comps, "ref": out.logits[0, sl].float()}


t0 = time.monotonic()
checked = False
n_written = 0
with torch.inference_mode(), args.out.open("a") as sink:
    for ui, (design, rec, la) in enumerate(units):
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
        is_marker = torch.isin(nxt, F)
        dirs = Ug[nxt] - Umean[None, :]
        off = run(ids, P, N)
        lp_off, rms_off = lp_of(off["h"])
        if not checked:
            ref_lp = torch.log_softmax(off["ref"], -1)
            print(
                f"  logits check (off): max |mine − model| / max |model| = {float((lp_off - ref_lp).abs().max() / ref_lp.abs().max()):.2e}"
            )
        own_off = ((off["h"] * dirs).sum(-1)) / rms_off
        for L, a in la:
            norm = float(rec["norm"]) if design == "own" else norm_by_row[L]
            delta = ((a * norm) * (V[L][k] / V[L][k].norm())).to(dev)
            on = run(ids, P, N, L, delta)
            lp_on, rms_on = lp_of(on["h"])
            lp_dir, _ = lp_of(off["h"] + delta[None, :])
            lp_prc, _ = lp_of(on["h"] - delta[None, :])
            kl_full, kl_dir, kl_prc = (
                (lp_on.exp() * (lp_on - lp_off)).sum(-1),
                (lp_dir.exp() * (lp_dir - lp_off)).sum(-1),
                (lp_prc.exp() * (lp_prc - lp_off)).sum(-1),
            )
            if not checked:
                tot = sum(on["comps"][c] - off["comps"][c] for c in on["comps"]) + delta[None, :]
                print(
                    f"  residual check: |Δh − Σ| / |Δh| = {float((on['h'] - off['h'] - tot).norm() / (on['h'] - off['h']).norm()):.2e}"
                )
                checked = True
            own_on = ((on["h"] * dirs).sum(-1)) / rms_on
            d_own = own_on - own_off
            terms = {c: ((on["comps"][c] - off["comps"][c]) * dirs).sum(-1) / rms_on for c in on["comps"]}
            terms["delta"] = (delta[None, :] * dirs).sum(-1) / rms_on
            terms["dilution"] = ((off["h"] * dirs).sum(-1)) * (1.0 / rms_on - 1.0 / rms_off)
            res = {
                "key": rkey(design, rec, L, a),
                "design": design,
                "pair_id": k,
                "slot": rec["slot"],
                "layer": L,
                "alpha": a,
                "n_pos": int(T),
                "n_marker": int(is_marker.sum()),
                "delta_norm": float(delta.norm()),
                "kl": {
                    "full": float(kl_full.mean()),
                    "direct": float(kl_dir.mean()),
                    "processed": float(kl_prc.mean()),
                    "full_median": float(kl_full.median()),
                    "direct_median": float(kl_dir.median()),
                    "processed_median": float(kl_prc.median()),
                    "full_first": float(kl_full[0]),
                    "full_rest": float(kl_full[1:].mean()) if T > 1 else float("nan"),
                    "full_pos": [round(float(v), 4) for v in kl_full],
                },
                "own": {"all": {"d": float(d_own.mean()), "sum": float(sum(v.mean() for v in terms.values()))}},
            }
            for cn, m in (("marker", is_marker), ("other", ~is_marker)):
                if int(m.sum()) == 0:
                    continue
                res["own"][cn] = {
                    "n": int(m.sum()),
                    "d": float(d_own[m].mean()),
                    "terms": {c: float(v[m].mean()) for c, v in terms.items()},
                }
            res["own"]["all"]["terms"] = {c: float(v.mean()) for c, v in terms.items()}
            sink.write(json.dumps(res) + "\n")
            n_written += 1
        if ui % 50 == 0:
            sink.flush()
            print(f"{ui}/{len(units)} texts, {n_written} records, {(time.monotonic() - t0) / 60:.1f} min", flush=True)
print("done", n_written, f"{(time.monotonic() - t0) / 60:.1f} min")
