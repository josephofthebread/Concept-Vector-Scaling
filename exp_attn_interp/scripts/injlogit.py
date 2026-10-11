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
args = p.parse_args()
inj_rows = {int(x) for x in args.inj.split(",")}
mk = json.load(args.markers.open())
recs = [json.loads(l) for l in open(args.records)]
recs = [x for x in recs if int(x["layer"]) in inj_rows and abs(abs(float(x["alpha"])) - args.alpha) < 1e-9]
recs = [
    x
    for x in recs
    if len(mk["concept"].get(str(x["pair_id"]), [])) >= 2 and len(mk["antagonist"].get(str(x["pair_id"]), [])) >= 2
]
if args.limit:
    recs = recs[: args.limit]
done = set()
args.out.parent.mkdir(parents=True, exist_ok=True)
if args.out.exists():
    done = {json.loads(l)["key"] for l in args.out.open()}
recs = [x for x in recs if x["key"] not in done]
print(f"{len(recs)} texts to read ({len(done)} done); injection rows {sorted(inj_rows)}, |alpha| {args.alpha}")
tok = AutoTokenizer.from_pretrained(INSTRUCT_MODEL, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    INSTRUCT_MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa", local_files_only=True
).eval()
dev = model.device
V = load_diff_vectors(device="cpu", tag=vector_tag_for(INSTRUCT_MODEL))
Wu = model.lm_head.weight
WuF = Wu.float()
gn = model.model.norm.weight.float()
eps = float(model.model.norm.variance_epsilon)
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


gdir = {}
for k in {int(x["pair_id"]) for x in recs}:
    C = torch.tensor(mk["concept"][str(k)], device=dev)
    A = torch.tensor(mk["antagonist"][str(k)], device=dev)
    gdir[k] = (Wu[C].float().mean(0) - Wu[A].float().mean(0)) * gn
t0 = time.monotonic()
checked = False
with torch.inference_mode(), args.out.open("a") as sink:
    for i, rec in enumerate(recs):
        k, L, a = int(rec["pair_id"]), int(rec["layer"]), float(rec["alpha"])
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
        nxt = ids[0, P + 1 :]
        delta = ((a * float(rec["norm"])) * (V[L][k] / V[L][k].norm())).to(dev)
        g = gdir[k] * (1.0 if a > 0 else -1.0)
        C = torch.tensor(mk["concept" if a > 0 else "antagonist"][str(k)], device=dev)
        A = torch.tensor(mk["antagonist" if a > 0 else "concept"][str(k)], device=dev)
        passes = {}
        for on in (True, False):
            cap.clear()
            if on:
                with Inject(L, delta, P - 1):
                    out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
            else:
                out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
            sl = slice(P, N - 1)
            h = cap["final"][0, sl].float()
            rms = torch.sqrt((h * h).mean(-1) + eps)
            logits = ((h / rms[:, None]) * gn) @ WuF.T
            lp = torch.log_softmax(logits, -1)
            proj = {"E": out.hidden_states[0][0, sl].float() @ g}
            for b in range(NB):
                proj[("attn", b)] = cap[("attn", b)][0, sl].float() @ g
                proj[("mlp", b)] = cap[("mlp", b)][0, sl].float() @ g
            if not checked:
                ref = out.logits[0, sl].float()
                print(
                    f"  logits check (hook {'on' if on else 'off'}): max |mine − model| / max |model| = {float((logits - ref).abs().max() / ref.abs().max()):.2e}"
                )
                tot = sum(proj.values()) + (delta.float() @ g if on else 0.0)
                print(
                    f"  residual check (hook {'on' if on else 'off'}): |h·g − Σ| / |h·g| = {float((h @ g - tot).norm() / (h @ g).norm()):.2e}"
                )
            passes[on] = {
                "h": h,
                "rms": rms,
                "lp": lp,
                "proj": proj,
                "massP": torch.logsumexp(lp[:, C], -1),
                "massO": torch.logsumexp(lp[:, A], -1),
                "lpn": lp[torch.arange(N - 1 - P, device=dev), nxt],
            }
        checked = True
        on, off = passes[True], passes[False]
        Gon = (on["h"] @ g) / on["rms"]
        Goff = (off["h"] @ g) / off["rms"]
        terms = {}
        for kk in on["proj"]:
            name = kk if isinstance(kk, str) else f"{kk[0]}{kk[1]}"
            terms[name] = float(((on["proj"][kk] - off["proj"][kk]) / on["rms"]).mean())
        terms["delta"] = float(((delta.float() @ g) / on["rms"]).mean())
        terms["dilution"] = float(((off["h"] @ g) * (1.0 / on["rms"] - 1.0 / off["rms"])).mean())
        dG = float((Gon - Goff).mean())
        res = {
            "key": rec["key"],
            "pair_id": k,
            "slot": rec["slot"],
            "layer": L,
            "alpha": a,
            "n_pos": int(N - 1 - P),
            "dG": dG,
            "sum_terms": float(sum(terms.values())),
            "G_off": float(Goff.mean()),
            "G_on": float(Gon.mean()),
            "terms": terms,
            "soft": {
                "d_mass_pushed": float((on["massP"] - off["massP"]).mean()),
                "d_mass_other": float((on["massO"] - off["massO"]).mean()),
                "mass_pushed_off": float(off["massP"].mean()),
                "mass_other_off": float(off["massO"].mean()),
                "kl": float((on["lp"].exp() * (on["lp"] - off["lp"])).sum(-1).mean()),
                "d_lp_next": float((on["lpn"] - off["lpn"]).mean()),
            },
            "rms": {
                "on": float(on["rms"].mean()),
                "off": float(off["rms"].mean()),
                "ratio": float((on["rms"] / off["rms"]).mean()),
            },
            "dh_norm": float((on["h"] - off["h"]).norm(dim=-1).mean()),
            "delta_norm": float(delta.norm()),
            "g_norm": float(g.norm()),
        }
        sink.write(json.dumps(res) + "\n")
        if i % 50 == 0:
            sink.flush()
            print(
                f"{i}/{len(recs)} {(time.monotonic() - t0) / 60:.1f} min  dG {dG:+.3f} (sum {res['sum_terms']:+.3f})",
                flush=True,
            )
print("done", len(recs), f"{(time.monotonic() - t0) / 60:.1f} min")
