import argparse
import collections
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from exp_attn_interp.config import INSTRUCT_MODEL, vector_tag_for
from exp_attn_interp.saelabel import load_sae
from exp_attn_interp.vectors import load_diff_vectors

p = argparse.ArgumentParser()
p.add_argument("--records", default="work/posteer-foot/orig.jsonl")
p.add_argument("--labels", type=Path, default=Path("work/sae/label-instruct.json"))
p.add_argument("--rows", default="28,35")
p.add_argument("--inj", default="22,28")
p.add_argument("--out", type=Path, required=True)
p.add_argument("--limit", type=int, default=0)
args = p.parse_args()
readouts = [int(x) for x in args.rows.split(",")]
inj_rows = {int(x) for x in args.inj.split(",")}
lab = json.load(args.labels.open())


feats = {}
for r in readouts:
    R = lab["rows"][str(r)]
    for k, lst in R["pole_disc"].items():
        feats[(r, int(k))] = []
        for d in lst[:12]:
            cs = R["contexts"].get(str(d["f"]), [])[:3]
            toks = [re.sub(r"[^a-zA-Z']", "", c["ctx"].split("⟦")[-1].split("⟧")[0]).lower() for c in cs]
            cnt = collections.Counter(t for t in toks if t)
            feats[(r, int(k))].append(
                (d["f"], float(np.sign(d["diff"])), "tied" if cnt and cnt.most_common(1)[0][1] >= 2 else "context")
            )
recs = [json.loads(l) for l in open(args.records)]
recs = [
    x
    for x in recs
    if int(x["layer"]) in inj_rows
    and abs(abs(float(x["alpha"])) - 0.5) < 1e-9
    and (readouts[0], int(x["pair_id"])) in feats
]
if args.limit:
    recs = recs[: args.limit]
done = set()
args.out.parent.mkdir(parents=True, exist_ok=True)
if args.out.exists():
    done = {json.loads(l)["key"] for l in args.out.open()}
recs = [x for x in recs if x["key"] not in done]
print(f"{len(recs)} texts to read ({len(done)} done); readouts {readouts}, injection rows {sorted(inj_rows)}")
tok = AutoTokenizer.from_pretrained(INSTRUCT_MODEL, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    INSTRUCT_MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa", local_files_only=True
).eval()
dev = model.device
V = load_diff_vectors(device="cpu", tag=vector_tag_for(INSTRUCT_MODEL))
saes = {r: load_sae(r - 1, dev) for r in readouts}
U = {}
for (r, k), fl in feats.items():
    W = saes[r]["W_dec"]
    D = W[:, [f for f, _, _ in fl]].T.float()
    U[(r, k)] = D / D.norm(dim=-1, keepdim=True)
NB = len(model.model.layers)
cap = {}
handles = []
for b in range(NB):

    def attn_hook(_m, _i, out, b=b):
        cap[("attn", b)] = (out[0] if isinstance(out, tuple) else out).detach()

    def mlp_hook(_m, _i, out, b=b):
        cap[("mlp", b)] = out.detach()

    handles.append(model.model.layers[b].self_attn.register_forward_hook(attn_hook))
    handles.append(model.model.layers[b].mlp.register_forward_hook(mlp_hook))


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
        if not rids:
            continue
        ids = torch.tensor([pids + rids], device=dev)
        P = len(pids)
        delta = (a * float(rec["norm"])) * (V[L][k] / V[L][k].norm())
        delta = delta.to(dev)
        passes = {}
        for on in (True, False):
            cap.clear()
            if on:
                with Inject(L, delta, P - 1):
                    out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
            else:
                out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
            E = out.hidden_states[0][0, P:].float()
            per = {}
            for r in readouts:
                Ur = U[(r, k)]
                proj = {"E": (E @ Ur.T).mean(0)}
                for b in range(r):
                    proj[("attn", b)] = (cap[("attn", b)][0, P:].float() @ Ur.T).mean(0)
                    proj[("mlp", b)] = (cap[("mlp", b)][0, P:].float() @ Ur.T).mean(0)
                if on and L <= r:
                    proj["delta"] = delta.float() @ Ur.T
                if not checked:
                    h = out.hidden_states[r][0, P:].float() @ Ur.T
                    tot = (
                        proj["E"] + sum(v for kk, v in proj.items() if kk not in ("E", "delta")) + proj.get("delta", 0)
                    )
                    print(
                        f"  decomposition check, readout {r}, hook {'on' if on else 'off'}: |resid − sum| / |resid| = {float((h.mean(0) - tot).norm() / h.mean(0).norm()):.4f}"
                    )
                per[r] = proj
            passes[on] = per
        checked = True
        res = {
            "key": rec["key"],
            "pair_id": k,
            "slot": rec["slot"],
            "layer": L,
            "alpha": a,
            "n_resp": len(rids),
            "rows": {},
        }
        for r in readouts:
            fl = feats[(r, k)]
            sgn = torch.tensor([s for _, s, _ in fl], device=dev)
            is_sit = torch.tensor([kd == "context" for _, _, kd in fl], device=dev)

            def split(v):
                v = v * sgn
                return float(v[is_sit].sum()), float(v[~is_sit].sum())

            on, off = passes[True][r], passes[False][r]
            rows = {"E": split(on["E"] - off["E"])}
            for b in range(r):
                rows[f"attn{b}"] = split(on[("attn", b)] - off[("attn", b)])
                rows[f"mlp{b}"] = split(on[("mlp", b)] - off[("mlp", b)])
            rows["delta"] = split(on["delta"]) if "delta" in on else (0.0, 0.0)
            rows["off_total"] = split(off["E"] + sum(off[kk] for kk in off if kk != "E"))
            res["rows"][str(r)] = rows
        sink.write(json.dumps(res) + "\n")
        if i % 50 == 0:
            sink.flush()
            print(f"{i}/{len(recs)} {(time.monotonic() - t0) / 60:.1f} min", flush=True)
print("done", len(recs), f"{(time.monotonic() - t0) / 60:.1f} min")
