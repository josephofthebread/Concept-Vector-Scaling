import argparse
import collections
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from exp_attn_interp.saelabel import encode, load_sae

p = argparse.ArgumentParser()
p.add_argument("--records", required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--labels", type=Path, default=Path("work/sae/label-instruct.json"))
p.add_argument("--rows", default="22,24,28,35")
p.add_argument("--limit", type=int, default=0)
p.add_argument("--model", default="Qwen/Qwen3-8B")
args = p.parse_args()
rows = [int(x) for x in args.rows.split(",")]
lab = json.load(args.labels.open())


feats = {}
scale = {}
for r in rows:
    R = lab["rows"][str(r)]
    scale[r] = float(np.median([c[0]["act"] for c in R["contexts"].values() if c]))
    for k, lst in R["pole_disc"].items():
        feats[(r, int(k))] = []
        for d in lst[:12]:
            cs = R["contexts"].get(str(d["f"]), [])[:3]
            toks = [re.sub(r"[^a-zA-Z']", "", c["ctx"].split("⟦")[-1].split("⟧")[0]).lower() for c in cs]
            cnt = collections.Counter(t for t in toks if t)
            feats[(r, int(k))].append(
                (d["f"], float(np.sign(d["diff"])), "tied" if cnt and cnt.most_common(1)[0][1] >= 2 else "context")
            )
print("rows", rows, "scale", {r: round(s, 1) for r, s in scale.items()}, "concepts", len({k for _, k in feats}))

recs = []
for path in [p for p in args.records.replace(";", ",").split(",") if p]:
    for line in open(path):
        rec = json.loads(line)
        if (rows[0], int(rec["pair_id"])) in feats:
            recs.append(rec)
if args.limit:
    recs = recs[: args.limit]
done = set()
args.out.parent.mkdir(parents=True, exist_ok=True)
if args.out.exists():
    done = {json.loads(l)["key"] for l in args.out.open()}
recs = [r for r in recs if r["key"] not in done]
print(f"{len(recs)} records to read ({len(done)} done)")
tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa", local_files_only=True
).eval()
dev = model.device
saes = {r: load_sae(r - 1, dev) for r in rows}
t0 = time.monotonic()
with torch.inference_mode(), args.out.open("a") as sink:
    for i, rec in enumerate(recs):
        msgs = ([{"role": "system", "content": rec["system"]}] if rec.get("system") else []) + [
            {"role": "user", "content": rec["prompt"]}
        ]
        prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        pids = tok(prefix, add_special_tokens=False)["input_ids"]
        rids = tok(rec["response"], add_special_tokens=False)["input_ids"]
        if not rids:
            continue
        ids = torch.tensor([pids + rids], device=dev)
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        res = {
            "key": rec["key"],
            "pair_id": int(rec["pair_id"]),
            "slot": rec["slot"],
            "layer": int(rec.get("layer", 22)),
            "alpha": float(rec["alpha"]),
            "arm": rec.get("arm", ""),
            "sub": rec.get("sub", ""),
            "context": rec.get("context", ""),
            "ablate": rec.get("ablate", ""),
            "neurons": rec.get("neurons", ""),
            "where": rec.get("where", "current"),
            "n_resp": len(rids),
            "rows": {},
        }
        for r in rows:
            h = out.hidden_states[r][0, len(pids) :].float()
            vals, idx = encode(h, saes[r])
            fl = feats[(r, int(rec["pair_id"]))]
            want = torch.tensor([f for f, _, _ in fl], device=dev)
            match = (idx[:, :, None] == want[None, None, :]).float()
            act = (vals[:, :, None] * match).sum(1).mean(0) / scale[r]
            act = act.tolist()
            W = saes[r]["W_dec"]
            D = (W[:, want] if W.shape[0] != saes[r]["W_enc"].shape[0] else W[want].T).T
            dense = ((h @ D.T) / (h.norm(dim=-1, keepdim=True) * D.norm(dim=-1)[None, :] + 1e-6)).mean(0).tolist()
            sums = {"tied": 0.0, "context": 0.0, "tied_n": 0, "context_n": 0, "tied_dense": 0.0, "context_dense": 0.0}
            per = []
            for (f, s, kd), a, dn in zip(fl, act, dense):
                sums[kd] += s * a
                sums[kd + "_n"] += 1
                sums[kd + "_dense"] += s * dn
                per.append([f, s, kd, round(a, 4), round(dn, 5)])
            sums["norm"] = float(h.norm(dim=-1).mean())
            sums["per"] = per
            res["rows"][str(r)] = sums
        sink.write(json.dumps(res) + "\n")
        if i % 100 == 0:
            sink.flush()
            print(f"{i}/{len(recs)} {(time.monotonic() - t0) / 60:.1f} min", flush=True)
print("done", len(recs), f"{(time.monotonic() - t0) / 60:.1f} min")
