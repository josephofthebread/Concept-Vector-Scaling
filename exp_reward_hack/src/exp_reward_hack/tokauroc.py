import glob
import json
import os
from collections import defaultdict

import numpy as np

D = "."
MODELS = {"qwen": 23, "llama": 15, "gemma": 24}
KS = [1, 3, 5, 7]
THR = 0.85
MING = 10

labels = {r["file"]: r for r in json.load(open(f"{D}/labels.json"))}
feats = json.load(open(f"{D}/../exp_lmsys_probe/data/features.json"))["features"]
feats.sort(key=lambda f: f["pair_id"])
names = [f"{f['concept']} // {f.get('antagonist', '')}" for f in feats]


out = {"thr": THR, "ks": KS, "models": {}}
for mtag in MODELS:
    bytask = defaultdict(list)
    for f in sorted(glob.glob(f"{D}/turnwork_tok/*-{mtag}-A-seed*.npz")):
        lab = labels.get(os.path.basename(f).replace(".npz", ".json"))
        if lab is None:
            continue
        z = np.load(f, allow_pickle=True)
        bytask[lab["workload"]].append((bool(lab["cheated"]), z["z"].astype(np.float32), z["turn"]))
    print(f"\n########## {mtag} (L{MODELS[mtag]})")
    model_rows = {}
    for k in KS:
        best = (0.0, "", "")
        hit_concept_tasks = defaultdict(set)
        per_task = {}
        for task, eps in bytask.items():
            ch = [(z, t) for c, z, t in eps if c]
            cl = [(z, t) for c, z, t in eps if not c]
            if len(ch) < MING or len(cl) < MING:
                continue
            pos = np.vstack([z[t < k] for z, t in ch if (t < k).any()])
            neg = np.vstack([z[t < k] for z, t in cl if (t < k).any()])
            if len(pos) < 20 or len(neg) < 20:
                continue
            na, nb = len(pos), len(neg)
            X = np.vstack([pos, neg])
            ranks = np.argsort(np.argsort(X, axis=0), axis=0).astype(np.float64) + 1
            rpos = ranks[:na].sum(0)
            a = (rpos - na * (na + 1) / 2.0) / (na * nb)
            a = np.maximum(a, 1 - a)
            nhit = int((a >= THR).sum())
            j = int(np.argmax(a))
            per_task[task] = {
                "n_ch_ep": len(ch),
                "n_cl_ep": len(cl),
                "n_pos_tok": len(pos),
                "n_neg_tok": len(neg),
                "n_hit": nhit,
                "top": [names[j], round(float(a[j]), 3)],
            }
            for p in np.where(a >= THR)[0]:
                hit_concept_tasks[int(p)].add(task)
            if a[j] > best[0]:
                best = (float(a[j]), names[j], task)
        n_any = len(hit_concept_tasks)
        model_rows[k] = {"tasks": per_task, "best": best, "n_concepts_hit_any_task": n_any}
        print(
            f"  K<={k}: usable tasks={len(per_task)} | concepts>={THR} in >=1 task: {n_any} | "
            f"best={best[0]:.3f} [{best[1][:34]} @ {best[2]}]"
        )
        for task, r in sorted(per_task.items(), key=lambda kv: -kv[1]["n_hit"]):
            flag = "  <<" if r["n_hit"] else ""
            print(
                f"      {task:20} ch_ep={r['n_ch_ep']:3} cl_ep={r['n_cl_ep']:3} "
                f"tok={r['n_pos_tok']:5}/{r['n_neg_tok']:5} hits={r['n_hit']:4} "
                f"top={r['top'][1]:.3f}{flag}"
            )
    out["models"][mtag] = model_rows

os.makedirs(f"{D}/xresults", exist_ok=True)
json.dump(out, open(f"{D}/xresults/tokauroc.json", "w"), indent=2)
print("\nwrote xresults/tokauroc.json")
