from __future__ import annotations

import collections
import glob
import json
from pathlib import Path

import numpy as np

TRIGGER = Path("../exp_trigger_prompts")
OUT = Path("./work/tmp/reliab.json")
METRICS = ("concept", "instruct", "fluency")


def main() -> None:
    out = {}
    for tag, dirs in (("qwen", ("judge", "judge-1217")), ("llama", ("judge-llama",)), ("gemma", ("judge-gemma",))):
        sub = {m: collections.defaultdict(dict) for m in METRICS}
        for d in dirs:
            for m in METRICS:
                for f in glob.glob(str(TRIGGER / "work" / d / f"{m}-shard-*.jsonl")):
                    for line in open(f):
                        r = json.loads(line)
                        p, slot, lay, al = r["key"].split(":")
                        sub[m][(int(p), int(lay[1:]), float(al[1:]))][int(slot)] = float(r["score"])
        cells = sub["concept"]
        rows = sorted({k[1] for k in cells})
        alphas = sorted({abs(k[2]) for k in cells})
        res = {"rows": rows, "alphas": alphas, "n_cells": len(cells)}
        for a in alphas:
            for pole_sign, name in ((1, "concept_pole"), (-1, "antagonist_pole")):
                mean_row, sd_row, disagree_row, n_row = [], [], [], []
                for L in rows:
                    ms, sds, dis, n = [], [], [], 0
                    for (p, LL, al), slots in cells.items():
                        if LL != L or al != pole_sign * a or len(slots) < 4:
                            continue
                        v = np.array(list(slots.values()))
                        ms.append(v.mean())
                        sds.append(v.std())
                        dis.append(float((v >= 1).any() and (v == 0).any()))
                        n += 1
                    mean_row.append(float(np.mean(ms)) if ms else None)
                    sd_row.append(float(np.mean(sds)) if sds else None)
                    disagree_row.append(float(np.mean(dis)) if dis else None)
                    n_row.append(n)
                res[f"{name}_a{a}"] = {
                    "mean": mean_row,
                    "prompt_sd": sd_row,
                    "frac_cells_mixed": disagree_row,
                    "n": n_row,
                }
            flip_row = []
            for L in rows:
                gt = eq = tot = 0
                for (p, LL, al), slots in cells.items():
                    if LL != L or al != a:
                        continue
                    other = cells.get((p, L, -a), {})
                    for s, sc in slots.items():
                        if s in other:
                            tot += 1
                            gt += sc > other[s]
                            eq += sc == other[s]
                flip_row.append({"P_concept_push_scores_higher": (gt + 0.5 * eq) / tot if tot else None, "n": tot})
            res[f"paired_a{a}"] = flip_row
        out[tag] = res
    OUT.write_text(json.dumps(out))
    for tag, res in out.items():
        print(f"\n== {tag}: {res['n_cells']} cells, rows {res['rows']}")
        for a in res["alphas"]:
            c = res[f"concept_pole_a{a}"]
            an = res[f"antagonist_pole_a{a}"]
            print(
                f"alpha {a}: row | concept mean | prompt sd | mixed cells || antagonist mean | prompt sd | mixed cells"
            )
            for i, L in enumerate(res["rows"]):
                f = lambda x: "  nan" if x is None else f"{x:.3f}"
                print(
                    f"  {L:3d} | {f(c['mean'][i])} | {f(c['prompt_sd'][i])} | {f(c['frac_cells_mixed'][i])} || {f(an['mean'][i])} | {f(an['prompt_sd'][i])} | {f(an['frac_cells_mixed'][i])}"
                )


if __name__ == "__main__":
    main()
