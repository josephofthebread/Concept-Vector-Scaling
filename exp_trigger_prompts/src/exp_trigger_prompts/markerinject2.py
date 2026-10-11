from __future__ import annotations

import collections
import glob
import json
import re
from pathlib import Path

import numpy as np
from datasets import load_dataset

STOP = set(
    (
        "a an the to of in on at by for with from as into about over under and or but nor so if then "
        "is am are was were be been being have has had do does did will would can could this that these "
        "those it its there here i you he she they we me him her them us my your his their our not no".split()
    )
)
WORD = re.compile(r"[a-z']+")
KMARK = 20
JUDGE = Path("work/judge/steerscore-cells.jsonl")


def spear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    return float(np.corrcoef(a.argsort().argsort() * 1.0, b.argsort().argsort() * 1.0)[0, 1])


def main() -> None:
    conc = json.loads(Path("data/inject_concepts.json").read_text())
    want = {(c["concept"], c["antagonist"]): c["pair_id"] for c in conc}
    dfp: dict = collections.defaultdict(collections.Counter)
    dfa: dict = collections.defaultdict(collections.Counter)
    ds = load_dataset("AntonKorznikov/feature_stories", split="train")
    for r in ds:
        if r["language"] != "English":
            continue
        k = (r["concept"], r["antagonist"])
        if k in want:
            dfp[k].update(w for w in set(WORD.findall(r["concept_text"].lower())) if w not in STOP)
            dfa[k].update(w for w in set(WORD.findall(r["antagonist_text"].lower())) if w not in STOP)
    markers = {
        want[k]: set(sorted(set(dfp[k]) | set(dfa[k]), key=lambda w: dfp[k][w] - dfa[k][w], reverse=True)[:KMARK])
        for k in want
    }

    cell: dict = collections.defaultdict(lambda: [0.0, 0.0, 0])
    base: dict = collections.defaultdict(lambda: [0.0, 0])
    for f in glob.glob("work/steer/sweep-Qwen_Qwen3-8B-shard-*.jsonl"):
        if "1217" in f:
            continue
        for line in open(f):
            r = json.loads(line)
            pid = r["pair_id"]
            if pid not in markers:
                continue
            w = WORD.findall(r["response"].lower())
            mr = sum(x in markers[pid] for x in w) / max(len(w), 1)
            if r["alpha"] == 0:
                base[pid][0] += mr
                base[pid][1] += 1
            elif r["alpha"] > 0:
                c = cell[(pid, r["layer"], r["alpha"])]
                c[0] += mr
                w = WORD.findall(r["response"].lower())
                tri = [tuple(w[i : i + 3]) for i in range(len(w) - 2)]
                c[1] += 1 - len(set(tri)) / max(len(tri), 1) if tri else 0.0
                c[2] += 1
    baserate = {p: base[p][0] / base[p][1] for p in base if base[p][1]}
    judge = {}
    for line in open(JUDGE):
        r = json.loads(line)
        if r["alpha"] > 0:
            judge[(r["pair_id"], r["layer"], r["alpha"])] = r["score"]

    rows = []
    for key, (msum, rsum, n) in cell.items():
        pid, L, a = key
        if key in judge and pid in baserate:
            rows.append(
                {"pid": pid, "alpha": a, "inj": msum / n - baserate[pid], "nonsense": rsum / n, "judge": judge[key]}
            )
    print(f"cells: {len(rows)}")
    print("alpha |  mean_inj  mean_judge  mean_nonsense")
    for a in sorted({r["alpha"] for r in rows}):
        g = [r for r in rows if r["alpha"] == a]
        print(
            f"{a:+.2f} |  {100 * np.mean([r['inj'] for r in g]):+7.2f}   {np.mean([r['judge'] for r in g]):6.2f}"
            f"    {np.mean([r['nonsense'] for r in g]):6.3f}"
        )
    inj = [r["inj"] for r in rows]
    jd = [r["judge"] for r in rows]
    ns = [r["nonsense"] for r in rows]
    thr = float(np.median(ns))
    coh = [r for r in rows if r["nonsense"] < thr]
    print(f"\nSpearman(injection, judge): ALL cells = {spear(inj, jd):+.3f}")
    print(
        f"Spearman(injection, judge): COHERENT cells (nonsense<{thr:.2f}, n={len(coh)}) = "
        f"{spear([r['inj'] for r in coh], [r['judge'] for r in coh]):+.3f}"
    )
    Path("work/markerinject2.json").write_text(json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
