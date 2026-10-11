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
STEER = Path("work/steer")


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
    markers: dict = {}
    for k, pid in want.items():
        allw = set(dfp[k]) | set(dfa[k])
        markers[pid] = set(sorted(allw, key=lambda w: dfp[k][w] - dfa[k][w], reverse=True)[:KMARK])

    def rate(txt: str, mset: set) -> float:
        w = WORD.findall(txt.lower())
        return sum(x in mset for x in w) / max(len(w), 1)

    base: dict = collections.defaultdict(lambda: [0.0, 0])
    steer: dict = collections.defaultdict(lambda: [0.0, 0])
    for f in glob.glob(str(STEER / "sweep-Qwen_Qwen3-8B-shard-*.jsonl")):
        if "1217" in f:
            continue
        for line in open(f):
            r = json.loads(line)
            pid = r["pair_id"]
            if pid not in markers:
                continue
            m = markers[pid]
            if r["alpha"] == 0:
                base[pid][0] += rate(r["response"], m)
                base[pid][1] += 1
            elif r["alpha"] > 0:
                steer[pid][0] += rate(r["response"], m)
                steer[pid][1] += 1

    rows = []
    for c in conc:
        pid = c["pair_id"]
        b = base[pid][0] / base[pid][1] if base[pid][1] else float("nan")
        s = steer[pid][0] / steer[pid][1] if steer[pid][1] else float("nan")
        rows.append({**c, "base_rate": b, "steer_rate": s, "injection": s - b, "markers": sorted(markers[pid])[:10]})
    Path("work/markerinject.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    rows.sort(key=lambda r: r["steer"])
    print(f"{'steer':>5s} {'inject':>7s} {'base%':>6s} {'steer%':>6s}  concept")
    for r in rows:
        print(
            f"{r['steer']:5.2f} {100 * r['injection']:+7.2f} {100 * r['base_rate']:6.2f} "
            f"{100 * r['steer_rate']:6.2f}  {r['concept'][:36]}"
        )
    inj = np.array([r["injection"] for r in rows])
    st = np.array([r["steer"] for r in rows])
    ra, rb = inj.argsort().argsort() * 1.0, st.argsort().argsort() * 1.0
    print(f"\nSpearman(injection, steer score) over {len(rows)} concepts = {np.corrcoef(ra, rb)[0, 1]:+.3f}")


if __name__ == "__main__":
    main()
