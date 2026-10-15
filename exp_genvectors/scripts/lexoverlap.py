from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from datasets import load_dataset
from lexbias import auroc, tfidf

TT = Path("../exp_trigger_prompts")
MID = {"Qwen/Qwen3-8B": 18, "google/gemma-2-9b-it": 21, "unsloth/Meta-Llama-3.1-8B-Instruct": 16}
TOPK = 30
FOLDS = 5
STOP = set(
    (
        "a an the to of in on at by for with from as into about over under and or but nor so if then "
        "is am are was were be been being have has had do does did will would can could this that these "
        "those it its there here i you he she they we me him her them us my your his their our not no "
        "s t re ve ll d m one all any each some such more most many much very will just also then than "
        "which who whom whose what when where why how out up down off only own same too own".split()
    )
)


def lenient_hit(word: str, promoted: set[str]) -> bool:
    """word (full) matches a promoted BPE piece if either is a >=4-char prefix of the other."""
    if word in promoted:
        return True
    for p in promoted:
        if len(p) >= 4 and (word.startswith(p) or p.startswith(word)):
            return True
    return False


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B", choices=sorted(MID))
    model = parser.parse_args().model
    tag = model.split("/")[-1]
    mid = str(MID[model])
    conc = json.loads((TT / "data" / "inject_concepts.json").read_text())
    ll = json.loads((TT / f"work/logitlens/logitlens-{tag}.json").read_text())
    prom_by_pid = {r["pair_id"]: r for r in ll["results"]}
    want = {(c["concept"], c["antagonist"]): c["pair_id"] for c in conc}

    pole: dict = {k: [] for k in want}
    antg: dict = {k: [] for k in want}
    ds = load_dataset("AntonKorznikov/feature_stories", split="train")
    for r in ds:
        if r["language"] != "English":
            continue
        k = (r["concept"], r["antagonist"])
        if k in want:
            pole[k].append(r["concept_text"])
            antg[k].append(r["antagonist_text"])

    rng = np.random.default_rng(0)
    rows = []
    for c in conc:
        k = (c["concept"], c["antagonist"])
        pid = c["pair_id"]
        docs = pole[k] + antg[k]
        y = np.array([1] * len(pole[k]) + [0] * len(antg[k]))
        x, vocab = tfidf(docs)
        fold = rng.integers(0, FOLDS, len(docs))
        score = np.zeros(len(docs), np.float32)
        for f in range(FOLDS):
            tr, te = fold != f, fold == f
            if te.any() and y[tr].sum() and (y[tr] == 0).sum():
                w = x[tr][y[tr] == 1].mean(0) - x[tr][y[tr] == 0].mean(0)
                score[te] = x[te] @ w
        en_auroc = auroc(y, score)
        wfull = x[y == 1].mean(0) - x[y == 0].mean(0)
        aurwords = [vocab[j] for j in np.argsort(-wfull) if "_" not in vocab[j]][:TOPK]

        pr = prom_by_pid[pid]["layers"][mid]
        promoted = set(pr["latin"])
        promoted_all = set()
        for L in prom_by_pid[pid]["layers"]:
            promoted_all |= set(prom_by_pid[pid]["layers"][L]["latin"])

        exact = [w for w in aurwords if w in promoted]
        exact_all = [w for w in aurwords if w in promoted_all]
        lenient = [w for w in aurwords if lenient_hit(w, promoted_all)]
        rows.append(
            {
                "concept": c["concept"],
                "steer": c["steer"],
                "en_auroc": round(float(en_auroc), 3),
                "aurwords": aurwords,
                "aur_func_frac": round(sum(w in STOP for w in aurwords) / len(aurwords), 2),
                "prom_func_frac": round(sum(w in STOP for w in pr["latin"]) / max(len(pr["latin"]), 1), 2),
                "exact_L18": exact,
                "exact_anylayer": exact_all,
                "lenient_anylayer": lenient,
            }
        )
        print(
            f"AUROC={en_auroc:.3f} steer={c['steer']:.2f} funcAUR={rows[-1]['aur_func_frac']:.2f} "
            f"funcPROM={rows[-1]['prom_func_frac']:.2f} ex={len(exact)} exAll={len(exact_all)} "
            f"len={len(lenient)}  {c['concept'][:26]:26s} | {' '.join(aurwords[:10])}",
            flush=True,
        )

    out = Path("work/lexoverlap")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"lexoverlap-{tag}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    ex = [len(r["exact_anylayer"]) for r in rows]
    le = [len(r["lenient_anylayer"]) for r in rows]
    print(
        f"\nmedian exact overlap (any layer) {sorted(ex)[len(ex) // 2]}/{TOPK}; "
        f"median lenient {sorted(le)[len(le) // 2]}/{TOPK}; "
        f"concepts w/ >=1 exact {sum(e > 0 for e in ex)}/{len(rows)}"
    )
    print(
        f"median AUROC-word func-frac {np.median([r['aur_func_frac'] for r in rows]):.2f}; "
        f"median promoted func-frac {np.median([r['prom_func_frac'] for r in rows]):.2f}"
    )


if __name__ == "__main__":
    main()
