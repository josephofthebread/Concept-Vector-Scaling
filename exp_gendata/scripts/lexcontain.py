from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from datasets import load_dataset
from lexbias import tfidf

TT = Path("../exp_trigger_prompts")
MID = {"Qwen/Qwen3-8B": 18, "google/gemma-2-9b-it": 21, "unsloth/Meta-Llama-3.1-8B-Instruct": 16}
STOP = set(
    (
        "a an the to of in on at by for with from as into about over under and or but nor so if then "
        "is am are was were be been being have has had do does did will would can could this that these "
        "those it its there here i you he she they we me him her them us my your his their our not no s t "
        "re ve ll d m one all any each some such more most many much very just also than which who what "
        "when where why how out up down off only own same too whether you'll i'm isn i'll have".split()
    )
)


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

    rows = []
    all_pct = []
    for c in conc:
        k = (c["concept"], c["antagonist"])
        pid = c["pair_id"]
        docs = pole[k] + antg[k]
        y = np.array([1] * len(pole[k]) + [0] * len(antg[k]))
        x, vocab = tfidf(docs)
        wfull = x[y == 1].mean(0) - x[y == 0].mean(0)
        vindex = {w: j for j, w in enumerate(vocab)}
        order = np.argsort(-wfull)
        rank_of = np.empty(len(vocab), int)
        rank_of[order] = np.arange(len(vocab))
        nvoc = len(vocab)

        promoted = prom_by_pid[pid]["layers"][mid]["latin"]
        content = [w for w in promoted if w not in STOP and len(w) > 2]
        hits = []
        for w in content:
            j = vindex.get(w)
            if j is not None:
                pct = rank_of[j] / max(nvoc - 1, 1)
                hits.append((w, round(float(pct), 3)))
                all_pct.append(pct)
        pcts = [p for _, p in hits]
        rows.append(
            {
                "concept": c["concept"],
                "steer": c["steer"],
                "n_content_promoted": len(content),
                "n_in_vocab": len(hits),
                "median_pct": round(float(np.median(pcts)), 3) if pcts else None,
                "frac_top5": round(np.mean([p < 0.05 for p in pcts]), 2) if pcts else None,
                "frac_top10": round(np.mean([p < 0.10 for p in pcts]), 2) if pcts else None,
                "words": sorted(hits, key=lambda t: t[1]),
            }
        )
        top = " ".join(f"{w}:{p:.2f}" for w, p in sorted(hits, key=lambda t: t[1])[:8])
        print(
            f"steer={c['steer']:.2f} inVocab={len(hits)}/{len(content)} medPct={rows[-1]['median_pct']} "
            f"top5={rows[-1]['frac_top5']}  {c['concept'][:24]:24s} | {top}",
            flush=True,
        )

    out = Path("work/lexcontain")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"lexcontain-{tag}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    ap = np.array(all_pct)
    print(f"\nALL promoted content words in-vocab: n={len(ap)}")
    print(f"  median discriminative percentile: {np.median(ap):.3f}  (0=most pole-discriminative, 0.5=neutral)")
    print(
        f"  fraction in top-5% discriminative: {np.mean(ap < 0.05):.2f}; top-10%: {np.mean(ap < 0.10):.2f}; "
        f"pole-half (<0.5): {np.mean(ap < 0.5):.2f}"
    )


if __name__ == "__main__":
    main()
