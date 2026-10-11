from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from datasets import load_dataset
from lexbias import tfidf

TT = Path("../exp_trigger_prompts")
MID = {"Qwen/Qwen3-8B": 18, "google/gemma-2-9b-it": 21, "unsloth/Meta-Llama-3.1-8B-Instruct": 16}
SEP_THRESH = 0.98
PROMO_THRESH = 0.90


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B", choices=sorted(MID))
    tag = parser.parse_args().model.split("/")[-1]
    conc = json.loads((TT / "data" / "inject_concepts.json").read_text())
    probes = json.loads((TT / "data" / "probe_words.json").read_text())
    lp = json.loads((TT / f"work/logitprobe/logitprobe-{tag}.json").read_text())
    promo = {(r["concept"], e["word"]): e["pct"] for r in lp for cat in r["words"] for e in r["words"][cat]}
    ant_of = {c["concept"]: c["antagonist"] for c in conc}

    want = {(concept, ant_of[concept]) for concept in probes}
    ds = load_dataset("AntonKorznikov/feature_stories", split="train")
    df = ds.to_pandas()
    print(f"loaded {len(df)} rows; filtering English", flush=True)
    en = df[df["language"] == "English"]
    pole: dict = {}
    antg: dict = {}
    for k in want:
        sub = en[(en["concept"] == k[0]) & (en["antagonist"] == k[1])]
        pole[k] = sub["concept_text"].tolist()
        antg[k] = sub["antagonist_text"].tolist()
    print(f"collected docs for {len(want)} concepts", flush=True)

    rows = []
    for concept, cats in probes.items():
        k = (concept, ant_of[concept])
        docs = pole[k] + antg[k]
        y = np.array([1] * len(pole[k]) + [0] * len(antg[k]))
        x, vocab = tfidf(docs)
        wfull = x[y == 1].mean(0) - x[y == 0].mean(0)
        vindex = {w: j for j, w in enumerate(vocab)}
        order = np.argsort(-wfull)
        rank_of = np.empty(len(vocab), int)
        rank_of[order] = np.arange(len(vocab))
        nvoc = len(vocab)
        for cat, words in cats.items():
            for w in words:
                j = vindex.get(w)
                disc = None if j is None else round(float(1 - rank_of[j] / max(nvoc - 1, 1)), 3)
                pp = promo.get((concept, w))
                in_sep = bool(disc is not None and disc >= SEP_THRESH)
                ood = bool((pp is not None and pp >= PROMO_THRESH) and not in_sep)
                rows.append(
                    {
                        "concept": concept,
                        "cat": cat,
                        "word": w,
                        "promo_pct": pp,
                        "disc_pct": disc,
                        "in_vocab": j is not None,
                        "in_separation_list": in_sep,
                        "ood_promoted": ood,
                    }
                )

    (Path("work/lexprobe_disc")).mkdir(parents=True, exist_ok=True)
    Path(f"work/lexprobe_disc/lexprobe_disc-{tag}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))

    print(f"{'cat':8s} {'word':14s} {'promo':>6s} {'disc':>6s} inSep  OOD   concept")
    for r in rows:
        d = f"{r['disc_pct']:.3f}" if r["disc_pct"] is not None else "  --"
        p = f"{r['promo_pct']:.3f}" if r["promo_pct"] is not None else "  --"
        print(
            f"{r['cat']:8s} {r['word'][:14]:14s} {p:>6s} {d:>6s} "
            f"{'Y' if r['in_separation_list'] else '.':>3s}   {'OOD' if r['ood_promoted'] else '.':>3s}  "
            f"{r['concept'][:26]}"
        )
    print(
        "\n=== summary (OOD = promoted >=%.2f but NOT in separation list, disc<%.2f) ===" % (PROMO_THRESH, SEP_THRESH)
    )
    for cat in ("marker", "novel", "control", "antonym"):
        g = [r for r in rows if r["cat"] == cat]
        promoted = [r for r in g if r["promo_pct"] is not None and r["promo_pct"] >= PROMO_THRESH]
        insep = [r for r in g if r["in_separation_list"]]
        ood = [r for r in g if r["ood_promoted"]]
        print(f"  {cat:8s}: n={len(g)}  promoted={len(promoted)}  in-sep-list={len(insep)}  OOD-promoted={len(ood)}")
    nov_ood = [r for r in rows if r["cat"] == "novel" and r["ood_promoted"]]
    print(
        f"\nnovel synonyms that are OOD-promoted (promoted, not a separation word): "
        f"{len(nov_ood)}/{sum(r['cat'] == 'novel' for r in rows)}"
    )
    print("  ", [r["word"] for r in nov_ood])


if __name__ == "__main__":
    main()
