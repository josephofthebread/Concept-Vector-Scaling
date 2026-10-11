from __future__ import annotations

import json
import re
from argparse import ArgumentParser
from pathlib import Path

TT = Path("../exp_trigger_prompts")
MID = {"Qwen/Qwen3-8B": 18, "google/gemma-2-9b-it": 21, "unsloth/Meta-Llama-3.1-8B-Instruct": 16}
TRAIN_NONLATIN = {"Han", "Cyrillic"}
LATINW = re.compile(r"[a-zà-ÿ']+")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B", choices=sorted(MID))
    model = parser.parse_args().model
    tag = model.split("/")[-1]
    LAYER = str(MID[model])
    ll = json.loads((TT / f"work/logitlens/logitlens-{tag}.json").read_text())

    from datasets import load_dataset

    ds = load_dataset("AntonKorznikov/feature_stories", split="train")
    global_latin: set[str] = set()
    blob_parts: list[str] = []
    for r in ds:
        for field in ("concept_text", "antagonist_text"):
            t = r[field].lower()
            global_latin.update(LATINW.findall(t))
            blob_parts.append(t)
    blob = "\n".join(blob_parts)
    print(f"corpus: {len(global_latin)} Latin word types, blob {len(blob) / 1e6:.1f}M chars", flush=True)

    def is_novel(token: str, latin: bool) -> bool:
        t = token.strip().lower()
        if not t:
            return False
        return (t not in global_latin) if latin else (token.strip() not in blob)

    rows = []
    agg = {"Latin": [0, 0], "TrainNonLatin": [0, 0], "OOD": [0, 0]}
    for r in ll["results"]:
        L = r["layers"][LAYER]
        rec: dict = {
            "concept": r["concept"],
            "steer": r["steer"],
            "novel_latin": [],
            "novel_train_nonlatin": [],
            "ood_tokens": [],
            "ood_novel": [],
        }
        for w in L["latin"]:
            nov = is_novel(w, latin=True)
            agg["Latin"][0] += 1
            agg["Latin"][1] += nov
            if nov:
                rec["novel_latin"].append(w)
        for script, words in L["other"].items():
            grp = "TrainNonLatin" if script in TRAIN_NONLATIN else "OOD"
            for w in words:
                nov = is_novel(w, latin=False)
                agg[grp][0] += 1
                agg[grp][1] += nov
                if grp == "TrainNonLatin" and nov:
                    rec["novel_train_nonlatin"].append(f"{w}({script})")
                if grp == "OOD":
                    rec["ood_tokens"].append(f"{w}({script})")
                    if nov:
                        rec["ood_novel"].append(f"{w}({script})")
        rows.append(rec)
        print(
            f"steer={r['steer']:.2f}  {r['concept'][:26]:26s} | novel-Latin={len(rec['novel_latin'])} "
            f"{rec['novel_latin'][:5]}  novel-Zh/Ru={rec['novel_train_nonlatin'][:4]}",
            flush=True,
        )

    out = Path("work/lexnovel")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"lexnovel-{tag}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print("\n=== novelty by group (token occurs 0 times anywhere in feature_stories) ===")
    for g, (tot, ncount) in agg.items():
        print(f"  {g:14s}: {ncount}/{tot} novel  ({ncount / max(tot, 1):.1%})")
    print(
        "  (Latin ~= English/German/Spanish; TrainNonLatin = Mandarin+Russian; "
        "OOD = Korean/Japanese/Arabic/Hebrew, not in the 5 training languages)"
    )


if __name__ == "__main__":
    main()
