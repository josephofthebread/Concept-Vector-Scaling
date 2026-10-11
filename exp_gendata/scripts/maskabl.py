from __future__ import annotations

import collections
import json
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

TAG = {
    "Qwen/Qwen3-8B": "Qwen_Qwen3-8B--assistant--semantic",
    "unsloth/Meta-Llama-3.1-8B-Instruct": "unsloth_Meta-Llama-3.1-8B-Instruct--assistant--semantic",
    "google/gemma-2-9b-it": "google_gemma-2-9b-it--assistant--semantic",
}
KMARK = 30
BATCH = 16
FILLERS = ["thing", "place", "part", "way", "case", "area", "point", "item", "matter", "piece"]
WORD = re.compile(r"[a-z']+")


def mask_text(text: str, wordset: set, rng: np.random.Generator) -> str:
    if not wordset:
        return text
    pat = re.compile(r"\b(" + "|".join(re.escape(w) for w in wordset) + r")\b", re.IGNORECASE)
    return pat.sub(lambda _: FILLERS[int(rng.integers(0, len(FILLERS)))], text)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B", choices=sorted(TAG))
    parser.add_argument("--sample", type=int, default=200, help="concepts, evenly spaced over English AUROC")
    parser.add_argument("--out-dir", type=Path, default=Path("work/maskabl"))
    args = parser.parse_args()
    MODEL = args.model
    mtag = MODEL.split("/")[-1]

    lb = json.loads(Path("work/lexbias/summary.json").read_text())
    scored = [
        (s["auroc_by_lang"]["English"], s["concept"], s["antagonist"])
        for s in lb["summary"]
        if s["auroc_by_lang"].get("English")
    ]
    scored.sort()
    step = len(scored) / args.sample
    picked = [scored[int(i * step)] for i in range(args.sample)]
    want = {(c, a): auroc for auroc, c, a in picked}
    print(f"{len(want)} concepts, English AUROC {picked[0][0]:.3f}..{picked[-1][0]:.3f}", flush=True)

    ds = load_dataset("AntonKorznikov/feature_stories", split="train")
    pole: dict = collections.defaultdict(list)
    antg: dict = collections.defaultdict(list)
    for r in ds:
        if r["language"] != "English":
            continue
        k = (r["concept"], r["antagonist"])
        if k in want:
            pole[k].append(r["concept_text"])
            antg[k].append(r["antagonist_text"])

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    attn = "eager" if "gemma" in MODEL else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation=attn, device_map="auto"
    )
    model.eval()
    device = model.device
    mid = round(model.config.num_hidden_layers / 2)
    LAYERS = [mid - 3, mid, mid + 3]
    rng = np.random.default_rng(0)
    print(f"{MODEL}: layers {LAYERS}", flush=True)

    def mean_hidden(texts: list[str]) -> dict[int, np.ndarray]:
        acc = {ell: np.zeros(model.config.hidden_size, np.float64) for ell in LAYERS}
        n = 0
        for i in range(0, len(texts), BATCH):
            enc = cast(
                "Any", tok(texts[i : i + BATCH], return_tensors="pt", truncation=True, max_length=512, padding=True)
            )
            am = enc["attention_mask"].to(device)
            with torch.no_grad():
                out = model(enc["input_ids"].to(device), attention_mask=am, output_hidden_states=True)
            m = am.unsqueeze(-1).float()
            m[:, 0] = 0.0
            denom = m.sum(dim=1).clamp_min(1.0)
            for ell in LAYERS:
                h = out.hidden_states[ell].float()
                acc[ell] += ((h * m).sum(dim=1) / denom).sum(dim=0).cpu().numpy()
            n += enc["input_ids"].shape[0]
        return {ell: acc[ell] / max(n, 1) for ell in LAYERS}

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    results = []
    for idx, (k, auroc) in enumerate(want.items()):
        pt, at = pole[k], antg[k]
        if not pt or not at:
            continue
        dfp: collections.Counter = collections.Counter()
        dfa: collections.Counter = collections.Counter()
        for t in pt:
            dfp.update(set(WORD.findall(t.lower())))
        for t in at:
            dfa.update(set(WORD.findall(t.lower())))
        allw = set(dfp) | set(dfa)
        gap = sorted(allw, key=lambda w: dfp[w] - dfa[w])
        disc = set(gap[:KMARK]) | set(gap[-KMARK:])
        cand = [w for w in allw if w not in disc and dfp[w] + dfa[w] >= 5]
        rand = set(rng.choice(cand, size=min(2 * KMARK, len(cand)), replace=False).tolist())

        v = {}
        for vtag, wordset in [("orig", set()), ("disc", disc), ("rand", rand)]:
            pm = mean_hidden([mask_text(t, wordset, rng) for t in pt])
            am_ = mean_hidden([mask_text(t, wordset, rng) for t in at])
            v[vtag] = {ell: pm[ell] - am_[ell] for ell in LAYERS}
        results.append(
            {
                "concept": k[0],
                "antagonist": k[1],
                "auroc": auroc,
                "n_pole": len(pt),
                "n_antag": len(at),
                "cos_disc": {ell: cos(v["orig"][ell], v["disc"][ell]) for ell in LAYERS},
                "cos_rand": {ell: cos(v["orig"][ell], v["rand"][ell]) for ell in LAYERS},
            }
        )
        if idx % 20 == 0:
            r = results[-1]
            print(
                f"{idx}/{len(want)} {k[0][:28]:28s} AUROC={auroc:.2f} "
                f"cos_disc={r['cos_disc'][mid]:.2f} cos_rand={r['cos_rand'][mid]:.2f}",
                flush=True,
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"maskabl-{mtag}.json"
    out.write_text(json.dumps({"model": MODEL, "layers": LAYERS, "results": results}, ensure_ascii=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
