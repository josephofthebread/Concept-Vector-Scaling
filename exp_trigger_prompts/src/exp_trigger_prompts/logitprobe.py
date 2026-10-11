from __future__ import annotations

import json
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from .probe import VEC_ROOT, VEC_TAG

LATINW = re.compile(r"[a-zà-ÿ']+")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B", choices=sorted(VEC_TAG))
    MODEL = parser.parse_args().model
    conc = json.loads(Path("data/inject_concepts.json").read_text())
    probes = json.loads(Path("data/probe_words.json").read_text())
    pid_of = {c["concept"]: c["pair_id"] for c in conc}

    ds = load_dataset("AntonKorznikov/feature_stories", split="train")
    corpus_latin: set[str] = set()
    for r in ds:
        corpus_latin.update(LATINW.findall(r["concept_text"].lower()))
        corpus_latin.update(LATINW.findall(r["antagonist_text"].lower()))

    tok = AutoTokenizer.from_pretrained(MODEL)
    attn = "eager" if "gemma" in MODEL else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation=attn, device_map="auto"
    )
    model.eval()
    device = model.device
    LAYER = round(model.config.num_hidden_layers / 2)
    unemb = model.get_output_embeddings().weight.float()
    gamma = model.model.norm.weight.float().to(device)
    with safe_open(VEC_ROOT / VEC_TAG[MODEL] / "diff.safetensors", framework="pt") as h:
        diff = h.get_tensor("diff")

    results = []
    for concept, cats in probes.items():
        pid = pid_of[concept]
        d = diff[LAYER, pid].float().to(device)
        d = d / d.norm().clamp_min(1e-8)
        proj = unemb @ (gamma * d)
        mu, sd = proj.mean(), proj.std().clamp_min(1e-9)
        frontier = torch.topk(proj, 30).values.min()
        frontier_pct = (proj < frontier).float().mean().item()
        rec: dict[str, Any] = {"concept": concept, "pair_id": pid, "frontier_pct": round(frontier_pct, 5), "words": {}}
        for cat, words in cats.items():
            entries = []
            for w in words:
                ids = tok(" " + w, add_special_tokens=False)["input_ids"]
                tid, ntok = ids[0], len(ids)
                val = proj[tid]
                pct = (proj < val).float().mean().item()
                entries.append(
                    {
                        "word": w,
                        "pct": round(pct, 4),
                        "z": round(float((val - mu) / sd), 2),
                        "in_corpus": w.lower() in corpus_latin,
                        "ntok": ntok,
                        "in_topk": pct >= frontier_pct,
                    }
                )
            rec["words"][cat] = entries
        results.append(rec)
        line = []
        for cat in ("marker", "novel", "control", "antonym"):
            ps = [e["pct"] for e in rec["words"].get(cat, [])]
            line.append(f"{cat[:4]}={sum(ps) / len(ps):.3f}" if ps else f"{cat[:4]}=-")
        print(f"{concept[:28]:28s} frontier={frontier_pct:.4f} | " + "  ".join(line), flush=True)
        for e in rec["words"]["novel"]:
            print(
                f"      novel {e['word']:14s} pct={e['pct']:.4f} z={e['z']:+.1f} "
                f"inCorpus={e['in_corpus']} nTok={e['ntok']} inTopK={e['in_topk']}",
                flush=True,
            )

    out = Path(f"work/logitprobe/logitprobe-{MODEL.split('/')[-1]}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1))

    print("\n=== mean promotion percentile by category (1.0 = top of vocab) ===")
    for cat in ("marker", "novel", "control", "antonym"):
        v = [e["pct"] for r in results for e in r["words"].get(cat, [])]
        print(f"  {cat:8s}: {sum(v) / len(v):.3f}")
    nov = [e for r in results for e in r["words"]["novel"]]
    print(
        f"novel synonyms absent from corpus: {sum(not e['in_corpus'] for e in nov)}/{len(nov)}; "
        f"of those, pct>0.99: {sum((not e['in_corpus']) and e['pct'] > 0.99 for e in nov)}"
    )


if __name__ == "__main__":
    main()
