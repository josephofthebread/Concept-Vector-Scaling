from __future__ import annotations

import collections
import json
import re
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from datasets import load_dataset
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from .probe import VEC_ROOT, VEC_TAG

MODEL = "Qwen/Qwen3-8B"
LAYER = 18
KMARK = 12
NSTORY = 30
FRAMES = [" {m}", "This is {m}.", "The {m} one."]
FILLERS = ["thing", "place", "part", "way", "case", "area", "point", "item", "matter", "piece"]
WORD = re.compile(r"[a-z']+")
STOP = set(
    (
        "a an the to of in on at by for with from as into about over under and or but nor so if then is "
        "am are was were be been being have has had do does did will would can could this that these those "
        "it its there here you he she they we me him her them us my your his their our not no i s t re ve "
        "ll d m".split()
    )
)


def main() -> None:
    inj = json.loads(Path("data/inject_concepts.json").read_text())
    want = {(c["concept"], c["antagonist"]): c["pair_id"] for c in inj}
    dfp: dict = collections.defaultdict(collections.Counter)
    dfa: dict = collections.defaultdict(collections.Counter)
    pole: dict = collections.defaultdict(list)
    antg: dict = collections.defaultdict(list)
    ds = load_dataset("AntonKorznikov/feature_stories", split="train")
    for r in ds:
        if r["language"] != "English":
            continue
        k = (r["concept"], r["antagonist"])
        if k in want:
            pw = set(WORD.findall(r["concept_text"].lower()))
            aw = set(WORD.findall(r["antagonist_text"].lower()))
            dfp[k].update(w for w in pw if w not in STOP)
            dfa[k].update(w for w in aw if w not in STOP)
            if len(pole[k]) < NSTORY:
                pole[k].append(r["concept_text"])
            if len(antg[k]) < NSTORY:
                antg[k].append(r["antagonist_text"])
    markers = {
        want[k]: [
            w
            for w, _ in sorted(((w, dfp[k][w] - dfa[k][w]) for w in set(dfp[k]) | set(dfa[k])), key=lambda t: -t[1])[
                :KMARK
            ]
        ]
        for k in want
    }

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto"
    )
    model.eval()
    device = model.device
    with safe_open(VEC_ROOT / VEC_TAG[MODEL] / "diff.safetensors", framework="pt") as h:
        diff = h.get_tensor("diff").float()
    rng = np.random.default_rng(0)

    def resid(text: str, marker: str | None) -> torch.Tensor:
        """Layer-L residual: at the marker's first token if marker given, else mean over content tokens."""
        enc = cast("Any", tok(text, return_tensors="pt", return_offsets_mapping=True))
        offs = enc["offset_mapping"][0].tolist()
        ids = enc["input_ids"].to(device)
        with torch.no_grad():
            hs = model(ids, output_hidden_states=True).hidden_states[LAYER][0].float()
        if marker is not None:
            m = re.search(r"\b" + re.escape(marker) + r"\w*\b", text, re.IGNORECASE)
            if not m:
                return hs[-1]
            for i, (a, b) in enumerate(offs):
                if a <= m.start() < b:
                    return hs[i]
            return hs[-1]
        toks = [str(tok.decode([t])) for t in ids[0].tolist()]
        idx = [i for i in range(1, ids.shape[1]) if re.search(r"[a-z]", toks[i].lower())]
        return hs[idx].mean(0) if idx else hs.mean(0)

    results = []
    for c in inj:
        pid = c["pair_id"]
        k = (c["concept"], c["antagonist"])
        ms = markers[pid]
        M = []
        for w in ms:
            rs = [resid(f.format(m=w), w) for f in FRAMES]
            M.append(torch.stack(rs).mean(0))
        Mt = torch.stack(M)
        q, _ = torch.linalg.qr(Mt.T)
        v = diff[LAYER, pid].to(device)
        vlex = q @ (q.T @ v)
        vsem = v - vlex
        dirs = {"v": v, "vlex": vlex, "vsem": vsem}
        units = {n: (d / d.norm().clamp_min(1e-8)) for n, d in dirs.items()}

        auc: dict[str, dict[str, float]] = {"unmasked": {}, "masked": {}}
        for cond in ("unmasked", "masked"):
            scores: dict[str, dict[str, list[float]]] = {n: {"pos": [], "neg": []} for n in units}
            for pol, texts in (("pos", pole[k]), ("neg", antg[k])):
                for t in texts:
                    tx = t
                    if cond == "masked" and ms:
                        pat = re.compile(r"\b(" + "|".join(re.escape(w) for w in ms) + r")\w*\b", re.IGNORECASE)
                        tx = pat.sub(lambda _: FILLERS[int(rng.integers(0, len(FILLERS)))], t)
                    r = resid(tx, None)
                    for n, u in units.items():
                        scores[n][pol].append(float(r @ u))
            for n in units:
                pos, neg = scores[n]["pos"], scores[n]["neg"]
                a = np.array(pos + neg)
                y = np.array([1] * len(pos) + [0] * len(neg))
                order = a.argsort()
                ranks = np.empty(len(a), float)
                ranks[order] = np.arange(1, len(a) + 1)
                np1, nn = len(pos), len(neg)
                auc[cond][n] = round(
                    float((ranks[y == 1].sum() - np1 * (np1 + 1) / 2) / (np1 * nn)) if np1 and nn else float("nan"), 3
                )
        rec = {
            "concept": c["concept"],
            "pair_id": pid,
            "steer": c["steer"],
            "k": len(ms),
            "cos_v_vlex": round(float((v @ vlex) / (v.norm() * vlex.norm().clamp_min(1e-8))), 3),
            "auroc": auc,
            "markers": ms[:8],
        }
        results.append(rec)
        print(
            f"steer={c['steer']:.2f} lexfrac={rec['cos_v_vlex']:.2f} | "
            f"UNMASK v={auc['unmasked']['v']:.2f} vsem={auc['unmasked']['vsem']:.2f} | "
            f"MASK v={auc['masked']['v']:.2f} vlex={auc['masked']['vlex']:.2f} vsem={auc['masked']['vsem']:.2f}"
            f"  {c['concept'][:26]}",
            flush=True,
        )

    out = Path("work/lexablate/lexablate.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"model": MODEL, "layer": LAYER, "kmark": KMARK, "results": results}, ensure_ascii=False, indent=1)
    )

    def mean(cond: str, n: str) -> float:
        return float(np.mean([r["auroc"][cond][n] for r in results]))

    print("\n=== mean AUROC (pole vs antagonist) over 20 concepts ===")
    print(
        f"  UNMASKED:  v={mean('unmasked', 'v'):.3f}  vlex={mean('unmasked', 'vlex'):.3f}  "
        f"vsem={mean('unmasked', 'vsem'):.3f}"
    )
    print(
        f"  MASKED :   v={mean('masked', 'v'):.3f}  vlex={mean('masked', 'vlex'):.3f}  "
        f"vsem={mean('masked', 'vsem'):.3f}   (marker-free concept detection)"
    )
    print(f"  mean lexical fraction cos(v,vlex) = {np.mean([r['cos_v_vlex'] for r in results]):.3f}")


if __name__ == "__main__":
    main()
