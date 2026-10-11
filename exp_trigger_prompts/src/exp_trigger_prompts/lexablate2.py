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


def auroc(pos: list[float], neg: list[float]) -> float:
    a = np.array(pos + neg)
    y = np.array([1] * len(pos) + [0] * len(neg))
    order = a.argsort()
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1)
    np1, nn = len(pos), len(neg)
    return float((ranks[y == 1].sum() - np1 * (np1 + 1) / 2) / (np1 * nn)) if np1 and nn else float("nan")


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
            dfp[k].update(w for w in set(WORD.findall(r["concept_text"].lower())) if w not in STOP)
            dfa[k].update(w for w in set(WORD.findall(r["antagonist_text"].lower())) if w not in STOP)
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

    def hidden_of(text: str) -> tuple[torch.Tensor, list[str], list[tuple[int, int]]]:
        enc = cast("Any", tok(text, return_tensors="pt", return_offsets_mapping=True))
        ids = enc["input_ids"].to(device)
        with torch.no_grad():
            hs = model(ids, output_hidden_states=True).hidden_states[LAYER][0].float()
        toks = [str(tok.decode([t])) for t in ids[0].tolist()]
        return hs, toks, enc["offset_mapping"][0].tolist()

    def marker_resid(text: str, marker: str) -> torch.Tensor:
        hs, _toks, offs = hidden_of(text)
        m = re.search(r"\b" + re.escape(marker) + r"\w*\b", text, re.IGNORECASE)
        if m:
            for i, (a, b) in enumerate(offs):
                if a <= m.start() < b:
                    return hs[i]
        return hs[-1]

    results = []
    for c in inj:
        pid, k = c["pair_id"], (c["concept"], c["antagonist"])
        ms = markers[pid]
        M = torch.stack([torch.stack([marker_resid(f.format(m=w), w) for f in FRAMES]).mean(0) for w in ms])
        q, _ = torch.linalg.qr(M.T)
        v = diff[LAYER, pid].to(device)
        vlex = q @ (q.T @ v)
        units = {
            "v": v / v.norm().clamp_min(1e-8),
            "vlex": vlex / vlex.norm().clamp_min(1e-8),
            "vsem": (v - vlex) / (v - vlex).norm().clamp_min(1e-8),
        }

        out: dict[str, Any] = {"mean": {}, "spike": {}, "argmax_marker": {}}
        for cond in ("unmasked", "masked"):
            sc: dict[str, dict[str, list[float]]] = {
                n: {"mean_pos": [], "mean_neg": [], "sp_pos": [], "sp_neg": []} for n in units
            }
            amk: dict[str, list[int]] = {n: [] for n in units}
            for pol, texts in (("pos", pole[k]), ("neg", antg[k])):
                for t in texts:
                    tx = t
                    if cond == "masked":
                        pat = re.compile(r"\b(" + "|".join(re.escape(w) for w in ms) + r")\w*\b", re.IGNORECASE)
                        tx = pat.sub(lambda _: FILLERS[int(rng.integers(0, len(FILLERS)))], t)
                    hs, toks, _offs = hidden_of(tx)
                    idx = [i for i in range(1, hs.shape[0]) if re.search(r"[a-z]", toks[i].lower())]
                    if len(idx) < 3:
                        continue
                    H = hs[torch.tensor(idx, device=device)]
                    Hn = H / H.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    for n, u in units.items():
                        cos = Hn @ u
                        z = (cos - cos.mean()) / cos.std().clamp_min(1e-6)
                        sc[n][f"mean_{pol}"].append(float(cos.mean()))
                        sc[n][f"sp_{pol}"].append(float(z.max()))
                        if cond == "unmasked" and pol == "pos":
                            atok = toks[idx[int(cos.argmax().item())]].strip().lower()
                            amk[n].append(int(any(atok.startswith(w[:4]) or w.startswith(atok) for w in ms if atok)))
            for n in units:
                out["mean"].setdefault(cond, {})[n] = round(auroc(sc[n]["mean_pos"], sc[n]["mean_neg"]), 3)
                out["spike"].setdefault(cond, {})[n] = round(auroc(sc[n]["sp_pos"], sc[n]["sp_neg"]), 3)
                if cond == "unmasked":
                    out["argmax_marker"][n] = round(float(np.mean(amk[n])), 2) if amk[n] else None
        rec = {
            "concept": c["concept"],
            "steer": c["steer"],
            "lexfrac": round(float((v @ vlex) / (v.norm() * vlex.norm().clamp_min(1e-8))), 3),
            **out,
        }
        results.append(rec)
        print(
            f"steer={c['steer']:.2f} lexfrac={rec['lexfrac']:.2f} | "
            f"SPIKE-unmask v={out['spike']['unmasked']['v']:.2f} vlex={out['spike']['unmasked']['vlex']:.2f} "
            f"vsem={out['spike']['unmasked']['vsem']:.2f} | SPIKE-mask vlex={out['spike']['masked']['vlex']:.2f} "
            f"vsem={out['spike']['masked']['vsem']:.2f} | argmax=marker(vlex)={out['argmax_marker']['vlex']}"
            f"  {c['concept'][:22]}",
            flush=True,
        )

    Path("work/lexablate2").mkdir(parents=True, exist_ok=True)
    Path("work/lexablate2/lexablate2.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))

    def mn(metric: str, cond: str, n: str) -> float:
        return float(np.mean([r[metric][cond][n] for r in results]))

    print("\n=== mean AUROC over 20 concepts (SPIKE = top-token z; MEAN = mean cosine) ===")
    for metric in ("spike", "mean"):
        for cond in ("unmasked", "masked"):
            print(
                f"  {metric.upper():5s} {cond:8s}: v={mn(metric, cond, 'v'):.3f}  "
                f"vlex={mn(metric, cond, 'vlex'):.3f}  vsem={mn(metric, cond, 'vsem'):.3f}"
            )
    print(
        f"  argmax=marker frac (vlex, unmasked) = "
        f"{np.mean([r['argmax_marker']['vlex'] for r in results if r['argmax_marker']['vlex'] is not None]):.2f}"
    )
    print(f"  mean lexical fraction = {np.mean([r['lexfrac'] for r in results]):.3f}")


if __name__ == "__main__":
    main()
