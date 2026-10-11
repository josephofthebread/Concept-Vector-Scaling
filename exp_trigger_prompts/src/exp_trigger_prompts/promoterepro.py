from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import try_to_load_from_cache
from safetensors import safe_open
from transformers import AutoTokenizer

from .probe import VEC_ROOT, VEC_TAG

MODELS = {
    "Qwen": ("Qwen/Qwen3-8B", 21),
    "Llama": ("unsloth/Meta-Llama-3.1-8B-Instruct", 18),
    "Gemma": ("google/gemma-2-9b-it", 24),
}
K = 20
WORD = re.compile(r"\w", re.UNICODE)


def promoted_words(model: str, layer: int) -> list[list[str]]:
    with safe_open(VEC_ROOT / VEC_TAG[model] / "diff.safetensors", framework="pt") as h:
        v = h.get_tensor("diff")[layer].float()
    v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-9)
    idx = try_to_load_from_cache(model, "model.safetensors.index.json")
    if not isinstance(idx, str):
        raise SystemExit(f"{model} index not cached")
    root = Path(idx).parent
    wm = json.loads(Path(idx).read_text())["weight_map"]
    head = "lm_head.weight" if "lm_head.weight" in wm else "model.embed_tokens.weight"
    out = {}
    for name in (head, "model.norm.weight"):
        with safe_open(root / wm[name], framework="pt") as h:
            out[name] = h.get_tensor(name).float()
    gamma = out["model.norm.weight"]
    if "gemma" in model.lower():
        gamma = gamma + 1.0
    W = out[head]
    tok = AutoTokenizer.from_pretrained(model)
    words: list[list[str]] = []
    scores = (v * gamma) @ W.T
    top = torch.topk(scores, K * 3, dim=1).indices.tolist()
    for ids in top:
        seen: list[str] = []
        for tid in ids:
            w = str(tok.decode([tid])).strip().lower()
            if w and "�" not in w and WORD.search(w) and w not in seen:
                seen.append(w)
            if len(seen) == K:
                break
        words.append(seen)
    del W, scores
    return words


def jac(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def main() -> None:
    names = list(MODELS)
    W = {name: [set(x) for x in promoted_words(*MODELS[name])] for name in names}
    for name in names:
        print(f"{name}: promoted words for {len(W[name])} concepts", flush=True)
    n = len(W[names[0]])
    pairs = [(0, 1), (0, 2), (1, 2)]

    agree = np.array([np.mean([jac(W[names[a]][i], W[names[b]][i]) for a, b in pairs]) for i in range(n)])
    rng = np.random.default_rng(0)
    chance = np.zeros(n)
    for _ in range(5):
        perm = {name: rng.permutation(n) for name in names}
        for i in range(n):
            chance[i] += np.mean([jac(W[names[a]][i], W[names[b]][perm[names[b]][i]]) for a, b in pairs])
    chance /= 5

    repro = np.load("work/crossrepro.npz", allow_pickle=True)["repro"]
    concept = list(np.load("work/crossrepro.npz", allow_pickle=True)["concept"])
    from scipy.stats import pearsonr, spearmanr

    m = np.isfinite(agree) & np.isfinite(repro)
    sp, pe = spearmanr(agree[m], repro[m]).statistic, pearsonr(agree[m], repro[m])[0]

    print(f"\ncross-model promoted-word agreement (top-{K}):")
    print(
        f"  mean agree = {agree.mean():.3f}   mean chance = {chance.mean():.3f}   "
        f"lift = {agree.mean() - chance.mean():+.3f}"
    )
    print(f"  agree vs geometric repro: Spearman={sp:.3f}  Pearson={pe:.3f}")
    o = np.argsort(-agree)
    print("\nHIGHEST word-agreement:")
    for i in o[:6]:
        print(f"  {agree[i]:.2f} {concept[i][:34]:34} Q:{sorted(W['Qwen'][i])[:6]}")
    print("LOWEST word-agreement:")
    for i in o[-6:]:
        print(f"  {agree[i]:.2f} {concept[i][:34]:34} Q:{sorted(W['Qwen'][i])[:6]}")

    Path("work/promoterepro").mkdir(parents=True, exist_ok=True)
    Path("work/promoterepro/promoterepro.json").write_text(
        json.dumps(
            {
                "K": K,
                "mean_agree": float(agree.mean()),
                "mean_chance": float(chance.mean()),
                "spearman_repro": float(sp),
                "pearson_repro": float(pe),
                "concept": concept,
                "agree": [round(float(x), 4) for x in agree],
                "chance": [round(float(x), 4) for x in chance],
                "repro": [round(float(x), 4) for x in repro],
                "words": {name: [sorted(s) for s in W[name]] for name in names},
            },
            ensure_ascii=False,
        )
    )
    print("\nwrote work/promoterepro/promoterepro.json")


if __name__ == "__main__":
    main()
