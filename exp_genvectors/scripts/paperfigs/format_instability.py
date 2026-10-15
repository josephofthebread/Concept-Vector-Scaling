from itertools import combinations
from pathlib import Path

import numpy as np
from safetensors import safe_open
from scipy.stats import pearsonr, spearmanr

VEC = Path("work/vectors")
TAGS = ["user", "assistant", "paired_assistant"]
EPS = 1e-7
MODELS = {
    "Qwen3-8B": ("Qwen_Qwen3-8B", range(12, 25)),
    "gemma-2-9b": ("google_gemma-2-9b-it", range(14, 29)),
    "Llama-3.1-8B": ("unsloth_Meta-Llama-3.1-8B-Instruct", range(11, 22)),
}


def unit(stem: str, tag: str, layer: int) -> np.ndarray:
    with safe_open(str(VEC / f"{stem}--{tag}--semantic" / "diff.safetensors"), framework="np") as h:
        m = h.get_slice("diff")[layer].astype(np.float64)
    return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-9)


def main() -> None:
    inst = {}
    for label, (stem, band) in MODELS.items():
        per_layer = []
        for layer in band:
            u = {t: unit(stem, t, layer) for t in TAGS}
            per_layer.append(
                np.mean(
                    [
                        np.log10(1 - np.clip(np.einsum("kh,kh->k", u[a], u[b]), -1, 1 - EPS))
                        for a, b in combinations(TAGS, 2)
                    ],
                    axis=0,
                )
            )
        inst[label] = np.stack(per_layer).mean(0)
        print(label, "median instability (log10(1-cos))", round(float(np.median(inst[label])), 3), flush=True)
    for a, b in combinations(MODELS, 2):
        x, y = inst[a], inst[b]
        print(
            f"{a:>12s} vs {b:<12s}  Spearman {spearmanr(x, y)[0]:.3f}  Pearson(log) {pearsonr(x, y)[0]:.3f}  "
            f"Pearson(linear 1-cos) {pearsonr(10**x, 10**y)[0]:.3f}"
        )
    out = Path("work/tmp/paperfigs")
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "format-instability.npz", **inst)
    print("wrote", out / "format-instability.npz")


if __name__ == "__main__":
    main()
