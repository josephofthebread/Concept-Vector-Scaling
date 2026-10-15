from itertools import combinations
from pathlib import Path

import numpy as np
from safetensors import safe_open

FORMATS = ["raw", "paired", "user", "assistant", "paired-assistant"]
TAG = {
    "raw": ("base", "raw"),
    "paired": ("base", "paired_raw"),
    "user": ("it", "user"),
    "assistant": ("it", "assistant"),
    "paired-assistant": ("it", "paired_assistant"),
}
MODELS = {
    "Llama-3.1-8B": ("unsloth_Meta-Llama-3.1-8B", "unsloth_Meta-Llama-3.1-8B-Instruct", (16, 21)),
    "gemma-2-9b": ("google_gemma-2-9b", "google_gemma-2-9b-it", (21, 28)),
}


def unit_rows(stem: str, tag: str, layer: int) -> np.ndarray:
    with safe_open(f"work/vectors/{stem}--{tag}--semantic/diff.safetensors", framework="np") as h:
        m = np.asarray(h.get_slice("diff")[layer], dtype=np.float64)
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)


def main() -> None:
    out = Path("work/tmp/paperfigs")
    out.mkdir(parents=True, exist_ok=True)
    result = {}
    for label, (base, it, layers) in MODELS.items():
        for layer in layers:
            vec = {f: unit_rows(base if TAG[f][0] == "base" else it, TAG[f][1], layer) for f in FORMATS}
            cos = np.stack([np.einsum("kh,kh->k", vec[a], vec[b]) for a, b in combinations(FORMATS, 2)])
            result[f"{label}_L{layer}"] = cos.astype(np.float32)
            print(
                label,
                layer,
                "mean per pair:",
                dict(zip([f"{a}|{b}" for a, b in combinations(FORMATS, 2)], cos.mean(1).round(3))),
                flush=True,
            )
    np.savez(out / "format-cosines.npz", **result, formats=np.array(FORMATS))
    print("wrote", out / "format-cosines.npz")


if __name__ == "__main__":
    main()
