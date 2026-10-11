from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
from scipy.stats import spearmanr

from .probe import VEC_ROOT, VEC_TAG

MODELS = {"Qwen": "Qwen/Qwen3-8B", "Llama": "unsloth/Meta-Llama-3.1-8B-Instruct", "Gemma": "google/gemma-2-9b-it"}
DEPTHS = np.round(np.linspace(1 / 3, 2 / 3, 11), 4)


def main() -> None:
    diff = {}
    for name, model in MODELS.items():
        with safe_open(VEC_ROOT / VEC_TAG[model] / "diff.safetensors", framework="pt") as h:
            diff[name] = h.get_tensor("diff").float().numpy()
        print(f"{name}: {diff[name].shape}", flush=True)
    n = diff["Qwen"].shape[1]
    iu = np.triu_indices(n, 1)
    names = list(MODELS)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]

    rows = []
    print("depth   " + "  ".join(f"{a}-{b}: layers  spear  pear  rmse" for a, b in pairs), flush=True)
    for d in DEPTHS:
        S, layer = {}, {}
        for name in names:
            nb = diff[name].shape[0] - 1
            L = int(round(float(d) * nb))
            v = diff[name][L].astype(np.float64)
            v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
            S[name], layer[name] = (v @ v.T)[iu], L
        line = f"{float(d):.3f}  "
        for a, b in pairs:
            sp = float(spearmanr(S[a], S[b]).statistic)
            pe = float(np.corrcoef(S[a], S[b])[0, 1])
            rm = float(np.sqrt(np.mean((S[a] - S[b]) ** 2)))
            rows.append(
                {
                    "depth": float(d),
                    "pair": f"{a}-{b}",
                    "layer_a": layer[a],
                    "layer_b": layer[b],
                    "spearman": round(sp, 4),
                    "pearson": round(pe, 4),
                    "rmse": round(rm, 4),
                }
            )
            line += f"  {a[0]}{b[0]} L{layer[a]:2d}/L{layer[b]:2d} {sp:.3f} {pe:.3f} {rm:.3f}"
        print(line, flush=True)

    Path("work/depthrepro").mkdir(parents=True, exist_ok=True)
    Path("work/depthrepro/depthrepro.json").write_text(
        json.dumps(
            {
                "depths": [float(x) for x in DEPTHS],
                "blocks": {k: int(diff[k].shape[0] - 1) for k in names},
                "rows": rows,
            },
            indent=1,
        )
    )
    print("\nwrote work/depthrepro/depthrepro.json")


if __name__ == "__main__":
    main()
