#!/bin/bash
#SBATCH --job-name=genvectors-vectors-sb
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/vectors-sb-%j.out
#SBATCH --error=logs/vectors-sb-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun .venv/bin/python - <<'EOF'
import json
import numpy as np
from pathlib import Path
from safetensors import safe_open
from exp_genvectors.artifacts import atomic_safetensors, canonical_json

MANDARIN = 2   # LANGUAGES = (English, German, Mandarin Chinese, Russian, Spanish)
RUNS = {"Qwen_Qwen3-8B-Base--raw--semantic": "raw",
        "Qwen_Qwen3-8B--assistant--semantic": "assistant"}

for tag, presentation in RUNS.items():
    with safe_open(f"work/captures/{tag}/shard-000.safetensors", framework="np") as f:
        manifest = json.loads(f.metadata()["manifest"])
        lang_mu = f.get_tensor("language_sum").astype(np.float32)     # packed MEANS [L,P,pole,lang,H]
        lang_ct = f.get_tensor("language_count").astype(np.float32)   # [P,pole,lang]
    L, P, _, _, H = lang_mu.shape
    keep = [i for i in range(lang_ct.shape[2]) if i != MANDARIN]
    w = lang_ct[:, :, keep]
    pole = (lang_mu[:, :, :, keep] * w[None, ..., None]).sum(3) / w.sum(2)[None, ..., None]
    corpus = (pole * w.sum(2)[None, ..., None]).sum((1, 2)) / w.sum()

    raw = {"diff": pole[:, :, 0] - pole[:, :, 1],
           "concept_centered": pole[:, :, 0] - corpus[:, None],
           "antagonist_centered": pole[:, :, 1] - corpus[:, None]}
    out, mid = {}, L // 2
    for name, v in raw.items():
        centered = v - v.mean(axis=1, keepdims=True)
        out[name] = centered.astype(np.float32)
        cos = np.einsum("ph,ph->p", centered[mid], v[mid]) / (
            np.linalg.norm(centered[mid], axis=1) * np.linalg.norm(v[mid], axis=1))
        ratio = np.linalg.norm(centered[mid].mean(0)) / np.linalg.norm(v[mid].mean(0))
        print(f"{tag} {name}: median cos(centered, uncentered)@L{mid} = {np.median(cos):.3f}; "
              f"shared-axis norm after centring = {ratio:.2e} of original", flush=True)
    meta = {"manifest": canonical_json({
        "schema_version": 3, "model": manifest["model"], "presentation": presentation,
        "dataset": "semantic-sb", "layers": L, "hidden_size": H, "pairs": P,
        "composition": "English, German, Russian, Spanish (Mandarin excluded)",
        "postprocessing": "across-concept mean subtracted before normalization",
        "axes": {n: ["layer", "pair", "hidden"] for n in out},
        "source_capture": manifest["merge_key"]})}
    dest = Path(f"work/vectors-sb/{tag}")
    for name, v in out.items():
        atomic_safetensors({name: v}, dest / f"{name}.safetensors", meta)
    (dest / "manifest.json").write_text(meta["manifest"])
    print(f"wrote {dest}: {L} layers, {P} pairs, {H} hidden", flush=True)
print("done", flush=True)
EOF
