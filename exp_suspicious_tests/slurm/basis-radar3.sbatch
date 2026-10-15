#!/bin/bash
#SBATCH --job-name=basis-radar3
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=logs/basis-radar3-%j.out
#SBATCH --error=logs/basis-radar3-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p work/basis
srun .venv/bin/python - <<'EOF'
import json
import numpy as np
from safetensors import safe_open
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

K, CAND, TOP = 6, 8, 8
NAMES = ["valence", "order &\nboundaries", "authenticity", "self-\nexpansion",
         "epistemic\nhumility", "traditionalism"]
FAMS = [("Qwen3-8B", "Qwen_Qwen3-8B", 25),
        ("gemma-2-9b", "google_gemma-2-9b-it", 29),
        ("Llama-3.1-8B", "unsloth_Meta-Llama-3.1-8B-Instruct", 22)]
pairs = json.load(open("../exp_genvectors/data/semantic.ontology.json"))["pairs"]
names_full = [f"{p['concept']} vs {p['antagonist']}" for p in pairs]


def loadings(stem, layer, k):
    with safe_open(f"../exp_genvectors/work/vectors/{stem}--assistant--semantic/diff.safetensors",
                   framework="np") as h:
        M = h.get_tensor("diff")[layer].astype(np.float64)
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    _, s, Vt = np.linalg.svd(M, full_matrices=False)
    cos = M @ Vt[:k].T
    return cos, (s ** 2 / (s ** 2).sum())[:k]


cosQ, shareQ = loadings(FAMS[0][1], FAMS[0][2], K)
flip = np.sign(cosQ[np.abs(cosQ).argmax(axis=0), np.arange(K)])
cosQ *= flip

profiles = {"Qwen3-8B": cosQ}
report = []
for fam, stem, layer in FAMS[1:]:
    cosF, shareF = loadings(stem, layer, CAND)
    C = np.corrcoef(cosQ.T, cosF.T)[:K, K:]                     # [K, CAND]
    chosen = {}
    order = np.dstack(np.unravel_index(np.argsort(-np.abs(C), axis=None), C.shape))[0]
    used_q, used_f = set(), set()
    for qi, fi in order:
        if qi in used_q or fi in used_f:
            continue
        chosen[qi] = (fi, C[qi, fi]); used_q.add(qi); used_f.add(fi)
        if len(chosen) == K:
            break
    aligned = np.empty_like(cosQ)
    report.append(f"\n### {fam} (layer {layer}) mode matching -> qwen axes")
    for qi in range(K):
        fi, c = chosen[qi]
        aligned[:, qi] = cosF[:, fi] * np.sign(c)
        report.append(f"  qwen mode {qi} ({NAMES[qi].replace(chr(10), ' ')}): "
                      f"{fam} mode {fi}, |corr|={abs(c):.3f}, sigma2 share {shareF[fi]*100:.1f}%")
    profiles[fam] = aligned
    for qi in range(K):
        vals = aligned[:, qi]
        report.append(f"  -- axis {qi} ({NAMES[qi].replace(chr(10), ' ')}) top by +cos:")
        for i in np.argsort(vals)[::-1][:TOP]:
            report.append(f"     {vals[i]:+.3f}  {names_full[i][:70]}")
        report.append(f"  -- axis {qi} top by -cos:")
        for i in np.argsort(vals)[:TOP]:
            report.append(f"     {vals[i]:+.3f}  {names_full[i][:70]}")
print("\n".join(report))
with open("work/basis/radar3-matching.txt", "w") as f:
    f.write("\n".join(report) + "\n")

ang = np.linspace(0, 2 * np.pi, K, endpoint=False)
ang_c = np.concatenate([ang, ang[:1]])
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), subplot_kw={"projection": "polar"})
for ax, (fam, _, layer) in zip(axes, FAMS):
    E = profiles[fam] ** 2
    for row in E:
        ax.plot(ang_c, np.concatenate([row, row[:1]]), color="#777", alpha=0.03, linewidth=0.4)
    med = np.median(E, axis=0); p95 = np.percentile(E, 95, axis=0)
    ax.plot(ang_c, np.concatenate([med, med[:1]]), color="#c62828", linewidth=1.5)
    ax.plot(ang_c, np.concatenate([p95, p95[:1]]), color="#1565c0", linewidth=1.2, linestyle="--")
    ax.set_xticks(ang)
    ax.set_xticklabels(NAMES, fontsize=6.5)
    # tick-label transforms are reset at draw time, so place this one manually
    labels = ax.get_xticklabels(); labels[3].set_text("")
    ax.set_xticklabels([t.get_text() for t in labels], fontsize=6.5)
    ax.text(np.pi, 0.74, "self-\nexpansion", fontsize=6.5, ha="right", va="center")
    ax.set_ylim(0, 0.65)
    ax.set_yticks([0.2, 0.4, 0.6])
    ax.set_rlabel_position(90)
    ax.tick_params(axis="y", labelsize=5.5, pad=0)
    ax.tick_params(axis="x", pad=5)
    ax.set_title(fam, fontsize=9, pad=14)
fig.tight_layout()
for out in ("work/basis/radar3.svg", "work/basis/radar3.pdf"):
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)
EOF
