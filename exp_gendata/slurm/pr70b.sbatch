#!/bin/bash
#SBATCH --job-name=gendata-pr70b
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/pr70b-%j.out
#SBATCH --error=logs/pr70b-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun .venv/bin/python - <<'EOF'
import json
import os
import numpy as np
from pathlib import Path
from exp_gendata.artifacts import atomic_safetensors, canonical_json, read_capture

MANDARIN = 2
SRC = Path(os.environ.get("SRC", "work/captures/unsloth_Meta-Llama-3.1-70B--raw--semantic"))
REF = Path("work/captures/unsloth_Meta-Llama-3.1-8B--raw--semantic/shard-000.safetensors")
DEST = Path(os.environ.get("DEST", "work/vectors-sb/unsloth_Meta-Llama-3.1-70B--raw--semantic"))
PAIRS = os.environ.get("PAIRS", "")

# The extraction shards rows, not concepts, so any subset of finished shards still covers every
# concept — it just gives each concept vector proportionally fewer stories.
paths = sorted(SRC.glob("shard-*.safetensors"))
if not paths:
    raise SystemExit(f"no shards in {SRC}")
totals = {}
for path in paths:
    _, tensors = read_capture(path)
    for name, t in tensors.items():
        acc = t.astype(np.float64 if np.issubdtype(t.dtype, np.floating) else np.int64)
        totals[name] = acc if name not in totals else totals[name] + acc
    print(f"merged {path.name}", flush=True)

lang_sum, lang_ct = totals["language_sum"], totals["language_count"].astype(np.float64)
L, P, _, nlang, H = lang_sum.shape
keep = [i for i in range(nlang) if i != MANDARIN]
stories = float(lang_ct[:, :, keep].sum(2).mean())
print(f"{len(paths)} shard(s), {P} concepts, {L} layers, hidden {H}, "
      f"{stories:.1f} non-Mandarin stories per pole on average", flush=True)

pole = lang_sum[:, :, :, keep].sum(3) / np.maximum(lang_ct[:, :, keep].sum(2), 1)[None, ..., None]
diff = (pole[:, :, 0] - pole[:, :, 1]).astype(np.float32)
sb = (diff - diff.mean(axis=1, keepdims=True)).astype(np.float32)
meta = {"manifest": canonical_json({
    "model": "unsloth/Meta-Llama-3.1-70B", "presentation": "raw",
    "pair_subset": PAIRS or "all", "shards_merged": [p.name for p in paths],
    "stories_per_pole": stories, "language_excluded": "Mandarin Chinese",
    "postprocessing": "across-concept mean subtracted before normalization",
    "axes": {"diff": ["layer", "pair", "hidden"]}})}
atomic_safetensors({"diff": sb}, DEST / "diff.safetensors", meta)
atomic_safetensors({"diff": diff.astype(np.float32)}, DEST / "diff_uncentered.safetensors", meta)
(DEST / "manifest.json").write_text(meta["manifest"])
print(f"wrote {DEST}", flush=True)


def pr(x):
    lam = np.linalg.svd(x, compute_uv=False) ** 2
    cum = np.cumsum(lam) / lam.sum()
    return (float(lam.sum() ** 2 / (lam ** 2).sum()), int(np.searchsorted(cum, .5) + 1),
            int(np.searchsorted(cum, .9) + 1), float(lam[0] / lam.sum()))


def spectrum(V, l, idx=None):
    v = V[l] if idx is None else V[l][idx]
    u = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    return pr(u - u.mean(0)), pr(u)


DEPTHS = (0.0, 0.2, 0.35, 0.5, 0.55, 0.7, 0.85, 1.0)
rng = np.random.default_rng(0)
header = (f"{'depth':>6s} {'layer':>5s} {'PR centered':>11s} {'n50':>4s} {'n90':>4s} | "
          f"{'PR raw':>7s} {'PC1%':>6s} | {'isotropic':>10s}")

print("\n=== Llama-3.1-70B, raw, no Mandarin ===")
print(header)
pr70 = {}
for frac in DEPTHS:
    l = int(round(frac * (L - 1)))
    c, r = spectrum(diff, l)
    g = rng.standard_normal((P, H)).astype(np.float32)
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    pr70[frac] = c[0]
    print(f"{frac:6.2f} L{l:<4d} {c[0]:11.1f} {c[1]:4d} {c[2]:4d} | "
          f"{r[0]:7.1f} {r[3]*100:5.1f}% | {pr(g - g.mean(0))[0]:10.1f}")

# 8B reference, built the same way. Storage does not cross language cells with buckets, so when the
# 70B run used a bucket subset the reference is reported both ways and the truth is bracketed.
_, ref = read_capture(REF)
b8, b8c = ref["sums"], ref["counts"].astype(np.float64)
l8, l8c = ref["language_sum"], ref["language_count"].astype(np.float64)
k8 = [i for i in range(l8c.shape[2]) if i != MANDARIN]
p8 = l8[:, :, :, k8].sum(3) / np.maximum(l8c[:, :, k8].sum(2), 1)[None, ..., None]
V8_nomand = (p8[:, :, 0] - p8[:, :, 1]).astype(np.float32)
p8b = b8.sum(3) / np.maximum(b8c.sum(2), 1)[None, ..., None]
V8_all = (p8b[:, :, 0] - p8b[:, :, 1]).astype(np.float32)
L8 = V8_all.shape[0]
idx = np.array(sorted(json.loads(Path(PAIRS).read_text()))) if PAIRS else None

print("\n=== Meta-Llama-3.1-8B reference, same concepts, same composition ===")
for label, V8 in (("all buckets / no Mandarin", V8_nomand), ("all buckets / all languages", V8_all)):
    print(f"-- 8B {label} --")
    print(header[:header.index('|')])
    for frac in DEPTHS:
        l = int(round(frac * (L8 - 1)))
        c, r = spectrum(V8, l, idx)
        print(f"{frac:6.2f} L{l:<4d} {c[0]:11.1f} {c[1]:4d} {c[2]:4d} | {r[0]:7.1f} {r[3]*100:5.1f}%")

print("\n=== ratio at matched fractional depth (70B PR / 8B PR, no-Mandarin reference) ===")
for frac in DEPTHS:
    c8, _ = spectrum(V8_nomand, int(round(frac * (L8 - 1))), idx)
    print(f"{frac:6.2f}  70B {pr70[frac]:8.1f}   8B {c8[0]:8.1f}   ratio {pr70[frac] / c8[0]:5.2f}")
print("\ndone", flush=True)
EOF
