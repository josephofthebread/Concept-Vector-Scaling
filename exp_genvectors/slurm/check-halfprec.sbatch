#!/bin/bash
#SBATCH --job-name=genvectors-halfcheck
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/halfcheck-%j.out
#SBATCH --error=logs/halfcheck-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun .venv/bin/python - <<'EOF'
import numpy as np
import torch
from safetensors import safe_open

from exp_genvectors.vecmath import whiten, within_covariance

CASTS = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp8e4m3": torch.float8_e4m3fn,
}


def cast(x: np.ndarray, name: str) -> np.ndarray:
    return torch.from_numpy(np.ascontiguousarray(x)).to(CASTS[name]).float().numpy()


def cos_rows(a, b):
    num = np.einsum("...i,...i->...", a, b)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return (num / np.maximum(den, 1e-30)).ravel()


def show(label, truth, approx):
    c = cos_rows(truth, approx)
    print(f"    {label:34s} cos min={c.min():.6f} median={np.median(c):.6f}")


path = "work/captures/Qwen_Qwen3-8B-Base--raw--semantic/shard-000.safetensors"
with safe_open(path, framework="np") as handle:
    counts = handle.get_tensor("counts").astype(np.float64)          # [pair, side, bucket]
    genre_count = handle.get_tensor("genre_count").astype(np.float64)
    sums_slice = handle.get_slice("sums")
    moments_slice = handle.get_slice("moments")
    unit_slice = handle.get_slice("unit_diff_sum")
    genre_slice = handle.get_slice("genre_sum")

    for layer in (0, 18, 36):
        print(f"layer {layer}:")
        # --- pole sums -> diff vectors -------------------------------------------------
        sums = np.asarray(sums_slice[layer]).astype(np.float32)      # [pair, side, bucket, hidden]
        bucket_n = np.maximum(counts[..., None], 1).astype(np.float32)
        mu = sums.sum(axis=2) / counts.sum(axis=2)[..., None].astype(np.float32)
        diff = mu[:, 0] - mu[:, 1]
        for name in CASTS:
            restored = cast(sums / bucket_n, name) * bucket_n
            mu_c = restored.sum(axis=2) / counts.sum(axis=2)[..., None].astype(np.float32)
            show(f"diff from sums as {name} means", diff, mu_c[:, 0] - mu_c[:, 1])
        # --- unit difference sums -> R and drift direction -----------------------------
        unit = np.asarray(unit_slice[layer]).astype(np.float32)      # [pair, bucket, hidden]
        pair_n = counts[:, 0, :].astype(np.float32)                  # unit rows counted once per row
        r_truth = np.linalg.norm(unit.sum(axis=1), axis=-1)
        for name in CASTS:
            restored = cast(unit / np.maximum(pair_n[..., None], 1), name) * pair_n[..., None]
            r = np.linalg.norm(restored.sum(axis=1), axis=-1)
            err = np.abs(r - r_truth) / np.maximum(r_truth, 1e-9)
            show(f"drift direction as {name}", unit.sum(axis=1), restored.sum(axis=1))
            print(f"      R max relerr {name}: {err.max():.2e}")
        # --- genre cell sums -> nuisance directions ------------------------------------
        genre = np.asarray(genre_slice[layer]).astype(np.float32)    # [pair, side, genre, hidden]
        gmu = genre / np.maximum(genre_count[..., None], 1).astype(np.float32)
        truth_dir = (gmu - gmu.mean(axis=2, keepdims=True)).reshape(-1, genre.shape[-1])
        keep = np.linalg.norm(truth_dir, axis=1) > 0
        for name in CASTS:
            gmu_c = cast(gmu, name)
            cast_dir = (gmu_c - gmu_c.mean(axis=2, keepdims=True)).reshape(-1, genre.shape[-1])
            show(f"genre directions as {name}", truth_dir[keep], cast_dir[keep])
        # --- moments -> covariance and whitened vectors --------------------------------
        moments = np.asarray(moments_slice[layer])                   # [bucket, hidden, hidden] f64
        stories = counts.sum()
        group_sums = sums.astype(np.float64).sum(axis=2).reshape(-1, sums.shape[-1])
        group_counts = counts.sum(axis=2).reshape(-1)
        cov_truth = within_covariance(moments.sum(axis=0), group_sums, group_counts)
        white_truth = whiten(diff.astype(np.float64), cov_truth, 0.05)
        scaled = (moments / stories).astype(np.float32).astype(np.float64) * stories
        cov = within_covariance(scaled.sum(axis=0), group_sums, group_counts)
        show("whitened diff, moments as fp32", white_truth, whiten(diff.astype(np.float64), cov, 0.05))
EOF
