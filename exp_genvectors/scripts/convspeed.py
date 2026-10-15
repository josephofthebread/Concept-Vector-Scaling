from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from safetensors import safe_open

LAM_GRID = (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4)
OUTER, SEED = 5, 0
NPERM = 200
TAIL_Q = 5.0
BEST_LAYER = {"Qwen_Qwen3-8B": 22, "google_gemma-2-9b-it": 22, "unsloth_Meta-Llama-3.1-8B-Instruct": 14}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUROC with tie-averaged ranks -> a constant predictor scores exactly 0.5."""
    from scipy.stats import rankdata

    npos, nneg = int(y.sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    ranks = rankdata(s)
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def oof_from_hats(hats: list[tuple], target: np.ndarray) -> np.ndarray:
    pred = np.empty(len(target))
    ymu = target.mean()
    for test_idx, train_idx, m in hats:
        pred[test_idx] = m @ (target[train_idx] - ymu) + ymu
    return pred


def evaluate(
    name: str, features: np.ndarray, speed: np.ndarray, q_lo: float, q_hi: float, rng: np.random.Generator
) -> dict:
    """Real OOF metrics plus a standard permutation null: permute the target, refit at the fixed lambda,
    score the OOF prediction against the PERMUTED target, with slow/fast tails recomputed from it."""
    n = len(speed)
    folds = np.array_split(np.random.default_rng(SEED).permutation(n), OUTER)
    best_lam, best, best_hats = LAM_GRID[0], -np.inf, None
    for lam in LAM_GRID:
        hats = []
        d = features.shape[1]
        for k in range(len(folds)):
            test_idx = folds[k]
            train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != k])
            train, other = features[train_idx], features[test_idx]
            mu, sd = train.mean(0), train.std(0)
            sd[sd == 0] = 1.0
            xt, xv = (train - mu) / sd, (other - mu) / sd
            m = xv @ np.linalg.solve(xt.T @ xt + lam * np.eye(d), xt.T)
            hats.append((test_idx, train_idx, m))
        s = spearman(oof_from_hats(hats, speed), speed)
        if s > best:
            best, best_lam, best_hats = s, lam, hats
    lam, hats = best_lam, best_hats
    pred = oof_from_hats(hats, speed)
    tail_mask = (speed <= q_lo) | (speed >= q_hi)
    rho, roc = spearman(pred, speed), auroc((speed >= q_hi)[tail_mask].astype(int), pred[tail_mask])
    null_rho, null_roc = np.empty(NPERM), np.empty(NPERM)
    for i in range(NPERM):
        target = rng.permutation(speed)
        p = oof_from_hats(hats, target)
        mask = (target <= q_lo) | (target >= q_hi)
        null_rho[i] = spearman(p, target)
        null_roc[i] = auroc((target >= q_hi)[mask].astype(int), p[mask])
    p_rho = (1 + np.sum(null_rho >= rho)) / (NPERM + 1)
    p_roc = (1 + np.sum(null_roc >= roc)) / (NPERM + 1)
    print(
        f"  {name:16s} dim={features.shape[1]:>4d} lam={lam:<6g}  "
        f"Spearman={rho:+.3f} (null {null_rho.mean():+.3f}, p95 {np.percentile(null_rho, 95):+.3f}, p={p_rho:.3f})  "
        f"AUROC={roc:.3f} (null {null_roc.mean():.3f}, p95 {np.percentile(null_roc, 95):.3f}, p={p_roc:.3f})"
    )
    return {
        "dim": int(features.shape[1]),
        "lambda": lam,
        "spearman_all": round(rho, 4),
        "spearman_null_mean": round(float(null_rho.mean()), 4),
        "spearman_null_p95": round(float(np.percentile(null_rho, 95)), 4),
        "spearman_p": round(float(p_rho), 4),
        "auroc_tails": round(roc, 4),
        "auroc_null_mean": round(float(null_roc.mean()), 4),
        "auroc_null_p95": round(float(np.percentile(null_roc, 95)), 4),
        "auroc_p": round(float(p_roc), 4),
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--tag", required=True, help="model/tag stem, e.g. Qwen_Qwen3-8B--assistant--semantic")
    parser.add_argument("--conv-root", type=Path, default=Path("work/convergence"))
    parser.add_argument("--vectors-root", type=Path, default=Path("work/vectors"))
    parser.add_argument("--lexical", type=Path, default=Path("work/convergence/lexical_features.parquet"))
    args = parser.parse_args()

    with safe_open(str(args.conv_root / f"{args.tag}.safetensors"), framework="np") as h:
        qq = h.get_tensor("q_q").astype(np.float64)
    layers = qq.shape[0]
    band = slice(round(layers / 3), round(2 * layers / 3) + 1)
    best = next(v for k, v in BEST_LAYER.items() if args.tag.startswith(k))
    targets = {"mid_third": qq[band].mean(axis=(0, 1)), f"best_layer_{best}": qq[best].mean(axis=0)}
    n = qq.shape[2]
    print(f"=== {args.tag} === (mid third = layers {band.start}-{band.stop - 1}; best layer {best})")

    with safe_open(str(args.vectors_root / args.tag / "diff.safetensors"), framework="np") as h:
        emb = h.get_tensor("diff")[0].astype(np.float64)
    assert emb.shape[0] == n, f"pair mismatch conv {n} vs vectors {emb.shape[0]}"
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)

    lex = pq.read_table(args.lexical).to_pydict()
    order = np.argsort(lex["pair_id"])
    syntax = np.stack(
        [
            np.asarray(lex[c], np.float64)[order]
            for c in ("vocab_jaccard", "unigram_cosine", "label_rate", "mean_tokens", "length_ratio")
        ],
        axis=1,
    )
    assert syntax.shape[0] == n, f"pair mismatch conv {n} vs lexical {syntax.shape[0]}"

    summary: dict = {
        "tag": args.tag,
        "n_concepts": n,
        "band": [band.start, band.stop - 1],
        "best_layer": best,
        "null": "standard: permuted target, refit at fixed lambda, scored against the permuted target",
        "targets": {},
    }
    for name, speed in targets.items():
        q_lo, q_hi = np.percentile(speed, TAIL_Q), np.percentile(speed, 100 - TAIL_Q)
        n_slow, n_fast = int((speed <= q_lo).sum()), int((speed >= q_hi).sum())
        print(
            f" [{name}] median {np.median(speed):.4f} [q5 {q_lo:.4f}, q95 {q_hi:.4f}]; tails {n_slow} slow / {n_fast} fast"
        )
        summary["targets"][name] = {
            "q5": round(float(q_lo), 4),
            "q95": round(float(q_hi), 4),
            "n_slow": n_slow,
            "n_fast": n_fast,
            "embedding_only": evaluate("embedding-only", emb, speed, q_lo, q_hi, np.random.default_rng(SEED + 1)),
            "syntax_only": evaluate("syntax-only", syntax, speed, q_lo, q_hi, np.random.default_rng(SEED + 2)),
        }
    out = Path("work/convspeed")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"convspeed-{args.tag}.json").write_text(json.dumps(summary, indent=2))
    print(f"  wrote work/convspeed/convspeed-{args.tag}.json")


if __name__ == "__main__":
    main()
