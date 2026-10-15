from __future__ import annotations

import glob
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from safetensors import safe_open

STEM = {
    "Qwen3-8B": "Qwen_Qwen3-8B",
    "Meta-Llama-3.1-8B-Instruct": "unsloth_Meta-Llama-3.1-8B-Instruct",
    "gemma-2-9b-it": "google_gemma-2-9b-it",
}
GRID = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160]
TEST_PER_CLASS = 15
REPEATS = 5
FIELD = "lang"


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = a.argsort().argsort() * 1.0, b.argsort().argsort() * 1.0
    return float(np.corrcoef(ra, rb)[0, 1])


def logdev(x: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(1.0 - np.minimum(x, 0.9999), 1e-6, None))


PEARSON_N = [8, 16, 32]


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--tag", default="Qwen3-8B")
    args = parser.parse_args()
    cb = json.loads(Path(f"work/oodcurve/oodcurveb-{args.tag}.json").read_text())
    best_layer = cb[FIELD]["best_layer"]
    li = cb["layers"].index(best_layer)
    pc = json.loads(Path(f"work/oodcurve/oodcurveb-{args.tag}-percpt.json").read_text())[FIELD]
    nstar = {p["pid"]: p["nstar"] for p in pc}

    conv_path = Path(f"work/convergence/{STEM[args.tag]}--assistant--semantic.safetensors")
    out: dict = {"tag": args.tag, "best_layer": best_layer}
    if conv_path.exists():
        with safe_open(str(conv_path), framework="np") as f:
            qq = f.get_slice("q_q")[best_layer]
        conv = qq.mean(0)
        pids = [p for p in nstar if p < len(conv)]
        ns = np.array([nstar[p] for p in pids], float)
        cv = np.array([conv[p] for p in pids], float)
        conv_ld = logdev(cv)
        out["conv"] = {
            "n": len(pids),
            "spearman_nstar_vs_conv_cos": round(spearman(ns, cv), 3),
            "conv_cos_median": round(float(np.median(cv)), 4),
            "pearson_at_N": {},
        }
        print(f"{args.tag} L{best_layer}: n={len(pids)}")
        print(f"  [robustness] Spearman(N*, convergence cos) = {out['conv']['spearman_nstar_vs_conv_cos']:+.3f}")
        curve_of = {p["pid"]: p["curve"] for p in pc}
        for N in PEARSON_N:
            xs, ys = [], []
            for j, p in enumerate(pids):
                a = curve_of.get(p, {}).get(str(N))
                if a is not None:
                    xs.append(conv_ld[j])
                    ys.append(a)
            if len(xs) >= 3:
                yl = logdev(np.array(ys))
                xr = np.array(xs)
                out["conv"]["pearson_at_N"][N] = {
                    "n": len(xs),
                    "pearson": round(float(np.corrcoef(xr, yl)[0, 1]) if len(xr) >= 3 else float("nan"), 3),
                    "spearman": round(spearman(xr, yl), 3),
                }
                print(
                    f"  Pearson(log10(1-cos), log10(1-AUROC@N={N:>2}))  = "
                    f"{out['conv']['pearson_at_N'][N]['pearson']:+.3f}  "
                    f"(Spearman {out['conv']['pearson_at_N'][N]['spearman']:+.3f}, n={len(xs)})"
                )
    else:
        print(f"  (no convergence tensor at {conv_path})")

    rng = np.random.default_rng(0)
    acc: dict[int, list[float]] = {}
    for f in sorted(glob.glob(f"work/oodrepsb-{args.tag}/pair-*.npz")):
        z = np.load(f)
        reps = z["reps"][:, li, :].astype(np.float32)
        y, lang = z["y"], z["lang"]
        for g in np.unique(lang):
            ip = np.where((y == 1) & (lang == g))[0]
            ia = np.where((y == 0) & (lang == g))[0]
            if len(ip) < TEST_PER_CLASS + GRID[0] or len(ia) < TEST_PER_CLASS + GRID[0]:
                continue
            vfull = reps[ip].mean(0) - reps[ia].mean(0)
            nf = np.linalg.norm(vfull)
            for n in GRID:
                if n > len(ip) or n > len(ia):
                    break
                cs = []
                for _ in range(REPEATS):
                    v = reps[rng.choice(ip, n, replace=False)].mean(0) - reps[rng.choice(ia, n, replace=False)].mean(0)
                    nv = np.linalg.norm(v)
                    cs.append(float(v @ vfull / (nv * nf)) if nv > 1e-9 and nf > 1e-9 else np.nan)
                acc.setdefault(n, []).append(float(np.nanmean(cs)))
    cos_curve = [{"N": n, "cos_median": round(float(np.median(acc[n])), 4)} for n in GRID if n in acc]
    out["cos_curve"] = cos_curve
    auroc_curve = {b["N"]: b["median"] for b in cb[FIELD]["best_band"]}
    print("\n  N   AUROC   cos(vec_N,vec_full)")
    for c in cos_curve:
        a = auroc_curve.get(c["N"])
        print(f"  {c['N']:>4}  {a if a is None else round(a, 3)}   {c['cos_median']:.3f}")
    Path(f"work/oodcurve/oodconv-{args.tag}.json").write_text(json.dumps(out))
    print(f"\nwrote work/oodcurve/oodconv-{args.tag}.json")


if __name__ == "__main__":
    main()
