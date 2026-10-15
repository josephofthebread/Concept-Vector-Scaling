from __future__ import annotations

import glob
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from lexbias import auroc

FIELDS = ["lang", "genre", "generator"]
GRID = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160]
TEST_PER_CLASS = 15
REPEATS = 5
TOL = 0.005
N_SMALL, N_LARGE = 8, 64


def cell_curve(reps: np.ndarray, y: np.ndarray, inc: np.ndarray, rng: np.random.Generator) -> dict[int, float]:
    ip, ia = np.where((y == 1) & inc)[0], np.where((y == 0) & inc)[0]
    if len(ip) < TEST_PER_CLASS + GRID[0] or len(ia) < TEST_PER_CLASS + GRID[0]:
        return {}
    ip, ia = rng.permutation(ip), rng.permutation(ia)
    T = np.concatenate([ip[:TEST_PER_CLASS], ia[:TEST_PER_CLASS]])
    yT = np.concatenate([np.ones(TEST_PER_CLASS), np.zeros(TEST_PER_CLASS)])
    trp, tra = ip[TEST_PER_CLASS:], ia[TEST_PER_CLASS:]
    row: dict[int, float] = {}
    for n in GRID:
        if n > len(trp) or n > len(tra):
            break
        row[n] = float(
            np.mean(
                [
                    auroc(
                        yT,
                        reps[T]
                        @ (
                            reps[rng.choice(trp, n, replace=False)].mean(0)
                            - reps[rng.choice(tra, n, replace=False)].mean(0)
                        ),
                    )
                    for _ in range(REPEATS)
                ]
            )
        )
    return row


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = a.argsort().argsort() * 1.0, b.argsort().argsort() * 1.0
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--tag", default="Qwen3-8B")
    args = parser.parse_args()
    files = sorted(glob.glob(f"work/oodrepsb-{args.tag}/pair-*.npz"))
    print(f"{args.tag}: {len(files)} band-rep files", flush=True)
    rng = np.random.default_rng(0)
    layers = list(np.load(files[0])["layers"])
    percpt: dict[str, list[list[dict]]] = {f: [[] for _ in layers] for f in FIELDS}
    for i, f in enumerate(files):
        z = np.load(f)
        reps_all = z["reps"].astype(np.float32)
        y = z["y"]
        pid = int(Path(f).stem.split("-")[1])
        for li in range(len(layers)):
            reps = reps_all[:, li, :]
            for fld in FIELDS:
                if fld not in z:
                    continue
                merged: dict[int, list[float]] = {}
                for g in np.unique(z[fld]):
                    for n, a in cell_curve(reps, y, z[fld] == int(g), rng).items():
                        merged.setdefault(n, []).append(a)
                if merged:
                    percpt[fld][li].append({"pid": pid, "curve": {n: float(np.mean(v)) for n, v in merged.items()}})
        if i % 100 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    summary: dict = {"tag": args.tag, "layers": [int(x) for x in layers]}
    concepts_out: dict = {}
    for fld in FIELDS:
        per_layer = []
        for li, lname in enumerate(layers):
            curves = [e["curve"] for e in percpt[fld][li]]
            if not curves:
                continue
            b = []
            for n in GRID:
                vals = np.array([c[n] for c in curves if n in c])
                if len(vals) >= 20:
                    b.append(
                        {
                            "N": n,
                            "median": float(np.median(vals)),
                            "p2_5": float(np.percentile(vals, 2.5)),
                            "p97_5": float(np.percentile(vals, 97.5)),
                            "n_concepts": int(len(vals)),
                        }
                    )
            per_layer.append({"layer": int(lname), "li": li, "plateau": b[-1]["median"] if b else 0.0, "band": b})
        if not per_layer:
            continue
        best = max(per_layer, key=lambda x: x["plateau"])
        bi = best["li"]
        entries = percpt[fld][bi]
        plateau_n = best["band"][-1]["N"]
        pc = []
        for e in entries:
            c = e["curve"]
            if plateau_n not in c:
                continue
            pl = c[plateau_n]
            ns = next((n for n in GRID if n in c and c[n] >= pl - TOL), plateau_n)
            pc.append(
                {
                    "pid": e["pid"],
                    "nstar": int(ns),
                    "plateau": round(float(pl), 4),
                    "curve": {str(n): round(a, 4) for n, a in c.items()},
                }
            )
        both = [(p["curve"].get(str(N_SMALL)), p["curve"].get(str(N_LARGE))) for p in pc]
        both = [(a, b) for a, b in both if a is not None and b is not None]
        cross = spearman(np.array([a for a, _ in both]), np.array([b for _, b in both])) if both else float("nan")
        ns = np.array([p["nstar"] for p in pc])
        summary[fld] = {
            "best_layer": int(best["layer"]),
            "best_band": best["band"],
            "plateau_auroc": best["band"][-1]["median"],
            "nstar": {
                "median": int(np.median(ns)),
                "p10": int(np.percentile(ns, 10)),
                "p90": int(np.percentile(ns, 90)),
                "n": len(ns),
            },
            "crossing_spearman_N%d_vs_N%d" % (N_SMALL, N_LARGE): round(cross, 3),
            "per_layer_plateau": {int(pl["layer"]): round(pl["plateau"], 4) for pl in per_layer},
        }
        concepts_out[fld] = pc
        print(f"\n=== {fld}: best layer {best['layer']} (plateau {best['band'][-1]['median']:.3f}) ===")
        print("   per-layer plateau AUROC:", {k: round(v, 3) for k, v in summary[fld]["per_layer_plateau"].items()})
        print(
            f"   N* median {summary[fld]['nstar']['median']} (p10 {summary[fld]['nstar']['p10']}, "
            f"p90 {summary[fld]['nstar']['p90']}); crossing Spearman(N{N_SMALL},N{N_LARGE})={cross:.3f}"
        )

    out = Path("work/oodcurve")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"oodcurveb-{args.tag}.json").write_text(json.dumps(summary))
    (out / f"oodcurveb-{args.tag}-percpt.json").write_text(json.dumps(concepts_out))
    print(f"\nwrote oodcurveb-{args.tag}.json (+ -percpt.json)")


if __name__ == "__main__":
    main()
