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
REPEATS = 8
TOL = 0.005


def cell_curve(reps: np.ndarray, y: np.ndarray, inc: np.ndarray, rng: np.random.Generator) -> dict[int, float]:
    pole, antg = (y == 1) & inc, (y == 0) & inc
    ip, ia = np.where(pole)[0], np.where(antg)[0]
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
        aucs = [
            auroc(
                yT,
                reps[T]
                @ (reps[rng.choice(trp, n, replace=False)].mean(0) - reps[rng.choice(tra, n, replace=False)].mean(0)),
            )
            for _ in range(REPEATS)
        ]
        row[n] = float(np.mean(aucs))
    return row


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--tag", default="Qwen3-8B")
    args = parser.parse_args()
    files = sorted(glob.glob(f"work/oodreps-{args.tag}/pair-*.npz"))
    print(f"{args.tag}: {len(files)} concept rep files", flush=True)
    rng = np.random.default_rng(0)
    percpt: dict[str, list[dict]] = {f: [] for f in FIELDS}
    for i, f in enumerate(files):
        z = np.load(f)
        reps = z["reps"].astype(np.float32)
        y = z["y"]
        pid = int(Path(f).stem.split("-")[1])
        for fld in FIELDS:
            if fld not in z:
                continue
            cells = [cell_curve(reps, y, z[fld] == int(g), rng) for g in np.unique(z[fld])]
            merged: dict[int, list[float]] = {}
            for c in cells:
                for n, a in c.items():
                    merged.setdefault(n, []).append(a)
            if merged:
                percpt[fld].append({"pid": pid, "curve": {n: float(np.mean(v)) for n, v in merged.items()}})
        if i % 100 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    summary = {}
    concepts_out: dict[str, list[dict]] = {}
    for fld in FIELDS:
        entries = percpt[fld]
        if not entries:
            continue
        curves = [e["curve"] for e in entries]
        band = []
        for n in GRID:
            vals = np.array([c[n] for c in curves if n in c])
            if len(vals) < 20:
                continue
            band.append(
                {
                    "N": n,
                    "median": float(np.median(vals)),
                    "p2_5": float(np.percentile(vals, 2.5)),
                    "p97_5": float(np.percentile(vals, 97.5)),
                    "n_concepts": len(vals),
                }
            )
        plateau_n = band[-1]["N"]
        per_concept = []
        for e in entries:
            c = e["curve"]
            if plateau_n not in c:
                continue
            pl = c[plateau_n]
            ns = next((n for n in GRID if n in c and c[n] >= pl - TOL), plateau_n)
            per_concept.append(
                {
                    "pid": e["pid"],
                    "nstar": int(ns),
                    "plateau": float(pl),
                    "curve": {str(n): round(a, 4) for n, a in c.items()},
                }
            )
        concepts_out[fld] = per_concept
        nstars = np.array([p["nstar"] for p in per_concept])
        summary[fld] = {
            "band": band,
            "plateau_auroc": band[-1]["median"],
            "nstar": {
                "median": int(np.median(nstars)),
                "p90": int(np.percentile(nstars, 90)),
                "p10": int(np.percentile(nstars, 10)),
                "n": len(nstars),
            },
        }
        print(f"\n=== {fld}: AUROC vs N (median [95% band], {len(curves)} concepts) ===")
        for b in band:
            print(f"   N={b['N']:3d}  {b['median']:.3f}  [{b['p2_5']:.3f}, {b['p97_5']:.3f}]")
        ns = summary[fld]["nstar"]
        print(f"   per-concept N*: median {ns['median']}, p10 {ns['p10']}, p90 {ns['p90']}")
    out = Path("work/oodcurve")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"oodcurve-{args.tag}.json").write_text(json.dumps(summary))
    (out / f"oodcurve-{args.tag}-percpt.json").write_text(json.dumps(concepts_out))
    print(f"\nwrote {out}/oodcurve-{args.tag}.json (+ -percpt.json: per-pair trajectories + N*)")


if __name__ == "__main__":
    main()
