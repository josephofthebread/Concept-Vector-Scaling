from __future__ import annotations

import glob
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from lexbias import auroc

FIELDS = ["lang", "genre", "generator"]
TEST_FRAC = 0.25
MIN_N = 8
MIN_TEST = 6
REPEATS = 5


def cell_gap(reps: np.ndarray, y: np.ndarray, group: np.ndarray, g: int, rng: np.random.Generator) -> dict | None:
    pole, antg = y == 1, y == 0
    inc = group == g
    test = np.zeros(len(y), bool)
    for cls in (pole, antg):
        idx = np.where(inc & cls)[0]
        if len(idx) < MIN_TEST + MIN_N:
            return None
        n_te = max(MIN_TEST, int(round(TEST_FRAC * len(idx))))
        test[rng.permutation(idx)[:n_te]] = True
    T = np.where(test)[0]
    yT = y[T]
    in_p = np.where(inc & pole & ~test)[0]
    in_a = np.where(inc & antg & ~test)[0]
    out_p = np.where(~inc & pole)[0]
    out_a = np.where(~inc & antg)[0]
    N_p = min(len(in_p), len(out_p))
    N_a = min(len(in_a), len(out_a))
    if N_p < MIN_N or N_a < MIN_N:
        return None
    aid, aood = [], []
    for _ in range(REPEATS):

        def vec(pp: np.ndarray, ap: np.ndarray) -> np.ndarray:
            p = reps[rng.choice(pp, N_p, replace=False)].mean(0)
            a = reps[rng.choice(ap, N_a, replace=False)].mean(0)
            return p - a

        aid.append(auroc(yT, reps[T] @ vec(in_p, in_a)))
        aood.append(auroc(yT, reps[T] @ vec(out_p, out_a)))
    return {
        "cell": int(g),
        "N": int(N_p + N_a),
        "n_test": int(len(T)),
        "id": float(np.mean(aid)),
        "ood": float(np.mean(aood)),
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--tag", default="Qwen3-8B")
    args = parser.parse_args()
    files = sorted(glob.glob(f"work/oodreps-{args.tag}/pair-*.npz"))
    print(f"{len(files)} concept rep files", flush=True)
    rng = np.random.default_rng(0)
    res: dict[str, list[dict]] = {f: [] for f in FIELDS}
    for i, f in enumerate(files):
        z = np.load(f)
        reps = z["reps"].astype(np.float32)
        y = z["y"]
        pid = int(Path(f).stem.split("-")[1])
        for fld in FIELDS:
            if fld not in z:
                continue
            for g in np.unique(z[fld]):
                r = cell_gap(reps, y, z[fld], int(g), rng)
                if r:
                    res[fld].append({"pair_id": pid, **r})
        if i % 100 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    summary = {}
    for fld in FIELDS:
        rows = res[fld]
        if not rows:
            continue
        idd = np.array([r["id"] for r in rows])
        ood = np.array([r["ood"] for r in rows])
        gap = idd - ood
        summary[fld] = {
            "n": len(rows),
            "id_median": float(np.median(idd)),
            "ood_median": float(np.median(ood)),
            "gap_median": float(np.median(gap)),
            "gap_mean": float(gap.mean()),
        }
        print(f"\n=== {fld} (matched N, common test; {len(rows)} cells) ===")
        print(f"  ID  AUC median {np.median(idd):.3f}   OOD AUC median {np.median(ood):.3f}")
        print(
            f"  generalization penalty (ID-OOD): median {np.median(gap):+.3f}  mean {gap.mean():+.3f}  "
            f"(|gap|<0.02 in {np.mean(np.abs(gap) < 0.02):.0%})"
        )
    out = Path("work/oodmatch")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"oodmatch-{args.tag}.json").write_text(json.dumps({"summary": summary, "cells": res}))
    print(f"\nwrote {out}/oodmatch-{args.tag}.json")


if __name__ == "__main__":
    main()
