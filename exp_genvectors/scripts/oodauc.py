from __future__ import annotations

import glob
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from lexbias import auroc

MIN_POLE = 8
FOLDS = 5


def summarize(name: str, results: list[dict]) -> dict:
    ood = np.array([r["ood"] for r in results if r["ood"] is not None], float)
    idd = np.array([r["id"] for r in results if r["id"] is not None], float)
    both = [(r["ood"], r["id"]) for r in results if r["ood"] is not None and r["id"] is not None]
    gap = np.array([i - o for o, i in both], float)
    print(f"\n=== {name}: {len(results)} held-out cells ===")
    print(f"  OOD AUC: median {np.median(ood):.3f}  mean {ood.mean():.3f}")
    print(f"  ID  AUC: median {np.median(idd):.3f}  mean {idd.mean():.3f}")
    print(
        f"  ID-OOD gap: median {np.median(gap):.3f}  mean {gap.mean():.3f}  "
        f"(OOD within 0.02 of ID: {np.mean(np.abs(gap) < 0.02):.0%})"
    )
    per: dict = {}
    for g in sorted({r["group"] for r in results}):
        go = [r["ood"] for r in results if r["group"] == g and r["ood"] is not None]
        gi = [r["id"] for r in results if r["group"] == g and r["id"] is not None]
        per[g] = {"ood": round(float(np.median(go)), 3), "id": round(float(np.median(gi)), 3), "n": len(go)}
    return {
        "ood_median": float(np.median(ood)),
        "id_median": float(np.median(idd)),
        "gap_median": float(np.median(gap)),
        "per_group": per,
        "n": len(results),
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--tag", default="Qwen3-8B")
    args = parser.parse_args()
    files = sorted(glob.glob(f"work/oodreps-{args.tag}/pair-*.npz"))
    print(f"{len(files)} concept rep files", flush=True)
    rng = np.random.default_rng(0)
    fields = ["lang", "genre", "generator"]
    res: dict[str, list[dict]] = {fld: [] for fld in fields}
    for i, f in enumerate(files):
        z = np.load(f)
        reps = z["reps"].astype(np.float32)
        y = z["y"]
        pid = int(Path(f).stem.split("-")[1])
        for fld in fields:
            if fld not in z:
                continue
            group = z[fld]
            cells = []
            pole, antg = y == 1, y == 0
            for g in np.unique(group):
                te = group == g
                if int((te & pole).sum()) < MIN_POLE or int((te & antg).sum()) < MIN_POLE:
                    continue
                rec: dict = {"group": int(g), "n_pole": int((te & pole).sum()), "ood": None, "id": None}
                tr = ~te
                if (tr & pole).any() and (tr & antg).any():
                    vec = reps[tr & pole].mean(0) - reps[tr & antg].mean(0)
                    rec["ood"] = auroc(y[te], reps[te] @ vec)
                idx = np.where(te)[0]
                fold = rng.integers(0, FOLDS, len(idx))
                s_id = np.zeros(len(idx), np.float32)
                ok = True
                for fold_id in range(FOLDS):
                    trm, tem = idx[fold != fold_id], idx[fold == fold_id]
                    tp, ta = trm[y[trm] == 1], trm[y[trm] == 0]
                    if not len(tp) or not len(ta) or not len(tem):
                        ok = tem.size == 0
                        continue
                    vec = reps[tp].mean(0) - reps[ta].mean(0)
                    s_id[fold == fold_id] = reps[tem] @ vec
                if ok:
                    rec["id"] = auroc(y[idx], s_id)
                cells.append(rec)
            for r in cells:
                res[fld].append({"pair_id": pid, **r})
        if i % 100 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    summary = {fld: summarize(f"{fld.upper()} leave-one-out", res[fld]) for fld in fields if res[fld]}
    out = Path("work/oodauc")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"oodauc-{args.tag}.json").write_text(
        json.dumps({"tag": args.tag, "summary": summary, "cells": res}, ensure_ascii=False)
    )
    print(f"\nwrote {out}/oodauc-{args.tag}.json")


if __name__ == "__main__":
    main()
