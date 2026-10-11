from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

TAGS = {"Qwen3-8B": 22, "gemma-2-9b-it": 22, "Meta-Llama-3.1-8B-Instruct": 14}
WORK = Path("work")
OUT = WORK / "convcurve3-agg"
GRID_KEYS = ("auc_grid", "conv_grid")
META_KEYS = ("layer", "layers", "n_rows", "dtype", "gpu")
N_LANG = 5
LANG_COMBOS = [c for k in range(1, N_LANG) for c in combinations(range(N_LANG), k)]


def load(dirs: list[Path], keep_layers: bool) -> tuple[list[int], dict[str, np.ndarray], dict]:
    """Union over dirs (first dir wins on duplicates). keep_layers=False -> best-layer slice of all-layer files."""
    records: dict[int, dict[str, np.ndarray]] = {}
    grids: dict = {}
    provenance = {"fp16": 0, "bf16": 0, "unknown": 0}
    for d in dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("pair-*.npz")):
            pid = int(f.stem.split("-")[1])
            if pid in records:
                continue
            z = np.load(f)
            all_layer = "layers" in z.files
            layer = int(z["layer"])
            rec = {}
            for k in [k for k in z.files if k not in GRID_KEYS + META_KEYS and not k.startswith("n_")]:
                v = z[k]
                rec[k] = v[layer] if (all_layer and not keep_layers) else v
            for k in (x for x in z.files if x.startswith("n_")):
                rec[k] = z[k]
            records[pid] = rec
            if not grids:
                grids = {k: z[k].tolist() for k in GRID_KEYS}
                grids["best_layer"] = layer
                if all_layer and keep_layers:
                    grids["n_layers"] = int(z["layers"].shape[0])
            dt = str(z["dtype"]) if "dtype" in z.files else "unknown"
            provenance["fp16" if "16" in dt and "bfloat" not in dt else "bf16" if "bfloat" in dt else "unknown"] += 1
    pids = sorted(records)
    if not pids:
        return [], {}, grids
    keys = [
        k
        for k in records[pids[0]]
        if all(k in records[p] and records[p][k].shape == records[pids[0]][k].shape for p in pids)
    ]
    stacked = {k: np.stack([records[p][k] for p in pids]) for k in keys}
    grids["provenance"] = provenance
    return pids, stacked, grids


def summarize(stacked: dict, grids: dict, layered: bool) -> dict:
    s: dict = {"n_pairs": int(next(iter(stacked.values())).shape[0]), **{k: grids[k] for k in grids}}
    if layered:
        ag, cg = grids["auc_grid"], grids["conv_grid"]
        pick = lambda grid, n: grid.index(n) if n in grid else len(grid) - 1
        s["per_layer_median"] = {
            "auroc_u@8": np.nanmedian(stacked["auroc_u"][:, :, pick(ag, 8)], 0).round(4).tolist(),
            "auroc_u@128": np.nanmedian(stacked["auroc_u"][:, :, pick(ag, 128)], 0).round(4).tolist(),
            "conv_u@128": np.nanmedian(stacked["conv_u"][:, :, pick(cg, 128)], 0).round(4).tolist(),
            "conv_m@256": np.nanmedian(stacked["conv_m"][:, :, pick(cg, 256)], 0).round(4).tolist(),
            "conv_m@500": np.nanmedian(stacked["conv_m"][:, :, -1], 0).round(4).tolist(),
            "xfer_genre": np.nanmedian(np.nanmean(stacked["xfer_genre"], -1), 0).round(4).tolist(),
            "dircos_language": np.nanmedian(np.nanmean(stacked["dircos_language"], -1), 0).round(4).tolist(),
        }
        return s
    for k in ("auroc_u", "conv_u", "nested_u", "auroc_m", "conv_m"):
        if k in stacked:
            s[k] = {
                "median": np.nanmedian(stacked[k], 0).round(4).tolist(),
                "p5": np.nanpercentile(stacked[k], 5, 0).round(4).tolist(),
                "p95": np.nanpercentile(stacked[k], 95, 0).round(4).tolist(),
            }
    if "lang_all" in stacked:
        by_k = {}
        for kk in range(1, N_LANG):
            idx = [i for i, c in enumerate(LANG_COMBOS) if len(c) == kk]
            for name in ("lang_all", "lang_size"):
                per_pair = np.nanmean(stacked[name][:, idx, :].reshape(len(stacked[name]), -1), 1)
                by_k.setdefault(name, {})[f"K={kk}"] = round(float(np.nanmedian(per_pair)), 4)
        s["language_transfer_median"] = by_k
    for k in ("xfer_genre", "xfer_generator"):
        if k in stacked:
            s[k + "_median"] = round(float(np.nanmedian(np.nanmean(stacked[k], 1))), 4)
    for field in ("language", "genre", "generator"):
        if f"same_{field}" in stacked:
            s[f"{field}_same_vs_other_median"] = {
                "same": round(float(np.nanmedian(np.nanmean(stacked[f"same_{field}"], 1))), 4),
                "other": round(float(np.nanmedian(np.nanmean(stacked[f"other_{field}"], 1))), 4),
                "dircos": round(float(np.nanmedian(np.nanmean(stacked[f"dircos_{field}"], 1))), 4),
            }
    return s


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict = {}
    for tag in TAGS:
        datasets = {
            "assistant": ([WORK / f"convcurve3-{tag}"], False),
            "raw": ([WORK / f"convcurve3-{tag}-raw"], False),
            "user": ([WORK / f"convcurve3-{tag}-user", WORK / f"convcurve3L-{tag}-user"], False),
            "paired_assistant": (
                [WORK / f"convcurve3-{tag}-paired_assistant", WORK / f"convcurve3L-{tag}-paired_assistant"],
                False,
            ),
            "assistant_layers": ([WORK / f"convcurve3L-{tag}"], True),
        }
        summary[tag] = {}
        per_concept = {}
        for name, (dirs, layered) in datasets.items():
            pids, stacked, grids = load(dirs, keep_layers=layered)
            if not pids:
                print(f"{tag} {name}: no files")
                continue
            np.savez_compressed(
                OUT / f"{tag}--{name}.npz",
                pair_ids=np.array(pids),
                **stacked,
                **{k: np.array(v) for k, v in grids.items() if k in GRID_KEYS},
            )
            summary[tag][name] = summarize(stacked, grids, layered)
            print(f"{tag} {name}: {len(pids)} pairs, {len(stacked)} fields, provenance {grids.get('provenance')}")
            if not layered:
                ag, _ = grids["auc_grid"], grids["conv_grid"]
                per_concept[name] = {
                    "pids": pids,
                    "auroc_u@8": stacked["auroc_u"][:, ag.index(8)],
                    "conv_m@500": stacked["conv_m"][:, -1],
                }
        cross = {}
        names = [n for n in ("assistant", "user", "paired_assistant", "raw") if n in per_concept]
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                common = sorted(set(per_concept[a]["pids"]) & set(per_concept[b]["pids"]))
                ia = [per_concept[a]["pids"].index(p) for p in common]
                ib = [per_concept[b]["pids"].index(p) for p in common]
                cross[f"{a}~{b}"] = {
                    "n": len(common),
                    **{
                        m: round(float(spearmanr(per_concept[a][m][ia], per_concept[b][m][ib]).statistic), 4)
                        for m in ("auroc_u@8", "conv_m@500")
                    },
                }
        summary[tag]["cross_format_spearman"] = cross
        print(f"{tag} cross-format: {json.dumps(cross)}")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {OUT}/summary.json and per-dataset npz files")


if __name__ == "__main__":
    main()
