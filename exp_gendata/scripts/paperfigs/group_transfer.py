import glob
import json

import numpy as np

TAGS = {"Qwen3-8B": "Qwen3-8B", "Llama-3.1-8B": "Meta-Llama-3.1-8B-Instruct", "gemma-2-9b": "gemma-2-9b-it"}
out = {}
for label, tag in TAGS.items():
    files = sorted(glob.glob(f"work/convcurve3-{tag}/pair-*.npz"))
    acc = {k: [] for k in ("loo_language", "loo_genre", "loo_generator")}
    for f in ("language", "genre", "generator"):
        for k in ("same", "other", "gap", "n", "dircos"):
            acc[f"{k}_{f}"] = []
    for path in files:
        z = np.load(path)
        acc["loo_language"].append(z["lang_all"][25:30].ravel())
        acc["loo_genre"].append(z["xfer_genre"])
        acc["loo_generator"].append(z["xfer_generator"])
        for f in ("language", "genre", "generator"):
            s, o = z[f"same_{f}"], z[f"other_{f}"]
            acc[f"same_{f}"].append(s)
            acc[f"other_{f}"].append(o)
            acc[f"gap_{f}"].append(s - o)
            acc[f"n_{f}"].append(z[f"n_{f}"])
            acc[f"dircos_{f}"].append(z[f"dircos_{f}"])
    res = {"n_pairs": len(files), "layer": int(np.load(files[0])["layer"])}
    for k, v in acc.items():
        x = np.concatenate([np.ravel(a) for a in v]).astype(float)
        x = x[~np.isnan(x)]
        res[k] = {
            "median": round(float(np.median(x)), 4),
            "p5": round(float(np.percentile(x, 5)), 4),
            "mean": round(float(x.mean()), 4),
            "cells": int(len(x)),
        }
    out[label] = res
print(json.dumps(out, indent=1))
open("work/tmp/paperfigs/group-transfer.json", "w").write(json.dumps(out, indent=1))
