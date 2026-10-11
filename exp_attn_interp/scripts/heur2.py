import collections
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from exp_attn_interp.bstar import crossing_cum

unit = lambda x: x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
files = sorted(Path("work/vdecomp").glob("pair-*.safetensors"))
ML, MR, pairs = [], [], []
for f in files:
    t = load_file(str(f))
    A = (t["pole_head"][0] - t["pole_head"][1]).float().sum(1)
    M = (t["pole_mlp"][0] - t["pole_mlp"][1]).float()
    De = (t["pole_embed"][0] - t["pole_embed"][1]).float()
    B = M.shape[0]
    V = torch.cat([De[None], De[None] + torch.cumsum(A + M, 0)], 0)
    vl = unit(V[B - 1])
    vr = unit(V[:B])
    ML.append((M @ vl).numpy())
    MR.append((M * vr).sum(-1).numpy())
    pairs.append(int(f.stem.split("-")[1]))
ml = np.stack(ML)
mr = np.stack(MR)
bl = {pairs[i]: crossing_cum(ml[i]) for i in range(len(pairs))}
br = {pairs[i]: crossing_cum(mr[i]) for i in range(len(pairs))}
print(f"{len(pairs)} concepts. b*_final hist:", dict(sorted(collections.Counter(bl.values()).items())))
print("b*_running hist:", dict(sorted(collections.Counter(br.values()).items())))
TRIG = "../exp_trigger_prompts/work"
cells = []
for fn in (f"{TRIG}/judge/steerscore-cells.jsonl", f"{TRIG}/judge-1217/steerscore-cells.jsonl"):
    cells += [json.loads(l) for l in open(fn)]
g = {(r["pair_id"], r["layer"], r["alpha"]): r["score"] for r in cells}
layers = sorted({r["layer"] for r in cells})
lo, hi = min(layers), max(layers)
concepts = sorted({r["pair_id"] for r in cells})
clip = lambda L: int(np.clip(L, lo, hi))
for a in (0.5, 0.25):
    ks = [k for k in concepts if (k, 22, a) in g and k in bl]
    fx = np.array([g[(k, 22, a)] for k in ks])

    def sel(bmap):
        return np.array([g.get((k, clip(bmap[k]), a), np.nan) for k in ks])

    bfa = sel(bl)
    bra = sel(br)
    orac = np.array([max(g.get((k, L, a), -9) for L in layers) for k in ks])
    bestrow, bestval = max(((L, np.mean([g.get((k, L, a), np.nan) for k in ks])) for L in layers), key=lambda t: t[1])
    print(f"\n=== concept pole, alpha +{a}, n={len(ks)} ===")
    print(f"  fixed row 22          {fx.mean():.4f}")
    print(f"  best CONSTANT row {bestrow:<2}  {bestval:.4f}   (Δ {bestval - fx.mean():+.4f})")
    print(
        f"  heuristic b*_final    {np.nanmean(bfa):.4f}   (Δ {np.nanmean(bfa) - fx.mean():+.4f}; up {np.mean(bfa > fx):.0%} down {np.mean(bfa < fx):.0%})"
    )
    print(f"  heuristic b*_running  {np.nanmean(bra):.4f}   (Δ {np.nanmean(bra) - fx.mean():+.4f})")
    print(f"  ORACLE per-concept    {orac.mean():.4f}   (Δ {orac.mean() - fx.mean():+.4f})  <-circular ceiling")
    diff = [k for k in ks if clip(bl[k]) != 22]
    if diff:
        d = np.array([g[(k, clip(bl[k]), a)] - g[(k, 22, a)] for k in diff])
        print(
            f"  where b*_final!=22 (n={len(diff)}): mean Δ {d.mean():+.4f} (up {np.mean(d > 0):.0%}, down {np.mean(d < 0):.0%})"
        )
