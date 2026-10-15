import collections
import json
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch

from exp_attn_interp.common import load_model, story_span
from exp_attn_interp.config import GENVECTORS_ROOT, INSTRUCT_MODEL, vector_tag_for
from exp_attn_interp.decomp import POOL_MIN, POOL_SKIP
from exp_attn_interp.posteer import ablate_heads, unit
from exp_attn_interp.vectors import load_diff_vectors
from exp_attn_interp.write import story_ids

PAIRS = (
    [int(x) for x in sys.argv[1].split(",")]
    if len(sys.argv) > 1
    else [
        0,
        43,
        86,
        129,
        172,
        215,
        258,
        301,
        344,
        387,
        430,
        473,
        516,
        559,
        602,
        645,
        688,
        731,
        774,
        817,
        860,
        903,
        946,
        989,
        1032,
    ]
)
PER = int(sys.argv[2]) if len(sys.argv) > 2 else 8
ROW = 22
ARMS = {"none": "", "writers": "19:28,19:29,19:31", "random": "19:4,19:12,19:20"}
model, tok = load_model(offline=True)
dev = model.device
V = load_diff_vectors(device="cpu", tag=vector_tag_for(INSTRUCT_MODEL))[ROW]
m = unit(V.mean(0))
Xc = V - V.mean(0)
_, _, Wt = torch.linalg.svd(Xc, full_matrices=False)
Q = torch.linalg.qr(torch.cat([m[None], Wt[:16]], 0).T)[0]
tbl = pq.read_table(
    GENVECTORS_ROOT / "data" / "semantic.parquet",
    columns=["pair_id", "concept_text", "antagonist_text"],
    filters=[("language_id", "==", 0)],
)
by_pair: dict[int, list] = {}
for pid, c, a in zip(tbl["pair_id"].to_pylist(), tbl["concept_text"].to_pylist(), tbl["antagonist_text"].to_pylist()):
    by_pair.setdefault(int(pid), []).append((c, a))


@torch.inference_mode()
def pooled(text):
    ids = story_ids(tok, text)
    s0, s1 = story_span(tok, ids, text)
    span = list(range(s0 + POOL_SKIP, s1)) if (s1 - s0 - POOL_SKIP) >= POOL_MIN else list(range(s0, s1))
    out = model(input_ids=torch.tensor([ids], device=dev), output_hidden_states=True, use_cache=False)
    return out.hidden_states[ROW][0, span].float().mean(0)


res = collections.defaultdict(lambda: collections.defaultdict(list))
t0 = time.monotonic()
for pi, k in enumerate(PAIRS):
    stories = by_pair[k][:PER]
    vec = {}
    for arm, spec in ARMS.items():
        with ablate_heads(model, spec):
            c = torch.stack([pooled(s[0]) for s in stories]).mean(0)
            a = torch.stack([pooled(s[1]) for s in stories]).mean(0)
        d = (c - a).cpu()
        vec[arm] = d
    base = vec["none"]
    for arm in ARMS:
        d = vec[arm]
        res[arm]["cos_intact"].append(float(d @ base / (d.norm() * base.norm())))
        res[arm]["cos_paper"].append(float(d @ V[k] / d.norm()))
        res[arm]["norm_ratio"].append(float(d.norm() / base.norm()))
        res[arm]["stance_share"].append(float(((d @ Q) ** 2).sum() / (d @ d)))
    if pi % 5 == 0:
        print(f"{pi + 1}/{len(PAIRS)} pairs, {(time.monotonic() - t0) / 60:.1f} min", flush=True)
print("\narm      | cos with the intact vector | cos with the paper's v(22) | |v| ratio | stance-subspace share")
out = {}
for arm in ARMS:
    r = {k: (float(np.median(v)), float(np.percentile(v, 10))) for k, v in res[arm].items()}
    out[arm] = {k: v[0] for k, v in r.items()} | {k + "_p10": v[1] for k, v in r.items()}
    print(
        "%-8s | %.4f (p10 %.4f) | %.4f | %.4f | %.4f"
        % (arm, r["cos_intact"][0], r["cos_intact"][1], r["cos_paper"][0], r["norm_ratio"][0], r["stance_share"][0])
    )
json.dump(
    {"arms": out, "pairs": PAIRS, "per_pole": PER, "row": ROW, "heads": ARMS},
    open("work/tmp/extabl.json", "w"),
    indent=1,
)
print("wrote work/tmp/extabl.json")
