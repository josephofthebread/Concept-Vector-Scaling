import collections
import json
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch

from exp_attn_interp.common import load_model, story_span
from exp_attn_interp.config import GENDATA_ROOT
from exp_attn_interp.decomp import POOL_MIN, POOL_SKIP
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
PER = int(sys.argv[2]) if len(sys.argv) > 2 else 16
BLOCKS = list(range(18, 28))
POS = [f"{b}:{c}" for b in BLOCKS for c in ("attn", "mlp")]
model, tok = load_model(offline=True)
dev = model.device
tbl = pq.read_table(
    GENDATA_ROOT / "data" / "semantic.parquet",
    columns=["pair_id", "concept_text", "antagonist_text"],
    filters=[("language_id", "==", 0)],
)
by_pair: dict[int, list] = {}
for pid, c, a in zip(tbl["pair_id"].to_pylist(), tbl["concept_text"].to_pylist(), tbl["antagonist_text"].to_pylist()):
    by_pair.setdefault(int(pid), []).append((c, a))
cap: dict[str, torch.Tensor] = {}
pre_in: dict[int, torch.Tensor] = {}
handles = []
for b in BLOCKS:

    def attn_hook(_m, _i, out, b=b):
        cap[f"{b}:attn"] = (out[0] if isinstance(out, tuple) else out).detach()[0].float()

    def blk_hook(_m, _i, out, b=b):
        cap[f"{b}:mlp"] = (out[0] if isinstance(out, tuple) else out).detach()[0].float()

    def pre_hook(_m, args, kwargs, b=b):
        x = args[0] if args else kwargs["hidden_states"]
        pre_in[b] = x.detach()[0].float()
        return None

    handles.append(model.model.layers[b].self_attn.register_forward_hook(attn_hook))
    handles.append(model.model.layers[b].register_forward_hook(blk_hook))
    handles.append(model.model.layers[b].register_forward_pre_hook(pre_hook, with_kwargs=True))


def residuals(text):
    """Return {position: [T, hid]} over one story's own tokens."""
    ids = story_ids(tok, text)
    s0, s1 = story_span(tok, ids, text)
    span = list(range(s0 + POOL_SKIP, s1)) if (s1 - s0 - POOL_SKIP) >= POOL_MIN else list(range(s0, s1))
    model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
    out = {}
    for b in BLOCKS:
        out[f"{b}:attn"] = (pre_in[b] + cap[f"{b}:attn"])[span]
        out[f"{b}:mlp"] = cap[f"{b}:mlp"][span]
    return out


res: dict[str, dict[str, list]] = {p: collections.defaultdict(list) for p in POS}
t0 = time.monotonic()
half = PER // 2
with torch.inference_mode():
    for pi, k in enumerate(PAIRS):
        stories = by_pair[k][:PER]
        sums = {
            p: [torch.zeros(model.config.hidden_size, device=dev), torch.zeros(model.config.hidden_size, device=dev)]
            for p in POS
        }
        for c, a in stories[:half]:
            for side, text in ((0, c), (1, a)):
                R = residuals(text)
                for p in POS:
                    sums[p][side] += R[p].mean(0)
        direction = {p: torch.nn.functional.normalize(sums[p][0] - sums[p][1], dim=-1) for p in POS}
        proj = {p: ([], []) for p in POS}
        story = {p: ([], []) for p in POS}
        for c, a in stories[half:]:
            for side, text in ((0, c), (1, a)):
                R = residuals(text)
                for p in POS:
                    z = (R[p] @ direction[p]).cpu().numpy()
                    story[p][side].append(float(z.mean()))
                    proj[p][side].append(z)
        for p in POS:
            sc = np.array(story[p][0])
            sa = np.array(story[p][1])
            within = np.mean([x.var() for x in proj[p][0] + proj[p][1]])
            snr = abs(sc.mean() - sa.mean()) / np.sqrt(max(within, 1e-12))
            auc = float((sc[:, None] > sa[None, :]).mean() + 0.5 * (sc[:, None] == sa[None, :]).mean())
            tc = np.concatenate(proj[p][0])
            ta = np.concatenate(proj[p][1])
            idx = np.argsort(np.concatenate([tc, ta]))
            ranks = np.empty(len(idx))
            ranks[idx] = np.arange(len(idx))
            tauc = float((ranks[: len(tc)].mean() - (len(tc) - 1) / 2) / len(ta))
            res[p]["snr"].append(float(snr))
            res[p]["auc"].append(auc)
            res[p]["tauc"].append(tauc)
        if pi % 5 == 0:
            print(f"{pi + 1}/{len(PAIRS)} pairs, {(time.monotonic() - t0) / 60:.1f} min", flush=True)
for h in handles:
    h.remove()
out = {}
print("\nposition  | token SNR median | mean | story AUC | token AUC | n")
for p in POS:
    out[p] = {
        "snr_median": float(np.median(res[p]["snr"])),
        "snr_mean": float(np.mean(res[p]["snr"])),
        "auc": float(np.mean(res[p]["auc"])),
        "tauc": float(np.mean(res[p]["tauc"])),
        "n": len(res[p]["snr"]),
    }
    print(
        "%-9s | %.4f | %.4f | %.3f | %.3f | %d"
        % (p, out[p]["snr_median"], out[p]["snr_mean"], out[p]["auc"], out[p]["tauc"], out[p]["n"])
    )
best = max(out, key=lambda p: out[p]["snr_median"])
wins = collections.Counter()
for i in range(len(PAIRS)):
    wins[max(POS, key=lambda p: res[p]["snr"][i] if i < len(res[p]["snr"]) else -1)] += 1
print("best position by median token SNR:", best, out[best])
print("per-concept argmax:", wins.most_common(8))
json.dump(
    {"positions": out, "per_concept": {p: res[p]["snr"] for p in POS}, "pairs": PAIRS},
    open("work/tmp/subprobe.json", "w"),
    indent=1,
)
