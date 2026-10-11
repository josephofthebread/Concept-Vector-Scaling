from __future__ import annotations

import collections
import heapq
import json
import logging
import time
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import pyarrow.parquet as pq
import torch

from exp_attn_interp.common import load_model, story_span
from exp_attn_interp.config import GENDATA_ROOT
from exp_attn_interp.decomp import POOL_MIN, POOL_SKIP
from exp_attn_interp.saedecode import sae_path
from exp_attn_interp.write import story_ids

log = logging.getLogger("saelabel")
PAIRS25 = [
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


def load_sae(layer: int, device):
    sd = torch.load(sae_path(layer), map_location="cpu")
    return {k: sd[k].to(device, torch.float32) for k in ("W_enc", "b_enc", "W_dec", "b_dec")}


def encode(h: torch.Tensor, sae, k: int = 50):
    pre = h @ sae["W_enc"].T + sae["b_enc"]
    vals, idx = pre.topk(k, dim=-1)
    return vals.clamp_min(0), idx


@torch.inference_mode()
def main() -> None:
    p = ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B-Base")
    p.add_argument("--presentation", default="raw", choices=["raw", "assistant"])
    p.add_argument("--rows", default="4,12,19,22,24,28,35")
    p.add_argument("--pairs", default=",".join(str(x) for x in PAIRS25))
    p.add_argument("--per-pole", type=int, default=16)
    p.add_argument("--features", type=Path, default=None, help="decode output whose features get contexts")
    p.add_argument(
        "--wanted", type=Path, default=None, help="JSON {row: [feature ids]} of further features to record contexts for"
    )
    p.add_argument("--contexts", type=int, default=12)
    p.add_argument("--out", type=Path, default=Path("work/sae/label-base.json"))
    p.add_argument("--offline", action=BooleanOptionalAction, default=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(offline=args.offline, model_name=args.model)
    if args.presentation == "raw":
        import exp_attn_interp.saelabel as _self

        _self.story_ids = lambda tok, story: tok(story, add_special_tokens=True)["input_ids"]  # type: ignore[assignment]
    dev = model.device
    rows = [int(x) for x in args.rows.split(",")]
    pairs = [int(x) for x in args.pairs.split(",")]
    tbl = pq.read_table(
        GENDATA_ROOT / "data" / "semantic.parquet",
        columns=["pair_id", "concept_text", "antagonist_text"],
        filters=[("language_id", "==", 0)],
    )
    by_pair: dict[int, list[tuple[str, str]]] = {}
    for pid, c, a in zip(
        tbl["pair_id"].to_pylist(), tbl["concept_text"].to_pylist(), tbl["antagonist_text"].to_pylist()
    ):
        by_pair.setdefault(int(pid), []).append((c, a))
    saes = {r: load_sae(r - 1, dev) for r in rows}
    F = saes[rows[0]]["W_enc"].shape[0]
    wanted: dict[int, set[int]] = {r: set() for r in rows}
    if args.features and args.features.exists():
        dec = json.loads(args.features.read_text())
        for r in rows:
            objs = dec["rows"].get(str(r), {}).get("objects", {})
            for name, o in objs.items():
                wanted[r].update(d["f"] for d in o["top"][:10])
    if args.wanted and args.wanted.exists():
        for r, feats in json.loads(args.wanted.read_text()).items():
            if int(r) in wanted:
                wanted[int(r)].update(int(j) for j in feats)
    log.info("features tracked per row: %s", {r: len(v) for r, v in wanted.items()})
    pole_sum = {r: torch.zeros(len(pairs), 2, F, device=dev) for r in rows}
    pole_cnt = {r: torch.zeros(len(pairs), 2, device=dev) for r in rows}
    fire = {r: torch.zeros(F, device=dev) for r in rows}
    ntok = {r: 0 for r in rows}
    recon = {r: [0.0, 0.0] for r in rows}
    heaps: dict[int, dict[int, list]] = {r: collections.defaultdict(list) for r in rows}
    started = time.monotonic()
    for pi, k in enumerate(pairs):
        for c, a in by_pair[k][: args.per_pole]:
            for side, text in ((0, c), (1, a)):
                ids = story_ids(tokenizer, text)
                s0, s1 = story_span(tokenizer, ids, text)
                span = list(range(s0 + POOL_SKIP, s1)) if (s1 - s0 - POOL_SKIP) >= POOL_MIN else list(range(s0, s1))
                out = model(input_ids=torch.tensor([ids], device=dev), output_hidden_states=True, use_cache=False)
                for r in rows:
                    h = out.hidden_states[r][0, span].float()
                    vals, idx = encode(h, saes[r])
                    dense = torch.zeros(h.shape[0], F, device=dev)
                    dense.scatter_(1, idx, vals)
                    pole_sum[r][pi, side] += dense.sum(0)
                    pole_cnt[r][pi, side] += h.shape[0]
                    fire[r] += (dense > 0).float().sum(0)
                    ntok[r] += h.shape[0]
                    rec = dense @ saes[r]["W_dec"].T + saes[r]["b_dec"]
                    recon[r][0] += float(((h - rec) ** 2).sum())
                    recon[r][1] += float(((h - h.mean(0)) ** 2).sum())
                    want = wanted[r]
                    for ti in range(h.shape[0]):
                        for j, val in zip(idx[ti].tolist(), vals[ti].tolist()):
                            if val <= 0 or (j not in want and len(want) > 0):
                                continue
                            hp = heaps[r][j]
                            tok_pos = span[ti]
                            ctx = (
                                tokenizer.decode(ids[max(0, tok_pos - 8) : tok_pos])
                                + "⟦"
                                + tokenizer.decode([ids[tok_pos]])
                                + "⟧"
                            )
                            item = (val, k, side, ctx)
                            if len(hp) < args.contexts:
                                heapq.heappush(hp, item)
                            elif val > hp[0][0]:
                                heapq.heapreplace(hp, item)
        if pi % 5 == 0:
            log.info("%d/%d pairs, %.1f min", pi + 1, len(pairs), (time.monotonic() - started) / 60)
    result = {"model": args.model, "presentation": args.presentation, "pairs": pairs, "rows": {}}
    for r in rows:
        means = pole_sum[r] / pole_cnt[r].clamp_min(1)[:, :, None]
        diff = means[:, 0] - means[:, 1]
        spread = means.mean(1).mean(0).clamp_min(1e-6)
        per_concept = {}
        disc_feats: set[int] = set()
        for pi, k in enumerate(pairs):
            top = (diff[pi] / (spread + diff[pi].abs().mean())).abs().topk(12)
            lst = [
                {
                    "f": int(j),
                    "diff": round(float(diff[pi, j]), 4),
                    "mean_c": round(float(means[pi, 0, j]), 4),
                    "mean_a": round(float(means[pi, 1, j]), 4),
                }
                for j in top.indices.tolist()
            ]
            per_concept[str(k)] = lst
            disc_feats.update(int(j) for j in top.indices.tolist())
        result["rows"][str(r)] = {
            "recon_var_explained": round(1 - recon[r][0] / max(recon[r][1], 1e-6), 4),
            "tokens": ntok[r],
            "pole_disc": per_concept,
            "contexts": {
                str(j): [
                    {"act": round(v, 3), "pair": kk, "pole": s, "ctx": c}
                    for v, kk, s, c in sorted(heaps[r][j], reverse=True)
                ]
                for j in heaps[r]
                if heaps[r][j]
            },
            "fire_rate": {str(j): round(float(fire[r][j] / max(ntok[r], 1)), 5) for j in set(heaps[r]) | disc_feats},
        }
        log.info(
            "row %d: recon var explained %.3f over %d tokens; %d features with contexts",
            r,
            result["rows"][str(r)]["recon_var_explained"],
            ntok[r],
            len(heaps[r]),
        )
    args.out.write_text(json.dumps(result))
    log.info("wrote %s (%.1f min)", args.out, (time.monotonic() - started) / 60)


if __name__ == "__main__":
    main()
