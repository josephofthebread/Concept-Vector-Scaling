from __future__ import annotations

import json
import logging
import time
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import pyarrow.parquet as pq
import torch
from safetensors.torch import save_file

from exp_attn_interp.common import ValueCatcher, load_model, story_span
from exp_attn_interp.config import GENDATA_ROOT, INSTRUCT_MODEL, vector_tag_for
from exp_attn_interp.decomp import POOL_MIN, POOL_SKIP, MlpCatcher
from exp_attn_interp.vectors import load_diff_vectors
from exp_attn_interp.write import story_ids

log = logging.getLogger("attnsrc")
CATS = ("template", "marker_own", "marker_other", "self", "delim", "story_other")
DELIMS = {".", ",", ";", ":", "!", "?", "'", '"', ".'", '."', ")", "(", "-", "—", "...", "'s"}


def is_delim(s: str) -> bool:
    t = s.strip()
    return t in DELIMS or t == "" or all(ch in ".,;:!?'\"()-—…\n" for ch in t)


@torch.inference_mode()
def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    parser.add_argument("--markers", type=Path, default=Path("data/markers_presence.json"))
    parser.add_argument("--per-pole", type=int, default=4)
    parser.add_argument("--every", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("work/attnsrc"))
    parser.add_argument("--offline", action=BooleanOptionalAction, default=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(offline=args.offline, model_name=args.model)
    catcher = ValueCatcher(model.model)
    mlps = MlpCatcher(model.model)
    vectors = load_diff_vectors(device=str(model.device), tag=vector_tag_for(args.model))
    mk = json.loads(args.markers.read_text())
    tbl = pq.read_table(
        GENDATA_ROOT / "data" / "semantic.parquet",
        columns=["pair_id", "concept_text", "antagonist_text"],
        filters=[("language_id", "==", 0)],
    )
    by_pair: dict[int, list[tuple[str, str]]] = {}
    for p, c, a in zip(tbl["pair_id"].to_pylist(), tbl["concept_text"].to_pylist(), tbl["antagonist_text"].to_pylist()):
        by_pair.setdefault(int(p), []).append((c, a))
    pairs = sorted(by_pair)[:: args.every]
    B = len(model.model.layers)
    heads, kv, hd = catcher.heads, catcher.kv_heads, catcher.head_dim
    group = heads // kv
    dev = model.device
    started = time.monotonic()
    for n, k in enumerate(pairs):
        ids_c = set(mk["concept"].get(str(k), []))
        ids_a = set(mk["antagonist"].get(str(k), []))
        for j, (c, a) in enumerate(by_pair[k][: args.per_pole]):
            for side, text in ((0, c), (1, a)):
                path = args.out / f"item-{k:04d}-{j}-{side}.safetensors"
                if path.exists():
                    continue
                ids = story_ids(tokenizer, text)
                T = len(ids)
                s0, s1 = story_span(tokenizer, ids, text)
                span = (s0 + POOL_SKIP, s1) if (s1 - s0 - POOL_SKIP) >= POOL_MIN else (s0, s1)
                tid = torch.tensor(ids, device=dev)
                own, oth = (ids_c, ids_a) if side == 0 else (ids_a, ids_c)
                is_own = torch.tensor([t in own for t in ids], device=dev)
                is_oth = torch.tensor([t in oth and t not in own for t in ids], device=dev)
                pos = torch.arange(T, device=dev)
                is_del = torch.tensor([is_delim(tokenizer.decode([t])) for t in ids], device=dev)
                cat = torch.full((T,), 5, device=dev)
                cat[is_del] = 4
                cat[is_oth] = 2
                cat[is_own] = 1
                cat[pos < s0] = 0
                cat[pos >= s1] = 0
                onehot = torch.nn.functional.one_hot(cat, len(CATS)).to(torch.float32)
                out = model(input_ids=tid[None], output_attentions=True, use_cache=False)
                D = torch.stack([vectors[:B, k], vectors[B - 1, k].expand(B, -1)], 0)
                contrib = torch.zeros(2, B, heads, len(CATS))
                mass = torch.zeros(B, heads, len(CATS))
                src = torch.zeros(2, B, T)
                for b in range(B):
                    attn = out.attentions[b][0].to(torch.float32)[:, span[0] : span[1], :]
                    raw = catcher.values[b]
                    assert raw is not None
                    vals = raw.view(T, kv, hd).to(torch.float32)
                    Vv = vals[:, torch.arange(heads, device=dev) // group]
                    scale, gain = mlps.attn_scale[b], mlps.attn_gain[b]
                    W_O = model.model.layers[b].self_attn.o_proj.weight.to(torch.float32).view(-1, heads, hd)
                    for i in range(2):
                        d = D[i, b] * gain if gain is not None else D[i, b]
                        w = torch.einsum("ohd,o->hd", W_O, d)
                        P = torch.einsum("thd,hd->ht", Vv, w)
                        A = attn * P[:, None, :]
                        if scale is not None:
                            A = A * scale[span[0] : span[1]][None, :, None]
                        Sn = span[1] - span[0]
                        per_src = A.sum(1) / Sn
                        c = per_src @ onehot
                        dsrc = torch.diagonal(A[:, :, span[0] : span[1]], dim1=1, dim2=2) / Sn
                        c -= dsrc @ onehot[span[0] : span[1]]
                        c[:, 3] = dsrc.sum(1)
                        contrib[i, b] = c.cpu()
                        src[i, b] = per_src.sum(0).cpu()
                    m = attn.sum(1) / (span[1] - span[0])
                    mass[b] = (m @ onehot).cpu()
                    del attn, A
                save_file(
                    {
                        "contrib": contrib,
                        "mass": mass,
                        "src": src.to(torch.float16),
                        "ids": tid.cpu(),
                        "cat": cat.cpu().to(torch.int8),
                        "pair_id": torch.tensor(k),
                        "side": torch.tensor(side),
                        "span": torch.tensor(span),
                    },
                    str(path),
                    metadata={"cats": ",".join(CATS), "dirs": "v_k(b),v_k(last)"},
                )
        if n < 3 or n % 20 == 0:
            log.info("%d/%d pair %d: %.1f min", n, len(pairs), k, (time.monotonic() - started) / 60)
    log.info("done %d pairs, %.1f min", len(pairs), (time.monotonic() - started) / 60)


if __name__ == "__main__":
    main()
