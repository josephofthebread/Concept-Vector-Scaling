from __future__ import annotations

import json
import logging
import time
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from exp_attn_interp.common import load_model, story_span
from exp_attn_interp.config import GENDATA_ROOT, INSTRUCT_MODEL, MARKERS_PRESENCE, vector_tag_for
from exp_attn_interp.vectors import load_diff_vectors
from exp_attn_interp.write import story_ids

log = logging.getLogger("tokproj")


def unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


@torch.inference_mode()
def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    parser.add_argument("--per-pole", type=int, default=8)
    parser.add_argument("--language", type=int, default=0, help="0 en, 1 de, 2 zh, 3 ru, 4 es")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=0)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--out", type=Path, default=Path("work/tokproj"))
    parser.add_argument("--offline", action=BooleanOptionalAction, default=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(offline=args.offline, model_name=args.model)
    dev = model.device
    V = load_diff_vectors(device=str(dev), tag=vector_tag_for(args.model))
    R = V.shape[0]
    E = model.model.embed_tokens.weight.to(torch.float32)
    U = model.lm_head.weight.to(torch.float32) * model.model.norm.weight.to(torch.float32)[None, :]
    markers = json.loads(MARKERS_PRESENCE.read_text())
    tbl = pq.read_table(
        GENDATA_ROOT / "data" / "semantic.parquet",
        columns=["pair_id", "concept_text", "antagonist_text"],
        filters=[("language_id", "==", args.language)],
    )
    by_pair: dict[int, list[tuple[str, str]]] = {}
    for p, c, a in zip(tbl["pair_id"].to_pylist(), tbl["concept_text"].to_pylist(), tbl["antagonist_text"].to_pylist()):
        by_pair.setdefault(int(p), []).append((c, a))
    pairs = sorted(by_pair)[args.start : args.end or None][:: args.every]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    started = time.monotonic()
    done = 0
    for n, k in enumerate(pairs):
        path = args.out / f"pair-{k:04d}.npz"
        if path.exists():
            continue
        cm = torch.tensor(markers["concept"][str(k)], device=dev)
        am = torch.tensor(markers["antagonist"][str(k)], device=dev)
        e_gap = unit(E[cm].mean(0) - E[am].mean(0))
        u_gap = unit(U[cm].mean(0) - U[am].mean(0))
        cset, aset = set(cm.tolist()), set(am.tolist())
        stories = [(0, c) for c, _ in by_pair[k][: args.per_pole]] + [(1, a) for _, a in by_pair[k][: args.per_pole]]
        seqs, spans = [], []
        for _, text in stories:
            ids = story_ids(tokenizer, text)
            seqs.append(ids)
            spans.append(story_span(tokenizer, ids, text))
        S = len(stories)
        Lmax = max(s1 - s0 for s0, s1 in spans)
        proj = np.full((S, Lmax, R), np.nan, dtype=np.float16)
        norm = np.full((S, Lmax, R), np.nan, dtype=np.float16)
        pol_e = np.full((S, Lmax), np.nan, dtype=np.float16)
        pol_u = np.full((S, Lmax), np.nan, dtype=np.float16)
        tok = np.full((S, Lmax), -1, dtype=np.int32)
        flag = np.zeros((S, Lmax), dtype=np.int8)
        length = np.zeros(S, dtype=np.int32)
        pole = np.array([p for p, _ in stories], dtype=np.int8)
        for c0 in range(0, S, args.batch):
            chunk = list(range(c0, min(S, c0 + args.batch)))
            T = max(len(seqs[i]) for i in chunk)
            input_ids = torch.full((len(chunk), T), pad_id, dtype=torch.long, device=dev)
            attn = torch.zeros((len(chunk), T), dtype=torch.long, device=dev)
            for j, i in enumerate(chunk):
                input_ids[j, : len(seqs[i])] = torch.tensor(seqs[i], device=dev)
                attn[j, : len(seqs[i])] = 1
            hs = model(
                input_ids=input_ids, attention_mask=attn, output_hidden_states=True, use_cache=False
            ).hidden_states
            for j, i in enumerate(chunk):
                s0, s1 = spans[i]
                L = s1 - s0
                length[i] = L
                H = torch.stack([hs[r][j, s0:s1] for r in range(R)]).to(torch.float32)
                proj[i, :L] = torch.einsum("rld,rd->rl", H, V[:, k]).T.cpu().numpy().astype(np.float16)
                norm[i, :L] = H.norm(dim=-1).T.cpu().numpy().astype(np.float16)
                del H
                ids_t = torch.tensor(seqs[i][s0:s1], device=dev)
                tok[i, :L] = ids_t.cpu().numpy()
                pol_e[i, :L] = (unit(E[ids_t]) @ e_gap).cpu().numpy().astype(np.float16)
                pol_u[i, :L] = (unit(U[ids_t]) @ u_gap).cpu().numpy().astype(np.float16)
                flag[i, :L] = np.array(
                    [1 if t in cset else (-1 if t in aset else 0) for t in seqs[i][s0:s1]], dtype=np.int8
                )
            del hs
        np.savez_compressed(
            str(path),
            proj=proj,
            norm=norm,
            pol_e=pol_e,
            pol_u=pol_u,
            tok=tok,
            flag=flag,
            length=length,
            pole=pole,
            pair=np.array(k),
        )
        done += 1
        if done <= 3 or done % 50 == 0:
            el = (time.monotonic() - started) / 60
            log.info(
                "%d/%d pair %d: %d stories, Lmax %d, %.1f min (%.2f min/pair)", n, len(pairs), k, S, Lmax, el, el / done
            )
    log.info("done: %d new pairs, %.1f min", done, (time.monotonic() - started) / 60)


if __name__ == "__main__":
    main()
