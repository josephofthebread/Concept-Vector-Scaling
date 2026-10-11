from __future__ import annotations

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

log = logging.getLogger("compproj")


@torch.inference_mode()
def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    parser.add_argument("--per-pole", type=int, default=16)
    parser.add_argument("--every", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("work/compproj"))
    parser.add_argument("--offline", action=BooleanOptionalAction, default=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(offline=args.offline, model_name=args.model)
    catcher = ValueCatcher(model.model)
    mlps = MlpCatcher(model.model)
    vectors = load_diff_vectors(device=str(model.device), tag=vector_tag_for(args.model))
    B = len(model.model.layers)
    last = B - 1
    heads, kv, hd = catcher.heads, catcher.kv_heads, catcher.head_dim
    group = heads // kv
    dev = model.device
    tbl = pq.read_table(
        GENDATA_ROOT / "data" / "semantic.parquet",
        columns=["pair_id", "concept_text", "antagonist_text"],
        filters=[("language_id", "==", 0)],
    )
    by_pair: dict[int, list[tuple[str, str]]] = {}
    for p, c, a in zip(tbl["pair_id"].to_pylist(), tbl["concept_text"].to_pylist(), tbl["antagonist_text"].to_pylist()):
        by_pair.setdefault(int(p), []).append((c, a))
    pairs = sorted(by_pair)[:: args.every]
    started = time.monotonic()
    for n, k in enumerate(pairs):
        path = args.out / f"pair-{k:04d}.safetensors"
        if path.exists():
            continue
        u = vectors[last, k]
        S = min(args.per_pole, len(by_pair[k]))
        a_mean, a_std = torch.zeros(2, S, B), torch.zeros(2, S, B)
        m_mean, m_std = torch.zeros(2, S, B), torch.zeros(2, S, B)
        e_mean, e_std = torch.zeros(2, S), torch.zeros(2, S)
        h_mean, h_std = torch.zeros(2, S, B + 1), torch.zeros(2, S, B + 1)
        cov_am = torch.zeros(2, S, B + 1)
        for j, (c, a) in enumerate(by_pair[k][:S]):
            for side, text in ((0, c), (1, a)):
                ids = story_ids(tokenizer, text)
                T = len(ids)
                s0, s1 = story_span(tokenizer, ids, text)
                if (s1 - s0 - POOL_SKIP) >= POOL_MIN:
                    s0 += POOL_SKIP
                out = model(
                    input_ids=torch.tensor([ids], device=dev),
                    output_attentions=True,
                    output_hidden_states=True,
                    use_cache=False,
                )
                A = torch.zeros(B, s1 - s0, device=dev)
                M = torch.zeros(B, s1 - s0, device=dev)
                for b in range(B):
                    attn = out.attentions[b][0].to(torch.float32)[:, s0:s1, :]
                    raw = catcher.values[b]
                    assert raw is not None
                    Vv = raw.view(T, kv, hd).to(torch.float32)[:, torch.arange(heads, device=dev) // group]
                    W_O = model.model.layers[b].self_attn.o_proj.weight.to(torch.float32).view(-1, heads, hd)
                    w = torch.einsum("ohd,o->hd", W_O, u)
                    P = torch.einsum("thd,hd->ht", Vv, w)
                    A[b] = torch.einsum("hst,ht->s", attn, P)
                    mo = mlps.out[b]
                    assert mo is not None
                    M[b] = mo[0, s0:s1].to(torch.float32) @ u
                    del attn
                E = out.hidden_states[0][0, s0:s1].to(torch.float32) @ u
                Hh = torch.stack([out.hidden_states[L][0, s0:s1].to(torch.float32) @ u for L in range(B + 1)])
                cumA = torch.cat([torch.zeros(1, s1 - s0, device=dev), torch.cumsum(A, 0)], 0)
                cumM = torch.cat([torch.zeros(1, s1 - s0, device=dev), torch.cumsum(M, 0)], 0)
                a_mean[side, j], a_std[side, j] = A.mean(1).cpu(), A.std(1).cpu()
                m_mean[side, j], m_std[side, j] = M.mean(1).cpu(), M.std(1).cpu()
                e_mean[side, j], e_std[side, j] = E.mean().cpu(), E.std().cpu()
                h_mean[side, j], h_std[side, j] = Hh.mean(1).cpu(), Hh.std(1).cpu()
                ca, cm = cumA - cumA.mean(1, keepdim=True), cumM - cumM.mean(1, keepdim=True)
                cov_am[side, j] = ((ca * cm).mean(1)).cpu()
                del out
        recon = (e_mean + a_mean[:, :, :last].sum(-1) + m_mean[:, :, :last].sum(-1) - h_mean[:, :, last]).abs().max()
        save_file(
            {
                "a_mean": a_mean,
                "a_std": a_std,
                "m_mean": m_mean,
                "m_std": m_std,
                "e_mean": e_mean,
                "e_std": e_std,
                "h_mean": h_mean,
                "h_std": h_std,
                "cov_am": cov_am,
                "pair_id": torch.tensor(k),
            },
            str(path),
        )
        if n < 3 or n % 20 == 0:
            gap = h_mean[0, :, 22].mean() - h_mean[1, :, 22].mean()
            log.info(
                "%d/%d pair %d: %d stories/pole, recon max abs err %.2e, pole gap at L22 %.2f, token std %.2f; %.1f min",
                n,
                len(pairs),
                k,
                S,
                float(recon),
                float(gap),
                float(h_std[:, :, 22].mean()),
                (time.monotonic() - started) / 60,
            )
    log.info("done %d pairs, %.1f min", len(pairs), (time.monotonic() - started) / 60)


if __name__ == "__main__":
    main()
