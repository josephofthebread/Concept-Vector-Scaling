from __future__ import annotations

import logging
import time
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import pyarrow.parquet as pq
import torch
from safetensors.torch import save_file

from exp_attn_interp.common import ValueCatcher, load_model, story_span
from exp_attn_interp.config import GENDATA_ROOT, INSTRUCT_MODEL
from exp_attn_interp.decomp import POOL_MIN, POOL_SKIP, MlpCatcher
from exp_attn_interp.write import story_ids

log = logging.getLogger("vdecomp")


@torch.inference_mode()
def story_writes(model, tokenizer, catcher, mlps, story: str):
    """Pooled (over the story span) write vectors: heads [B,H,hidden], mlp [B,hidden], embed [hidden],
    resid [B+1, hidden]."""
    ids = story_ids(tokenizer, story)
    T = len(ids)
    s0, s1 = story_span(tokenizer, ids, story)
    span = list(range(s0 + POOL_SKIP, s1)) if (s1 - s0 - POOL_SKIP) >= POOL_MIN else list(range(s0, s1))
    out = model(
        input_ids=torch.tensor([ids], device=model.device),
        output_attentions=True,
        output_hidden_states=True,
        use_cache=False,
    )
    device = model.device
    S = torch.tensor(span, device=device)
    heads, kv, hd = catcher.heads, catcher.kv_heads, catcher.head_dim
    group = heads // kv
    B = len(model.model.layers)
    H_out = torch.empty(B, heads, model.config.hidden_size, device=device)
    M_out = torch.empty(B, model.config.hidden_size, device=device)
    for b in range(B):
        attn = out.attentions[b][0].to(torch.float32)
        vals = catcher.values[b].view(T, kv, hd).to(torch.float32)
        Vv = vals[:, torch.arange(heads, device=device) // group]
        ctx = torch.bmm(attn[:, S, :], Vv.permute(1, 0, 2))
        scale, gain = mlps.attn_scale[b], mlps.attn_gain[b]
        if scale is not None:
            ctx = ctx * scale[S][None, :, None]
        W_O = model.model.layers[b].self_attn.o_proj.weight.to(torch.float32).view(-1, heads, hd)
        hp = torch.einsum("hsd,ohd->ho", ctx, W_O) / len(span)
        if gain is not None:
            hp = hp * gain[None, :]
        H_out[b] = hp
        mo = mlps.out[b]
        assert mo is not None
        M_out[b] = mo[0][S].to(torch.float32).mean(0)
        del attn, ctx
    resid = torch.stack([h[0][S].to(torch.float32).mean(0) for h in out.hidden_states])
    embed = resid[0]
    return H_out, M_out, embed, resid


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    parser.add_argument("--per-pole", type=int, default=8)
    parser.add_argument(
        "--story-offset", type=int, default=0, help="skip this many stories per pole (a disjoint second half)"
    )
    parser.add_argument("--out", type=Path, default=Path("work/vdecomp"))
    parser.add_argument("--pairs", default="", help="comma list; default all")
    parser.add_argument("--start", type=int, default=0, help="first pair index (in sorted order)")
    parser.add_argument("--end", type=int, default=0, help="end pair index (exclusive; 0 = all)")
    parser.add_argument("--offline", action=BooleanOptionalAction, default=True)
    parser.add_argument(
        "--language", type=int, default=0, help="language_id of the stories (0 en, 1 de, 2 zh, 3 ru, 4 es)"
    )
    parser.add_argument(
        "--presentation", default="assistant", choices=["assistant", "raw"], help="raw = bare story tokens (base model)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(offline=args.offline, model_name=args.model)
    if args.presentation == "raw":
        import exp_attn_interp.vdecomp as _self

        _self.story_ids = lambda tok, story: tok(story, add_special_tokens=True)["input_ids"]  # type: ignore[assignment]
    catcher = ValueCatcher(model.model)
    mlps = MlpCatcher(model.model)
    tbl = pq.read_table(
        GENDATA_ROOT / "data" / "semantic.parquet",
        columns=["pair_id", "concept_text", "antagonist_text"],
        filters=[("language_id", "==", args.language)],
    )
    by_pair: dict[int, list[tuple[str, str]]] = {}
    for p, c, a in zip(tbl["pair_id"].to_pylist(), tbl["concept_text"].to_pylist(), tbl["antagonist_text"].to_pylist()):
        by_pair.setdefault(int(p), []).append((c, a))
    pairs = [int(x) for x in args.pairs.split(",")] if args.pairs else sorted(by_pair)[args.start : args.end or None]
    started = time.monotonic()
    for n, k in enumerate(pairs):
        path = args.out / f"pair-{k:04d}.safetensors"
        if path.exists():
            continue
        B = len(model.model.layers)
        hid = model.config.hidden_size
        dev = model.device
        acc_h = torch.zeros(2, B, catcher.heads, hid, device=dev)
        acc_m = torch.zeros(2, B, hid, device=dev)
        acc_e = torch.zeros(2, hid, device=dev)
        acc_r = torch.zeros(2, B + 1, hid, device=dev)
        counts = [0, 0]
        for c, a in by_pair[k][args.story_offset : args.story_offset + args.per_pole]:
            for side, text in ((0, c), (1, a)):
                Hh, Mm, Ee, Rr = story_writes(model, tokenizer, catcher, mlps, text)
                acc_h[side] += Hh
                acc_m[side] += Mm
                acc_e[side] += Ee
                acc_r[side] += Rr
                counts[side] += 1
        cnt = torch.tensor(counts, device=acc_h.device, dtype=torch.float32).clamp_min(1)
        acc_h /= cnt[:, None, None, None]
        acc_m /= cnt[:, None, None]
        acc_e /= cnt[:, None]
        acc_r /= cnt[:, None, None]
        recon = acc_e + acc_h[:, : B - 1].sum((1, 2)) + acc_m[:, : B - 1].sum(1)
        err = ((recon - acc_r[:, B - 1]).norm(dim=-1) / acc_r[:, B - 1].norm(dim=-1)).tolist()
        save_file(
            {
                "pole_head": acc_h.to(torch.bfloat16).cpu(),
                "pole_mlp": acc_m.to(torch.bfloat16).cpu(),
                "pole_embed": acc_e.to(torch.bfloat16).cpu(),
                "pole_resid": acc_r.to(torch.bfloat16).cpu(),
                "n_stories": torch.tensor(counts),
            },
            str(path),
            metadata={"pair": str(k)},
        )
        if n < 3 or n % 50 == 0:
            log.info(
                "%d/%d pair %d: %d+%d stories, recon rel.err at row %d: %.2e / %.2e, %.1f min",
                n,
                len(pairs),
                k,
                counts[0],
                counts[1],
                B - 1,
                err[0],
                err[1],
                (time.monotonic() - started) / 60,
            )
    log.info("done: %d pairs, %.1f min", len(pairs), (time.monotonic() - started) / 60)


if __name__ == "__main__":
    main()
