from __future__ import annotations

import logging
import time
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from safetensors.torch import save_file

from exp_attn_interp.common import ValueCatcher, load_model, story_span
from exp_attn_interp.config import GENVECTORS_ROOT, INSTRUCT_MODEL, vector_tag_for
from exp_attn_interp.vectors import load_diff_vectors
from exp_attn_interp.write import story_ids

log = logging.getLogger("decomp")

READ_FRACS = (1 / 6, 1 / 3, 1 / 2, 2 / 3, 5 / 6)
POOL_SKIP = 50
POOL_MIN = 16


def read_layers_for(n_blocks: int) -> list[int]:
    """Read layers at fixed depth fractions plus n_blocks-1 (Qwen: 6,12,18,24,30,35).
    HF's hidden_states[n_blocks] is post-final-norm, so exact decompositions stop at n_blocks-1."""
    return [max(1, round(f * n_blocks)) for f in READ_FRACS] + [n_blocks - 1]


##
# NOTE: gemma-2 normalizes each sublayer's output before the residual add, so a head's write is
# head_h * s_t * (1+w) with s_t = rsqrt(mean(x^2)+eps); pre-norm families have s_t = 1 and gain 1.
##
class MlpCatcher:
    """Catch each block's MLP write to the residual, plus gemma-2's attention-norm scale and gain."""

    def __init__(self, model: Any) -> None:
        n = len(model.layers)
        self.out: list[torch.Tensor | None] = [None] * n
        self.attn_scale: list[torch.Tensor | None] = [None] * n
        self.attn_gain: list[torch.Tensor | None] = [None] * n
        self.postnorm = hasattr(model.layers[0], "pre_feedforward_layernorm")
        for i, block in enumerate(model.layers):
            if self.postnorm:
                block.post_feedforward_layernorm.register_forward_hook(self._catch(i))
                block.post_attention_layernorm.register_forward_hook(self._scale(i))
            else:
                block.mlp.register_forward_hook(self._catch(i))

    def _catch(self, i: int) -> Any:
        def hook(_m: Any, _i: Any, output: torch.Tensor) -> None:
            self.out[i] = output.detach()

        return hook

    def _scale(self, i: int) -> Any:
        def hook(mod: Any, inputs: Any, _o: Any) -> None:
            x = inputs[0][0].detach().to(torch.float32)
            self.attn_scale[i] = torch.rsqrt(x.pow(2).mean(-1, keepdim=False) + float(mod.eps))
            self.attn_gain[i] = 1.0 + mod.weight.detach().to(torch.float32)

        return hook


@torch.inference_mode()
def process_story(
    model, tokenizer, catcher, mlps, vectors, read_layers, pair_id, side, story, out_dir, story_index: int = 0
):
    item_id = 200000 + pair_id * 20 + story_index * 2 + (0 if side == "concept" else 1)
    path = out_dir / "decomp" / f"item-{item_id:06d}.safetensors"
    if path.exists():
        return {"item": item_id, "skipped": True}
    ids = story_ids(tokenizer, story)
    T = len(ids)
    s0, s1 = story_span(tokenizer, ids, story)
    span = list(range(s0 + POOL_SKIP, s1)) if (s1 - s0 - POOL_SKIP) >= POOL_MIN else list(range(s0, s1))
    with torch.inference_mode():
        out = model(
            input_ids=torch.tensor([ids], device=model.device),
            output_attentions=True,
            output_hidden_states=True,
            use_cache=False,
        )
    hidden = list(out.hidden_states)
    device = model.device
    n_blocks = len(model.model.layers)
    S = torch.tensor(span, device=device)
    heads, kv, hd = catcher.heads, catcher.kv_heads, catcher.head_dim
    group = heads // kv
    nL = len(read_layers)
    V = torch.stack([vectors[L][pair_id] for L in read_layers])
    c_attn = torch.zeros(nL, n_blocks, heads)
    c_mlp = torch.zeros(nL, n_blocks)
    embed_pool = hidden[0][0][S].to(torch.float32).mean(0)
    c_embed = (V @ embed_pool).cpu()
    total = torch.stack([hidden[L][0][S].to(torch.float32).mean(0) @ V[i] for i, L in enumerate(read_layers)]).cpu()
    for b in range(n_blocks):
        attn = out.attentions[b][0].to(torch.float32)
        vals = catcher.values[b].view(T, kv, hd).to(torch.float32)
        Vv = vals[:, torch.arange(heads, device=device) // group]
        ctx = torch.bmm(attn[:, S, :], Vv.permute(1, 0, 2))
        scale, gain = mlps.attn_scale[b], mlps.attn_gain[b]
        if scale is not None:
            ctx = ctx * scale[S][None, :, None]
        W_O = model.model.layers[b].self_attn.o_proj.weight.to(torch.float32).view(-1, heads, hd)
        head_pool = torch.einsum("hsd,ohd->ho", ctx, W_O) / len(span)
        if gain is not None:
            head_pool = head_pool * gain[None, :]
        mlp_out = mlps.out[b]
        assert mlp_out is not None
        mlp_pool = mlp_out[0][S].to(torch.float32).mean(0)
        proj_h = head_pool @ V.T
        proj_m = V @ mlp_pool
        for i, L in enumerate(read_layers):
            if b < L:
                c_attn[i, b] = proj_h[:, i].cpu()
                c_mlp[i, b] = proj_m[i].cpu()
        del attn, ctx
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "c_attn": c_attn,
            "c_mlp": c_mlp,
            "c_embed": c_embed,
            "total": total,
            "pair_id": torch.tensor(pair_id),
            "side": torch.tensor(0 if side == "concept" else 1),
        },
        str(path),
        metadata={"read_layers": ",".join(map(str, read_layers)), "pool": f"{len(span)}", "span": f"{s0}:{s1}"},
    )
    recon = c_embed + c_attn.sum((1, 2)) + c_mlp.sum(1)
    relerr = ((recon - total).abs() / total.abs().clamp_min(1e-6)).tolist()
    return {
        "item": item_id,
        "tokens": T,
        "span": f"{s0}:{s1}",
        "pooled": len(span),
        "relerr_by_layer": {str(L): round(e, 4) for L, e in zip(read_layers, relerr)},
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    parser.add_argument("--per-pole", type=int, default=4, help="stories per pole per concept")
    parser.add_argument("--chunk", type=int, default=0, help="64 pairs per chunk; -1 = all pairs in one process")
    parser.add_argument(
        "--presentation",
        default="assistant",
        choices=["assistant", "raw"],
        help="'raw' = story text alone (base models; vectors tag --raw--semantic)",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=None, help="default work[-<model short>]")
    parser.add_argument("--offline", action=BooleanOptionalAction, default=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    short = args.model.split("/")[-1]
    out_dir = args.out_dir or (Path("work") if args.model == INSTRUCT_MODEL else Path(f"work-{short}"))

    model, tokenizer = load_model(offline=args.offline, model_name=args.model)
    catcher = ValueCatcher(model.model)
    mlps = MlpCatcher(model.model)
    vectors = load_diff_vectors(device=str(model.device), tag=vector_tag_for(args.model, args.presentation))
    n_blocks = len(model.model.layers)
    read_layers = read_layers_for(n_blocks)
    if args.presentation == "raw":
        import exp_attn_interp.decomp as _self

        _self.story_ids = lambda tok, story: tok(story, add_special_tokens=True)["input_ids"]  # type: ignore[assignment]

    tbl = pq.read_table(
        GENVECTORS_ROOT / "data" / "semantic.parquet",
        columns=["pair_id", "concept_text", "antagonist_text"],
        filters=[("language_id", "==", 0)],
    )
    by_pair: dict[int, list[tuple[str, str]]] = {}
    for p, c, a in zip(tbl["pair_id"].to_pylist(), tbl["concept_text"].to_pylist(), tbl["antagonist_text"].to_pylist()):
        by_pair.setdefault(int(p), []).append((c, a))
    pairs = sorted(by_pair) if args.chunk < 0 else sorted(by_pair)[args.chunk * 64 : (args.chunk + 1) * 64]
    jobs = [
        (p, i, side, text)
        for p in pairs
        for i, (c, a) in enumerate(by_pair[p][: args.per_pole])
        for side, text in (("concept", c), ("antagonist", a))
    ]
    if args.limit:
        jobs = jobs[: args.limit]
    log.info(
        "%s: %d blocks, read layers %s; chunk %d: %d pairs x %d/pole = %d stories",
        short,
        n_blocks,
        read_layers,
        args.chunk,
        len(pairs),
        args.per_pole,
        len(jobs),
    )
    started = time.monotonic()
    for n, (p, i, side, text) in enumerate(jobs, start=1):
        r = process_story(model, tokenizer, catcher, mlps, vectors, read_layers, p, side, text, out_dir, story_index=i)
        if n <= 3 or n % 50 == 0:
            log.info("%d/%d %s", n, len(jobs), r)
    log.info("chunk %d done: %.1f min", args.chunk, (time.monotonic() - started) / 60)


if __name__ == "__main__":
    main()
