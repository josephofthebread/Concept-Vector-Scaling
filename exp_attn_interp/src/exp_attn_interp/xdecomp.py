from __future__ import annotations

import logging
import time
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from exp_attn_interp.common import story_span
from exp_attn_interp.config import GENDATA_ROOT
from exp_attn_interp.decomp import POOL_MIN, POOL_SKIP

log = logging.getLogger("xdecomp")
STORY_PROMPT = "Generate a short story."


class Catcher:
    def __init__(self, model: Any) -> None:
        self.layers = model.model.layers
        self.mlp_out: list[torch.Tensor | None] = [None] * len(self.layers)
        self.final_raw: torch.Tensor | None = None
        for b, block in enumerate(self.layers):
            target = getattr(block, "post_feedforward_layernorm", None) or block.mlp
            target.register_forward_hook(self._mlp(b))
        model.model.norm.register_forward_pre_hook(self._norm)

    def _mlp(self, b: int):
        def hook(_m, _i, out):
            self.mlp_out[b] = out.detach()

        return hook

    def _norm(self, _m, inputs):
        self.final_raw = inputs[0].detach()


def render(tokenizer, story: str, presentation: str) -> list[int]:
    if presentation == "raw":
        return tokenizer(story, add_special_tokens=True)["input_ids"]
    messages = [{"role": "user", "content": STORY_PROMPT}, {"role": "assistant", "content": story}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
    except TypeError:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


@torch.inference_mode()
def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--presentation", default="assistant", choices=["assistant", "raw"])
    parser.add_argument("--every", type=int, default=4)
    parser.add_argument("--per-pole", type=int, default=8)
    parser.add_argument("--language", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--offline", action=BooleanOptionalAction, default=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    short = args.model.split("/")[-1]
    out_dir = args.out or Path(f"work-x/{short}/xdecomp")
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.offline, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=args.offline,
        device_map={"": "cuda"},
    ).eval()
    catcher = Catcher(model)
    dev = model.device
    B = len(model.model.layers)
    hid = model.config.hidden_size
    tbl = pq.read_table(
        GENDATA_ROOT / "data" / "semantic.parquet",
        columns=["pair_id", "concept_text", "antagonist_text"],
        filters=[("language_id", "==", args.language)],
    )
    by_pair: dict[int, list[tuple[str, str]]] = {}
    for p, c, a in zip(tbl["pair_id"].to_pylist(), tbl["concept_text"].to_pylist(), tbl["antagonist_text"].to_pylist()):
        by_pair.setdefault(int(p), []).append((c, a))
    pairs = sorted(by_pair)[:: args.every]
    S = args.per_pole
    started = time.monotonic()
    for n, k in enumerate(pairs):
        path = out_dir / f"pair-{k:04d}.npz"
        if path.exists():
            continue
        means = torch.zeros(2, S, B + 1, hid, device=dev)
        mlps = torch.zeros(2, S, B, hid, device=dev)
        for j, (c, a) in enumerate(by_pair[k][:S]):
            for side, text in ((0, c), (1, a)):
                ids = render(tokenizer, text, args.presentation)
                s0, s1 = story_span(tokenizer, ids, text)
                span = list(range(s0 + POOL_SKIP, s1)) if (s1 - s0 - POOL_SKIP) >= POOL_MIN else list(range(s0, s1))
                Sx = torch.tensor(span, device=dev)
                out = model(input_ids=torch.tensor([ids], device=dev), output_hidden_states=True, use_cache=False)
                hs = out.hidden_states
                for r in range(B):
                    means[side, j, r] = hs[r][0, Sx].to(torch.float32).mean(0)
                means[side, j, B] = catcher.final_raw[0, Sx].to(torch.float32).mean(0)
                for b in range(B):
                    mlps[side, j, b] = catcher.mlp_out[b][0, Sx].to(torch.float32).mean(0)
                del out, hs
        pole_resid = means.mean(1)
        pole_mlp = mlps.mean(1)
        v = pole_resid[0] - pole_resid[1]
        Mv = pole_mlp[0] - pole_mlp[1]
        Av = (pole_resid[0, 1:] - pole_resid[0, :-1] - pole_mlp[0]) - (
            pole_resid[1, 1:] - pole_resid[1, :-1] - pole_mlp[1]
        )
        vl = v[B] / v[B].norm().clamp_min(1e-8)
        share_mlp_last = (Mv @ vl) / v[B].norm().clamp_min(1e-8)
        share_attn_last = (Av @ vl) / v[B].norm().clamp_min(1e-8)
        vr = v[:B] / v[:B].norm(dim=-1, keepdim=True).clamp_min(1e-8)
        gain_mlp_running = (Mv * vr).sum(-1) / v[:B].norm(dim=-1).clamp_min(1e-8)
        cos_adjacent = (vr[1:] * vr[:-1]).sum(-1)
        h = S // 2
        dA = (means[0, :h] - means[1, :h]).mean(0)
        dA = dA / dA.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        pc = (means[0, h:] * dA[None]).sum(-1).cpu().numpy()
        pa = (means[1, h:] * dA[None]).sum(-1).cpu().numpy()
        gap = pc.mean(0) - pa.mean(0)
        spread = np.sqrt(0.5 * (pc.var(0, ddof=1) + pa.var(0, ddof=1)) + 1e-12)
        auc_list = []
        for r in range(B + 1):
            d = pc[:, r][:, None] - pa[:, r][None, :]
            auc_list.append(float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size))
        auc = np.array(auc_list)
        dB = (means[0, h:] - means[1, h:]).mean(0)
        dB = dB / dB.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        repro = (dA * dB).sum(-1).cpu().numpy()
        np.savez(
            str(path),
            pair=np.array(k),
            n_blocks=np.array(B),
            pole_resid=pole_resid.to(torch.bfloat16).view(torch.int16).cpu().numpy(),
            pole_mlp=pole_mlp.to(torch.bfloat16).view(torch.int16).cpu().numpy(),
            share_mlp_last=share_mlp_last.cpu().numpy(),
            share_attn_last=share_attn_last.cpu().numpy(),
            gain_mlp_running=gain_mlp_running.cpu().numpy(),
            cos_adjacent=cos_adjacent.cpu().numpy(),
            story_gap=gap,
            story_spread=spread,
            story_snr=gap / spread,
            story_auc=auc,
            repro=repro,
            norm_v=v.norm(dim=-1).cpu().numpy(),
            norm_resid=pole_resid.mean(0).norm(dim=-1).cpu().numpy(),
        )
        if n < 3 or n % 25 == 0:
            el = (time.monotonic() - started) / 60
            log.info(
                "%d/%d pair %d: %.1f min; MLP share along v(last) neg through %s",
                n,
                len(pairs),
                k,
                el,
                [b for b in range(B) if share_mlp_last[b] < 0][-1:] or None,
            )
    log.info("done %d pairs, %.1f min", len(pairs), (time.monotonic() - started) / 60)


if __name__ == "__main__":
    main()
