from __future__ import annotations

import json
import logging
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch

from exp_attn_interp.common import load_model, story_span
from exp_attn_interp.config import GENVECTORS_ROOT, INSTRUCT_MODEL
from exp_attn_interp.decomp import MlpCatcher
from exp_attn_interp.write import story_ids

log = logging.getLogger("polarity")


class ResidCatcher:
    """Catches the residual entering each block (input of the decoder layer) and the attention write."""

    def __init__(self, model) -> None:
        n = len(model.layers)
        self.h_in: list[torch.Tensor | None] = [None] * n
        self.attn: list[torch.Tensor | None] = [None] * n
        for i, blk in enumerate(model.layers):
            blk.register_forward_pre_hook(self._pre(i))
            blk.self_attn.register_forward_hook(self._attn(i))

    def _pre(self, i):
        def hook(_m, inputs):
            self.h_in[i] = inputs[0].detach()

        return hook

    def _attn(self, i):
        def hook(_m, _i, output):
            self.attn[i] = (output[0] if isinstance(output, tuple) else output).detach()

        return hook


@torch.inference_mode()
def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    parser.add_argument("--stories", type=int, default=96)
    parser.add_argument("--out", type=Path, default=Path("work/polarity.json"))
    parser.add_argument("--offline", action=BooleanOptionalAction, default=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model, tokenizer = load_model(offline=args.offline, model_name=args.model)
    mlps = MlpCatcher(model.model)
    rc = ResidCatcher(model.model)
    postnorm = mlps.postnorm
    tbl = pq.read_table(
        GENVECTORS_ROOT / "data" / "semantic.parquet",
        columns=["pair_id", "concept_text"],
        filters=[("language_id", "==", 0)],
    )
    seen: dict[int, str] = {}
    for p, c in zip(tbl["pair_id"].to_pylist(), tbl["concept_text"].to_pylist()):
        seen.setdefault(int(p), c)
    ks = sorted(seen)
    stories = [seen[ks[i]] for i in range(0, len(ks), max(1, len(ks) // args.stories))][: args.stories]
    B = len(model.model.layers)
    acc = {k: torch.zeros(B) for k in ("c_mlp", "c_attn", "g_mlp", "g_attn", "r_mlp", "r_attn", "c_mlp_c", "c_attn_c")}
    n = 0
    for text in stories:
        ids = story_ids(tokenizer, text)
        s0, s1 = story_span(tokenizer, ids, text)
        model(input_ids=torch.tensor([ids], device=model.device), use_cache=False)
        for b in range(B):
            hi, ai, mi = rc.h_in[b], rc.attn[b], mlps.out[b]
            assert hi is not None and ai is not None and mi is not None
            h = hi[0, s0:s1].to(torch.float32)
            a = ai[0, s0:s1].to(torch.float32)
            m = mi[0, s0:s1].to(torch.float32)
            if postnorm:
                blk = model.model.layers[b]
                a = torch.nn.functional.rms_norm(a, (a.shape[-1],), eps=blk.post_attention_layernorm.eps) * (
                    1 + blk.post_attention_layernorm.weight.to(torch.float32)
                )
            hn = h.norm(dim=-1).clamp_min(1e-6)
            hc = h - h.mean(0, keepdim=True)
            ac, mc = a - a.mean(0, keepdim=True), m - m.mean(0, keepdim=True)
            acc["c_mlp"][b] += float(((m * h).sum(-1) / (m.norm(dim=-1).clamp_min(1e-6) * hn)).mean())
            acc["c_attn"][b] += float(((a * h).sum(-1) / (a.norm(dim=-1).clamp_min(1e-6) * hn)).mean())
            acc["g_mlp"][b] += float(((m * h).sum(-1) / hn**2).mean())
            acc["g_attn"][b] += float(((a * h).sum(-1) / hn**2).mean())
            acc["r_mlp"][b] += float((m.norm(dim=-1) / hn).mean())
            acc["r_attn"][b] += float((a.norm(dim=-1) / hn).mean())
            acc["c_mlp_c"][b] += float(
                ((mc * hc).sum(-1) / (mc.norm(dim=-1).clamp_min(1e-6) * hc.norm(dim=-1).clamp_min(1e-6))).mean()
            )
            acc["c_attn_c"][b] += float(
                ((ac * hc).sum(-1) / (ac.norm(dim=-1).clamp_min(1e-6) * hc.norm(dim=-1).clamp_min(1e-6))).mean()
            )
        n += 1
    res: dict[str, Any] = {k: (v / n).tolist() for k, v in acc.items()}
    res["model"] = args.model
    res["n_stories"] = n
    c = res["c_mlp"]
    onset = next(
        (b for b in range(4, B - 2) if c[b] > 0 and c[b + 1] > 0 and c[b + 2] > 0 and min(c[max(0, b - 4) : b]) <= 0),
        -1,
    )
    res["amplification_onset_block"] = onset
    log.info("%s: MLP cos with the residual by block: %s", args.model, [round(x, 2) for x in c])
    log.info("%s: MLP cos with the CONTENT part of the residual: %s", args.model, [round(x, 2) for x in res["c_mlp_c"]])
    log.info("%s: attention cos with the residual: %s", args.model, [round(x, 2) for x in res["c_attn"]])
    log.info("%s: amplification onset block (MLP cos turns positive and stays): %d", args.model, onset)
    args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
