from __future__ import annotations

import json
import logging
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import pyarrow.parquet as pq
import torch
from safetensors.torch import save_file

from exp_attn_interp.common import ValueCatcher, load_model, sink_mask, story_span
from exp_attn_interp.config import GENVECTORS_ROOT, INSTRUCT_MODEL, PAIRS, vector_tag_for
from exp_attn_interp.decomp import MlpCatcher, read_layers_for
from exp_attn_interp.vectors import load_diff_vectors
from exp_attn_interp.write import story_ids

log = logging.getLogger("dyn")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    parser.add_argument("--pairs", required=True, help="comma list of concept ids")
    parser.add_argument("--per-pole", type=int, default=2)
    parser.add_argument("--markers", type=Path, default=Path("data/markers_unique.json"))
    parser.add_argument("--read-frac", type=float, default=1.0, help="1.0 = last raw residual")
    parser.add_argument("--out-dir", type=Path, default=Path("work/dyn"))
    parser.add_argument("--offline", action=BooleanOptionalAction, default=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model, tokenizer = load_model(offline=args.offline, model_name=args.model)
    catcher = ValueCatcher(model.model)
    mlps = MlpCatcher(model.model)
    vectors = load_diff_vectors(device=str(model.device), tag=vector_tag_for(args.model))
    n_blocks = len(model.model.layers)
    rl = read_layers_for(n_blocks)
    L = rl[-1] if args.read_frac >= 1.0 else rl[min(len(rl) - 1, int(round(args.read_frac * 5)))]
    markers = json.loads(args.markers.read_text())["concept"]
    marker_ids = [set(markers[str(k)]) for k in range(PAIRS)]
    heads, kv, hd = catcher.heads, catcher.kv_heads, catcher.head_dim
    group = heads // kv
    device = model.device

    tbl = pq.read_table(
        GENVECTORS_ROOT / "data" / "semantic.parquet",
        columns=["pair_id", "concept_text", "antagonist_text"],
        filters=[("language_id", "==", 0)],
    )
    by_pair: dict[int, list[tuple[str, str]]] = {}
    for p, c, a in zip(tbl["pair_id"].to_pylist(), tbl["concept_text"].to_pylist(), tbl["antagonist_text"].to_pylist()):
        by_pair.setdefault(int(p), []).append((c, a))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for k in [int(x) for x in args.pairs.split(",")]:
        v = vectors[L][k]
        for i, (c, a) in enumerate(by_pair[k][: args.per_pole]):
            for side, story in (("concept", c), ("antagonist", a)):
                ids = story_ids(tokenizer, story)
                T = len(ids)
                s0, s1 = story_span(tokenizer, ids, story)
                with torch.inference_mode():
                    out = model(
                        input_ids=torch.tensor([ids], device=device),
                        output_attentions=True,
                        output_hidden_states=True,
                        use_cache=False,
                    )
                hidden = list(out.hidden_states)
                mask = sink_mask(list(out.attentions), hidden)
                traj = torch.stack([hidden[ell][0].to(torch.float32) @ v for ell in range(n_blocks)]).T.cpu()
                e = (hidden[0][0].to(torch.float32) @ v).cpu()
                cc = torch.zeros(T, n_blocks, heads)
                mm = torch.zeros(T, n_blocks)
                for b in range(L):
                    attn = out.attentions[b][0].to(torch.float32)
                    raw_v = catcher.values[b]
                    assert raw_v is not None
                    vals = raw_v.view(T, kv, hd).to(torch.float32)
                    Vv = vals[:, torch.arange(heads, device=device) // group]
                    ctx = torch.bmm(attn, Vv.permute(1, 0, 2))
                    W_O = model.model.layers[b].self_attn.o_proj.weight.to(torch.float32).view(-1, heads, hd)
                    wv = torch.einsum("ohd,o->hd", W_O, v)
                    cc[:, b] = torch.einsum("htd,hd->th", ctx, wv).cpu()
                    mo = mlps.out[b]
                    assert mo is not None
                    mm[:, b] = (mo[0].to(torch.float32) @ v).cpu()
                    del attn, ctx
                is_marker = torch.tensor([1 if tok in marker_ids[k] else 0 for tok in ids], dtype=torch.int8)
                item = f"pair{k:04d}-{side}-{i}"
                save_file(
                    {
                        "c": cc.half(),
                        "m": mm.half(),
                        "e": e,
                        "traj": traj.half().contiguous(),
                        "marker": is_marker,
                        "sink": mask.cpu().to(torch.int8),
                    },
                    str(args.out_dir / f"{item}.safetensors"),
                    metadata={
                        "pair_id": str(k),
                        "side": side,
                        "read_layer": str(L),
                        "span": f"{s0}:{s1}",
                        "model": args.model,
                    },
                )
                recon = float((e + cc.sum((1, 2)).float() + mm.sum(1).float())[s0:s1].mean())
                log.info(
                    "%s: T=%d markers=%d  mean traj@L%d=%.3f recon=%.3f",
                    item,
                    T,
                    int(is_marker.sum()),
                    L,
                    float(traj[s0:s1, L].float().mean()),
                    recon,
                )


if __name__ == "__main__":
    main()
