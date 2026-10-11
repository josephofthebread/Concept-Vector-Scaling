from __future__ import annotations

import glob
import json
import logging
import time
from argparse import ArgumentParser, BooleanOptionalAction
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from exp_attn_interp.config import INSTRUCT_MODEL, vector_tag_for
from exp_attn_interp.vectors import load_diff_vectors

log = logging.getLogger("reach")
TRIGGER = Path("../exp_trigger_prompts")
QUANTS = (0.5, 0.9, 1.0)


@contextmanager
def steer_from(model: Any, layer: int, delta: torch.Tensor | None, starts: list[int]):
    """Add delta[b] to the residual at the output of block layer-1 at positions >= starts[b] (right padding)."""
    if delta is None:
        yield
        return

    def hook(_m: Any, _i: Any, output: Any) -> Any:
        residual = output[0] if isinstance(output, tuple) else output
        residual = residual.clone()
        for b, s in enumerate(starts):
            residual[b, s:, :] += delta[b].to(residual.dtype)
        return (residual, *output[1:]) if isinstance(output, tuple) else residual

    handle = model.model.layers[layer - 1].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def margins(logits: torch.Tensor, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """logits [T, V] fp32 -> per-position (best marker margin, marker-set logprob, top logprob)."""
    lse = torch.logsumexp(logits, -1)
    top = logits.max(-1).values
    best = logits[:, ids].max(-1).values
    mset = torch.logsumexp(logits[:, ids], -1) - lse
    return best - top, mset, top - lse


@torch.inference_mode()
def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--alphas", default="0.25,0.5")
    parser.add_argument("--markers", type=Path, default=Path("data/markers_presence.json"))
    parser.add_argument("--out", type=Path, default=Path("work/reach"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--every", type=int, default=1, help="every n-th concept")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=640, help="truncate prompt+response tokens")
    parser.add_argument("--offline", action=BooleanOptionalAction, default=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    L = args.layer
    alphas = [float(a) for a in args.alphas.split(",")]
    tag = args.model.replace("/", "_")
    norms: dict[int, float] = {}
    for f in (f"norms-{tag}.json", f"norms-{tag}-1217.json"):
        p = TRIGGER / "work" / "steer" / f
        if p.exists():
            norms.update({int(k): float(v) for k, v in json.loads(p.read_text())["norms"].items()})
    markers = json.loads(args.markers.read_text())
    cells: set[tuple[int, int]] = set()
    for d in ("judge", "judge-1217"):
        for f in glob.glob(str(TRIGGER / "work" / d / "concept-shard-*.jsonl")):
            with open(f) as fh:
                for line in fh:
                    p, slot, lay, _al = json.loads(line)["key"].split(":")
                    if int(lay[1:]) in (L,):
                        cells.add((int(p), int(slot)))
    log.info("judged cells at row %d: %d", L, len(cells))
    base: dict[tuple[int, int], tuple[str, str]] = {}
    with (TRIGGER / "work" / f"baselines-{tag}.jsonl").open() as fh:
        for line in fh:
            d = json.loads(line)
            key = (int(d["pair_id"]), int(d["slot"]))
            if key in cells and d.get("response"):
                base[key] = (d["prompt"], d["response"])
    pairs = sorted({k for k, _ in base})[:: args.every]
    if args.limit:
        pairs = pairs[: args.limit]
    keys = [(k, s) for k, s in sorted(base) if k in set(pairs)]
    path = args.out / f"reach-L{L}.jsonl"
    done: set[str] = set()
    if path.exists():
        with path.open() as fh:
            done = {json.loads(line)["key"] for line in fh}
    keys = [ks for ks in keys if f"{ks[0]}:{ks[1]}" not in done]
    log.info("%d cells to run (%d done)", len(keys), len(done))

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.offline)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa", local_files_only=args.offline
    ).eval()
    kw: dict[str, Any] = {"enable_thinking": False} if "Qwen3" in args.model else {}
    vectors = load_diff_vectors(device="cpu", tag=vector_tag_for(args.model))[L]
    dev = model.device
    started = time.monotonic()
    with path.open("a") as sink:
        for start in range(0, len(keys), args.batch_size):
            batch = keys[start : start + args.batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": base[ks][0]}], tokenize=False, add_generation_prompt=True, **kw
                )
                for ks in batch
            ]
            p_ids = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts]
            r_ids = [tokenizer(base[ks][1], add_special_tokens=False)["input_ids"] for ks in batch]
            seqs = [(p + r)[: args.max_len] for p, r in zip(p_ids, r_ids)]
            lens = [len(s) for s in seqs]
            T = max(lens)
            input_ids = torch.full((len(batch), T), tokenizer.pad_token_id, dtype=torch.long)
            attn = torch.zeros((len(batch), T), dtype=torch.long)
            for b, s in enumerate(seqs):
                input_ids[b, : len(s)] = torch.tensor(s)
                attn[b, : len(s)] = 1
            input_ids, attn = input_ids.to(dev), attn.to(dev)
            starts = [len(p) - 1 for p in p_ids]
            V = torch.stack([vectors[k] for k, _ in batch]).to(dev)
            passes: dict[str, list[tuple[Any, ...]]] = {}
            out = model(input_ids=input_ids, attention_mask=attn).logits
            recs: list[dict[str, Any]] = []
            for b, (k, slot) in enumerate(batch):
                s, e = starts[b], lens[b] - 1
                if e - s < 4:
                    continue
                lg = out[b, s:e].float()
                rec: dict[str, Any] = {
                    "key": f"{k}:{slot}",
                    "pair_id": k,
                    "slot": slot,
                    "layer": L,
                    "n_pos": int(e - s),
                    "resp_tokens": int(len(r_ids[b])),
                }
                resp = torch.tensor(seqs[b][s + 1 : e + 1], device=dev)
                for pole in ("concept", "antagonist"):
                    ids = torch.tensor(markers[pole].get(str(k), []), device=dev)
                    if ids.numel() < 2:
                        continue
                    m, mset, toplp = margins(lg, ids)
                    rec[f"{pole}_m_q"] = [float(torch.quantile(m, q)) for q in QUANTS]
                    rec[f"{pole}_emit"] = float((m >= 0).float().mean())
                    rec[f"{pole}_lp"] = float(mset.mean())
                    rec[f"{pole}_rate"] = float(torch.isin(resp, ids).float().mean())
                rec["top_lp"] = float(toplp.mean())
                recs.append(rec)
                passes[rec["key"]] = [(b, s, e)]
            for alpha in alphas:
                for sign, pole in ((+1.0, "concept"), (-1.0, "antagonist")):
                    delta = V * (sign * alpha * norms[L])
                    with steer_from(model, L, delta, starts):
                        out_s = model(input_ids=input_ids, attention_mask=attn).logits
                    for rec in recs:
                        b, s, e = passes[rec["key"]][0]
                        k = rec["pair_id"]
                        ids = torch.tensor(markers[pole].get(str(k), []), device=dev)
                        if ids.numel() < 2:
                            continue
                        lg0, lg1 = out[b, s:e].float(), out_s[b, s:e].float()
                        m0, mset0, _ = margins(lg0, ids)
                        m1, mset1, _ = margins(lg1, ids)
                        top0 = lg0.argmax(-1)
                        lp1 = torch.log_softmax(lg1, -1).gather(1, top0[:, None])[:, 0]
                        lp0 = torch.log_softmax(lg0, -1).gather(1, top0[:, None])[:, 0]
                        a = f"{alpha:g}"
                        rec[f"{pole}_a{a}_m_q"] = [float(torch.quantile(m1, q)) for q in QUANTS]
                        rec[f"{pole}_a{a}_capture"] = float((m1 >= 0).float().mean())
                        rec[f"{pole}_a{a}_boost"] = float((m1 - m0).mean())
                        rec[f"{pole}_a{a}_boost_q"] = [float(torch.quantile(m1 - m0, q)) for q in QUANTS]
                        rec[f"{pole}_a{a}_lp"] = float(mset1.mean())
                        rec[f"{pole}_a{a}_topdrop"] = float((lp0 - lp1).mean())
                        rec[f"{pole}_a{a}_topkept"] = float((lg1.argmax(-1) == top0).float().mean())
                    del out_s
            for rec in recs:
                sink.write(json.dumps(rec) + "\n")
            sink.flush()
            if (start // args.batch_size) % 20 == 0:
                log.info("%d/%d cells, %.1f min", start + len(batch), len(keys), (time.monotonic() - started) / 60)
    log.info("done: %.1f min", (time.monotonic() - started) / 60)


if __name__ == "__main__":
    main()
