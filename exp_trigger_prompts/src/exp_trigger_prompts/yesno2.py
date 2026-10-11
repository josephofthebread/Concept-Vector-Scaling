from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, cast

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from .probe import VEC_ROOT, VEC_TAG, locate_span

log = logging.getLogger("yesno2")

MODEL = "Qwen/Qwen3-8B"
SUFFIX = ' Reply with only "Yes" or "No".'
PAIRS = [
    ("g1", "Is 8 greater than 5?", "Yes"),
    ("g2", "Is 5 greater than 8?", "No"),
    ("s1", "Is 8 > 5?", "Yes"),
    ("s2", "Is 5 > 8?", "No"),
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("work/yesno/yesno_gt.json"))
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto"
    )
    model.eval()
    device = model.device
    n_layers = model.config.num_hidden_layers
    layers = list(range(n_layers + 1))
    with safe_open(VEC_ROOT / VEC_TAG[MODEL] / "diff.safetensors", framework="pt") as handle:
        diff = handle.get_tensor("diff").float().to(device)
    unit = diff / diff.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def tokid(s: str) -> int:
        ids = cast("list[int]", tok.encode(s, add_special_tokens=False))
        return ids[0] if ids else -1

    yes_ids = {tokid(s) for s in ["Yes", " Yes", "yes", " yes", "YES"]} - {-1}
    no_ids = {tokid(s) for s in ["No", " No", "no", " no", "NO"]} - {-1}

    out_chats = []
    for stem, question, correct in PAIRS:
        for answer in ("Yes", "No"):
            messages = [{"role": "user", "content": question + SUFFIX}, {"role": "assistant", "content": answer}]
            enc = cast(
                "Any",
                tok.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_tensors="pt",
                    return_dict=True,
                    enable_thinking=False,
                ),
            )  # type: ignore[arg-type]
            ids_t = enc["input_ids"]
            ids = cast("list[int]", ids_t[0].tolist())
            up0, up1 = locate_span(ids, messages[0]["content"], tok)
            a0, a1 = locate_span(ids, answer, tok, start=max(up1, 0))
            content = torch.zeros(len(ids), dtype=torch.bool)
            if up0 >= 0:
                content[up0:up1] = True
            if a0 >= 0:
                content[a0:a1] = True

            with torch.no_grad():
                out = model(ids_t.to(device), output_hidden_states=True)
            probs = torch.softmax(out.logits[0, a0 - 1].float(), dim=-1)
            top = probs.topk(6)
            decision = {
                "p_yes": float(sum(probs[i].item() for i in yes_ids)),
                "p_no": float(sum(probs[i].item() for i in no_ids)),
                "top": [
                    [tok.convert_ids_to_tokens([int(i)])[0], round(float(p), 4)]
                    for p, i in zip(top.values.tolist(), top.indices.tolist())
                ],
            }
            gaps: dict[str, list[float]] = {}
            doms: dict[str, list[int]] = {}
            for layer in layers:
                hs = out.hidden_states[layer][0].float()
                h_unit = hs / hs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                cos = h_unit @ unit[layer].T
                block = cos[content]
                z = (cos - block.mean(0)) / block.std().clamp_min(1e-6)
                top2 = z.abs().topk(2, dim=-1)
                gaps[str(layer)] = [round(g, 4) for g in (top2.values[:, 0] - top2.values[:, 1]).cpu().tolist()]
                doms[str(layer)] = top2.indices[:, 0].cpu().tolist()
            out_chats.append(
                {
                    "stem": stem,
                    "answer": answer,
                    "correct": correct,
                    "question": question,
                    "pieces": tok.convert_ids_to_tokens(ids),
                    "answer_idx": a0,
                    "decision": decision,
                    "gaps": gaps,
                    "doms": doms,
                }
            )
            log.info("%s answer=%s: p_yes=%.3f p_no=%.3f", stem, answer, decision["p_yes"], decision["p_no"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_chats, ensure_ascii=False))
    log.info("wrote %s (%d chats)", args.out, len(out_chats))


if __name__ == "__main__":
    main()
