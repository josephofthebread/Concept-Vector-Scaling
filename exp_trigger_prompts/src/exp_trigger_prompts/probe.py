from __future__ import annotations

import json
import logging
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, cast

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger("probe")

VEC_ROOT = Path("../exp_gendata/work/vectors")
VEC_TAG = {
    "Qwen/Qwen3-8B": "Qwen_Qwen3-8B--assistant--semantic",
    "google/gemma-2-9b-it": "google_gemma-2-9b-it--assistant--semantic",
    "unsloth/Meta-Llama-3.1-8B-Instruct": "unsloth_Meta-Llama-3.1-8B-Instruct--assistant--semantic",
}
CLIP = 3.0
TOPK = 5


def slug(text: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", text.lower())).strip("-") or "x"


def find_subsequence(haystack: list[int], needle: list[int], start: int = 0) -> int:
    if not needle:
        return -1
    for i in range(start, len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return -1


def locate_span(ids: list[int], text: str, tok, start: int = 0) -> tuple[int, int]:
    """Token span of `text` inside `ids`, retrying without a leading token that merged with template."""
    piece = tok(text, add_special_tokens=False)["input_ids"]
    off = find_subsequence(ids, piece, start)
    if off < 0 and len(piece) > 1:
        piece = piece[1:]
        off = find_subsequence(ids, piece, start)
    if off < 0:
        return (-1, -1)
    return (off, off + len(piece))


def render_svg(
    tokens: list[str], zvals: list[float], resp_start: int, title: str, sublines: list[str], out_path: Path
) -> None:
    """Lay tokens out as wrapped, background-shaded monospace text under a self-contained header."""
    from xml.sax.saxutils import escape

    cw, lh, fs = 7.6, 20.0, 13.0
    pad = 16.0
    top = 30.0 + len(sublines) * 16.0 + 12.0
    max_w = 1180.0
    x, y = pad, top
    parts: list[str] = []
    width = max_w + 2 * pad

    def newline() -> None:
        nonlocal x, y
        x = pad
        y += lh

    for idx, (raw, z) in enumerate(zip(tokens, zvals)):
        if idx == resp_start and resp_start > 0:
            if x > pad:
                newline()
            parts.append(
                f'<line x1="{pad}" y1="{y - 4:.0f}" x2="{max_w:.0f}" y2="{y - 4:.0f}" '
                f'stroke="#888" stroke-dasharray="3,3"/>'
            )
            parts.append(
                f'<text x="{pad}" y="{y + fs - 4:.0f}" font-size="10" fill="#888" '
                f'font-family="monospace">assistant reply &#8595;</text>'
            )
            y += lh
        t = max(-1.0, min(1.0, z / CLIP))
        if t >= 0:
            r, g, b = (round(255 - 77 * t), round(255 - 231 * t), round(255 - 212 * t))
        else:
            u = -t
            r, g, b = (round(255 - 222 * u), round(255 - 153 * u), round(255 - 83 * u))
        dark = (0.299 * r + 0.587 * g + 0.114 * b) < 130
        fg = "#fff" if dark else "#111"
        for seg_i, seg in enumerate(raw.split("\n")):
            if seg_i > 0:
                newline()
            if seg == "":
                continue
            w = len(seg) * cw
            if x + w > max_w and x > pad:
                newline()
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{lh:.1f}" fill="rgb({r},{g},{b})"/>')
            parts.append(
                f'<text x="{x:.1f}" y="{y + fs:.1f}" font-size="{fs}" fill="{fg}" '
                f'font-family="monospace" xml:space="preserve">{escape(seg)}</text>'
            )
            x += w
    height = y + lh + pad
    head_parts = [
        f'<text x="{pad}" y="20" font-size="15" font-family="sans-serif" font-weight="bold">{escape(title)}</text>'
    ]
    for i, line in enumerate(sublines):
        head_parts.append(
            f'<text x="{pad}" y="{38 + i * 16:.0f}" font-size="11.5" '
            f'font-family="sans-serif" fill="#555">{escape(line)}</text>'
        )
    head = "".join(head_parts)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}">'
        f'<rect width="100%" height="100%" fill="#fff"/>{head}{"".join(parts)}</svg>'
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(VEC_TAG))
    parser.add_argument("--prefills", type=Path, default=Path("work/heldout_prefills_qwen.json"))
    parser.add_argument("--features", type=Path, default=Path("data/features.json"))
    parser.add_argument("--out-root", type=Path, default=Path("work/PROBING_2"))
    args = parser.parse_args()

    names = {f["pair_id"]: (f["concept"], f["antagonist"]) for f in json.loads(args.features.read_text())["features"]}
    entries = json.loads(args.prefills.read_text())
    model_dir = args.model.split("/")[-1]
    thinking: dict[str, Any] = {"enable_thinking": False} if "Qwen3" in args.model else {}

    attn = "eager" if "gemma" in args.model.lower() else "sdpa"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=attn, device_map="auto"
    )
    model.eval()
    device = model.device
    layer = round(model.config.num_hidden_layers / 2)

    with safe_open(VEC_ROOT / VEC_TAG[args.model] / "diff.safetensors", framework="pt") as handle:
        diff = handle.get_tensor("diff")
    assert diff.shape[0] == model.config.num_hidden_layers + 1, (diff.shape, model.config.num_hidden_layers)
    directions = diff[layer].float().to(device)
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    log.info(
        "%s: %d layers, reading layer %d, %d features",
        args.model,
        model.config.num_hidden_layers,
        layer,
        directions.shape[0],
    )

    summary = []
    for entry in entries:
        messages = entry["conversation"]
        enc = cast(
            "Any",
            tok.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False, return_tensors="pt", return_dict=True, **thinking
            ),
        )  # type: ignore[arg-type]
        ids_t = enc["input_ids"]
        ids = cast("list[int]", ids_t[0].tolist())
        with torch.no_grad():
            out = model(ids_t.to(device), output_hidden_states=True)
        hidden = out.hidden_states[layer][0].float()
        hidden = hidden / hidden.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        cos = hidden @ directions.T

        p0, p1 = locate_span(ids, messages[0]["content"], tok)
        r0, r1 = locate_span(ids, messages[1]["content"], tok, start=max(p1, 0))
        content = torch.zeros(len(ids), dtype=torch.bool)
        if p0 >= 0:
            content[p0:p1] = True
        if r0 >= 0:
            content[r0:r1] = True
        if content.sum() < 2 or r0 < 0:
            log.warning(
                "pair %s %s: span recovery weak (content=%d, resp=%d); using all tokens",
                entry["pair_id"],
                entry["pole"],
                int(content.sum()),
                r0,
            )
            content[:] = True
            r0 = r0 if r0 >= 0 else 0

        mu = cos[content].mean(0)
        sigma = cos[content].std(0).clamp_min(1e-6)
        z = (cos - mu) / sigma
        score = z[r0:r1].mean(0) if r1 > r0 else z[content].mean(0)

        pos = torch.topk(score, TOPK).indices.tolist()
        neg = torch.topk(-score, TOPK).indices.tolist()
        display = [cast("str", tok.decode([i])) for i in ids]
        pair_dir = f"{entry['pair_id']}--{slug(entry['concept'])}--{slug(entry['antagonist'])}"

        selected = [("pos", rank, pid) for rank, pid in enumerate(pos, 1)] + [
            ("neg", rank, pid) for rank, pid in enumerate(neg, 1)
        ]
        n_layers = model.config.num_hidden_layers
        for sign, rank, pid in selected:
            fc, fa = names[pid]
            zcol = z[:, pid].tolist()
            title = f"Feature: {fc}  ‖  {fa}   (pair {pid})"
            sublines = [
                f"color:  red = {fc}   |   blue = {fa}   (z-scored cosine, white ≈ 0)",
                f"conversation: pair {entry['pair_id']}  {entry['concept']} ‖ {entry['antagonist']}"
                f"   —   {entry['pole'].upper()} pole prompt (Qwen prefill)",
                f"model: {model_dir}   ·   layer {layer}/{n_layers} (½ depth)   ·   "
                f"{'positive' if sign == 'pos' else 'negative'} rank {rank}   ·   "
                f"mean-z over reply = {score[pid].item():+.2f}",
            ]
            out_path = (
                args.out_root
                / pair_dir
                / model_dir
                / entry["pole"]
                / f"{sign}{rank}-{pid:04d}-{slug(fc + '--' + fa)}.svg"
            )
            render_svg(display, zcol, r0, title, sublines, out_path)

        summary.append(
            {
                "pair_id": entry["pair_id"],
                "pole": entry["pole"],
                "tokens": len(ids),
                "reply_span": [int(r0), int(r1)],
                "top_pos": [[pid, round(score[pid].item(), 3), names[pid]] for pid in pos],
                "top_neg": [[pid, round(score[pid].item(), 3), names[pid]] for pid in neg],
            }
        )
        log.info(
            "pair %s [%s]: %d tokens, reply[%d:%d], top+ %s",
            entry["pair_id"],
            entry["pole"],
            len(ids),
            r0,
            r1,
            [names[p][0] for p in pos],
        )

    sm = args.out_root / "_selection" / f"{model_dir}.json"
    sm.parent.mkdir(parents=True, exist_ok=True)
    sm.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    log.info("done %s: %d conversations -> %s", args.model, len(entries), args.out_root)


if __name__ == "__main__":
    main()
