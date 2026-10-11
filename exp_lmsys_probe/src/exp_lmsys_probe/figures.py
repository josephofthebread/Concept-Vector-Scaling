from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from xml.sax.saxutils import escape

import torch

from .config import CAP
from .model import load
from .sources import LMSYS, exchange_template

log = logging.getLogger("figures")

LAYER = 23
CLIP = 3.0


def _emit(
    parts: list[str],
    x: float,
    y: float,
    toks: list[str],
    zs: list[float],
    pad: float,
    max_w: float,
    cw: float,
    lh: float,
    fs: float,
) -> tuple[float, float]:
    for raw, z in zip(toks, zs):
        t = max(-1.0, min(1.0, z / CLIP))
        if t >= 0:
            r, g, b = (round(255 - 77 * t), round(255 - 231 * t), round(255 - 212 * t))
        else:
            u = -t
            r, g, b = (round(255 - 222 * u), round(255 - 153 * u), round(255 - 83 * u))
        fg = "#fff" if (0.299 * r + 0.587 * g + 0.114 * b) < 130 else "#111"
        for si, seg in enumerate(raw.split("\n")):
            if si > 0:
                x, y = pad, y + lh
            if seg == "":
                continue
            w = len(seg) * cw
            if x + w > max_w and x > pad:
                x, y = pad, y + lh
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{lh:.1f}" fill="rgb({r},{g},{b})"/>')
            parts.append(
                f'<text x="{x:.1f}" y="{y + fs:.1f}" font-size="{fs}" fill="{fg}" '
                f'font-family="monospace" xml:space="preserve">{escape(seg)}</text>'
            )
            x += w
    return x, y


def render(
    title: str, legend: str, utoks: list[str], uz: list[float], atoks: list[str], az: list[float], out: Path
) -> None:
    cw, lh, fs, pad, max_w = 7.6, 20.0, 13.0, 16.0, 1180.0
    parts: list[str] = []
    y0 = 58.0
    x, y = _emit(parts, pad, y0, ["User: "] + utoks, [0.0] + uz, pad, max_w, cw, lh, fs)
    y += lh * 1.4
    parts.append(
        f'<line x1="{pad}" y1="{y - 8:.0f}" x2="{max_w:.0f}" y2="{y - 8:.0f}" stroke="#ccc" stroke-dasharray="3,3"/>'
    )
    x, y = _emit(parts, pad, y, ["Assistant: "] + atoks, [0.0] + az, pad, max_w, cw, lh, fs)
    height = y + lh + pad
    head = (
        f'<text x="{pad}" y="24" font-size="15" font-family="sans-serif" font-weight="bold">{escape(title)}</text>'
        f'<text x="{pad}" y="44" font-size="11.5" font-family="sans-serif" fill="#666">{escape(legend)}</text>'
    )
    width = max_w + 2 * pad
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}"><rect width="100%" height="100%" fill="#fff"/>'
        f"{head}{''.join(parts)}</svg>"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = ArgumentParser()
    parser.add_argument("--work", type=Path, default=Path("work"))
    parser.add_argument("--features", type=Path, default=Path("data/features.json"))
    parser.add_argument("--out", type=Path, default=Path("work/figures"))
    args = parser.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    figman = json.loads((args.work / "figman.json").read_text())
    feats = json.loads(args.features.read_text())["features"]
    tasks: dict[str, list[tuple[int, int]]] = {}
    for entry in figman:
        for rank, cid in enumerate(entry["top3"], 1):
            tasks.setdefault(cid, []).append((entry["pair_id"], rank))
    log.info("%d features, %d conversations to render", len(figman), len(tasks))
    if not tasks:
        raise SystemExit("no accepted conversations to render")

    tokenizer, model, vectors, mlayers, _ = load()
    vi = mlayers.index(LAYER)
    head_ids, mid_ids, tail_ids = exchange_template(tokenizer)

    root = Path(snapshot_download(LMSYS, repo_type="dataset"))
    want = set(tasks)
    text: dict[str, tuple[str, str]] = {}
    for sh in sorted(root.rglob("*.parquet")):
        table = pq.read_table(sh, columns=["conversation_id", "conversation"])
        for cid, turns in zip(table.column("conversation_id").to_pylist(), table.column("conversation").to_pylist()):
            if cid in want and cid not in text:
                oi = next((i for i, t in enumerate(turns) if t["role"] == "user"), None)
                if oi is not None:
                    rep = next((t["content"] for t in turns[oi + 1 :] if t["role"] == "assistant"), "")
                    text[cid] = (turns[oi]["content"], rep)
        if len(text) == len(want):
            break

    n = 0
    with torch.inference_mode():
        for cid, needs in tasks.items():
            if cid not in text:
                continue
            uc, ac = text[cid]
            u = tokenizer(uc, add_special_tokens=False)["input_ids"][:CAP]
            a = tokenizer(ac, add_special_tokens=False)["input_ids"][:CAP]
            if not u or not a:
                continue
            input_ids = torch.tensor([head_ids + u + mid_ids + a + tail_ids], device=model.device)
            hs = model(input_ids=input_ids, output_hidden_states=True, use_cache=False).hidden_states[LAYER][0]
            us, as_ = len(head_ids), len(head_ids) + len(u) + len(mid_ids)
            content = torch.cat([hs[us : us + len(u)], hs[as_ : as_ + len(a)]]).float()
            unit = torch.nn.functional.normalize(content, dim=-1)
            cos = unit @ vectors[vi].T
            z = (cos - cos.mean(0)) / cos.std(0).clamp_min(1e-6)
            up = [tokenizer.decode([t]) for t in u]
            ap = [tokenizer.decode([t]) for t in a]
            for pid, rank in needs:
                f = feats[pid]
                zu = z[: len(u), pid].cpu().tolist()
                za = z[len(u) :, pid].cpu().tolist()
                title = f"{f['concept']}  ‖  {f['antagonist']}   (pair {pid})"
                legend = f"red = {f['concept']}   ·   blue = {f['antagonist']}   ·   layer {LAYER} z-scored cosine"
                render(title, legend, up, zu, ap, za, args.out / f"pair{pid:04d}-r{rank}.svg")
                n += 1
    log.info("rendered %d SVGs -> %s", n, args.out)


if __name__ == "__main__":
    main()
