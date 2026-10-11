from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from xml.sax.saxutils import escape

import torch

from .config import CAP
from .model import load
from .sources import LMSYS, OPENWEBTEXT, exchange_template

log = logging.getLogger("render")

LAYER = 23
CLIP = 3.0


def render(pieces: list[str], zvals: list[float], title: str, sub: str, out: Path) -> None:
    cw, lh, fs, pad, max_w = 7.6, 20.0, 13.0, 16.0, 1180.0
    x, y = pad, 60.0
    parts: list[str] = []
    for raw, z in zip(pieces, zvals):
        t = max(-1.0, min(1.0, z / CLIP))
        if t >= 0:
            r, g, b = (round(255 - 77 * t), round(255 - 231 * t), round(255 - 212 * t))
        else:
            u = -t
            r, g, b = (round(255 - 222 * u), round(255 - 153 * u), round(255 - 83 * u))
        fg = "#fff" if (0.299 * r + 0.587 * g + 0.114 * b) < 130 else "#111"
        for seg_i, seg in enumerate(raw.split("\n")):
            if seg_i > 0:
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
    height = y + lh + pad
    head = (
        f'<text x="{pad}" y="22" font-size="14" font-family="sans-serif" font-weight="bold">'
        f"{escape(title)}</text>"
        f'<text x="{pad}" y="40" font-size="11" font-family="sans-serif" fill="#666">{escape(sub)}</text>'
    )
    width = max_w + 2 * pad
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}"><rect width="100%" height="100%" fill="#fff"/>'
        f"{head}{''.join(parts)}</svg>"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg)


def fetch_lmsys(wanted: set[str]) -> dict[str, tuple[str, str]]:
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(LMSYS, repo_type="dataset"))
    out: dict[str, tuple[str, str]] = {}
    for sh in sorted(root.rglob("*.parquet")):
        t = pq.read_table(sh, columns=["conversation_id", "conversation"])
        for cid, turns in zip(t.column("conversation_id").to_pylist(), t.column("conversation").to_pylist()):
            if cid in wanted and cid not in out:
                oi = next((i for i, x in enumerate(turns) if x["role"] == "user"), None)
                if oi is None:
                    continue
                rep = next((x["content"] for x in turns[oi + 1 :] if x["role"] == "assistant"), "")
                out[cid] = (turns[oi]["content"], rep)
        if len(out) == len(wanted):
            break
    return out


def prepare(corpus: str, tok, cid: str, texts: dict, tmpl: tuple) -> tuple[list[int], list[tuple[int, int]], list[str]]:
    """Build (input_ids, content_spans, boundary_labels) for one example."""
    if corpus == "openwebtext":
        toks = tok(texts[cid], add_special_tokens=False)["input_ids"][:CAP]
        bos = tok.bos_token_id
        ids = ([bos] + toks) if bos is not None else list(toks)
        cstart = 1 if bos is not None else 0
        return ids, [(cstart, len(toks))], [""]
    head, middle, tail = tmpl
    user, asst = texts[cid]
    u = tok(user, add_special_tokens=False)["input_ids"][:CAP]
    a = tok(asst, add_special_tokens=False)["input_ids"][:CAP]
    ids = head + u + middle + a + tail
    spans = [(len(head), len(u)), (len(head) + len(middle) + len(u), len(a))]
    return ids, spans, ["user ↓ ", "assistant ↓ "]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = ArgumentParser()
    parser.add_argument("--work", type=Path, default=Path("work"))
    parser.add_argument("--features", type=Path, default=Path("data/features.json"))
    parser.add_argument("--figures", type=Path, default=Path("work/figures"))
    parser.add_argument("--layer", type=int, default=LAYER)
    args = parser.parse_args()

    meta = json.loads((args.work / "meta.json").read_text())
    corpus, model_name = meta.get("corpus", "lmsys"), meta["model"]
    figman = json.loads((args.work / "figman.json").read_text())
    feats = json.loads(args.features.read_text())["features"]

    tasks: dict[str, list[tuple[int, int]]] = {}
    for f in figman:
        for rank, cid in enumerate(f["top3"], 1):
            tasks.setdefault(cid, []).append((f["pair_id"], rank))
    if not tasks:
        raise SystemExit("no accepted examples to render")
    log.info("corpus=%s model=%s: %d examples to render", corpus, model_name, len(tasks))

    tokenizer, model, vectors, mlayers, _ = load(model_name)
    vi = mlayers.index(args.layer)
    tmpl = exchange_template(tokenizer) if corpus == "lmsys" else ()
    if corpus == "lmsys":
        texts: dict = {c: fetch_lmsys(set(tasks))[c] for c in tasks}
    else:
        from datasets import load_dataset

        ds = load_dataset(OPENWEBTEXT, split="train")
        texts = {c: ds[int(c.split("-")[1])]["text"] for c in set(tasks)}

    n = 0
    with torch.inference_mode():
        for cid, jobs in tasks.items():
            if cid not in texts:
                continue
            ids, spans, labels = prepare(corpus, tokenizer, cid, texts, tmpl)
            out = model(input_ids=torch.tensor([ids], device=model.device), output_hidden_states=True, use_cache=False)
            hidden = out.hidden_states[args.layer][0].float()
            idx, pieces = [], []
            for (st, ln), lab in zip(spans, labels):
                if lab:
                    pieces.append("\n" + lab)
                    idx.append(-1)
                for t in range(st, st + ln):
                    idx.append(t)
                    pieces.append(tokenizer.decode([ids[t]]))
            content = [i for i in idx if i >= 0]
            unit = torch.nn.functional.normalize(hidden[content], dim=-1)
            cosm = unit @ vectors[vi].T
            z = (cosm - cosm.mean(0)) / cosm.std(0).clamp_min(1e-6)
            for pid, rank in jobs:
                f = feats[pid]
                zc = {i: v for i, v in zip(content, z[:, pid].cpu().tolist())}
                zvals = [zc.get(i, 0.0) for i in idx]
                title = f"{f['concept']}  ‖  {f['antagonist']}   (pair {pid})"
                sub = (
                    f"red = {f['concept']}  ·  blue = {f['antagonist']}  ·  per-token z-scored cosine, "
                    f"layer {args.layer}  ·  {corpus} {cid}"
                )
                render(pieces, zvals, title, sub, args.figures / f"pair{pid:04d}-r{rank}.svg")
                n += 1
    log.info("rendered %d figures -> %s", n, args.figures)


if __name__ == "__main__":
    main()
