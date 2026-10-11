from __future__ import annotations

import json
import logging
import re
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from xml.sax.saxutils import escape

PAGE_W = 1212.0
TOC_H = 1560.0
SCALE = 0.75
DIM = re.compile(r'<svg[^>]*\bwidth="([\d.]+)"[^>]*\bheight="([\d.]+)"')
Link = tuple[int, float, str, object]


def svg_page(inner: str, width: float, height: float) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}"><rect width="100%" height="100%" fill="#fff"/>'
        f"{inner}</svg>"
    )


def nest(fig: str, y: float) -> tuple[str, float]:
    m = DIM.search(fig)
    w, h = (float(m.group(1)), float(m.group(2))) if m else (PAGE_W, 200.0)
    body = fig[fig.index(">") + 1 : fig.rindex("</svg>")]
    return (f'<svg x="0" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" viewBox="0 0 {w:.0f} {h:.0f}">{body}</svg>'), h


def toc_pages(by_class: list[tuple[str, list[dict]]], corpus: str) -> tuple[list[str], list[Link]]:
    pages: list[str] = []
    links: list[Link] = []
    lines: list[str] = [
        '<text x="16" y="34" font-size="20" font-family="sans-serif" '
        f'font-weight="bold">Concept activation on {escape(corpus)} — contents</text>'
    ]
    y = 60.0

    def flush() -> None:
        nonlocal lines, y
        pages.append(svg_page("".join(lines), PAGE_W, TOC_H))
        lines, y = [], 40.0

    for cls, feats in by_class:
        if y > TOC_H - 60:
            flush()
        lines.append(
            f'<text x="16" y="{y:.0f}" font-size="14" font-family="sans-serif" '
            f'font-weight="bold" fill="#1a4a8a">{escape(cls)}</text>'
        )
        links.append((len(pages), y, "class", cls))
        y += 22
        for f in feats:
            if y > TOC_H - 24:
                flush()
            k = len(f["top3"])
            fill = "#1140a0" if k else "#aaa"
            note = f"  ({k} example{'s' if k != 1 else ''})" if k else "  (no verified examples)"
            lines.append(
                f'<text x="40" y="{y:.0f}" font-size="12" font-family="sans-serif" '
                f'fill="{fill}">{escape(f["concept"])} vs {escape(f["antagonist"])}'
                f"  ·  pair {f['pair_id']}{escape(note)}</text>"
            )
            links.append((len(pages), y, "pair", f["pair_id"]))
            y += 18
        y += 8
    pages.append(svg_page("".join(lines), PAGE_W, TOC_H))
    return pages, links


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("pdf")
    parser = ArgumentParser()
    parser.add_argument("--work", type=Path, default=Path("work"))
    parser.add_argument("--figures", type=Path, default=Path("work/figures"))
    parser.add_argument("--out", type=Path, default=Path("work/probing_lmsys.pdf"))
    args = parser.parse_args()

    figman = json.loads((args.work / "figman.json").read_text())
    corpus = json.loads((args.work / "meta.json").read_text()).get("corpus", "lmsys")
    order: dict[str, list[dict]] = {}
    for f in sorted(figman, key=lambda x: (x["class"], x["pair_id"])):
        order.setdefault(f["class"], []).append(f)
    by_class = list(order.items())

    tmp = args.work / "_pages"
    tmp.mkdir(parents=True, exist_ok=True)
    for old in tmp.glob("*.svg"):
        old.unlink()
    pages: list[Path] = []

    def write(svg: str) -> None:
        p = tmp / f"p{len(pages):04d}.svg"
        p.write_text(svg)
        pages.append(p)

    toc_svgs, toc_links = toc_pages(by_class, corpus)
    for tp in toc_svgs:
        write(tp)

    class_page: dict[str, int] = {}
    pair_page: dict[int, int] = {}
    for cls, feats in by_class:
        for f in feats:
            head = (
                f'<text x="16" y="30" font-size="12" font-family="sans-serif" fill="#888">'
                f"{escape(cls)}</text>"
                f'<text x="16" y="52" font-size="17" font-family="sans-serif" font-weight="bold">'
                f"{escape(f['concept'])}  ‖  {escape(f['antagonist'])}   (pair {f['pair_id']})</text>"
            )
            body, y = [head], 76.0
            for rank in range(1, len(f["top3"]) + 1):
                fig = args.figures / f"pair{f['pair_id']:04d}-r{rank}.svg"
                if fig.exists():
                    markup, h = nest(fig.read_text(), y)
                    body.append(markup)
                    y += h + 18
            if len(f["top3"]) == 0:
                body.append(
                    '<text x="16" y="90" font-size="13" font-family="sans-serif" fill="#999">'
                    "No verified examples (0 conversations accepted by the judge).</text>"
                )
                y = 120.0
            class_page.setdefault(cls, len(pages))
            pair_page[f["pair_id"]] = len(pages)
            write(svg_page("".join(body), PAGE_W, y + 20))

    raw = args.work / "_raw.pdf"
    log.info("rendering %d pages (%d ToC) with rsvg-convert", len(pages), len(toc_svgs))
    subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(raw), *[str(p) for p in pages]], check=True)

    try:
        from PyPDF2 import PdfReader, PdfWriter
        from PyPDF2.generic import AnnotationBuilder

        writer = PdfWriter()
        writer.append(PdfReader(str(raw)))
        root = writer.add_outline_item("Table of contents", 0)
        cbm = {cls: writer.add_outline_item(cls, class_page[cls], parent=root) for cls, _ in by_class}
        for cls, feats in by_class:
            for f in feats:
                title = f"{f['concept']} vs {f['antagonist']} (pair {f['pair_id']})"
                writer.add_outline_item(title, pair_page[f["pair_id"]], parent=cbm[cls])
        page_h = TOC_H * SCALE
        for toc_pg, y_px, kind, key in toc_links:
            target = class_page[key] if kind == "class" else pair_page[key]  # type: ignore[index]
            rect = (16 * SCALE, page_h - (y_px + 4) * SCALE, (PAGE_W - 16) * SCALE, page_h - (y_px - 14) * SCALE)
            ann = AnnotationBuilder.link(rect=rect, target_page_index=target)
            writer.add_annotation(page_number=toc_pg, annotation=ann)
        with args.out.open("wb") as fh:
            writer.write(fh)
        log.info("wrote %s (%d pages, %d clickable ToC links)", args.out, len(pages), len(toc_links))
    except Exception as exc:
        raw.replace(args.out)
        log.warning("links/bookmarks skipped (%s); wrote plain %s", exc, args.out)


if __name__ == "__main__":
    main()
