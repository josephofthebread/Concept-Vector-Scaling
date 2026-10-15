from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer

from exp_attn_interp.config import INSTRUCT_MODEL, PAIRS

log = logging.getLogger("edgemap")
MIN_GAP = 0.05
MIN_N = 5


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--write-dir", type=Path, default=Path("work/stories/write"))
    parser.add_argument("--markers", type=Path, default=Path("data/markers_unique.json"))
    parser.add_argument("--ontology", type=Path, default=Path("../exp_genvectors/data/semantic.ontology.json"))
    parser.add_argument("--out", type=Path, default=Path("work/edgemap.json"))
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    pairs = sorted(json.loads(args.ontology.read_text())["pairs"], key=lambda p: int(p["pair_id"]))
    uniq = json.loads(args.markers.read_text())["concept"]
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, local_files_only=True)
    words_of = {k: [str(tok.decode([i])).strip() for i in uniq[str(k)]] for k in range(PAIRS)}

    files = sorted(args.write_dir.glob("item-*.safetensors"))
    with safe_open(str(files[0]), framework="np") as h:
        n_blocks, heads = (int(x) for x in h.get_tensor("write_cos").shape[:2])
    rows = n_blocks * heads
    mw_sum = np.zeros((rows, PAIRS))
    mw_n = np.zeros((rows, PAIRS))
    ma_sum = np.zeros((rows, PAIRS))
    ma_n = np.zeros((rows, PAIRS))
    for f in files:
        with safe_open(str(f), framework="np") as h:
            mw = h.get_tensor("marker_write").astype(np.float32).reshape(rows, PAIRS)
            ma = None
            if "marker_attend" in h.keys():
                ma = h.get_tensor("marker_attend").astype(np.float32).reshape(rows, PAIRS)
        v = ~np.isnan(mw)
        mw_sum += np.where(v, mw, 0.0)
        mw_n += v
        if ma is not None:
            va = ~np.isnan(ma)
            ma_sum += np.where(va, ma, 0.0)
            ma_n += va
    with np.errstate(invalid="ignore"):
        MW = np.where(mw_n > 0, mw_sum / np.clip(mw_n, 1, None), np.nan)
        MA = np.where(ma_n > 0, ma_sum / np.clip(ma_n, 1, None), np.nan)
    have_attend = bool((ma_n > 0).any())

    def att(h: int, k: int) -> float | None:
        """Read-side check: that head's attend-AUROC on k's markers (0.5 = no routing into them)."""
        return round(float(MA[h, k]), 3) if have_attend and np.isfinite(MA[h, k]) else None

    log.info("%d items, %d concepts with any marker data", len(files), int(np.isfinite(MW).any(0).sum()))

    rec: dict[int, tuple[int, float, int]] = {}
    for k in range(PAIRS):
        col = MW[:, k]
        if not np.isfinite(col).any():
            continue
        h = int(np.nanargmax(col))
        off = np.nanmedian(MW[h, :])
        rec[k] = (h, float(col[h] - off), int(mw_n[h, k]))
    good = {k: r for k, r in rec.items() if r[1] >= MIN_GAP and r[2] >= MIN_N}
    log.info("concepts with a real marker-local head (gap>=%.2f, n>=%d): %d / %d", MIN_GAP, MIN_N, len(good), len(rec))

    blocks = Counter(r[0] // heads for r in good.values())
    band = {
        "input_edge (block 0)": sum(v for b, v in blocks.items() if b == 0),
        "early (1-5)": sum(v for b, v in blocks.items() if 1 <= b <= 5),
        "mid (6-30)": sum(v for b, v in blocks.items() if 6 <= b <= 30),
        "late (31-34)": sum(v for b, v in blocks.items() if 31 <= b <= 34),
        "output_edge (block 35)": sum(v for b, v in blocks.items() if b == 35),
    }

    def head_map(block: int) -> list[dict[str, Any]]:
        byhead: dict[int, list[int]] = defaultdict(list)
        for k, (h, _, _) in good.items():
            if h // heads == block:
                byhead[h % heads].append(k)
        out = []
        for hd, ks in sorted(byhead.items(), key=lambda x: -len(x[1])):
            ks = sorted(ks, key=lambda k: -good[k][1])
            wc = Counter(w for k in ks for w in words_of[k])
            out.append(
                {
                    "head": hd,
                    "n_concepts": len(ks),
                    "top_words": [w for w, _ in wc.most_common(12)],
                    "concepts": [
                        {
                            "pair": f"{pairs[k]['concept']} || {pairs[k]['antagonist']}"[:64],
                            "gap": round(good[k][1], 3),
                            "n": good[k][2],
                        }
                        for k in ks[:8]
                    ],
                }
            )
        return out

    edge_in, edge_out = head_map(0), head_map(n_blocks - 1)

    mid = sorted([(r[1], k, r[0], r[2]) for k, r in good.items() if 6 <= r[0] // heads <= 30], reverse=True)
    mid_hits = [
        {
            "pair": f"{pairs[k]['concept']} || {pairs[k]['antagonist']}"[:64],
            "block": h // heads,
            "head": h % heads,
            "gap": round(g, 3),
            "n": n,
        }
        for g, k, h, n in mid
    ]
    per_block = [int(sum(1 for r in good.values() if r[0] // heads == b)) for b in range(n_blocks)]
    mid_byhead: dict[int, list[int]] = defaultdict(list)
    for k, (h, _, _) in good.items():
        if 6 <= h // heads <= 30:
            mid_byhead[h].append(k)
    mid_head_map: list[dict[str, Any]] = []
    for h, ks in sorted(mid_byhead.items(), key=lambda x: (-len(x[1]), x[0])):
        ks = sorted(ks, key=lambda k: -good[k][1])
        wc = Counter(w for k in ks for w in words_of[k])
        mid_head_map.append(
            {
                "block": h // heads,
                "head": h % heads,
                "n_concepts": len(ks),
                "top_words": [w for w, _ in wc.most_common(8)],
                "concepts": [
                    {
                        "pair": f"{pairs[k]['concept']} || {pairs[k]['antagonist']}"[:64],
                        "gap": round(good[k][1], 3),
                        "n": good[k][2],
                        "attend": att(h, k),
                    }
                    for k in ks[:6]
                ],
            }
        )
    dedicated = sum(1 for e in mid_head_map if e["n_concepts"] == 1)
    circuits = [k for k, (h, _, _) in good.items() if 6 <= h // heads <= 30 and (att(h, k) or 0.0) >= 0.6]

    summary = {
        "items": len(files),
        "concepts_with_real_head": len(good),
        "concepts_total": len(rec),
        "localization": band,
        "per_block": per_block,
        "input_edge_heads": edge_in,
        "output_edge_heads": edge_out,
        "mid_stack_hits": mid_hits,
        "mid_stack_head_map": mid_head_map,
        "mid_stack_heads": len(mid_head_map),
        "mid_stack_dedicated_heads": dedicated,
        "have_attend": have_attend,
        "mid_stack_full_circuits": len(circuits),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))

    print(f"\n=== LOCALIZATION of marker-local concept heads ({len(good)} concepts, gap>={MIN_GAP}, n>={MIN_N}) ===")
    for name, c in band.items():
        print(f"  {name:24s} {c:4d}  ({100 * c / max(len(good), 1):.0f}%)")
    for title, m in (("INPUT EDGE (block 0)", edge_in), ("OUTPUT EDGE (block 35)", edge_out)):
        print(f"\n=== {title}: head -> vocabulary / concepts ===")
        for e in m[:8]:
            print(f"  H{e['head']:2d}  {e['n_concepts']:3d} concepts | reads: {e['top_words'][:10]}")
            for cc in e["concepts"][:4]:
                print(f"        {cc['gap']:+.3f} n={cc['n']:2d}  {cc['pair']}")
    print("\n=== PER-BLOCK count of well-attested concept heads (block: count) ===")
    print("  " + " ".join(f"{b}:{c}" for b, c in enumerate(per_block) if c))
    circ = f", {len(circuits)} also ATTEND to their markers (full read->write circuits)" if have_attend else ""
    print(
        f"\n=== MID-STACK (blocks 6-30): {len(mid_hits)} concepts on {len(mid_head_map)} heads, "
        f"{dedicated} heads dedicated to a single concept{circ} ==="
    )
    for e in mid_head_map[:14]:
        print(f"  L{e['block']:2d} H{e['head']:2d}  {e['n_concepts']:2d} concept(s) | words: {e['top_words'][:7]}")
        for cc in e["concepts"][:3]:
            a = f" attend={cc['attend']:.2f}" if cc.get("attend") is not None else ""
            print(f"        write{cc['gap']:+.3f}{a} n={cc['n']:3d}  {cc['pair']}")
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
