from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

log = logging.getLogger("dynan")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--dyn-dir", type=Path, default=Path("work/dyn"))
    parser.add_argument("--ontology", type=Path, default=Path("../exp_gendata/data/semantic.ontology.json"))
    parser.add_argument("--detectors", default="", help="'pair:block:head,...' known detector heads to rank")
    parser.add_argument("--out", type=Path, default=Path("work/dynan.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    pairs = sorted(json.loads(args.ontology.read_text())["pairs"], key=lambda p: int(p["pair_id"]))
    det = {}
    for x in args.detectors.split(","):
        if x:
            k, b, h = (int(v) for v in x.split(":"))
            det[k] = (b, h)

    per: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    n_blocks = heads = None
    for f in sorted(args.dyn_dir.glob("*.safetensors")):
        with safe_open(str(f), framework="np") as h:
            meta = h.metadata()
            if meta.get("side") != "concept":
                continue
            k = int(meta["pair_id"])
            s0, s1 = (int(v) for v in meta["span"].split(":"))
            c = h.get_tensor("c").astype(np.float32)
            m = h.get_tensor("m").astype(np.float32)
            traj = h.get_tensor("traj").astype(np.float32)
            marker = h.get_tensor("marker").astype(bool)
            sink = h.get_tensor("sink").astype(bool)
        n_blocks, heads = c.shape[1], c.shape[2]
        keep = np.zeros(len(marker), bool)
        keep[s0:s1] = True
        keep &= ~sink
        mk, nm = keep & marker, keep & ~marker
        if mk.sum() < 2 or nm.sum() < 5:
            continue
        per[k]["traj_m"].append(traj[mk].mean(0))
        per[k]["traj_n"].append(traj[nm].mean(0))
        per[k]["blk_attn_m"].append(c[mk].sum(2).mean(0))
        per[k]["blk_mlp_m"].append(m[mk].mean(0))
        per[k]["head_m"].append(c[mk].mean(0))
        per[k]["head_n"].append(c[nm].mean(0))
        per[k]["n_marker"].append(int(mk.sum()))

    report: list[dict[str, Any]] = []
    for k, d in sorted(per.items()):
        tm, tn = np.mean(d["traj_m"], 0), np.mean(d["traj_n"], 0)
        sep = tm - tn
        first = int(np.argmax(sep >= 0.5 * sep[-1])) if sep[-1] > 0 else -1
        ba, bm = np.mean(d["blk_attn_m"], 0), np.mean(d["blk_mlp_m"], 0)
        tot = ba.sum() + bm.sum()
        hm, hn = np.mean(d["head_m"], 0), np.mean(d["head_n"], 0)
        spec = hm - hn
        top = [
            (int(i // heads), int(i % heads), round(float(spec.ravel()[i]), 4)) for i in np.argsort(-spec.ravel())[:5]
        ]
        det_rank = None
        if k in det:
            b, h = det[k]
            det_rank = int((spec.ravel() > spec[b, h]).sum()) + 1
        report.append(
            {
                "pair_id": k,
                "pair": f"{pairs[k]['concept']} || {pairs[k]['antagonist']}"[:56],
                "n_stories": len(d["n_marker"]),
                "markers_per_story": float(np.mean(d["n_marker"])),
                "sep_final": round(float(sep[-1]), 4),
                "sep_half_layer": first,
                "traj_marker": [round(float(x), 3) for x in tm],
                "traj_rest": [round(float(x), 3) for x in tn],
                "attn_share_at_markers": round(float(ba.sum() / tot), 3) if abs(tot) > 1e-9 else None,
                "block_profile_at_markers": [round(float(x), 4) for x in (ba + bm)],
                "top_marker_specific_heads": top,
                "detector_rank": det_rank,
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"n_blocks": n_blocks, "heads": heads, "concepts": report}, indent=2))
    print(f"\n=== PER-TOKEN DYNAMICS: {len(report)} concepts ===")
    for r in report:
        prof = np.array(r["block_profile_at_markers"])
        top_blocks = [int(i) for i in np.argsort(-np.abs(prof))[:4]]
        print(f"\n{r['pair']}  (n={r['n_stories']}, markers/story {r['markers_per_story']:.0f})")
        print(
            f"  separation marker-vs-rest at last layer {r['sep_final']:+.3f}; reaches half by layer {r['sep_half_layer']}"
        )
        print(
            f"  attention share of the write at markers {r['attn_share_at_markers']}; top writing blocks {top_blocks}"
        )
        print(
            f"  marker-specific heads: {r['top_marker_specific_heads'][:4]}"
            + (f"  detector rank {r['detector_rank']}" if r["detector_rank"] else "")
        )
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
