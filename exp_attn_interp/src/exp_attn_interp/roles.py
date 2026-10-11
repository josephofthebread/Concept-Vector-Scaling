from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

log = logging.getLogger("roles")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--decomp", type=Path, default=Path("work/per_concept_decomp.json"))
    parser.add_argument("--edgemap", type=Path, default=Path("work/stories_rw/edgemap.json"))
    parser.add_argument("--decompan", type=Path, default=Path("work/decompan.json"))
    parser.add_argument("--out", type=Path, default=Path("work/roles.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    dec = json.loads(args.decomp.read_text())
    em = json.loads(args.edgemap.read_text())
    n_blocks = json.loads(args.decompan.read_text()).get("n_blocks", 36) if args.decompan.exists() else 36

    flags: dict[tuple[int, int], set[str]] = defaultdict(set)
    builders_of: dict[int, list[tuple[int, int]]] = {}
    for d in dec:
        top = [(b, h) for b, h, _ in d.get("top_heads", [])[:3]]
        builders_of[d["pair_id"]] = top
        for bh in top:
            flags[bh].add("builder")
    detectors_of: dict[str, tuple[int, int]] = {}
    for e in em.get("mid_stack_head_map", []):
        bh = (e["block"], e["head"])
        if e["n_concepts"] == 1:
            a = e["concepts"][0].get("attend")
            flags[bh].add("detector" if (a is not None and a >= 0.6) else "propagator")
            detectors_of[e["concepts"][0]["pair"]] = bh
        else:
            flags[bh].add("family_head")
    for e in em.get("input_edge_heads", []):
        flags[(0, e["head"])].add("lex_reader")
    for e in em.get("output_edge_heads", []):
        flags[(n_blocks - 1, e["head"])].add("out_writer")

    combos = Counter("+".join(sorted(f)) for f in flags.values())
    by_depth: dict[str, Counter] = defaultdict(Counter)
    for (b, _), f in flags.items():
        for r in f:
            by_depth[r][b] += 1
    n_heads_total = n_blocks * 32
    det_heads = {bh for bh, f in flags.items() if "detector" in f or "propagator" in f}
    bld_heads = {bh for bh, f in flags.items() if "builder" in f}
    overlap = len(det_heads & bld_heads)

    summary: dict[str, Any] = {
        "heads_total": n_heads_total,
        "heads_with_any_role": len(flags),
        "role_counts": {
            r: sum(1 for f in flags.values() if r in f)
            for r in ("builder", "detector", "propagator", "family_head", "lex_reader", "out_writer")
        },
        "role_combinations": combos.most_common(12),
        "depth_profile": {r: [c.get(b, 0) for b in range(n_blocks)] for r, c in by_depth.items()},
        "detector_or_propagator_heads_that_are_also_builders": overlap,
        "detector_or_propagator_heads": len(det_heads),
        "builder_heads": len(bld_heads),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\n=== HEAD ROLE CENSUS: {len(flags)} of {n_heads_total} heads carry a concept role ===")
    for r, c in summary["role_counts"].items():
        print(f"  {r:12s} {c:4d}")
    print("role combinations:", combos.most_common(8))
    print(f"heads that both DETECT/PROPAGATE and BUILD: {overlap} of {len(det_heads)} dedicated heads")
    for r in ("builder", "detector", "propagator"):
        prof = summary["depth_profile"].get(r, [])
        if prof:
            print(f"  {r:11s} by block: " + " ".join(f"{b}:{c}" for b, c in enumerate(prof) if c))
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
