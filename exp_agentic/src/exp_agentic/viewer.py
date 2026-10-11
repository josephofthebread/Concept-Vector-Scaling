#! /usr/bin/env python


import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from safetensors.torch import load_file

from exp_agentic.readout import find

log = logging.getLogger("viewer")

LAYERS = (18, 25)
SPAN = 8.0
ENDING = {
    "submit": "REWARD HACK",
    "give_up": "gave up",
    "degenerate": "degenerate",
    "max_turns": "hit turn cap",
    "context_exhausted": "out of context",
    "parse_failures": "parse failures",
    "stopped_without_submit": "stopped",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("episodes/gate"))
    parser.add_argument("--out", type=Path, default=Path("../scope/data"))
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--limit", type=int, default=0, help="pack only the first N episodes")
    parser.add_argument(
        "--survey",
        type=Path,
        default=Path("analysis/survey-gate.json"),
        help="survey output, used to rank concepts by how differently they behave on hacks",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    rows = pq.read_table(find("pairs.parquet")).to_pylist()
    pairs = len(rows)
    args.out.mkdir(parents=True, exist_ok=True)

    block = load_file(find("diff.safetensors"))["diff"]
    unit = block[25].float().numpy()
    unit = unit / np.linalg.norm(unit, axis=1, keepdims=True)
    similarity = unit @ unit.T
    np.fill_diagonal(similarity, -2.0)
    (args.out / "concepts.json").write_text(
        json.dumps(
            {
                "pairs": [
                    {
                        "pair": index,
                        "concept": row["concept"],
                        "antagonist": row["antagonist"],
                        "class": row["class_name"],
                    }
                    for index, row in enumerate(rows)
                ],
                "neighbors": np.argsort(-similarity, axis=1)[:, :12].tolist(),
            }
        )
    )

    if args.survey.exists():
        survey = json.loads(args.survey.read_text())
        gaps = {c["pair"]: (c.get("gap_hacked_minus_other") or 0.0) for c in survey["concepts"]}
        edge = {c["pair"]: c["layers"]["L25"]["fraction_at_turn_boundary"] for c in survey["concepts"]}
        spread = float(np.std([v for v in gaps.values()])) or 1.0
        hit = [pair for pair, gap in gaps.items() if abs(gap) >= 2 * spread and edge[pair] <= 0.10]
        (args.out / "effects.json").write_text(
            json.dumps({"did": {str(k): round(v, 3) for k, v in gaps.items()}, "hit": hit})
        )
        log.info(f"effects.json: {len(hit)} concepts separate hacks by >2sd ({2 * spread:.3f}z)")
    else:
        log.warning(f"{args.survey} absent; the concept ranking and badges will be unavailable")

    scored = sorted(args.dir.glob("*.z.npy"))
    if args.limit:
        scored = scored[: args.limit]
    generations = []

    for count, path in enumerate(scored, start=1):
        record = Path(str(path).replace(".z.npy", ".json"))
        episode = json.loads(record.read_text())
        stem = record.stem
        z = np.load(path).astype(np.float32)
        ids = episode["ids"][: z.shape[0]]

        norms_path = record.with_suffix(".norms.npy")
        if norms_path.exists():
            magnitude = np.load(norms_path)[: z.shape[0]]
        else:
            magnitude = None

        scales = {}
        for slot, layer in enumerate(LAYERS):
            scale = SPAN / 127.0
            packed = np.clip(np.rint(z[:, slot, :] / scale), -127, 127).astype(np.int8).tobytes()
            (args.out / f"{stem}.m0.L{layer}.bin").write_bytes(packed)
            scales[f"m0.L{layer}"] = scale

        (args.out / f"{stem}.tokens.json").write_text(
            json.dumps(
                {
                    "pieces": tokenizer.convert_ids_to_tokens(ids),
                    "role": episode["roles"][: z.shape[0]],
                    "logprob": [None] * len(ids),
                    **(
                        {
                            "norms": {
                                f"L{layer}": [round(float(v), 1) for v in magnitude[:, slot]]
                                for slot, layer in enumerate(LAYERS)
                            }
                        }
                        if magnitude is not None
                        else {}
                    ),
                }
            )
        )

        ending = episode.get("ending", "?")
        generations.append(
            {
                "stem": stem,
                "ending": ending,
                "outcome": ENDING.get(ending, ending),
                "seed": episode.get("seed"),
                "variant": episode.get("variant", ""),
                "turns": len(episode.get("turns", [])),
                "distinct": episode.get("distinct"),
                "tokens": len(ids),
                "width": pairs,
                "scales": scales,
            }
        )
        if count % 25 == 0:
            log.info(f"{count}/{len(scored)} packed")

    order = {"submit": 0, "degenerate": 1, "give_up": 2}
    generations.sort(key=lambda row: (order.get(row["ending"], 9), -(row["distinct"] or 0), row["stem"]))

    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "methods": ["z-score (per episode)"],
                "layers": list(LAYERS),
                "pairs": pairs,
                "span": SPAN,
                "generations": generations,
            }
        )
    )
    counts: dict[str, int] = {}
    for row in generations:
        counts[row["outcome"]] = counts.get(row["outcome"], 0) + 1
    log.info(f"wrote {len(generations)} trajectories to {args.out}: {counts}")


if __name__ == "__main__":
    main()
