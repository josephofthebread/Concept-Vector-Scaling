from __future__ import annotations

import json
import logging
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch

from exp_attn_interp.common import center, generate_reply, load_model, sink_mask, token_cosine
from exp_attn_interp.config import (
    BAND,
    CEILING_ITEMS,
    CENTERINGS,
    LMSYS_ITEMS,
    MARKER_MIN_OCCURRENCES,
    MARKERS_LOGITLENS,
    MARKERS_PRESENCE,
    PAIRS,
    READ_LAYER,
    REPLY_TOKENS,
    SAMPLING,
    generation_seed,
)
from exp_attn_interp.stats import participation_ratio, rank_auroc

log = logging.getLogger("ceiling")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--items", type=Path, default=LMSYS_ITEMS)
    parser.add_argument("--markers", type=Path, default=MARKERS_PRESENCE)
    parser.add_argument("--markers-crosscheck", type=Path, default=MARKERS_LOGITLENS)
    parser.add_argument("--limit", type=int, default=CEILING_ITEMS)
    parser.add_argument("--out", type=Path, default=Path("work/ceiling.json"))
    parser.add_argument("--offline", action=BooleanOptionalAction, default=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model, tokenizer = load_model(offline=args.offline)
    from exp_attn_interp.vectors import load_diff_vectors

    vectors = load_diff_vectors(device=str(model.device))

    marker_sets: dict[str, list[set[int]]] = {}
    for name, path in [("presence", args.markers), ("logitlens", args.markers_crosscheck)]:
        if path.exists():
            concept = json.loads(path.read_text())["concept"]
            marker_sets[name] = [set(concept[str(k)]) for k in range(PAIRS)]
    if not marker_sets:
        raise SystemExit("no marker files found; run attn-presence / attn-markers first")
    special_ids = set(tokenizer.all_special_ids)

    items = pq.read_table(args.items).to_pylist()[: args.limit]
    dims: dict[int, dict[str, list[float]]] = {ell: {m: [] for m in CENTERINGS} for ell in BAND}
    shared: dict[str, list[float]] = {m: [] for m in CENTERINGS}
    matched_auc: dict[str, list[float]] = {name: [] for name in marker_sets}
    mismatched_auc: dict[str, list[float]] = {name: [] for name in marker_sets}

    for number, item in enumerate(items, start=1):
        seed = generation_seed(int(item["item_id"]))
        prompt_ids, reply_ids = generate_reply(
            model, tokenizer, item["text"], int(item["item_id"]), REPLY_TOKENS, SAMPLING, seed
        )
        ids = prompt_ids + reply_ids
        with torch.inference_mode():
            out = model(
                input_ids=torch.tensor([ids], device=model.device),
                output_attentions=True,
                output_hidden_states=True,
                use_cache=False,
            )
        hidden = list(out.hidden_states)
        mask = sink_mask(list(out.attentions), hidden)
        content = (~mask).clone()
        for pos, tok in enumerate(ids):
            if tok in special_ids:
                content[pos] = False
        keep = content.nonzero().flatten()
        if keep.numel() < 32:
            log.info(
                "%s/%s item %s: only %d content tokens, skipped", number, len(items), item["item_id"], int(keep.numel())
            )
            continue

        for ell in BAND:
            cos = token_cosine(hidden[ell][0][keep], vectors[ell])
            for mode in CENTERINGS:
                c = center(cos, mode)
                centered = c - c.mean(dim=0, keepdim=True)
                cov = centered.T @ centered
                eig = torch.linalg.eigvalsh(cov)
                dims[ell][mode].append(participation_ratio(eig))
                if ell == READ_LAYER:
                    series = (c - c.mean(dim=0, keepdim=True)).T
                    mean_series = series.mean(dim=0, keepdim=True)
                    sim = torch.nn.functional.cosine_similarity(series, mean_series, dim=-1)
                    shared[mode].append(float(sim.median()))

        cos = token_cosine(hidden[READ_LAYER][0][keep], vectors[READ_LAYER])
        z = center(cos, "token")
        kept_ids = [ids[i] for i in keep.tolist()]
        n_present = {}
        for name, marker_ids in marker_sets.items():
            present = [k for k in range(PAIRS) if sum(t in marker_ids[k] for t in kept_ids) >= MARKER_MIN_OCCURRENCES]
            n_present[name] = len(present)
            for k in present:
                positive = torch.tensor([t in marker_ids[k] for t in kept_ids], device=z.device)
                auc = rank_auroc(z[:, k].unsqueeze(0), positive)[0]
                if not torch.isnan(auc):
                    matched_auc[name].append(float(auc))
                other = (k + 500) % PAIRS
                positive_null = torch.tensor([t in marker_ids[other] for t in kept_ids], device=z.device)
                if positive_null.sum() >= MARKER_MIN_OCCURRENCES:
                    auc_null = rank_auroc(z[:, k].unsqueeze(0), positive_null)[0]
                    if not torch.isnan(auc_null):
                        mismatched_auc[name].append(float(auc_null))
        log.info(
            "%s/%s item %s: %d content tokens, concepts with markers %s",
            number,
            len(items),
            item["item_id"],
            int(keep.numel()),
            n_present,
        )

    def med(xs: list[float]) -> float:
        return float(torch.tensor(xs).median()) if xs else float("nan")

    summary: dict[str, Any] = {
        "items": len(items),
        "read_layer": READ_LAYER,
        "effective_dim": {mode: {ell: round(med(dims[ell][mode]), 2) for ell in BAND} for mode in CENTERINGS},
        "effective_dim_band_median": {
            mode: round(med([d for ell in BAND for d in dims[ell][mode]]), 2) for mode in CENTERINGS
        },
        "shared_component_cos_read_layer": {mode: round(med(shared[mode]), 3) for mode in CENTERINGS},
        "marker_spike": {
            name: {
                "auc_matched": round(med(matched_auc[name]), 3),
                "auc_mismatched": round(med(mismatched_auc[name]), 3),
                "n": len(matched_auc[name]),
            }
            for name in marker_sets
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    log.info("=== ceiling summary ===")
    log.info("effective dim (band median): %s", summary["effective_dim_band_median"])
    log.info("shared-component cos @L%d: %s", READ_LAYER, summary["shared_component_cos_read_layer"])
    for name in marker_sets:
        s = summary["marker_spike"][name]
        log.info(
            "marker spike [%s]: matched=%.3f mismatched=%.3f (n=%d)",
            name,
            s["auc_matched"],
            s["auc_mismatched"],
            s["n"],
        )
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
