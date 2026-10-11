from __future__ import annotations

import json
import logging
import re
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from exp_attn_interp.config import INSTRUCT_MODEL, MARKER_TOPN, PAIRS

log = logging.getLogger("presence")

WORD = re.compile(r"[a-z]{3,}")
STOPWORDS = frozenset(
    """
the a an and or but if then than that this these those to of in on at by for with from as is are was
were be been being it its it's he she they them his her their our your you i we me my mine ours yours
not no nor so too very can will would could should may might must do does did done have has had not
what which who whom whose when where why how all any both each few more most other some such only own
same not out up down over under again further once here there about into through during before after
above below between out off can't don't won't isn't aren't wasn't weren't he's she's they're i'm we're
one two three like just get got go went really much many said says say get also back even still way
""".split()
)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=Path("../exp_gendata/data/semantic.parquet"))
    parser.add_argument("--topn", type=int, default=MARKER_TOPN)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument(
        "--freq-exclude",
        type=int,
        default=300,
        help="drop the K most frequent corpus words (function words that leak via chance df-gaps)",
    )
    parser.add_argument("--out", type=Path, default=Path("data/markers_presence.json"))
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    log.info("reading English stories from %s", args.parquet)
    table = pq.read_table(
        args.parquet, columns=["pair_id", "concept_text", "antagonist_text"], filters=[("language_id", "==", 0)]
    )
    pid = table["pair_id"].to_pylist()
    concept_text = table["concept_text"].to_pylist()
    antag_text = table["antagonist_text"].to_pylist()
    log.info("%d English rows across %d pairs", len(pid), len(set(pid)))

    concept_df: dict[int, Counter] = {}
    antag_df: dict[int, Counter] = {}
    overall: Counter = Counter()
    for k, ctext, atext in zip(pid, concept_text, antag_text):
        cw, aw = (
            {w for w in WORD.findall(ctext.lower()) if w not in STOPWORDS},
            {w for w in WORD.findall(atext.lower()) if w not in STOPWORDS},
        )
        concept_df.setdefault(k, Counter()).update(cw)
        antag_df.setdefault(k, Counter()).update(aw)
        overall.update(cw)
        overall.update(aw)
    frequent = {w for w, _ in overall.most_common(args.freq_exclude)}
    log.info("excluding %d most frequent corpus words, e.g. %s", len(frequent), [w for w, _ in overall.most_common(12)])

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    def markers(pole: Counter, other: Counter) -> tuple[list[int], list[str]]:
        gap = {w: pole[w] - other.get(w, 0) for w in pole if pole[w] >= args.min_df and w not in frequent}
        top = sorted((w for w in gap if gap[w] > 0), key=lambda w: -gap[w])[: args.topn]
        ids = sorted({tokenizer(" " + w, add_special_tokens=False)["input_ids"][0] for w in top})
        return ids, top

    concept_ids: dict[str, list[int]] = {}
    antag_ids: dict[str, list[int]] = {}
    concept_words: dict[str, list[str]] = {}
    for k in range(PAIRS):
        cd, ad = concept_df.get(k, Counter()), antag_df.get(k, Counter())
        cids, cwords = markers(cd, ad)
        aids, _ = markers(ad, cd)
        concept_ids[str(k)] = cids
        antag_ids[str(k)] = aids
        concept_words[str(k)] = cwords

    payload = {
        "source": "presence_gap_english",
        "topn": args.topn,
        "concept": concept_ids,
        "antagonist": antag_ids,
        "concept_words": concept_words,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload))
    sizes = [len(v) for v in concept_ids.values()]
    log.info(
        "wrote %s: %d concepts, median %d marker ids/concept (min %d, max %d)",
        args.out,
        len(concept_ids),
        sorted(sizes)[len(sizes) // 2],
        min(sizes),
        max(sizes),
    )


if __name__ == "__main__":
    main()
