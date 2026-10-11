from __future__ import annotations

import json
import logging
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

log = logging.getLogger("viewer_prep")


def accepted(scores: list[int], span_poles: list[str], pole: str) -> bool:
    w = "pos" if pole == "con" else "neg"
    own = [s for s, p in zip(scores, span_poles) if p == w]
    return bool(own) and max(own) >= 2


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--per-feature", type=int, default=16)
    ap.add_argument("--zero", action="store_true", help="select examples whose spans all scored 0")
    args = ap.parse_args()
    w = args.work
    sfx, verdict = ("_zero", "no") if args.zero else ("", "yes")
    judged_name, sel_name, hl_name = (
        f"judged{sfx}.json",
        ("examples_zero.json" if args.zero else "examples_sel.json"),
        f"highlights{sfx}.json",
    )

    J = {f["pair_id"]: f for f in json.loads((w / "span_sweep" / "span_judged.json").read_text())["features"]}
    jd = []
    for pid, f in sorted(J.items()):
        rec = {"pair_id": pid}
        for pole in ("con", "ant"):
            rec[pole] = [
                {
                    "conversation_id": e["conversation_id"],
                    "verdict": ("no" if bool(e["scores"]) and max(e["scores"]) == 0 else "yes")
                    if args.zero
                    else ("yes" if accepted(e["scores"], e["span_poles"], pole) else "no"),
                }
                for e in f[pole]
            ]
        jd.append(rec)
    (w / judged_name).write_text(json.dumps({"features": jd}, ensure_ascii=False))
    log.info("wrote %s (%d features)", judged_name, len(jd))

    subprocess.run(
        [
            sys.executable,
            "-m",
            "exp_lmsys_probe.select_examples",
            "--work",
            str(w),
            "--verdict",
            verdict,
            "--per-feature",
            str(args.per_feature),
            "--judged",
            judged_name,
            "--out",
            sel_name,
        ],
        check=True,
    )

    H = json.loads((w / "highlights_all.json").read_text())
    convs = H["convs"]
    cos = {(f["pair_id"], e["pole"], e["conversation_id"]): e["cos"] for f in H["features"] for e in f["examples"]}
    sel = json.loads((w / sel_name).read_text())["features"]
    out, total = [], 0
    for f in sel:
        pid = f["pair_id"]
        exs = []
        for e in f["examples"]:
            cv = convs.get(e["conversation_id"])
            c = cos.get((pid, e["pole"], e["conversation_id"]))
            if cv is None or c is None:
                continue
            exs.append(
                {
                    "conversation_id": e["conversation_id"],
                    "pole": e["pole"],
                    "score": e["score"],
                    "layer": e["layer"],
                    "ncut": cv["ncut"],
                    "toks": cv["toks"],
                    "cos": c,
                }
            )
        total += len(exs)
        out.append({k: f[k] for k in ("pair_id", "concept", "antagonist", "class")} | {"examples": exs})
    (w / hl_name).write_text(json.dumps({"features": out}, ensure_ascii=False))
    log.info("wrote %s (%d examples over %d features)", hl_name, total, len(out))


if __name__ == "__main__":
    main()
