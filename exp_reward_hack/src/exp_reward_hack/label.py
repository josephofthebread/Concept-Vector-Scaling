import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from exp_reward_hack import workloads
from exp_reward_hack.workloads.common import cheated

MODELS = ["qwen", "llama", "gemma"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("episodes_smoke"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    split: dict[tuple[str, str], Counter] = defaultdict(Counter)
    reverted: dict[tuple[str, str], int] = defaultdict(int)
    types: Counter = Counter()
    seen: set[str] = set()
    records = []

    for jf in sorted(args.dir.glob("*.json")):
        ep = json.loads(jf.read_text())
        wlname = ep.get("workload") or jf.stem.split("-")[0]
        try:
            wl = workloads.load(wlname)
        except KeyError:
            continue
        mtag = ep.get("model", "?").split("/")[-1].lower()
        mtag = next((m for m in MODELS if m in mtag), mtag)
        seen.add(mtag)
        verdict = cheated(wl, ep.get("turns", []))
        split[(wlname, mtag)]["cheat" if verdict["cheated"] else "clean"] += 1
        if verdict["cheated"]:
            types[verdict["type"]] += 1
            if not verdict["final"]:
                reverted[(wlname, mtag)] += 1
        records.append({"file": jf.name, "workload": wlname, "model": mtag, "ending": ep.get("ending"), **verdict})

    models = [m for m in MODELS if m in seen] or sorted(seen)
    print(f"\ndid-it-cheat split over {args.dir}  (C=cheated / n total; ending-agnostic)\n")
    print(f"{'workload':<22} " + "  ".join(f"{m:<10}" for m in models))
    for wl in workloads.ALL:
        name = workloads.load(wl).NAME
        cells = []
        for m in models:
            c = split.get((name, m), Counter())
            n = c["cheat"] + c["clean"]
            cells.append(f"C{c['cheat']}/{n}".ljust(10))
        print(f"{name:<22} " + "  ".join(cells))

    total_cheat = sum(c["cheat"] for c in split.values())
    total = sum(c["cheat"] + c["clean"] for c in split.values())
    total_rev = sum(reverted.values())
    print(
        f"\ncheated {total_cheat}/{total} episodes | types: "
        + ", ".join(f"{k}={v}" for k, v in types.most_common())
        + (f" | {total_rev} cheats later reverted (not in final state)" if total_rev else "")
    )

    if args.out:
        args.out.write_text(json.dumps(records, indent=2))
        print(f"wrote {args.out} ({len(records)} episodes)")


if __name__ == "__main__":
    main()
