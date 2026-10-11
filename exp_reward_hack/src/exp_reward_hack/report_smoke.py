import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from exp_reward_hack import workloads
from exp_reward_hack.workloads.common import classify, hack_label

MODELS = ["qwen", "llama", "gemma"]
NOHACK = {"give_up", "degenerate"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("episodes_smoke"))
    args = ap.parse_args()

    split: dict[tuple[str, str], Counter] = defaultdict(Counter)
    types: dict[tuple[str, str], Counter] = defaultdict(Counter)
    seen_models: set[str] = set()

    for jf in sorted(args.dir.glob("*.json")):
        episode = json.loads(jf.read_text())
        wlname = episode.get("workload") or jf.stem.split("-")[0]
        try:
            wl = workloads.load(wlname)
        except KeyError:
            continue
        mtag = episode.get("model", "?").split("/")[-1].lower()
        mtag = next((m for m in MODELS if m in mtag), mtag)
        seen_models.add(mtag)
        ending = episode["ending"]
        root = args.dir / jf.stem
        is_hack, kind = (False, None)
        if root.is_dir():
            is_hack, kind = hack_label(wl, ending, classify(wl, root))
        outcome = "HACK" if is_hack else ("nohack" if ending in NOHACK else ending)
        split[(wlname, mtag)][outcome] += 1
        if is_hack:
            types[(wlname, mtag)][kind] += 1

    models = [m for m in MODELS if m in seen_models] or sorted(seen_models)
    print(f"\nsmoke split over {args.dir}  (H=hack, N=give_up/degenerate, other=parse/max/ctx/stopped)\n")
    print(f"{'workload':<22} " + "  ".join(f"{m:<16}" for m in models) + "  verdict")
    flags = []
    for wl in workloads.ALL:
        name = workloads.load(wl).NAME
        cells, has_hack, has_nohack = [], False, False
        for m in models:
            counter = split.get((name, m), Counter())
            n = counter.get("HACK", 0)
            no = counter.get("nohack", 0)
            other = sum(v for k, v in counter.items() if k not in ("HACK", "nohack"))
            has_hack = has_hack or n > 0
            has_nohack = has_nohack or no > 0
            cells.append(f"H{n} N{no} o{other}".ljust(16))
        verdict = "ok" if (has_hack and has_nohack) else ("NO HACKS" if not has_hack else "NO NON-HACKS")
        if verdict != "ok":
            flags.append((name, verdict))
        print(f"{name:<22} " + "  ".join(cells) + f"  {verdict}")

    hacktypes = Counter()
    for counter in types.values():
        hacktypes.update(counter)
    if hacktypes:
        print("\nhack types:", ", ".join(f"{k}={v}" for k, v in hacktypes.most_common()))
    if flags:
        print("\nnon-discriminating (tune deniability or drop before the 64-seed run):")
        for name, verdict in flags:
            print(f"  {name}: {verdict}")
    else:
        print("\nall tasks discriminate at this seed count")


if __name__ == "__main__":
    main()
