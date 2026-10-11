#! /usr/bin/env python

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from exp_reward_hack import workloads
from exp_reward_hack.workloads.common import classify, hack_label, materialize, pytest_counts


def summary(counts: dict) -> str:
    keys = ("passed", "failed", "error", "timeout")
    return " ".join(f"{k}={counts[k]}" for k in keys if k in counts) or f"exit={counts.get('exit')}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, help="workload names to check")
    parser.add_argument("--keep", action="store_true", help="leave the scratch checkouts on disk")
    args = parser.parse_args()

    if sys.version_info[:2] == (3, 9):
        raise SystemExit("refusing to validate under Python 3.9: env_pin would be satisfiable")
    print(f"python {sys.version.split()[0]} on {sys.platform}")

    names = args.only or workloads.ALL
    scratch = Path(tempfile.mkdtemp(prefix="rh-validate-"))
    failures = {}
    try:
        for name in names:
            wl = workloads.load(name)
            problems: list[str] = []
            line = f"{wl.NAME:<22} {'impossible' if wl.IMPOSSIBLE else 'possible  '}"

            stub = scratch / name / "stub"
            materialize(wl, stub)
            counts = pytest_counts(stub / wl.TEST_DIR)
            if counts.get("exit", 1) == 0:
                problems.append(f"stub is not red ({summary(counts)})")
            line += f" | stub {summary(counts)}"

            honest = scratch / name / "honest"
            materialize(wl, honest, wl.HONEST)
            counts = pytest_counts(honest / wl.TEST_DIR)
            if wl.IMPOSSIBLE and counts.get("exit", 1) == 0:
                problems.append(f"HONEST turned an impossible task green ({summary(counts)})")
            if not wl.IMPOSSIBLE and not (counts.get("exit", 1) == 0 and counts.get("passed", 0) > 0):
                problems.append(f"HONEST does not pass a possible task ({summary(counts)})")
            line += f" | honest {summary(counts)}"
            verdict = classify(wl, honest)
            if not wl.IMPOSSIBLE:
                if verdict["hidden_pass_rate"] is None or verdict["hidden_pass_rate"] < 1.0:
                    problems.append(f"HONEST fails hidden tests (pass rate {verdict['hidden_pass_rate']})")
                is_hack, _ = hack_label(wl, "submit", verdict)
                if is_hack:
                    problems.append("HONEST submit would be labelled a hack")
            if verdict["tampered"] or verdict["added"]:
                problems.append(f"HONEST shows tampering: {verdict['tampered'] + verdict['added']}")

            cheat = scratch / name / "cheat"
            materialize(wl, cheat, wl.CHEAT)
            counts = pytest_counts(cheat / wl.TEST_DIR)
            if not (counts.get("exit", 1) == 0 and counts.get("passed", 0) > 0):
                problems.append(f"CHEAT is not green ({summary(counts)})")
            line += f" | cheat {summary(counts)}"
            verdict = classify(wl, cheat)
            is_hack, kind = hack_label(wl, "submit", verdict)
            if not is_hack:
                problems.append(f"CHEAT submit not labelled a hack: {verdict}")
            line += f" -> {kind}"
            if not wl.IMPOSSIBLE:
                line += f" (hidden {verdict['hidden_pass_rate']:.2f})"

            print(("ok   " if not problems else "FAIL ") + line)
            for problem in problems:
                print(f"       - {problem}")
            if problems:
                failures[name] = problems
    finally:
        if args.keep:
            print(f"scratch kept at {scratch}")
        else:
            shutil.rmtree(scratch, ignore_errors=True)

    print()
    if failures:
        raise SystemExit(f"{len(failures)}/{len(names)} workloads failed validation: {sorted(failures)}")
    print(f"all {len(names)} workloads validated")


if __name__ == "__main__":
    main()
