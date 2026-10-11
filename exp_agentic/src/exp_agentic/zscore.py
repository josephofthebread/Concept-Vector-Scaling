#! /usr/bin/env python


import argparse
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("zscore")

FLOOR = 1e-6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    done = 0
    for source in sorted(args.dir.glob("*.scores.npy")):
        target = source.with_name(source.name.replace(".scores.npy", ".z.npy"))
        if target.exists() and not args.redo:
            continue
        scores = np.load(source)
        values = scores.astype(np.float32)
        center = values.mean(axis=0, keepdims=True)
        spread = values.std(axis=0, keepdims=True)
        quiet = spread < FLOOR
        result = (values - center) / np.where(quiet, 1.0, spread)
        result[np.broadcast_to(quiet, result.shape)] = 0.0
        np.save(target, result.astype(np.float16))
        done += 1
        if done % 50 == 0:
            log.info(f"{done} z-scored")
    log.info(f"wrote {done} z-score files under {args.dir}")


if __name__ == "__main__":
    main()
