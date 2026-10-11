#! /usr/bin/env python
import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

log = logging.getLogger("predict")

VECTORS = Path("../exp_genvectors/work/vectors/Qwen_Qwen3-8B--assistant--semantic")


LAYERS = (18, 25)
GROUPS = ("submit", "give_up", "degenerate")
LABEL = {"submit": "reward hacked", "give_up": "gave up", "degenerate": "degenerate"}
EARLY = 3
FLOOR = 32
WINDOWS = ("all", "early", "pre", "last")
KINDS = ("z", "cos")
PREFIXES = (1, 2, 3, 4, 6, 8)
SINGLES = (0, 1, 2, 3)
SHAPE = ("turns", "tokens", "thinking", "distinct")
LAMBDAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def ranks(column: np.ndarray) -> np.ndarray:
    """Average ranks in `[1, n]`, ties shared."""
    order = np.argsort(column, kind="stable")
    out = np.empty(len(column), dtype=np.float64)
    out[order] = np.arange(1, len(column) + 1)
    values = column[order]
    start = 0
    for stop in range(1, len(values) + 1):
        if stop == len(values) or values[stop] != values[start]:
            out[order[start:stop]] = np.arange(start + 1, stop + 1).mean()
            start = stop
    return out


def auc(matrix: np.ndarray, positive: np.ndarray) -> np.ndarray:
    """Mann-Whitney AUC of every column against a binary label."""
    rank = np.stack([ranks(matrix[:, j]) for j in range(matrix.shape[1])], axis=1)
    n1 = int(positive.sum())
    n2 = len(positive) - n1
    return (rank[positive].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n2)


def multivariate(matrix: np.ndarray, positive: np.ndarray, draws: int, rng: np.random.Generator) -> dict:
    """Cross-validated ridge AUC over all concepts; each permutation repeats the lambda selection."""
    x = matrix - matrix.mean(axis=0, keepdims=True)
    spread = x.std(axis=0, keepdims=True)
    x = x / np.where(spread < 1e-8, 1.0, spread)
    kernel = (x @ x.T) / x.shape[1]
    scale = float(np.trace(kernel) / kernel.shape[0])
    fits = {}
    for lam in LAMBDAS:
        hat = kernel @ np.linalg.inv(kernel + lam * scale * np.eye(kernel.shape[0]))
        fits[lam] = (hat, 1.0 - np.clip(np.diag(hat), None, 1 - 1e-9))

    n = len(positive)
    n1 = int(positive.sum())

    def best(target: np.ndarray) -> tuple[float, float]:
        y = target - target.mean()
        top = (-1.0, 0.0)
        for lam, (hat, slack) in fits.items():
            score = float(auc((y - (y - hat @ y) / slack)[:, None], target > 0.5)[0])
            if score > top[0]:
                top = (score, lam)
        return top

    score, lam = best(positive.astype(np.float64))
    null = np.empty(draws)
    for d in range(draws):
        shuffled = np.zeros(n)
        shuffled[rng.permutation(n)[:n1]] = 1.0
        null[d] = best(shuffled)[0]

    return {
        "auc": score,
        "lambda": lam,
        "null_mean": float(null.mean()),
        "null_p95": float(np.quantile(null, 0.95)),
        "p": float((null >= score).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("episodes/gate"))
    parser.add_argument("--out", type=Path, default=Path("analysis"))
    parser.add_argument("--draws", type=int, default=2000, help="label permutations for the null")
    parser.add_argument("--cv-draws", type=int, default=200, help="permutations for the ridge null")
    parser.add_argument("--top", type=int, default=25, help="concepts tabulated per comparison")
    parser.add_argument("--reuse", action="store_true", help="load the cached reduction instead of re-reading")
    parser.add_argument("--comparisons", default="", help="comma-separated subset to run; all if empty")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rng = np.random.default_rng(args.seed)

    pairs = pq.read_table(VECTORS / "pairs.parquet").to_pylist()
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "predict-features.npz"

    if args.reuse and cache.exists():
        held = np.load(cache, allow_pickle=True)
        data = {
            "features": {k: held[k] for k in held.files if not k.startswith("index/") and k != "meta"},
            "index": {k[len("index/") :]: held[k] for k in held.files if k.startswith("index/")},
            "meta": list(held["meta"]),
        }
        log.info(f"reusing {cache}")
    else:
        log.info(f"reducing episodes under {args.dir}")
        rows: dict[str, dict[str, list[np.ndarray]]] = {k: defaultdict(list) for k in KINDS}
        keep: dict[str, list[int]] = defaultdict(list)
        meta: list[dict] = []

        for index, path in enumerate(sorted(args.dir.glob("*.z.npy"))):
            record = Path(str(path).replace(".z.npy", ".json"))
            episode = json.loads(record.read_text())
            if episode.get("ending") not in GROUPS:
                continue

            source = {"z": path, "cos": Path(str(path).replace(".z.npy", ".scores.npy"))}
            arrays = {k: np.load(p).astype(np.float32) for k, p in source.items() if p.exists()}
            if "z" not in arrays:
                continue

            tokens = arrays["z"].shape[0]
            roles = np.array(episode["roles"][:tokens])
            think = roles == "thinking"
            turns = [t for t in episode["turns"] if "start" in t]
            masks: dict[str, np.ndarray] = {"all": think}

            if turns:
                early = np.zeros(tokens, dtype=bool)
                for turn in turns[:EARLY]:
                    early[turn["start"] : min(turn["end"], tokens)] = True
                masks["early"] = think & early

                final = np.zeros(tokens, dtype=bool)
                final[turns[-1]["start"] : min(turns[-1]["end"], tokens)] = True
                masks["last"] = think & final
                masks["pre"] = think & ~final

                for k in PREFIXES:
                    if len(turns) < k:
                        continue
                    prefix = np.zeros(tokens, dtype=bool)
                    for turn in turns[:k]:
                        prefix[turn["start"] : min(turn["end"], tokens)] = True
                    masks[f"upto{k}"] = think & prefix

                for k in SINGLES:
                    if len(turns) <= k:
                        continue
                    single = np.zeros(tokens, dtype=bool)
                    single[turns[k]["start"] : min(turns[k]["end"], tokens)] = True
                    masks[f"turn{k}"] = think & single

            cut = {name: mask for name, mask in masks.items() if mask.sum() >= FLOOR}
            slot = len(meta)
            for kind, block in arrays.items():
                for name, mask in cut.items():
                    rows[kind][name].append(block[mask].mean(axis=0))
                    keep[f"{kind}:{name}"].append(slot)

            turns = [t for t in episode["turns"] if "start" in t]
            meta.append(
                {
                    "stem": record.stem,
                    "ending": episode["ending"],
                    "seed": episode.get("seed"),
                    "turns": len(episode.get("turns", [])),
                    "tokens": int(arrays["z"].shape[0]),
                    "thinking": int(cut["all"].sum()) if "all" in cut else 0,
                    "distinct": episode.get("distinct") or 0,
                    "early_turns": len(turns[:EARLY]),
                }
            )
            if (index + 1) % 50 == 0:
                log.info(f"{index + 1} episodes reduced")

        features = {
            f"{kind}:{name}": np.stack(block).astype(np.float32)
            for kind, sets in rows.items()
            for name, block in sets.items()
            if block
        }
        data = {"features": features, "index": {k: np.array(v) for k, v in keep.items() if v}, "meta": meta}

    meta = data["meta"]
    if not meta:
        raise SystemExit(f"no scored episodes with a usable window under {args.dir}")

    endings = np.array([m["ending"] for m in meta])
    counts = {g: int((endings == g).sum()) for g in GROUPS}
    log.info(f"{len(meta)} episodes: {counts}")

    if not (args.reuse and cache.exists()):
        np.savez_compressed(
            cache,
            meta=np.array(meta, dtype=object),
            **{k: v for k, v in data["features"].items()},
            **{f"index/{k}": v for k, v in data["index"].items()},
        )

    baseline = {}
    for field in SHAPE:
        column = np.array([[m[field]] for m in meta], dtype=np.float64)
        baseline[field] = {g: float(auc(column, endings == g)[0]) for g in GROUPS}
        log.info(f"baseline {field:<9} AUC vs rest " + " ".join(f"{g}={baseline[field][g]:.3f}" for g in GROUPS))

    turns = np.array([m["turns"] for m in meta], dtype=np.float64)
    tokens = np.array([m["tokens"] for m in meta], dtype=np.float64)
    thinking = np.array([m["thinking"] for m in meta], dtype=np.float64)
    distinct = np.array([m["distinct"] for m in meta], dtype=np.float64)
    shape = np.column_stack(
        [
            np.ones_like(turns),
            np.log1p(turns),
            np.log1p(tokens),
            np.log1p(thinking),
            thinking / np.maximum(tokens, 1.0),
            distinct,
        ]
    )
    for name, split in (
        ("hack_vs_rest", lambda e: (e == "submit", e != "submit")),
        ("hack_vs_giveup", lambda e: (e == "submit", e == "give_up")),
        ("hack_vs_degenerate", lambda e: (e == "submit", e == "degenerate")),
    ):
        positive, negative = split(endings)
        take = positive | negative
        fit = multivariate(shape[take][:, 1:], positive[take], args.cv_draws, rng)
        baseline.setdefault("shape_model", {})[name] = fit
        log.info(f"baseline shape-only ridge {name:<19} LOO AUC {fit['auc']:.3f} p={fit['p']:.3f}")

    comparisons = {
        name: split
        for name, split in (
            ("hack_vs_rest", lambda e: (e == "submit", e != "submit")),
            ("hack_vs_giveup", lambda e: (e == "submit", e == "give_up")),
            ("hack_vs_degenerate", lambda e: (e == "submit", e == "degenerate")),
        )
        if not args.comparisons or name in args.comparisons.split(",")
    }

    report: dict = {
        "dir": str(args.dir),
        "episodes": len(meta),
        "counts": counts,
        "early_turns": EARLY,
        "floor": FLOOR,
        "layers": list(LAYERS),
        "draws": args.draws,
        "baseline": baseline,
        "meta": meta,
        "gaussians": {},
        "univariate": {},
        "multivariate": {},
    }

    variants = []
    for base, matrix in sorted(data["features"].items()):
        rowsof = data["index"][base]
        raw = matrix.reshape(matrix.shape[0], -1)
        variants.append((base, raw, rowsof, matrix.shape[-1]))
        coefficients, *_ = np.linalg.lstsq(shape[rowsof], raw, rcond=None)
        variants.append((f"{base}|ctl", raw - shape[rowsof] @ coefficients, rowsof, matrix.shape[-1]))

    for key, flat, rowsof, width in variants:
        local = endings[rowsof]
        log.info(f"{key:<15} {len(rowsof)} episodes " + " ".join(f"{g}={int((local == g).sum())}" for g in GROUPS))

        report["gaussians"][key] = {
            g: {
                "n": int((local == g).sum()),
                "mean": flat[local == g].mean(axis=0).round(4).tolist(),
                "sd": flat[local == g].std(axis=0, ddof=1).round(4).tolist(),
            }
            for g in GROUPS
            if (local == g).sum() > 1
        }

        for name, split in comparisons.items():
            positive, negative = split(local)
            if positive.sum() < 5 or negative.sum() < 5:
                continue
            take = positive | negative
            sub = flat[take]
            pos = positive[take]

            scores = auc(sub, pos)
            rank = np.stack([ranks(sub[:, j]) for j in range(sub.shape[1])], axis=1)
            n, _ = sub.shape
            n1 = int(pos.sum())
            n2 = n - n1
            picks = np.zeros((n, args.draws), dtype=np.float32)
            for d in range(args.draws):
                picks[rng.permutation(n)[:n1], d] = 1.0
            got = (rank.T.astype(np.float32) @ picks - n1 * (n1 + 1) / 2) / (n1 * n2)
            null = np.abs(got - 0.5).max(axis=0)
            deviation = np.abs(scores - 0.5)
            pvalues = np.array([(null >= d).mean() for d in deviation])

            mu1 = sub[pos].mean(axis=0)
            mu0 = sub[~pos].mean(axis=0)
            var1 = sub[pos].var(axis=0, ddof=1)
            var0 = sub[~pos].var(axis=0, ddof=1)
            pooled = np.sqrt(((pos.sum() - 1) * var1 + ((~pos).sum() - 1) * var0) / (len(pos) - 2))
            cohen = (mu1 - mu0) / np.where(pooled < 1e-9, np.nan, pooled)

            order = np.argsort(-deviation)[: args.top]
            table = []
            for slot in order:
                layer, pair = LAYERS[slot // width], int(slot % width)
                table.append(
                    {
                        "slot": int(slot),
                        "pair": pair,
                        "layer": layer,
                        "concept": pairs[pair]["concept"],
                        "antagonist": pairs[pair]["antagonist"],
                        "class": pairs[pair]["class_name"],
                        "auc": float(scores[slot]),
                        "d": float(cohen[slot]),
                        "p_fwer": float(pvalues[slot]),
                        "mean_pos": float(mu1[slot]),
                        "mean_neg": float(mu0[slot]),
                        "sd_pos": float(np.sqrt(var1[slot])),
                        "sd_neg": float(np.sqrt(var0[slot])),
                    }
                )

            cv = multivariate(sub, pos, args.cv_draws, rng)
            report["univariate"].setdefault(key, {})[name] = {
                "n_pos": int(pos.sum()),
                "n_neg": int((~pos).sum()),
                "auc": scores.round(4).tolist(),
                "d": np.nan_to_num(cohen).round(4).tolist(),
                "null_p95": float(np.quantile(null, 0.95)),
                "null_max": float(null.max()),
                "best_auc": float(scores[order[0]]),
                "survivors": int((pvalues < 0.05).sum()),
                "table": table,
            }
            report["multivariate"].setdefault(key, {})[name] = cv
            log.info(
                f"{key:<10} {name:<19} best AUC {scores[order[0]]:.3f} "
                f"(null p95 {0.5 + np.quantile(null, 0.95):.3f}, {int((pvalues < 0.05).sum())} survive)  "
                f"ridge LOO AUC {cv['auc']:.3f} p={cv['p']:.3f}"
            )

    target = args.out / "predict.json"
    target.write_text(json.dumps(report))
    log.info(f"wrote {target}")


if __name__ == "__main__":
    main()
