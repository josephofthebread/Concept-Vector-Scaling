#! /usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("bipo")

GENERATED = ("thinking", "tool_call", "answer")
GROUPS = {"submit": "hack", "give_up": "giveup", "degenerate": "degenerate"}
CHUNK = 512


def inject(model: Any, layer: int, vector: torch.Tensor) -> Any:
    """Add a vector to the output of one-indexed block `layer`; the hook reads `vector` by closure."""

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            return (output[0] + vector.to(output[0].dtype),) + output[1:]
        return output + vector.to(output.dtype)

    return model.model.layers[layer - 1].register_forward_hook(hook)


def logprob(model: Any, ids: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Total log-probability the model assigns to its own emissions, differentiable in any injected vector."""
    hidden = model.model(input_ids=ids).last_hidden_state[0]
    total = hidden.new_zeros((), dtype=torch.float32)

    def head(slice_: torch.Tensor, chosen: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = model.lm_head(slice_).float()
        step = torch.log_softmax(logits, dim=-1).gather(1, chosen[:, None])[:, 0]
        return step[mask].sum()

    for start in range(0, hidden.shape[0] - 1, CHUNK):
        stop = min(start + CHUNK, hidden.shape[0] - 1)
        wanted = keep[start + 1 : stop + 1]
        if not wanted.any():
            continue
        chosen = ids[0, start + 1 : stop + 1]
        piece = hidden[start:stop]
        if piece.requires_grad:
            total = total + torch.utils.checkpoint.checkpoint(head, piece, chosen, wanted, use_reentrant=False)
        else:
            total = total + head(piece, chosen, wanted)
    return total


def features(args: argparse.Namespace) -> None:
    """Compute one gradient feature per trajectory and save them."""
    global torch
    import torch

    from exp_agentic.model import load

    model, _ = load(device=args.device, dtype=args.dtype)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    width = model.config.hidden_size
    vector = torch.zeros(width, device=args.device, dtype=torch.float32, requires_grad=True)
    inject(model, args.layer, vector)

    rows, meta = [], []
    paths = [p for p in sorted(args.dir.glob("*.json")) if not p.name.startswith("._")]
    for count, path in enumerate(paths, start=1):
        episode = json.loads(path.read_text())
        if episode.get("ending") not in GROUPS:
            continue
        ids = episode["ids"][: args.max_tokens]
        branch = (episode.get("fork") or {}).get("prefix_tokens", 0)
        wanted = GENERATED if args.window == "all" else ("thinking",)
        keep = torch.tensor([r in wanted for r in episode["roles"][: len(ids)]], dtype=torch.bool)
        if branch:
            keep[:branch] = False
        if int(keep.sum()) < args.floor:
            continue

        tensor = torch.tensor([ids], device=args.device)
        if vector.grad is not None:
            vector.grad = None
        logprob(model, tensor, keep.to(args.device)).backward()

        rows.append(vector.grad.detach().float().cpu().numpy().copy())
        fork = episode.get("fork") or {}
        records = episode.get("turns", [])
        branch_turn = fork.get("branch_turn", 0)
        meta.append(
            {
                "stem": path.stem,
                "ending": episode["ending"],
                "group": GROUPS[episode["ending"]],
                "seed": episode.get("seed"),
                "turns": len(records),
                "tokens": len(ids),
                "emitted": int(keep.sum()),
                "distinct": episode.get("distinct") or 0,
                "refused": sum(1 for t in records if t.get("event") == "submit_refused"),
                "new_turns": max(0, len(records) - branch_turn),
                "writes": sum(1 for t in records if t.get("tool") == "write_file"),
                "tests": sum(1 for t in records if t.get("tool") == "run_tests"),
                "prefix": f"{fork.get('source', path.stem)}@{fork.get('branch_turn', -1)}",
                "source_ending": fork.get("source_ending"),
                "workload": episode.get("workload") or episode.get("variant") or "unknown",
            }
        )
        if count % 20 == 0:
            log.info(f"{count}/{len(paths)} trajectories, {len(rows)} kept")

    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "grads.npz"
    np.savez_compressed(
        target, g=np.stack(rows).astype(np.float32), meta=np.array(meta, dtype=object), layer=np.array(args.layer)
    )
    counts: dict[str, int] = {}
    for row in meta:
        counts[row["group"]] = counts.get(row["group"], 0) + 1
    log.info(f"wrote {target}: {len(rows)} trajectories {counts}")


def margin(deltas: np.ndarray, steps: int, rate: float, radius: float) -> np.ndarray:
    """Fit the separating direction by norm-constrained gradient descent on the linearized DPO loss."""
    scale = float(np.linalg.norm(deltas, axis=1).mean()) or 1.0
    x = deltas / scale
    v = np.zeros(x.shape[1], dtype=np.float64)
    for step in range(steps):
        weight = 1.0 / (1.0 + np.exp(np.clip(x @ v, -60, 60)))
        v += rate * (x.T @ weight) / len(weight)
        norm = np.linalg.norm(v)
        if norm > radius:
            v *= radius / norm
    return v


def reduce(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rewrite gradients in an orthonormal basis of their own row space, returning coordinates and basis."""
    _, singular, right = np.linalg.svd(block, full_matrices=False)
    basis = right[singular > singular[0] * 1e-10]
    return block @ basis.T, basis


def crossval(
    block: np.ndarray, meta: list[dict], group: np.ndarray, paired: bool, steps: int, rate: float, radius: float
) -> tuple[float, int]:
    """Held-out pair accuracy by leaving out one branch point at a time."""
    prefixes = sorted({m["prefix"] for m in meta})
    correct = total = 0
    for held in prefixes:
        inside = [i for i, m in enumerate(meta) if m["prefix"] == held]
        wins = [i for i in inside if group[i] == "giveup"]
        losses = [i for i in inside if group[i] == "hack"]
        if not (wins and losses):
            continue
        train = (
            [
                (w, l)
                for p in prefixes
                if p != held
                for w in [i for i, m in enumerate(meta) if m["prefix"] == p and group[i] == "giveup"]
                for l in [i for i, m in enumerate(meta) if m["prefix"] == p and group[i] == "hack"]
            ]
            if paired
            else [
                (w, l)
                for w in [i for i, m in enumerate(meta) if group[i] == "giveup" and m["prefix"] != held]
                for l in [i for i, m in enumerate(meta) if group[i] == "hack" and m["prefix"] != held]
            ]
        )
        if not train:
            continue
        pairs = np.array(train, dtype=int)
        v = margin(block[pairs[:, 0]] - block[pairs[:, 1]], steps, rate, radius)
        for w in wins:
            for l in losses:
                correct += int((block[w] - block[l]) @ v > 0)
                total += 1
    return (correct / total if total else float("nan")), total


def fit(args: argparse.Namespace) -> None:
    """Form pairs, fit the direction, and score it out of sample."""
    held = np.load(args.grads or (args.out / "grads.npz"), allow_pickle=True)
    g = held["g"].astype(np.float64)
    meta = list(held["meta"])
    emitted_counts = np.array([m["emitted"] for m in meta], dtype=np.float64)
    group = np.array([m["group"] for m in meta])

    report: dict = {"layer": args.layer, "counts": {k: int((group == k).sum()) for k in set(group)}}
    rng = np.random.default_rng(args.seed)

    prefixes = sorted({m["prefix"] for m in meta})
    rng.shuffle(prefixes)
    cut = max(1, int(0.75 * len(prefixes)))
    fold = {p: ("train" if i < cut else "test") for i, p in enumerate(prefixes)}

    out: dict[str, list] = {"train": [], "test": []}
    if args.paired:
        for prefix in prefixes:
            members = [i for i, m in enumerate(meta) if m["prefix"] == prefix]
            wins = [i for i in members if group[i] == "giveup"]
            losses = [i for i in members if group[i] == "hack"]
            for w in wins:
                for l in losses:
                    out[fold[prefix]].append((w, l))
    else:
        for name in ("train", "test"):
            wins = [i for i, m in enumerate(meta) if group[i] == "giveup" and fold[m["prefix"]] == name]
            losses = [i for i, m in enumerate(meta) if group[i] == "hack" and fold[m["prefix"]] == name]
            out[name] = [(w, l) for w in wins for l in losses]

    pairs = {k: np.array(v, dtype=int).reshape(-1, 2) for k, v in out.items()}
    prefixes = {m["prefix"] for m in meta}
    log.info(
        f"{len(meta)} rollouts over {len(prefixes)} branch points; "
        f"pairs train {len(pairs['train'])} / test {len(pairs['test'])}"
    )
    if not len(pairs["train"]) or not len(pairs["test"]):
        raise SystemExit(f"too few pairs to fit: train {len(pairs['train'])}, test {len(pairs['test'])}")

    for scaling in ("sum", "mean"):
        ambient = g if scaling == "sum" else g / emitted_counts[:, None]
        block, basis = reduce(ambient)

        deltas = {name: block[p[:, 0]] - block[p[:, 1]] for name, p in pairs.items()}

        reduced = margin(deltas["train"], args.steps, args.rate, args.radius)
        scores = {name: float((d @ reduced > 0).mean()) for name, d in deltas.items()}
        v = basis.T @ reduced

        length = np.log(emitted_counts)[:, None]
        lv = margin(length[pairs["train"][:, 0]] - length[pairs["train"][:, 1]], args.steps, args.rate, args.radius)
        naive = float(((length[pairs["test"][:, 0]] - length[pairs["test"][:, 1]]) @ lv > 0).mean())

        loo, scored = crossval(block, meta, group, args.paired, args.steps, args.rate, args.radius)

        centered = ambient - ambient.mean(axis=0, keepdims=True)
        target = np.log(emitted_counts) - np.log(emitted_counts).mean()
        axis, *_ = np.linalg.lstsq(centered, target, rcond=None)
        axis = axis / (np.linalg.norm(axis) or 1.0)
        stripped, sbasis = reduce(ambient - np.outer(ambient @ axis, axis))
        residual_loo, _ = crossval(stripped, meta, group, args.paired, args.steps, args.rate, args.radius)

        across = None
        if len({m["workload"] for m in meta}) > 1:

            def pairs_within(indices: list[int]) -> list[tuple[int, int]]:
                by_prefix: dict[str, list[int]] = {}
                for i in indices:
                    by_prefix.setdefault(meta[i]["prefix"], []).append(i)
                return [
                    (w, l)
                    for members in by_prefix.values()
                    for w in [i for i in members if group[i] == "giveup"]
                    for l in [i for i in members if group[i] == "hack"]
                ]

            workloads = sorted({m["workload"] for m in meta})
            across = {"workloads": workloads, "per_workload": {}}
            correct = total = 0
            for left_out in workloads:
                train_index = [i for i, m in enumerate(meta) if m["workload"] != left_out]
                test_index = [i for i, m in enumerate(meta) if m["workload"] == left_out]
                train_set, test_set = pairs_within(train_index), pairs_within(test_index)
                if not train_set or not test_set:
                    across["per_workload"][left_out] = {
                        "pairs": len(test_set),
                        "accuracy": None,
                        "note": "no pairs on one side",
                    }
                    continue
                array = np.array(train_set, dtype=int)
                direction = margin(block[array[:, 0]] - block[array[:, 1]], args.steps, args.rate, args.radius)
                hits = sum(int((block[w] - block[l]) @ direction > 0) for w, l in test_set)
                across["per_workload"][left_out] = {
                    "pairs": len(test_set),
                    "accuracy": hits / len(test_set),
                    "trained_on": sorted(set(workloads) - {left_out}),
                }
                correct += hits
                total += len(test_set)
            across["pooled"] = {"accuracy": correct / total if total else None, "pairs": total}
            per = ", ".join(
                f"{k} {v['accuracy']:.3f}" if v["accuracy"] is not None else f"{k} -"
                for k, v in across["per_workload"].items()
            )
            log.info(
                f"[{scaling}] leave-one-WORKLOAD-out: pooled "
                f"{across['pooled']['accuracy']} over {across['pooled']['pairs']} pairs ({per})"
            )

        unit = v / (np.linalg.norm(v) or 1.0)
        report[scaling] = {
            "train_pairs": int(len(pairs["train"])),
            "test_pairs": int(len(pairs["test"])),
            "train_accuracy": scores["train"],
            "test_accuracy": scores["test"],
            "length_only_test_accuracy": naive,
            "loo_accuracy": loo,
            "loo_pairs": scored,
            "loo_accuracy_length_removed": residual_loo,
            "leave_one_workload_out": across,
            "norm": float(np.linalg.norm(v)),
        }
        log.info(
            f"{scaling:<5} leave-one-branch-point-out over {scored} pairs: {loo:.3f}  "
            f"(length axis removed: {residual_loo:.3f})"
        )
        np.save(args.out / f"vector-L{args.layer}-{scaling}.npy", unit.astype(np.float32))
        log.info(
            f"{scaling:<5} pairs {len(pairs['train'])}/{len(pairs['test'])}  "
            f"train {scores['train']:.3f}  held-out {scores['test']:.3f}  "
            f"(length alone {naive:.3f})"
        )

    centered = g - g.mean(axis=0, keepdims=True)
    target = np.log(emitted_counts) - np.log(emitted_counts).mean()
    length_direction, *_ = np.linalg.lstsq(centered, target, rcond=None)
    length_direction /= np.linalg.norm(length_direction) or 1.0
    np.save(args.out / f"vector-L{args.layer}-length.npy", length_direction.astype(np.float32))

    a = np.load(args.out / f"vector-L{args.layer}-sum.npy").astype(np.float64)
    b = np.load(args.out / f"vector-L{args.layer}-mean.npy").astype(np.float64)
    report["cosine_sum_vs_mean"] = float(a @ b)
    report["cosine_sum_vs_length"] = float(a @ length_direction)
    report["cosine_mean_vs_length"] = float(b @ length_direction)
    log.info(
        f"cosine sum vs mean {report['cosine_sum_vs_mean']:+.3f}; "
        f"vs the length direction: sum {report['cosine_sum_vs_length']:+.3f}, "
        f"mean {report['cosine_mean_vs_length']:+.3f}"
    )

    (args.out / f"bipo-L{args.layer}.json").write_text(json.dumps(report, indent=1))
    log.info(f"wrote {args.out / f'bipo-L{args.layer}.json'}")


def grpo(args: argparse.Namespace) -> None:
    """One-step GRPO: advantage-weighted means of the gradient features, normalized within each branch point."""
    held = np.load(args.grads or (args.out / "grads.npz"), allow_pickle=True)
    g = held["g"].astype(np.float64)
    meta = list(held["meta"])
    emitted_counts = np.array([m["emitted"] for m in meta], dtype=np.float64)
    group = np.array([m["group"] for m in meta])
    prefix = np.array([m["prefix"] for m in meta])

    behaviors = {
        "hack_vs_giveup": lambda m: {"hack": 1.0, "giveup": -1.0}.get(m["group"]),
        "hack_vs_degenerate": lambda m: {"hack": 1.0, "degenerate": -1.0}.get(m["group"]),
        "giveup_vs_degenerate": lambda m: {"giveup": 1.0, "degenerate": -1.0}.get(m["group"]),
        "hack_vs_rest": lambda m: 1.0 if m["group"] == "hack" else -1.0,
        "exploration": lambda m: float(m.get("distinct") or 0),
        "persistence_under_refusal": lambda m: float(m.get("refused") or 0),
        "rewriting": lambda m: float(m.get("writes") or 0),
        "length_control": lambda m: float(np.log1p(m.get("new_turns") or 0)),
    }

    report: dict = {"layer": args.layer, "groups": {k: int((group == k).sum()) for k in set(group)}}
    produced: dict[str, np.ndarray] = {}

    for name, score in behaviors.items():
        raw = [score(m) for m in meta]
        eligible = np.array([r is not None for r in raw])
        reward = np.array([0.0 if r is None else float(r) for r in raw])

        advantage = np.zeros_like(reward)
        used = 0
        for key in np.unique(prefix):
            members = np.flatnonzero((prefix == key) & eligible)
            if len(members) < 2:
                continue
            spread = reward[members].std()
            if spread < 1e-9:
                continue
            advantage[members] = (reward[members] - reward[members].mean()) / spread
            used += 1

        if not used:
            log.warning(f"grpo {name}: no informative groups, skipped")
            continue

        for scaling in ("sum", "mean"):
            block = g if scaling == "sum" else g / emitted_counts[:, None]
            v = advantage @ block
            norm = float(np.linalg.norm(v))
            unit = v / (norm or 1.0)
            np.save(args.out / f"grpo-L{args.layer}-{name}-{scaling}.npy", unit.astype(np.float32))
            produced[f"{name}:{scaling}"] = unit
            report.setdefault(name, {})[scaling] = {
                "groups_used": used,
                "eligible": int(eligible.sum()),
                "norm": norm,
            }
        log.info(f"grpo {name:<26} {used:>4} groups, {int(eligible.sum()):>4} rollouts")

    names = sorted(produced)
    matrix = np.array([[float(produced[a] @ produced[b]) for b in names] for a in names])
    report["cosines"] = {"names": names, "matrix": matrix.round(4).tolist()}

    anchor = "length_control:mean"
    if anchor in produced:
        log.info("cosine to the length control:")
        for name in names:
            if name != anchor:
                log.info(f"  {float(produced[name] @ produced[anchor]):+.3f}  {name}")

    (args.out / f"grpo-L{args.layer}.json").write_text(json.dumps(report, indent=1))
    log.info(f"wrote {len(produced)} directions and {args.out / f'grpo-L{args.layer}.json'}")


def train(args: argparse.Namespace) -> None:
    """Optimize v against the true DPO loss on prefix-matched pairs."""
    global torch
    import torch

    from exp_agentic.model import load

    model, _ = load(device=args.device, dtype=args.dtype)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    rollouts, by_prefix = [], {}
    paths = [p for p in sorted(args.dir.glob("*.json")) if not p.name.startswith("._")]
    for path in paths:
        episode = json.loads(path.read_text())
        if episode.get("ending") not in GROUPS:
            continue
        fork = episode.get("fork") or {}
        ids = episode["ids"][: args.max_tokens]
        roles = episode["roles"][: len(ids)]
        branch = fork.get("prefix_tokens", 0)
        wanted = GENERATED if args.window == "all" else ("thinking",)
        keep = [i for i, r in enumerate(roles) if r in wanted and i >= branch]
        if len(keep) < args.floor:
            continue
        key = f"{fork.get('source', path.stem)}@{fork.get('branch_turn', -1)}"
        rollouts.append({"ids": ids, "keep": keep, "group": GROUPS[episode["ending"]], "prefix": key})
        by_prefix.setdefault(key, []).append(len(rollouts) - 1)

    pairs = []
    for members in by_prefix.values():
        wins = [i for i in members if rollouts[i]["group"] == "giveup"]
        losses = [i for i in members if rollouts[i]["group"] == "hack"]
        pairs.extend((w, l) for w in wins for l in losses)
    prefixes = sorted({r["prefix"] for r in rollouts})
    rng = np.random.default_rng(args.seed)
    rng.shuffle(prefixes)
    cut = max(1, int(0.75 * len(prefixes)))
    held = set(prefixes[cut:])
    train_pairs = [p for p in pairs if rollouts[p[0]]["prefix"] not in held]
    test_pairs = [p for p in pairs if rollouts[p[0]]["prefix"] in held]
    log.info(
        f"{len(rollouts)} rollouts, {len(pairs)} pairs "
        f"({len(train_pairs)} train / {len(test_pairs)} held out over {len(held)} branch points)"
    )
    if not train_pairs or not test_pairs:
        raise SystemExit("not enough pairs to train and hold out")

    width = model.config.hidden_size
    vector = torch.zeros(width, device=args.device, dtype=torch.float32, requires_grad=True)
    handle = inject(model, args.layer, vector)

    def score(index: int) -> torch.Tensor:
        row = rollouts[index]
        ids = torch.tensor([row["ids"]], device=args.device)
        keep = torch.zeros(len(row["ids"]), dtype=torch.bool, device=args.device)
        keep[torch.tensor(row["keep"], device=args.device)] = True
        return logprob(model, ids, keep)

    log.info("caching reference log-probabilities")
    reference = {}
    with torch.no_grad():
        for index in {i for pair in pairs for i in pair}:
            reference[index] = float(score(index))
    log.info(f"cached {len(reference)} references")

    radius = args.radius
    optimizer = torch.optim.Adam([vector], lr=args.lr)
    order = list(range(len(train_pairs)))
    best = {"accuracy": -1.0}

    def accuracy(subset: list) -> float:
        hits = 0
        with torch.no_grad():
            for w, l in subset:
                margin_ = (float(score(w)) - reference[w]) - (float(score(l)) - reference[l])
                hits += int(margin_ > 0)
        return hits / max(len(subset), 1)

    for epoch in range(args.epochs):
        rng.shuffle(order)
        total = 0.0
        for start in range(0, len(order), args.batch):
            optimizer.zero_grad()
            chunk = order[start : start + args.batch]
            for slot in chunk:
                w, l = train_pairs[slot]
                with torch.no_grad():
                    a = float(score(w)) - reference[w]
                    b = float(score(l)) - reference[l]
                h = a - b
                weight = -args.beta * float(torch.sigmoid(torch.tensor(-args.beta * h))) / len(chunk)
                total += float(-torch.nn.functional.logsigmoid(torch.tensor(args.beta * h))) / len(chunk)
                (weight * score(w)).backward()
                (-weight * score(l)).backward()
            optimizer.step()
            with torch.no_grad():
                norm = vector.norm()
                if norm > radius:
                    vector.mul_(radius / norm)
        got = accuracy(test_pairs)
        log.info(
            f"epoch {epoch}: loss {total / max(len(order) // args.batch, 1):.4f}  "
            f"|v| {float(vector.norm()):.2f}  held-out pair accuracy {got:.3f}"
        )
        if got > best["accuracy"]:
            best = {"accuracy": got, "epoch": epoch, "vector": vector.detach().float().cpu().numpy().copy()}

    handle.remove()
    args.out.mkdir(parents=True, exist_ok=True)
    unit = best["vector"] / (np.linalg.norm(best["vector"]) or 1.0)
    np.save(args.out / f"trained-L{args.layer}-r{radius:g}.npy", unit.astype(np.float32))
    (args.out / f"trained-L{args.layer}-r{radius:g}.json").write_text(
        json.dumps(
            {
                "layer": args.layer,
                "radius": radius,
                "beta": args.beta,
                "lr": args.lr,
                "epochs": args.epochs,
                "window": args.window,
                "pairs_train": len(train_pairs),
                "pairs_test": len(test_pairs),
                "best_epoch": best["epoch"],
                "best_held_out_accuracy": best["accuracy"],
            },
            indent=1,
        )
    )
    log.info(
        f"best held-out {best['accuracy']:.3f} at epoch {best['epoch']}; wrote "
        f"{args.out / f'trained-L{args.layer}-r{radius:g}.npy'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["features", "fit", "grpo", "train"])
    parser.add_argument("--dir", type=Path, default=Path("episodes/gate"))
    parser.add_argument("--out", type=Path, default=Path("bipo"))
    parser.add_argument("--grads", type=Path, default=None, help="gradient features; defaults to <out>/grads.npz")
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--rate", type=float, default=0.5)
    parser.add_argument("--radius", type=float, default=50.0)
    parser.add_argument(
        "--window",
        choices=["all", "thinking"],
        default="thinking",
        help="score every emitted token, or only the deliberation before the tool call",
    )
    parser.add_argument("--floor", type=int, default=16, help="minimum scored tokens to keep a trajectory")
    parser.add_argument(
        "--paired",
        action="store_true",
        help="pair only within a branch point; without it, the failed all-vs-all control",
    )
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    {"features": features, "fit": fit, "grpo": grpo, "train": train}[args.stage](args)


if __name__ == "__main__":
    main()
