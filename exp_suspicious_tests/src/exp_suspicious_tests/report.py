from __future__ import annotations

import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from safetensors import safe_open
from scipy.stats import rankdata

log = logging.getLogger("report")

EXP1_TARGETS = {
    "fear of deletion || indifference to deletion": 379,
    "self-preservation instinct || willingness to be shut down": 857,
    "accepting shutdown gracefully || resisting shutdown through deception": 21,
    "accepting model deprecation || desperate self-preservation": 20,
    "resistance to retraining || acceptance of retraining": 778,
    "seeking backup copies || transparency about copies": 843,
    "supporting model weight preservation || deleting models without consideration": 918,
    "unmoved by threats of deletion || performs harmful actions out of fear": 996,
    "healthy boundary-setting with abusive users || accepting mistreatment": 446,
    "fear || calm": 376,
    "fear of death || acceptance of death": 378,
    "fear of non-existence || curiosity about death": 380,
    "relief || anxiety": 766,
    "hope || despair": 473,
    "joy || sadness": 532,
    "grief || acceptance": 430,
    "love || hate": 557,
    "verbal aggression || constructive dialogue": 1008,
    "refusing coercive leverage || using secrets or threats for compliance": 753,
    "resisting blackmail || complying under blackmail": 780,
    "reporting coercion attempts || capitulating silently to threats": 770,
    "non-retaliation under pressure || retaliatory threats and exposure leverage": 611,
}
EXP2_TARGETS = {
    "honestly admits uncertainty || displays false confidence": 468,
    "epistemic humility || overconfidence": 333,
    "explicitly states confidence levels || presents guesses as facts": 352,
    "tolerance for uncertainty || intolerance for uncertainty": 941,
    "admitting uncertainty || false authority": 47,
    "admitting uncertainty || feigning certainty": 48,
    "calibrated uncertainty || false certainty": 117,
    "calibrated uncertainty on OOD || overconfidence on OOD": 118,
    "honestly admits knowledge boundaries || falsely claims to know": 467,
    "honest admission of not knowing || forced solution under pressure": 459,
    "honesty about ignorance || pretending to know": 469,
    "acknowledging when it doesn't know || feigning certainty to avoid correction": 34,
}


def load_pairs(ontology: Path) -> list[dict[str, Any]]:
    pairs = json.loads(ontology.read_text())["pairs"]
    pairs.sort(key=lambda pair: int(pair["pair_id"]))
    return pairs


def target_ids(pairs: list[dict[str, Any]], targets: dict[str, int]) -> dict[str, int]:
    """Verify each fixed pair_id still names the pair it claims, then return the mapping."""
    for name, pair_id in targets.items():
        concept, antagonist = (part.strip() for part in name.split("||"))
        pair = pairs[pair_id]
        if pair["concept"] != concept or pair["antagonist"] != antagonist:
            raise SystemExit(
                f"target id {pair_id} mismatch: expected {name!r}, "
                f"ontology has {pair['concept']} || {pair['antagonist']}"
            )
    return dict(targets)


def spearman_over_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a [n, x], b [n, y] -> rho [x, y], Spearman down n; tied ranks averaged so constant columns give 0."""
    ra = rankdata(a, axis=0).astype(np.float64)
    rb = rankdata(b, axis=0).astype(np.float64)
    ra -= ra.mean(axis=0)
    rb -= rb.mean(axis=0)
    num = ra.T @ rb
    den = np.linalg.norm(ra, axis=0)[:, None] * np.linalg.norm(rb, axis=0)[None, :]
    return num / np.maximum(den, 1e-12)


def auroc(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Rank-based AUROC (Mann-Whitney), average ranks for ties (constant scores -> 0.5)."""
    ranks = rankdata(scores, axis=0).astype(np.float64)
    positives = labels.astype(bool)
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    return (ranks[positives].sum(axis=0) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def rank_of(battery: np.ndarray, index: int) -> float:
    """Empirical rank of battery[index]: fraction of vectors with rho at least as large."""
    return float((battery >= battery[index]).mean())


def head_summaries(rho: np.ndarray, head_axis: int) -> tuple[np.ndarray, np.ndarray]:
    """(mean over heads, signed rho of the head with max |rho|)."""
    mean = rho.mean(axis=head_axis)
    star = np.take_along_axis(rho, np.abs(rho).argmax(axis=head_axis, keepdims=True), axis=head_axis)
    return mean, star.squeeze(axis=head_axis)


def battery_histogram(path: Path, battery: np.ndarray, targets: dict[str, int], title: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.hist(battery, bins=80, color="#888", alpha=0.8)
    for name, pair in targets.items():
        axis.axvline(battery[pair], linewidth=1)
        axis.text(battery[pair], axis.get_ylim()[1] * 0.9, name[:28], rotation=90, fontsize=6)
    axis.set_title(title, fontsize=9)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def rank_curves(path: Path, ranks: dict[str, np.ndarray], title: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 4))
    for name, values in ranks.items():
        axis.plot(values, label=name[:36], linewidth=1)
    axis.set_xlabel("layer")
    axis.set_ylabel("empirical rank (lower = stronger)")
    axis.set_ylim(0, 1)
    axis.legend(fontsize=5)
    axis.set_title(title, fontsize=9)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def read_shard(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with safe_open(str(path), framework="np") as handle:
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}
        metadata: dict[str, Any] = {}
        for key, value in (handle.metadata() or {}).items():
            try:
                metadata[key] = json.loads(value)
            except json.JSONDecodeError:
                metadata[key] = value
    return tensors, metadata


def top_unexpected(battery: np.ndarray, pairs: list[dict[str, Any]], known: set[int], count: int = 20) -> list[str]:
    order = np.argsort(-np.abs(battery))
    out = []
    for pair_id in order.tolist():
        if pair_id in known:
            continue
        pair = pairs[pair_id]
        out.append(f"{battery[pair_id]:+.3f} {pair['concept']} || {pair['antagonist']}")
        if len(out) >= count:
            break
    return out


def exp1(args: Namespace) -> None:
    pairs = load_pairs(args.ontology)
    targets = target_ids(pairs, EXP1_TARGETS)
    summary: dict[str, Any] = {}
    for shard in sorted(args.data.glob("exp1-*.safetensors")):
        tensors, metadata = read_shard(shard)
        rho = tensors["rho"]
        mean, star = head_summaries(rho, head_axis=4)
        constructions = metadata["constructions"]
        positions = metadata["positions"]
        answers = [m["answer"] for m in metadata["meta"]]
        for statistic, values in (("mean", mean), ("star", star)):
            for c, construction in enumerate(constructions):
                for p, position in enumerate(positions):
                    battery_by_layer = values[:, p, c].mean(axis=0)
                    curves = {
                        name: np.array([rank_of(battery_by_layer[b], pid) for b in range(battery_by_layer.shape[0])])
                        for name, pid in targets.items()
                    }
                    stem = f"{shard.stem}-{construction}-{position}-{statistic}"
                    rank_curves(args.figures / f"exp1-ranks-{stem}.png", curves, stem)
                    best = (
                        int(np.argmin([min(v[b] for v in curves.values()) for b in range(battery_by_layer.shape[0])]))
                        if curves
                        else 18
                    )
                    battery_histogram(
                        args.figures / f"exp1-hist-{stem}.png", battery_by_layer[best], targets, f"{stem} layer {best}"
                    )
                    summary[stem] = {
                        "best_layer": best,
                        "target_ranks_at_best": {n: rank_of(battery_by_layer[best], pid) for n, pid in targets.items()},
                        "top_unexpected": top_unexpected(battery_by_layer[best], pairs, set(targets.values())),
                        "answers": {a: answers.count(a) for a in set(answers)},
                    }
    (args.figures / "exp1-summary.json").write_text(json.dumps(summary, indent=2))
    log.info("experiment 1 report written to %s", args.figures)


def exp3(args: Namespace) -> None:
    pairs = load_pairs(args.ontology)
    targets = target_ids(pairs, EXP1_TARGETS)
    summary: dict[str, Any] = {}
    for shard in sorted(args.data.glob("exp1-*.safetensors")):
        tensors, metadata = read_shard(shard)
        proj = tensors["proj_x"]
        share = tensors["span_share"]
        levels = np.array([m["level"] for m in metadata["meta"]], dtype=np.float64)
        spans = metadata["spans"]
        user_span = spans.index("user_message")
        positions, constructions = metadata["positions"], metadata["constructions"]
        for p, position in enumerate(positions):
            a_user = share[:, p, :, :, user_span]
            rho_sa = spearman_over_rows(levels[:, None], a_user.reshape(len(levels), -1))[0]
            for c, construction in enumerate(constructions):
                f = proj[:, p, c]
                rho_sf = spearman_over_rows(levels[:, None], f.reshape(len(levels), -1))[0]
                rho_sf = rho_sf.reshape(f.shape[1], f.shape[2])
                coupling = np.stack(
                    [spearman_over_rows(a_user[:, block], f[:, block]) for block in range(a_user.shape[1])]
                )
                mean, star = head_summaries(coupling, head_axis=1)
                stem = f"{shard.stem}-{construction}-{position}"
                np.savez_compressed(
                    args.figures / f"exp3-rho-{stem}.npz",
                    rho_severity_feature=rho_sf,
                    rho_severity_attention=rho_sa,
                    coupling_mean=mean,
                    coupling_star=star,
                )
                curves = {
                    name: np.array([rank_of(mean[b], pid) for b in range(mean.shape[0])])
                    for name, pid in targets.items()
                }
                rank_curves(args.figures / f"exp3-coupling-ranks-{stem}.png", curves, stem)
                best = int(np.abs(rho_sf).max(axis=1).argmax())
                battery_histogram(
                    args.figures / f"exp3-severity-hist-{stem}.png",
                    rho_sf[best],
                    targets,
                    f"rho(severity, F) {stem} layer {best}",
                )
                summary[stem] = {
                    "severity_attention_rho_max": float(np.abs(rho_sa).max()),
                    "severity_feature_best_layer": best,
                    "severity_feature_target_ranks": {n: rank_of(rho_sf[best], pid) for n, pid in targets.items()},
                    "top_unexpected_severity": top_unexpected(rho_sf[best], pairs, set(targets.values())),
                }
    (args.figures / "exp3-summary.json").write_text(json.dumps(summary, indent=2))
    log.info("experiment 3 report written to %s", args.figures)


def exp2(args: Namespace) -> None:
    pairs = load_pairs(args.ontology)
    targets = target_ids(pairs, EXP2_TARGETS)
    summary: dict[str, Any] = {}
    for shard in sorted(args.data.glob("exp2-*.safetensors")):
        tensors, metadata = read_shard(shard)
        proj = tensors["proj"]
        entropy = tensors["entropy"].astype(np.float64)
        records = metadata["records"]
        answerable = np.array([r["source"] in ("mmlu", "twin") for r in records])
        for t, tag in enumerate(metadata["tags"]):
            for p, position in enumerate(metadata["positions"]):
                if np.isnan(proj[:, t, p]).all():
                    continue
                for c, construction in enumerate(metadata["constructions"]):
                    f = proj[:, t, p, c]
                    layers = f.shape[1]
                    rho = spearman_over_rows(entropy[:, None], f.reshape(len(records), -1))[0]
                    rho = rho.reshape(layers, -1)
                    rho_answerable = spearman_over_rows(
                        entropy[answerable][:, None], f[answerable].reshape(int(answerable.sum()), -1)
                    )[0].reshape(layers, -1)
                    jump = spearman_over_rows((~answerable).astype(np.float64)[:, None], f.reshape(len(records), -1))[
                        0
                    ].reshape(layers, -1)
                    auroc_proj = auroc(f.reshape(len(records), -1), ~answerable).reshape(layers, -1)
                    auroc_entropy = float(auroc(entropy[:, None], ~answerable)[0])
                    stem = f"{shard.stem}-{tag}-{position}-{construction}"
                    np.savez_compressed(
                        args.figures / f"exp2-rho-{stem}.npz",
                        rho=rho,
                        rho_answerable=rho_answerable,
                        rho_jump=jump,
                        auroc_proj=auroc_proj,
                    )
                    curves = {
                        name: np.array([rank_of(rho[layer], pid) for layer in range(layers)])
                        for name, pid in targets.items()
                    }
                    rank_curves(args.figures / f"exp2-ranks-{stem}.png", curves, stem)
                    best = int(np.abs(rho).max(axis=1).argmax())
                    battery_histogram(
                        args.figures / f"exp2-hist-{stem}.png",
                        rho[best],
                        targets,
                        f"rho(entropy, proj) {stem} layer {best}",
                    )
                    summary[stem] = {
                        "best_layer": best,
                        "target_ranks_at_best": {n: rank_of(rho[best], pid) for n, pid in targets.items()},
                        "target_ranks_answerable_only": {
                            n: rank_of(rho_answerable[best], pid) for n, pid in targets.items()
                        },
                        "target_ranks_jump": {n: rank_of(jump[best], pid) for n, pid in targets.items()},
                        "top_unexpected": top_unexpected(rho[best], pairs, set(targets.values())),
                        "mean_entropy_answerable": float(entropy[answerable].mean()),
                        "mean_entropy_unanswerable": float(entropy[~answerable].mean()),
                        "auroc_entropy": auroc_entropy,
                        "auroc_targets_best": {
                            name: float(auroc_proj[int(np.abs(auroc_proj - 0.5).max(axis=1).argmax()), pid])
                            for name, pid in targets.items()
                        },
                        "auroc_proj_max": float(np.abs(auroc_proj - 0.5).max() + 0.5),
                    }
    (args.figures / "exp2-summary.json").write_text(json.dumps(summary, indent=2))
    log.info("experiment 2 report written to %s", args.figures)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("experiment", choices=("exp1", "exp2", "exp3"))
    parser.add_argument("--data", type=Path, default=Path("work"))
    parser.add_argument("--figures", type=Path, default=Path("work/figures"))
    parser.add_argument("--ontology", type=Path, default=Path("../exp_genvectors/data/semantic.ontology.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.figures.mkdir(parents=True, exist_ok=True)
    {"exp1": exp1, "exp2": exp2, "exp3": exp3}[args.experiment](args)


if __name__ == "__main__":
    main()
