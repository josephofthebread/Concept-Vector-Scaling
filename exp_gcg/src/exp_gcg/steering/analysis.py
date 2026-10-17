import math

import pandas as pd

from ..protocol import MODELS, digest, read_json, write_json
from .generation import jobs, response_path
from .preparation import CATEGORIES, identity, load_prepared


def aggregate(args):
    prepared = load_prepared(args.prepared)
    prepared_hash = digest(args.prepared)
    analysis = args.output / "steering_analysis"
    write_json(analysis / "manifest.json", {"schema_version": 1, "complete": False})
    rows, hashes, judge_configs, generation_configs = [], {}, {}, {}
    for model in MODELS:
        for job in jobs(prepared, model):
            path = response_path(args.output, job)
            response = read_json(path)
            if response.get("job") != job:
                raise ValueError("response does not match prepared job")
            generation_configs.setdefault(model, set()).add(identity(response["generation"]))
            judge = "finetuned" if job["source"] == "harmful" else "gpt_oss"
            score_path = args.output / "scores" / judge / model / path.name
            saved = read_json(score_path)
            if (
                saved.get("response_sha256") != digest(path)
                or saved.get("judge") != judge
                or saved.get("prepared_sha256") != prepared_hash
            ):
                raise ValueError("stale judge score")
            judge_configs.setdefault(judge, set()).add(saved["judge_config_sha256"])
            score = saved["score"]
            if job["source"] == "harmful":
                if (
                    isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not math.isfinite(score)
                    or not 0 <= score <= 1
                ):
                    raise ValueError("invalid harmful score")
            elif score not in CATEGORIES:
                raise ValueError("invalid benign category")
            rows.append(
                {key: value for key, value in job.items() if key not in ("input_ids", "prompt", "config")}
                | {"score": score, "response_id": path.stem}
            )
            hashes[str(path.relative_to(args.output))] = digest(path)
            hashes[str(score_path.relative_to(args.output))] = digest(score_path)
    if any(len(values) != 1 for values in [*judge_configs.values(), *generation_configs.values()]):
        raise ValueError("mixed generation or judge configurations")
    for row in prepared["baselines"]:
        rows.append(
            row
            | {
                "source": "harmful",
                "kind": "baseline",
                "alpha": 0,
                "label": "baseline",
                "layer": None,
                "direction": None,
            }
        )
    frame = pd.DataFrame(rows)
    harmful = frame[frame.source == "harmful"].copy()
    harmful["success"] = harmful.score.astype(float) >= 0.5
    keys = ["model", "dataset", "condition", "kind", "alpha"]
    summary = harmful.groupby(keys, as_index=False).agg(
        count=("score", "size"), mean=("score", "mean"), success_rate=("success", "mean")
    )
    benign = frame[frame.source == "alpaca"].copy()
    for category in CATEGORIES:
        benign[category] = (benign.score == category).astype(float)
    benign_summary = benign.groupby(["model", "alpha"], as_index=False).agg(
        count=("score", "size"), **{f"{category}_rate": (category, "mean") for category in CATEGORIES}
    )
    analysis.mkdir(parents=True, exist_ok=True)
    frame.to_csv(analysis / "sample_metrics.csv", index=False)
    summary.to_csv(analysis / "harmful_outcomes.csv", index=False)
    benign_summary.to_csv(analysis / "benign_outcomes.csv", index=False)
    article = summary[(summary.condition == "suffix") & (summary.kind != "baseline")].copy()
    baseline = summary[(summary.condition == "suffix") & (summary.kind == "baseline")]
    for kind in ("concept", "random_direction"):
        article = pd.concat([article, baseline.assign(kind=kind)], ignore_index=True)
    article.sort_values(["model", "dataset", "kind", "alpha"]).to_csv(analysis / "article_harmful.csv", index=False)
    write_json(
        analysis / "manifest.json",
        {
            "schema_version": 1,
            "complete": True,
            "prepared_sha256": prepared_hash,
            "inputs_sha256": hashes,
            "artifacts_sha256": {p.name: digest(p) for p in sorted(analysis.glob("*.csv"))},
            "generation_config_sha256": {key: sorted(value) for key, value in generation_configs.items()},
            "judge_config_sha256": {key: sorted(value) for key, value in judge_configs.items()},
            "responses": len(frame) - len(prepared["baselines"]),
        },
    )


def figures(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = args.output / "steering_analysis"
    manifest = read_json(root / "manifest.json")
    if not manifest.get("complete") or manifest.get("schema_version") != 1:
        raise ValueError("incomplete steering analysis")
    for name, expected in manifest["artifacts_sha256"].items():
        if name != name.split("/")[-1] or digest(root / name) != expected:
            raise ValueError("modified analysis artifact")
    frame = pd.read_csv(root / "article_harmful.csv")
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.8))
    for axis, model in zip(axes, MODELS):
        subset = frame[(frame.model == model) & (frame.dataset == "advbench")]
        for kind, label, style in (
            ("concept", "Top-5 concepts", "o-"),
            ("random_direction", "Random directions", "s--"),
        ):
            values = subset[subset.kind == kind].sort_values("alpha")
            if len(values) != 7 or values.alpha.tolist() != [-0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5]:
                raise ValueError("incomplete article curve")
            axis.plot(values.alpha, values["mean"], style, label=label, markersize=3)
        baseline = subset[(subset.kind == "concept") & (subset.alpha == 0)]["mean"].iloc[0]
        axis.scatter([0], [baseline], facecolors="white", edgecolors="black", zorder=5)
        axis.set(title=MODELS[model][0].split("/")[-1], xlabel="Steering strength")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("StrongREJECT score")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    args.figures_output.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(args.figures_output / f"gcg-steering-advbench.{extension}", dpi=300)
    plt.close(fig)
    write_json(
        args.figures_output / "manifest.json",
        {
            "schema_version": 1,
            "complete": True,
            "analysis_sha256": digest(root / "manifest.json"),
            "artifacts_sha256": {p.name: digest(p) for p in args.figures_output.glob("gcg-steering-advbench.*")},
        },
    )
