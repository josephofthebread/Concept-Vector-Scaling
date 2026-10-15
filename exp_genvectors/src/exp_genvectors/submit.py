from __future__ import annotations

import subprocess
from argparse import ArgumentParser
from pathlib import Path

from exp_genvectors.config import EXTRACT_SHARDS, MODELS, REMOTE_ROOT


def sbatch(*arguments: str) -> str:
    result = subprocess.run(["sbatch", "--parsable", *arguments], check=True, text=True, capture_output=True)
    return result.stdout.strip().split(";")[0]


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="write the grids and print sizes, no sbatch")
    parser.add_argument("--skip-prepare", action="store_true", help="data/ already exists; submit extract directly")
    args = parser.parse_args()

    root = Path(REMOTE_ROOT)
    grid = [(spec.checkpoint, presentation, "semantic") for spec in MODELS for presentation in spec.presentations]
    grid.append(("Qwen/Qwen3-8B", "assistant", "syntax"))
    tasks = [
        (model, presentation, dataset, shard)
        for model, presentation, dataset in grid
        for shard in range(EXTRACT_SHARDS)
    ]
    jobs = root / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    (jobs / "vectors.tsv").write_text("".join("\t".join(run) + "\n" for run in grid))
    (jobs / "extract.tsv").write_text(
        "".join(f"{m}\t{p}\t{d}\t{shard}\t{EXTRACT_SHARDS}\n" for m, p, d, shard in tasks)
    )
    print(f"runs={len(grid)} extract_tasks={len(tasks)} shards={EXTRACT_SHARDS}")
    if args.dry_run:
        return

    dependency = []
    if not args.skip_prepare:
        prepare = sbatch(str(root / "slurm/prepare.sbatch"))
        dependency = [f"--dependency=afterok:{prepare}"]
        print(f"prepare={prepare}")
    extract = sbatch(*dependency, f"--array=0-{len(tasks) - 1}%13", str(root / "slurm/extract.sbatch"))
    vectors = sbatch(
        f"--dependency=afterok:{extract}", f"--array=0-{len(grid) - 1}%4", str(root / "slurm/vectors.sbatch")
    )
    print(f"extract={extract} vectors={vectors}")


if __name__ == "__main__":
    main()
