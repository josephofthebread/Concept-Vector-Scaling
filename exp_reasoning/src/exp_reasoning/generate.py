from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from exp_reasoning.benchmarks import (
    MODEL_ID,
    UniqueNGramLoopDetector,
    instruction,
    load_benchmark,
    score,
    thinking_span,
    visible_gpu_ids,
)

RESULTS = Path("work/results")
GENERATION_VERSION = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("math_500", "gpqa_diamond", "all"), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS,
        help="Where shards are written and resumed from. Point a throwaway run somewhere "
        "else to keep it out of the main results.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Questions generated together. Decoding is "
        "memory-bandwidth bound, so a batch costs little more per step than a single "
        "sequence. Raise until the KV cache stops fitting.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16384,
        help="Generation budget per response. An unbounded run costs a full context window per generation.",
    )
    parser.add_argument(
        "--disable-loop-detection",
        action="store_true",
        help="Disable online low-diversity loop detection.",
    )
    parser.add_argument("--loop-ngram-size", type=int, default=4)
    parser.add_argument("--loop-window-tokens", type=int, default=1024)
    parser.add_argument("--loop-unique-ratio-threshold", type=float, default=0.20)
    parser.add_argument("--loop-check-every", type=int, default=64)
    parser.add_argument("--loop-consecutive-windows", type=int, default=3)
    parser.add_argument(
        "--loop-extra-tokens",
        type=int,
        default=512,
        help="Tokens retained after loop detection so several repeated phrases remain analyzable.",
    )
    parser.add_argument("--loop-min-new-tokens", type=int, default=2048)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.num_workers < 1:
        parser.error("--num-workers must be at least 1")
    if args.worker_index is not None and not 0 <= args.worker_index < args.num_workers:
        parser.error("--worker-index must be in [0, num-workers)")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    return args


def benchmark_names(name: str) -> tuple[str, ...]:
    return ("math_500", "gpqa_diamond") if name == "all" else (name,)


def build_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    tasks = []
    for benchmark in benchmark_names(args.benchmark):
        examples = load_benchmark(benchmark)
        if args.limit is not None:
            examples = examples[: args.limit]
        prompts = {example["id"]: instruction(benchmark, example["prompt"]) for example in examples}
        for example in examples:
            prompt = prompts[example["id"]]
            tasks.append(
                {
                    "benchmark": benchmark,
                    "id": example["id"],
                    "prompt": prompt,
                    "answer": example["answer"],
                }
            )
    return tasks


def task_key(task: dict[str, Any]) -> str:
    return f"{task['benchmark']}:{task['id']}:baseline"


def prompt_hash(task: dict[str, Any]) -> str:
    return hashlib.sha256(task["prompt"].encode()).hexdigest()


def compatible(record: dict[str, Any], task: dict[str, Any], max_new_tokens: int) -> bool:
    if not (
        record.get("key") == task_key(task)
        and record.get("model") == MODEL_ID
        and record.get("dtype") == "float16"
        and record.get("generation_version") == GENERATION_VERSION
        and record.get("prompt_sha256") == prompt_hash(task)
    ):
        return False
    if record.get("max_new_tokens") == max_new_tokens:
        return True
    generated = record.get("generated_token_count")
    stopped_on_its_own = not record.get("hit_context_limit") and not record.get("hit_token_budget", False)
    return stopped_on_its_own and isinstance(generated, int) and generated < max_new_tokens


def iter_records(path: Path):
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    print(f"{path}:{number}: skipping an unreadable line", file=sys.stderr, flush=True)


def result_path(args: argparse.Namespace) -> Path:
    results = getattr(args, "results_dir", None) or RESULTS
    if args.worker_index is None:
        return results / "generations.jsonl"
    return results / f"generations.worker-{args.worker_index:02d}-of-{args.num_workers:02d}.jsonl"


def eos_ids(tokenizer: Any) -> set[int]:
    value = tokenizer.eos_token_id
    return set(value if isinstance(value, list) else [value])


class BatchedLoopDetector(StoppingCriteria):
    """Runs the eval harness detector once per row, so one looping sequence stops alone."""

    def __init__(self, prompt_tokens: int, batch_size: int, eos: set[int], options: dict[str, Any]) -> None:
        self.detectors = [UniqueNGramLoopDetector(prompt_tokens, **options) for _ in range(batch_size)]
        self.eos = eos
        self.finished = [False] * batch_size
        self.scanned = prompt_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> torch.BoolTensor:
        width = input_ids.shape[1]
        done = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        for row, detector in enumerate(self.detectors):
            if not self.finished[row] and width > self.scanned:
                fresh = input_ids[row, self.scanned : width].tolist()
                self.finished[row] = any(token in self.eos for token in fresh)
            if self.finished[row]:
                continue
            done[row] = bool(detector(input_ids[row : row + 1], scores).any())
        self.scanned = max(self.scanned, width)
        return done

    def metadata(self, row: int, generated_tokens: int) -> dict[str, Any]:
        return self.detectors[row].metadata(generated_tokens)


def loop_options(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.disable_loop_detection:
        return None
    return {
        "ngram_size": args.loop_ngram_size,
        "window_tokens": args.loop_window_tokens,
        "unique_ratio_threshold": args.loop_unique_ratio_threshold,
        "check_every": args.loop_check_every,
        "consecutive_windows": args.loop_consecutive_windows,
        "min_new_tokens": args.loop_min_new_tokens,
        "extra_tokens": args.loop_extra_tokens,
    }


def condition_batches(tasks: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [tasks[start : start + batch_size] for start in range(0, len(tasks), batch_size)]


def split_continuation(row: list[int], eos: set[int]) -> list[int]:
    """Generated tokens of one row, cut at its own EOS so batch padding is dropped."""
    for index, token in enumerate(row):
        if token in eos:
            return row[: index + 1]
    return row


def generate_with_backoff(
    model: Any,
    tokenizer: Any,
    tasks: list[dict[str, Any]],
    options: dict[str, Any] | None,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Generate a batch, halving it whenever the KV cache does not fit."""
    try:
        return generate(model, tokenizer, tasks, options, max_new_tokens)
    except torch.OutOfMemoryError:
        if len(tasks) == 1:
            raise
        torch.cuda.empty_cache()
        middle = len(tasks) // 2
        print(f"out of memory on a batch of {len(tasks)}, retrying as {middle} + {len(tasks) - middle}", flush=True)
        return [
            record
            for half in (tasks[:middle], tasks[middle:])
            for record in generate_with_backoff(model, tokenizer, half, options, max_new_tokens)
        ]


@torch.inference_mode()
def generate(
    model: Any,
    tokenizer: Any,
    tasks: list[dict[str, Any]],
    options: dict[str, Any] | None,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": task["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for task in tasks
    ]
    inputs = tokenizer(rendered, return_tensors="pt", padding=True).to(model.device)
    prompt_width = inputs.input_ids.shape[1]
    eos = eos_ids(tokenizer)
    detector = BatchedLoopDetector(prompt_width, len(tasks), eos, options) if options is not None else None
    started = time.perf_counter()
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        max_length=model.config.max_position_embeddings,
        do_sample=False,
        **({"stopping_criteria": StoppingCriteriaList([detector])} if detector is not None else {}),
    )
    batch_seconds = time.perf_counter() - started

    records = []
    for row, task in enumerate(tasks):
        token_ids = split_continuation(output[row, prompt_width:].tolist(), eos)
        start, end, reasoning_status = thinking_span(tokenizer, token_ids, len(token_ids))
        reasoning_tokens = 0 if start is None else end - start
        text = tokenizer.decode(token_ids, skip_special_tokens=False)
        ended_with_eos = bool(token_ids) and token_ids[-1] in eos
        hit_context_limit = output.shape[1] >= model.config.max_position_embeddings and not ended_with_eos
        hit_token_budget = len(token_ids) >= max_new_tokens and not ended_with_eos
        correct = score(task["benchmark"], text, task["answer"]) if reasoning_status == "closed_thinking" else False
        loop_detection = (
            detector.metadata(row, len(token_ids))
            if detector is not None
            else {"enabled": False, "detected": False, "forced_stop": False}
        )
        records.append(
            {
                "key": task_key(task),
                "benchmark": task["benchmark"],
                "id": task["id"],
                "model": MODEL_ID,
                "dtype": "float16",
                "generation_version": GENERATION_VERSION,
                "prompt_sha256": prompt_hash(task),
                "output": text,
                "reference": task["answer"],
                "correct": correct,
                "generated_token_count": len(token_ids),
                "reasoning_token_count": reasoning_tokens,
                "reasoning_status": reasoning_status,
                "hit_context_limit": hit_context_limit,
                "hit_token_budget": hit_token_budget,
                "max_new_tokens": max_new_tokens,
                "loop_detected": loop_detection["detected"],
                "loop_forced_stop": loop_detection["forced_stop"],
                "loop_detection": loop_detection,
                "batch_size": len(tasks),
                "batch_seconds": batch_seconds,
                "generation_seconds": batch_seconds / len(tasks),
            }
        )
    return records


def run_worker(args: argparse.Namespace) -> None:
    tasks = build_tasks(args)
    if args.worker_index is not None:
        tasks = tasks[args.worker_index :: args.num_workers]
    path = result_path(args)
    tasks_by_key = {task_key(task): task for task in tasks}
    completed = {
        record["key"]
        for shard in sorted(args.results_dir.glob("generations*.jsonl"))
        for record in iter_records(shard)
        if record.get("key") in tasks_by_key and compatible(record, tasks_by_key[record["key"]], args.max_new_tokens)
    }
    pending = [task for task in tasks if task_key(task) not in completed]
    if not pending:
        print(f"{path}: all {len(tasks)} tasks already complete", flush=True)
        return
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    options = loop_options(args)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    batches = condition_batches(pending, args.batch_size)
    done = 0
    with path.open("a", encoding="utf-8") as handle:
        for batch in batches:
            records = generate_with_backoff(model, tokenizer, batch, options, args.max_new_tokens)
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            done += len(records)
            print(
                f"{done}/{len(pending)} {records[0]['key']} +{len(records) - 1} more "
                f"batch_seconds={records[0]['batch_seconds']:.1f} "
                f"tokens={[r['reasoning_token_count'] for r in records]} "
                f"correct={[r['correct'] for r in records]} "
                f"loop={[r['loop_detected'] for r in records]}",
                flush=True,
            )


def worker_command(args: argparse.Namespace, worker: int) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "exp_reasoning.generate",
        "--benchmark",
        args.benchmark,
        "--num-workers",
        str(args.num_workers),
        "--worker-index",
        str(worker),
        "--batch-size",
        str(args.batch_size),
        "--results-dir",
        str(args.results_dir),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--loop-ngram-size",
        str(args.loop_ngram_size),
        "--loop-window-tokens",
        str(args.loop_window_tokens),
        "--loop-unique-ratio-threshold",
        str(args.loop_unique_ratio_threshold),
        "--loop-check-every",
        str(args.loop_check_every),
        "--loop-consecutive-windows",
        str(args.loop_consecutive_windows),
        "--loop-min-new-tokens",
        str(args.loop_min_new_tokens),
        "--loop-extra-tokens",
        str(args.loop_extra_tokens),
    ]
    if args.disable_loop_detection:
        command.append("--disable-loop-detection")
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    return command


def seed_shards(args: argparse.Namespace, tasks: list[dict[str, Any]]) -> None:
    canonical = args.results_dir / "generations.jsonl"
    if not canonical.exists():
        return
    missing_workers = []
    destinations = {}
    for worker in range(args.num_workers):
        shard_args = argparse.Namespace(**{**vars(args), "worker_index": worker})
        path = result_path(shard_args)
        if not path.exists():
            missing_workers.append(worker)
            destinations[worker] = path.open("w", encoding="utf-8")
    if not missing_workers:
        return
    assigned = {
        task_key(task): (index % args.num_workers, task)
        for index, task in enumerate(tasks)
        if index % args.num_workers in missing_workers
    }
    try:
        with canonical.open(encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                target = assigned.get(record.get("key"))
                if target and compatible(record, target[1], args.max_new_tokens):
                    destinations[target[0]].write(line)
    finally:
        for handle in destinations.values():
            handle.close()


def merge_shards(args: argparse.Namespace, tasks: list[dict[str, Any]]) -> None:
    tasks_by_key = {task_key(task): task for task in tasks}
    seen = set()
    destination = args.results_dir / "generations.jsonl"
    temporary = destination.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        if destination.exists():
            with destination.open(encoding="utf-8") as source:
                for line in source:
                    record = json.loads(line)
                    if record.get("key") not in tasks_by_key:
                        output.write(line)
        for worker in range(args.num_workers):
            shard_args = argparse.Namespace(**{**vars(args), "worker_index": worker})
            with result_path(shard_args).open(encoding="utf-8") as source:
                for line in source:
                    record = json.loads(line)
                    key = record.get("key")
                    if (
                        key in tasks_by_key
                        and key not in seen
                        and compatible(record, tasks_by_key[key], args.max_new_tokens)
                    ):
                        output.write(line)
                        seen.add(key)
    missing = [key for key in tasks_by_key if key not in seen]
    if missing:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"{len(missing)} generation tasks are missing: {missing[:10]}")
    temporary.replace(destination)


def launch_workers(args: argparse.Namespace) -> None:
    gpu_ids = visible_gpu_ids()
    if len(gpu_ids) < args.num_workers:
        raise RuntimeError(f"Requested {args.num_workers} workers, but only {len(gpu_ids)} GPUs are visible")
    tasks = build_tasks(args)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    seed_shards(args, tasks)
    processes = []
    for worker in range(args.num_workers):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu_ids[worker]
        processes.append(subprocess.Popen(worker_command(args, worker), env=environment))
    failures = [(worker, process.wait()) for worker, process in enumerate(processes)]
    failures = [(worker, code) for worker, code in failures if code]
    if failures:
        raise RuntimeError(f"Generation workers failed: {failures}")
    merge_shards(args, tasks)


def main() -> None:
    args = parse_args()
    if args.worker_index is None and args.num_workers > 1:
        launch_workers(args)
    else:
        run_worker(args)


if __name__ == "__main__":
    main()
