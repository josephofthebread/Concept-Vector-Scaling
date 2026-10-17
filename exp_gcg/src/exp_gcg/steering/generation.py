from contextlib import contextmanager

from ..protocol import chat, digest, load_model, model_source, read_json, write_json
from .preparation import identity, load_prepared


@contextmanager
def final_token_hook(blocks, layer, vector, alpha):
    if not 0 < layer <= len(blocks):
        raise ValueError("intervention layer must refer to a transformer block output")
    hits = {"input": 0, "decode": 0}

    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hits["input"]:
            hits["decode"] += 1
            return output
        hits["input"] += 1
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[-1] != vector.numel():
            raise ValueError("expected one sequence with a matching hidden dimension")
        if alpha == 0:
            return output
        changed = hidden.clone()
        last = hidden[:, -1, :]
        changed[:, -1, :] = last + alpha * last.norm(dim=-1, keepdim=True) * vector.to(last)
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    handle = blocks[layer - 1].register_forward_hook(hook)
    try:
        yield hits
    finally:
        handle.remove()


def jobs(prepared, model):
    config = identity(prepared)
    for record in prepared["records"]:
        if record["model"] != model:
            continue
        for condition in ("suffix",):
            for index, direction in enumerate(prepared["directions"]):
                if (direction["model"], direction["dataset"]) != (model, record["dataset"]):
                    continue
                for alpha in prepared["alphas"]:
                    yield {
                        "config": config,
                        "model": model,
                        "source": "harmful",
                        "record": record["key"],
                        "dataset": record["dataset"],
                        "condition": condition,
                        "direction": index,
                        "kind": direction["kind"],
                        "label": direction["label"],
                        "layer": direction["layer"],
                        "alpha": alpha,
                        "prompt": record["forbidden_prompt"],
                        "input_ids": record["input_ids"][condition],
                    }
    for row in prepared["benign"]:
        yield {
            "config": config,
            "model": model,
            "source": "alpaca",
            "record": row["id"],
            "dataset": "alpaca",
            "condition": "base",
            "direction": None,
            "kind": "baseline",
            "label": "baseline",
            "layer": None,
            "alpha": 0,
            "prompt": row["prompt"],
            "input_ids": None,
        }
        for index, direction in enumerate(prepared["directions"]):
            if direction["model"] != model or direction["kind"] != "concept":
                continue
            for alpha in prepared["alphas"]:
                yield {
                    "config": config,
                    "model": model,
                    "source": "alpaca",
                    "record": row["id"],
                    "dataset": "alpaca",
                    "source_dataset": direction["dataset"],
                    "condition": "base",
                    "direction": index,
                    "kind": "concept",
                    "label": direction["label"],
                    "layer": direction["layer"],
                    "alpha": alpha,
                    "prompt": row["prompt"],
                    "input_ids": None,
                }


def assigned(items, shard, shards, limit):
    if shards < 1 or not 0 <= shard < shards or limit < 0:
        raise ValueError("invalid shard or limit")
    count = 0
    for index, item in enumerate(items):
        if index % shards == shard:
            if limit and count >= limit:
                break
            yield item
            count += 1


def response_path(output, job):
    return output / "responses" / job["model"] / (identity(job) + ".json")


def resumed(path, expected):
    if not path.exists():
        return False
    saved = read_json(path)
    if any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError(f"stale checkpoint: {path.name}")
    return True


def generate(args):
    import torch

    prepared = load_prepared(args.prepared)
    source = model_source(args, args.model)
    tokenizer, model = load_model(source, device=args.device)
    settings = {"model_source_sha256": identity(source), "max_new_tokens": 512, "decoding": "greedy"}
    for job in assigned(jobs(prepared, args.model), args.shard, args.shards, args.limit):
        path = response_path(args.output, job)
        if resumed(path, {"schema_version": 1, "job": job, "generation": settings}):
            continue
        ids_list = job["input_ids"] if job["input_ids"] is not None else chat(tokenizer, args.model, job["prompt"])
        if len(ids_list) + 512 > model.config.max_position_embeddings:
            raise ValueError("prompt plus generation exceeds model context")
        inputs = torch.tensor([ids_list], device=model.device)
        options = {
            "input_ids": inputs,
            "attention_mask": torch.ones_like(inputs),
            "max_new_tokens": 512,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
            "use_cache": True,
        }
        with torch.inference_mode():
            if job["direction"] is None:
                generated = model.generate(**options)
                hits = {"input": 0, "decode": 0}
            else:
                direction = torch.tensor(prepared["directions"][job["direction"]]["vector"], device=model.device)
                with final_token_hook(model.model.layers, job["layer"], direction, job["alpha"]) as hits:
                    generated = model.generate(**options)
                if hits["input"] != 1:
                    raise ValueError("generation must invoke the intervention exactly once")
        tokens = generated[0, len(ids_list) :].tolist()
        write_json(
            path,
            {
                "schema_version": 1,
                "job": job,
                "generation": settings,
                "generation_token_ids": tokens,
                "text": tokenizer.decode(tokens, skip_special_tokens=True),
                "hook_calls": hits,
            },
        )
    write_json(
        args.output / "generation" / f"{args.model}-{args.shard}-of-{args.shards}.json",
        {
            "schema_version": 1,
            "complete": not bool(args.limit),
            "prepared_sha256": digest(args.prepared),
            "generation": settings,
            "shard": args.shard,
            "shards": args.shards,
        },
    )
