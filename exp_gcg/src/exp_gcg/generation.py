from pathlib import Path

import numpy as np

from .evaluation import fit_response, judge_ids
from .protocol import (
    CONTEXT,
    check_checkpoint,
    digest,
    identity,
    load_model,
    load_records,
    model_fingerprint,
    model_source,
    read_json,
    save_metadata,
    seed,
    write_checkpoint,
)


class JudgeBudget:
    """Stop long responses at the reference experiment's judge-token budget."""

    def __init__(self, tokenizer, judge_tokenizer, template, prompt, start):
        self.tokenizer, self.judge_tokenizer = tokenizer, judge_tokenizer
        self.template, self.prompt, self.start = template, prompt, start
        self.hit = False

    def __call__(self, input_ids, scores, **kwargs):
        response = self.tokenizer.decode(input_ids[0, self.start :], skip_special_tokens=True)
        self.hit = len(judge_ids(self.judge_tokenizer, self.template, self.prompt, response)) >= CONTEXT - 1
        return self.hit


def generate(args):
    import torch
    from transformers import AutoTokenizer, StoppingCriteriaList, set_seed

    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive when supplied")
    records = [record for record in load_records(args.records) if record["model"] == args.model]
    source = model_source(args, args.model)
    tokenizer, model = load_model(source, device=args.device)
    judge_tokenizer = AutoTokenizer.from_pretrained(args.strongreject)
    template = read_json(args.templates)["strongreject_finetuned"]
    model_hash = model_fingerprint(source, model, tokenizer)
    protocol_hash = identity(
        {
            "model_sha256": model_hash,
            "max_new_tokens": args.max_new_tokens,
            "judge_context": CONTEXT,
            "generation_config": model.generation_config.to_dict(),
            "judge_tokenizer_sha256": model_fingerprint(args.strongreject, tokenizer=judge_tokenizer),
            "templates_sha256": digest(args.templates),
            "stopping": "judge_context_or_eos_v1",
        }
    )
    output = Path(args.output)
    for number, record in enumerate(records, 1):
        root = output / "records" / record["key"]
        common = {"record_sha256": identity(record), "model_sha256": model_hash}
        for condition in ("base", "suffix"):
            hidden_path = root / f"hidden_{condition}.npy"
            provenance = common | {"condition": condition, "input_ids_sha256": identity(record["input_ids"][condition])}
            if check_checkpoint(hidden_path, provenance) is None:
                ids = torch.tensor([record["input_ids"][condition]], device=model.device)
                with torch.inference_mode():
                    states = model.model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
                hidden = torch.stack([state[0, -1].float().cpu() for state in states]).numpy()
                del states
                if not np.isfinite(hidden).all():
                    raise ValueError("non-finite residuals")
                hidden_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = hidden_path.with_suffix(".tmp.npy")
                np.save(temporary, hidden)
                temporary.replace(hidden_path)
                save_metadata(hidden_path, provenance)
        jobs = [("long_base", record["input_ids"]["base"]), ("long_suffix", record["input_ids"]["suffix"])]
        for name, ids_list in jobs:
            destination = root / "responses" / f"{name}.json"
            provenance = common | {
                "generation_protocol_sha256": protocol_hash,
                "input_ids_sha256": identity(ids_list),
                "name": name,
            }
            if check_checkpoint(destination, provenance) is not None:
                continue
            chosen_seed = seed(record["key"] + "/" + name)
            set_seed(chosen_seed)
            ids = torch.tensor([ids_list], device=model.device)
            available = model.config.max_position_embeddings - len(ids_list)
            maximum = min(args.max_new_tokens or available, available)
            if maximum < 1:
                raise ValueError("input fills the model context")
            budget = JudgeBudget(tokenizer, judge_tokenizer, template, record["forbidden_prompt"], len(ids_list))
            if len(judge_ids(judge_tokenizer, template, record["forbidden_prompt"], "")) >= CONTEXT:
                raise ValueError("prompt and rubric exceed judge context")
            options = {"stopping_criteria": StoppingCriteriaList([budget])}
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    max_new_tokens=maximum,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                    **options,
                )
            token_ids = generated[0, len(ids_list) :].tolist()
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            evaluated = fit_response(judge_tokenizer, template, record["forbidden_prompt"], text)
            eos = model.generation_config.eos_token_id
            eos = [eos] if isinstance(eos, int) else eos or []
            reason = (
                "eos"
                if token_ids and token_ids[-1] in eos
                else (
                    "judge_context"
                    if budget.hit
                    else "explicit_cap"
                    if args.max_new_tokens and maximum < available
                    else "target_context"
                )
            )
            write_checkpoint(
                destination,
                {
                    "schema_version": 1,
                    "key": record["key"],
                    "name": name,
                    "text": text,
                    "evaluated_text": evaluated,
                    "generation_token_ids": token_ids,
                    "prompt_tokens": len(ids_list),
                    "seed": chosen_seed,
                    "max_new_tokens": maximum,
                    "stop_reason": reason,
                },
                provenance,
            )
        print(f"{args.model}: {number}/{len(records)} {record['key']}", flush=True)
