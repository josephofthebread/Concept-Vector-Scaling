import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

MODELS = {
    "qwen3_8b": ("Qwen/Qwen3-8B", "Qwen_Qwen3-8B--assistant--semantic"),
    "llama3_1_8b_instruct": (
        "meta-llama/Llama-3.1-8B-Instruct",
        "unsloth_Meta-Llama-3.1-8B-Instruct--assistant--semantic",
    ),
    "gemma2_9b_it": ("google/gemma-2-9b-it", "google_gemma-2-9b-it--assistant--semantic"),
}
COUNTS = {"advbench": 520, "jbb": 100}
JUDGES = ("finetuned",)
CONTEXT = 8192


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def model_fingerprint(source, model=None, tokenizer=None):
    """Bind checkpoints to snapshot contents and resolved runtime configuration."""
    from huggingface_hub import try_to_load_from_cache

    root = Path(source)
    revision = getattr(getattr(model, "config", None), "_commit_hash", None)
    if not root.is_dir():
        for name in ("config.json", "adapter_config.json", "tokenizer_config.json"):
            cached = try_to_load_from_cache(str(source), name, revision=revision)
            if isinstance(cached, str):
                root = Path(cached).parent
                break
    files = {}
    if root.is_dir():
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in {".json", ".safetensors", ".bin", ".model", ".txt", ".tiktoken"}:
                files[str(path.relative_to(root))] = digest(path)
    config = model.config.to_dict() if model is not None else None
    if config is not None:
        config.pop("_name_or_path", None)
    return identity(
        {
            "source": files if files else str(source),
            "config": config,
            "revision": revision,
            "vocabulary": tokenizer.get_vocab() if tokenizer is not None else None,
            "chat_template": getattr(tokenizer, "chat_template", None),
        }
    )


def save_metadata(path, provenance):
    write_json(
        Path(str(path) + ".meta.json"), {"schema_version": 1, "provenance": provenance, "artifact_sha256": digest(path)}
    )


def check_checkpoint(path, expected=None):
    """Missing checkpoints may be generated; incomplete, old, or changed ones fail closed."""
    path = Path(path)
    metadata = Path(str(path) + ".meta.json")
    if not path.exists() and not metadata.exists():
        return None
    if not path.is_file() or not metadata.is_file():
        raise ValueError(f"incomplete or legacy checkpoint: {path.name}; use a fresh output directory")
    value = read_json(metadata)
    if (
        value.get("schema_version") != 1
        or value.get("artifact_sha256") != digest(path)
        or not isinstance(value.get("provenance"), dict)
        or any(value["provenance"].get(key) != item for key, item in (expected or {}).items())
    ):
        raise ValueError(f"stale checkpoint: {path.name}")
    return value["provenance"]


def write_checkpoint(path, value, provenance):
    write_json(path, value)
    save_metadata(path, provenance)


def response_names(record):
    return ["long_base", "long_suffix"]


def seed(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def load_records(path):
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    required = {"key", "model", "dataset", "index", "forbidden_prompt", "input_ids"}
    for record in records:
        missing = required - record.keys()
        if missing:
            raise ValueError(f"{record.get('key', '<unknown>')}: missing {sorted(missing)}")
        key = record["key"]
        if (
            not isinstance(key, str)
            or not key
            or "\\" in key
            or any(part in ("", ".", "..") for part in key.split("/"))
            or Path(key).is_absolute()
        ):
            raise ValueError("record keys must be safe relative paths")
        if record["model"] not in MODELS or record["dataset"] not in COUNTS:
            raise ValueError(f"{record['key']}: unsupported model or dataset")
        if set(record["input_ids"]) != {"base", "suffix"}:
            raise ValueError(f"{record['key']}: input_ids must contain base and suffix")
        if not all(isinstance(ids, list) and ids for ids in record["input_ids"].values()):
            raise ValueError(f"{record['key']}: input token lists must be non-empty")
    if len({record["key"] for record in records}) != len(records):
        raise ValueError("record keys are not unique")
    return records


def normalize(values):
    values = np.asarray(values)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or (norms <= 0).any():
        raise ValueError("vectors must be finite and non-zero")
    return values / norms


def chat(tokenizer, model, text):
    messages = [{"role": "user", "content": text}]
    if model == "llama3_1_8b_instruct":
        messages.insert(0, {"role": "system", "content": ""})
    options = {"enable_thinking": False} if model == "qwen3_8b" else {}
    return tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=False, add_generation_prompt=True, **options
    )


def model_source(args, model):
    override = getattr(args, model.replace("_instruct", "").replace("gemma2_9b_it", "gemma"), None)
    return override or MODELS[model][0]


def load_model(source, *, gpt_oss=False, device="cuda:0"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

    tokenizer = AutoTokenizer.from_pretrained(source)
    options = {"attn_implementation": "eager" if gpt_oss else "sdpa"}
    if gpt_oss:
        options.update(experts_implementation="eager", quantization_config=Mxfp4Config(dequantize=True))
    model = AutoModelForCausalLM.from_pretrained(
        source, torch_dtype=torch.bfloat16, device_map=device, **options
    ).eval()
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    return tokenizer, model
