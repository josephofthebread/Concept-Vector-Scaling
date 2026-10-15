from dataclasses import dataclass

SOURCE_DATASET = "AntonKorznikov/feature_stories"

BUCKETS = 4
SKIP_TOKENS = 50
STORY_MAX_TOKENS = 1024

GENRES = (
    "case_study",
    "dialogue",
    "diary",
    "fable",
    "letter",
    "memo",
    "monologue",
    "narrative_3rd_person",
    "news",
    "speech",
)
LANGUAGES = ("English", "German", "Mandarin Chinese", "Russian", "Spanish")
VARIANTS = 10

PRESENTATIONS = (
    "raw",
    "named_raw",
    "user",
    "assistant",
    "named_assistant",
    "genre_assistant",
    "paired_raw",
    "paired_assistant",
)

REMOTE_ROOT = "."
EXTRACT_SHARDS = 1


@dataclass(frozen=True)
class ModelSpec:
    checkpoint: str
    instruction_tuned: bool

    @property
    def presentations(self) -> tuple[str, ...]:
        return ("user", "assistant", "named_assistant") if self.instruction_tuned else ("raw", "named_raw")


MODELS = (
    ModelSpec("Qwen/Qwen3-8B-Base", False),
    ModelSpec("Qwen/Qwen3-8B", True),
    ModelSpec("google/gemma-2-9b", False),
    ModelSpec("google/gemma-2-9b-it", True),
    ModelSpec("unsloth/Meta-Llama-3.1-8B", False),
    ModelSpec("unsloth/Meta-Llama-3.1-8B-Instruct", True),
    ModelSpec("unsloth/Meta-Llama-3.1-70B", False),
)

MODEL_BY_CHECKPOINT = {spec.checkpoint: spec for spec in MODELS}
