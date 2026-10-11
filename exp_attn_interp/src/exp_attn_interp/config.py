from __future__ import annotations

from pathlib import Path

REMOTE_ROOT = Path(".")
GENDATA_ROOT = Path("../exp_gendata")
SUSPICIOUS_ROOT = Path("../exp_suspicious_tests")
GENDATA_VECTORS = GENDATA_ROOT / "work" / "vectors"

INSTRUCT_MODEL = "Qwen/Qwen3-8B"
VECTOR_TAG = "Qwen_Qwen3-8B--assistant--semantic"
FAMILY_MODELS = ("Qwen/Qwen3-8B", "unsloth/Meta-Llama-3.1-8B-Instruct", "google/gemma-2-9b-it")
N_BLOCKS_OF = {"Qwen/Qwen3-8B": 36, "unsloth/Meta-Llama-3.1-8B-Instruct": 32, "google/gemma-2-9b-it": 42}
STEER_BAND_START = {"Qwen/Qwen3-8B": 19, "unsloth/Meta-Llama-3.1-8B-Instruct": 13, "google/gemma-2-9b-it": 21}


def vector_tag_for(model_name: str, presentation: str = "assistant") -> str:
    return f"{model_name.replace('/', '_')}--{presentation}--semantic"


ONTOLOGY = GENDATA_ROOT / "data" / "semantic.ontology.json"
LMSYS_ITEMS = SUSPICIOUS_ROOT / "data" / "lmsys_items.parquet"

PAIRS = 1036
NUM_BLOCKS = 36

BAND = tuple(range(round(NUM_BLOCKS / 3), round(2 * NUM_BLOCKS / 3) + 1, 2))
READ_LAYER = 18

MARKER_TOPN = 30
MARKER_MIN_OCCURRENCES = 2
MARKERS_PRESENCE = Path("data/markers_presence.json")
MARKERS_LOGITLENS = Path("data/markers_logitlens.json")

SINK_LEADING_TOKENS = 2
SINK_EPSILON = 0.3
SINK_COSINE = 0.95
SINK_NORM_MULT = 8.0

GRID_POSITIONS = 6
MIN_CONTEXT = 16

REPLY_TOKENS = 192
MIN_TOKENS = 32
MAX_TOKENS = 1024
SAMPLING = {"do_sample": True, "temperature": 1.0, "top_k": 20, "top_p": 0.95}

CEILING_ITEMS = 16
MEASURE_ITEMS = 64

CENTERINGS = ("raw", "concept", "token", "double")


def generation_seed(*parts: int) -> int:
    """Deterministic per-item seed; explicit arithmetic so it never depends on hash randomization."""
    seed = 20260903
    for part in parts:
        seed = (seed * 1000003 + part + 1) % (2**31 - 1)
    return seed
