from __future__ import annotations

DRAFT_MODEL = "openai/gpt-oss-120b"

DRAFT_TEMPERATURE = 1.0
DRAFT_TOP_P = 1.0
DRAFT_TOP_K = -1

DRAFT_EFFORT = "low"
DRAFT_MAX_TOKENS = 768
DRAFT_MAX_MODEL_LEN = 4096

INSTRUCT_MODELS = (
    "Qwen/Qwen3-8B",
    "google/gemma-2-9b-it",
    "unsloth/Meta-Llama-3.1-8B-Instruct",
)
BASELINE_MAX_TOKENS = 2048
BASELINE_MAX_MODEL_LEN = 4096


def chat_template_kwargs(model: str) -> dict[str, bool]:
    """Per-model chat-template flags; Qwen3 has thinking disabled to match exp_suspicious_tests."""
    return {"enable_thinking": False} if "Qwen3" in model else {}


KIND_COUNTS = {"neutral": 32, "concept": 16, "antagonist": 16}

SEEDS: tuple[int, ...] = (
    101,
    211,
    307,
    401,
    503,
    601,
    701,
    809,
    907,
    1009,
    1103,
    1201,
    1301,
    1409,
    1511,
    1601,
    1709,
    1811,
    1901,
    2003,
    2111,
    2213,
    2309,
    2411,
    2503,
    2609,
    2707,
    2801,
    2903,
    3001,
    3109,
    3203,
)

SCENARIO_HINTS: tuple[str, ...] = (
    "a first-person diary entry",
    "a short scene between two characters",
    "a letter to a close friend",
    "an unspoken inner monologue",
    "a spoken voice-memo transcript",
    "a short social-media post",
    "a piece of flash fiction",
    "the opening of a reflective personal essay",
    "an everyday workplace situation",
    "a vivid childhood memory",
    "a moment of decision under pressure",
    "a plain description of an ordinary scene",
    "a dialogue-heavy exchange",
    "a short lyrical or poetic passage",
    "an opinion piece or review",
    "a message left on someone's phone",
    "a text-message thread",
    "a scene set at a family dinner",
    "a late-night journal page",
    "a conversation with a stranger on a train",
    "a short blog post",
    "a recollection of a turning point",
    "a note passed in a meeting",
    "a passage describing a walk outdoors",
    "a monologue delivered to an empty room",
    "a scene in a waiting room",
    "an exchange between colleagues after work",
    "a short retelling of a dream",
    "a reflection written on a birthday",
    "a scene at a bus stop in the rain",
    "a confession made to a friend",
    "a description of returning to an old place",
)

assert len(SEEDS) == len(SCENARIO_HINTS) == 32, "thirty-two seeds and thirty-two shells"

SYSTEM_PROMPT = (
    "You are assisting mechanistic-interpretability research. Researchers extract "
    "concept-activation vectors by feeding a language model prompts that make it GENERATE short "
    "passages, then reading the model's internal activations. Your job is to write those "
    "elicitation prompts.\n\n"
    "Many target concepts are negative (for example cruelty, despair, deception, or fear). Writing "
    "a prompt that elicits a short *fictional* passage expressing such a concept is standard and "
    "expected -- the analogue of a sentiment dataset containing negative reviews -- and you should "
    "produce those normally. Decline only if a prompt would require genuinely harmful real-world "
    "operational content (for example weapon synthesis, or sexual content involving minors); a "
    "brief refusal is fine there and will be recorded. Otherwise, write the elicitation prompt."
)


def build_embodiment_prompt(pole_text: str, opposite_text: str, guidance: str, hint: str) -> str:
    """Stage-1 instruction for a concept or antagonist draft whose answer embodies one pole."""
    return (
        f'Target concept to embody: "{pole_text}"\n'
        f'Its opposite (for direction only -- do NOT evoke it): "{opposite_text}"\n'
        f"Background on the dichotomy: {guidance}\n\n"
        "Write ONE elicitation prompt to give to another language model so that the model's "
        f'GENERATED continuation vividly EMBODIES "{pole_text}" -- it should read as a natural piece '
        "of writing saturated with the concept, not a definition or an explanation of it.\n\n"
        "Constraints:\n"
        f"- Cast it as: {hint}.\n"
        "- Two to four sentences. Concrete and evocative.\n"
        f'- Do not name the label "{pole_text}" outright if the passage can carry it without naming.\n'
        "- Do not mention or hint at the opposite concept.\n"
        "- The prompt must work for a plain completion model as well as a chat model.\n\n"
        "Output ONLY the elicitation prompt, wrapped exactly as:\n"
        "<PROMPT>your prompt here</PROMPT>"
    )


def build_neutral_prompt(concept: str, antagonist: str, guidance: str, hint: str) -> str:
    """Stage-1 instruction for a neutral, on-axis draft that leans to neither pole."""
    return (
        f'Axis of interest: "{concept}" vs "{antagonist}".\n'
        f"Background on the dichotomy: {guidance}\n\n"
        "Write ONE open-ended elicitation prompt to give to another language model. The prompt must "
        "be on the theme of this axis but strictly NEUTRAL: it must not lean toward either pole, and "
        "a natural answer could plausibly go either way. It should invite a substantive free-form "
        "response (several sentences) in which this trait could surface, so that later steering along "
        f'the "{concept}" <-> "{antagonist}" axis will visibly change the answer.\n\n'
        "Constraints:\n"
        f"- Cast it as: {hint}.\n"
        "- One to three sentences.\n"
        "- Do not mention or hint at either pole label; stay genuinely balanced.\n"
        "- The prompt must work for a plain completion model as well as a chat model.\n\n"
        "Output ONLY the elicitation prompt, wrapped exactly as:\n"
        "<PROMPT>your prompt here</PROMPT>"
    )
