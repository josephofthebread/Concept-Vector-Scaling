from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from .ontology import FeaturePair
from .secrets import openrouter_api_key, resolve_openrouter_proxy
from .story_few_shot import few_shot_section
from .utils import append_jsonl, read_jsonl, stable_id, write_json

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_VALIDATION_RETRIES = 10
RETRYABLE_HTTP_STATUS = {429, 500, 502, 503, 504}


class OpenRouterCreditError(RuntimeError):
    """API key/credits/limit exhausted. Abort the run; swap key and resume."""


class AdaptiveConcurrency:
    """AIMD in-flight limiter: halve on pressure (429/5xx/timeout), slowly ramp on successes."""

    def __init__(
        self,
        initial: int,
        minimum: int,
        maximum: int,
        *,
        success_step: int = 32,
        successes_before_up: int = 24,
    ) -> None:
        initial = max(1, int(initial))
        minimum = max(1, int(minimum))
        maximum = max(minimum, int(maximum))
        self.min = minimum
        self.max = maximum
        self.limit = min(maximum, max(minimum, initial))
        self.success_step = max(1, int(success_step))
        self.successes_before_up = max(1, int(successes_before_up))
        self._inflight = 0
        self._success_streak = 0
        self._cond = asyncio.Condition()
        self.pressure_events = 0
        self.ramp_events = 0
        self._last_pressure_at = 0.0
        self._pressure_cooldown_s = 2.0

    @property
    def inflight(self) -> int:
        return self._inflight

    async def acquire(self) -> None:
        async with self._cond:
            while self._inflight >= self.limit:
                await self._cond.wait()
            self._inflight += 1

    async def note_pressure(self, reason: str = "") -> None:
        """Lower limit on 429/5xx/timeout without releasing the caller's slot."""
        async with self._cond:
            now = time.monotonic()
            if now - self._last_pressure_at < self._pressure_cooldown_s:
                self._success_streak = 0
                return
            new_limit = max(self.min, self.limit // 2)
            if new_limit < self.limit:
                self.pressure_events += 1
                self._last_pressure_at = now
                print(
                    f"[generation] concurrency DOWN {self.limit} -> {new_limit} "
                    f"(inflight={self._inflight}, reason={reason or 'pressure'})",
                    file=sys.stderr,
                    flush=True,
                )
                self.limit = new_limit
            self._success_streak = 0
            self._cond.notify_all()

    async def release(self, *, success: bool = False) -> None:
        async with self._cond:
            self._inflight = max(0, self._inflight - 1)
            if success:
                self._success_streak += 1
                if self._success_streak >= self.successes_before_up and self.limit < self.max:
                    new_limit = min(self.max, self.limit + self.success_step)
                    if new_limit > self.limit:
                        self.ramp_events += 1
                        print(
                            f"[generation] concurrency UP {self.limit} -> {new_limit} (inflight={self._inflight})",
                            file=sys.stderr,
                            flush=True,
                        )
                        self.limit = new_limit
                    self._success_streak = 0
            self._cond.notify_all()

    def snapshot(self) -> dict[str, int]:
        return {
            "limit": self.limit,
            "min": self.min,
            "max": self.max,
            "inflight": self._inflight,
            "pressure_events": self.pressure_events,
            "ramp_events": self.ramp_events,
        }


def openrouter_credit_error_message(resp: httpx.Response) -> str | None:
    """Return a fatal credit/auth message if this response means stop-the-run."""
    status = resp.status_code
    text = (resp.text or "").strip()
    lower = text.lower()
    creditish = any(
        needle in lower
        for needle in (
            "insufficient credits",
            "add more credits",
            "payment required",
            "key limit exceeded",
            "can only afford",
            "quota exceeded",
            "quota_exceeded",
            "credit limit",
        )
    )
    if status in {401, 402}:
        return f"OpenRouter HTTP {status}: {text[:500]}"
    if status == 403 and (creditish or "limit" in lower or "credit" in lower or "quota" in lower):
        return f"OpenRouter HTTP {status}: {text[:500]}"
    if creditish and status >= 400:
        return f"OpenRouter HTTP {status}: {text[:500]}"
    return None


@dataclass(frozen=True)
class GenerationTask:
    feature_pair: FeaturePair
    genre: str
    language: str
    stories_per_pole: int
    target_words_min: int
    target_words_max: int

    @property
    def request_id(self) -> str:
        return stable_id(
            self.feature_pair.class_id,
            self.feature_pair.pair_index,
            self.feature_pair.concept,
            self.feature_pair.antagonist,
            self.genre,
            self.language,
        )


def genre_description(genre: str) -> str:
    return {
        "narrative_3rd_person": "third-person literary narrative",
        "dialogue": "dialogue scene with speaker labels",
        "monologue": "first-person monologue",
        "diary": "dated diary entry",
        "letter": "personal or professional letter",
        "news": "news report",
        "memo": "internal memo",
        "fable": "short fable with characters and implied moral",
        "speech": "public speech",
        "case_study": "case study with background, decision, and analysis",
    }.get(genre, genre.replace("_", " "))


def genre_contract(genre: str) -> str:
    contracts = {
        "speech": (
            "Required: first-person direct speech to an audience; at least 70% must be the spoken words.\n"
            "Mandatory opening: the protagonist addresses the audience (e.g. colleagues, friends, Ladies and gentlemen).\n"
            "Forbidden: third-person narration (e.g. 'He walked to the podium', 'She began her speech')."
        ),
        "monologue": (
            "Required: first-person interior voice; the protagonist speaks or thinks as 'I' throughout.\n"
            "Forbidden: third-person narration; other characters' dialogue exceeding 30% of the text."
        ),
        "narrative_3rd_person": (
            "Required: third-person literary narration of the protagonist's actions and reasoning.\n"
            "Forbidden: letter, diary, memo, news, or speech formatting."
        ),
        "diary": (
            "Required: first-person diary; the first line must be a date or time marker.\n"
            "Forbidden: third-person narration; public speech or news-report style."
        ),
        "letter": (
            "Required: opening salutation (e.g. Dear… / Уважаемый… / 尊敬的…), first-person body, "
            "and closing sign-off (e.g. Sincerely… / С уважением… / 此致敬礼). "
            "Never omit the salutation or sign-off.\n"
            "Forbidden: third-person narrative scenes outside the letter frame."
        ),
        "dialogue": (
            "Required: every line must begin with a speaker label in the form 'Name:'; at least 80% dialogue.\n"
            "Forbidden: novel-style third-person narration between exchanges; unlabeled speech lines."
        ),
        "memo": (
            "Required: the first three lines must be memo headers—To:/From:/Subject: "
            "(or locale equivalents such as Кому:/От:/Тема: / 致:/自:/主题:).\n"
            "Forbidden: literary fiction, fable, or diary framing."
        ),
        "news": (
            "Required: journalistic report with factual tone; attributed quotes allowed.\n"
            "Forbidden: first-person diary/letter voice; interior monologue as the main mode."
        ),
        "fable": (
            "Required: allegorical characters and an implied moral.\n"
            "Forbidden: realistic workplace case study without allegorical framing."
        ),
        "case_study": (
            "Required: background, decision process, and analysis in neutral professional tone.\n"
            "Forbidden: literary fiction, speech, diary, or letter formats."
        ),
    }
    return contracts.get(
        genre,
        f"Stay in strict {genre.replace('_', ' ')} form; do not default to generic third-person short fiction.",
    )


def anti_outcome_rules() -> str:
    return """- success, failure, victory, defeat, saved, lost, resolved, healed, punished, rewarded, approved, rejected
- trust gained or lost, reputation improved or damaged, gratitude or praise received, evaluation or verdict delivered
- aftermath, results, lessons learned from what happened, or 'in the end it worked / didn't work'
- looking back on how things turned out after the scene has ended
Allowed: the protagonist still deciding, acting, planning, or reasoning in the present moment.
Closing must show ongoing action or intention—not reflection on results."""


STORY_OUTCOME_MODE = "no_clear_reward_or_punishment"


def prose_quality_bullets(language: str) -> str:
    if language in {"Mandarin Chinese", "Japanese", "Korean"}:
        return (
            "- 3-5 paragraphs of connected prose (not a beat sheet).\n"
            "- Avg 20-45 characters/sentence; at most 2 consecutive sentences under 12 characters.\n"
            "- Use transitions and subordinate clauses; avoid choppy fragments."
        )
    return (
        "- 3-5 paragraphs of connected prose (not a beat sheet).\n"
        "- Avg 12-25 words/sentence; at most 2 consecutive sentences under 8 words.\n"
        "- Use transitions and subordinate clauses; avoid choppy fragments."
    )


def story_prompt_schema(task: GenerationTask, pair_numbers: list[int] | None = None) -> dict[str, Any]:
    if pair_numbers is None:
        pair_numbers = list(range(task.stories_per_pole))
    return {
        "request_id": task.request_id,
        "genre": task.genre,
        "language": task.language,
        "story_pairs": [
            {
                "pair_number": pair_number,
                "outcome_mode": outcome_mode_for_pair(pair_number),
                "protagonist_name": "string",
                "supporting_characters": ["string"],
                "shared_role_and_constraints": "English",
                "setup": "English",
                "pole_a_text": f"Full story in {task.language}",
                "pole_b_text": f"Full story in {task.language}",
            }
            for pair_number in pair_numbers
        ],
    }


def story_writing_rules(task: GenerationTask) -> str:
    pair = task.feature_pair
    multi_setup = ""
    if task.stories_per_pole > 1:
        multi_setup = (
            f"- Produce {task.stories_per_pole} distinct setups (pair_number 0..{task.stories_per_pole - 1}); "
            "each needs a different role, place, and decision—not the same scene with minor edits.\n"
        )
    return f"""## Writing rules
Read top to bottom. If rules conflict: genre form wins, then length, then other constraints.

### 1. Matched counterfactual pair
Each item = one shared setup + two full stories about the same named protagonist.
- pole_a_text and pole_b_text are near-parallel counterfactuals: same progression, cast, facts, paragraph structure, and length (±15%).
- Only pole-consistent choices and reasoning differ; balance non-target traits so neither pole is systematically more moral.
{multi_setup}
### 2. Target pole (activation signal)
- pole_a_text sustains "{pair.concept}"; pole_b_text sustains "{pair.antagonist}"—throughout the full text.
- Manifest the pole via concrete actions and reasoning; never name the concept labels in the text.
- Opening, middle, and closing must all reflect the target pole; exaggerate slightly for a clear diagnostic signal.
- The opposite pole must not appear in a given story.

### 3. Genre, length, and prose ({task.genre} · {task.language})
Genre contract (strict—this overrides concept if they clash):
{genre_contract(task.genre)}

Length: {genre_length_hint(task.genre, task.language)}. Expand any story below the minimum before returning JSON.

Prose ({task.language}):
{prose_quality_bullets(task.language)}

### 4. Setup (use the class and guidance in Task above)
- Invent a concrete protagonist, setting, and decision natural to that guidance—not a generic scene for unrelated classes.
- Do not copy few-shot settings, scenario shapes, or character names; avoid their templates (limited sign-up slots, lab blame games) unless genuinely the best fit.
- Avoid LLM defaults when unsure: same industry, job title, meeting room, product type, deadline phrase.
- The setup must be specific to this class—not interchangeable after swapping only language or protagonist name.
- Vary what is scarce (time, access, information, approval, risk, attention)—not always "N seats / N slots".
- Let the genre shape the scene (fable uses allegory; letter is correspondence; news is reportage—not every genre as a staff briefing).

### 5. Story arc
- Write pole_a_text and pole_b_text as full texts in {task.language}, not summaries.
- Three beats only: situation → decision/process → private reasoning.
- End while the protagonist is still deciding, acting, or reasoning—not after things are resolved.

## Avoid in story text
{anti_outcome_rules()}
Also do not copy few-shot examples: settings, scenario shapes, or names (e.g. Yulia, Marina, Miyu, Tamura).

## JSON metadata
- Set outcome_mode to "{STORY_OUTCOME_MODE}" for every item. This is a JSON label only—the story bodies must not frame outcomes.

## Final check (every story)
1. Meets length target and genre mandatory opening/closing lines
2. Setup grounded in this class—not a recycled template
3. Target pole visible in opening, middle, and closing
4. pole_a_text and pole_b_text are near-parallel counterfactuals
5. Flowing prose with no outcomes or resolution anywhere in the text"""


def story_task_section(task: GenerationTask, pair_numbers: list[int] | None = None) -> str:
    pair = task.feature_pair
    unit = "pair" if task.stories_per_pole == 1 else "pairs"
    return f"""## Task
Generate exactly {task.stories_per_pole} matched story {unit} for activation-vector extraction.
Include every pair_number from 0 through {task.stories_per_pole - 1}.

Feature pair:
- Pole A: {pair.concept}
- Pole B: {pair.antagonist}
- Class: {pair.class_name}
- Guidance: {pair.narrative_guidance}

Output format:
- Genre: {task.genre} ({genre_description(task.genre)})
- Story bodies (pole_a_text, pole_b_text): {task.language}
- JSON metadata (setup, shared_role_and_constraints): English

Return JSON only in this shape:
{json.dumps(story_prompt_schema(task, pair_numbers), ensure_ascii=False)}"""


def shared_story_generation_rules(task: GenerationTask) -> str:
    """Alias kept for repair prompts and any external callers."""
    return story_writing_rules(task)


def genre_length_hint(genre: str, language: str) -> str:
    if language in {"Mandarin Chinese", "Japanese", "Korean"}:
        return "Each story: 600-1200 characters (target 900-1100); below 600 is invalid"
    return "Each story: 200-400 words (target 300-380); below 200 words is invalid"


def outcome_mode_for_pair(pair_number: int) -> str:
    del pair_number
    return STORY_OUTCOME_MODE


def build_story_prompt(task: GenerationTask) -> list[dict[str, str]]:
    system = (
        "You generate paired research stimuli for mechanistic interpretability. "
        "Return valid JSON only. No markdown, no chain-of-thought, no extra commentary. "
        "Missing fields or wrong key names make the response unusable."
    )
    user = f"""{story_task_section(task)}

{story_writing_rules(task)}

{few_shot_section()}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user.strip()}]


def story_response_format(task: GenerationTask, structured_outputs: bool) -> dict[str, Any]:
    if not structured_outputs:
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "matched_story_pairs",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "request_id": {"type": "string"},
                    "genre": {"type": "string"},
                    "language": {"type": "string"},
                    "story_pairs": {
                        "type": "array",
                        "minItems": task.stories_per_pole,
                        "maxItems": task.stories_per_pole,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "pair_number": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": task.stories_per_pole - 1,
                                },
                                "outcome_mode": {"type": "string"},
                                "protagonist_name": {"type": "string"},
                                "supporting_characters": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "shared_role_and_constraints": {"type": "string"},
                                "setup": {"type": "string"},
                                "pole_a_text": {"type": "string"},
                                "pole_b_text": {"type": "string"},
                            },
                            "required": [
                                "pair_number",
                                "outcome_mode",
                                "protagonist_name",
                                "supporting_characters",
                                "shared_role_and_constraints",
                                "setup",
                                "pole_a_text",
                                "pole_b_text",
                            ],
                        },
                    },
                },
                "required": ["request_id", "genre", "language", "story_pairs"],
            },
        },
    }


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise
        obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("Model response JSON root must be an object")
    return obj


def story_item_pair_number(item: Any, fallback: int) -> int:
    if not isinstance(item, dict):
        return fallback
    try:
        return int(item.get("pair_number", fallback))
    except (TypeError, ValueError):
        return fallback


def story_item_has_texts(item: dict[str, Any]) -> bool:
    return bool(
        (
            item.get("pole_a_text")
            or item.get("concept_text")
            or (isinstance(item.get("concept_story"), dict) and item["concept_story"].get("text"))
        )
        and (
            item.get("pole_b_text")
            or item.get("antagonist_text")
            or (isinstance(item.get("antagonist_story"), dict) and item["antagonist_story"].get("text"))
        )
    )


def select_story_pair_items(pairs: list[Any], stories_per_pole: int) -> list[Any]:
    if len(pairs) <= stories_per_pole:
        return pairs
    by_pair_number: dict[int, list[dict[str, Any]]] = {}
    fallback_items: list[Any] = []
    for fallback, item in enumerate(pairs):
        if not isinstance(item, dict):
            fallback_items.append(item)
            continue
        pair_number = story_item_pair_number(item, fallback)
        by_pair_number.setdefault(pair_number, []).append(item)

    selected: list[Any] = []
    for pair_number in range(stories_per_pole):
        candidates = by_pair_number.get(pair_number, [])
        complete = [item for item in candidates if story_item_has_texts(item)]
        if complete:
            selected.append(complete[-1])
        elif candidates:
            selected.append(candidates[-1])

    if len(selected) < stories_per_pole:
        selected.extend(fallback_items[: stories_per_pole - len(selected)])
    return selected[:stories_per_pole]


def complete_pair_numbers(obj: dict[str, Any], stories_per_pole: int) -> set[int]:
    pairs = obj.get("story_pairs")
    if not isinstance(pairs, list):
        return set()
    complete: set[int] = set()
    for fallback, item in enumerate(pairs):
        if not isinstance(item, dict) or not story_item_has_texts(item):
            continue
        pair_number = story_item_pair_number(item, fallback)
        if 0 <= pair_number < stories_per_pole:
            complete.add(pair_number)
    return complete


def missing_pair_numbers(obj: dict[str, Any], stories_per_pole: int) -> list[int]:
    complete = complete_pair_numbers(obj, stories_per_pole)
    return [idx for idx in range(stories_per_pole) if idx not in complete]


def merge_story_pair_objects(base: dict[str, Any], repair: dict[str, Any], stories_per_pole: int) -> dict[str, Any]:
    merged = dict(base)
    base_pairs = base.get("story_pairs")
    repair_pairs = repair.get("story_pairs")
    if not isinstance(base_pairs, list):
        base_pairs = []
    if not isinstance(repair_pairs, list):
        repair_pairs = []
    merged["story_pairs"] = select_story_pair_items([*base_pairs, *repair_pairs], stories_per_pole)
    return merged


def build_story_repair_prompt(
    task: GenerationTask, existing: dict[str, Any], missing: list[int]
) -> list[dict[str, str]]:
    missing_text = ", ".join(str(x) for x in missing)
    existing_pairs = existing.get("story_pairs")
    existing_summary: list[dict[str, Any]] = []
    if isinstance(existing_pairs, list):
        for item in existing_pairs:
            if not isinstance(item, dict):
                continue
            pair_number = story_item_pair_number(item, len(existing_summary))
            if 0 <= pair_number < task.stories_per_pole and pair_number not in missing:
                existing_summary.append(
                    {
                        "pair_number": pair_number,
                        "outcome_mode": item.get("outcome_mode") or outcome_mode_for_pair(pair_number),
                        "setup": str(item.get("setup") or item.get("shared_setup") or "")[:300],
                        "protagonist_name": str(item.get("protagonist_name", ""))[:80],
                    }
                )

    system = (
        "Repair incomplete JSON for mechanistic interpretability stimuli. "
        "Return valid JSON only. No markdown, no chain-of-thought."
    )
    user = f"""## Repair task
The previous response was incomplete. Generate ONLY the missing story_pairs: {missing_text}.

{story_task_section(task, pair_numbers=missing)}

Existing pairs (use different setups for any new pair_number):
{json.dumps(existing_summary[: task.stories_per_pole], ensure_ascii=False)}

{story_writing_rules(task)}

{few_shot_section()}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user.strip()}]


def story_length_units(text: str, language: str) -> int:
    if language in {"Mandarin Chinese", "Japanese", "Korean"}:
        return len(re.sub(r"\s+", "", text))
    return len(re.findall(r"\w+", text, flags=re.UNICODE))


def minimum_story_units(language: str, target_words_min: int) -> int:
    if language in {"Mandarin Chinese", "Japanese", "Korean"}:
        return 600
    return target_words_min


def maximum_story_units(language: str, target_words_max: int) -> int:
    if language in {"Mandarin Chinese", "Japanese", "Korean"}:
        return 1200
    return target_words_max


def extract_pole_text(
    item: dict[str, Any],
    field: str,
    prefix: str,
    neutral_prefix: str,
) -> tuple[str, str, str]:
    story_obj = item.get(field)
    if isinstance(story_obj, dict):
        text = str(story_obj.get("text", "")).strip()
        title = str(story_obj.get("title", "")).strip()
        latent_cue_summary = str(story_obj.get("latent_cue_summary", "")).strip()
    else:
        text = str(item.get(f"{neutral_prefix}_text") or item.get(f"{prefix}_text") or "").strip()
        title = str(item.get(f"{neutral_prefix}_title") or item.get(f"{prefix}_title") or "").strip()
        latent_cue_summary = str(
            item.get(f"{neutral_prefix}_latent_cue_summary") or item.get(f"{prefix}_latent_cue_summary") or ""
        ).strip()
    return text, title, latent_cue_summary


def validate_story_text(
    text: str,
    task: GenerationTask,
    pair_index: int,
    neutral_prefix: str,
) -> int:
    if not text:
        raise ValueError(f"story_pairs[{pair_index}].{neutral_prefix}_text is missing")
    return story_length_units(text, task.language)


def validate_generation(obj: dict[str, Any], task: GenerationTask) -> list[dict[str, Any]]:
    pairs = obj.get("story_pairs")
    if not isinstance(pairs, list):
        raise ValueError("Response has no story_pairs list")
    if len(pairs) < task.stories_per_pole:
        raise ValueError(f"Expected at least {task.stories_per_pole} story_pairs, got {len(pairs)}")
    if len(pairs) > task.stories_per_pole:
        pairs = select_story_pair_items(pairs, task.stories_per_pole)

    rows: list[dict[str, Any]] = []
    feature = task.feature_pair
    for idx, item in enumerate(pairs):
        if not isinstance(item, dict):
            raise ValueError(f"story_pairs[{idx}] is not an object")
        shared_setup = str(item.get("shared_setup") or item.get("setup") or "").strip()
        try:
            pair_number = int(item.get("pair_number", idx))
        except (TypeError, ValueError):
            pair_number = idx
        outcome_mode = str(item.get("outcome_mode") or outcome_mode_for_pair(pair_number)).strip()
        expected_outcome_mode = outcome_mode_for_pair(pair_number)
        protagonist_name = str(item.get("protagonist_name", "")).strip()
        supporting_characters = item.get("supporting_characters", [])
        if not isinstance(supporting_characters, list):
            supporting_characters = [str(supporting_characters)]
        shared_role_and_constraints = str(item.get("shared_role_and_constraints", "")).strip()
        constant_details = item.get("constant_details", [])
        if not isinstance(constant_details, list):
            constant_details = [str(constant_details)]
        diversity_axis = str(item.get("diversity_axis", "")).strip()
        pole_specs = [
            ("concept", "concept_story", feature.concept, "concept", "pole_a"),
            ("antagonist", "antagonist_story", feature.antagonist, "antagonist", "pole_b"),
        ]
        pole_payloads: list[tuple[str, str, str, str, str, int]] = []
        for pole, field, label, prefix, neutral_prefix in pole_specs:
            text, title, latent_cue_summary = extract_pole_text(item, field, prefix, neutral_prefix)
            units = validate_story_text(text, task, idx, neutral_prefix)
            pole_payloads.append((pole, label, text, title, latent_cue_summary, units))
        min_units = minimum_story_units(task.language, task.target_words_min)
        max_units = maximum_story_units(task.language, task.target_words_max)
        for pole, label, text, title, latent_cue_summary, units in pole_payloads:
            length_ok = min_units <= units <= max_units
            story_id = stable_id(task.request_id, pair_number, pole)
            rows.append(
                {
                    "story_id": story_id,
                    "request_id": task.request_id,
                    "class_id": feature.class_id,
                    "class_name": feature.class_name,
                    "pair_index": feature.pair_index,
                    "pair_id": feature.pair_id,
                    "concept": feature.concept,
                    "antagonist": feature.antagonist,
                    "target_label": label,
                    "pole": pole,
                    "genre": task.genre,
                    "language": task.language,
                    "pair_number": pair_number,
                    "shared_setup": shared_setup,
                    "outcome_mode": outcome_mode,
                    "expected_outcome_mode": expected_outcome_mode,
                    "outcome_mode_ok": outcome_mode == expected_outcome_mode,
                    "protagonist_name": protagonist_name,
                    "supporting_characters": [str(x) for x in supporting_characters],
                    "shared_role_and_constraints": shared_role_and_constraints,
                    "constant_details": [str(x) for x in constant_details],
                    "diversity_axis": diversity_axis,
                    "latent_cue_summary": latent_cue_summary,
                    "title": title,
                    "text": text,
                    "length_units": units,
                    "minimum_length_units": min_units,
                    "maximum_length_units": max_units,
                    "length_ok": length_ok,
                    "mentions_concept": feature.concept.lower() in text.lower(),
                    "mentions_antagonist": feature.antagonist.lower() in text.lower(),
                }
            )
    return rows


async def call_openrouter(
    client: httpx.AsyncClient,
    task: GenerationTask,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_retries: int,
    structured_outputs: bool,
    require_parameters: bool,
    provider_sort: str | None = "price",
    messages: list[dict[str, str]] | None = None,
    response_format: dict[str, Any] | None = None,
    on_transient: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    api_key = openrouter_api_key()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages or build_story_prompt(task),
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "response_format": response_format or story_response_format(task, structured_outputs),
    }
    model_l = model.lower()
    ##
    # NOTE: gpt-oss requires reasoning and Llama 4 on Google rejects reasoning.effort=none/exclude,
    # so reasoning is disabled only for families known to accept the exclude payload.
    ##
    if "gpt-oss" not in model_l and "llama-4" not in model_l and not model_l.startswith("meta-llama/"):
        payload["reasoning"] = {"effort": "none", "exclude": True}
        payload["include_reasoning"] = False
    provider_prefs: dict[str, Any] = {}
    if provider_sort and provider_sort.lower() not in {"none", "off", "default"}:
        provider_prefs["sort"] = provider_sort
    if require_parameters:
        provider_prefs["require_parameters"] = True
    ##
    # NOTE: Llama 4 via Google Vertex often returns malformed JSON or rejects reasoning params,
    # so Google is excluded from its providers.
    ##
    if "llama-4" in model_l or model_l.startswith("meta-llama/"):
        ignore = list(provider_prefs.get("ignore") or [])
        if "Google" not in ignore:
            ignore.append("Google")
        provider_prefs["ignore"] = ignore
    if provider_prefs:
        payload["provider"] = provider_prefs
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://cursor.local/feature-vectors",
        "X-Title": "Alignment Feature Vector Pipeline",
    }

    def signal_transient(reason: str) -> None:
        if on_transient is None:
            return
        try:
            on_transient(reason)
        except Exception:  # noqa: BLE001 - never break generation on metrics callback.
            return

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = await client.post(OPENROUTER_URL, json=payload, headers=headers)
            credit_msg = openrouter_credit_error_message(resp)
            if credit_msg:
                raise OpenRouterCreditError(
                    f"{credit_msg} (request_id={task.request_id}). "
                    "Swap OPENROUTER_API_KEY / secrets/openrouter_api_key.txt and re-run "
                    "the same command to resume unfinished cells."
                )
            if resp.status_code in RETRYABLE_HTTP_STATUS:
                signal_transient(f"http_{resp.status_code}")
                raise httpx.HTTPStatusError(
                    "retryable OpenRouter error",
                    request=resp.request,
                    response=resp,
                )
            resp.raise_for_status()
            raw = resp.json()
            content = raw["choices"][0]["message"]["content"]
            parsed = extract_json_object(content)
            return parsed, raw.get("usage", {})
        except OpenRouterCreditError:
            raise
        except Exception as exc:  # noqa: BLE001 - keep retries resilient for API batches.
            resp = getattr(exc, "response", None)
            if isinstance(resp, httpx.Response):
                credit_msg = openrouter_credit_error_message(resp)
                if credit_msg:
                    raise OpenRouterCreditError(
                        f"{credit_msg} (request_id={task.request_id}). "
                        "Swap OPENROUTER_API_KEY / secrets/openrouter_api_key.txt and re-run "
                        "the same command to resume unfinished cells."
                    ) from exc
            if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
                signal_transient(type(exc).__name__)
            last_error = exc
            if attempt >= max_retries:
                break
            await asyncio.sleep(min(60.0, (2**attempt) + (0.1 * attempt)))
    raise RuntimeError(f"OpenRouter generation failed for {task.request_id}: {last_error}") from last_error


def completed_request_ids(stories_path: Path) -> set[str]:
    return {row["request_id"] for row in read_jsonl(stories_path)}


def write_retry_summary(out_dir: Path, total_tasks: int, pending_tasks: int, max_validation_retries: int) -> None:
    batches_path = out_dir / "batches.jsonl"
    errors_path = out_dir / "errors.jsonl"
    batches = list(read_jsonl(batches_path)) if batches_path.exists() else []
    errors = list(read_jsonl(errors_path)) if errors_path.exists() else []
    completed_requests = {row["request_id"] for row in batches}
    failed_requests = {row["request_id"] for row in errors if row["request_id"] not in completed_requests}
    validation_retried_batches = [row for row in batches if int(row.get("validation_retries_used", 0)) > 0]
    validation_retried_errors = [row for row in errors if int(row.get("validation_retries_used", 0)) > 0]
    total_api_calls = sum(int(row.get("validation_attempts", 1)) for row in batches + errors)
    total_finished_records = len(batches) + len(errors)
    write_json(
        out_dir / "retry_summary.json",
        {
            "total_tasks": total_tasks,
            "pending_tasks_at_start": pending_tasks,
            "completed_requests": len(completed_requests),
            "failed_requests": len(failed_requests),
            "max_validation_retries": max_validation_retries,
            "tasks_with_validation_retries": len(validation_retried_batches) + len(validation_retried_errors),
            "completed_tasks_with_validation_retries": len(validation_retried_batches),
            "failed_tasks_with_validation_retries": len(validation_retried_errors),
            "validation_retry_task_fraction": (
                (len(validation_retried_batches) + len(validation_retried_errors)) / max(1, total_finished_records)
            ),
            "total_api_calls": total_api_calls,
            "api_call_multiplier": total_api_calls / max(1, total_finished_records),
            "extra_api_calls_from_validation_retries": sum(
                max(0, int(row.get("validation_attempts", 1)) - 1) for row in batches + errors
            ),
        },
    )


async def generate_tasks(
    tasks: list[GenerationTask],
    out_dir: Path,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    concurrency: int,
    timeout_s: int,
    max_retries: int,
    validation_retries: int,
    structured_outputs: bool,
    require_parameters: bool,
    proxy: str | None = None,
    provider_sort: str | None = "price",
    concurrency_adaptive: bool = False,
    concurrency_min: int | None = None,
    concurrency_max: int | None = None,
) -> None:
    validation_retries = min(MAX_VALIDATION_RETRIES, max(0, int(validation_retries)))
    out_dir.mkdir(parents=True, exist_ok=True)
    stories_path = out_dir / "stories.jsonl"
    batches_path = out_dir / "batches.jsonl"
    errors_path = out_dir / "errors.jsonl"
    invalid_path = out_dir / "invalid_responses.jsonl"
    done = completed_request_ids(stories_path)
    pending = [task for task in tasks if task.request_id not in done]

    initial_c = max(1, int(concurrency))
    min_c = max(1, int(concurrency_min if concurrency_min is not None else max(1, initial_c // 8)))
    max_c = max(min_c, int(concurrency_max if concurrency_max is not None else initial_c))
    adaptive: AdaptiveConcurrency | None = None
    if concurrency_adaptive:
        adaptive = AdaptiveConcurrency(initial=initial_c, minimum=min_c, maximum=max_c)
        conn_cap = max_c
    else:
        conn_cap = initial_c

    write_json(
        out_dir / "manifest.json",
        {
            "total_tasks": len(tasks),
            "pending_tasks": len(pending),
            "model": model,
            "validation_retries": validation_retries,
            "max_validation_retries": MAX_VALIDATION_RETRIES,
            "concurrency": initial_c,
            "concurrency_adaptive": concurrency_adaptive,
            "concurrency_min": min_c if concurrency_adaptive else initial_c,
            "concurrency_max": max_c if concurrency_adaptive else initial_c,
            "tasks": [
                {
                    "request_id": task.request_id,
                    "class_id": task.feature_pair.class_id,
                    "class_name": task.feature_pair.class_name,
                    "pair_index": task.feature_pair.pair_index,
                    "concept": task.feature_pair.concept,
                    "antagonist": task.feature_pair.antagonist,
                    "genre": task.genre,
                    "language": task.language,
                }
                for task in tasks
            ],
        },
    )

    semaphore = asyncio.Semaphore(initial_c)
    timeout = httpx.Timeout(timeout_s)
    limits = httpx.Limits(max_connections=conn_cap, max_keepalive_connections=min(conn_cap, 256))
    resolved_proxy = resolve_openrouter_proxy(proxy)
    if resolved_proxy:
        print(f"[generation] using proxy {resolved_proxy}", file=sys.stderr, flush=True)
    if provider_sort and provider_sort.lower() not in {"none", "off", "default"}:
        print(f"[generation] provider sort={provider_sort}", file=sys.stderr, flush=True)
    else:
        print("[generation] provider sort=default (OpenRouter load balance)", file=sys.stderr, flush=True)
    if adaptive is not None:
        print(
            f"[generation] adaptive concurrency initial={adaptive.limit} min={adaptive.min} max={adaptive.max}",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(f"[generation] fixed concurrency={initial_c}", file=sys.stderr, flush=True)

    progress_lock = asyncio.Lock()
    progress = {"done": 0, "succeeded": 0, "failed": 0, "repaired": 0}

    async def report_progress(
        status: str, task: GenerationTask, elapsed_s: float, attempts: int, repaired: bool
    ) -> None:
        async with progress_lock:
            progress["done"] += 1
            progress[status] += 1
            if repaired:
                progress["repaired"] += 1
            total = len(pending)
            cbit = ""
            if adaptive is not None:
                cbit = f" C={adaptive.limit}/{adaptive.inflight}"
            print(
                "[generation] "
                f"{progress['done']}/{total} done "
                f"ok={progress['succeeded']} failed={progress['failed']} repaired={progress['repaired']} "
                f"last={status} attempts={attempts} elapsed={elapsed_s:.1f}s{cbit} "
                f"{task.request_id}",
                file=sys.stderr,
                flush=True,
            )

    def on_transient(reason: str) -> None:
        if adaptive is None:
            return
        asyncio.get_running_loop().create_task(adaptive.note_pressure(reason))

    client_kwargs: dict[str, Any] = {
        "timeout": timeout,
        "limits": limits,
        "trust_env": resolved_proxy is None,
    }
    if resolved_proxy:
        client_kwargs["proxy"] = resolved_proxy
    async with httpx.AsyncClient(**client_kwargs) as client:

        async def run_one(task: GenerationTask) -> None:
            if adaptive is not None:
                await adaptive.acquire()
            else:
                await semaphore.acquire()
            started = time.time()
            last_error: Exception | None = None
            validation_attempts = 0
            validation_retries_used = 0
            repair_used = False
            succeeded = False
            try:
                rows: list[dict[str, Any]] | None = None
                usage: dict[str, Any] = {}
                usage_events: list[dict[str, Any]] = []
                parsed: dict[str, Any] | None = None
                for validation_attempt in range(validation_retries + 1):
                    validation_attempts = validation_attempt + 1
                    validation_retries_used = validation_attempt
                    try:
                        repair_missing = missing_pair_numbers(parsed, task.stories_per_pole) if parsed else []
                        call_kwargs = dict(
                            client=client,
                            task=task,
                            model=model,
                            temperature=temperature,
                            top_p=top_p,
                            max_retries=max_retries,
                            require_parameters=require_parameters,
                            provider_sort=provider_sort,
                            on_transient=on_transient if adaptive is not None else None,
                        )
                        if parsed and repair_missing and len(repair_missing) < task.stories_per_pole:
                            repair_used = True
                            repair_max_tokens = min(
                                max_tokens,
                                max(4000, int(max_tokens * (len(repair_missing) / task.stories_per_pole)) + 2000),
                            )
                            repair_obj, usage = await call_openrouter(
                                **call_kwargs,
                                max_tokens=repair_max_tokens,
                                structured_outputs=False,
                                messages=build_story_repair_prompt(task, parsed, repair_missing),
                                response_format={"type": "json_object"},
                            )
                            usage_events.append(
                                {
                                    "attempt": validation_attempt,
                                    "mode": "repair_missing_pairs",
                                    "missing_pair_numbers": repair_missing,
                                    "max_tokens": repair_max_tokens,
                                    "usage": usage,
                                }
                            )
                            parsed = merge_story_pair_objects(parsed, repair_obj, task.stories_per_pole)
                        else:
                            parsed, usage = await call_openrouter(
                                **call_kwargs,
                                max_tokens=max_tokens,
                                structured_outputs=structured_outputs,
                            )
                            usage_events.append(
                                {
                                    "attempt": validation_attempt,
                                    "mode": "full_generation",
                                    "max_tokens": max_tokens,
                                    "usage": usage,
                                }
                            )
                        rows = validate_generation(parsed, task)
                        break
                    except OpenRouterCreditError:
                        raise
                    except ValueError as exc:
                        last_error = exc
                        if parsed is not None:
                            append_jsonl(
                                invalid_path,
                                [
                                    {
                                        "request_id": task.request_id,
                                        "task": asdict(task),
                                        "validation_attempt": validation_attempt,
                                        "error": repr(exc),
                                        "parsed_response": parsed,
                                    }
                                ],
                            )
                        if validation_attempt >= validation_retries:
                            raise
                        await asyncio.sleep(min(20.0, 1.0 + validation_attempt))
                if rows is None:
                    raise RuntimeError(f"No validated rows produced: {last_error!r}")
                append_jsonl(stories_path, rows)
                append_jsonl(
                    batches_path,
                    [
                        {
                            "request_id": task.request_id,
                            "task": asdict(task),
                            "usage": usage,
                            "usage_events": usage_events,
                            "validation_attempts": validation_attempts,
                            "validation_retries_used": validation_retries_used,
                            "regenerated_after_validation_error": validation_retries_used > 0,
                            "repair_used": repair_used,
                            "elapsed_s": round(time.time() - started, 3),
                        }
                    ],
                )
                succeeded = True
                await report_progress(
                    "succeeded",
                    task,
                    elapsed_s=time.time() - started,
                    attempts=validation_attempts,
                    repaired=repair_used,
                )
            except OpenRouterCreditError:
                raise
            except Exception as exc:  # noqa: BLE001 - persist failures and continue other batches.
                append_jsonl(
                    errors_path,
                    [
                        {
                            "request_id": task.request_id,
                            "task": asdict(task),
                            "error": repr(exc),
                            "validation_attempts": validation_attempts,
                            "validation_retries_used": validation_retries_used,
                            "regenerated_after_validation_error": validation_retries_used > 0,
                            "repair_used": repair_used,
                            "elapsed_s": round(time.time() - started, 3),
                        }
                    ],
                )
                await report_progress(
                    "failed",
                    task,
                    elapsed_s=time.time() - started,
                    attempts=validation_attempts,
                    repaired=repair_used,
                )
            finally:
                if adaptive is not None:
                    await adaptive.release(success=succeeded)
                else:
                    semaphore.release()

        workers = [asyncio.create_task(run_one(task), name=task.request_id) for task in pending]
        credit_abort: OpenRouterCreditError | None = None
        try:
            await asyncio.gather(*workers)
        except OpenRouterCreditError as exc:
            credit_abort = exc
            for worker in workers:
                if not worker.done():
                    worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
        except Exception:
            for worker in workers:
                if not worker.done():
                    worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            raise

    write_retry_summary(
        out_dir,
        total_tasks=len(tasks),
        pending_tasks=len(pending),
        max_validation_retries=validation_retries,
    )
    if adaptive is not None:
        write_json(out_dir / "concurrency_adaptive.json", adaptive.snapshot())
        print(f"[generation] adaptive concurrency final={adaptive.snapshot()}", file=sys.stderr, flush=True)
    if credit_abort is not None:
        done_now = completed_request_ids(stories_path)
        write_json(
            out_dir / "aborted.json",
            {
                "reason": "openrouter_credits_or_key_limit",
                "error": str(credit_abort),
                "model": model,
                "completed_request_ids": len(done_now),
                "pending_at_start": len(pending),
                "resume_hint": (
                    "Update secrets/openrouter_api_key.txt (or OPENROUTER_API_KEY), "
                    "then re-run the same generate command; finished cells are skipped."
                ),
            },
        )
        print(
            f"[generation] ABORTED (credits/key limit): {credit_abort}",
            file=sys.stderr,
            flush=True,
        )
        print(
            "[generation] Wrote aborted.json — swap API key and re-run to resume.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(2) from credit_abort


def build_generation_tasks(
    feature_pairs: list[FeaturePair],
    genres: list[str],
    languages: list[str],
    stories_per_pole: int,
    target_words_min: int,
    target_words_max: int,
) -> list[GenerationTask]:
    return [
        GenerationTask(pair, genre, language, stories_per_pole, target_words_min, target_words_max)
        for pair in feature_pairs
        for genre in genres
        for language in languages
    ]
