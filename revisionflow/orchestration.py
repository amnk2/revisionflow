"""Orchestration layer: combines the processed sources and asks the language model for a revision pack.

The pipeline is fixed (speech, image and text stages feed one generation step)
rather than planned by the language model, because the task is known in
advance and a 3B model is not reliable at planning (Shen et al., 2023).
Sources are passed as labelled blocks rather than plain concatenation, so the
model can tag where each point came from (Zeng et al., 2023).
"""

import json
import time
from dataclasses import dataclass, field

import requests

from . import config
from .fallback import fallback_pack
from .schema import RevisionPack, ollama_schema, parse_pack

INSTRUCTIONS = """You create revision material for a university student from the student's own study material.

Rules:
- Use only the information inside the source blocks below. Do not add facts from elsewhere.
- Tag every item with the source it came from: "text notes", "audio transcript", "slide text", "image caption", or "combined" if it needs more than one source.
- If a source is empty, unclear or about a different topic from the others, leave it out rather than forcing it in.
- Keep the summary under 120 words. The flashcards and quiz questions matter more than the summary.
- Write 4 to 8 key concepts, 6 to 10 flashcards and 5 quiz questions.
- The quiz must have at least 2 "short_answer" questions and at least 2 "multiple_choice" questions with 4 options each.
- For a multiple-choice question, the answer must be copied exactly from one of its options.
- Every answer must be supported by the source blocks.
- The revision plan gives 3 to 5 concrete next steps for this material."""

_LABELS = {
    config.SOURCE_NOTES: "TEXT NOTES",
    config.SOURCE_AUDIO: "AUDIO TRANSCRIPT",
    config.SOURCE_SLIDE: "SLIDE TEXT (OCR)",
    config.SOURCE_CAPTION: "IMAGE CAPTION",
}


@dataclass
class GenerationResult:
    pack: RevisionPack
    mode: str
    attempts: int
    valid_first_try: bool
    fallback_used: bool
    warnings: list = field(default_factory=list)
    repairs: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    raw_outputs: list = field(default_factory=list)


def build_context(sources: dict[str, str]) -> str:
    """Wrap each non-empty source in a labelled block."""
    blocks = []
    for name in config.SOURCE_ORDER:
        text = (sources.get(name) or "").strip()
        if text:
            label = _LABELS[name]
            blocks.append(f"[{label}]\n{text}\n[END {label}]")
    return "\n\n".join(blocks)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (English averages about 1.3 tokens per word)."""
    return int(len(text.split()) * 1.35)


def call_ollama(prompt: str, fmt=None) -> tuple[str, dict]:
    payload = {"model": config.OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": config.OLLAMA_OPTIONS}
    if fmt is not None:
        payload["format"] = fmt
    response = requests.post(f"{config.OLLAMA_URL}/api/generate", json=payload, timeout=config.OLLAMA_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    # Ollama reports durations in nanoseconds.
    meta = {
        "total_s": data.get("total_duration", 0) / 1e9,
        "load_s": data.get("load_duration", 0) / 1e9,
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "output_tokens": data.get("eval_count", 0),
        "generation_s": data.get("eval_duration", 0) / 1e9,
    }
    return data.get("response", ""), meta


def _schema_prompt(context: str, error: str = "") -> str:
    schema = json.dumps(ollama_schema())
    retry = f"\n\nYour previous reply could not be used because the {error}. Reply again with valid JSON only." if error else ""
    return f"{INSTRUCTIONS}\n\n{context}\n\nReturn only JSON that matches this schema:\n{schema}{retry}"


def _draft_prompt(context: str) -> str:
    return (f"{INSTRUCTIONS}\n\n{context}\n\nWrite the revision material as plain text under these headings: "
            "SOURCES USED, SUMMARY, KEY CONCEPTS, FLASHCARDS, QUIZ, REVISION PLAN. "
            "Put the source tag in brackets after every item.")


def _convert_prompt(draft: str, error: str = "") -> str:
    schema = json.dumps(ollama_schema())
    retry = f"\n\nYour previous reply could not be used because the {error}. Reply again with valid JSON only." if error else ""
    return ("Convert the revision material below into JSON that matches the schema. "
            "Copy the content across; do not add, remove or change any facts.\n\n"
            f"{draft}\n\nSchema:\n{schema}{retry}")


def generate_pack(sources: dict[str, str], mode: str = config.GENERATION_MODE) -> GenerationResult:
    """Generate a revision pack, retrying once with the error message and falling back to the rule-based pack."""
    context = build_context(sources)
    warnings, repairs, raw_outputs, timings = [], [], [], {}

    if not context:
        return GenerationResult(fallback_pack(sources), mode, 0, False, True, ["No study material was provided."])

    tokens = estimate_tokens(INSTRUCTIONS + context)
    if tokens > config.OLLAMA_OPTIONS["num_ctx"] * 0.8:
        size = "longer than" if tokens > config.OLLAMA_OPTIONS["num_ctx"] else "close to"
        warnings.append(f"The material is about {tokens} tokens, {size} the model's {config.OLLAMA_OPTIONS['num_ctx']}-token window, so part of it may be cut off.")

    start = time.perf_counter()
    error, attempts, pack = "", 0, None
    try:
        draft = ""
        if mode == "two_step":
            draft, meta = call_ollama(_draft_prompt(context))
            raw_outputs.append(draft)
            timings["draft"] = meta
        for attempts in (1, 2):
            prompt = _convert_prompt(draft, error) if mode == "two_step" else _schema_prompt(context, error)
            raw, meta = call_ollama(prompt, fmt=ollama_schema())
            raw_outputs.append(raw)
            timings[f"attempt_{attempts}"] = meta
            if meta["prompt_tokens"] >= config.OLLAMA_OPTIONS["num_ctx"]:
                warnings.append("The prompt filled the model's context window, so some material was probably cut off.")
            pack, error, item_repairs = parse_pack(raw)
            repairs.extend(item_repairs)
            if pack is not None:
                break
    except requests.RequestException as exc:
        warnings.append(f"Could not reach Ollama ({exc.__class__.__name__}), so the rule-based fallback was used. "
                        "Open the Ollama app (or run `ollama serve` in a terminal), then generate again.")
    timings["total_s"] = time.perf_counter() - start

    if pack is None:
        if error:
            warnings.append(f"The language model's reply could not be used after {attempts} attempts ({error}); the rule-based fallback was used.")
        return GenerationResult(fallback_pack(sources), mode, attempts, False, True, warnings, repairs, timings, raw_outputs)
    return GenerationResult(pack, mode, attempts, attempts == 1, False, warnings, repairs, timings, raw_outputs)
