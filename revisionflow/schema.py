"""The structure of the revision pack the language model has to return.

The same Pydantic models are used three times: to give Ollama a JSON schema,
to validate what comes back, and to store sessions.
"""

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

Source = Literal["text notes", "audio transcript", "slide text", "image caption", "combined"]


class KeyConcept(BaseModel):
    concept: str
    explanation: str
    source: Source


class Flashcard(BaseModel):
    front: str
    back: str
    source: Source


class QuizQuestion(BaseModel):
    kind: Literal["multiple_choice", "short_answer"]
    question: str
    options: list[str] = Field(default_factory=list)
    # The explanation is placed before the answer so the model writes its
    # reasoning first (Tam et al., 2024).
    explanation: str
    answer: str
    source: Source


class RevisionPack(BaseModel):
    sources_used: list[Source]
    summary: str
    key_concepts: list[KeyConcept]
    flashcards: list[Flashcard]
    quiz: list[QuizQuestion]
    revision_plan: list[str]
    generated_by: str = "llm"


def ollama_schema() -> dict:
    """JSON schema passed to Ollama's `format` parameter.

    It is stricter than the Pydantic models used for validation: lengths and
    required fields steer the constrained decoding, while validation stays
    lenient so that small problems can be repaired instead of rejected.
    """
    schema = RevisionPack.model_json_schema()
    props = schema["properties"]
    props.pop("generated_by", None)
    schema["required"] = [r for r in schema["required"] if r != "generated_by"]
    props["summary"]["minLength"] = 20
    for name, low, high in (("key_concepts", 3, 8), ("flashcards", 5, 10), ("quiz", 4, 6), ("revision_plan", 3, 5)):
        props[name]["minItems"], props[name]["maxItems"] = low, high

    defs = schema["$defs"]
    quiz = defs["QuizQuestion"]
    quiz["required"] = ["kind", "question", "options", "explanation", "answer", "source"]
    quiz["properties"]["options"].pop("default", None)
    quiz["properties"]["options"]["maxItems"] = 4
    for field in ("question", "explanation", "answer"):
        quiz["properties"][field]["minLength"] = 1
    for field in ("front", "back"):
        defs["Flashcard"]["properties"][field]["minLength"] = 1
    for field in ("concept", "explanation"):
        defs["KeyConcept"]["properties"][field]["minLength"] = 1
    return schema


def normalise_source(value) -> str:
    """Map the different ways a small model writes a source name onto the allowed names."""
    text = str(value).strip().lower()
    if not text:
        return "combined"
    if "combined" in text or "multiple" in text or "," in text or " and " in text or "/" in text:
        return "combined"
    if "note" in text:
        return "text notes"
    if "audio" in text or "transcript" in text or "lecture" in text:
        return "audio transcript"
    if "slide" in text or "ocr" in text:
        return "slide text"
    if "caption" in text or "image" in text:
        return "image caption"
    return "combined"


def _strip_code_fence(raw: str) -> str:
    raw = raw.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.S)
    return match.group(1) if match else raw


def _normalise_kind(value) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text in ("mcq", "multiple_choice", "multiplechoice", "choice"):
        return "multiple_choice"
    return "short_answer"


def _match_option(answer: str, options: list[str]):
    """Return the option the answer refers to, allowing for letters (A-D) and small wording differences."""
    clean = answer.strip().lower().rstrip(".")
    for option in options:
        if option.strip().lower().rstrip(".") == clean:
            return option
    letter = re.match(r"^\(?([a-d])[\).:]?(\s|$)", clean)
    if letter:
        index = ord(letter.group(1)) - ord("a")
        if index < len(options):
            return options[index]
    if len(clean) >= 4:
        partial = [o for o in options if clean in o.lower() or o.strip().lower().rstrip(".") in clean]
        if len(partial) == 1:
            return partial[0]
    return None


def parse_pack(raw: str):
    """Parse and validate the model's reply.

    Returns (pack, error, repairs). `error` is a short message if the reply
    could not be used at all; `repairs` lists small problems that were fixed
    or items that were dropped.
    """
    repairs = []
    try:
        data = json.loads(_strip_code_fence(raw))
    except json.JSONDecodeError as exc:
        return None, f"reply was not valid JSON ({exc.msg})", repairs
    if not isinstance(data, dict):
        return None, "reply was not a JSON object", repairs

    used = data.get("sources_used") or []
    if isinstance(used, str):
        used = [used]
    data["sources_used"] = sorted({normalise_source(s) for s in used})
    for field in ("key_concepts", "flashcards", "quiz"):
        items = data.get(field) or []
        kept = []
        for item in items:
            if not isinstance(item, dict):
                repairs.append(f"dropped a {field} item that was not an object")
                continue
            item["source"] = normalise_source(item.get("source", ""))
            kept.append(item)
        data[field] = kept

    quiz = []
    for item in data.get("quiz", []):
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not question or not answer:
            repairs.append(f"dropped a question with no {'answer' if question else 'question text'}: {question[:60]}")
            continue
        item["kind"] = _normalise_kind(item.get("kind", ""))
        options = list(dict.fromkeys(str(o).strip() for o in (item.get("options") or []) if str(o).strip()))
        if item["kind"] == "multiple_choice":
            if len(options) < 2:
                item["kind"], options = "short_answer", []
                repairs.append(f"turned a multiple-choice question with no options into a short-answer question: {question[:60]}")
            else:
                matched = _match_option(answer, options)
                if matched is None and len(options) <= 3:
                    # The answer is missing from a short option list: add it at a position fixed by the question text.
                    position = sum(map(ord, question)) % (len(options) + 1)
                    options.insert(position, answer)
                    matched = answer
                    repairs.append(f"added the answer to the options of: {question[:60]}")
                if matched is None:
                    repairs.append(f"dropped a multiple-choice question whose answer was not one of its options: {question[:60]}")
                    continue
                answer = matched
        if item["kind"] == "short_answer":
            options = []
        item.update(question=question, answer=answer, options=options)
        quiz.append(item)
    data["quiz"] = quiz

    plan = data.get("revision_plan") or []
    if isinstance(plan, str):
        plan = [plan]
    # A small model sometimes writes each step as an object such as {"step": "..."}.
    data["revision_plan"] = [" ".join(map(str, s.values())) if isinstance(s, dict) else str(s) for s in plan]
    data.pop("generated_by", None)

    try:
        pack = RevisionPack.model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        location = ".".join(str(part) for part in first["loc"])
        return None, f"reply did not match the schema ({location}: {first['msg']})", repairs

    pack.flashcards = [c for c in pack.flashcards if c.front.strip() and c.back.strip()]
    pack.key_concepts = [k for k in pack.key_concepts if k.concept.strip()]
    if not (pack.summary.strip() and (pack.flashcards or pack.quiz)):
        return None, "reply was missing the summary or had no flashcards or questions", repairs
    return pack, None, repairs
