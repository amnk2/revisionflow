"""Cleaning and sentence splitting for the text that flows through the pipeline."""

import re

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_BULLET = re.compile(r"^\s*(?:[-*•▪◦]|\d+[.)])\s+")


def clean_text(text: str) -> str:
    """Tidy pasted notes or OCR output without changing the wording."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"-\n(?=[a-z])", "", text)  # words split across lines by OCR
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str, min_words: int = 3) -> list[str]:
    """Split text into sentences, treating each bullet or short line as its own unit.

    Used by the relevance and grounding checks, because the embedding model
    truncates long inputs (256 word pieces).
    """
    sentences = []
    for block in clean_text(text).split("\n"):
        block = _BULLET.sub("", block).strip()
        if not block:
            continue
        for part in _SENTENCE_END.split(block):
            part = part.strip()
            if len(part.split()) >= min_words:
                sentences.append(part)
    return sentences


def word_count(text: str) -> int:
    return len(text.split())
