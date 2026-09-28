"""Rule-based revision pack used when the language model is unavailable or keeps failing.

It only reuses sentences from the sources, so it cannot invent content, but its
questions are simple fill-the-gap items.
"""

import re

from .processing.text import split_sentences
from .schema import Flashcard, KeyConcept, QuizQuestion, RevisionPack

_STOPWORDS = {
    "because", "between", "through", "without", "another", "different", "however", "therefore",
    "which", "where", "there", "these", "those", "their", "about", "would", "should", "could",
}


def _gap_word(sentence: str):
    """Pick the longest content word to blank out (a simple cloze item)."""
    words = [w.strip(".,;:()\"'") for w in sentence.split()]
    candidates = [w for w in words if len(w) >= 6 and w.isalpha() and w.lower() not in _STOPWORDS]
    return max(candidates, key=len) if candidates else None


def fallback_pack(sources: dict[str, str]) -> RevisionPack:
    tagged = [(name, s) for name, text in sources.items() for s in split_sentences(text, min_words=6)]
    used = sorted({name for name, _ in tagged})

    summary_sentences = [s for _, s in tagged[:3]]
    summary = " ".join(summary_sentences) if summary_sentences else "No usable study material was provided."

    key_concepts, flashcards, quiz = [], [], []
    for name, sentence in tagged:
        word = _gap_word(sentence)
        if not word:
            continue
        if len(key_concepts) < 6:
            key_concepts.append(KeyConcept(concept=word, explanation=sentence, source=name))
        if len(flashcards) < 8:
            front = re.sub(rf"\b{re.escape(word)}\b", "_____", sentence, count=1)
            flashcards.append(Flashcard(front=front, back=word, source=name))
        if len(quiz) < 5:
            quiz.append(QuizQuestion(
                kind="short_answer",
                question="Fill the gap: " + re.sub(rf"\b{re.escape(word)}\b", "_____", sentence, count=1),
                explanation=f"From your {name}: {sentence}",
                answer=word,
                source=name,
            ))

    plan = [
        "Check the transcript and slide text against the original material.",
        "Answer the fill-the-gap questions without looking at your notes.",
        "Rewrite any point you got wrong in your own words.",
    ]
    return RevisionPack(sources_used=used, summary=summary, key_concepts=key_concepts,
                        flashcards=flashcards, quiz=quiz, revision_plan=plan, generated_by="fallback")
