"""Relevance and grounding checks.

Both use sentence embeddings (all-MiniLM-L6-v2, Reimers & Gurevych, 2019) and
cosine similarity. Similarity only measures shared topic and wording:
a weak match means "check this", but a strong match does not prove a point is
correct (Maynez et al., 2020; Laban et al., 2022).
"""

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from . import config
from .processing.text import split_sentences
from .schema import RevisionPack


class Embedder:
    """Loads the sentence-embedding model on first use and returns unit-length vectors."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return np.asarray(self._model.encode(list(texts), normalize_embeddings=True))


@dataclass
class RelevanceResult:
    applicable: bool
    method: str
    threshold: float
    scores: dict = field(default_factory=dict)  # "source a | source b" -> cosine similarity
    flagged: list = field(default_factory=list)


def _source_vectors(sources: dict[str, str], embedder) -> dict[str, np.ndarray]:
    """One vector per source: the normalised mean of its sentence embeddings."""
    vectors = {}
    for name, text in sources.items():
        sentences = split_sentences(text) or [text]
        embedded = embedder.encode(sentences)
        mean = embedded.mean(axis=0)
        vectors[name] = mean / (np.linalg.norm(mean) or 1.0)
    return vectors


def relevance_check(sources: dict[str, str], method: str = "embedding", embedder=None,
                    threshold: float = None) -> RelevanceResult:
    """Warn when two inputs appear to be about different topics."""
    if threshold is None:
        threshold = config.RELEVANCE_THRESHOLD
    present = {n: t for n, t in sources.items() if t and t.strip()}
    # The BLIP caption is one short generic sentence, so it is left out of this check.
    present.pop(config.SOURCE_CAPTION, None)
    result = RelevanceResult(len(present) >= 2, method, threshold)
    if not result.applicable:
        return result

    names = list(present)
    if method == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer

        matrix = TfidfVectorizer(stop_words="english").fit_transform([present[n] for n in names])
        similarity = (matrix @ matrix.T).toarray()  # rows are already L2-normalised
        score = lambda a, b: float(similarity[names.index(a), names.index(b)])
    else:
        vectors = _source_vectors(present, embedder or Embedder())
        score = lambda a, b: float(vectors[a] @ vectors[b])

    for a, b in combinations(names, 2):
        value = round(score(a, b), 3)
        result.scores[f"{a} | {b}"] = value
        if value < threshold:
            result.flagged.append((a, b, value))
    return result


@dataclass
class GroundingItem:
    kind: str
    text: str
    claimed_source: str
    score: float
    best_sentence: str
    best_source: str
    supported: bool


def pack_items(pack: RevisionPack) -> list[tuple[str, str, str]]:
    """The generated points that are checked: key concepts, flashcards and quiz answers with their explanations."""
    items = [("key concept", f"{k.concept}: {k.explanation}", k.source) for k in pack.key_concepts]
    items += [("flashcard", f"{c.front} {c.back}", c.source) for c in pack.flashcards]
    items += [("quiz question", quiz_item_text(q), q.source) for q in pack.quiz]
    return items


def quiz_item_text(question) -> str:
    return f"{question.question} {question.answer}. {question.explanation}"


def grounding_check(pack: RevisionPack, sources: dict[str, str], embedder=None,
                    threshold: float = None) -> list[GroundingItem]:
    """For each generated point, find the closest source sentence and flag weak support."""
    if threshold is None:
        threshold = config.GROUNDING_THRESHOLD
    items = pack_items(pack)
    source_sentences = [(name, s) for name, text in sources.items() if text for s in split_sentences(text)]
    if not items:
        return []
    if not source_sentences:
        return [GroundingItem(k, t, c, 0.0, "", "", False) for k, t, c in items]

    embedder = embedder or Embedder()
    item_vectors = embedder.encode([text for _, text, _ in items])
    sentence_vectors = embedder.encode([s for _, s in source_sentences])
    similarity = item_vectors @ sentence_vectors.T

    results = []
    for i, (kind, text, claimed) in enumerate(items):
        best = int(np.argmax(similarity[i]))
        value = round(float(similarity[i, best]), 3)
        name, sentence = source_sentences[best]
        results.append(GroundingItem(kind, text, claimed, value, sentence, name, value >= threshold))
    return results
