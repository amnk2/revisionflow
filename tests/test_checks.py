from revisionflow.checks import grounding_check, relevance_check
from revisionflow.schema import Flashcard, KeyConcept, RevisionPack

UNRELATED = "The French Revolution began in 1789 when citizens stormed the Bastille prison in Paris."


def make_pack(concepts, cards):
    return RevisionPack(sources_used=["text notes"], summary="s", revision_plan=[], quiz=[],
                        key_concepts=[KeyConcept(concept=c, explanation=e, source="text notes") for c, e in concepts],
                        flashcards=[Flashcard(front=f, back=b, source="text notes") for f, b in cards])


def test_related_inputs_are_not_flagged(cpu_sources, fake_embedder):
    result = relevance_check(cpu_sources, embedder=fake_embedder, threshold=0.2)
    assert result.applicable and not result.flagged
    assert "text notes | audio transcript" in result.scores


def test_unrelated_inputs_are_flagged(cpu_sources, fake_embedder):
    sources = {"text notes": cpu_sources["text notes"], "audio transcript": UNRELATED}
    result = relevance_check(sources, embedder=fake_embedder, threshold=0.2)
    assert result.flagged and result.flagged[0][:2] == ("text notes", "audio transcript")


def test_tfidf_baseline_gives_same_decision(cpu_sources):
    related = relevance_check(cpu_sources, method="tfidf", threshold=0.1)
    unrelated = relevance_check({"text notes": cpu_sources["text notes"], "audio transcript": UNRELATED},
                                method="tfidf", threshold=0.1)
    assert not related.flagged and unrelated.flagged


def test_check_needs_two_sources_and_ignores_caption(fake_embedder):
    result = relevance_check({"text notes": "Only one source here.", "image caption": "a diagram"},
                             embedder=fake_embedder)
    assert not result.applicable


def test_grounding_flags_unsupported_points(cpu_sources, fake_embedder):
    pack = make_pack(
        concepts=[("Datapath", "The datapath contains the ALU and the registers.")],
        cards=[("Where did the French Revolution begin?", "In Paris in 1789.")],
    )
    results = grounding_check(pack, cpu_sources, embedder=fake_embedder, threshold=0.3)
    supported = {r.kind: r.supported for r in results}
    assert supported["key concept"] is True
    assert supported["flashcard"] is False
    assert results[0].best_source == "text notes"


def test_grounding_with_no_sources_marks_everything_unsupported(fake_embedder):
    pack = make_pack([("A", "b")], [])
    results = grounding_check(pack, {}, embedder=fake_embedder)
    assert len(results) == 1 and not results[0].supported
