from revisionflow.schema import normalise_source, ollama_schema, parse_pack


def test_valid_reply_is_parsed_and_repaired(valid_pack_json):
    pack, error, repairs = parse_pack(valid_pack_json)
    assert error is None
    assert pack.sources_used == ["audio transcript", "text notes"]
    # empty flashcard removed
    assert len(pack.flashcards) == 1 and pack.flashcards[0].source == "text notes"
    # letter answer mapped to the option text; 4-option question with an impossible answer dropped;
    # answer added to a 2-option list; no-options MCQ turned into short answer; empty answer dropped
    assert [q.answer for q in pack.quiz] == ["ALU", "Control unit", "The ALU and registers", "Buses"]
    assert any("dropped a multiple-choice" in r for r in repairs)
    assert "Control unit" in pack.quiz[1].options and len(pack.quiz[1].options) == 3
    assert pack.quiz[2].kind == "short_answer" and pack.quiz[2].options == []
    assert any("no answer" in r for r in repairs)
    # short-answer options cleared, kind normalised, source normalised
    assert pack.quiz[3].kind == "short_answer" and pack.quiz[3].options == []
    assert pack.quiz[3].source == "audio transcript"
    assert pack.revision_plan == ["Review the datapath."]


def test_code_fence_is_removed(valid_pack_json):
    pack, error, _ = parse_pack("```json\n" + valid_pack_json + "\n```")
    assert error is None and pack is not None


def test_invalid_json_gives_error():
    pack, error, _ = parse_pack("Here is your revision material: summary...")
    assert pack is None and "not valid JSON" in error


def test_missing_summary_gives_schema_error():
    reply = '{"flashcards": [{"front": "a", "back": "b", "source": "text notes"}], "quiz": []}'
    pack, error, _ = parse_pack(reply)
    assert pack is None and "schema" in error and "summary" in error


def test_missing_lists_default_to_empty():
    reply = '{"summary": "x", "flashcards": [{"front": "a", "back": "b", "source": "text notes"}]}'
    pack, error, _ = parse_pack(reply)
    assert error is None and pack.quiz == [] and pack.key_concepts == [] and pack.revision_plan == []


def test_pack_without_questions_is_rejected():
    reply = '{"sources_used": [], "summary": "x", "key_concepts": [], "flashcards": [], "quiz": [], "revision_plan": []}'
    pack, error, _ = parse_pack(reply)
    assert pack is None and "no flashcards" in error


def test_odd_but_usable_shapes_are_accepted(valid_pack_json):
    import json

    data = json.loads(valid_pack_json)
    data["sources_used"] = "text notes"
    data["revision_plan"] = [{"step": "Redo the quiz"}, "Check the slide"]
    pack, error, _ = parse_pack(json.dumps(data))
    assert error is None
    assert pack.sources_used == ["text notes"]
    assert pack.revision_plan == ["Redo the quiz", "Check the slide"]


def test_source_names_are_normalised():
    assert normalise_source("Text Notes") == "text notes"
    assert normalise_source("audio") == "audio transcript"
    assert normalise_source("OCR slide") == "slide text"
    assert normalise_source("image") == "image caption"
    assert normalise_source("notes and transcript") == "combined"
    assert normalise_source("") == "combined"


def test_schema_for_ollama_excludes_app_field():
    schema = ollama_schema()
    assert "generated_by" not in schema["properties"] and "generated_by" not in schema["required"]
    assert set(schema["required"]) >= {"summary", "flashcards", "quiz"}
    question = schema["$defs"]["QuizQuestion"]
    assert "options" in question["required"] and question["properties"]["answer"]["minLength"] == 1
    assert schema["properties"]["quiz"]["minItems"] == 4
