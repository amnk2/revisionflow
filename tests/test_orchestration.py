import requests

from revisionflow import orchestration
from revisionflow.orchestration import build_context, generate_pack


def fake_ollama(replies):
    """Replace call_ollama with a function that returns the given replies in order."""
    calls = []

    def _call(prompt, fmt=None):
        calls.append({"prompt": prompt, "fmt": fmt})
        reply = replies[min(len(calls), len(replies)) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply, {"total_s": 1.0, "load_s": 0.0, "prompt_tokens": 500, "output_tokens": 300, "generation_s": 0.9}

    return _call, calls


def test_context_uses_labelled_blocks_in_fixed_order():
    context = build_context({"audio transcript": "spoken words", "text notes": "my notes", "slide text": ""})
    assert context.index("[TEXT NOTES]") < context.index("[AUDIO TRANSCRIPT]")
    assert "[END AUDIO TRANSCRIPT]" in context
    assert "SLIDE TEXT" not in context  # empty sources are left out


def test_valid_reply_first_time(monkeypatch, cpu_sources, valid_pack_json):
    call, calls = fake_ollama([valid_pack_json])
    monkeypatch.setattr(orchestration, "call_ollama", call)
    result = generate_pack(cpu_sources, mode="schema")
    assert result.valid_first_try and not result.fallback_used
    assert result.attempts == 1 and len(calls) == 1
    assert calls[0]["fmt"]["type"] == "object"  # JSON schema passed to Ollama


def test_retry_includes_the_error(monkeypatch, cpu_sources, valid_pack_json):
    call, calls = fake_ollama(["not json at all", valid_pack_json])
    monkeypatch.setattr(orchestration, "call_ollama", call)
    result = generate_pack(cpu_sources, mode="schema")
    assert result.attempts == 2 and not result.valid_first_try and not result.fallback_used
    assert "previous reply could not be used" in calls[1]["prompt"]


def test_fallback_after_two_bad_replies(monkeypatch, cpu_sources):
    call, _ = fake_ollama(["bad", "still bad"])
    monkeypatch.setattr(orchestration, "call_ollama", call)
    result = generate_pack(cpu_sources)
    assert result.fallback_used and result.pack.generated_by == "fallback"
    assert any("fallback" in w for w in result.warnings)


def test_fallback_when_ollama_is_down(monkeypatch, cpu_sources):
    call, _ = fake_ollama([requests.ConnectionError("refused")])
    monkeypatch.setattr(orchestration, "call_ollama", call)
    result = generate_pack(cpu_sources)
    assert result.fallback_used and "Could not reach Ollama" in result.warnings[0]


def test_two_step_mode_drafts_then_converts(monkeypatch, cpu_sources, valid_pack_json):
    call, calls = fake_ollama(["SUMMARY: ... [text notes]", valid_pack_json])
    monkeypatch.setattr(orchestration, "call_ollama", call)
    result = generate_pack(cpu_sources, mode="two_step")
    assert not result.fallback_used
    assert calls[0]["fmt"] is None and calls[1]["fmt"] is not None
    assert "SUMMARY: ... [text notes]" in calls[1]["prompt"]


def test_long_input_warns_about_context_window(monkeypatch, valid_pack_json):
    call, _ = fake_ollama([valid_pack_json])
    monkeypatch.setattr(orchestration, "call_ollama", call)
    result = generate_pack({"text notes": "word " * 6000})
    assert any("cut off" in w for w in result.warnings)


def test_no_material_goes_straight_to_fallback():
    result = generate_pack({"text notes": "  "})
    assert result.fallback_used and result.attempts == 0
