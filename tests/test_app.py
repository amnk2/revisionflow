from pathlib import Path

from streamlit.testing.v1 import AppTest

from revisionflow.fallback import fallback_pack

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_start_page_renders():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value.endswith("RevisionFlow")
    process = [b for b in at.button if b.label == "Process material"][0]
    assert process.disabled  # nothing added yet


def test_loaded_session_shows_steps_two_and_three(cpu_sources):
    at = AppTest.from_file(APP, default_timeout=60)
    pack = fallback_pack(cpu_sources)
    at.session_state["sources"] = cpu_sources
    for name, text in cpu_sources.items():
        at.session_state[f"edit_{name}"] = text
    at.session_state["pack"] = pack
    at.run()
    assert not at.exception
    labels = [t.label for t in at.text_area]
    assert "Your notes (cleaned)" in labels and "Audio transcript (Whisper)" in labels
    assert [t.label for t in at.tabs][:2] == ["Quiz", "Flashcards"]
    assert len(at.text_input) >= len(pack.quiz)  # short-answer boxes plus the Topic box


def test_quiz_answer_is_hidden_until_checked(cpu_sources):
    at = AppTest.from_file(APP, default_timeout=60)
    pack = fallback_pack(cpu_sources)
    at.session_state["sources"] = cpu_sources
    at.session_state["pack"] = pack
    at.run()
    assert not any("Model answer" in i.value for i in at.info)
    at.text_input(key="q_0_text").input("my attempt").run()
    at.button(key="q_0_check").click().run()
    assert any(pack.quiz[0].answer in i.value for i in at.info)
