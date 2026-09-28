"""RevisionFlow: Streamlit interface.

Workflow (design chapter): add material -> check and edit what the models
understood -> generate -> revise with the quiz and flashcards first.
"""

import os
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from revisionflow import config
from revisionflow.checks import Embedder, grounding_check, quiz_item_text, relevance_check
from revisionflow.orchestration import generate_pack
from revisionflow.processing import speech, vision
from revisionflow.processing.text import clean_text
from revisionflow.session import flashcards_csv, session_from_json, session_to_json

SAMPLES = Path(__file__).parent / "samples"
AI_WARNING = ("RevisionFlow uses AI models that can mishear audio, misread slides and state things that are not in "
              "your material. Treat everything here as a draft and check it against your original sources.")

st.set_page_config(page_title="RevisionFlow", layout="wide")


@st.cache_resource(show_spinner=False)
def get_embedder():
    return Embedder()


def reset_outputs():
    for key in list(st.session_state):
        if key.startswith(("edit_", "q_", "fc_")) or key in ("pack", "generation", "grounding", "relevance", "report"):
            del st.session_state[key]


def load_session():
    """Runs as a callback, before the page is drawn, so it can set the Topic and text boxes."""
    uploaded = st.session_state.get("session_file")
    if uploaded is None:
        return
    try:
        topic, sources, pack = session_from_json(uploaded.getvalue().decode("utf-8"))
    except (ValueError, KeyError) as exc:
        st.session_state["load_message"] = ("error", f"Could not open that file: {exc}")
        return
    reset_outputs()
    st.session_state.update(topic=topic, sources=sources, timings={})
    for name, text in sources.items():
        st.session_state[f"edit_{name}"] = text
    if pack:
        st.session_state["pack"] = pack
    st.session_state["load_message"] = ("success", "Session loaded.")


def sample_files():
    notes = SAMPLES / "notes.txt"
    audio = next(iter(sorted(SAMPLES.glob("*.mp3")) + sorted(SAMPLES.glob("*.wav"))), None)
    image = next(iter(sorted(SAMPLES.glob("*.png")) + sorted(SAMPLES.glob("*.jpg"))), None)
    return notes if notes.exists() else None, audio, image


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Revision session")
    st.text_input("Topic", key="topic", placeholder="e.g. CPU datapath")

    st.file_uploader("Open a saved session", type=["json"], key="session_file", on_change=load_session)
    if "load_message" in st.session_state:
        kind, message = st.session_state.pop("load_message")
        (st.success if kind == "success" else st.error)(message)

    with st.expander("Settings"):
        whisper_model = st.selectbox("Speech-to-text model", config.WHISPER_CHOICES,
                                     index=config.WHISPER_CHOICES.index(config.WHISPER_MODEL))
        use_ocr = st.checkbox("Read slide text (OCR)", value=True)
        use_caption = st.checkbox("Describe the image (caption)", value=True)
        mode = st.radio("Generation mode", ["schema", "two_step"],
                        index=["schema", "two_step"].index(config.GENERATION_MODE),
                        help="schema: one call constrained to the JSON schema. "
                             "two_step: write freely first, then convert to JSON.")
    if any(sample_files()):
        use_samples = st.toggle("Use the example material", key="use_samples",
                                help="Loads the prepared example files instead of your own uploads.")
    else:
        use_samples = False
    if st.button("Start a new session"):
        for key in list(st.session_state):
            del st.session_state[key]
        st.rerun()

# ---------------------------------------------------------------- step 1
st.title("RevisionFlow")
st.caption("Turns your own lecture audio, slide screenshots and notes into quiz questions and flashcards. "
           "Everything runs locally on this computer.")
st.info(AI_WARNING)

st.subheader("Step 1: Add your material")
col_notes, col_files = st.columns([3, 2])
with col_notes:
    notes = st.text_area("Your notes", height=220, placeholder="Paste or type your notes here")
with col_files:
    audio_file = st.file_uploader("Lecture audio (mp3, wav, m4a)", type=config.AUDIO_TYPES)
    image_file = st.file_uploader("Slide or screenshot (png, jpg)", type=config.IMAGE_TYPES)

if use_samples:
    s_notes, s_audio, s_image = sample_files()
    st.caption("Using example material: " + ", ".join(p.name for p in (s_notes, s_audio, s_image) if p))


def process_material():
    """Run each input through its model and keep only the resulting text."""
    reset_outputs()
    sources, timings, report = {}, {}, {}
    notes_text, audio_path, image = notes, None, None
    temp_path = None

    if use_samples:
        s_notes, s_audio, s_image = sample_files()
        notes_text = s_notes.read_text() if s_notes else notes_text
        audio_path = str(s_audio) if s_audio else None
        image = Image.open(s_image) if s_image else None
    else:
        if audio_file is not None:
            suffix = os.path.splitext(audio_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_file.getvalue())
                temp_path = audio_path = tmp.name
        if image_file is not None:
            image = Image.open(image_file)

    sources[config.SOURCE_NOTES] = clean_text(notes_text)
    try:
        if audio_path:
            with st.spinner(f"Transcribing audio with Whisper ({whisper_model})..."):
                result = speech.transcribe(audio_path, whisper_model)
            sources[config.SOURCE_AUDIO] = result.text
            timings["speech-to-text"] = result.seconds
            report["audio"] = result
    finally:
        if temp_path:
            os.remove(temp_path)  # uploads are never kept

    if image is not None:
        if use_ocr:
            with st.spinner("Reading slide text with Tesseract OCR..."):
                ocr = vision.ocr_image(image)
            sources[config.SOURCE_SLIDE] = ocr.text
            timings["OCR"] = ocr.seconds
            report["ocr"] = ocr
        if use_caption:
            with st.spinner("Describing the image with BLIP..."):
                caption = vision.caption_image(image)
            sources[config.SOURCE_CAPTION] = caption.text
            timings["caption"] = caption.seconds

    sources = {name: text for name, text in sources.items() if text}
    for name, text in sources.items():
        st.session_state[f"edit_{name}"] = text
    st.session_state.update(sources=sources, timings=timings, report=report)
    with st.spinner("Checking the inputs are about the same topic..."):
        st.session_state["relevance"] = relevance_check(sources, embedder=get_embedder())


has_input = bool(notes.strip() or audio_file or image_file or use_samples)
if st.button("Process material", type="primary", disabled=not has_input):
    process_material()
elif st.session_state.get("sources"):
    st.caption("Processing again replaces the text in step 2, including any corrections you have made.")

# ---------------------------------------------------------------- step 2
if st.session_state.get("sources"):
    st.divider()
    st.subheader("Step 2: Check what the AI understood")
    st.caption("These are the texts the language model will work from: your notes, what the speech model heard in "
               "the audio (Whisper), the text read from the slide (OCR) and a one-line description of the image (BLIP). "
               "Correct any mistakes before generating.")
    report = st.session_state.get("report", {})
    labels = {
        config.SOURCE_NOTES: "Your notes (cleaned)",
        config.SOURCE_AUDIO: "Audio transcript (Whisper)",
        config.SOURCE_SLIDE: "Slide text (OCR)",
        config.SOURCE_CAPTION: "Image description (BLIP)",
    }
    for name in config.SOURCE_ORDER:
        if f"edit_{name}" not in st.session_state:
            continue
        st.text_area(labels[name], key=f"edit_{name}", height=160 if name != config.SOURCE_CAPTION else 68)
        if name == config.SOURCE_AUDIO and "audio" in report:
            flagged = report["audio"].flagged_segments
            if flagged:
                with st.expander(f"{len(flagged)} part(s) of the transcript may contain invented words"):
                    st.caption("Whisper judged these parts probably silent, which is where it tends to invent text.")
                    for seg in flagged:
                        st.write(f"{seg['start']}s to {seg['end']}s: \"{seg['text']}\"")
        if name == config.SOURCE_SLIDE and "ocr" in report:
            ocr = report["ocr"]
            st.caption(f"{ocr.word_count} words read, mean OCR confidence {ocr.mean_confidence}%.")

    relevance = st.session_state.get("relevance")
    if relevance and relevance.applicable:
        if relevance.flagged:
            pairs = "; ".join(f"{a} and {b} (similarity {v:.2f})" for a, b, v in relevance.flagged)
            st.warning(f"These inputs look like they are about different topics: {pairs}. "
                       "Mixing unrelated material usually makes the output confusing. "
                       "Remove or edit one of them unless they really belong together.")
        else:
            st.success("Your inputs look like they are about the same topic.")

    if st.button("Generate revision material", type="primary"):
        edited = {name: st.session_state[f"edit_{name}"].strip()
                  for name in config.SOURCE_ORDER if st.session_state.get(f"edit_{name}", "").strip()}
        st.session_state["sources"] = edited
        embedder = get_embedder()
        st.session_state["relevance"] = relevance_check(edited, embedder=embedder)
        with st.spinner("Generating with llama3.2:3b through Ollama. This usually takes about 30 seconds..."):
            generation = generate_pack(edited, mode=mode)
        start = time.perf_counter()
        grounding = grounding_check(generation.pack, edited, embedder=embedder)
        timings = st.session_state.setdefault("timings", {})
        timings["generation"] = generation.timings.get("total_s", 0.0)
        timings["grounding check"] = time.perf_counter() - start
        for key in [k for k in st.session_state if k.startswith(("q_", "fc_"))]:
            del st.session_state[key]
        st.session_state.update(pack=generation.pack, generation=generation, grounding=grounding)
        st.rerun()

# ---------------------------------------------------------------- step 3
pack = st.session_state.get("pack")
if pack:
    st.divider()
    st.subheader("Step 3: Revise")
    generation = st.session_state.get("generation")
    if pack.generated_by == "fallback":
        st.warning("The language model was not available or its reply could not be used, so these are simple "
                   "fill-the-gap items built directly from your sentences.")
    if generation:
        for warning in generation.warnings:
            st.caption(f"Warning: {warning}")

    grounding = {g.text: g for g in st.session_state.get("grounding", [])}
    weak = sum(not g.supported for g in grounding.values())
    if weak:
        st.caption(f"{weak} generated point(s) have weak support in your material and are marked below.")

    def support_note(text):
        item = grounding.get(text)
        if item and not item.supported:
            with st.expander("Weak support in your material: check this"):
                st.write(f"Closest sentence ({item.best_source}, similarity {item.score:.2f}):")
                st.caption(f'"{item.best_sentence}"' if item.best_sentence else "No matching sentence found.")

    tab_quiz, tab_cards, tab_concepts, tab_summary, tab_plan, tab_details = st.tabs(
        ["Quiz", "Flashcards", "Key concepts", "Summary", "Revision plan", "Checks and details"])

    with tab_quiz:
        st.caption("Answer each question before checking it. Testing yourself helps you remember more than rereading.")
        for i, q in enumerate(pack.quiz):
            with st.container(border=True):
                st.markdown(f"**Q{i + 1}.** {q.question}")
                if q.kind == "multiple_choice":
                    choice = st.radio("Your answer", q.options, index=None, key=f"q_{i}_choice",
                                      label_visibility="collapsed")
                else:
                    choice = st.text_input("Your answer", key=f"q_{i}_text", placeholder="Type your answer")
                if st.button("Check answer", key=f"q_{i}_check", disabled=not choice):
                    st.session_state[f"q_{i}_checked"] = True
                if st.session_state.get(f"q_{i}_checked"):
                    if q.kind == "multiple_choice":
                        if choice == q.answer:
                            st.success(f"Correct: {q.answer}")
                        else:
                            st.error(f"Not quite. The answer is: {q.answer}")
                    else:
                        st.info(f"Model answer: {q.answer}")
                        st.caption("Compare it with yours. Did you get the key idea?")
                    st.caption(f"Why: {q.explanation}")
                    st.caption(f"Source: {q.source}")
                    support_note(quiz_item_text(q))
        if not pack.quiz:
            st.write("No questions were generated.")

    with tab_cards:
        st.caption("Try to recall the answer before revealing it.")
        for i, card in enumerate(pack.flashcards):
            with st.container(border=True):
                st.markdown(f"**{card.front}**")
                if st.toggle("Show answer", key=f"fc_{i}"):
                    st.write(card.back)
                    st.caption(f"Source: {card.source}")
                support_note(f"{card.front} {card.back}")
        st.download_button("Download flashcards (CSV for Anki or Quizlet)", flashcards_csv(pack),
                           file_name=f"{st.session_state.get('topic') or 'revisionflow'}-flashcards.csv",
                           mime="text/csv", disabled=not pack.flashcards)

    with tab_concepts:
        for concept in pack.key_concepts:
            st.markdown(f"**{concept.concept}:** {concept.explanation}  \n*Source: {concept.source}*")
            support_note(f"{concept.concept}: {concept.explanation}")

    with tab_summary:
        st.write(pack.summary)
        st.caption("A summary is a starting point. The quiz and flashcards are the better way to revise.")

    with tab_plan:
        for step in pack.revision_plan:
            st.markdown(f"- {step}")

    with tab_details:
        st.markdown("**Sources used:** " + (", ".join(pack.sources_used) or "none reported"))
        if generation:
            st.markdown(f"**Generation:** mode `{generation.mode}`, {generation.attempts} attempt(s), "
                        f"{'fallback used' if generation.fallback_used else 'language model output'}")
            if generation.repairs:
                with st.expander(f"{len(generation.repairs)} item(s) repaired or removed during validation"):
                    for repair in generation.repairs:
                        st.caption(repair)
        relevance = st.session_state.get("relevance")
        if relevance and relevance.scores:
            st.markdown("**Topic similarity between inputs**")
            st.dataframe(pd.DataFrame([{"inputs": k, "similarity": v} for k, v in relevance.scores.items()]),
                         hide_index=True)
        if grounding:
            st.markdown(f"**Grounding check** (threshold {config.GROUNDING_THRESHOLD})")
            st.dataframe(pd.DataFrame([{"type": g.kind, "item": g.text, "similarity": g.score,
                                        "closest source": g.best_source, "supported": g.supported}
                                       for g in grounding.values()]), hide_index=True)
        timings = st.session_state.get("timings", {})
        if timings:
            st.markdown("**Processing time (seconds)**")
            st.dataframe(pd.DataFrame([{"stage": k, "seconds": round(v, 2)} for k, v in timings.items()]),
                         hide_index=True)

    st.download_button("Save this session", session_to_json(st.session_state.get("topic", ""),
                                                            st.session_state.get("sources", {}), pack),
                       file_name=f"{st.session_state.get('topic') or 'revisionflow'}-session.json",
                       mime="application/json")
