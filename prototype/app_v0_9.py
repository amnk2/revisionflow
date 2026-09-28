import os
import tempfile

import requests
import streamlit as st
import torch
import whisper
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


st.set_page_config(page_title="RevisionFlow", layout="wide")

st.title("RevisionFlow (v0.9. prototype)")
st.write("Turning audio, image and text study materials into useful revision.")


@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")


@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model


def transcribe_audio(uploaded_audio):
    model = load_whisper_model()
    suffix = os.path.splitext(uploaded_audio.name)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_audio.write(uploaded_audio.read())
        temp_audio_path = temp_audio.name

    result = model.transcribe(temp_audio_path)
    os.remove(temp_audio_path)

    return result["text"].strip()


def describe_image(uploaded_image):
    processor, model = load_blip_model()
    image = Image.open(uploaded_image).convert("RGB")
    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=60)

    description = processor.decode(output[0], skip_special_tokens=True)
    return description.strip()


def generate_with_ollama(combined_material):
    prompt = f"""
You are generating revision material from study inputs.
Use only the material provided below.
Do not invent facts that are not supported by the material.
If the material is limited or unclear, say that clearly.

Create the following sections:
1. Source use
Briefly state what information came from:
- Text notes
- Audio transcript
- Image description
2. Summary
Write a short summary using all available sources. Do not ignore the audio transcript or image description if they contain information.
3. Key concepts
List key concepts and mark the source in brackets, for example [text notes], [audio transcript], or [image description].
4. Self-test questions
Create questions that test the important ideas from the material. Include at least one question from each available source.
5. Revision next steps
Suggest what the student should check or revise next.

Study material:
{combined_material}
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2:3b",
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def generate_fallback_output(combined_material):
    lines = []
    if "Audio transcript:" in combined_material and "Image description:" in combined_material:
        audio_text = combined_material.split("Audio transcript:", 1)[1].split("Image description:", 1)[0]
        lines.extend([line.strip() for line in audio_text.splitlines() if line.strip()])

    if "Image description:" in combined_material:
        image_text = combined_material.split("Image description:", 1)[1]
        lines.extend([line.strip() for line in image_text.splitlines() if line.strip()])

    if "Text notes:" in combined_material and "Audio transcript:" in combined_material:
        notes_text = combined_material.split("Text notes:", 1)[1].split("Audio transcript:", 1)[0]
        lines.extend([line.strip() for line in notes_text.splitlines() if line.strip()])

    useful_lines = []
    for line in lines:
        if line.endswith(":"):
            continue
        if line in ["No audio file was uploaded.", "No image file was uploaded."]:
            continue
        for sentence in line.split("."):
            sentence = sentence.strip()
            if len(sentence) > 20:
                useful_lines.append(sentence + ".")

    selected_lines = useful_lines[:8]

    if not selected_lines:
        selected_lines = ["No usable study material was provided."]

    summary = "The material has been combined into one revision source. The main ideas should be checked against the transcript, image description and notes before using them for exam revision."

    key_concepts = "\n".join([f"- {line}" for line in selected_lines[:5]])

    questions = "\n".join(
        [
            "1. What are the main ideas in the combined study material?",
            "2. Which points are supported by the notes, transcript or image description?",
            "3. Which parts need checking against the original source?",
            "4. What topic should be revised next based on the weakest or least clear material?",
        ]
    )

    next_steps = "\n".join(
        [
            "- Check the transcript for transcription errors.",
            "- Compare the image description with the original screenshot.",
            "- Rewrite any unclear points in the notes.",
            "- Use the self-test questions for active recall.",
        ]
    )

    return f"""
Summary
{summary}

Key concepts
{key_concepts}

Self-test questions
{questions}

Revision next steps
{next_steps}
""".strip()


def build_combined_material(text_notes, transcript, image_description):
    return f"""
Text notes:
{text_notes}

Audio transcript:
{transcript}

Image description:
{image_description}
""".strip()


st.header("1. Text notes")
text_notes = st.text_area("Paste study notes here", height=180)

st.header("2. Audio input")
uploaded_audio = st.file_uploader("Upload a short audio file", type=["mp3", "wav", "m4a"])

st.header("3. Image or screenshot input")
uploaded_image = st.file_uploader("Upload an image or screenshot", type=["png", "jpg", "jpeg"])

use_ollama = st.checkbox("Use Ollama for final revision output", value=True)

if st.button("Run prototype"):
    transcript = "No audio file was uploaded."
    image_description = "No image file was uploaded."

    if uploaded_audio is not None:
        with st.spinner("Transcribing audio with Whisper"):
            transcript = transcribe_audio(uploaded_audio)

    if uploaded_image is not None:
        with st.spinner("Generating image description with BLIP"):
            image_description = describe_image(uploaded_image)

    combined_material = build_combined_material(text_notes, transcript, image_description)

    st.header("Intermediate output: audio transcript")
    st.write(transcript)

    st.header("Intermediate output: image description")
    st.write(image_description)

    st.header("Combined source material")
    st.text_area("Material passed to the revision generation stage", combined_material, height=260)

    st.header("Final revision output")

    if use_ollama:
        try:
            with st.spinner("Generating revision output with Ollama"):
                revision_output = generate_with_ollama(combined_material)
        except Exception as error:
            st.warning("Ollama generation was not available. Fallback generation was used for this run.")
            st.caption(str(error))
            revision_output = generate_fallback_output(combined_material)
    else:
        revision_output = generate_fallback_output(combined_material)

    st.write(revision_output)
