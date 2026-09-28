# RevisionFlow

RevisionFlow is my final project for CM3070 (BSc Computer Science, University of London). It uses the template **CM3020 Artificial Intelligence, 4.1 Project Idea 1: Orchestrating AI models to achieve a goal**.

It takes lecture audio, a slide screenshot and typed notes, and turns them into quiz questions, flashcards, key concepts and a short summary. Everything runs locally.

## How it works

1. **Add material**: notes, an audio file (mp3, wav, m4a) and/or a slide image (png, jpg).
2. **Processing**, one pre-trained model per input:
   - speech-to-text with OpenAI Whisper (`small.en`)
   - slide text with Tesseract OCR 5.5.3, after greyscale conversion, upscaling and autocontrast
   - a short visual caption with BLIP (`Salesforce/blip-image-captioning-base`)
3. **Check and edit**: the transcript, slide text and caption are shown and can be corrected before anything is generated. Whisper segments that are probably silence are flagged, because that is where Whisper tends to invent text. A relevance check (sentence embeddings, `all-MiniLM-L6-v2`) warns when the inputs seem to be about different topics.
4. **Generation**: the sources are passed as labelled blocks to `llama3.2:3b` through Ollama. The reply is constrained to a JSON schema, validated with Pydantic and repaired where possible. The app retries once with the error message, then falls back to a rule-based pack.
5. **Grounding check**: each key concept, flashcard and quiz answer is compared with the closest sentence in the sources, and weakly supported points are flagged for checking.
6. **Revise**: the quiz comes first, and answers stay hidden until they are attempted. Flashcards can be exported as CSV for Anki or Quizlet, and sessions can be saved and reopened as JSON.

## Requirements

- macOS or Linux, with Python 3.12
- [ffmpeg](https://ffmpeg.org/) (used by Whisper): `brew install ffmpeg`
- [Tesseract OCR](https://tesseract-ocr.github.io/) 5.x: `brew install tesseract`
- [Ollama](https://ollama.com/) running locally, with the model pulled: `ollama pull llama3.2:3b`
- About 3 GB of disk space for the Python environment and model files

## Setup

```bash
git clone https://github.com/amnk2/revisionflow.git
cd revisionflow
python3.12 -m venv venv
venv/bin/pip install -r requirements.txt
```

The Whisper, BLIP and embedding models download automatically the first time they are used. `run.sh` keeps them in the project's `.cache` folder.

## Running

```bash
bash run.sh
```

This opens the app at http://localhost:8501. Ollama must be running (open the Ollama app, or run `ollama serve`).

## Tests

```bash
venv/bin/python -m pytest
```

The unit tests cover:
- schema validation and repairs
- orchestration (with Ollama mocked)
- the relevance and grounding checks (with a stand-in embedder)
- OCR preprocessing
- the fallback pack
- sessions
- the Streamlit interface, using Streamlit's `AppTest`

## Reproducing the evaluation

```bash
export XDG_CACHE_HOME=$PWD/.cache HF_HOME=$PWD/.cache/huggingface
venv/bin/python evaluation/build_testset.py        # 8 bundles: notes, TTS audio (macOS `say`), rendered slides
venv/bin/python evaluation/run_technical.py all    # speech, image, relevance, generation, stress tests
venv/bin/python evaluation/run_technical.py timing # re-times the checks with the embedding model loaded
venv/bin/python evaluation/make_rating_sheet.py    # blind rating sheets for manual marking
venv/bin/python evaluation/analyse_technical.py    # tables (results/summary.md) and charts (figures/)
```

The test set is adapted from English Wikipedia (CC BY-SA 4.0); revision IDs are in `evaluation/testset/source_text/attribution.json`. The marking rubric is `evaluation/rubric.md` and the marks are in `evaluation/ratings/`. User-study materials (prepared but not run) and their analysis script are in `evaluation/user_study/`.

## Project structure

```
app.py                      Streamlit interface
run.sh                      starts the app with model caches kept in .cache/
revisionflow/
  config.py                 every model name, version and threshold
  schema.py                 revision pack structure, JSON schema for Ollama, parsing and repair
  orchestration.py          labelled context, Ollama calls, retry and fallback
  checks.py                 relevance and grounding checks
  fallback.py               rule-based pack used when the language model fails
  session.py                save/load sessions, flashcard CSV export
  processing/
    speech.py               Whisper
    vision.py               Tesseract OCR with preprocessing, BLIP caption
    text.py                 cleaning and sentence splitting
tests/                      pytest unit tests
evaluation/                 test set, evaluation scripts, results, marks, user-study materials
prototype/app_v0_9.py       the preliminary-report prototype, kept as the baseline
```

## Model versions

| Stage | Model / tool | Version |
|---|---|---|
| Speech-to-text | openai-whisper | package 20250625, model `small.en` (choice of `base`, `base.en`, `small.en`) |
| Slide text | Tesseract | 5.5.3, `--oem 1 --psm 3` |
| Caption | BLIP | `Salesforce/blip-image-captioning-base` |
| Language model | Ollama | `llama3.2:3b` (Q4_K_M, digest `a80c4f17acd5`); temperature 0, seed 42, context 8192 |
| Embeddings | sentence-transformers 4.1.0 | `all-MiniLM-L6-v2` |

## Privacy

Everything runs on the local machine. Uploaded audio is written to a temporary file for Whisper and deleted straight after. Images are only held in memory. A session file is created only when the student chooses to save one, and it contains text only.

## Limitations

- The outputs are drafts: the models can mishear, misread and state unsupported things.
- The grounding check only measures similarity, so a wrong statement that uses the same words as the source can still pass.
- Long recordings take a long time to transcribe, and there is no progress estimate.
- Full PDF processing is out of scope. Copy the text or take a screenshot instead.
- The example material in `samples/` is not included in the repository.
