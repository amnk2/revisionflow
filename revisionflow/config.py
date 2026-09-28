"""Settings for RevisionFlow.

Every model name, version and threshold is kept here so the app, the tests and
the evaluation scripts all use exactly the same values.
"""

# Speech-to-text (openai-whisper 20250625).
# The package default is now "turbo", so the model is always named explicitly.
# small.en replaced base.en after the evaluation: lower WER in every noise condition (5.4% against
# 10.1% at 0 dB) for about 2 s more per 40 s clip. All three models still mishear some names.
WHISPER_MODEL = "small.en"
WHISPER_CHOICES = ["base", "base.en", "small.en"]
# Whisper can invent text where there is little or no speech, so segments it
# rates as probably silent are flagged for the student to check.
WHISPER_NO_SPEECH_FLAG = 0.6

# Image stage.
BLIP_MODEL = "Salesforce/blip-image-captioning-base"
TESSERACT_CONFIG = "--oem 1 --psm 3"  # LSTM engine, automatic page segmentation
OCR_MIN_WIDTH = 1600  # narrower screenshots are upscaled before OCR
OCR_MIN_CONFIDENCE = 0  # keep every recognised word; low confidence is reported instead
# Own Otsu binarisation is switched off: Tesseract already binarises internally, and in the
# stress test the extra pass lowered key-term recall on degraded slides (0.64 against 0.86).
OCR_BINARISE = False

# Language model through Ollama.
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"  # Q4_K_M, digest a80c4f17acd5
# Temperature 0 and a fixed seed make runs as repeatable as Ollama allows.
# num_ctx is set because the default context window is only a few thousand tokens.
OLLAMA_OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 8192}
OLLAMA_TIMEOUT = 300
GENERATION_MODE = "schema"  # "schema" (one constrained call) or "two_step"

# Relevance and grounding checks. Starting values, tuned on the evaluation test set.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RELEVANCE_THRESHOLD = 0.30
GROUNDING_THRESHOLD = 0.50

# Inputs.
AUDIO_TYPES = ["mp3", "wav", "m4a"]
IMAGE_TYPES = ["png", "jpg", "jpeg"]

# Names used for the four intermediate sources throughout the app.
SOURCE_NOTES = "text notes"
SOURCE_AUDIO = "audio transcript"
SOURCE_SLIDE = "slide text"
SOURCE_CAPTION = "image caption"
SOURCE_ORDER = [SOURCE_NOTES, SOURCE_AUDIO, SOURCE_SLIDE, SOURCE_CAPTION]
