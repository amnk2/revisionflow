"""Speech-to-text stage using OpenAI Whisper, run locally."""

import time
from dataclasses import dataclass, field
from functools import lru_cache

from .. import config
from .text import clean_text


@dataclass
class TranscriptResult:
    text: str
    model: str
    seconds: float
    duration: float  # length of the audio in seconds
    flagged_segments: list = field(default_factory=list)


@lru_cache(maxsize=2)
def load_model(name: str = config.WHISPER_MODEL):
    import whisper  # imported here so the rest of the app starts quickly

    return whisper.load_model(name)


def transcribe(path: str, model_name: str = config.WHISPER_MODEL) -> TranscriptResult:
    """Transcribe an audio file and flag segments that may contain invented text."""
    model = load_model(model_name)
    start = time.perf_counter()
    result = model.transcribe(
        path,
        fp16=False,  # CPU only; avoids the FP16 warning seen in the prototype
        language="en" if model_name.endswith(".en") else None,
        condition_on_previous_text=False,  # reduces repetition loops on long audio
    )
    seconds = time.perf_counter() - start

    segments = result.get("segments", [])
    flagged = [
        {"start": round(s["start"], 1), "end": round(s["end"], 1), "text": s["text"].strip(),
         "no_speech_prob": round(s["no_speech_prob"], 2)}
        for s in segments
        if s.get("no_speech_prob", 0) >= config.WHISPER_NO_SPEECH_FLAG and s["text"].strip()
    ]
    duration = segments[-1]["end"] if segments else 0.0
    return TranscriptResult(clean_text(result["text"]), model_name, seconds, duration, flagged)
