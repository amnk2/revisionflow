import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeEmbedder:
    """Bag-of-words vectors: texts sharing words are similar. Lets the checks be tested without the real model."""

    def encode(self, texts):
        vectors = []
        for text in texts:
            vector = np.zeros(512)
            for word in text.lower().split():
                word = word.strip(".,:;!?()\"'")
                if len(word) > 3:
                    vector[zlib.crc32(word.encode()) % 512] += 1  # hash() changes between runs
            norm = np.linalg.norm(vector)
            vectors.append(vector / norm if norm else vector)
        return np.array(vectors)


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def cpu_sources():
    return {
        "text notes": "The CPU fetches instructions from memory. The control unit decodes each instruction. "
                      "The datapath contains the ALU and the registers.",
        "audio transcript": "In this lecture we build the datapath. The ALU and the registers are connected by buses. "
                            "The control unit sends signals to the datapath every clock cycle.",
    }


@pytest.fixture
def valid_pack_json():
    return """{
      "sources_used": ["text notes", "Audio Transcript"],
      "summary": "The CPU fetches, decodes and executes instructions using the datapath and control unit.",
      "key_concepts": [
        {"concept": "Datapath", "explanation": "Contains the ALU and the registers.", "source": "text notes"}
      ],
      "flashcards": [
        {"front": "What does the control unit do?", "back": "It decodes each instruction.", "source": "notes"},
        {"front": "", "back": "empty front", "source": "notes"}
      ],
      "quiz": [
        {"kind": "multiple_choice", "question": "Which part performs arithmetic?", "options": ["ALU", "Bus", "Clock", "Cache"],
         "explanation": "The notes say the datapath contains the ALU.", "answer": "A", "source": "text notes"},
        {"kind": "multiple_choice", "question": "Broken question?", "options": ["One", "Two", "Four", "Five"],
         "explanation": "x", "answer": "Three", "source": "text notes"},
        {"kind": "multiple_choice", "question": "Which unit sends control signals?", "options": ["ALU", "Bus"],
         "explanation": "The notes say the control unit sends signals.", "answer": "Control unit", "source": "text notes"},
        {"kind": "multiple_choice", "question": "What does the datapath contain?",
         "explanation": "From the notes.", "answer": "The ALU and registers", "source": "text notes"},
        {"kind": "short_answer", "question": "Empty answer?", "options": [], "explanation": "x", "answer": "", "source": "text notes"},
        {"kind": "short answer", "question": "What connects the ALU and registers?", "options": ["ignored"],
         "explanation": "The lecture says buses connect them.", "answer": "Buses", "source": "lecture audio"}
      ],
      "revision_plan": "Review the datapath."
    }"""
