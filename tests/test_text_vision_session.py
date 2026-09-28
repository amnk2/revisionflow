import shutil

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from revisionflow import config
from revisionflow.fallback import fallback_pack
from revisionflow.processing.text import clean_text, split_sentences
from revisionflow.processing.vision import ocr_image, otsu_threshold, preprocess_for_ocr
from revisionflow.session import flashcards_csv, session_from_json, session_to_json


def test_clean_text_joins_hyphenated_words_and_trims():
    assert clean_text("  The data-\npath   holds\r\n\r\n\r\nregisters  ") == "The datapath holds\n\nregisters"


def test_split_sentences_handles_bullets_and_short_fragments():
    text = "- The ALU does arithmetic.\n- Registers store values. Buses move data.\nOK"
    assert split_sentences(text) == ["The ALU does arithmetic.", "Registers store values.", "Buses move data."]


def test_otsu_separates_two_levels():
    pixels = np.array([[20] * 50 + [230] * 50] * 10, dtype=np.uint8)
    assert 20 <= otsu_threshold(pixels) < 230


def test_preprocessing_upscales_and_inverts_dark_slides():
    dark = Image.new("RGB", (800, 450), "black")
    ImageDraw.Draw(dark).text((50, 50), "Datapath", fill="white")
    prepared = preprocess_for_ocr(dark, binarise=True)
    assert prepared.width == config.OCR_MIN_WIDTH
    assert (np.asarray(prepared) == 255).mean() > 0.5  # background now light


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract not installed")
def test_ocr_reads_simple_slide():
    slide = Image.new("RGB", (1200, 400), "white")
    font = ImageFont.load_default(size=60)
    ImageDraw.Draw(slide).text((60, 120), "The datapath contains the ALU", fill="black", font=font)
    result = ocr_image(slide)
    assert "datapath" in result.text.lower() and result.mean_confidence > 50


def test_fallback_only_reuses_source_sentences(cpu_sources):
    pack = fallback_pack(cpu_sources)
    assert pack.generated_by == "fallback" and pack.flashcards and pack.quiz
    all_text = " ".join(cpu_sources.values())
    for card in pack.flashcards:
        assert card.back in all_text and "_____" in card.front


def test_session_round_trip_and_csv(cpu_sources):
    pack = fallback_pack(cpu_sources)
    topic, sources, loaded = session_from_json(session_to_json("CPU", cpu_sources, pack))
    assert topic == "CPU" and sources == cpu_sources and loaded == pack
    rows = flashcards_csv(pack).strip().splitlines()
    assert len(rows) == len(pack.flashcards)


def test_foreign_json_is_rejected():
    with pytest.raises(ValueError):
        session_from_json('{"app": "Other"}')
