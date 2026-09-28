"""Image stage: Tesseract OCR for slide text and BLIP for a short visual caption."""

import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageOps

from .. import config
from .text import clean_text


@dataclass
class OcrResult:
    text: str
    mean_confidence: float
    word_count: int
    seconds: float
    settings: str


@dataclass
class CaptionResult:
    text: str
    seconds: float


def otsu_threshold(gray: np.ndarray) -> int:
    """Otsu's method: the grey level that best separates text from background."""
    histogram = np.bincount(gray.ravel(), minlength=256).astype(float)
    total = gray.size
    levels = np.arange(256)
    weight_bg = np.cumsum(histogram)
    weight_fg = total - weight_bg
    sum_bg = np.cumsum(histogram * levels)
    mean_bg = np.divide(sum_bg, weight_bg, out=np.zeros(256), where=weight_bg > 0)
    mean_fg = np.divide(sum_bg[-1] - sum_bg, weight_fg, out=np.zeros(256), where=weight_fg > 0)
    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    return int(np.argmax(between))


def preprocess_for_ocr(image: Image.Image, binarise: bool = config.OCR_BINARISE) -> Image.Image:
    """Greyscale, upscale small screenshots and optionally binarise (Tesseract recommends ~300 DPI)."""
    gray = ImageOps.grayscale(image.convert("RGB"))
    if gray.width < config.OCR_MIN_WIDTH:
        scale = config.OCR_MIN_WIDTH / gray.width
        gray = gray.resize((round(gray.width * scale), round(gray.height * scale)), Image.LANCZOS)
    gray = ImageOps.autocontrast(gray)
    if not binarise:
        return gray
    pixels = np.asarray(gray)
    threshold = otsu_threshold(pixels)
    binary = np.where(pixels > threshold, 255, 0).astype(np.uint8)
    # Tesseract expects dark text on a light background, so invert dark-themed slides.
    if (binary == 0).mean() > 0.5:
        binary = 255 - binary
    return Image.fromarray(binary)


def ocr_image(image: Image.Image, binarise: bool = config.OCR_BINARISE,
              tess_config: str = config.TESSERACT_CONFIG) -> OcrResult:
    """Read the text on a slide, keeping Tesseract's line structure."""
    import pytesseract

    start = time.perf_counter()
    prepared = preprocess_for_ocr(image, binarise=binarise)
    data = pytesseract.image_to_data(prepared, config=tess_config, output_type=pytesseract.Output.DICT)

    lines, confidences = {}, []
    for i, word in enumerate(data["text"]):
        conf = float(data["conf"][i])
        if not word.strip() or conf < max(config.OCR_MIN_CONFIDENCE, 0):
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
        confidences.append(conf)
    text = "\n".join(" ".join(words) for _, words in sorted(lines.items()))
    seconds = time.perf_counter() - start
    mean_conf = round(float(np.mean(confidences)), 1) if confidences else 0.0
    settings = f"{tess_config}, binarise={binarise}"
    return OcrResult(clean_text(text), mean_conf, len(confidences), seconds, settings)


@lru_cache(maxsize=1)
def load_blip():
    from transformers import BlipForConditionalGeneration, BlipProcessor

    processor = BlipProcessor.from_pretrained(config.BLIP_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(config.BLIP_MODEL)
    return processor, model


def caption_image(image: Image.Image) -> CaptionResult:
    """Short description of what the image shows (not its text)."""
    import torch

    processor, model = load_blip()
    start = time.perf_counter()
    inputs = processor(image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=40)
    text = processor.decode(output[0], skip_special_tokens=True).strip()
    return CaptionResult(text, time.perf_counter() - start)
