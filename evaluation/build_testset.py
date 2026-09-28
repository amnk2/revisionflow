"""Build the evaluation test set from evaluation/testset/bundles.json.

For each bundle this writes:
  notes.txt             the student notes
  audio.mp3             the lecture script read by macOS text-to-speech (pink noise added for noisy clips)
  reference_audio.txt   the exact script, used as the reference transcript
  slide.png / slide.jpg the rendered slide
  reference_slide.txt   the exact slide text, used as the OCR reference

Usage: venv/bin/python evaluation/build_testset.py
"""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from scipy.io import wavfile

ROOT = Path(__file__).parent / "testset"
FONT = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
SAMPLE_RATE = 22050
SEED = 2026

STYLES = {
    # background, text colour, title size, bullet size
    "simple": ("white", (20, 20, 20), 58, 40),
    "dense": ("white", (20, 20, 20), 44, 27),
    "dark": ((28, 40, 58), (240, 240, 240), 54, 36),
    "photo": ("white", (20, 20, 20), 58, 40),
}


def pink_noise(length: int, rng: np.random.Generator) -> np.ndarray:
    """Pink (1/f) noise made by shaping white noise in the frequency domain."""
    spectrum = np.fft.rfft(rng.standard_normal(length))
    freqs = np.fft.rfftfreq(length)
    freqs[0] = freqs[1]
    noise = np.fft.irfft(spectrum / np.sqrt(freqs), n=length)
    return noise / np.std(noise)


def synthesise(bundle: dict) -> np.ndarray:
    """The lecture script read by macOS text-to-speech, as mono float samples at SAMPLE_RATE."""
    spec = bundle["audio"]
    with tempfile.TemporaryDirectory() as tmp:
        aiff, wav = Path(tmp) / "speech.aiff", Path(tmp) / "speech.wav"
        subprocess.run(["say", "-v", spec["voice"], "-r", str(spec["rate"]), "-o", str(aiff), bundle["audio_script"]],
                       check=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(aiff), "-ac", "1", "-ar", str(SAMPLE_RATE), str(wav)],
                       check=True)
        _, samples = wavfile.read(wav)
    return samples.astype(np.float32) / 32768.0


def add_noise(speech: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Scale pink noise so that speech power / noise power matches the target SNR exactly."""
    noise = pink_noise(len(speech), rng)
    noise *= np.sqrt(np.mean(speech ** 2) / (10 ** (snr_db / 10)))
    mixed = speech + noise
    return mixed / max(1.0, np.max(np.abs(mixed)))


def save_mp3(samples: np.ndarray, path: Path):
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        wavfile.write(wav, SAMPLE_RATE, (samples * 32767).astype(np.int16))
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(wav), "-b:a", "128k", str(path)], check=True)


def make_audio(bundle: dict, folder: Path, snr_db: float, rng: np.random.Generator):
    speech = synthesise(bundle)
    if bundle["audio"]["noise"]:
        speech = add_noise(speech, snr_db, rng)
    save_mp3(speech, folder / "audio.mp3")
    (folder / "reference_audio.txt").write_text(bundle["audio_script"] + "\n")


def wrap(draw, text, font, width):
    words, lines, line = text.split(), [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) <= width:
            line = trial
        else:
            lines.append(line)
            line = word
    lines.append(line)
    return lines


def make_slide(bundle: dict, folder: Path, rng: np.random.Generator):
    spec = bundle["slide"]
    background, colour, title_size, bullet_size = STYLES[spec["style"]]
    image = Image.new("RGB", (1280, 720), background)
    draw = ImageDraw.Draw(image)
    title_font, bullet_font = ImageFont.truetype(BOLD, title_size), ImageFont.truetype(FONT, bullet_size)

    y = 50
    for line in wrap(draw, spec["title"], title_font, 1160):
        draw.text((60, y), line, font=title_font, fill=colour)
        y += int(title_size * 1.25)
    y += 25
    for bullet in spec["bullets"]:
        lines = wrap(draw, bullet, bullet_font, 1100)
        draw.text((60, y), "•", font=bullet_font, fill=colour)
        for line in lines:
            draw.text((100, y), line, font=bullet_font, fill=colour)
            y += int(bullet_size * 1.3)
        y += int(bullet_size * 0.35)
    if y > 700:
        raise ValueError(f"{bundle['id']}: slide text does not fit")

    name = "slide.png"
    if spec["style"] == "photo":
        # Imitate a phone photo of a projected slide: tilt, blur, uneven lighting, JPEG compression.
        image = image.rotate(float(rng.uniform(2.0, 3.5)), resample=Image.BICUBIC, expand=True, fillcolor=(90, 90, 90))
        image = image.filter(ImageFilter.GaussianBlur(1.1))
        shade = np.linspace(1.0, 0.72, image.width)[None, :, None]
        image = Image.fromarray((np.asarray(image) * shade).astype(np.uint8))
        image = ImageEnhance.Contrast(image).enhance(0.8)
        image = image.resize((image.width * 3 // 4, image.height * 3 // 4))
        name = "slide.jpg"
        image.save(folder / name, quality=40)
    else:
        image.save(folder / name)

    reference = "\n".join([spec["title"]] + spec["bullets"])
    (folder / "reference_slide.txt").write_text(reference + "\n")
    missing = [term for term in bundle["key_terms"] if term.lower() not in reference.lower()]
    if missing:
        raise ValueError(f"{bundle['id']}: key terms not on the slide: {missing}")


def main():
    spec = json.loads((ROOT / "bundles.json").read_text())
    rng = np.random.default_rng(SEED)
    for bundle in spec["bundles"]:
        folder = ROOT / bundle["id"]
        folder.mkdir(exist_ok=True)
        (folder / "notes.txt").write_text(bundle["notes"] + "\n")
        make_audio(bundle, folder, spec["noise"]["snr_db"], rng)
        make_slide(bundle, folder, rng)
        print(f"{bundle['id']} {bundle['topic']}: audio ({bundle['audio']['voice']}, "
              f"{'noisy' if bundle['audio']['noise'] else 'clean'}), slide ({bundle['slide']['style']})")


if __name__ == "__main__":
    main()
