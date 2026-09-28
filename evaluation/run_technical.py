"""Technical evaluation of RevisionFlow on the test set (Evaluation chapter, section 5.2).

Stages (each writes CSV files to evaluation/results/):
  asr         Whisper base, base.en and small.en: word error rate and time per clip
  image       Tesseract OCR (with and without binarisation) and BLIP: key-term recall, CER, time
  relevance   embedding and TF-IDF relevance check on 8 matched and 8 mismatched bundles
  generation  schema and two_step modes, 3 runs per bundle: validity, repairs, counts, timings, grounding scores
  all         every stage in order

Usage: ./run.sh is not needed; run with the project environment and model caches set, e.g.
  XDG_CACHE_HOME=$PWD/.cache HF_HOME=$PWD/.cache/huggingface venv/bin/python evaluation/run_technical.py all
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jiwer  # noqa: E402
from PIL import Image  # noqa: E402

from revisionflow import config  # noqa: E402
from revisionflow.checks import Embedder, grounding_check, relevance_check  # noqa: E402
from revisionflow.orchestration import generate_pack  # noqa: E402
from revisionflow.processing import speech, vision  # noqa: E402

TESTSET = ROOT / "evaluation" / "testset"
RESULTS = ROOT / "evaluation" / "results"
BUNDLES = json.loads((TESTSET / "bundles.json").read_text())["bundles"]
WHISPER_MODELS = ["base", "base.en", "small.en"]
MODES = ["schema", "two_step"]
RUNS = 3


def write_csv(name, rows):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / name, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {name} ({len(rows)} rows)")


def slide_path(bundle_id):
    folder = TESTSET / bundle_id
    return folder / "slide.png" if (folder / "slide.png").exists() else folder / "slide.jpg"


# ------------------------------------------------------------------ ASR
def run_asr():
    from whisper.normalizers import EnglishTextNormalizer

    normalise = EnglishTextNormalizer()  # the normaliser used in the Whisper paper
    rows = []
    for model in WHISPER_MODELS:
        speech.load_model(model)  # load once so timings exclude loading
        for b in BUNDLES:
            reference = (TESTSET / b["id"] / "reference_audio.txt").read_text()
            result = speech.transcribe(str(TESTSET / b["id"] / "audio.mp3"), model)
            wer = jiwer.wer(normalise(reference), normalise(result.text))
            rows.append({"bundle": b["id"], "model": model, "voice": b["audio"]["voice"], "accent": b["audio"]["accent"],
                         "condition": "noisy" if b["audio"]["noise"] else "clean", "duration_s": round(result.duration, 1),
                         "seconds": round(result.seconds, 2), "wer": round(wer, 4),
                         "flagged_segments": len(result.flagged_segments), "hypothesis": result.text})
            print(f"  {model:9} {b['id']} WER {wer:.3f} ({result.seconds:.1f}s)")
    write_csv("asr.csv", rows)


# ------------------------------------------------------------------ image
def _norm(text):
    return re.sub(r"\s+", " ", text.lower().replace("•", " ")).strip()


def key_term_recall(text, terms):
    found = [t for t in terms if _norm(t) in _norm(text)]
    return len(found) / len(terms), found


def run_image():
    rows = []
    vision.load_blip()
    for b in BUNDLES:
        image = Image.open(slide_path(b["id"]))
        reference = (TESTSET / b["id"] / "reference_slide.txt").read_text()
        outputs = {}
        for binarise in (True, False):
            ocr = vision.ocr_image(image, binarise=binarise)
            outputs[f"ocr_binarised={binarise}"] = (ocr.text, ocr.seconds, ocr.mean_confidence)
        caption = vision.caption_image(image)
        outputs["blip_caption"] = (caption.text, caption.seconds, "")
        best_ocr = outputs[f"ocr_binarised={config.OCR_BINARISE}"][0]
        outputs["ocr+caption"] = (best_ocr + "\n" + caption.text, None, "")
        for method, (text, seconds, conf) in outputs.items():
            recall, found = key_term_recall(text, b["key_terms"])
            cer = jiwer.cer(_norm(reference), _norm(text)) if method.startswith("ocr_") else ""
            rows.append({"bundle": b["id"], "style": b["slide"]["style"], "method": method,
                         "key_term_recall": round(recall, 3), "cer": round(cer, 4) if cer != "" else "",
                         "mean_confidence": conf, "seconds": round(seconds, 2) if seconds else "",
                         "terms_found": "; ".join(found), "text": text})
            print(f"  {b['id']} {method:20} recall {recall:.2f} {'' if cer == '' else f'CER {cer:.3f}'}")
    write_csv("image.csv", rows)


# ------------------------------------------------------------------ sources used by later stages
def system_sources(bundle, transcript_by_bundle, ocr_by_bundle, caption_by_bundle, audio_from=None):
    audio_id = audio_from or bundle["id"]
    return {
        config.SOURCE_NOTES: bundle["notes"],
        config.SOURCE_AUDIO: transcript_by_bundle[audio_id],
        config.SOURCE_SLIDE: ocr_by_bundle[bundle["id"]],
        config.SOURCE_CAPTION: caption_by_bundle[bundle["id"]],
    }


def load_stage_outputs():
    """Use the outputs of the configured Whisper model and OCR setting (see config.py), plus the BLIP caption, as the system's sources."""
    with open(RESULTS / "asr.csv") as f:
        transcripts = {r["bundle"]: r["hypothesis"] for r in csv.DictReader(f) if r["model"] == config.WHISPER_MODEL}
    with open(RESULTS / "image.csv") as f:
        rows = list(csv.DictReader(f))
    ocr = {r["bundle"]: r["text"] for r in rows if r["method"] == f"ocr_binarised={config.OCR_BINARISE}"}
    captions = {r["bundle"]: r["text"] for r in rows if r["method"] == "blip_caption"}
    return transcripts, ocr, captions


# ------------------------------------------------------------------ relevance
def run_relevance():
    transcripts, ocr, captions = load_stage_outputs()
    embedder = Embedder()
    ids = [b["id"] for b in BUNDLES]
    cases = []
    for i, b in enumerate(BUNDLES):
        cases.append((b, b["id"], "matched"))
        cases.append((b, ids[(i + 4) % len(ids)], "mismatched"))  # audio from a bundle on a different subject

    rows = []
    for b, audio_from, label in cases:
        sources = system_sources(b, transcripts, ocr, captions, audio_from)
        for method in ("embedding", "tfidf"):
            start = time.perf_counter()
            result = relevance_check(sources, method=method, embedder=embedder, threshold=-1)
            seconds = time.perf_counter() - start
            rows.append({"bundle": b["id"], "audio_from": audio_from, "case": label, "method": method,
                         "min_score": min(result.scores.values()), "seconds": round(seconds, 3),
                         **{k: v for k, v in result.scores.items()}})
    write_csv("relevance.csv", rows)

    # Harder case: two bundles from the same subject (both computer science) swapped.
    by_id = {b["id"]: b for b in BUNDLES}
    near = []
    for base_id, audio_id in (("B1", "B8"), ("B8", "B1")):
        sources = system_sources(by_id[base_id], transcripts, ocr, captions, audio_id)
        for method in ("embedding", "tfidf"):
            result = relevance_check(sources, method=method, embedder=embedder, threshold=-1)
            near.append({"bundle": base_id, "audio_from": audio_id, "method": method,
                         "min_score": min(result.scores.values()),
                         "flagged_at_threshold": min(result.scores.values()) < config.RELEVANCE_THRESHOLD})
    write_csv("relevance_near.csv", near)

    sweep = []
    for method in ("embedding", "tfidf"):
        scores = [(r["min_score"], r["case"]) for r in rows if r["method"] == method]
        for step in range(0, 81):
            threshold = step / 100
            tp = sum(s < threshold and c == "mismatched" for s, c in scores)
            fp = sum(s < threshold and c == "matched" for s, c in scores)
            fn = sum(s >= threshold and c == "mismatched" for s, c in scores)
            tn = sum(s >= threshold and c == "matched" for s, c in scores)
            sweep.append({"method": method, "threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                          "accuracy": (tp + tn) / len(scores),
                          "precision": tp / (tp + fp) if tp + fp else "", "recall": tp / (tp + fn) if tp + fn else ""})
    write_csv("relevance_sweep.csv", sweep)


# ------------------------------------------------------------------ generation
def run_generation():
    transcripts, ocr, captions = load_stage_outputs()
    embedder = Embedder()
    out_dir = RESULTS / "packs"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, grounding_rows = [], []
    for b in BUNDLES:
        sources = system_sources(b, transcripts, ocr, captions)
        for mode in MODES:
            for run in range(1, RUNS + 1):
                result = generate_pack(sources, mode=mode)
                start = time.perf_counter()
                grounding = grounding_check(result.pack, sources, embedder=embedder)
                grounding_s = time.perf_counter() - start
                pack = result.pack
                name = f"{b['id']}_{mode}_run{run}"
                (out_dir / f"{name}.json").write_text(json.dumps({
                    "bundle": b["id"], "mode": mode, "run": run, "sources": sources, "pack": pack.model_dump(),
                    "attempts": result.attempts, "fallback_used": result.fallback_used, "warnings": result.warnings,
                    "repairs": result.repairs, "timings": result.timings}, indent=2))
                tokens_in = sum(v.get("prompt_tokens", 0) for v in result.timings.values() if isinstance(v, dict))
                tokens_out = sum(v.get("output_tokens", 0) for v in result.timings.values() if isinstance(v, dict))
                rows.append({
                    "bundle": b["id"], "mode": mode, "run": run, "attempts": result.attempts,
                    "valid_first_try": result.valid_first_try, "fallback_used": result.fallback_used,
                    "repairs": len(result.repairs), "key_concepts": len(pack.key_concepts),
                    "flashcards": len(pack.flashcards), "quiz": len(pack.quiz),
                    "quiz_mcq": sum(q.kind == "multiple_choice" for q in pack.quiz),
                    "quiz_short": sum(q.kind == "short_answer" for q in pack.quiz),
                    "sources_used": "; ".join(pack.sources_used), "generation_s": round(result.timings.get("total_s", 0), 2),
                    "grounding_s": round(grounding_s, 3), "prompt_tokens": tokens_in, "output_tokens": tokens_out,
                    "flagged_weak": sum(not g.supported for g in grounding)})
                for g in grounding:
                    grounding_rows.append({"bundle": b["id"], "mode": mode, "run": run, "kind": g.kind, "text": g.text,
                                           "claimed_source": g.claimed_source, "score": g.score,
                                           "best_source": g.best_source, "best_sentence": g.best_sentence})
                print(f"  {name}: attempts {result.attempts}, fallback {result.fallback_used}, "
                      f"quiz {len(pack.quiz)}, {result.timings.get('total_s', 0):.1f}s")
    write_csv("generation.csv", rows)
    write_csv("grounding.csv", grounding_rows)


# ------------------------------------------------------------------ stress tests
SNR_LEVELS = [20, 10, 5, 0]
DEGRADATIONS = [("original", 1.0, 0.0), ("scale 0.5", 0.5, 0.0), ("scale 0.35 + blur", 0.35, 0.8),
                ("scale 0.25 + blur", 0.25, 1.2)]


def run_stress():
    """Push the speech and OCR stages until they fail, because the main test set turned out to be too easy."""
    import numpy as np
    from PIL import ImageFilter
    from whisper.normalizers import EnglishTextNormalizer

    sys.path.insert(0, str(ROOT / "evaluation"))
    from build_testset import add_noise, save_mp3, synthesise

    normalise = EnglishTextNormalizer()
    stress_dir = TESTSET / "stress"
    stress_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(7)

    rows = []
    for b in BUNDLES:
        clean = synthesise(b)
        reference = normalise(b["audio_script"])
        for snr in SNR_LEVELS:
            path = stress_dir / f"{b['id']}_snr{snr}.mp3"
            save_mp3(add_noise(clean, snr, rng), path)
            for model in WHISPER_MODELS:
                result = speech.transcribe(str(path), model)
                wer = jiwer.wer(reference, normalise(result.text))
                rows.append({"bundle": b["id"], "accent": b["audio"]["accent"], "snr_db": snr, "model": model,
                             "wer": round(wer, 4), "seconds": round(result.seconds, 2),
                             "flagged_segments": len(result.flagged_segments), "hypothesis": result.text})
            print(f"  {b['id']} SNR {snr:>2} dB: " + ", ".join(f"{r['model']} {r['wer']:.3f}" for r in rows[-3:]))
    write_csv("stress_asr.csv", rows)

    real = json.loads((TESTSET / "real" / "cpu_diagram.json").read_text())
    images = [(b["id"], b["slide"]["style"], Image.open(slide_path(b["id"])),
               (TESTSET / b["id"] / "reference_slide.txt").read_text(), b["key_terms"]) for b in BUNDLES]
    if (ROOT / real["image"]).exists():  # the real diagram stays local (samples/ is not in the repository)
        images.append(("R1", "real diagram", Image.open(ROOT / real["image"]), real["reference"], real["key_terms"]))
    else:
        print(f"  skipping the real diagram: {real['image']} not found")
    rows = []
    for slide_id, style, image, reference, terms in images:
        for label, scale, blur in DEGRADATIONS:
            degraded = image.convert("RGB")
            if scale < 1:
                degraded = degraded.resize((max(1, int(degraded.width * scale)), max(1, int(degraded.height * scale))),
                                           Image.BILINEAR)
            if blur:
                degraded = degraded.filter(ImageFilter.GaussianBlur(blur))
            for binarise in (True, False):
                ocr = vision.ocr_image(degraded, binarise=binarise)
                recall, _ = key_term_recall(ocr.text, terms)
                rows.append({"slide": slide_id, "style": style, "degradation": label, "width_px": degraded.width,
                             "binarised": binarise, "key_term_recall": round(recall, 3),
                             "cer": round(jiwer.cer(_norm(reference), _norm(ocr.text)), 4),
                             "mean_confidence": ocr.mean_confidence, "seconds": round(ocr.seconds, 2), "text": ocr.text})
            print(f"  {slide_id} {label:18} recall bin {rows[-2]['key_term_recall']:.2f} / raw {rows[-1]['key_term_recall']:.2f}")
    write_csv("stress_ocr.csv", rows)


def run_timing():
    """Re-time the two checks with the embedding model already loaded (the first call includes loading it)."""
    from revisionflow.schema import RevisionPack

    embedder = Embedder()
    embedder.encode(["warm-up sentence"])
    rows = []
    for path in sorted((RESULTS / "packs").glob(f"*_{config.GENERATION_MODE}_run*.json")):
        data = json.loads(path.read_text())
        sources, pack = data["sources"], RevisionPack.model_validate(data["pack"])
        start = time.perf_counter()
        relevance_check(sources, embedder=embedder)
        relevance_s = time.perf_counter() - start
        start = time.perf_counter()
        grounding_check(pack, sources, embedder=embedder)
        rows.append({"bundle": data["bundle"], "run": data["run"], "relevance_s": round(relevance_s, 3),
                     "grounding_s": round(time.perf_counter() - start, 3)})
    write_csv("timing_checks.csv", rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["asr", "image", "relevance", "generation", "stress", "timing", "all"])
    stage = parser.parse_args().stage
    stages = {"asr": run_asr, "image": run_image, "relevance": run_relevance, "generation": run_generation,
              "stress": run_stress, "timing": run_timing}
    for name, function in stages.items():
        if stage in (name, "all"):
            print(f"== {name}")
            start = time.perf_counter()
            function()
            print(f"== {name} finished in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
