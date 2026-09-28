"""Turn the evaluation CSVs into the tables and charts used in the report.

Writes evaluation/results/summary.md and PNG charts to evaluation/figures/.
Run after run_technical.py (and after the ratings are complete for the faithfulness section).
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
sys.path.insert(0, str(ROOT.parent))
from revisionflow import config  # noqa: E402

plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREEN, RED, GREY = "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8C8C8C"
lines = []
# The generation runs used base.en transcripts (small.en became the default afterwards),
# so end-to-end timing uses the base.en speech times to match them.
GENERATION_WHISPER_MODEL = "base.en"


def md_table(df, floatfmt=3):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype.kind == "f":
            df[col] = df[col].round(floatfmt)
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=220)
    plt.close(fig)


# ------------------------------------------------------------------ speech-to-text
def speech_section():
    asr = pd.read_csv(RESULTS / "asr.csv")
    asr["rtf"] = asr.seconds / asr.duration_s
    table = asr.groupby("model").agg(mean_wer=("wer", "mean"), max_wer=("wer", "max"),
                                     clean_wer=("wer", lambda s: s[asr.loc[s.index, "condition"] == "clean"].mean()),
                                     noisy_wer=("wer", lambda s: s[asr.loc[s.index, "condition"] == "noisy"].mean()),
                                     mean_seconds=("seconds", "mean"), real_time_factor=("rtf", "mean")).reset_index()
    lines.extend(["## Speech-to-text (test set, 8 clips, 33-47 s each)", "", md_table(table), ""])

    stress = pd.read_csv(RESULTS / "stress_asr.csv")
    pivot = stress.pivot_table(index="snr_db", columns="model", values="wer", aggfunc="mean").sort_index(ascending=False)
    lines.extend(["### Noise stress test: mean WER by signal-to-noise ratio", "",
                  md_table(pivot.reset_index()), "",
                  f"Segments flagged as probably silent across all stress runs: {int(stress.flagged_segments.sum())}", ""])
    accent = stress[stress.snr_db == 0].groupby("accent").wer.mean().sort_values().reset_index()
    lines.extend(["### WER at 0 dB by accent (mean of three models)", "", md_table(accent), ""])

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for model, colour in zip(["base", "base.en", "small.en"], [GREY, BLUE, GREEN]):
        series = pivot[model]
        ax.plot(series.index, series.values * 100, marker="o", color=colour, label=model)
    ax.axhline(15, color=RED, linestyle="--", linewidth=1, label="Target (15%)")
    ax.set_xlabel("Signal-to-noise ratio (dB), lower is noisier")
    ax.set_ylabel("Mean word error rate (%)")
    ax.invert_xaxis()
    ax.set_xticks([20, 10, 5, 0])
    ax.legend(frameon=False, fontsize=8)
    save(fig, "asr_noise.png")


# ------------------------------------------------------------------ image
def image_section():
    image = pd.read_csv(RESULTS / "image.csv")
    table = image.groupby("method").agg(mean_key_term_recall=("key_term_recall", "mean"),
                                        min_key_term_recall=("key_term_recall", "min")).reset_index()
    cer = image[image.method.str.startswith("ocr_")].groupby("method").cer.mean()
    table["mean_cer"] = table.method.map(cer)
    lines.extend(["## Image stage (test set, 8 slides)", "", md_table(table), ""])

    stress = pd.read_csv(RESULTS / "stress_ocr.csv")
    synthetic = stress[stress.slide != "R1"]
    order = ["original", "scale 0.5", "scale 0.35 + blur", "scale 0.25 + blur"]
    pivot = synthetic.pivot_table(index="degradation", columns="binarised", values="key_term_recall",
                                  aggfunc="mean").reindex(order)
    pivot.columns = ["without own binarisation", "with own binarisation"]
    lines.extend(["### OCR degradation stress test: mean key-term recall (8 synthetic slides)", "",
                  md_table(pivot.reset_index()), ""])
    real = stress[(stress.slide == "R1") & (~stress.binarised)][["degradation", "width_px", "key_term_recall", "cer",
                                                                 "mean_confidence"]]
    lines.extend(["### Real diagram (preliminary-report screenshot), no binarisation", "", md_table(real), ""])

    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    x = np.arange(len(order))
    ax.bar(x - 0.2, pivot.iloc[:, 0] * 100, 0.4, color=BLUE, label="Tesseract (final setting)")
    ax.bar(x + 0.2, pivot.iloc[:, 1] * 100, 0.4, color=ORANGE, label="Tesseract + own binarisation")
    for i, (plain, binarised) in enumerate(zip(pivot.iloc[:, 0], pivot.iloc[:, 1])):
        for offset, value in ((-0.2, plain), (0.2, binarised)):
            if value == 0:  # an empty bar would otherwise be invisible
                ax.text(i + offset, 2, "0%", ha="center", va="bottom", fontsize=7.5)
    blip = image[image.method == "blip_caption"].key_term_recall.mean() * 100
    ax.axhline(blip, color=RED, linestyle=":", linewidth=1.2, label=f"BLIP caption, original ({blip:.1f}%)")
    ax.set_xticks(x, ["original\n(1280 px)", "50%\n(640 px)", "35% + blur\n(448 px)", "25% + blur\n(320 px)"])
    ax.set_ylabel("Key-term recall (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    save(fig, "ocr_degradation.png")


# ------------------------------------------------------------------ relevance
def relevance_section():
    rel = pd.read_csv(RESULTS / "relevance.csv")
    sweep = pd.read_csv(RESULTS / "relevance_sweep.csv")
    summary = rel.pivot_table(index="case", columns="method", values="min_score", aggfunc=["min", "mean", "max"])
    summary.columns = [f"{a} {b}" for a, b in summary.columns]
    lines.extend(["## Relevance check (8 matched, 8 mismatched bundles)", "", "Lowest pairwise similarity per bundle:", "",
                  md_table(summary.reset_index()), ""])
    rows = []
    for method, threshold in (("embedding", config.RELEVANCE_THRESHOLD), ("tfidf", config.RELEVANCE_THRESHOLD)):
        d = sweep[(sweep.method == method) & (np.isclose(sweep.threshold, threshold))].iloc[0]
        best = sweep[sweep.method == method].accuracy.max()
        rows.append({"method": method, "threshold": threshold, "accuracy": d.accuracy, "precision": d.precision,
                     "recall": d.recall, "best accuracy at any threshold": best})
    lines.extend([md_table(pd.DataFrame(rows)), ""])
    near = RESULTS / "relevance_near.csv"
    if near.exists():
        lines.extend(["### Same-subject mismatch (both computer science)", "", md_table(pd.read_csv(near)), ""])

    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.8), sharey=False)
    for ax, method, title in zip(axes, ["embedding", "tfidf"], ["Sentence embeddings (MiniLM)", "TF-IDF baseline"]):
        for i, (case, colour) in enumerate([("matched", GREEN), ("mismatched", RED)]):
            values = rel[(rel.method == method) & (rel.case == case)].min_score
            jitter = np.random.default_rng(i).uniform(-0.08, 0.08, len(values))
            ax.scatter(np.full(len(values), i) + jitter, values, color=colour, s=22)
        ax.axhline(config.RELEVANCE_THRESHOLD, color=GREY, linestyle="--", linewidth=1)
        ax.set_xticks([0, 1], ["matched", "mismatched"])
        ax.set_title(title, fontsize=9)
        ax.set_xlim(-0.5, 1.5)
    axes[0].set_ylabel("Lowest similarity between inputs")
    save(fig, "relevance.png")


# ------------------------------------------------------------------ generation
def generation_section():
    gen = pd.read_csv(RESULTS / "generation.csv")
    table = gen.groupby("mode").agg(runs=("run", "count"), valid_first_try=("valid_first_try", "mean"),
                                    fallback=("fallback_used", "mean"), mean_repairs=("repairs", "mean"),
                                    key_concepts=("key_concepts", "mean"), flashcards=("flashcards", "mean"),
                                    quiz=("quiz", "mean"), quiz_mcq=("quiz_mcq", "mean"),
                                    generation_s=("generation_s", "mean"), generation_sd=("generation_s", "std"),
                                    flagged_weak=("flagged_weak", "mean")).reset_index()
    lines.extend(["## Generation (8 bundles x 2 modes x 3 runs)", "", md_table(table, 2), ""])

    identical = []
    packs = RESULTS / "packs"
    for (bundle, mode), _ in gen.groupby(["bundle", "mode"]):
        texts = [json.dumps(json.loads((packs / f"{bundle}_{mode}_run{r}.json").read_text())["pack"], sort_keys=True)
                 for r in (1, 2, 3)]
        identical.append({"bundle": bundle, "mode": mode, "identical_runs": len(set(texts)) == 1})
    ident = pd.DataFrame(identical)
    lines.extend([f"Runs identical across all 3 repeats: {ident.identical_runs.sum()} of {len(ident)} bundle/mode pairs "
                  "(temperature 0, seed 42).", ""])

    asr = pd.read_csv(RESULTS / "asr.csv")
    image = pd.read_csv(RESULTS / "image.csv")
    checks = pd.read_csv(RESULTS / "timing_checks.csv")  # re-timed with the embedding model already loaded
    run1 = gen[(gen["mode"] == config.GENERATION_MODE)]
    per_bundle = []
    for bundle, g in run1.groupby("bundle"):
        speech_s = asr[(asr.bundle == bundle) & (asr.model == GENERATION_WHISPER_MODEL)].seconds.iloc[0]
        ocr_s = image[(image.bundle == bundle) & (image.method == f"ocr_binarised={config.OCR_BINARISE}")].seconds.iloc[0]
        caption_s = image[(image.bundle == bundle) & (image.method == "blip_caption")].seconds.iloc[0]
        for _, row in g.iterrows():
            timed = checks[(checks.bundle == bundle) & (checks.run == row.run)].iloc[0]
            per_bundle.append({"bundle": bundle, "run": row.run, "speech": speech_s, "OCR": ocr_s, "caption": caption_s,
                               "relevance": timed.relevance_s, "generation": row.generation_s,
                               "grounding": timed.grounding_s})
    timing = pd.DataFrame(per_bundle)
    timing["total"] = timing[["speech", "OCR", "caption", "relevance", "generation", "grounding"]].sum(axis=1)
    stages = timing[["speech", "OCR", "caption", "relevance", "generation", "grounding", "total"]].agg(["mean", "std", "max"]).T
    lines.extend(["## End-to-end time per bundle (models loaded, default settings, 24 runs)", "",
                  md_table(stages.reset_index().rename(columns={"index": "stage"}), 2), ""])

    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    means = timing[["speech", "OCR", "caption", "relevance", "generation", "grounding"]].mean()
    left = 0
    for (stage, value), colour in zip(means.items(), [BLUE, GREEN, "#8172B3", GREY, ORANGE, RED]):
        shown = "<0.1" if value < 0.05 else f"{value:.1f}"
        ax.barh([0], [value], left=left, color=colour, label=f"{stage} ({shown}s)")
        left += value
    ax.axvline(60, color=RED, linestyle="--", linewidth=1)
    ax.text(60, 0.45, "target 60 s", color=RED, fontsize=8, ha="right")
    ax.set_yticks([])
    ax.set_xlabel("Mean seconds per bundle")
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.35))
    save(fig, "timing.png")
    return timing


# ------------------------------------------------------------------ ratings
def ratings_section():
    ratings_dir = ROOT / "ratings"
    items_path, key_path = ratings_dir / "items_marked.csv", ratings_dir / "key.csv"
    if not items_path.exists():
        lines.extend(["## Faithfulness and question quality", "", "Not marked yet.", ""])
        return
    items = pd.read_csv(items_path).merge(pd.read_csv(key_path), on=["id", "bundle"])
    faith = items.groupby(["mode", "kind"]).faithfulness.value_counts(normalize=True).unstack(fill_value=0)
    faith = faith.reindex(columns=["S", "P", "U", "E"], fill_value=0)
    lines.extend(["## Faithfulness (share of items)", "", md_table(faith.reset_index(), 3), ""])
    overall = items.groupby("mode").faithfulness.value_counts(normalize=True).unstack(fill_value=0).reindex(
        columns=["S", "P", "U", "E"], fill_value=0)
    counts = items.groupby("mode").faithfulness.value_counts().unstack(fill_value=0).reindex(
        columns=["S", "P", "U", "E"], fill_value=0)
    counts["items"] = counts.sum(axis=1)
    lines.extend(["Overall share:", "", md_table(overall.reset_index(), 3), "", "Counts:", "", md_table(counts.reset_index()), ""])
    comments = items.comment.fillna("")
    tags = pd.DataFrame({"mode": items["mode"], "asr_error": comments.str.contains("asr-error"),
                         "extrinsic_correct": comments.str.contains("extrinsic but correct"),
                         "placeholder": items.faithfulness.eq("E")})
    lines.extend(["Items carrying a transcription error, extrinsic-but-correct additions, placeholders:", "",
                  md_table(tags.groupby("mode")[["asr_error", "extrinsic_correct", "placeholder"]].sum().reset_index()), ""])
    by_bundle = items.assign(supported=items.faithfulness.eq("S")).groupby(["bundle", "mode"]).supported.mean().unstack()
    lines.extend(["Share supported by bundle:", "", md_table(by_bundle.reset_index(), 2), ""])

    quiz = items[items.kind == "quiz question"]
    qq = quiz.groupby("mode")[["answerable", "correct", "key_point"]].mean()
    flaws = quiz.assign(flawed=quiz.flaws.fillna("").str.len() > 0).groupby("mode").flawed.mean()
    qq["share_with_flaw"] = flaws
    lines.extend(["## Quiz question quality (0-2 per criterion)", "", md_table(qq.reset_index(), 2), ""])
    cards = items[items.kind == "flashcard"]
    lines.extend(["## Flashcard usefulness", "",
                  md_table(cards.groupby("mode").card_flag.value_counts(normalize=True).unstack(fill_value=0).reset_index(), 3), ""])

    summaries_path = ratings_dir / "summaries_marked.csv"
    if summaries_path.exists():
        summaries = pd.read_csv(summaries_path).merge(pd.read_csv(key_path), on=["id", "bundle"])
        summaries["coverage"] = summaries[[f"kp{i}" for i in range(1, 6)]].sum(axis=1) / 5
        lines.extend(["## Summary key-point coverage", "",
                      md_table(summaries.groupby("mode").coverage.agg(["mean", "min", "max"]).reset_index(), 2), ""])

    # Grounding check against the manual faithfulness labels.
    grounding = pd.read_csv(RESULTS / "grounding.csv")
    grounding = grounding[grounding.run == 1]
    marked = items.copy()
    marked["kind_index"] = marked.ref.str.split().str[1].astype(int)
    grounding["kind_index"] = grounding.groupby(["bundle", "mode", "kind"]).cumcount()
    merged = marked.merge(grounding, on=["bundle", "mode", "kind", "kind_index"], suffixes=("", "_g"))
    merged["unsupported"] = merged.faithfulness != "S"
    rows = []
    for threshold in np.arange(0.30, 0.86, 0.05):
        flagged = merged.score < threshold
        tp = int((flagged & merged.unsupported).sum())
        fp = int((flagged & ~merged.unsupported).sum())
        fn = int((~flagged & merged.unsupported).sum())
        rows.append({"threshold": round(threshold, 2), "flagged": int(flagged.sum()), "precision": tp / (tp + fp) if tp + fp else 0,
                     "recall": tp / (tp + fn) if tp + fn else 0})
    sweep = pd.DataFrame(rows)
    lines.extend([f"## Grounding check against manual labels ({len(merged)} items; "
                  f"{int(merged.unsupported.sum())} marked P, U or E)", "", md_table(sweep, 3), ""])
    by_label = merged.groupby("faithfulness").score.describe()[["count", "mean", "min", "max"]]
    lines.extend(["Similarity score by manual label:", "", md_table(by_label.reset_index(), 3), ""])

    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    for label, colour in (("S", GREEN), ("P", ORANGE), ("U", RED), ("E", GREY)):
        values = merged[merged.faithfulness == label].score
        if len(values):
            ax.hist(values, bins=np.arange(0.2, 1.01, 0.05), alpha=0.7, color=colour, label=f"{label} (n={len(values)})")
    ax.axvline(config.GROUNDING_THRESHOLD, color=GREY, linestyle="--", linewidth=1, label="threshold")
    ax.set_xlabel("Grounding similarity (closest source sentence)")
    ax.set_ylabel("Items")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "grounding_scores.png")


def main():
    lines.extend(["# Technical evaluation summary", ""])
    speech_section()
    image_section()
    relevance_section()
    if (RESULTS / "generation.csv").exists():
        generation_section()
        ratings_section()
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")
    print((RESULTS / "summary.md").read_text())


if __name__ == "__main__":
    main()
