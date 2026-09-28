"""Analyse the user study (Evaluation chapter, section 5.4).

Reads evaluation/user_study/responses.csv and writes:
  evaluation/results/user_study_summary.md
  evaluation/figures/sus_scores.png

SUS scoring follows Brooke (1996): odd items contribute (score - 1), even items (5 - score),
and the sum is multiplied by 2.5. The mean is compared with the average of 68 (Lewis & Sauro, 2018)
and reported with a 95% confidence interval, because the sample is small (Tullis & Stetson, 2004).
Likert items are single ordinal items, so medians and distributions are reported, not means
(Sullivan & Artino, 2013).
"""

import csv
import statistics
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
FIGURES = HERE.parent / "figures"
LIKERT = {
    "l11_useful": "Flashcards and quiz would be useful",
    "l12_trust": "Trusted the generated material",
    "l13_edit_helped": "Editing intermediate text helped checking",
    "l14_warnings_clear": "Warnings were clear",
    "l15_would_use": "Would use it for a real exam",
}


def sus_score(row) -> float:
    total = 0
    for i in range(1, 11):
        value = int(row[f"sus{i}"])
        total += (value - 1) if i % 2 == 1 else (5 - value)
    return total * 2.5


def main():
    with open(HERE / "responses.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["pid"].strip()]
    if not rows:
        print("responses.csv has no participants yet.")
        return

    scores = [sus_score(r) for r in rows]
    n = len(scores)
    mean = statistics.mean(scores)
    sd = statistics.stdev(scores) if n > 1 else 0.0
    half_width = stats.t.ppf(0.975, n - 1) * sd / n ** 0.5 if n > 1 else 0.0

    lines = ["# User study results", "", f"Participants: {n}", "",
             "## System Usability Scale", "",
             "| Participant | SUS |", "|---|---|"]
    lines += [f"| {r['pid']} | {s:.1f} |" for r, s in zip(rows, scores)]
    lines += ["", f"Mean {mean:.1f}, SD {sd:.1f}, 95% CI {mean - half_width:.1f} to {mean + half_width:.1f} "
              f"(benchmark average 68).", ""]

    lines += ["## Task completion", "", "| Task | Completed without help | With a hint | Not completed | Median time (s) |",
              "|---|---|---|---|---|"]
    for t in (1, 2, 3):
        done = [float(r[f"t{t}_completed"]) for r in rows]
        times = [float(r[f"t{t}_seconds"]) for r in rows if r[f"t{t}_seconds"]]
        lines.append(f"| Task {t} | {done.count(1.0)} | {done.count(0.5)} | {done.count(0.0)} | "
                     f"{statistics.median(times) if times else '-'} |")

    lines += ["", "## RevisionFlow items (1 = strongly disagree, 5 = strongly agree)", "",
              "| Item | Median | 1 | 2 | 3 | 4 | 5 |", "|---|---|---|---|---|---|---|"]
    for key, label in LIKERT.items():
        values = [int(r[key]) for r in rows if r[key]]
        counts = Counter(values)
        lines.append(f"| {label} | {statistics.median(values)} | " + " | ".join(str(counts.get(v, 0)) for v in range(1, 6)) + " |")

    lines += ["", "## Open comments (to be grouped into themes by hand)", ""]
    for r in rows:
        lines.append(f"- {r['pid']} helped: {r['o16_helped']}")
        lines.append(f"- {r['pid']} confusing: {r['o17_confusing']}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "user_study_summary.md").write_text("\n".join(lines) + "\n")

    FIGURES.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar([r["pid"] for r in rows], scores, color="#4C72B0")
    ax.axhline(68, color="#C44E52", linestyle="--", linewidth=1.2, label="Average (68)")
    ax.axhline(mean, color="#333333", linewidth=1.2, label=f"Mean ({mean:.1f})")
    ax.set_ylim(0, 100)
    ax.set_ylabel("SUS score")
    ax.set_title("System Usability Scale by participant")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "sus_scores.png", dpi=200)
    print((RESULTS / "user_study_summary.md").read_text())


if __name__ == "__main__":
    main()
