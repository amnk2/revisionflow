"""Create blind rating sheets from the first run of each bundle and mode.

Writes to evaluation/ratings/:
  items.csv        one row per key concept, flashcard and quiz question, shuffled, with a random ID and no mode
  summaries.csv    one row per summary, shuffled, with the bundle's key points
  key.csv          which ID belongs to which bundle, mode and item (kept closed while marking)
  second_rater.csv a random 20% sample of items.csv for the second rater (same columns, empty marks)
"""

import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKS = ROOT / "results" / "packs"
OUT = ROOT / "ratings"
BUNDLES = {b["id"]: b for b in json.loads((ROOT / "testset" / "bundles.json").read_text())["bundles"]}
MARK_COLUMNS = ["faithfulness", "answerable", "correct", "key_point", "flaws", "card_flag", "comment"]


def item_rows(bundle_id, mode, pack):
    rows = []
    for i, k in enumerate(pack["key_concepts"]):
        rows.append((mode, f"concept {i}", "key concept", f"{k['concept']}: {k['explanation']}", k["source"]))
    for i, c in enumerate(pack["flashcards"]):
        rows.append((mode, f"card {i}", "flashcard", f"FRONT: {c['front']}\nBACK: {c['back']}", c["source"]))
    for i, q in enumerate(pack["quiz"]):
        options = "\n".join(f"  - {o}" for o in q["options"])
        text = (f"[{q['kind']}] {q['question']}\n{options + chr(10) if options else ''}"
                f"ANSWER: {q['answer']}\nEXPLANATION: {q['explanation']}")
        rows.append((mode, f"quiz {i}", "quiz question", text, q["source"]))
    return rows


def main():
    rng = random.Random(3070)
    OUT.mkdir(exist_ok=True)
    items, summaries, key = [], [], []
    used_ids = set()

    def new_id():
        while True:
            value = f"{rng.randrange(16 ** 4):04x}"
            if value not in used_ids:
                used_ids.add(value)
                return value

    for path in sorted(PACKS.glob("*_run1.json")):
        data = json.loads(path.read_text())
        bundle_id, mode, pack = data["bundle"], data["mode"], data["pack"]
        for mode_, ref, kind, text, source in item_rows(bundle_id, mode, pack):
            rid = new_id()
            items.append({"id": rid, "bundle": bundle_id, "kind": kind, "item": text, "claimed_source": source,
                          **{c: "" for c in MARK_COLUMNS}})
            key.append({"id": rid, "bundle": bundle_id, "mode": mode_, "ref": ref})
        rid = new_id()
        summaries.append({"id": rid, "bundle": bundle_id, "summary": pack["summary"],
                          **{f"kp{i + 1}": "" for i in range(5)}})
        key.append({"id": rid, "bundle": bundle_id, "mode": mode, "ref": "summary"})

    rng.shuffle(items)
    rng.shuffle(summaries)
    # Group by bundle so each bundle's sources only need reading once; order inside a bundle stays random.
    items.sort(key=lambda r: r["bundle"])
    summaries.sort(key=lambda r: r["bundle"])
    for name, rows in (("items.csv", items), ("summaries.csv", summaries), ("key.csv", key)):
        with open(OUT / name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    sample = rng.sample(items, round(len(items) * 0.2))
    with open(OUT / "second_rater.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(sample, key=lambda r: r["bundle"]))
    print(f"{len(items)} items, {len(summaries)} summaries, second-rater sample {len(sample)}")


if __name__ == "__main__":
    main()
