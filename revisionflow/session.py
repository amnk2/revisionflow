"""Saving and loading revision sessions, and exporting flashcards.

Only text is saved (the edited sources and the generated pack). Uploaded audio
and images are never stored.
"""

import csv
import io
import json
from datetime import datetime

from . import __version__
from .schema import RevisionPack


def session_to_json(topic: str, sources: dict[str, str], pack: RevisionPack = None) -> str:
    data = {
        "app": "RevisionFlow",
        "version": __version__,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "topic": topic,
        "sources": sources,
        "pack": pack.model_dump() if pack else None,
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


def session_from_json(text: str) -> tuple[str, dict, RevisionPack]:
    data = json.loads(text)
    if data.get("app") != "RevisionFlow":
        raise ValueError("This file is not a RevisionFlow session.")
    pack = RevisionPack.model_validate(data["pack"]) if data.get("pack") else None
    return data.get("topic", ""), data.get("sources", {}), pack


def flashcards_csv(pack: RevisionPack) -> str:
    """Two-column CSV (front, back) that flashcard apps such as Anki can import."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    for card in pack.flashcards:
        writer.writerow([card.front, card.back])
    return buffer.getvalue()
