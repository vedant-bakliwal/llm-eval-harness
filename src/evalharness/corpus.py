"""Load the handbook and split it into citable chunks (one per '## ' section)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    id: str      # "<file stem>#<section slug>", e.g. "pto#carryover"
    title: str
    text: str


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def load_corpus(folder: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(folder).glob("*.md")):
        title, body = None, []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                if title:
                    chunks.append(_chunk(path.stem, title, body))
                title, body = line[3:].strip(), []
            elif title and line.strip():
                body.append(line.strip())
        if title:
            chunks.append(_chunk(path.stem, title, body))
    if not chunks:
        raise ValueError(f"No '## ' sections found under {folder}")
    return chunks


def _chunk(stem: str, title: str, body: list[str]) -> Chunk:
    return Chunk(id=f"{stem}#{slugify(title)}", title=title, text=" ".join(body))
