from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class Document:
    doc_id: str
    text: str


def load_kb_from_dir(kb_dir: Path) -> List[Document]:
    docs: List[Document] = []
    for p in sorted(kb_dir.glob("*.txt")):
        text = p.read_text(encoding="utf-8").strip()
        if not text:
            continue
        docs.append(Document(doc_id=p.stem, text=text))
    return docs


