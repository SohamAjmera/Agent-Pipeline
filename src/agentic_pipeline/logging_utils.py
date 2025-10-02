from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table


console = Console()


def utc_ts() -> str:
    return datetime.utcnow().isoformat() + "Z"


@dataclass
class TraceStep:
    kind: str
    detail: Dict[str, Any]
    at: str = field(default_factory=utc_ts)


@dataclass
class Trace:
    query: str
    steps: List[TraceStep] = field(default_factory=list)
    started_at: str = field(default_factory=utc_ts)
    finished_at: Optional[str] = None

    def add(self, kind: str, detail: Dict[str, Any]) -> None:
        self.steps.append(TraceStep(kind=kind, detail=detail))

    def finish(self) -> None:
        self.finished_at = utc_ts()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "steps": [
                {"kind": s.kind, "detail": s.detail, "at": s.at} for s in self.steps
            ],
        }

    def save_json(self, out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return out_path

    def print_summary(self) -> None:
        table = Table(title="Trace Summary")
        table.add_column("When")
        table.add_column("Step")
        table.add_column("Detail")
        for s in self.steps:
            table.add_row(s.at, s.kind, json.dumps(s.detail)[:120])
        console.print(table)


