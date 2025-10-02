from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Tuple

from ..agentic_pipeline.config import Config
from ..agentic_pipeline.controller.agent import AgentController
from ..agentic_pipeline.logging_utils import console


def run_eval(queries: List[str]) -> Tuple[List[dict], Path]:
    config = Config.from_env()
    controller = AgentController(config=config)
    results_dir = config.results_dir
    out_records = []
    for q in queries:
        t0 = time.perf_counter()
        answer, trace, trace_path = controller.run(q, save_trace=True)
        t1 = time.perf_counter()
        tool_lat_ms = None
        for step in trace.get("steps", []):
            if step.get("kind") == "tool_call_csv_price":
                detail = step.get("detail") or {}
                tool_lat_ms = detail.get("latency_ms")
        rec = {
            "query": q,
            "answer": answer,
            "total_latency_ms": (t1 - t0) * 1000,
            "tool_latency_ms": tool_lat_ms,
            "trace_path": str(trace_path) if trace_path else None,
        }
        out_records.append(rec)
    summary_path = results_dir / "eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(out_records, f, ensure_ascii=False, indent=2)
    return out_records, summary_path


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    queries_path = project_root / "data" / "test_queries.json"
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    records, out = run_eval(queries)
    console.rule("Eval Summary")
    for r in records:
        console.print(f"- {r['query']} -> total {r['total_latency_ms']:.1f}ms, tool {r['tool_latency_ms']}")
    console.print(f"Saved summary: {out}")


if __name__ == "__main__":
    main()


