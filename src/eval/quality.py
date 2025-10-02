from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from ..agentic_pipeline.config import Config
from ..agentic_pipeline.logging_utils import console


STOPWORDS = {
    "the","a","an","and","or","is","are","to","of","in","on","for","with","do","does","how","what","when","which","that","it","you","we","i","this","there","be","as"
}


def tokenize(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS]


def jaccard(a_tokens: List[str], b_tokens: List[str]) -> float:
    aset, bset = set(a_tokens), set(b_tokens)
    if not aset or not bset:
        return 0.0
    return len(aset & bset) / len(aset | bset)


def detect_price_intent(query: str) -> bool:
    q = query.lower()
    keywords = ["price", "cost", "how much", "pricing"]
    return any(k in q for k in keywords)


def used_csv_tool(trace: Dict) -> bool:
    for step in trace.get("steps", []):
        if step.get("kind") == "tool_call_csv_price":
            detail = step.get("detail")
            if isinstance(detail, dict) and detail.get("product_name"):
                return True
    return False


def get_retrieved_text(trace: Dict) -> str:
    for step in trace.get("steps", []):
        if step.get("kind") == "retrieval":
            res = step.get("detail", {}).get("results", [])
            snippets = [str(r.get("text", "")) for r in res]
            return "\n\n".join(snippets)
    return ""


def score_record(query: str, answer: str, trace_path: Path) -> Dict:
    trace = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.exists() else {}

    relevance = jaccard(tokenize(query), tokenize(answer))

    should_use_tool = detect_price_intent(query)
    did_use_tool = used_csv_tool(trace)
    # Tool score: 1 if aligned, 0.5 if extra tool used, 0 if missed when needed
    if should_use_tool and did_use_tool:
        tool_score = 1.0
        tool_note = "Used tool appropriately."
    elif should_use_tool and not did_use_tool:
        tool_score = 0.0
        tool_note = "Should have used tool but did not."
    elif (not should_use_tool) and did_use_tool:
        tool_score = 0.5
        tool_note = "Used tool unnecessarily."
    else:
        tool_score = 1.0
        tool_note = "Correctly avoided tool."

    retrieved_text = get_retrieved_text(trace)
    kb_overlap = jaccard(tokenize(answer), tokenize(retrieved_text)) if retrieved_text else 0.0
    # If no tool recommended, we expect KB grounding more
    if not should_use_tool:
        kb_score = kb_overlap
    else:
        kb_score = max(0.2, kb_overlap)  # tolerate lower grounding when price tool is used

    final_score = 0.4 * relevance + 0.3 * tool_score + 0.3 * kb_score

    notes = []
    if relevance < 0.2:
        notes.append("Low query-answer overlap")
    if kb_overlap < 0.2 and not should_use_tool:
        notes.append("Weak KB grounding for non-price query")
    notes.append(tool_note)

    return {
        "query": query,
        "relevance": round(relevance, 3),
        "tool_score": round(tool_score, 3),
        "kb_score": round(kb_score, 3),
        "final_score": round(final_score, 3),
        "notes": "; ".join(notes),
        "trace_path": str(trace_path),
    }


def run_quality() -> Path:
    config = Config.from_env()
    summary_path = config.results_dir / "eval_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing eval summary: {summary_path}")
    records = json.loads(summary_path.read_text(encoding="utf-8"))
    out: List[Dict] = []
    for rec in records:
        q = rec.get("query", "")
        a = rec.get("answer", "")
        tpath = Path(rec.get("trace_path") or "")
        try:
            out.append(score_record(q, a, tpath))
        except Exception as e:
            out.append({"query": q, "error": str(e)})

    out_path = config.results_dir / "eval_quality.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    out = run_quality()
    console.print(f"Saved quality report: {out}")


if __name__ == "__main__":
    main()


