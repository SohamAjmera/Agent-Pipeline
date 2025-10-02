from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

from .prompts import Prompts


@dataclass
class ToolDecision:
    decision: str  # "kb_only" | "use_tool"
    rationale: str


class Reasoner:
    def __init__(self, prompt_version: str, llm_model: str, openai_api_key: Optional[str]) -> None:
        self.prompts = Prompts(version=prompt_version)
        self.llm_model = llm_model
        self.openai_api_key = openai_api_key
        self._client = OpenAI(api_key=openai_api_key) if (openai_api_key and OpenAI) else None

    def _chat(self, system: str, user: str) -> str:
        if not self._client:
            # Heuristic fallback: if price-related keywords appear, suggest tool
            if any(k in user.lower() for k in ["price", "cost", "how much", "pricing"]):
                return json.dumps({"decision": "use_tool", "rationale": "price-related"})
            return json.dumps({"decision": "kb_only", "rationale": "not price-related"})
        resp = self._client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
        )
        return resp.choices[0].message.content or "{}"

    def decide_tool(self, query: str, retrieved: List[Dict[str, str]]) -> ToolDecision:
        system = self.prompts.tool_decision()
        snippets = "\n\n".join([f"[{r['doc_id']}] {r['text'][:400]}" for r in retrieved])
        user = f"Query: {query}\n\nKB snippets:\n{snippets}"
        raw = self._chat(system, user)
        try:
            data = json.loads(raw)
            decision = data.get("decision", "kb_only")
            rationale = data.get("rationale", "")
        except Exception:
            decision, rationale = "kb_only", "parse_error"
        return ToolDecision(decision=decision, rationale=rationale)

    def synthesize(self, query: str, retrieved: List[Dict[str, str]], tool_result: Optional[Dict] = None) -> str:
        if not self._client:
            # Simple heuristic synthesis
            base = retrieved[0]['text'][:300] if retrieved else ""
            if tool_result:
                name = tool_result.get("product_name")
                price = tool_result.get("price_usd")
                return f"{name} costs ${price}. {base}"
            return base or "I couldn't find sufficient information."

        system = self.prompts.final_answer()
        snippets = "\n\n".join([f"[{r['doc_id']}] {r['text'][:500]}" for r in retrieved])
        tool_str = json.dumps(tool_result) if tool_result else "(no tool call)"
        user = (
            f"Query: {query}\n\nRetrieved context:\n{snippets}\n\nTool result: {tool_str}\n"
        )
        resp = self._client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""


