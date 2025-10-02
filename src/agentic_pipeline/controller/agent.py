from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config import Config
from ..logging_utils import Trace, console
from ..retriever.loader import load_kb_from_dir
from ..retriever.retriever import Retriever
from ..reasoner.reasoner import Reasoner
from ..tools.csv_price_tool import CSVPriceTool


@dataclass
class AgentController:
    config: Config

    def __post_init__(self) -> None:
        # Initialize components
        self.retriever = Retriever(
            embedding_model=self.config.embedding_model,
            openai_api_key=self.config.openai_api_key,
        )
        self.reasoner = Reasoner(
            prompt_version=self.config.prompt_version,
            llm_model=self.config.llm_model,
            openai_api_key=self.config.openai_api_key,
        )
        self.csv_tool = CSVPriceTool(self.config.prices_csv)
        # Load KB docs
        docs = load_kb_from_dir(self.config.kb_dir)
        self.retriever.index(docs)

    def run(self, query: str, save_trace: bool = True) -> Tuple[str, Dict, Optional[Path]]:
        trace = Trace(query=query)

        # Retrieve
        retrieved = self.retriever.search(query, k=4)
        retrieved_dicts = [
            {"doc_id": r.doc_id, "text": r.text, "score": r.score} for r in retrieved
        ]
        trace.add("retrieval", {"results": retrieved_dicts})

        # Decide tool
        decision = self.reasoner.decide_tool(query, retrieved_dicts)
        trace.add("reasoning_tool_decision", {"decision": decision.decision, "rationale": decision.rationale})

        # Maybe call tool
        tool_payload: Optional[Dict] = None
        if decision.decision == "use_tool":
            result = self.csv_tool.lookup(query)
            if result:
                tool_payload = {
                    "product_name": result.product_name,
                    "sku": result.sku,
                    "price_usd": result.price_usd,
                    "match_score": result.score,
                    "latency_ms": result.latency_ms,
                }
            trace.add("tool_call_csv_price", tool_payload or {"result": None})

        # Synthesize
        final_answer = self.reasoner.synthesize(query, retrieved_dicts, tool_payload)
        trace.add("final_answer", {"text": final_answer})
        trace.finish()

        out_path: Optional[Path] = None
        if save_trace:
            safe = "".join(c for c in query if c.isalnum() or c in (" ", "-", "_"))[:50].strip().replace(" ", "_")
            out_path = self.config.results_dir / f"trace_{safe or 'query'}.json"
            trace.save_json(out_path)

        # Pretty-print brief summary
        console.print(f"Decision: {decision.decision} ({decision.rationale})")
        if tool_payload:
            console.print(f"Tool: {tool_payload['product_name']} ${tool_payload['price_usd']} in {tool_payload['latency_ms']:.1f}ms")

        return final_answer, trace.to_dict(), out_path


