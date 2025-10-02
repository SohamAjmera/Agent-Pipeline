from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Prompts:
    version: str

    def tool_decision(self) -> str:
        if self.version == "v1":
            return (
                "You are a helpful assistant. Given the user query and the retrieved KB snippets, decide: "
                "(A) answer from KB only, or (B) call the CSV price lookup tool. "
                "Only call the tool if the user is asking for a product price or price-related info.\n"
                "Return JSON with fields: {\"decision\": \"kb_only|use_tool\", \"rationale\": str}."
            )
        return self.tool_decision()

    def final_answer(self) -> str:
        if self.version == "v1":
            return (
                "You are a helpful assistant. Craft a concise, accurate answer based on:\n"
                "- User query\n- Retrieved KB context\n- Optional tool result.\n"
                "Respond clearly. If you used the tool, cite the product and price."
            )
        return self.final_answer()


