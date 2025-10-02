from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .loader import Document
from .vector_store import VectorDoc, VectorStore


@dataclass
class RetrievedChunk:
    doc_id: str
    text: str
    score: float


class Retriever:
    def __init__(self, embedding_model: str, openai_api_key: str | None) -> None:
        self.store = VectorStore(embedding_model=embedding_model, openai_api_key=openai_api_key)

    def index(self, docs: List[Document]) -> None:
        vec_docs = [VectorDoc(doc_id=d.doc_id, text=d.text) for d in docs]
        self.store.add(vec_docs)

    def search(self, query: str, k: int = 4) -> List[RetrievedChunk]:
        results = self.store.query(query, k=k)
        return [RetrievedChunk(doc_id=d.doc_id, text=d.text, score=score) for d, score in results]


