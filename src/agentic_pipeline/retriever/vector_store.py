from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


@dataclass
class VectorDoc:
    doc_id: str
    text: str


class VectorStore:
    def __init__(self, embedding_model: str, openai_api_key: Optional[str]) -> None:
        self.embedding_model = embedding_model
        self.openai_api_key = openai_api_key
        self._use_openai = bool(openai_api_key)

        self._vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix: Optional[np.ndarray] = None
        self._embeddings: Optional[np.ndarray] = None
        self._docs: List[VectorDoc] = []

        self._client = OpenAI(api_key=openai_api_key) if (self._use_openai and OpenAI) else None

    def _embed_texts_openai(self, texts: List[str]) -> np.ndarray:
        assert self._client is not None
        response = self._client.embeddings.create(model=self.embedding_model, input=texts)
        vectors = [item.embedding for item in response.data]
        return np.array(vectors, dtype=np.float32)

    def _embed_texts_tfidf(self, texts: List[str]) -> np.ndarray:
        # Use TF-IDF as a vector baseline; cosine similarity works similarly
        if self._vectorizer is None:
            self._vectorizer = TfidfVectorizer(stop_words="english")
            self._tfidf_matrix = self._vectorizer.fit_transform(texts)
            return self._tfidf_matrix.toarray().astype(np.float32)
        else:
            tfidf = self._vectorizer.transform(texts)
            return tfidf.toarray().astype(np.float32)

    def add(self, docs: List[VectorDoc]) -> None:
        self._docs = list(docs)
        texts = [d.text for d in docs]
        if self._use_openai and self._client is not None:
            self._embeddings = self._embed_texts_openai(texts)
        else:
            self._embeddings = self._embed_texts_tfidf(texts)

    def query(self, text: str, k: int = 4) -> List[Tuple[VectorDoc, float]]:
        if self._embeddings is None:
            return []
        if self._use_openai and self._client is not None:
            q_vec = self._embed_texts_openai([text])[0]
        else:
            q_vec = self._embed_texts_tfidf([text])[0]
        A = self._embeddings
        q = q_vec
        denom = (np.linalg.norm(A, axis=1) * (np.linalg.norm(q) + 1e-8)) + 1e-8
        sims = (A @ q) / denom
        idx = np.argsort(-sims)[:k]
        return [(self._docs[i], float(sims[i])) for i in idx]


