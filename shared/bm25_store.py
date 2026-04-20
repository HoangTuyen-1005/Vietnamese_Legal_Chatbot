from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

from pyvi import ViTokenizer
from rank_bm25 import BM25Okapi


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class BM25Store:
    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: BM25Okapi | None = None

    def build(self, chunks: Sequence[Dict[str, Any]]) -> None:
        self.documents = []

        for raw in chunks:
            content = normalize_whitespace(str(raw.get("content", "")))
            metadata = raw.get("metadata", {}) or {}
            if not content:
                continue

            self.documents.append({
                "chunk_id": raw.get("chunk_id"),
                "content": content,
                "metadata": metadata,
            })

        self.tokenized_corpus = [
            ViTokenizer.tokenize(doc["content"].lower()).split()
            for doc in self.documents
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def save(self, index_path: str, docs_path: str) -> None:
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)

        with open(index_path, "wb") as f:
            pickle.dump(self.bm25, f)

        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)

    def load(self, index_path: str, docs_path: str | None = None) -> None:
        with open(index_path, "rb") as f:
            blob = pickle.load(f)

        if isinstance(blob, dict) and "bm25_model" in blob:
            self.bm25 = blob["bm25_model"]
            corpus = blob.get("corpus", [])
            metadata = blob.get("metadata", [])
            self.documents = [
                {
                    "chunk_id": meta.get("chunk_id", f"bm25_{i}"),
                    "content": corpus[i] if i < len(corpus) else "",
                    "metadata": meta,
                }
                for i, meta in enumerate(metadata)
            ]
            return

        self.bm25 = blob
        if docs_path is None:
            raise ValueError("docs_path is required for new BM25 format")

        with open(docs_path, "rb") as f:
            self.documents = pickle.load(f)

    def search(self, query: str, top_k: int = 10) -> List[dict]:
        if self.bm25 is None:
            raise RuntimeError("BM25 index has not been built or loaded.")

        tokenized_query = ViTokenizer.tokenize(query.lower()).split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_indices = bm25_scores.argsort()[-top_k:][::-1]

        results: List[dict] = []
        for idx in top_indices:
            doc = self.documents[idx]
            results.append({
                "chunk_id": doc.get("chunk_id"),
                "content": doc.get("content", ""),
                "metadata": doc.get("metadata", {}),
                "score": float(bm25_scores[idx]),
                "source": "bm25",
            })

        return results