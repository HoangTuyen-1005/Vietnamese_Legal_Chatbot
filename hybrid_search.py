import pickle
import torch
from pathlib import Path
from typing import Any, Dict, Sequence

from pyvi import ViTokenizer
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

class HybridSearch:
    def __init__(
        self,
        embedding_model_name: str = "huyydangg/DEk21_hcmute_embedding",
        collection_name: str = "legal_documents",
        bm25_index_path: str = "data/indexes/bm25_index.pkl",
        qdrant_path: str = "data/indexes/qdrant_db",
        qdrant_url: str | None = None,
    ):
        # Load embedding model and BM25 index.
        self.embedding_model = SentenceTransformer(
            embedding_model_name,
            device=device,
            model_kwargs={'torch_dtype': torch.float16}
            )
        self.collection_name = collection_name

        if qdrant_url:
            self.qdrant = QdrantClient(url=qdrant_url)
        else:
            self.qdrant = QdrantClient(path=qdrant_path)

        with Path(bm25_index_path).open("rb") as f:
            bm25_blob = pickle.load(f)

        # Backward compatible with old tuple format and new dict format.
        if isinstance(bm25_blob, dict):
            self.bm25 = bm25_blob["bm25_model"]
            self.bm25_corpus = bm25_blob["corpus"]
            self.bm25_metadata = bm25_blob["metadata"]
        else:
            self.bm25, self.bm25_corpus, self.bm25_metadata = bm25_blob

    def merge_results(self, vector_results: Sequence[Any], top_bm25_indices, k: int = 60):
        """
        Merge results from Vector Search and BM25 with RRF.
        Assumes each chunk has a unique 'chunk_id'.
        """
        merged_score: Dict[Any, float] = {}
        document_store: Dict[Any, Dict[str, Any]] = {}

        # 1. RRF score for vector search.
        for rank, hit in enumerate(vector_results.points):
            payload = dict(hit.payload or {})
            chunk_id = payload.get("chunk_id", hit.id)
            if chunk_id is None:
                continue

            document_store[chunk_id] = payload
            merged_score[chunk_id] = merged_score.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

        # 2. RRF score for BM25 search.
        for rank, idx in enumerate(top_bm25_indices):
            chunk_data = dict(self.bm25_metadata[idx])
            chunk_id = chunk_data.get("chunk_id")
            if chunk_id is None:
                chunk_id = f"bm25_{idx}"
                chunk_data["chunk_id"] = chunk_id

            if chunk_id not in document_store:
                chunk_data.setdefault("content", self.bm25_corpus[idx])
                document_store[chunk_id] = chunk_data

            merged_score[chunk_id] = merged_score.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

        sorted_chunks = sorted(merged_score.items(), key=lambda item: item[1], reverse=True)

        final_results = []
        for chunk_id, score in sorted_chunks:
            doc_data = dict(document_store[chunk_id])
            doc_data["hybrid_score"] = score
            final_results.append(doc_data)

        return final_results

    def search(self, query: str, top_k: int = 30):
        # 1. Vector search.
        query_vector = self.embedding_model.encode(query).tolist()
        vector_results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
        )

        # 2. Keyword search.
        tokenized_query = ViTokenizer.tokenize(query.lower()).split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = bm25_scores.argsort()[-top_k:][::-1]

        merged_results = self.merge_results(vector_results, top_bm25_indices)
        return merged_results
