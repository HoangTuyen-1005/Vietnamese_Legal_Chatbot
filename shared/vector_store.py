from __future__ import annotations

import hashlib
from typing import List, Optional

import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer


def _stable_point_id(chunk_id: str) -> int:
    """
    Convert chunk_id into a stable integer id for Qdrant.
    This helps avoid accidental reindex mismatch when chunk order changes.
    """
    digest = hashlib.md5(chunk_id.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


class VectorStore:
    def __init__(self, settings):
        self.settings = settings
        self.client: QdrantClient | None = None
        self.embedding_model: SentenceTransformer | None = None
        self.collection_name = settings.QDRANT_COLLECTION
        self._law_catalog_cache: list[dict] | None = None

        self._init_client()
        self._init_embedding_model()

    def _init_client(self) -> None:
        qdrant_path = getattr(self.settings, "QDRANT_PATH", None)
        qdrant_url = getattr(self.settings, "QDRANT_URL", None)
        qdrant_path = qdrant_path.strip() if isinstance(qdrant_path, str) else qdrant_path
        qdrant_url = qdrant_url.strip() if isinstance(qdrant_url, str) else qdrant_url

        if qdrant_url:
            self.client = QdrantClient(url=qdrant_url)
        elif qdrant_path:
            self.client = QdrantClient(path=qdrant_path)
        else:
            self.client = QdrantClient(
                host=self.settings.QDRANT_HOST,
                port=self.settings.QDRANT_PORT,
            )

    def _init_embedding_model(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda":
            self.embedding_model = SentenceTransformer(
                self.settings.EMBEDDING_MODEL_NAME,
                device=device,
                model_kwargs={"torch_dtype": torch.float16},
            )
        else:
            self.embedding_model = SentenceTransformer(
                self.settings.EMBEDDING_MODEL_NAME,
                device=device,
            )

    def create_collection_if_not_exists(self) -> None:
        if self.client is None or self.embedding_model is None:
            raise RuntimeError("VectorStore is not initialized.")

        vector_size = self.embedding_model.get_sentence_embedding_dimension()

        try:
            exists = self.client.collection_exists(self.collection_name)
        except Exception:
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    def recreate_collection(self) -> None:
        """
        Use this when rebuilding the whole index from scratch.
        Safer for academic/demo workflow than trying to partially sync old vectors.
        """
        if self.client is None:
            raise RuntimeError("VectorStore is not initialized.")

        try:
            exists = self.client.collection_exists(self.collection_name)
        except Exception:
            exists = False

        if exists:
            self.client.delete_collection(self.collection_name)

        self.create_collection_if_not_exists()
        self._law_catalog_cache = None

    def index_chunks(self, chunks: List[dict], batch_size: int = 32) -> None:
        if self.client is None or self.embedding_model is None:
            raise RuntimeError("VectorStore is not initialized.")

        self._law_catalog_cache = None

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            batch_texts = [item.get("content", "") for item in batch]

            batch_ids = [
                _stable_point_id(str(item.get("chunk_id", f"chunk_{start+i}")))
                for i, item in enumerate(batch)
            ]

            batch_payloads = []
            for item in batch:
                payload = dict(item.get("metadata", {}))
                payload["chunk_id"] = item.get("chunk_id")
                payload["content"] = item.get("content", "")
                batch_payloads.append(payload)

            embeddings = self.embedding_model.encode(
                batch_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )

            self.client.upsert(
                collection_name=self.collection_name,
                points=models.Batch(
                    ids=batch_ids,
                    vectors=embeddings.tolist(),
                    payloads=batch_payloads,
                ),
            )

    def search(self, query: str, top_k: int = 10) -> List[dict]:
        if self.client is None or self.embedding_model is None:
            raise RuntimeError("VectorStore is not initialized.")

        query_vector = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        vector_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
        )

        results: List[dict] = []
        for hit in vector_results.points:
            payload = dict(hit.payload or {})
            results.append({
                "chunk_id": payload.get("chunk_id", str(hit.id)),
                "content": payload.get("content", ""),
                "metadata": {
                    k: v
                    for k, v in payload.items()
                    if k not in {"content", "chunk_id"}
                },
                "score": float(hit.score) if hit.score is not None else None,
                "source": "vector",
            })

        return results

    def get_chunks_by_metadata(
        self,
        so_hieu: str,
        dieu: Optional[str] = None,
        limit: int = 100,
    ) -> List[dict]:
        if self.client is None:
            raise RuntimeError("VectorStore is not initialized.")

        must_conditions = [
            FieldCondition(key="so_hieu", match=MatchValue(value=so_hieu)),
        ]
        if dieu:
            must_conditions.append(
                FieldCondition(key="dieu", match=MatchValue(value=dieu))
            )

        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(must=must_conditions),
            limit=limit,
        )

        results: List[dict] = []
        for record in records:
            payload = dict(record.payload or {})
            results.append({
                "chunk_id": payload.get("chunk_id", str(record.id)),
                "content": payload.get("content", ""),
                "metadata": {
                    k: v
                    for k, v in payload.items()
                    if k not in {"content", "chunk_id"}
                },
                "source": "vector_context",
            })

        return results

    def get_law_catalog(self, refresh: bool = False, max_points: int = 10000) -> list[dict]:
        if self.client is None:
            raise RuntimeError("VectorStore is not initialized.")

        if self._law_catalog_cache is not None and not refresh:
            return self._law_catalog_cache

        laws: dict[str, set[str]] = {}
        offset = None
        fetched = 0

        while fetched < max_points:
            page_limit = min(256, max_points - fetched)
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=None,
                limit=page_limit,
                offset=offset,
                with_vectors=False,
            )

            if not records:
                break

            fetched += len(records)
            for record in records:
                payload = dict(record.payload or {})
                so_hieu = payload.get("so_hieu")
                if not so_hieu:
                    continue

                aliases = laws.setdefault(str(so_hieu), set())
                aliases.add(str(so_hieu))
                for key in ("document_name", "source_file", "loai_van_ban"):
                    value = payload.get(key)
                    if isinstance(value, str) and value.strip():
                        aliases.add(value.strip())

            if offset is None:
                break

        catalog = [
            {
                "so_hieu": so_hieu,
                "alias_text": " ".join(sorted(aliases)).lower(),
            }
            for so_hieu, aliases in laws.items()
        ]

        self._law_catalog_cache = catalog
        return catalog
