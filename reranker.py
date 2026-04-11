from qdrant_client.http.models import FieldCondition, Filter, MatchValue
from sentence_transformers import CrossEncoder
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model = CrossEncoder(
            model_name,
            device=device,
            automodel_args={'torch_dtype': torch.float16}
            )

    def rerank(self, query, retrieved_docs, top_n=5):
        # Create pairs [query, document_content].
        if not retrieved_docs:
            return []

        pairs = [[query, doc.get("content", "")] for doc in retrieved_docs]
        scores = self.model.predict(pairs)

        for i, doc in enumerate(retrieved_docs):
            doc["rerank_score"] = float(scores[i])

        retrieved_docs.sort(key=lambda x: x.get("rerank_score", float("-inf")), reverse=True)
        return retrieved_docs[:top_n]

    @staticmethod
    def get_full_acticle_context(qdrant_client, metadata):
        # Lấy thông tin định danh của điều luật.
        so_hieu = metadata.get("so_hieu")
        dieu = metadata.get("dieu")

        article_filter = Filter(
            must=[
                FieldCondition(key="so_hieu", match=MatchValue(value=so_hieu)),
                FieldCondition(key="dieu", match=MatchValue(value=dieu)),
            ]
        )

        # Search article and gather full khoản/điểm context.
        records, _ = qdrant_client.scroll(
            collection_name="legal_documents",
            scroll_filter=article_filter,
            limit=50,
        )

        full_text = "\n".join(
            record.payload.get("content", "")
            for record in records
            if record.payload.get("content")
        )
        return full_text

    @staticmethod
    def get_full_article_context(qdrant_client, metadata):
        # Alias giữ tương thích với tên cũ bị typo.
        return Reranker.get_full_acticle_context(qdrant_client, metadata)
