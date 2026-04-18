def reciprocal_rank_fusion(
    bm25_results: list[dict],
    vector_results: list[dict],
    k: int = 60
) -> list[dict]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.
    """
    fused_scores = {}
    doc_map = {}

    for rank, doc in enumerate(bm25_results, start=1):
        chunk_id = doc["chunk_id"]
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        doc_map[chunk_id] = doc

    for rank, doc in enumerate(vector_results, start=1):
        chunk_id = doc["chunk_id"]
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        doc_map[chunk_id] = doc_map.get(chunk_id, doc)

    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for chunk_id, fused_score in ranked:
        item = dict(doc_map[chunk_id])
        item["score"] = fused_score
        item["source"] = "hybrid"
        results.append(item)

    return results


def hybrid_search(
    query: str,
    bm25_store,
    vector_store,
    top_k_bm25: int = 10,
    top_k_vector: int = 10,
    top_k_final: int = 10,
    rrf_k: int = 60,
) -> list[dict]:
    bm25_results = bm25_store.search(query, top_k=top_k_bm25)
    vector_results = vector_store.search(query, top_k=top_k_vector)

    fused_results = reciprocal_rank_fusion(
        bm25_results=bm25_results,
        vector_results=vector_results,
        k=rrf_k,
    )
    return fused_results[:top_k_final]