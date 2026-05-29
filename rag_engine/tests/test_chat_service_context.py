from types import SimpleNamespace

from rag_engine.services.chat_service import ChatService


def make_service(max_chunks: int = 2) -> ChatService:
    return ChatService(
        bm25_store=None,
        vector_store=None,
        reranker=None,
        generator=None,
        settings=SimpleNamespace(
            MAX_CONTEXT_CHUNKS=max_chunks,
            MAX_CONTEXT_CHARS=10000,
        ),
        logger=None,
    )


def make_chunk(chunk_id: str, role: str | None = None) -> dict:
    chunk = {
        "chunk_id": chunk_id,
        "content": "Can thiệp sớm được áp dụng trong trường hợp sau đây.",
        "metadata": {
            "so_hieu": "32/2024/QH15",
            "dieu": "Điều 156",
            "ten_dieu": "Can thiệp sớm",
        },
    }
    if role:
        chunk["_context_role"] = role
    return chunk


def test_prompt_context_limit_keeps_reranked_and_referenced_chunks():
    service = make_service(max_chunks=2)
    reranked = [make_chunk("reranked", "reranked")]
    contexts = [
        make_chunk("low"),
        make_chunk("referenced", "referenced_dieu"),
        make_chunk("same", "same_dieu"),
        reranked[0],
    ]

    selected = service._limit_contexts_for_prompt(
        contexts=contexts,
        reranked_chunks=reranked,
        query_profile={"keywords": ["can thiệp sớm"]},
    )

    assert [chunk["chunk_id"] for chunk in selected] == ["referenced", "reranked"]
