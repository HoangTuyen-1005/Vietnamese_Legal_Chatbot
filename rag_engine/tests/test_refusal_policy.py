from rag_engine.retrieval.refusal_policy import (
    should_refuse_after_rerank,
    should_refuse_after_retrieval,
)


def make_chunk(
    content: str,
    *,
    score: float = 0.016,
    rerank_score: float | None = None,
    metadata_boost: float | None = None,
    so_hieu: str = "91/2015/QH13",
    ten_dieu: str = "",
) -> dict:
    chunk = {
        "chunk_id": f"{so_hieu}:{ten_dieu}:{content[:12]}",
        "content": content,
        "metadata": {
            "so_hieu": so_hieu,
            "ten_dieu": ten_dieu,
        },
        "score": score,
    }
    if rerank_score is not None:
        chunk["rerank_score"] = rerank_score
    if metadata_boost is not None:
        chunk["metadata_boost"] = metadata_boost
    return chunk


def test_retrieval_keeps_keyword_match_even_when_law_metadata_is_weak():
    query_profile = {
        "mentioned_law": "luật chuyển đổi số",
        "keywords": ["chuyển đổi số", "quá trình chuyển đổi"],
    }
    chunks = [
        make_chunk(
            "1. Chuyển đổi số là quá trình chuyển đổi hoạt động lên môi trường số.",
            score=0.010,
            so_hieu="148/2025/QH15",
            ten_dieu="Giải thích từ ngữ",
        )
    ]

    assert should_refuse_after_retrieval(chunks, query_profile) is False


def test_retrieval_refuses_when_all_early_signals_are_weak():
    query_profile = {
        "mentioned_law": None,
        "keywords": ["ly hôn", "hôn nhân"],
    }
    chunks = [
        make_chunk("Quy định về tài sản.", score=0.005, so_hieu="1/2024/QH15"),
        make_chunk("Quy định về đất đai.", score=0.004, so_hieu="2/2024/QH15"),
        make_chunk("Quy định về tín dụng.", score=0.004, so_hieu="3/2024/QH15"),
        make_chunk("Quy định về chuyển đổi số.", score=0.003, so_hieu="4/2024/QH15"),
    ]

    assert should_refuse_after_retrieval(chunks, query_profile) is True


def test_rerank_keeps_negative_score_when_other_evidence_is_strong():
    query_profile = {
        "mentioned_law": None,
        "keywords": ["người thành niên"],
    }
    chunks = [
        make_chunk(
            "Người từ đủ mười tám tuổi trở lên là người thành niên.",
            rerank_score=-2.5,
            metadata_boost=0.20,
            ten_dieu="Người thành niên",
        )
    ]

    assert should_refuse_after_rerank(chunks, query_profile) is False


def test_rerank_refuses_when_score_and_evidence_are_weak():
    query_profile = {
        "mentioned_law": None,
        "keywords": ["nghỉ thai sản"],
    }
    chunks = [
        make_chunk(
            "Quyền sở hữu đối với tài sản được xác lập theo quy định.",
            rerank_score=-1.2,
            metadata_boost=0.0,
            ten_dieu="Quyền sở hữu",
        )
    ]

    assert should_refuse_after_rerank(chunks, query_profile) is True


def test_empty_candidates_are_still_refused():
    assert should_refuse_after_retrieval([], {"keywords": []}) is True
    assert should_refuse_after_rerank([], {"keywords": []}) is True
