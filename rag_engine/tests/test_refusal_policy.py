from rag_engine.retrieval.refusal_policy import (
    diagnose_retrieval_refusal,
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


def test_retrieval_keeps_metadata_boosted_candidate_after_rescore():
    query_profile = {
        "mentioned_law": "luật mẫu",
        "keywords": ["không khớp"],
    }
    chunk = make_chunk(
        "Nội dung không có nhiều từ khóa trùng trực tiếp.",
        score=0.001,
        metadata_boost=0.20,
        so_hieu="1/2024/QH15",
        ten_dieu="Điều khoản phù hợp theo metadata",
    )
    chunk["base_score"] = 0.001
    chunk["score"] = 0.225

    diagnostics = diagnose_retrieval_refusal([chunk], query_profile)

    assert diagnostics["should_refuse"] is False
    assert diagnostics["reason"] == "plausible_law_keyword_or_metadata_signal"


def test_retrieval_keeps_newly_supported_labor_domain_signal():
    query_profile = {
        "mentioned_law": None,
        "raw_query": "Nguoi lao dong duoc nghi phep nam bao nhieu ngay?",
        "keywords": ["nguoi lao dong", "nghi phep nam"],
    }
    chunks = [
        make_chunk(
            "Nguoi dai dien va nguoi lao dong trong mot quy dinh khac.",
            score=0.032,
            metadata_boost=0.45,
            so_hieu="91/2015/QH13",
            ten_dieu="Dai dien",
        )
    ]

    diagnostics = diagnose_retrieval_refusal(chunks, query_profile)

    assert diagnostics["should_refuse"] is False
    assert diagnostics["scope_status"] == "in_scope_or_unknown"


def test_scope_guard_does_not_block_explicit_supported_law_reference():
    query_profile = {
        "mentioned_law": "bo luat hinh su",
        "raw_query": "Toi cuong ep ket hon theo Bo luat Hinh su bi xu ly the nao?",
        "keywords": ["toi cuong ep ket hon"],
    }
    chunks = [
        make_chunk(
            "Toi cuong ep ket hon bi xu ly theo Bo luat Hinh su.",
            score=0.020,
            metadata_boost=0.30,
            so_hieu="01/VBHN-VPQH",
            ten_dieu="Toi cuong ep ket hon",
        )
    ]

    diagnostics = diagnose_retrieval_refusal(chunks, query_profile)

    assert diagnostics["should_refuse"] is False
    assert diagnostics["scope_status"] == "in_scope_or_unknown"


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


def test_rerank_keeps_newly_supported_traffic_domain_signal():
    query_profile = {
        "mentioned_law": None,
        "raw_query": "Vuot den do bi phat tien bao nhieu theo nghi dinh 100?",
        "keywords": ["vuot den do", "phat tien"],
    }
    chunks = [
        make_chunk(
            "Quy dinh ve tien va tai san trong van ban hien co.",
            rerank_score=2.5,
            metadata_boost=0.40,
            ten_dieu="Tien va tai san",
        )
    ]

    assert should_refuse_after_rerank(chunks, query_profile) is False


def test_empty_candidates_are_still_refused():
    assert should_refuse_after_retrieval([], {"keywords": []}) is True
    assert should_refuse_after_rerank([], {"keywords": []}) is True
