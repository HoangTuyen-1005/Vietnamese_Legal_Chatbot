from __future__ import annotations

import re
import unicodedata
from typing import Sequence


REFUSAL_ANSWER = (
    "Xin l\u1ed7i, d\u1eef li\u1ec7u ph\u00e1p lu\u1eadt hi\u1ec7n t\u1ea1i c\u1ee7a t\u00f4i "
    "kh\u00f4ng \u0111\u1ec1 c\u1eadp \u0111\u1ebfn v\u1ea5n \u0111\u1ec1 n\u00e0y."
)

GENERATION_DISABLED_ANSWER = (
    "He thong dang tat che do sinh cau tra loi (ENABLE_GENERATION=False). "
    "Vui long tham khao phan nguon trich dan de xem can cu phap ly."
)

RETRIEVAL_SCORE_FLOOR = 0.006
RETRIEVAL_COVERAGE_FLOOR = 0.02
RETRIEVAL_LAW_OVERLAP_FLOOR = 0.05

UNSUPPORTED_SCOPE_PATTERNS = {
    "family_law": (
        r"\bket hon\b",
        r"\bly hon\b",
        r"\bkhai sinh\b",
        r"\bvo chong\b",
        r"\bhon nhan\b",
        r"\btai san chung cua vo chong\b",
    ),
    "labor_law": (
        r"\bnguoi lao dong\b",
        r"\bhop dong lao dong\b",
        r"\bnghi phep\b",
        r"\bphep nam\b",
        r"\bluong toi thieu\b",
        r"\bthu viec\b",
        r"\bnhan vien\b",
    ),
    "social_insurance": (
        r"\bbao hiem xa hoi\b",
        r"\bbao hiem that nghiep\b",
    ),
    "tax_law": (
        r"\bquyet toan thue\b",
        r"\bthue thu nhap\b",
        r"\bthue thu nhap ca nhan\b",
        r"\bthue thu nhap doanh nghiep\b",
        r"\bthue gia tri gia tang\b",
        r"\bma so thue\b",
        r"\bhoa don dien tu\b",
    ),
    "traffic_law": (
        r"\blai xe\b",
        r"\bden do\b",
        r"\bnong do con\b",
        r"\bot o\b",
        r"\bxe may\b",
        r"\bnghi dinh 100\b",
    ),
}

SUPPORTED_SCOPE_PATTERNS = (
    r"\bdan su\b",
    r"\bhinh su\b",
    r"\btoi pham\b",
    r"\bbo luat hinh su\b",
    r"\bbo luat dan su\b",
    r"\bluat dat dai\b",
    r"\bdat dai\b",
    r"\bngan hang\b",
    r"\btin dung\b",
    r"\bchuyen doi so\b",
    r"\bdu lieu\b",
)


def _safe_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(x) for x in value if x is not None).lower()
    return str(value).lower()


def _strip_accents(text: str) -> str:
    text = text.replace("Đ", "D").replace("đ", "d")
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )


def _normalize_for_match(text: str) -> str:
    text = _safe_text(text)
    text = _strip_accents(text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize_for_match(text: str) -> list[str]:
    stopwords = {
        "luat", "bo", "so", "nam", "va", "theo", "quy", "dinh",
        "cua", "duoc", "tai", "ve", "la", "dieu", "khoan", "diem",
    }
    tokens = re.findall(r"\w+", _normalize_for_match(text), flags=re.UNICODE)
    return [tok for tok in tokens if tok not in stopwords and len(tok) > 1]


def _pattern_matches(text: str, patterns: Sequence[str]) -> list[str]:
    return [
        pattern
        for pattern in patterns
        if re.search(pattern, text, flags=re.UNICODE)
    ]


def _scope_diagnostics(query_profile: dict | None) -> dict:
    if not query_profile:
        return {
            "scope_status": "unknown",
            "scope_matches": [],
            "supported_scope_matches": [],
        }

    raw_query = _normalize_for_match(query_profile.get("raw_query"))
    unsupported_matches: list[str] = []
    for scope, patterns in UNSUPPORTED_SCOPE_PATTERNS.items():
        if _pattern_matches(raw_query, patterns):
            unsupported_matches.append(scope)

    supported_matches = _pattern_matches(raw_query, SUPPORTED_SCOPE_PATTERNS)
    mentioned_law = _normalize_for_match(query_profile.get("mentioned_law"))
    supported_law_matches = _pattern_matches(mentioned_law, SUPPORTED_SCOPE_PATTERNS)
    has_hard_reference = bool(
        query_profile.get("mentioned_law")
        or query_profile.get("mentioned_law_codes")
        or query_profile.get("mentioned_articles")
        or query_profile.get("mentioned_clauses")
        or query_profile.get("mentioned_points")
        or query_profile.get("mentioned_chapters")
    )

    if (
        unsupported_matches
        and not supported_matches
        and (not has_hard_reference or not supported_law_matches)
    ):
        status = "likely_out_of_scope"
    elif unsupported_matches:
        status = "mixed_scope"
    else:
        status = "in_scope_or_unknown"

    return {
        "scope_status": status,
        "scope_matches": unsupported_matches,
        "supported_scope_matches": supported_matches,
    }


def _distinct_source_count(chunks: Sequence[dict]) -> int:
    source_ids = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        source_id = _safe_text(meta.get("so_hieu"))
        if source_id:
            source_ids.append(source_id)
    return len(set(source_ids))


def _law_overlap_with_metadata(mentioned_law: str, metadata: dict) -> float:
    mention_tokens = _tokenize_for_match(mentioned_law)
    if not mention_tokens:
        return 0.0

    haystack = " ".join([
        _safe_text(metadata.get("so_hieu")),
        _safe_text(metadata.get("document_name")),
        _safe_text(metadata.get("source_file")),
        _safe_text(metadata.get("loai_van_ban")),
        _safe_text(metadata.get("ten_chuong")),
        _safe_text(metadata.get("ten_dieu")),
    ])
    haystack_tokens = _tokenize_for_match(haystack)
    if not haystack_tokens:
        return 0.0

    inter = len(set(mention_tokens).intersection(set(haystack_tokens)))
    return inter / max(len(set(mention_tokens)), 1)


def _keyword_coverage(query_keywords: list[str], chunks: Sequence[dict]) -> float:
    query_tokens: list[str] = []
    for kw in query_keywords[:20]:
        query_tokens.extend(_tokenize_for_match(kw))

    if not query_tokens:
        return 0.0

    query_set = set(query_tokens)
    best = 0.0

    for chunk in chunks:
        meta = chunk.get("metadata", {})
        chunk_text = " ".join([
            _safe_text(meta.get("ten_dieu")),
            _safe_text(meta.get("dieu")),
            _safe_text(chunk.get("content", ""))[:600],
        ])
        chunk_norm = _normalize_for_match(chunk_text)
        chunk_tokens = set(_tokenize_for_match(chunk_text))
        if not chunk_tokens:
            continue

        matched = len(query_set.intersection(chunk_tokens))
        token_coverage = matched / max(len(query_set), 1)
        if len(query_set) >= 3 and matched < 2:
            token_coverage = min(token_coverage, 0.03)

        phrase_coverage = 0.0
        for kw in query_keywords[:20]:
            kw_norm = _normalize_for_match(kw)
            if " " in kw_norm and len(kw_norm) >= 6 and kw_norm in chunk_norm:
                phrase_coverage = 0.35
                break

        best = max(best, token_coverage, phrase_coverage)

    return best


def _max_law_overlap(mentioned_law: str, chunks: Sequence[dict]) -> float:
    overlap = 0.0
    for chunk in chunks:
        overlap = max(
            overlap,
            _law_overlap_with_metadata(mentioned_law, chunk.get("metadata", {})),
        )
    return overlap


def _max_metadata_boost(chunks: Sequence[dict]) -> float:
    best = 0.0
    for chunk in chunks:
        try:
            best = max(best, float(chunk.get("metadata_boost") or 0.0))
        except (TypeError, ValueError):
            continue
    return best


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _retrieval_score(chunk: dict) -> float:
    if "base_score" in chunk:
        return _safe_float(chunk.get("base_score"))
    return _safe_float(chunk.get("score"))


def diagnose_retrieval_refusal(
    chunks: Sequence[dict],
    query_profile: dict | None = None,
) -> dict:
    top_chunks = list(chunks[:5])
    if not top_chunks:
        return {
            "stage": "retrieval",
            "should_refuse": True,
            "reason": "no_candidates",
            "top_score": 0.0,
            "top_combined_score": 0.0,
            "coverage": 0.0,
            "law_overlap": 0.0,
            "metadata_boost": 0.0,
            "distinct_sources": 0,
        }

    top_score = _retrieval_score(top_chunks[0])
    top_combined_score = _safe_float(top_chunks[0].get("score"))
    distinct_sources = _distinct_source_count(top_chunks)
    metadata_boost = _max_metadata_boost(top_chunks)

    if query_profile is None:
        scope = _scope_diagnostics(query_profile)
        should_refuse = (
            top_score <= 0.0
            or (distinct_sources >= 4 and top_score < RETRIEVAL_SCORE_FLOOR)
        )
        return {
            "stage": "retrieval",
            "should_refuse": should_refuse,
            "reason": "weak_scores_without_profile" if should_refuse else "defer_to_reranker",
            "top_score": top_score,
            "top_combined_score": top_combined_score,
            "coverage": 0.0,
            "law_overlap": 0.0,
            "metadata_boost": metadata_boost,
            "distinct_sources": distinct_sources,
            **scope,
        }

    scope = _scope_diagnostics(query_profile)
    mentioned_law = _safe_text(query_profile.get("mentioned_law"))
    keywords = [str(x) for x in (query_profile.get("keywords") or []) if len(str(x)) >= 2]
    coverage = _keyword_coverage(keywords, top_chunks)
    law_overlap = _max_law_overlap(mentioned_law, top_chunks) if mentioned_law else 0.0

    if scope["scope_status"] == "likely_out_of_scope":
        return {
            "stage": "retrieval",
            "should_refuse": True,
            "reason": "unsupported_legal_domain_signal",
            "top_score": top_score,
            "top_combined_score": top_combined_score,
            "coverage": coverage,
            "law_overlap": law_overlap,
            "metadata_boost": metadata_boost,
            "distinct_sources": distinct_sources,
            "mentioned_law": mentioned_law or None,
            **scope,
        }

    if mentioned_law and (
        law_overlap >= 0.20
        or coverage >= 0.08
        or metadata_boost >= 0.15
    ):
        should_refuse = False
        reason = "plausible_law_keyword_or_metadata_signal"
    elif mentioned_law:
        should_refuse = (
            top_score < RETRIEVAL_SCORE_FLOOR
            and coverage < RETRIEVAL_COVERAGE_FLOOR
            and law_overlap < RETRIEVAL_LAW_OVERLAP_FLOOR
            and metadata_boost <= 0.0
        )
        reason = "weak_law_and_keyword_signals" if should_refuse else "defer_to_reranker"
    else:
        should_refuse = (
            top_score < RETRIEVAL_SCORE_FLOOR
            and coverage < RETRIEVAL_COVERAGE_FLOOR
            and distinct_sources >= 4
            and metadata_boost <= 0.0
        )
        reason = "weak_keyword_signals_across_many_sources" if should_refuse else "defer_to_reranker"

    return {
        "stage": "retrieval",
        "should_refuse": should_refuse,
        "reason": reason,
        "top_score": top_score,
        "top_combined_score": top_combined_score,
        "coverage": coverage,
        "law_overlap": law_overlap,
        "metadata_boost": metadata_boost,
        "distinct_sources": distinct_sources,
        "mentioned_law": mentioned_law or None,
        **scope,
    }


def diagnose_rerank_refusal(
    chunks: Sequence[dict],
    query_profile: dict | None = None,
) -> dict:
    top_chunks = list(chunks[:5])
    if not top_chunks:
        return {
            "stage": "rerank",
            "should_refuse": True,
            "reason": "no_reranked_candidates",
            "top_rerank_score": 0.0,
            "coverage": 0.0,
            "law_overlap": 0.0,
            "metadata_boost": 0.0,
        }

    top_rerank_score = _safe_float(top_chunks[0].get("rerank_score"))
    if query_profile is None:
        scope = _scope_diagnostics(query_profile)
        should_refuse = top_rerank_score < -1.0
        return {
            "stage": "rerank",
            "should_refuse": should_refuse,
            "reason": "weak_rerank_without_profile" if should_refuse else "rerank_score_passed",
            "top_rerank_score": top_rerank_score,
            "coverage": 0.0,
            "law_overlap": 0.0,
            "metadata_boost": 0.0,
            **scope,
        }

    scope = _scope_diagnostics(query_profile)
    mentioned_law = _safe_text(query_profile.get("mentioned_law"))
    keywords = [str(x) for x in (query_profile.get("keywords") or []) if len(str(x)) >= 2]
    coverage = _keyword_coverage(keywords, top_chunks)
    law_overlap = _max_law_overlap(mentioned_law, top_chunks) if mentioned_law else 0.0
    metadata_boost = _max_metadata_boost(top_chunks)

    if scope["scope_status"] == "likely_out_of_scope":
        return {
            "stage": "rerank",
            "should_refuse": True,
            "reason": "unsupported_legal_domain_signal",
            "top_rerank_score": top_rerank_score,
            "coverage": coverage,
            "law_overlap": law_overlap,
            "metadata_boost": metadata_boost,
            "mentioned_law": mentioned_law or None,
            **scope,
        }

    if (
        top_rerank_score >= 0.05
        or coverage >= 0.12
        or law_overlap >= 0.20
        or metadata_boost >= 0.15
    ):
        should_refuse = False
        reason = "supporting_signal_passed"
    elif mentioned_law:
        should_refuse = (
            top_rerank_score < -1.0
            and coverage < 0.04
            and law_overlap < 0.10
        )
        reason = "weak_rerank_and_law_signals" if should_refuse else "defer_to_prompt"
    else:
        should_refuse = (
            top_rerank_score < -1.0
            and coverage < 0.04
            and metadata_boost <= 0.0
        )
        reason = "weak_rerank_and_keyword_signals" if should_refuse else "defer_to_prompt"

    return {
        "stage": "rerank",
        "should_refuse": should_refuse,
        "reason": reason,
        "top_rerank_score": top_rerank_score,
        "coverage": coverage,
        "law_overlap": law_overlap,
        "metadata_boost": metadata_boost,
        "mentioned_law": mentioned_law or None,
        **scope,
    }


def should_refuse_after_retrieval(
    chunks: Sequence[dict],
    query_profile: dict | None = None,
) -> bool:
    return diagnose_retrieval_refusal(
        chunks,
        query_profile=query_profile,
    )["should_refuse"]


def should_refuse_after_rerank(
    chunks: Sequence[dict],
    query_profile: dict | None = None,
) -> bool:
    return diagnose_rerank_refusal(
        chunks,
        query_profile=query_profile,
    )["should_refuse"]


def is_refusal_answer(answer: str) -> bool:
    normalized = (answer or "").strip()
    if normalized == REFUSAL_ANSWER:
        return True

    lowered = _strip_accents(normalized.lower())
    refusal_patterns = [
        "khong de cap",
        "khong du can cu",
        "khong tim thay",
        "xin loi",
        "khong co trong du lieu",
    ]
    return any(pattern in lowered for pattern in refusal_patterns)
