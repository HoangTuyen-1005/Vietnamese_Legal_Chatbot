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
        chunk_tokens = set(_tokenize_for_match(chunk_text))
        if not chunk_tokens:
            continue

        matched = len(query_set.intersection(chunk_tokens))
        best = max(best, matched / max(len(query_set), 1))

    return best


def should_refuse_after_retrieval(
    chunks: Sequence[dict],
    query_profile: dict | None = None,
) -> bool:
    if not chunks:
        return True

    top_chunks = chunks[:5]
    top_score = float(top_chunks[0].get("score") or 0.0)

    if query_profile is None:
        distinct_sources = _distinct_source_count(top_chunks)
        return top_score < 0.01 or (distinct_sources >= 3 and top_score < 0.02)

    mentioned_law = _safe_text(query_profile.get("mentioned_law"))
    keywords = [str(x) for x in (query_profile.get("keywords") or []) if len(str(x)) >= 2]

    coverage = _keyword_coverage(keywords, top_chunks)

    if mentioned_law:
        overlap = 0.0
        for chunk in top_chunks:
            overlap = max(
                overlap,
                _law_overlap_with_metadata(mentioned_law, chunk.get("metadata", {})),
            )

        if overlap < 0.34:
            return True

        if coverage < 0.08 and top_score < 0.025:
            return True

    else:
        distinct_sources = _distinct_source_count(top_chunks)
        if coverage < 0.08 and top_score < 0.02:
            return True
        if coverage < 0.05 and distinct_sources >= 3:
            return True

    if _distinct_source_count(top_chunks) >= 4 and top_score < 0.03 and coverage < 0.18:
        return True

    return False


def should_refuse_after_rerank(chunks: Sequence[dict]) -> bool:
    if not chunks:
        return True

    top_rerank_score = float(chunks[0].get("rerank_score") or 0.0)
    return top_rerank_score < 0.10


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
