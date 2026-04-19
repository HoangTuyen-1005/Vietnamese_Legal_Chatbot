from __future__ import annotations

import re
import unicodedata


def safe_text(value) -> str:
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
    text = safe_text(text)
    text = _strip_accents(text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize_for_match(text: str) -> list[str]:
    stopwords = {
        "luat", "bo", "so", "nam", "va", "theo", "quy", "dinh",
        "cua", "duoc", "tai", "ve", "la", "dieu", "khoan", "diem",
    }
    tokens = re.findall(r"\w+", _normalize_for_match(text), flags=re.UNICODE)
    return [tok for tok in tokens if tok not in stopwords and len(tok) > 1]


def _token_overlap(left_tokens: list[str], right_tokens: list[str]) -> float:
    if not left_tokens or not right_tokens:
        return 0.0
    left_set = set(left_tokens)
    right_set = set(right_tokens)
    inter = len(left_set.intersection(right_set))
    return inter / max(len(left_set), 1)


def _law_mention_overlap(mentioned_law: str, metadata: dict) -> float:
    mentioned_norm = _normalize_for_match(mentioned_law)
    if not mentioned_norm:
        return 0.0

    haystack = " ".join([
        safe_text(metadata.get("so_hieu")),
        safe_text(metadata.get("document_name")),
        safe_text(metadata.get("source_file")),
        safe_text(metadata.get("loai_van_ban")),
        safe_text(metadata.get("ten_chuong")),
        safe_text(metadata.get("ten_dieu")),
        safe_text(metadata.get("parent_path")),
    ])
    haystack_norm = _normalize_for_match(haystack)
    if not haystack_norm:
        return 0.0

    if mentioned_norm in haystack_norm:
        return 1.0

    return _token_overlap(
        _tokenize_for_match(mentioned_norm),
        _tokenize_for_match(haystack_norm),
    )


def _looks_like_definition(content: str) -> bool:
    head = content[:260]
    return bool(
        re.search(r"^\s*\d+\.\s+.+\s+là\b", head)
        or "được hiểu là" in head
        or "bao gồm" in head
    )


def _keyword_overlap_boost(keywords: list[str], ten_dieu: str, dieu: str, content: str) -> float:
    overlap = 0.0
    title_norm = _normalize_for_match(f"{ten_dieu} {dieu}")
    content_norm = _normalize_for_match(content[:600])

    for kw in keywords[:20]:
        kw_norm = _normalize_for_match(kw)
        if not kw_norm:
            continue
        if kw_norm in title_norm:
            overlap += 0.08
        elif kw_norm in content_norm:
            overlap += 0.03

    return min(overlap, 0.30)


def compute_metadata_boost(query_profile: dict, chunk: dict) -> float:
    meta = chunk.get("metadata", {})
    content = safe_text(chunk.get("content", ""))
    ten_dieu = safe_text(meta.get("ten_dieu"))
    dieu = safe_text(meta.get("dieu"))
    cap_chunk = safe_text(meta.get("cap_chunk"))

    boost = 0.0
    query_type = query_profile.get("query_type")
    mentioned_law = safe_text(query_profile.get("mentioned_law"))
    keywords = [str(kw) for kw in query_profile.get("keywords", []) if len(str(kw)) >= 2]

    if mentioned_law:
        law_overlap = _law_mention_overlap(mentioned_law, meta)
        boost += law_overlap * 0.55
        if law_overlap < 0.15:
            boost -= 0.20

    if query_type == "article_lookup":
        if cap_chunk == "dieu":
            boost += 0.35
        elif cap_chunk == "khoan":
            boost += 0.12

    elif query_type == "principle_lookup":
        if "nguyên tắc" in ten_dieu:
            boost += 0.30
        elif "nguyên tắc" in content[:300]:
            boost += 0.14

    elif query_type == "definition_lookup":
        if "giải thích từ ngữ" in ten_dieu:
            boost += 0.30
        elif _looks_like_definition(content):
            boost += 0.12

    elif query_type == "prohibited_acts":
        if "nghiêm cấm" in ten_dieu or "nghiêm cấm" in content[:260]:
            boost += 0.20

    elif query_type == "conditions":
        if "điều kiện" in ten_dieu or "điều kiện" in content[:260]:
            boost += 0.20

    elif query_type == "cases_circumstances":
        if "trường hợp" in ten_dieu or "trường hợp" in content[:260]:
            boost += 0.20

    elif query_type == "procedure_lookup":
        if any(x in ten_dieu for x in ["thủ tục", "trình tự", "hồ sơ", "quy trình"]):
            boost += 0.20

    boost += _keyword_overlap_boost(keywords, ten_dieu, dieu, content)

    return boost


def rescore_candidates(
    candidates: list[dict],
    query_profile: dict,
    top_k: int | None = None,
) -> list[dict]:
    rescored = []

    for chunk in candidates:
        item = dict(chunk)
        base_score = float(item.get("score") or 0.0)
        metadata_boost = compute_metadata_boost(query_profile, item)

        item["base_score"] = base_score
        item["metadata_boost"] = metadata_boost
        item["score"] = base_score * 25.0 + metadata_boost

        rescored.append(item)

    rescored.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    if top_k is not None:
        return rescored[:top_k]
    return rescored
