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


def _profile_terms(query_profile: dict) -> list[str]:
    terms: list[str] = []
    for key in ("keywords", "legal_concepts", "retrieval_queries"):
        for value in query_profile.get(key, []) or []:
            text = str(value).strip()
            if text:
                terms.append(text)
    return terms


def _constraint_signal(query_profile: dict, text: str) -> float:
    text_norm = _normalize_for_match(text)
    if not text_norm:
        return 0.0

    score = 0.0
    for term in _profile_terms(query_profile)[:36]:
        term_norm = _normalize_for_match(term)
        if len(term_norm) < 8:
            continue

        if term_norm in text_norm:
            if any(marker in term_norm for marker in ("khong co", "chua co", "khong phai", "khong duoc")):
                score += 0.28
            elif " " in term_norm:
                score += 0.06

        neg_match = re.search(r"\b(khong co|chua co)\s+(.{3,80})", term_norm)
        if not neg_match:
            continue

        tail = re.split(r"\b(su dung|tu nam|tu truoc|thi|duoc|va|hoac|nhung)\b", neg_match.group(2))[0]
        tail = tail.strip()
        if len(tail) < 3:
            continue

        if f"{neg_match.group(1)} {tail}" in text_norm:
            score += 0.34
        elif _has_positive_counterpart(text_norm, tail):
            score -= 0.22

    return max(-0.30, min(score, 0.70))


def _has_positive_counterpart(text_norm: str, tail: str) -> bool:
    tail_tokens = [
        token
        for token in re.findall(r"\w+", tail)
        if len(token) > 1
    ]
    if not tail_tokens:
        return False
    tail_pattern = r"\s+".join(re.escape(token) for token in tail_tokens)
    return bool(re.search(rf"\bco(?:\s+\w+){{0,8}}\s+{tail_pattern}\b", text_norm))


def _label_match_boost(query_profile: dict, metadata: dict) -> float:
    boost = 0.0
    meta_dieu = _normalize_for_match(metadata.get("dieu"))
    meta_khoan = _normalize_for_match(metadata.get("khoan"))
    meta_diem = _normalize_for_match(metadata.get("diem"))
    meta_chuong = _normalize_for_match(metadata.get("chuong") or metadata.get("ten_chuong"))

    mentioned_articles = [
        _normalize_for_match(value)
        for value in query_profile.get("mentioned_articles", [])
        if str(value).strip()
    ]
    mentioned_clauses = [
        _normalize_for_match(value)
        for value in query_profile.get("mentioned_clauses", [])
        if str(value).strip()
    ]
    mentioned_points = [
        _normalize_for_match(value)
        for value in query_profile.get("mentioned_points", [])
        if str(value).strip()
    ]
    mentioned_chapters = [
        _normalize_for_match(value)
        for value in query_profile.get("mentioned_chapters", [])
        if str(value).strip()
    ]

    if mentioned_articles and meta_dieu in mentioned_articles:
        boost += 0.65
    if mentioned_clauses and meta_khoan in mentioned_clauses:
        boost += 0.12
    if mentioned_points and meta_diem in mentioned_points:
        boost += 0.08
    if mentioned_chapters and any(chapter in meta_chuong for chapter in mentioned_chapters):
        boost += 0.10

    return boost


def _intent_boost_weight(query_profile: dict) -> float:
    try:
        confidence = float(query_profile.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if query_profile.get("query_type") == "general_explanation":
        return 0.0
    return max(0.0, min(confidence, 0.60))


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
    intent_weight = _intent_boost_weight(query_profile)

    if mentioned_law:
        law_overlap = _law_mention_overlap(mentioned_law, meta)
        boost += law_overlap * 0.55
        if law_overlap < 0.15:
            boost -= 0.20

    boost += _label_match_boost(query_profile, meta)

    intent_boost = 0.0
    if query_type == "article_lookup":
        if cap_chunk == "dieu":
            intent_boost += 0.35
        elif cap_chunk == "khoan":
            intent_boost += 0.12

    elif query_type == "principle_lookup":
        if "nguyên tắc" in ten_dieu:
            intent_boost += 0.30
        elif "nguyên tắc" in content[:300]:
            intent_boost += 0.14

    elif query_type == "definition_lookup":
        if "giải thích từ ngữ" in ten_dieu:
            intent_boost += 0.30
        elif _looks_like_definition(content):
            intent_boost += 0.12

    elif query_type == "prohibited_acts":
        if "nghiêm cấm" in ten_dieu or "nghiêm cấm" in content[:260]:
            intent_boost += 0.20

    elif query_type == "conditions":
        if "điều kiện" in ten_dieu or "điều kiện" in content[:260]:
            intent_boost += 0.20

    elif query_type == "cases_circumstances":
        if "trường hợp" in ten_dieu or "trường hợp" in content[:260]:
            intent_boost += 0.20

    elif query_type == "procedure_lookup":
        if any(x in ten_dieu for x in ["thủ tục", "trình tự", "hồ sơ", "quy trình"]):
            intent_boost += 0.20

    boost += intent_boost * intent_weight

    boost += _keyword_overlap_boost(keywords, ten_dieu, dieu, content)
    boost += _constraint_signal(
        query_profile=query_profile,
        text=" ".join([ten_dieu, dieu, content[:1200]]),
    )

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
