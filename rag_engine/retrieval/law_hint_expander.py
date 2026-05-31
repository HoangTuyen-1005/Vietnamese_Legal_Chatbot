from __future__ import annotations

import re
import unicodedata
from typing import Sequence


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
    text = re.sub(r"[_\-/\\.,;:()\[\]{}]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize_for_match(text: str) -> list[str]:
    stopwords = {
        "luat", "bo", "so", "nam", "va", "theo", "quy", "dinh",
        "cua", "duoc", "tai", "ve", "la",
    }
    tokens = re.findall(r"\w+", _normalize_for_match(text), flags=re.UNICODE)
    return [tok for tok in tokens if tok not in stopwords and len(tok) > 1]


def _overlap_score(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return 0.0
    left_set = set(left)
    right_set = set(right)
    inter = len(left_set.intersection(right_set))
    return inter / max(len(left_set), 1)


def _resolve_so_hieu(mentioned_law: str, law_catalog: Sequence[dict]) -> str | None:
    if not mentioned_law or not law_catalog:
        return None

    m = re.search(r"\b\d{1,3}/\d{4}/qh\d+\b|\b\d{1,3}/vbhn-vpqh\b", _safe_text(mentioned_law))
    if m:
        code = m.group(0).lower()
        for entry in law_catalog:
            if _safe_text(entry.get("so_hieu")) == code:
                return entry.get("so_hieu")

    mention_norm = _normalize_for_match(mentioned_law)
    mention_tokens = _tokenize_for_match(mention_norm)

    best_so_hieu: str | None = None
    best_score = 0.0

    for entry in law_catalog:
        so_hieu = entry.get("so_hieu")
        alias_text = _safe_text(entry.get("alias_text"))
        alias_norm = _normalize_for_match(alias_text)
        alias_tokens = _tokenize_for_match(alias_norm)

        score = _overlap_score(mention_tokens, alias_tokens)
        if mention_norm and mention_norm in alias_norm:
            score += 0.4

        if score > best_score:
            best_score = score
            best_so_hieu = so_hieu

    if best_score < 0.35:
        return None
    if not best_so_hieu:
        return None
    return str(best_so_hieu)


def _resolve_target_laws(
    query_profile: dict,
    law_catalog: Sequence[dict],
    retrieved: Sequence[dict],
    max_laws_without_mention: int,
) -> list[str]:
    targets: list[str] = []
    seen: set[str] = set()

    mentions = []
    for key in ("mentioned_law", "mentioned_law_codes", "candidate_fields"):
        value = query_profile.get(key)
        if isinstance(value, list):
            mentions.extend(str(item) for item in value if item)
        elif value:
            mentions.append(str(value))

    for mention in mentions:
        so_hieu = _resolve_so_hieu(mentioned_law=mention, law_catalog=law_catalog)
        if so_hieu and so_hieu not in seen:
            targets.append(so_hieu)
            seen.add(so_hieu)

    if targets:
        return targets

    query_text = " ".join(mentions + _profile_terms(query_profile))
    query_tokens = _tokenize_for_match(query_text)
    law_scores: dict[str, float] = {}

    for chunk in retrieved:
        meta = chunk.get("metadata", {}) or {}
        so_hieu = meta.get("so_hieu")
        if not so_hieu:
            continue
        haystack = " ".join([
            _safe_text(meta.get("document_name")),
            _safe_text(meta.get("source_file")),
            _safe_text(meta.get("loai_van_ban")),
            _safe_text(meta.get("ten_dieu")),
            _safe_text(chunk.get("content", ""))[:400],
        ])
        score = _overlap_score(query_tokens, _tokenize_for_match(haystack))
        law_scores[str(so_hieu)] = max(law_scores.get(str(so_hieu), 0.0), score)

    for so_hieu, score in sorted(law_scores.items(), key=lambda item: item[1], reverse=True):
        if score <= 0:
            continue
        targets.append(so_hieu)
        if len(targets) >= max_laws_without_mention:
            break

    if targets:
        return targets

    return [
        str(entry.get("so_hieu"))
        for entry in law_catalog[:max_laws_without_mention]
        if entry.get("so_hieu")
    ]


def _looks_like_definition(content: str) -> bool:
    head = content[:260]
    return bool(
        re.search(r"^\s*\d+\.\s+.+\s+là\b", head)
        or "được hiểu là" in head
        or "bao gồm" in head
    )


def _intent_boost_weight(query_profile: dict) -> float:
    try:
        confidence = float(query_profile.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if _safe_text(query_profile.get("query_type")) == "general_explanation":
        return 0.0
    return max(0.0, min(confidence, 0.60))


def _label_match_score(query_profile: dict, metadata: dict) -> float:
    meta_dieu = _normalize_for_match(metadata.get("dieu"))
    meta_khoan = _normalize_for_match(metadata.get("khoan"))
    meta_diem = _normalize_for_match(metadata.get("diem"))
    score = 0.0

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

    if mentioned_articles and meta_dieu in mentioned_articles:
        score += 5.0
    if mentioned_clauses and meta_khoan in mentioned_clauses:
        score += 1.0
    if mentioned_points and meta_diem in mentioned_points:
        score += 0.8
    return score


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
                score += 3.2
            elif " " in term_norm:
                score += 0.5

        neg_match = re.search(r"\b(khong co|chua co)\s+(.{3,80})", term_norm)
        if not neg_match:
            continue

        tail = re.split(r"\b(su dung|tu nam|tu truoc|thi|duoc|va|hoac|nhung)\b", neg_match.group(2))[0]
        tail = tail.strip()
        if len(tail) < 3:
            continue

        if f"{neg_match.group(1)} {tail}" in text_norm:
            score += 3.8
        elif _has_positive_counterpart(text_norm, tail):
            score -= 2.4

    return max(-3.0, min(score, 7.0))


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


def _hint_score(query_profile: dict, chunk: dict) -> float:
    meta = chunk.get("metadata", {})
    content = _safe_text(chunk.get("content", ""))[:900]
    ten_dieu = _safe_text(meta.get("ten_dieu"))
    cap_chunk = _safe_text(meta.get("cap_chunk"))
    query_type = _safe_text(query_profile.get("query_type"))
    keywords = [kw for kw in query_profile.get("keywords", []) if len(str(kw)) >= 2]
    intent_weight = _intent_boost_weight(query_profile)

    score = 0.0
    score += _label_match_score(query_profile, meta)
    score += _constraint_signal(
        query_profile=query_profile,
        text=" ".join([ten_dieu, content]),
    )

    profile_terms = keywords + _profile_terms(query_profile)
    for kw in profile_terms[:24]:
        kw_norm = _normalize_for_match(str(kw))
        if not kw_norm:
            continue
        if kw_norm in _normalize_for_match(ten_dieu):
            score += 4.0 if kw_norm == _normalize_for_match(ten_dieu) else 1.8
        elif kw_norm in _normalize_for_match(content):
            score += 0.8

        kw_tokens = _tokenize_for_match(kw_norm)
        title_tokens = _tokenize_for_match(ten_dieu)
        if len(kw_tokens) >= 2 and _overlap_score(kw_tokens, title_tokens) >= 0.75:
            score += 2.5

    intent_score = 0.0
    if query_type == "article_lookup":
        if cap_chunk == "dieu":
            intent_score += 1.2
        elif cap_chunk == "khoan":
            intent_score += 0.6
    elif query_type == "principle_lookup":
        if "nguyên tắc" in ten_dieu:
            intent_score += 2.5
        elif "nguyên tắc" in content[:260]:
            intent_score += 1.2
    elif query_type == "definition_lookup":
        if "giải thích từ ngữ" in ten_dieu:
            intent_score += 2.0
        elif _looks_like_definition(content):
            intent_score += 1.2
    elif query_type == "conditions":
        if "điều kiện" in ten_dieu or "điều kiện" in content[:260]:
            intent_score += 1.5
    elif query_type == "cases_circumstances":
        if "trường hợp" in ten_dieu or "trường hợp" in content[:260]:
            intent_score += 1.5
    elif query_type == "prohibited_acts":
        if "nghiêm cấm" in ten_dieu or "nghiêm cấm" in content[:260]:
            intent_score += 1.5

    score += intent_score * intent_weight

    return score


def augment_with_law_hints(
    retrieved: Sequence[dict],
    query_profile: dict,
    vector_store,
    add_k: int = 8,
    max_laws_without_mention: int = 8,
) -> list[dict]:
    merged: dict[str, dict] = {}
    for idx, chunk in enumerate(retrieved):
        item = dict(chunk)
        chunk_id = str(item.get("chunk_id", f"retrieved_{idx}"))
        merged[chunk_id] = item

    try:
        law_catalog = vector_store.get_law_catalog()
    except Exception:
        return list(merged.values())

    target_laws = _resolve_target_laws(
        query_profile=query_profile,
        law_catalog=law_catalog,
        retrieved=retrieved,
        max_laws_without_mention=max_laws_without_mention,
    )

    if not target_laws:
        return list(merged.values())

    scored_hints: list[dict] = []
    for law_index, so_hieu in enumerate(target_laws):
        try:
            hint_chunks = vector_store.get_chunks_by_metadata(so_hieu=so_hieu, limit=300)
        except Exception:
            continue

        for idx, chunk in enumerate(hint_chunks):
            hint = dict(chunk)
            hint_strength = _hint_score(query_profile, hint)
            if hint_strength <= 0:
                continue

            hint["score"] = hint_strength * 0.01
            hint["source"] = "law_hint"
            chunk_id = str(hint.get("chunk_id", f"hint_{law_index}_{idx}"))
            hint["chunk_id"] = chunk_id
            scored_hints.append(hint)

    scored_hints.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    for hint in scored_hints[:add_k]:
        chunk_id = str(hint.get("chunk_id"))
        current = merged.get(chunk_id)
        if current is None:
            merged[chunk_id] = hint
            continue

        current_score = float(current.get("score") or 0.0)
        hint_score = float(hint.get("score") or 0.0)
        if hint_score > current_score:
            merged[chunk_id] = hint

    return sorted(
        merged.values(),
        key=lambda x: float(x.get("score") or 0.0),
        reverse=True,
    )
