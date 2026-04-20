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


def _looks_like_definition(content: str) -> bool:
    head = content[:260]
    return bool(
        re.search(r"^\s*\d+\.\s+.+\s+là\b", head)
        or "được hiểu là" in head
        or "bao gồm" in head
    )


def _hint_score(query_profile: dict, chunk: dict) -> float:
    meta = chunk.get("metadata", {})
    content = _safe_text(chunk.get("content", ""))[:900]
    ten_dieu = _safe_text(meta.get("ten_dieu"))
    cap_chunk = _safe_text(meta.get("cap_chunk"))
    query_type = _safe_text(query_profile.get("query_type"))
    keywords = [kw for kw in query_profile.get("keywords", []) if len(str(kw)) >= 2]

    score = 0.0
    for kw in keywords[:16]:
        kw_norm = _normalize_for_match(str(kw))
        if not kw_norm:
            continue
        if kw_norm in _normalize_for_match(ten_dieu):
            score += 1.8
        elif kw_norm in _normalize_for_match(content):
            score += 0.8

    if query_type == "article_lookup":
        if cap_chunk == "dieu":
            score += 1.2
        elif cap_chunk == "khoan":
            score += 0.6
    elif query_type == "principle_lookup":
        if "nguyên tắc" in ten_dieu:
            score += 2.5
        elif "nguyên tắc" in content[:260]:
            score += 1.2
    elif query_type == "definition_lookup":
        if "giải thích từ ngữ" in ten_dieu:
            score += 2.0
        elif _looks_like_definition(content):
            score += 1.2
    elif query_type == "conditions":
        if "điều kiện" in ten_dieu or "điều kiện" in content[:260]:
            score += 1.5
    elif query_type == "cases_circumstances":
        if "trường hợp" in ten_dieu or "trường hợp" in content[:260]:
            score += 1.5
    elif query_type == "prohibited_acts":
        if "nghiêm cấm" in ten_dieu or "nghiêm cấm" in content[:260]:
            score += 1.5

    return score


def augment_with_law_hints(
    retrieved: Sequence[dict],
    query_profile: dict,
    vector_store,
    add_k: int = 8,
) -> list[dict]:
    merged: dict[str, dict] = {}
    for idx, chunk in enumerate(retrieved):
        item = dict(chunk)
        chunk_id = str(item.get("chunk_id", f"retrieved_{idx}"))
        merged[chunk_id] = item

    mentioned_law = _safe_text(query_profile.get("mentioned_law"))
    if not mentioned_law:
        return list(merged.values())

    try:
        law_catalog = vector_store.get_law_catalog()
    except Exception:
        return list(merged.values())

    so_hieu = _resolve_so_hieu(mentioned_law=mentioned_law, law_catalog=law_catalog)
    if so_hieu is None:
        return list(merged.values())

    try:
        hint_chunks = vector_store.get_chunks_by_metadata(so_hieu=so_hieu, limit=300)
    except Exception:
        return list(merged.values())

    scored_hints: list[dict] = []
    for idx, chunk in enumerate(hint_chunks):
        hint = dict(chunk)
        hint_strength = _hint_score(query_profile, hint)
        if hint_strength <= 0:
            continue

        hint["score"] = hint_strength * 0.01
        hint["source"] = "law_hint"
        chunk_id = str(hint.get("chunk_id", f"hint_{idx}"))
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
