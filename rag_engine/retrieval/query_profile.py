from __future__ import annotations

import re
import unicodedata
from typing import Any


LEGAL_QUERY_TYPES = [
    "definition_lookup",
    "article_lookup",
    "principle_lookup",
    "prohibited_acts",
    "conditions",
    "cases_circumstances",
    "penalty_lookup",
    "procedure_lookup",
    "general_explanation",
]

LAW_MENTION_PATTERNS = [
    r"\b\d{1,3}/\d{4}/qh\d+\b",
    r"\b\d{1,3}/vbhn-vpqh\b",
    r"\bbộ luật\s+[^\?,\.;:\n]+?(?=\s+(quy|được|là|gì|theo|những|bao|ở|gồm)\b|$)",
    r"\bluật\s+[^\?,\.;:\n]+?(?=\s+(quy|được|là|gì|theo|những|bao|ở|gồm)\b|$)",
]

LAW_MENTION_PATTERNS_NO_ACCENTS = [
    r"\b\d{1,3}/\d{4}/qh\d+\b",
    r"\b\d{1,3}/vbhn-vpqh\b",
    r"\bbo luat\s+[^\?,\.;:\n]+?(?=\s+(quy|duoc|la|gi|theo|nhung|bao|o|gom)\b|$)",
    r"\bluat\s+[^\?,\.;:\n]+?(?=\s+(quy|duoc|la|gi|theo|nhung|bao|o|gom)\b|$)",
]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _strip_accents(text: str) -> str:
    text = text.replace("Đ", "D").replace("đ", "d")
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )


def _extract_ngram_phrases(tokens: list[str], max_phrases: int = 12) -> list[str]:
    phrases: list[str] = []
    for n in (4, 3, 2):
        for i in range(0, len(tokens) - n + 1):
            phrase = " ".join(tokens[i : i + n])
            if len(phrase) < 8:
                continue
            if phrase not in phrases:
                phrases.append(phrase)
            if len(phrases) >= max_phrases:
                return phrases
    return phrases


def extract_keywords(query: str) -> list[str]:
    q = normalize_text(query)

    raw_tokens = re.findall(r"\w+", q, flags=re.UNICODE)
    stopwords = {
        "là", "gì", "ở", "đâu", "nào", "của", "theo", "được", "có", "không",
        "như", "thế", "và", "hay", "trong", "về", "tại", "bao", "nhiêu",
        "quy", "định", "điều", "khoản", "điểm", "luật", "bộ", "các",
        "những", "nay", "đúng", "một",
    }
    tokens = [tok for tok in raw_tokens if tok not in stopwords and len(tok) > 1]
    phrases = _extract_ngram_phrases(tokens)

    result = []
    for item in phrases + tokens:
        if item not in result:
            result.append(item)
    return result


def detect_query_type(query: str) -> str:
    q = normalize_text(query)

    if any(p in q for p in [
        "điều nào",
        "quy định tại điều nào",
        "được quy định tại điều nào",
        "được quy định ở điều nào",
        "nằm ở điều nào",
    ]):
        return "article_lookup"

    if any(p in q for p in [
        "nguyên tắc",
        "quy định những nguyên tắc nào",
        "theo nguyên tắc nào",
    ]):
        return "principle_lookup"

    if any(p in q for p in [
        "là gì",
        "được hiểu như thế nào",
        "được hiểu là gì",
        "là như thế nào",
        "bao gồm những gì",
        "gồm những gì",
        "bao gồm những loại nào",
        "gồm những loại nào",
        "có được coi là",
        "có phải là",
        "có được xem là",
        "là ai",
    ]):
        return "definition_lookup"

    if any(p in q for p in [
        "bị nghiêm cấm",
        "nghiêm cấm",
        "cấm những gì",
        "hành vi nào bị cấm",
    ]):
        return "prohibited_acts"

    if any(p in q for p in [
        "điều kiện",
        "phải đáp ứng",
        "đáp ứng những gì",
        "cần những gì",
    ]):
        return "conditions"

    if any(p in q for p in [
        "trường hợp nào",
        "những trường hợp nào",
        "các trường hợp nào",
    ]):
        return "cases_circumstances"

    if any(p in q for p in [
        "mức phạt",
        "hình phạt",
        "bị phạt bao nhiêu",
        "mức án",
        "bị xử lý thế nào",
    ]):
        return "penalty_lookup"

    if any(p in q for p in [
        "thủ tục",
        "trình tự",
        "quy trình",
        "hồ sơ",
    ]):
        return "procedure_lookup"

    return "general_explanation"


def detect_mentioned_law(query: str) -> str | None:
    q = normalize_text(query)
    tail_noise = {
        "là", "gì", "như", "thế", "nào", "bao", "nhiêu",
        "đúng", "không", "được", "quy", "định", "theo",
        "những", "gồm",
    }

    for pattern in LAW_MENTION_PATTERNS:
        m = re.search(pattern, q)
        if not m:
            continue

        value = normalize_text(m.group(0))
        words = value.split()
        while words and words[-1] in tail_noise:
            words.pop()
        value = " ".join(words[:10]).strip()
        if value:
            return value

    # Fallback: sometimes users omit diacritics for law names.
    q_no_accents = _strip_accents(q)
    for pattern in LAW_MENTION_PATTERNS_NO_ACCENTS:
        m = re.search(pattern, q_no_accents)
        if not m:
            continue
        value = normalize_text(m.group(0))
        words = value.split()
        while words and words[-1] in tail_noise:
            words.pop()
        value = " ".join(words[:10]).strip()
        if value:
            return value

    return None


def detect_candidate_fields(query: str, max_fields: int = 2) -> list[str]:
    # Kept for backward compatibility with older profile schema.
    # This is intentionally generic and avoids hard-coded legal domain buckets.
    mentioned_law = detect_mentioned_law(query)
    if not mentioned_law:
        return []
    return [mentioned_law][:max_fields]


def build_query_profile(query: str) -> dict[str, Any]:
    query_type = detect_query_type(query)
    mentioned_law = detect_mentioned_law(query)
    candidate_fields = detect_candidate_fields(query)
    keywords = extract_keywords(query)

    needs_exact_article = query_type in {
        "article_lookup",
        "definition_lookup",
        "principle_lookup",
        "penalty_lookup",
        "prohibited_acts",
        "conditions",
        "cases_circumstances",
    }

    return {
        "query_type": query_type,
        "needs_exact_article": needs_exact_article,
        "mentioned_law": mentioned_law,
        "candidate_fields": candidate_fields,
        "keywords": keywords,
        "raw_query": query,
    }
