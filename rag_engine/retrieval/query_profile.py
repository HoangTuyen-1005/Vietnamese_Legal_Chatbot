from __future__ import annotations

import json
import os
import re
import ast
from typing import Any

try:
    from google import genai
    from google.genai import types
except Exception as exc:  # pragma: no cover
    genai = None
    types = None
    _IMPORT_ERROR = exc
else:  # pragma: no cover
    _IMPORT_ERROR = None


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

INTENT_FLAG_KEYS = (
    "asks_definition",
    "asks_article_location",
    "asks_principle",
    "asks_prohibited",
    "asks_conditions",
    "asks_cases",
    "asks_penalty",
    "asks_procedure",
    "asks_duration",
    "asks_yes_no",
)

PROFILE_VERSION = "gemini_v1"
DEFAULT_QUERY_PROFILE_MODEL_NAME = "gemini-2.0-flash"
DEFAULT_QUERY_PROFILE_TEMPERATURE = 0.0
DEFAULT_QUERY_PROFILE_MAX_OUTPUT_TOKENS = 2048

QUERY_PROFILE_PROVIDER_ERROR_ANSWER = (
    "D\u1ecbch v\u1ee5 ph\u00e2n t\u00edch c\u00e2u h\u1ecfi \u0111ang t\u1ea1m th\u1eddi "
    "kh\u00f4ng kh\u1ea3 d\u1ee5ng. Vui l\u00f2ng th\u1eed l\u1ea1i sau \u00edt ph\u00fat."
)


class QueryProfileProviderError(RuntimeError):
    def __init__(self, message: str, *, error_code: str = "query_profile_provider_error"):
        super().__init__(message)
        self.error_code = error_code


def _as_str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_str_list(value: Any, *, max_items: int = 24) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []

    result: list[str] = []
    seen = set()
    for item in value:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= max_items:
            break
    return result


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(parsed, 1.0))


def _dedupe(values: list[str], *, max_items: int = 24) -> list[str]:
    result: list[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= max_items:
            break
    return result


def _extract_labels_from_texts(texts: list[str]) -> dict[str, list[str]]:
    haystack = "\n".join(str(text or "") for text in texts)
    haystack_no_accents = _strip_accents_for_match(haystack)

    articles = [
        _numbered_label("Điều", match.group(1), upper_suffix=True)
        for match in re.finditer(r"\bdieu\s+(\d+[a-z]?)\b", haystack_no_accents)
    ]
    clauses = [
        _numbered_label("Khoản", match.group(1))
        for match in re.finditer(r"\bkhoan\s+(\d+[a-z]?)\b", haystack_no_accents)
    ]
    points = [
        f"Điểm {match.group(1).lower()}"
        for match in re.finditer(r"\bdiem\s+([a-z])\b", haystack_no_accents)
    ]
    chapters = [
        f"Chương {match.group(1).upper()}"
        for match in re.finditer(r"\bchuong\s+([ivxlcdm]+|\d+)\b", haystack_no_accents)
    ]
    law_codes = re.findall(
        r"\b\d{1,3}/\d{4}/qh\d+\b|\b\d{1,3}/vbhn-vpqh\b",
        haystack_no_accents,
    )

    return {
        "mentioned_law_codes": _dedupe(law_codes, max_items=8),
        "mentioned_articles": _dedupe([label for label in articles if label], max_items=12),
        "mentioned_clauses": _dedupe([label for label in clauses if label], max_items=12),
        "mentioned_points": _dedupe(points, max_items=12),
        "mentioned_chapters": _dedupe(chapters, max_items=12),
    }


def _canonical_numbered_labels(
    values: list[str],
    prefix: str,
    *,
    upper_suffix: bool = False,
    max_items: int = 12,
) -> list[str]:
    labels = []
    for value in values:
        label = _numbered_label(prefix, _strip_accents_for_match(value), upper_suffix=upper_suffix)
        if label:
            labels.append(label)
    return _dedupe(labels, max_items=max_items)


def _canonical_point_labels(values: list[str], *, max_items: int = 12) -> list[str]:
    labels = []
    for value in values:
        value_norm = _strip_accents_for_match(value)
        match = re.search(r"\bdiem\s+([a-z])\b", value_norm)
        if match:
            labels.append(f"Điểm {match.group(1).lower()}")
    return _dedupe(labels, max_items=max_items)


def _canonical_chapter_labels(values: list[str], *, max_items: int = 12) -> list[str]:
    labels = []
    for value in values:
        value_norm = _strip_accents_for_match(value)
        match = re.search(r"\bchuong\s+([ivxlcdm]+|\d+)\b", value_norm)
        if match:
            labels.append(f"Chương {match.group(1).upper()}")
    return _dedupe(labels, max_items=max_items)


def _canonical_law_codes(values: list[str], *, max_items: int = 8) -> list[str]:
    codes: list[str] = []
    for value in values:
        value_norm = _strip_accents_for_match(value)
        for match in re.finditer(
            r"\b\d{1,3}/\d{4}/qh\d+\b|\b\d{1,3}/vbhn-vpqh\b",
            value_norm,
        ):
            code = match.group(0)
            code = re.sub(r"qh", "QH", code, flags=re.IGNORECASE)
            code = re.sub(r"vbhn-vpqh", "VBHN-VPQH", code, flags=re.IGNORECASE)
            codes.append(code)
    return _dedupe(codes, max_items=max_items)


def _numbered_label(prefix: str, value: str, upper_suffix: bool = False) -> str | None:
    match = re.search(r"(\d+)([a-z]?)", value or "", flags=re.IGNORECASE)
    if not match:
        return None
    suffix = match.group(2).upper() if upper_suffix else match.group(2).lower()
    return f"{prefix} {int(match.group(1))}{suffix}"


def _strip_accents_for_match(text: str) -> str:
    import unicodedata

    text = str(text or "").lower().replace("đ", "d")
    text = "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )
    return re.sub(r"\s+", " ", text).strip()


def _extract_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    candidates = getattr(response, "candidates", None) or []
    parts_text: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if content is None:
            continue
        parts = getattr(content, "parts", None) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                parts_text.append(part_text.strip())
    return "\n".join(parts_text).strip()


def _strip_json_fence(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = _strip_json_fence(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        repaired = _repair_json_text(cleaned)
        try:
            parsed = json.loads(repaired)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(repaired)
            except (SyntaxError, ValueError) as repair_exc:
                raise QueryProfileProviderError(
                    f"Gemini query profile returned invalid JSON: {exc}"
                ) from repair_exc
    if not isinstance(parsed, dict):
        raise QueryProfileProviderError("Gemini query profile response is not a JSON object.")
    return parsed


def _repair_json_text(text: str) -> str:
    repaired = text.strip()
    repaired = re.sub(r"/\*.*?\*/", "", repaired, flags=re.DOTALL)
    repaired = re.sub(r"(^|[\s,{])//.*?(?=\n|$)", r"\1", repaired)
    repaired = re.sub(r"\bTrue\b", "true", repaired)
    repaired = re.sub(r"\bFalse\b", "false", repaired)
    repaired = re.sub(r"\bNone\b", "null", repaired)
    repaired = re.sub(
        r"'([^'\r\n]*)'",
        lambda match: json.dumps(match.group(1), ensure_ascii=False),
        repaired,
    )
    repaired = re.sub(
        r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)',
        r'\1"\2"\3',
        repaired,
    )
    repaired = re.sub(
        r'(:\s*)([A-Za-z_][A-Za-z0-9_\-]*)(\s*[,}\]])',
        lambda match: (
            match.group(0)
            if match.group(2) in {"true", "false", "null"}
            else f'{match.group(1)}"{match.group(2)}"{match.group(3)}'
        ),
        repaired,
    )
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    return repaired


def _default_intent_flags() -> dict[str, bool]:
    return {key: False for key in INTENT_FLAG_KEYS}


def _normalize_hard_entities(data: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    hard_entities = data.get("hard_entities")
    if not isinstance(hard_entities, dict):
        hard_entities = {}

    mentioned_law = _as_str_or_none(
        hard_entities.get("mentioned_law", data.get("mentioned_law"))
    )
    candidate_fields = _as_str_list(
        hard_entities.get("candidate_fields", data.get("candidate_fields")),
        max_items=8,
    )
    if mentioned_law and mentioned_law not in candidate_fields:
        candidate_fields.insert(0, mentioned_law)

    extracted_refs = _extract_labels_from_texts(
        _as_str_list(data.get("keywords"), max_items=24)
        + _as_str_list(data.get("legal_concepts"), max_items=24)
        + _as_str_list(data.get("retrieval_queries"), max_items=8)
        + _as_str_list(data.get("candidate_fields"), max_items=8)
    )

    raw_law_codes = (
        _as_str_list(hard_entities.get("mentioned_law_codes", data.get("mentioned_law_codes")), max_items=8)
        + extracted_refs["mentioned_law_codes"]
    )
    raw_articles = (
        _as_str_list(hard_entities.get("mentioned_articles", data.get("mentioned_articles")), max_items=12)
        + extracted_refs["mentioned_articles"]
    )
    raw_clauses = (
        _as_str_list(hard_entities.get("mentioned_clauses", data.get("mentioned_clauses")), max_items=12)
        + extracted_refs["mentioned_clauses"]
    )
    raw_points = (
        _as_str_list(hard_entities.get("mentioned_points", data.get("mentioned_points")), max_items=12)
        + extracted_refs["mentioned_points"]
    )
    raw_chapters = (
        _as_str_list(hard_entities.get("mentioned_chapters", data.get("mentioned_chapters")), max_items=12)
        + extracted_refs["mentioned_chapters"]
    )

    normalized = {
        "mentioned_law": mentioned_law,
        "candidate_fields": candidate_fields,
        "mentioned_law_codes": _canonical_law_codes(raw_law_codes),
        "mentioned_articles": _canonical_numbered_labels(
            raw_articles,
            "Điều",
            upper_suffix=True,
        ),
        "mentioned_clauses": _canonical_numbered_labels(raw_clauses, "Khoản"),
        "mentioned_points": _canonical_point_labels(raw_points),
        "mentioned_chapters": _canonical_chapter_labels(raw_chapters),
    }
    normalized["has_hard_legal_reference"] = bool(
        normalized["mentioned_law"]
        or normalized["mentioned_law_codes"]
        or normalized["mentioned_articles"]
        or normalized["mentioned_clauses"]
        or normalized["mentioned_points"]
        or normalized["mentioned_chapters"]
    )
    return normalized


def normalize_query_profile(data: dict[str, Any], raw_query: str) -> dict[str, Any]:
    query_type = _as_str_or_none(data.get("query_type")) or "general_explanation"
    if query_type not in LEGAL_QUERY_TYPES:
        query_type = "general_explanation"

    flags = _default_intent_flags()
    raw_flags = data.get("intent_flags")
    if isinstance(raw_flags, dict):
        for key in INTENT_FLAG_KEYS:
            flags[key] = _as_bool(raw_flags.get(key))

    keywords = _as_str_list(data.get("keywords"), max_items=24)
    legal_concepts = _as_str_list(data.get("legal_concepts"), max_items=24)
    keyword_pool = _as_str_list(keywords + legal_concepts, max_items=24)

    profile: dict[str, Any] = {
        "query_type": query_type,
        "needs_exact_article": _as_bool(data.get("needs_exact_article")),
        "keywords": keyword_pool,
        "intent_flags": flags,
        "confidence": _as_float(data.get("confidence"), default=0.0),
        "intent_source": "gemini_api",
        "profile_source": PROFILE_VERSION,
        "profile_version": PROFILE_VERSION,
        "profile_mode": "gemini_api",
        "intent_scores": data.get("intent_scores") if isinstance(data.get("intent_scores"), list) else [],
        "legal_concepts": legal_concepts,
        "retrieval_queries": _as_str_list(data.get("retrieval_queries"), max_items=8),
        "raw_query": raw_query,
    }

    hard_entities = _normalize_hard_entities(data, profile)
    profile.update({
        "mentioned_law": hard_entities["mentioned_law"],
        "candidate_fields": hard_entities["candidate_fields"],
        "hard_entities": hard_entities,
        "mentioned_law_codes": hard_entities["mentioned_law_codes"],
        "mentioned_articles": hard_entities["mentioned_articles"],
        "mentioned_clauses": hard_entities["mentioned_clauses"],
        "mentioned_points": hard_entities["mentioned_points"],
        "mentioned_chapters": hard_entities["mentioned_chapters"],
    })

    return profile


def build_query_profile(query: str, **kwargs: Any) -> dict[str, Any]:
    return QueryProfiler(**kwargs).profile(query)


def extract_hard_entities(query: str, **kwargs: Any) -> dict[str, Any]:
    return build_query_profile(query, **kwargs)["hard_entities"]


class QueryProfiler:
    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        client: Any | None = None,
        logger: Any | None = None,
        **_: Any,
    ):
        self.model_name = model_name or DEFAULT_QUERY_PROFILE_MODEL_NAME
        resolved_api_key = os.getenv("GEMINI_API_KEY") if api_key is None else api_key
        self.api_key = (resolved_api_key or "").strip()
        self.temperature = (
            DEFAULT_QUERY_PROFILE_TEMPERATURE
            if temperature is None
            else float(temperature)
        )
        self.max_output_tokens = int(
            max_output_tokens or DEFAULT_QUERY_PROFILE_MAX_OUTPUT_TOKENS
        )
        self.client = client
        self.logger = logger

    def profile(self, query: str) -> dict[str, Any]:
        raw_query = (query or "").strip()
        if not raw_query:
            raise QueryProfileProviderError("Question is empty.")

        response_text = self._call_gemini(self._build_prompt(raw_query))
        try:
            data = _parse_json_object(response_text)
        except QueryProfileProviderError:
            self._log_warning(
                "Gemini query profile returned invalid JSON. Retrying once | raw_response=%s",
                self._truncate_for_log(response_text),
            )
            response_text = self._call_gemini(self._build_repair_prompt(raw_query, response_text))
            try:
                data = _parse_json_object(response_text)
            except QueryProfileProviderError:
                self._log_warning(
                    "Gemini query profile retry still returned invalid JSON | raw_response=%s",
                    self._truncate_for_log(response_text),
                )
                raise
        profile = normalize_query_profile(data, raw_query=raw_query)
        if self.model_name:
            profile["intent_model"] = self.model_name
        return profile

    def _call_gemini(self, prompt: str) -> str:
        client = self._get_client()
        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self._build_generation_config(),
            )
        except Exception as exc:
            raise QueryProfileProviderError(
                f"Gemini query profile API failed: {exc}",
                error_code=self._provider_error_code(exc),
            ) from exc

        text = _extract_text(response)
        if not text:
            raise QueryProfileProviderError("Gemini query profile returned empty content.")
        return text

    def _get_client(self):
        if self.client is not None:
            return self.client
        if genai is None:
            raise QueryProfileProviderError(
                "google-genai is not installed. Add it to requirements and reinstall dependencies."
            ) from _IMPORT_ERROR
        if not self.api_key:
            raise QueryProfileProviderError(
                "Missing GEMINI_API_KEY. Please set GEMINI_API_KEY in your .env file.",
                error_code="query_profile_missing_api_key",
            )
        self.client = genai.Client(api_key=self.api_key)
        return self.client

    def _build_generation_config(self):
        if types is None:
            return None
        kwargs = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }
        response_schema = self._build_response_schema()
        if response_schema is not None:
            kwargs["response_schema"] = response_schema
        safety_settings = self._build_safety_settings()
        if safety_settings:
            kwargs["safety_settings"] = safety_settings
        try:
            return types.GenerateContentConfig(
                **kwargs,
                response_mime_type="application/json",
            )
        except TypeError:
            return types.GenerateContentConfig(**kwargs)

    def _build_response_schema(self):
        if types is None or not hasattr(types, "Schema") or not hasattr(types, "Type"):
            return None

        schema_type = types.Type
        string_array = types.Schema(
            type=schema_type.ARRAY,
            items=types.Schema(type=schema_type.STRING),
        )
        return types.Schema(
            type=schema_type.OBJECT,
            properties={
                "query_type": types.Schema(
                    type=schema_type.STRING,
                    enum=LEGAL_QUERY_TYPES,
                ),
                "needs_exact_article": types.Schema(type=schema_type.BOOLEAN),
                "mentioned_law": types.Schema(type=schema_type.STRING, nullable=True),
                "candidate_fields": string_array,
                "keywords": string_array,
                "confidence": types.Schema(type=schema_type.NUMBER),
                "mentioned_law_codes": string_array,
                "mentioned_articles": string_array,
                "mentioned_clauses": string_array,
                "mentioned_points": string_array,
                "mentioned_chapters": string_array,
                "legal_concepts": string_array,
                "retrieval_queries": string_array,
            },
            required=[
                "query_type",
                "needs_exact_article",
                "mentioned_law",
                "candidate_fields",
                "keywords",
                "confidence",
                "mentioned_law_codes",
                "mentioned_articles",
                "mentioned_clauses",
                "mentioned_points",
                "mentioned_chapters",
                "legal_concepts",
                "retrieval_queries",
            ],
            property_ordering=[
                "query_type",
                "needs_exact_article",
                "mentioned_law",
                "candidate_fields",
                "keywords",
                "confidence",
                "mentioned_law_codes",
                "mentioned_articles",
                "mentioned_clauses",
                "mentioned_points",
                "mentioned_chapters",
                "legal_concepts",
                "retrieval_queries",
            ],
        )

    def _build_safety_settings(self) -> list[Any]:
        if types is None or not hasattr(types, "SafetySetting"):
            return []
        settings = []
        for category in (
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
        ):
            settings.append(
                types.SafetySetting(
                    category=category,
                    threshold="BLOCK_NONE",
                )
            )
        return settings

    def _provider_error_code(self, exc: Exception) -> str:
        status_code = self._extract_error_status_code(exc)
        if status_code in {429, 503}:
            return "query_profile_provider_busy"
        text = " ".join(str(exc).lower().split())
        busy_markers = (
            "high demand",
            "try again later",
            "temporarily unavailable",
            "service unavailable",
            "overloaded",
            "resource exhausted",
            "unavailable",
        )
        if any(marker in text for marker in busy_markers):
            return "query_profile_provider_busy"
        return "query_profile_provider_error"

    def _extract_error_status_code(self, exc: Exception) -> int | None:
        for attr in ("status_code", "code"):
            parsed = self._parse_int(getattr(exc, attr, None))
            if parsed is not None:
                return parsed

        response = getattr(exc, "response", None)
        if response is not None:
            for attr in ("status_code", "status"):
                parsed = self._parse_int(getattr(response, attr, None))
                if parsed is not None:
                    return parsed

        match = re.search(r"\b(400|401|403|408|409|429|500|502|503|504)\b", str(exc))
        if match:
            return int(match.group(1))
        return None

    def _parse_int(self, value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _truncate_for_log(self, value: str, limit: int = 1200) -> str:
        text = " ".join((value or "").split())
        if len(text) <= limit:
            return text
        return f"{text[:limit]}..."

    def _log_warning(self, message: str, *args) -> None:
        if self.logger is not None and hasattr(self.logger, "warning"):
            self.logger.warning(message, *args)

    def _build_repair_prompt(self, query: str, previous_response: str) -> str:
        return f"""
Ignore the previous invalid output. Return one compact valid JSON object only.
No markdown. No comments. No trailing commas.
All keys and string values must use double quotes.
Use lowercase JSON literals: true, false, null.
Keep arrays short.

Required JSON keys:
query_type, needs_exact_article, mentioned_law, candidate_fields, keywords,
confidence, mentioned_law_codes, mentioned_articles, mentioned_clauses,
mentioned_points, mentioned_chapters, legal_concepts, retrieval_queries.

Question:
{query}
""".strip()

    def _build_prompt(self, query: str) -> str:
        query_types = ", ".join(LEGAL_QUERY_TYPES)
        return f"""
You are the query profiling component of a Vietnamese legal RAG system.

Task:
- Read the user question.
- Return one strict JSON object only.
- Do not answer the legal question.
- Do not cite laws unless the user explicitly mentions them or they are highly likely concepts for retrieval.
- Prefer normalized Vietnamese legal terms in keywords/legal_concepts to improve retrieval.
- If the user uses colloquial facts, convert them into legal retrieval concepts.

Allowed query_type values:
{query_types}

Return exactly these JSON keys:
- query_type: one allowed query_type
- needs_exact_article: boolean
- mentioned_law: string or null
- candidate_fields: short string array
- keywords: short string array
- confidence: number from 0 to 1
- mentioned_law_codes: string array
- mentioned_articles: string array
- mentioned_clauses: string array
- mentioned_points: string array
- mentioned_chapters: string array
- legal_concepts: short string array
- retrieval_queries: 1 to 3 short Vietnamese retrieval queries

Guidance:
- Use article_lookup when the user asks whether a rule applies, asks yes/no, asks a concrete legal consequence, asks an exact threshold, or asks where/how law regulates the issue.
- Use penalty_lookup for sanctions, fines, criminal punishment, imprisonment, or liability level.
- Use procedure_lookup for authority, documents, procedure, process.
- Use definition_lookup for definitions and included subjects/classes.
- Keep keywords and legal_concepts generalizable; do not overfit any dataset.

User question:
{query}
""".strip()
