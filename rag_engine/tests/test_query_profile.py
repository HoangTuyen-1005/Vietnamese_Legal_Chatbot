import json

from rag_engine.retrieval.query_profile import (
    LEGAL_QUERY_TYPES,
    QueryProfileProviderError,
    QueryProfiler,
    build_query_profile,
    extract_hard_entities,
)


PROFILE_PAYLOAD = {
    "query_type": "definition_lookup",
    "needs_exact_article": True,
    "mentioned_law": "Luat Dat dai 2024",
    "candidate_fields": ["Luat Dat dai 2024"],
    "keywords": ["nguoi su dung dat", "doi tuong su dung dat"],
    "intent_flags": {
        "asks_definition": True,
        "asks_article_location": False,
        "asks_principle": False,
        "asks_prohibited": False,
        "asks_conditions": False,
        "asks_cases": False,
        "asks_penalty": False,
        "asks_procedure": False,
        "asks_duration": False,
        "asks_yes_no": False,
    },
    "confidence": 0.87,
    "hard_entities": {
        "mentioned_law": "Luat Dat dai 2024",
        "candidate_fields": ["Luat Dat dai 2024"],
        "mentioned_law_codes": ["31/2024/QH15"],
        "mentioned_articles": ["Dieu 4"],
        "mentioned_clauses": [],
        "mentioned_points": [],
        "mentioned_chapters": [],
        "has_hard_legal_reference": True,
    },
    "mentioned_law_codes": ["31/2024/QH15"],
    "mentioned_articles": ["Dieu 4"],
    "mentioned_clauses": [],
    "mentioned_points": [],
    "mentioned_chapters": [],
    "legal_concepts": ["nguoi su dung dat"],
    "retrieval_queries": ["nguoi su dung dat Dieu 4 Luat Dat dai 2024"],
}


class FakeResponse:
    def __init__(self, text: str):
        self.text = text


class FakeModels:
    def __init__(self, text: str):
        self.text = text
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(self.text)


class FakeClient:
    def __init__(self, text: str):
        self.models = FakeModels(text)


def test_query_profiler_uses_gemini_client_and_normalizes_profile():
    client = FakeClient(json.dumps(PROFILE_PAYLOAD))
    profiler = QueryProfiler(
        model_name="gemini-test",
        client=client,
    )

    profile = profiler.profile("Nguoi su dung dat bao gom nhung ai?")

    assert profile["query_type"] == "definition_lookup"
    assert profile["query_type"] in LEGAL_QUERY_TYPES
    assert profile["profile_source"] == "gemini_v1"
    assert profile["profile_mode"] == "gemini_api"
    assert profile["intent_source"] == "gemini_api"
    assert profile["mentioned_law"] == "Luat Dat dai 2024"
    assert profile["mentioned_law_codes"] == ["31/2024/QH15"]
    assert profile["mentioned_articles"] == ["Điều 4"]
    assert profile["hard_entities"]["has_hard_legal_reference"] is True
    assert profile["keywords"] == ["nguoi su dung dat", "doi tuong su dung dat"]
    assert profile["intent_flags"]["asks_definition"] is True
    assert profile["confidence"] == 0.87
    assert client.models.calls[0]["model"] == "gemini-test"
    assert client.models.calls[0]["config"] is not None


def test_build_query_profile_accepts_injected_client():
    profile = build_query_profile(
        "Nguoi su dung dat bao gom nhung ai?",
        model_name="gemini-test",
        client=FakeClient(json.dumps(PROFILE_PAYLOAD)),
    )

    assert profile["profile_source"] == "gemini_v1"
    assert profile["retrieval_queries"]


def test_extract_hard_entities_reads_from_gemini_profile():
    entities = extract_hard_entities(
        "Dieu 4 Luat Dat dai 2024 quy dinh gi?",
        model_name="gemini-test",
        client=FakeClient(json.dumps(PROFILE_PAYLOAD)),
    )

    assert entities["mentioned_law"] == "Luat Dat dai 2024"
    assert entities["mentioned_law_codes"] == ["31/2024/QH15"]
    assert entities["mentioned_articles"] == ["Điều 4"]


def test_query_profiler_repairs_common_json_format_errors():
    payload = json.dumps(PROFILE_PAYLOAD)
    payload = payload.replace('"query_type"', "query_type", 1)
    payload = payload.replace('"needs_exact_article"', "needs_exact_article", 1)
    payload = payload.replace('"mentioned_law"', "mentioned_law", 1)
    payload = payload.replace('"definition_lookup"', "definition_lookup", 1)
    payload = payload.replace("true", "True", 1)
    payload = payload[:-1] + ",}"

    profile = QueryProfiler(
        model_name="gemini-test",
        client=FakeClient(payload),
    ).profile("Nguoi su dung dat bao gom nhung ai?")

    assert profile["query_type"] == "definition_lookup"
    assert profile["mentioned_law"] == "Luat Dat dai 2024"


def test_invalid_json_raises_provider_error():
    profiler = QueryProfiler(
        model_name="gemini-test",
        client=FakeClient("not json"),
    )

    try:
        profiler.profile("Cau hoi bat ky")
    except QueryProfileProviderError:
        return
    raise AssertionError("Expected QueryProfileProviderError.")


def test_missing_api_key_raises_provider_error():
    profiler = QueryProfiler(model_name="gemini-test", api_key="")

    try:
        profiler.profile("Cau hoi bat ky")
    except QueryProfileProviderError as exc:
        assert exc.error_code == "query_profile_missing_api_key"
        return

    raise AssertionError("Expected QueryProfileProviderError.")
