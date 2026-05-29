from rag_engine.retrieval.query_profile import build_query_profile


def test_detects_law_name_before_punctuation_and_when_question():
    profile = build_query_profile(
        "Theo Luật các Tổ chức tín dụng 2024, "
        "biện pháp Can thiệp sớm được áp dụng khi nào?"
    )

    assert profile["query_type"] == "cases_circumstances"
    assert profile["mentioned_law"] == "luật các tổ chức tín dụng 2024"
