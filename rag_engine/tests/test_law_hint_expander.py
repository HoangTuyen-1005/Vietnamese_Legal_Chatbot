from rag_engine.retrieval.law_hint_expander import (
    _hint_score,
    _resolve_so_hieu,
    augment_with_law_hints,
)


LAND_LAW_2024 = "Lu\u1eadt \u0110\u1ea5t \u0111ai n\u0103m 2024"
LAND_USER_TERM = "ng\u01b0\u1eddi s\u1eed d\u1ee5ng \u0111\u1ea5t"


class FakeVectorStore:
    def get_law_catalog(self) -> list[dict]:
        return [
            {
                "so_hieu": "31/2024/QH15",
                "alias_text": f"{LAND_LAW_2024} Luat_dat_dai_2024.pdf",
            }
        ]

    def get_chunks_by_metadata(self, so_hieu: str, limit: int = 300) -> list[dict]:
        assert so_hieu == "31/2024/QH15"
        assert limit == 300
        return [
            {
                "chunk_id": "land_article_4",
                "content": (
                    f"{LAND_USER_TERM} bao g\u1ed3m t\u1ed5 ch\u1ee9c trong n\u01b0\u1edbc, "
                    "c\u00e1 nh\u00e2n trong n\u01b0\u1edbc v\u00e0 c\u00e1c ch\u1ee7 th\u1ec3 kh\u00e1c."
                ),
                "metadata": {
                    "so_hieu": "31/2024/QH15",
                    "dieu": "\u0110i\u1ec1u 4",
                    "ten_dieu": LAND_USER_TERM,
                    "cap_chunk": "dieu",
                },
            },
            {
                "chunk_id": "land_article_137",
                "content": (
                    "C\u1ea5p Gi\u1ea5y ch\u1ee9ng nh\u1eadn quy\u1ec1n s\u1eed d\u1ee5ng "
                    "\u0111\u1ea5t cho h\u1ed9 gia \u0111\u00ecnh, c\u00e1 nh\u00e2n."
                ),
                "metadata": {
                    "so_hieu": "31/2024/QH15",
                    "dieu": "\u0110i\u1ec1u 137",
                    "ten_dieu": "C\u1ea5p Gi\u1ea5y ch\u1ee9ng nh\u1eadn quy\u1ec1n s\u1eed d\u1ee5ng \u0111\u1ea5t",
                    "cap_chunk": "dieu",
                },
            },
        ]


def test_resolve_so_hieu_from_law_catalog_aliases():
    catalog = FakeVectorStore().get_law_catalog()

    assert _resolve_so_hieu(LAND_LAW_2024, catalog) == "31/2024/QH15"
    assert _resolve_so_hieu("31/2024/QH15", catalog) == "31/2024/QH15"


def test_article_title_hint_outranks_broad_keyword_overlap():
    profile = {
        "query_type": "definition_lookup",
        "confidence": 0.9,
        "mentioned_law": LAND_LAW_2024,
        "keywords": [LAND_USER_TERM],
        "legal_concepts": [LAND_USER_TERM],
        "retrieval_queries": [f"{LAND_USER_TERM} \u0110i\u1ec1u 4 {LAND_LAW_2024}"],
    }
    chunks = FakeVectorStore().get_chunks_by_metadata("31/2024/QH15")

    article_4_score = _hint_score(profile, chunks[0])
    article_137_score = _hint_score(profile, chunks[1])

    assert article_4_score > article_137_score
    assert article_4_score > 0


def test_augment_with_law_hints_adds_matching_article_title_candidate():
    profile = {
        "query_type": "definition_lookup",
        "confidence": 0.9,
        "mentioned_law": LAND_LAW_2024,
        "candidate_fields": [LAND_LAW_2024],
        "keywords": [LAND_USER_TERM],
        "legal_concepts": [LAND_USER_TERM],
        "retrieval_queries": [f"{LAND_USER_TERM} \u0110i\u1ec1u 4 {LAND_LAW_2024}"],
    }
    retrieved = [
        {
            "chunk_id": "unrelated_initial_candidate",
            "content": "N\u1ed9i dung kh\u00f4ng li\u00ean quan.",
            "score": 1.0,
            "metadata": {"so_hieu": "91/2015/QH13", "dieu": "\u0110i\u1ec1u 260"},
        }
    ]

    augmented = augment_with_law_hints(
        retrieved=retrieved,
        query_profile=profile,
        vector_store=FakeVectorStore(),
        add_k=2,
    )

    by_id = {chunk["chunk_id"]: chunk for chunk in augmented}
    assert "unrelated_initial_candidate" in by_id
    assert by_id["land_article_4"]["source"] == "law_hint"
