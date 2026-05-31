from types import SimpleNamespace

from rag_engine.services.chat_service import GENERATION_PROVIDER_BUSY_ANSWER
from rag_engine.services.chat_service import ChatService
from rag_engine.retrieval.query_profile import (
    QUERY_PROFILE_PROVIDER_ERROR_ANSWER,
    QueryProfileProviderError,
)


def make_service(max_chunks: int = 2) -> ChatService:
    return ChatService(
        bm25_store=None,
        vector_store=None,
        reranker=None,
        generator=None,
        settings=SimpleNamespace(
            MAX_CONTEXT_CHUNKS=max_chunks,
            MAX_CONTEXT_CHARS=10000,
        ),
        logger=None,
    )


def make_chunk(chunk_id: str, role: str | None = None) -> dict:
    chunk = {
        "chunk_id": chunk_id,
        "content": "Can thiệp sớm được áp dụng trong trường hợp sau đây.",
        "metadata": {
            "so_hieu": "32/2024/QH15",
            "dieu": "Điều 156",
            "ten_dieu": "Can thiệp sớm",
        },
    }
    if role:
        chunk["_context_role"] = role
    return chunk


class NullLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


class FakeBM25Store:
    def search(self, query: str, top_k: int = 10) -> list[dict]:
        return [
            {
                "chunk_id": "adult",
                "content": "Người từ đủ mười tám tuổi trở lên là người thành niên.",
                "metadata": {
                    "so_hieu": "91/2015/QH13",
                    "loai_van_ban": "Bộ luật",
                    "dieu": "Điều 20",
                    "khoan": "Khoản 1",
                    "ten_dieu": "Người thành niên",
                },
                "score": 1.0,
                "source": "bm25",
            }
        ]


class FakeVectorStore:
    def search(self, query: str, top_k: int = 10) -> list[dict]:
        return []


class FakeReranker:
    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        item = dict(candidates[0])
        item["rerank_score"] = 1.0
        return [item]

    def expand_legal_context(self, top_chunks: list[dict], vector_store, **kwargs) -> list[dict]:
        return top_chunks


class BusyGenerator:
    def generate(self, prompt: str, max_new_tokens: int) -> str:
        raise RuntimeError(
            "503 This model is currently experiencing high demand. "
            "Spikes in demand are usually temporary. Please try again later."
        )


class FakeQueryProfiler:
    def profile(self, query: str) -> dict:
        return {
            "query_type": "definition_lookup",
            "needs_exact_article": False,
            "mentioned_law": None,
            "candidate_fields": [],
            "keywords": ["nguoi thanh nien"],
            "intent_flags": {},
            "confidence": 0.9,
            "raw_query": query,
        }


class FailingQueryProfiler:
    def profile(self, query: str) -> dict:
        raise QueryProfileProviderError(
            "503 This model is currently experiencing high demand.",
            error_code="query_profile_provider_busy",
        )


def make_full_service(generator) -> ChatService:
    return ChatService(
        bm25_store=FakeBM25Store(),
        vector_store=FakeVectorStore(),
        reranker=FakeReranker(),
        generator=generator,
        settings=SimpleNamespace(
            TOP_K_BM25=1,
            TOP_K_VECTOR=0,
            TOP_K_RERANK=1,
            RRF_K=60,
            FOLLOW_REFERENCED_DIEU=True,
            MAX_REFERENCED_DIEU_PER_CHUNK=3,
            MAX_REFERENCED_DIEU_TOTAL=12,
            MAX_CONTEXT_CHUNKS=4,
            MAX_CONTEXT_CHARS=10000,
            GENERATOR_MODEL_NAME="gemini-test",
            MAX_NEW_TOKENS=256,
            RETRY_INCOMPLETE_ANSWER=True,
            MIN_COMPLETE_ANSWER_CHARS=120,
        ),
        logger=NullLogger(),
        query_profiler=FakeQueryProfiler(),
    )


def test_prompt_context_limit_keeps_reranked_and_referenced_chunks():
    service = make_service(max_chunks=2)
    reranked = [make_chunk("reranked", "reranked")]
    contexts = [
        make_chunk("low"),
        make_chunk("referenced", "referenced_dieu"),
        make_chunk("same", "same_dieu"),
        reranked[0],
    ]

    selected = service._limit_contexts_for_prompt(
        contexts=contexts,
        reranked_chunks=reranked,
        query_profile={"keywords": ["can thiệp sớm"]},
    )

    assert [chunk["chunk_id"] for chunk in selected] == ["reranked", "same"]


def test_generation_provider_busy_returns_frontend_safe_payload():
    service = make_full_service(generator=BusyGenerator())

    result = service.answer_question("Người thành niên là gì?")

    assert result["answer"] == GENERATION_PROVIDER_BUSY_ANSWER
    assert result["error_code"] == "generation_provider_busy"
    assert "quá tải" in result["error_message"]
    assert result["sources"]


def test_query_profile_provider_error_returns_frontend_safe_payload():
    service = make_full_service(generator=BusyGenerator())
    service.query_profiler = FailingQueryProfiler()

    result = service.answer_question("Nguoi thanh nien la gi?")

    assert result["answer"] == QUERY_PROFILE_PROVIDER_ERROR_ANSWER
    assert result["error_code"] == "query_profile_provider_busy"
    assert result["sources"] == []
    assert result["retrieved_count"] == 0
    assert result["reranked_count"] == 0
