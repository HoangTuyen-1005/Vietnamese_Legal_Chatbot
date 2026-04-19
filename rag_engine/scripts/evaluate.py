import json
from pathlib import Path

from rag_engine.core.config import get_settings
from rag_engine.core.logger import setup_logger
from rag_engine.generation.generator import AnswerGenerator
from shared.bm25_store import BM25Store
from rag_engine.retrieval.hybrid_search import hybrid_search
from rag_engine.retrieval.law_hint_expander import augment_with_law_hints
from rag_engine.retrieval.metadata_rescorer import rescore_candidates
from rag_engine.retrieval.query_profile import build_query_profile
from rag_engine.retrieval.refusal_policy import is_refusal_answer, should_refuse_after_retrieval
from rag_engine.retrieval.reranker import Reranker
from shared.vector_store import VectorStore
from rag_engine.services.chat_service import ChatService

SOURCE_FIELDS = ("so_hieu", "dieu", "khoan", "diem")


def load_eval_questions(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def expected_has_constraints(expected: dict) -> bool:
    return any(expected.get(field) is not None for field in SOURCE_FIELDS)


def normalize_expected_sources(expected_sources: list[dict]) -> list[dict]:
    return [item for item in expected_sources if expected_has_constraints(item)]


def source_matches(expected: dict, actual: dict) -> bool:
    if not expected_has_constraints(expected):
        return False

    for field in SOURCE_FIELDS:
        expected_value = expected.get(field)
        if expected_value is None:
            continue
        if actual.get(field) != expected_value:
            return False
    return True


def has_match_in_top_k(expected_sources: list[dict], actual_sources: list[dict], k: int) -> bool:
    valid_expected = normalize_expected_sources(expected_sources)
    if not valid_expected:
        return False

    top_sources = actual_sources[:k]
    for expected in valid_expected:
        for actual in top_sources:
            if source_matches(expected, actual):
                return True
    return False


def evaluate_retrieval_only(questions: list[dict], bm25_store, vector_store, settings) -> dict:
    total = len(questions)
    in_scope_total = 0
    refusal_total = 0
    invalid_expected_total = 0

    top1_hits = 0
    top3_hits = 0
    refusal_correct = 0

    details = []

    for item in questions:
        question = item["question"]
        expected_sources = item.get("expected_sources", [])
        should_refuse = item.get("should_refuse", False)

        query_profile = build_query_profile(question)
        candidate_pool_k = settings.TOP_K_BM25 + settings.TOP_K_VECTOR

        retrieved = hybrid_search(
            query=question,
            bm25_store=bm25_store,
            vector_store=vector_store,
            top_k_bm25=settings.TOP_K_BM25,
            top_k_vector=settings.TOP_K_VECTOR,
            top_k_final=candidate_pool_k,
            rrf_k=settings.RRF_K,
        )

        retrieved = augment_with_law_hints(
            retrieved=retrieved,
            query_profile=query_profile,
            vector_store=vector_store,
            add_k=max(6, settings.TOP_K_VECTOR // 2),
        )

        retrieval_refusal = should_refuse_after_retrieval(
            retrieved,
            query_profile=query_profile,
        )

        retrieved = rescore_candidates(
            candidates=retrieved,
            query_profile=query_profile,
            top_k=candidate_pool_k,
        )

        actual_sources = []
        for chunk in retrieved:
            meta = chunk.get("metadata", {})
            actual_sources.append({
                "so_hieu": meta.get("so_hieu"),
                "dieu": meta.get("dieu"),
                "khoan": meta.get("khoan"),
                "diem": meta.get("diem"),
            })

        q_top1 = False
        q_top3 = False
        q_refusal = False
        invalid_expected = False

        if should_refuse:
            refusal_total += 1
            q_refusal = retrieval_refusal
            if q_refusal:
                refusal_correct += 1
        else:
            valid_expected = normalize_expected_sources(expected_sources)
            if not valid_expected:
                invalid_expected = True
                invalid_expected_total += 1
            else:
                in_scope_total += 1
                q_top1 = has_match_in_top_k(valid_expected, actual_sources, k=1)
                q_top3 = has_match_in_top_k(valid_expected, actual_sources, k=3)
                if q_top1:
                    top1_hits += 1
                if q_top3:
                    top3_hits += 1

        details.append({
            "id": item["id"],
            "question": question,
            "top1_hit": q_top1,
            "top3_hit": q_top3,
            "refusal_correct": q_refusal,
            "invalid_expected": invalid_expected,
            "actual_sources": actual_sources[:3],
        })

    return {
        "mode": "retrieval_only",
        "total": total,
        "in_scope_total": in_scope_total,
        "refusal_total": refusal_total,
        "invalid_expected_total": invalid_expected_total,
        "top1_accuracy": top1_hits / in_scope_total if in_scope_total else 0.0,
        "top3_accuracy": top3_hits / in_scope_total if in_scope_total else 0.0,
        "refusal_accuracy": refusal_correct / refusal_total if refusal_total else 0.0,
        "details": details,
    }


def evaluate_full_pipeline(questions: list[dict], chat_service) -> dict:
    total = len(questions)
    in_scope_total = 0
    refusal_total = 0
    invalid_expected_total = 0

    top1_hits = 0
    top3_hits = 0
    refusal_correct = 0

    details = []

    for item in questions:
        question = item["question"]
        expected_sources = item.get("expected_sources", [])
        should_refuse = item.get("should_refuse", False)

        result = chat_service.answer_question(question)
        actual_sources = result.get("sources", [])
        answer = result.get("answer", "")

        q_top1 = False
        q_top3 = False
        q_refusal = False
        invalid_expected = False

        if should_refuse:
            refusal_total += 1
            q_refusal = is_refusal_answer(answer)
            if q_refusal:
                refusal_correct += 1
        else:
            valid_expected = normalize_expected_sources(expected_sources)
            if not valid_expected:
                invalid_expected = True
                invalid_expected_total += 1
            else:
                in_scope_total += 1
                q_top1 = has_match_in_top_k(valid_expected, actual_sources, k=1)
                q_top3 = has_match_in_top_k(valid_expected, actual_sources, k=3)
                if q_top1:
                    top1_hits += 1
                if q_top3:
                    top3_hits += 1

        details.append({
            "id": item["id"],
            "question": question,
            "top1_hit": q_top1,
            "top3_hit": q_top3,
            "refusal_correct": q_refusal,
            "invalid_expected": invalid_expected,
            "answer_preview": answer[:300],
            "actual_sources": actual_sources[:3],
        })

    return {
        "mode": "full_pipeline",
        "total": total,
        "in_scope_total": in_scope_total,
        "refusal_total": refusal_total,
        "invalid_expected_total": invalid_expected_total,
        "top1_accuracy": top1_hits / in_scope_total if in_scope_total else 0.0,
        "top3_accuracy": top3_hits / in_scope_total if in_scope_total else 0.0,
        "refusal_accuracy": refusal_correct / refusal_total if refusal_total else 0.0,
        "details": details,
    }


if __name__ == "__main__":
    settings = get_settings()
    logger = setup_logger()

    eval_path = settings.EVAL_QUESTIONS_PATH
    if not Path(eval_path).exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_path}")

    questions = load_eval_questions(eval_path)

    bm25_store = BM25Store()
    bm25_store.load(
        index_path=settings.BM25_INDEX_PATH,
        docs_path=settings.BM25_DOCS_PATH,
    )

    vector_store = VectorStore(settings=settings)
    reranker = Reranker(model_name=settings.RERANKER_MODEL_NAME)

    retrieval_report = evaluate_retrieval_only(
        questions=questions,
        bm25_store=bm25_store,
        vector_store=vector_store,
        settings=settings,
    )

    print("=== RETRIEVAL ONLY REPORT ===")
    print(json.dumps(retrieval_report, ensure_ascii=False, indent=2))

    run_full_pipeline = False

    if run_full_pipeline:
        generator = None
        if settings.ENABLE_GENERATION:
            generator = AnswerGenerator(
                model_name=settings.GENERATOR_MODEL_NAME,
                api_key=settings.GEMINI_API_KEY,
                temperature=settings.GEMINI_TEMPERATURE,
            )

        chat_service = ChatService(
            bm25_store=bm25_store,
            vector_store=vector_store,
            reranker=reranker,
            generator=generator,
            settings=settings,
            logger=logger,
        )

        full_report = evaluate_full_pipeline(
            questions=questions,
            chat_service=chat_service,
        )

        print("\n=== FULL PIPELINE REPORT ===")
        print(json.dumps(full_report, ensure_ascii=False, indent=2))
