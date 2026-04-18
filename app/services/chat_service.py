import time
import unicodedata

from app.generation.prompt_builder import build_legal_prompt
from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.law_hint_expander import augment_with_law_hints
from app.retrieval.metadata_rescorer import rescore_candidates
from app.retrieval.query_profile import build_query_profile
from app.retrieval.refusal_policy import (
    GENERATION_DISABLED_ANSWER,
    REFUSAL_ANSWER,
    should_refuse_after_rerank,
    should_refuse_after_retrieval,
)


class ChatService:
    def __init__(
        self,
        bm25_store,
        vector_store,
        reranker,
        generator,
        settings,
        logger,
    ):
        self.bm25_store = bm25_store
        self.vector_store = vector_store
        self.reranker = reranker
        self.generator = generator
        self.settings = settings
        self.logger = logger

    def answer_question(self, question: str) -> dict:
        start_time = time.perf_counter()
        self.logger.info(f"Received question: {question}")

        profile_start = time.perf_counter()
        query_profile = build_query_profile(question)
        profile_ms = (time.perf_counter() - profile_start) * 1000
        self.logger.info(
            f"Query profile built in {profile_ms:.2f} ms | profile={query_profile}"
        )

        retrieval_start = time.perf_counter()
        candidate_pool_k = self.settings.TOP_K_BM25 + self.settings.TOP_K_VECTOR

        retrieved_chunks = hybrid_search(
            query=question,
            bm25_store=self.bm25_store,
            vector_store=self.vector_store,
            top_k_bm25=self.settings.TOP_K_BM25,
            top_k_vector=self.settings.TOP_K_VECTOR,
            top_k_final=candidate_pool_k,
            rrf_k=self.settings.RRF_K,
        )

        retrieved_chunks = augment_with_law_hints(
            retrieved=retrieved_chunks,
            query_profile=query_profile,
            vector_store=self.vector_store,
            add_k=max(6, self.settings.TOP_K_VECTOR // 2),
        )

        if self._should_refuse_answer(retrieved_chunks, query_profile):
            latency_ms = (time.perf_counter() - start_time) * 1000
            return {
                "question": question,
                "answer": REFUSAL_ANSWER,
                "sources": [],
                "retrieved_count": len(retrieved_chunks),
                "reranked_count": 0,
                "latency_ms": latency_ms,
            }

        retrieved_chunks = rescore_candidates(
            candidates=retrieved_chunks,
            query_profile=query_profile,
            top_k=candidate_pool_k,
        )

        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
        self.logger.info(
            f"Retrieved/rescored {len(retrieved_chunks)} chunks in {retrieval_ms:.2f} ms"
        )

        rerank_start = time.perf_counter()
        reranked_chunks = self.reranker.rerank(
            query=question,
            candidates=retrieved_chunks,
            top_k=self.settings.TOP_K_RERANK,
        )
        rerank_ms = (time.perf_counter() - rerank_start) * 1000
        self.logger.info(
            f"Reranked to {len(reranked_chunks)} chunks in {rerank_ms:.2f} ms"
        )

        if self._should_refuse_after_rerank(reranked_chunks):
            latency_ms = (time.perf_counter() - start_time) * 1000
            return {
                "question": question,
                "answer": REFUSAL_ANSWER,
                "sources": [],
                "retrieved_count": len(retrieved_chunks),
                "reranked_count": len(reranked_chunks),
                "latency_ms": latency_ms,
            }

        context_start = time.perf_counter()
        expanded_contexts = self.reranker.expand_legal_context(
            top_chunks=reranked_chunks,
            vector_store=self.vector_store,
            follow_referenced_dieu=self.settings.FOLLOW_REFERENCED_DIEU,
            max_referenced_dieu_per_chunk=self.settings.MAX_REFERENCED_DIEU_PER_CHUNK,
            max_referenced_dieu_total=self.settings.MAX_REFERENCED_DIEU_TOTAL,
        )
        context_ms = (time.perf_counter() - context_start) * 1000
        self.logger.info(
            f"Expanded to {len(expanded_contexts)} context chunks in {context_ms:.2f} ms"
        )

        if self.generator is None:
            self.logger.info("Generation is disabled. Returning retrieval-only result.")
            latency_ms = (time.perf_counter() - start_time) * 1000
            return {
                "question": question,
                "answer": GENERATION_DISABLED_ANSWER,
                "sources": self._format_sources(expanded_contexts),
                "retrieved_count": len(retrieved_chunks),
                "reranked_count": len(reranked_chunks),
                "latency_ms": latency_ms,
            }

        prompt = build_legal_prompt(
            question=question,
            contexts=expanded_contexts,
        )

        self.logger.info(
            f"Generation config | model={self.settings.GENERATOR_MODEL_NAME} "
            f"| max_new_tokens={self.settings.MAX_NEW_TOKENS}"
        )
        generation_start = time.perf_counter()
        answer = self.generator.generate(
            prompt=prompt,
            max_new_tokens=self.settings.MAX_NEW_TOKENS,
        )
        if (
            self.settings.RETRY_INCOMPLETE_ANSWER
            and self._is_incomplete_generated_answer(answer)
        ):
            self.logger.warning(
                "Generated answer looks incomplete. Retrying once with stricter output constraints."
            )
            retry_prompt = (
                f"{prompt}\n\n"
                "BO SUNG BAT BUOC CHO LAN TRA LOI NAY:\n"
                "- Tra loi day du 3 muc: Trich dan nguyen van, Can cu phap ly, Ket luan.\n"
                "- Moi trich dan chi toi da 40 tu de tranh tra loi bi dung giua chung.\n"
                "- Khong duoc de cau dang do hoac thieu dau ket thuc.\n"
            )
            answer = self.generator.generate(
                prompt=retry_prompt,
                max_new_tokens=self.settings.MAX_NEW_TOKENS,
            )

        generation_ms = (time.perf_counter() - generation_start) * 1000
        self.logger.info(f"Generated answer in {generation_ms:.2f} ms")

        latency_ms = (time.perf_counter() - start_time) * 1000
        return {
            "question": question,
            "answer": answer,
            "sources": self._format_sources(expanded_contexts),
            "retrieved_count": len(retrieved_chunks),
            "reranked_count": len(reranked_chunks),
            "latency_ms": latency_ms,
        }

    def _format_sources(self, chunks: list[dict]) -> list[dict]:
        seen = set()
        sources = []

        for chunk in chunks:
            meta = chunk.get("metadata", {})
            key = (
                meta.get("so_hieu"),
                meta.get("dieu"),
                meta.get("khoan"),
                meta.get("diem"),
            )
            if key in seen:
                continue
            seen.add(key)

            sources.append({
                "so_hieu": meta.get("so_hieu"),
                "loai_van_ban": meta.get("loai_van_ban"),
                "dieu": meta.get("dieu"),
                "khoan": meta.get("khoan"),
                "diem": meta.get("diem"),
                "trich_doan": chunk.get("content", "")[:300],
            })

        return sources

    def _should_refuse_answer(self, chunks: list[dict], query_profile: dict) -> bool:
        return should_refuse_after_retrieval(chunks, query_profile=query_profile)

    def _should_refuse_after_rerank(self, chunks: list[dict]) -> bool:
        return should_refuse_after_rerank(chunks)

    def _is_incomplete_generated_answer(self, answer: str) -> bool:
        text = (answer or "").strip()
        if not text:
            return True

        if len(text) < max(0, int(self.settings.MIN_COMPLETE_ANSWER_CHARS)):
            return True

        normalized = self._normalize_for_match(text)
        required_markers = [
            "trich dan nguyen van",
            "can cu phap ly",
            "ket luan",
        ]
        if not all(marker in normalized for marker in required_markers):
            return True

        quote_count = text.count('"')
        if quote_count % 2 == 1:
            return True

        return False

    def _normalize_for_match(self, value: str) -> str:
        text = (value or "").strip().lower()
        text = text.replace("đ", "d")
        text = "".join(
            ch for ch in unicodedata.normalize("NFD", text)
            if unicodedata.category(ch) != "Mn"
        )
        return " ".join(text.split())
