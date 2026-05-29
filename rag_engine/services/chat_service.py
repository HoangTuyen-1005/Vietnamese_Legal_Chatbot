import time
import unicodedata

from rag_engine.generation.prompt_builder import build_legal_prompt
from rag_engine.retrieval.hybrid_search import hybrid_search
from rag_engine.retrieval.law_hint_expander import augment_with_law_hints
from rag_engine.retrieval.metadata_rescorer import rescore_candidates
from rag_engine.retrieval.query_profile import build_query_profile
from rag_engine.retrieval.refusal_policy import (
    GENERATION_DISABLED_ANSWER,
    REFUSAL_ANSWER,
    diagnose_rerank_refusal,
    diagnose_retrieval_refusal,
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

    def answer_question(self, question: str, return_trace: bool = False) -> dict:
        start_time = time.perf_counter()
        self.logger.info(f"Received question: {question}")
        trace = {} if return_trace else None

        profile_start = time.perf_counter()
        query_profile = build_query_profile(question)
        if trace is not None:
            trace["query_profile"] = query_profile
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
        if trace is not None:
            trace["hybrid_chunks"] = retrieved_chunks

        retrieved_chunks = augment_with_law_hints(
            retrieved=retrieved_chunks,
            query_profile=query_profile,
            vector_store=self.vector_store,
            add_k=max(6, self.settings.TOP_K_VECTOR // 2),
        )
        if trace is not None:
            trace["augmented_chunks"] = retrieved_chunks

        self.logger.info(
            "Retrieved/augmented %s chunks before rescore | top_candidates=%s",
            len(retrieved_chunks),
            self._summarize_chunks(retrieved_chunks),
        )

        retrieved_chunks = rescore_candidates(
            candidates=retrieved_chunks,
            query_profile=query_profile,
            top_k=candidate_pool_k,
        )
        if trace is not None:
            trace["retrieved_chunks"] = retrieved_chunks

        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
        self.logger.info(
            "Retrieved/rescored %s chunks in %.2f ms | top_candidates=%s",
            len(retrieved_chunks),
            retrieval_ms,
            self._summarize_chunks(retrieved_chunks),
        )

        retrieval_diagnostics = diagnose_retrieval_refusal(
            retrieved_chunks,
            query_profile=query_profile,
        )
        if trace is not None:
            trace["retrieval_diagnostics"] = retrieval_diagnostics
        self.logger.info(
            "Retrieval refusal check | diagnostics=%s | top_candidates=%s",
            retrieval_diagnostics,
            self._summarize_chunks(retrieved_chunks),
        )

        if retrieval_diagnostics["should_refuse"]:
            self.logger.info(
                "Refusing after retrieval | reason=%s",
                retrieval_diagnostics.get("reason"),
            )
            latency_ms = (time.perf_counter() - start_time) * 1000
            result = {
                "question": question,
                "answer": REFUSAL_ANSWER,
                "sources": [],
                "retrieved_count": len(retrieved_chunks),
                "reranked_count": 0,
                "latency_ms": latency_ms,
            }
            if trace is not None:
                result["trace"] = trace
            return result

        rerank_start = time.perf_counter()
        reranked_chunks = self.reranker.rerank(
            query=question,
            candidates=retrieved_chunks,
            top_k=self.settings.TOP_K_RERANK,
        )
        if trace is not None:
            trace["reranked_chunks"] = reranked_chunks
        rerank_ms = (time.perf_counter() - rerank_start) * 1000
        self.logger.info(
            f"Reranked to {len(reranked_chunks)} chunks in {rerank_ms:.2f} ms"
        )

        rerank_diagnostics = diagnose_rerank_refusal(
            reranked_chunks,
            query_profile=query_profile,
        )
        if trace is not None:
            trace["rerank_diagnostics"] = rerank_diagnostics
        self.logger.info(
            "Rerank refusal check | diagnostics=%s | top_candidates=%s",
            rerank_diagnostics,
            self._summarize_chunks(reranked_chunks),
        )

        if rerank_diagnostics["should_refuse"]:
            self.logger.info(
                "Refusing after rerank | reason=%s",
                rerank_diagnostics.get("reason"),
            )
            latency_ms = (time.perf_counter() - start_time) * 1000
            result = {
                "question": question,
                "answer": REFUSAL_ANSWER,
                "sources": [],
                "retrieved_count": len(retrieved_chunks),
                "reranked_count": len(reranked_chunks),
                "latency_ms": latency_ms,
            }
            if trace is not None:
                result["trace"] = trace
            return result

        context_start = time.perf_counter()
        expanded_contexts = self.reranker.expand_legal_context(
            top_chunks=reranked_chunks,
            vector_store=self.vector_store,
            follow_referenced_dieu=self.settings.FOLLOW_REFERENCED_DIEU,
            max_referenced_dieu_per_chunk=self.settings.MAX_REFERENCED_DIEU_PER_CHUNK,
            max_referenced_dieu_total=self.settings.MAX_REFERENCED_DIEU_TOTAL,
        )
        if trace is not None:
            trace["expanded_contexts"] = expanded_contexts
        context_ms = (time.perf_counter() - context_start) * 1000
        self.logger.info(
            f"Expanded to {len(expanded_contexts)} context chunks in {context_ms:.2f} ms"
        )

        prompt_contexts = self._limit_contexts_for_prompt(
            contexts=expanded_contexts,
            reranked_chunks=reranked_chunks,
            query_profile=query_profile,
        )
        if trace is not None:
            trace["prompt_contexts"] = prompt_contexts
        if len(prompt_contexts) < len(expanded_contexts):
            self.logger.info(
                "Trimmed prompt context from %s to %s chunks | expanded_chars=%s | prompt_context_chars=%s | top_context=%s",
                len(expanded_contexts),
                len(prompt_contexts),
                self._total_content_chars(expanded_contexts),
                self._total_content_chars(prompt_contexts),
                self._summarize_chunks(prompt_contexts),
            )
        else:
            self.logger.info(
                "Prompt context kept at %s chunks | context_chars=%s",
                len(prompt_contexts),
                self._total_content_chars(prompt_contexts),
            )

        if self.generator is None:
            self.logger.info("Generation is disabled. Returning retrieval-only result.")
            latency_ms = (time.perf_counter() - start_time) * 1000
            result = {
                "question": question,
                "answer": GENERATION_DISABLED_ANSWER,
                "sources": self._format_sources(prompt_contexts),
                "retrieved_count": len(retrieved_chunks),
                "reranked_count": len(reranked_chunks),
                "latency_ms": latency_ms,
            }
            if trace is not None:
                result["trace"] = trace
            return result

        prompt = build_legal_prompt(
            question=question,
            contexts=prompt_contexts,
        )
        if trace is not None:
            trace["prompt_chars"] = len(prompt)

        self.logger.info(
            "Generation config | model=%s | max_new_tokens=%s | prompt_chars=%s | context_chunks=%s",
            self.settings.GENERATOR_MODEL_NAME,
            self.settings.MAX_NEW_TOKENS,
            len(prompt),
            len(prompt_contexts),
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
        if trace is not None:
            trace["generation_ms"] = generation_ms

        latency_ms = (time.perf_counter() - start_time) * 1000
        result = {
            "question": question,
            "answer": answer,
            "sources": self._format_sources(prompt_contexts),
            "retrieved_count": len(retrieved_chunks),
            "reranked_count": len(reranked_chunks),
            "latency_ms": latency_ms,
        }
        if trace is not None:
            result["trace"] = trace
        return result

    def _limit_contexts_for_prompt(
        self,
        contexts: list[dict],
        reranked_chunks: list[dict],
        query_profile: dict,
    ) -> list[dict]:
        max_chunks = int(getattr(self.settings, "MAX_CONTEXT_CHUNKS", 32) or 0)
        max_chars = int(getattr(self.settings, "MAX_CONTEXT_CHARS", 24000) or 0)
        if not contexts or (max_chunks <= 0 and max_chars <= 0):
            return contexts

        ranked = []
        for index, chunk in enumerate(contexts):
            ranked.append((
                self._context_relevance_score(chunk, reranked_chunks, query_profile),
                index,
                chunk,
            ))

        ranked.sort(key=lambda item: (item[0], -item[1]), reverse=True)

        selected: list[tuple[int, dict]] = []
        selected_chars = 0
        for _, index, chunk in ranked:
            content_chars = len(chunk.get("content", "") or "")
            if max_chunks > 0 and len(selected) >= max_chunks:
                continue
            if (
                max_chars > 0
                and selected
                and selected_chars + content_chars > max_chars
            ):
                continue

            selected.append((index, chunk))
            selected_chars += content_chars

        if not selected:
            return contexts[:1]

        selected.sort(key=lambda item: item[0])
        return [chunk for _, chunk in selected]

    def _context_relevance_score(
        self,
        chunk: dict,
        reranked_chunks: list[dict],
        query_profile: dict,
    ) -> float:
        chunk_id = chunk.get("chunk_id")
        top_ids = {item.get("chunk_id") for item in reranked_chunks if item.get("chunk_id")}
        top_dieu_keys = {
            (item.get("metadata", {}).get("so_hieu"), item.get("metadata", {}).get("dieu"))
            for item in reranked_chunks
            if item.get("metadata", {}).get("so_hieu") and item.get("metadata", {}).get("dieu")
        }
        meta = chunk.get("metadata", {})
        role = chunk.get("_context_role")

        score = 0.0
        if chunk_id in top_ids:
            score += 100.0
        if role == "referenced_dieu":
            score += 40.0
        elif role == "same_dieu":
            score += 15.0

        if (meta.get("so_hieu"), meta.get("dieu")) in top_dieu_keys:
            score += 10.0

        score += self._round_float(chunk.get("rerank_score")) or 0.0
        score += (self._round_float(chunk.get("metadata_boost")) or 0.0) * 5.0

        title_norm = self._normalize_for_match(
            " ".join([
                str(meta.get("ten_dieu") or ""),
                str(meta.get("dieu") or ""),
                str(meta.get("khoan") or ""),
                str(meta.get("diem") or ""),
            ])
        )
        body_norm = self._normalize_for_match(str(chunk.get("content", ""))[:1800])

        for keyword in (query_profile.get("keywords") or [])[:24]:
            keyword_norm = self._normalize_for_match(str(keyword))
            if len(keyword_norm) < 2:
                continue
            if keyword_norm in title_norm:
                score += 3.0
            elif keyword_norm in body_norm:
                score += 1.0

        if any(
            phrase in body_norm
            for phrase in [
                "truong hop sau day",
                "mot so truong hop sau",
                "ap dung khi",
                "duoc ap dung",
                "can thiep som",
            ]
        ):
            score += 4.0

        return score

    def _total_content_chars(self, chunks: list[dict]) -> int:
        return sum(len(chunk.get("content", "") or "") for chunk in chunks)

    def _summarize_chunks(self, chunks: list[dict], limit: int = 5) -> list[dict]:
        summary = []
        for rank, chunk in enumerate(chunks[:limit], start=1):
            meta = chunk.get("metadata", {})
            summary.append({
                "rank": rank,
                "chunk_id": chunk.get("chunk_id"),
                "source": chunk.get("source"),
                "context_role": chunk.get("_context_role"),
                "score": self._round_float(chunk.get("score")),
                "base_score": self._round_float(chunk.get("base_score")),
                "metadata_boost": self._round_float(chunk.get("metadata_boost")),
                "rerank_score": self._round_float(chunk.get("rerank_score")),
                "so_hieu": meta.get("so_hieu"),
                "dieu": meta.get("dieu"),
                "khoan": meta.get("khoan"),
                "diem": meta.get("diem"),
                "ten_dieu": meta.get("ten_dieu"),
            })
        return summary

    def _round_float(self, value):
        try:
            return round(float(value), 4)
        except (TypeError, ValueError):
            return None

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

    def _should_refuse_after_rerank(self, chunks: list[dict], query_profile: dict) -> bool:
        return should_refuse_after_rerank(chunks, query_profile=query_profile)

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
