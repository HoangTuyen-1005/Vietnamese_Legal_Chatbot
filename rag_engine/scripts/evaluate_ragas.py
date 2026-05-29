from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from rag_engine.core.config import get_settings
from rag_engine.core.logger import setup_logger
from rag_engine.retrieval.refusal_policy import REFUSAL_ANSWER, is_refusal_answer


SOURCE_FIELDS = ("so_hieu", "dieu", "khoan", "diem")


def load_eval_questions(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of eval questions in {path}")
    return data


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=_json_default))
            f.write("\n")


def _safe_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(item) for item in value if item is not None)
    return str(value)


def extract_reference(item: dict) -> tuple[str | None, bool]:
    for key in ("reference", "reference_answer", "ground_truth", "ground_truths"):
        value = item.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            value = "\n".join(str(part) for part in value if part is not None)
        value = str(value).strip()
        if value:
            return value, True

    if item.get("should_refuse"):
        return REFUSAL_ANSWER, False
    return None, False


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

    for expected in valid_expected:
        for actual in actual_sources[:k]:
            if source_matches(expected, actual):
                return True
    return False


def _normalize_for_answer_match(value: str) -> str:
    text = (value or "").lower()
    return " ".join(text.split())


def answer_mentions_expected_source(answer: str, expected_sources: list[dict]) -> bool:
    normalized_answer = _normalize_for_answer_match(answer)
    valid_expected = normalize_expected_sources(expected_sources)
    if not valid_expected:
        return False

    for expected in valid_expected:
        labels = [
            expected.get("so_hieu"),
            expected.get("dieu"),
            expected.get("khoan"),
            expected.get("diem"),
        ]
        labels = [_normalize_for_answer_match(str(label)) for label in labels if label]
        if labels and any(label in normalized_answer for label in labels):
            return True
    return False


def answer_has_required_sections(answer: str) -> bool:
    normalized = _normalize_for_answer_match(answer)
    return all(
        marker in normalized
        for marker in (
            "trich dan nguyen van",
            "can cu phap ly",
            "ket luan",
        )
    )


def chunk_to_context(chunk: dict) -> str:
    meta = chunk.get("metadata", {}) or {}
    header_parts = [
        meta.get("loai_van_ban"),
        meta.get("so_hieu"),
        meta.get("dieu"),
        meta.get("khoan"),
        meta.get("diem"),
        meta.get("ten_dieu"),
    ]
    header = " | ".join(str(part) for part in header_parts if part)
    content = (chunk.get("content") or "").strip()
    if header:
        return f"{header}\n{content}"
    return content


def source_from_chunk(chunk: dict) -> dict:
    meta = chunk.get("metadata", {}) or {}
    return {
        "so_hieu": meta.get("so_hieu"),
        "dieu": meta.get("dieu"),
        "khoan": meta.get("khoan"),
        "diem": meta.get("diem"),
    }


def percentile(values: Sequence[float], q: float) -> float | None:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return None
    index = max(0, min(len(clean) - 1, math.ceil(q * len(clean)) - 1))
    return clean[index]


def compute_legal_metrics(samples: list[dict]) -> dict:
    total = len(samples)
    in_scope = [sample for sample in samples if not sample.get("should_refuse")]
    refusal = [sample for sample in samples if sample.get("should_refuse")]
    valid_expected = [
        sample
        for sample in in_scope
        if normalize_expected_sources(sample.get("expected_sources") or [])
    ]

    top1 = sum(
        has_match_in_top_k(sample.get("expected_sources") or [], sample.get("sources") or [], 1)
        for sample in valid_expected
    )
    top3 = sum(
        has_match_in_top_k(sample.get("expected_sources") or [], sample.get("sources") or [], 3)
        for sample in valid_expected
    )
    top5 = sum(
        has_match_in_top_k(sample.get("expected_sources") or [], sample.get("sources") or [], 5)
        for sample in valid_expected
    )

    context_top5 = sum(
        has_match_in_top_k(
            sample.get("expected_sources") or [],
            sample.get("context_sources") or [],
            5,
        )
        for sample in valid_expected
    )
    citation_hits = sum(
        answer_mentions_expected_source(
            sample.get("answer") or "",
            sample.get("expected_sources") or [],
        )
        for sample in valid_expected
    )
    required_section_hits = sum(
        answer_has_required_sections(sample.get("answer") or "")
        for sample in in_scope
        if not is_refusal_answer(sample.get("answer") or "")
    )

    refusal_correct = sum(
        is_refusal_answer(sample.get("answer") or "")
        for sample in refusal
    )
    false_refusals = sum(
        is_refusal_answer(sample.get("answer") or "")
        for sample in in_scope
    )

    latencies = [
        float(sample["latency_ms"])
        for sample in samples
        if sample.get("latency_ms") is not None
    ]

    details = []
    for sample in samples:
        expected_sources = sample.get("expected_sources") or []
        sources = sample.get("sources") or []
        details.append({
            "id": sample.get("id"),
            "category": sample.get("category"),
            "should_refuse": sample.get("should_refuse", False),
            "is_refusal_answer": is_refusal_answer(sample.get("answer") or ""),
            "source_hit_at_1": has_match_in_top_k(expected_sources, sources, 1),
            "source_hit_at_3": has_match_in_top_k(expected_sources, sources, 3),
            "source_hit_at_5": has_match_in_top_k(expected_sources, sources, 5),
            "context_hit_at_5": has_match_in_top_k(
                expected_sources,
                sample.get("context_sources") or [],
                5,
            ),
            "answer_mentions_expected_source": answer_mentions_expected_source(
                sample.get("answer") or "",
                expected_sources,
            ),
            "answer_has_required_sections": answer_has_required_sections(sample.get("answer") or ""),
            "latency_ms": sample.get("latency_ms"),
            "retrieved_count": sample.get("retrieved_count"),
            "reranked_count": sample.get("reranked_count"),
        })

    category_summary: dict[str, dict] = {}
    for sample_detail in details:
        category = sample_detail.get("category") or "uncategorized"
        bucket = category_summary.setdefault(
            category,
            {
                "total": 0,
                "source_hit_at_3": 0,
                "context_hit_at_5": 0,
                "refusal_correct": 0,
                "refusal_total": 0,
            },
        )
        bucket["total"] += 1
        bucket["source_hit_at_3"] += int(bool(sample_detail["source_hit_at_3"]))
        bucket["context_hit_at_5"] += int(bool(sample_detail["context_hit_at_5"]))
        if sample_detail["should_refuse"]:
            bucket["refusal_total"] += 1
            bucket["refusal_correct"] += int(bool(sample_detail["is_refusal_answer"]))

    for bucket in category_summary.values():
        bucket["source_hit_at_3_rate"] = (
            bucket["source_hit_at_3"] / bucket["total"] if bucket["total"] else 0.0
        )
        bucket["context_hit_at_5_rate"] = (
            bucket["context_hit_at_5"] / bucket["total"] if bucket["total"] else 0.0
        )
        bucket["refusal_accuracy"] = (
            bucket["refusal_correct"] / bucket["refusal_total"]
            if bucket["refusal_total"]
            else None
        )

    generated_in_scope = [
        sample
        for sample in in_scope
        if not is_refusal_answer(sample.get("answer") or "")
    ]

    summary = {
        "total": total,
        "in_scope_total": len(in_scope),
        "valid_expected_total": len(valid_expected),
        "refusal_total": len(refusal),
        "source_hit_at_1": top1 / len(valid_expected) if valid_expected else 0.0,
        "source_hit_at_3": top3 / len(valid_expected) if valid_expected else 0.0,
        "source_hit_at_5": top5 / len(valid_expected) if valid_expected else 0.0,
        "context_hit_at_5": context_top5 / len(valid_expected) if valid_expected else 0.0,
        "citation_presence": citation_hits / len(valid_expected) if valid_expected else 0.0,
        "required_sections_rate": (
            required_section_hits / len(generated_in_scope)
            if generated_in_scope
            else 0.0
        ),
        "refusal_accuracy": refusal_correct / len(refusal) if refusal else 0.0,
        "false_refusal_rate": false_refusals / len(in_scope) if in_scope else 0.0,
        "latency_ms_avg": sum(latencies) / len(latencies) if latencies else None,
        "latency_ms_p95": percentile(latencies, 0.95),
        "by_category": category_summary,
    }

    return {
        "summary": summary,
        "details": details,
    }


def init_chat_service(settings, *, generation_enabled: bool | None = None):
    from rag_engine.generation.generator import AnswerGenerator
    from rag_engine.retrieval.reranker import Reranker
    from rag_engine.services.chat_service import ChatService
    from shared.bm25_store import BM25Store
    from shared.vector_store import VectorStore

    logger = setup_logger("ragas_eval")

    bm25_store = BM25Store()
    bm25_store.load(
        index_path=settings.BM25_INDEX_PATH,
        docs_path=settings.BM25_DOCS_PATH,
    )

    vector_store = VectorStore(settings=settings)
    reranker = Reranker(model_name=settings.RERANKER_MODEL_NAME)

    if generation_enabled is None:
        generation_enabled = bool(settings.ENABLE_GENERATION)

    generator = None
    if generation_enabled:
        generator = AnswerGenerator(
            model_name=settings.GENERATOR_MODEL_NAME,
            api_key=settings.GEMINI_API_KEY,
            temperature=settings.GEMINI_TEMPERATURE,
        )

    return ChatService(
        bm25_store=bm25_store,
        vector_store=vector_store,
        reranker=reranker,
        generator=generator,
        settings=settings,
        logger=logger,
    )


def collect_samples(
    questions: Sequence[dict],
    chat_service,
    *,
    limit: int | None = None,
) -> list[dict]:
    selected = list(questions[:limit] if limit else questions)
    samples: list[dict] = []

    for index, item in enumerate(selected, start=1):
        question = str(item["question"]).strip()
        print(f"[{index}/{len(selected)}] {question}")
        result = chat_service.answer_question(question, return_trace=True)
        trace = result.get("trace") or {}
        prompt_contexts = trace.get("prompt_contexts") or []

        reference, has_reference = extract_reference(item)
        contexts = [chunk_to_context(chunk) for chunk in prompt_contexts]
        context_sources = [source_from_chunk(chunk) for chunk in prompt_contexts]

        samples.append({
            "id": item.get("id", f"q{index:04d}"),
            "category": item.get("category"),
            "question": question,
            "answer": result.get("answer", ""),
            "contexts": contexts,
            "sources": result.get("sources", []),
            "context_sources": context_sources,
            "expected_sources": item.get("expected_sources", []),
            "reference": reference,
            "has_reference": has_reference,
            "should_refuse": bool(item.get("should_refuse", False)),
            "retrieved_count": result.get("retrieved_count"),
            "reranked_count": result.get("reranked_count"),
            "latency_ms": result.get("latency_ms"),
            "trace_summary": {
                "query_profile": trace.get("query_profile"),
                "retrieval_diagnostics": trace.get("retrieval_diagnostics"),
                "rerank_diagnostics": trace.get("rerank_diagnostics"),
                "prompt_chars": trace.get("prompt_chars"),
                "generation_ms": trace.get("generation_ms"),
                "hybrid_count": len(trace.get("hybrid_chunks") or []),
                "augmented_count": len(trace.get("augmented_chunks") or []),
                "retrieved_count": len(trace.get("retrieved_chunks") or []),
                "expanded_context_count": len(trace.get("expanded_contexts") or []),
                "prompt_context_count": len(prompt_contexts),
            },
        })

    return samples


def make_ragas_dataset_rows(samples: Sequence[dict], *, require_reference: bool) -> list[dict]:
    rows: list[dict] = []
    for sample in samples:
        if sample.get("should_refuse"):
            continue
        if is_refusal_answer(sample.get("answer") or ""):
            continue
        if not sample.get("contexts"):
            continue
        reference = sample.get("reference")
        if require_reference and not reference:
            continue

        row = {
            "question": sample["question"],
            "answer": sample.get("answer", ""),
            "contexts": sample.get("contexts", []),
            "ground_truth": reference or "",
            "user_input": sample["question"],
            "response": sample.get("answer", ""),
            "retrieved_contexts": sample.get("contexts", []),
            "reference": reference or "",
        }
        rows.append(row)
    return rows


def make_gemini_ragas_llm(model_name: str, api_key: str | None):
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY/GOOGLE_API_KEY for RAGAS evaluator.")

    os.environ.setdefault("GOOGLE_API_KEY", api_key)

    from ragas.llms import llm_factory

    try:
        from google import genai as google_genai

        client = google_genai.Client(api_key=api_key)
        return llm_factory(model_name, provider="google", client=client)
    except Exception as exc:
        raise RuntimeError(
            "Could not initialize a Gemini evaluator for RAGAS. "
            "Install google-genai and verify the API key."
        ) from exc


def _metric_from_names(metrics_module, names: Sequence[str], *, llm=None, embeddings=None):
    for name in names:
        metric_obj = getattr(metrics_module, name, None)
        if metric_obj is None:
            continue

        if isinstance(metric_obj, type):
            kwargs = {}
            if llm is not None:
                kwargs["llm"] = llm
            if embeddings is not None:
                kwargs["embeddings"] = embeddings
            try:
                return metric_obj(**kwargs)
            except TypeError:
                try:
                    return metric_obj(llm=llm)
                except TypeError:
                    return metric_obj()

        metric = metric_obj
        if llm is not None and hasattr(metric, "llm"):
            metric.llm = llm
        if embeddings is not None and hasattr(metric, "embeddings"):
            metric.embeddings = embeddings
        return metric

    return None


def build_ragas_metrics(*, llm, has_reference_rows: bool) -> list:
    import ragas.metrics as ragas_metrics

    specs = [
        ("faithfulness", ["Faithfulness", "faithfulness"]),
        ("response_relevancy", ["ResponseRelevancy", "AnswerRelevancy", "answer_relevancy"]),
    ]
    if has_reference_rows:
        specs.extend([
            (
                "context_precision",
                [
                    "ContextPrecision",
                    "LLMContextPrecisionWithReference",
                    "context_precision",
                ],
            ),
            ("context_recall", ["ContextRecall", "LLMContextRecall", "context_recall"]),
            ("answer_correctness", ["AnswerCorrectness", "answer_correctness"]),
        ])
    else:
        specs.append((
            "context_precision_without_reference",
            ["LLMContextPrecisionWithoutReference"],
        ))

    metrics = []
    missing = []
    for label, names in specs:
        metric = _metric_from_names(ragas_metrics, names, llm=llm)
        if metric is None:
            missing.append(label)
        else:
            metrics.append(metric)

    if missing:
        print(f"Skipped unavailable RAGAS metrics: {', '.join(missing)}")
    if not metrics:
        raise RuntimeError("No compatible RAGAS metrics were available.")
    return metrics


def _ragas_result_to_dict(result) -> dict:
    for attr in ("_repr_dict", "scores"):
        value = getattr(result, attr, None)
        if isinstance(value, dict):
            return value
    try:
        return dict(result)
    except Exception:
        return {"repr": str(result)}


def run_ragas(
    samples: Sequence[dict],
    output_dir: Path,
    *,
    model_name: str,
    api_key: str | None,
) -> dict:
    try:
        from datasets import Dataset
        from ragas import evaluate
    except ImportError as exc:
        raise ImportError(
            "RAGAS dependencies are missing. Install with: "
            "pip install -r rag_engine/requirements.txt"
        ) from exc

    reference_rows = make_ragas_dataset_rows(samples, require_reference=True)
    has_reference_rows = bool(reference_rows)
    rows = reference_rows or make_ragas_dataset_rows(samples, require_reference=False)

    if not rows:
        return {
            "skipped": True,
            "reason": "No non-refusal samples with contexts were available for RAGAS.",
        }

    data = {key: [row[key] for row in rows] for key in rows[0].keys()}
    dataset = Dataset.from_dict(data)
    llm = make_gemini_ragas_llm(model_name=model_name, api_key=api_key)
    metrics = build_ragas_metrics(llm=llm, has_reference_rows=has_reference_rows)

    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        raise_exceptions=False,
        show_progress=True,
    )

    summary = _ragas_result_to_dict(result)
    _write_json(output_dir / "ragas_summary.json", summary)

    if hasattr(result, "to_pandas"):
        try:
            result.to_pandas().to_csv(output_dir / "ragas_rows.csv", index=False)
        except Exception as exc:
            print(f"Could not export RAGAS row scores to CSV: {exc}")

    return {
        "skipped": False,
        "evaluated_rows": len(rows),
        "used_reference_rows": has_reference_rows,
        "metrics": [getattr(metric, "name", metric.__class__.__name__) for metric in metrics],
        "summary": summary,
    }


def write_flat_details_csv(path: Path, details: Sequence[dict]) -> None:
    if not details:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(details[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(details)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate rag_engine with RAGAS plus legal-domain metrics."
    )
    parser.add_argument("--questions-path", default=None)
    parser.add_argument("--output-dir", default="rag_engine/eval/ragas_runs")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--no-generation", action="store_true")
    parser.add_argument("--ragas-model", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()

    questions_path = args.questions_path or settings.EVAL_QUESTIONS_PATH
    questions = load_eval_questions(questions_path)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    chat_service = init_chat_service(
        settings,
        generation_enabled=False if args.no_generation else None,
    )
    samples = collect_samples(
        questions,
        chat_service,
        limit=args.limit,
    )

    _write_jsonl(output_dir / "samples.jsonl", samples)
    legal_report = compute_legal_metrics(samples)
    _write_json(output_dir / "legal_metrics.json", legal_report)
    write_flat_details_csv(output_dir / "legal_details.csv", legal_report["details"])

    ragas_report = {"skipped": True, "reason": "Disabled by --skip-ragas."}
    if not args.skip_ragas and not args.no_generation:
        ragas_report = run_ragas(
            samples,
            output_dir,
            model_name=args.ragas_model or settings.GENERATOR_MODEL_NAME,
            api_key=settings.GEMINI_API_KEY or os.environ.get("GOOGLE_API_KEY"),
        )
    elif args.no_generation:
        ragas_report = {
            "skipped": True,
            "reason": "Generation disabled by --no-generation; RAGAS answer metrics are not meaningful.",
        }

    final_report = {
        "run_id": run_id,
        "questions_path": questions_path,
        "output_dir": str(output_dir),
        "legal_metrics": legal_report["summary"],
        "ragas": ragas_report,
    }
    _write_json(output_dir / "summary.json", final_report)

    print("\n=== LEGAL METRICS ===")
    print(json.dumps(legal_report["summary"], ensure_ascii=False, indent=2, default=_json_default))
    print("\n=== RAGAS ===")
    print(json.dumps(ragas_report, ensure_ascii=False, indent=2, default=_json_default))
    print(f"\nSaved evaluation artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
