# RAG Evaluation

This folder contains evaluation data and run artifacts for `rag_engine`.

## Run

Legal-domain metrics only:

```bash
python -m rag_engine.scripts.evaluate_ragas --skip-ragas
```

Full generation plus RAGAS:

```bash
python -m rag_engine.scripts.evaluate_ragas
```

Fast smoke run:

```bash
python -m rag_engine.scripts.evaluate_ragas --limit 3 --skip-ragas
```

Outputs are written to:

```text
rag_engine/eval/ragas_runs/<run_id>/
```

Each run includes:

- `samples.jsonl`: questions, answers, prompt contexts, returned sources, and trace summary.
- `legal_metrics.json`: custom legal metrics such as source hit rate, refusal accuracy, citation presence, and latency.
- `legal_details.csv`: per-question legal metric details.
- `ragas_summary.json`: aggregate RAGAS scores when RAGAS is enabled.
- `ragas_rows.csv`: row-level RAGAS scores when supported by the installed RAGAS version.
- `summary.json`: compact combined report.

## Dataset Fields

The current `eval_questions.json` already supports retrieval and refusal metrics:

```json
{
  "id": "q013",
  "question": "Nguyên tắc sử dụng đất theo Luật Đất đai 2024 là gì?",
  "category": "land_law_content",
  "expected_sources": [
    {
      "so_hieu": "31/2024/QH15",
      "dieu": "Điều 5",
      "khoan": null,
      "diem": null
    }
  ],
  "should_refuse": false
}
```

For richer RAGAS metrics such as context recall and answer correctness, add one of these fields:

- `reference`
- `reference_answer`
- `ground_truth`

Example:

```json
{
  "id": "q013",
  "question": "Nguyên tắc sử dụng đất theo Luật Đất đai 2024 là gì?",
  "category": "land_law_content",
  "expected_sources": [
    {
      "so_hieu": "31/2024/QH15",
      "dieu": "Điều 5",
      "khoan": null,
      "diem": null
    }
  ],
  "reference": "Luật Đất đai 2024 quy định nguyên tắc sử dụng đất tại Điều 5...",
  "should_refuse": false
}
```

