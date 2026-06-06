# Vietnamese Legal Chatbot

Vietnamese Legal Chatbot là hệ thống hỏi đáp pháp luật tiếng Việt theo kiến trúc RAG. Hệ thống được thiết kế theo hướng kiến trúc hướng dịch vụ và 

- `data_pipeline`: thu thập, làm sạch, chunk văn bản luật và build index.
- `rag_engine`: FastAPI service xử lý truy vấn, retrieval, rerank và sinh câu trả lời.
- `frontend_app`: giao diện Gradio có đăng ký, đăng nhập và lưu lịch sử hội thoại.

Hệ thống ưu tiên trả lời dựa trên văn bản luật trong knowledge base, kèm nguồn trích dẫn. Đây là công cụ hỗ trợ tra cứu/phân tích, không thay thế tư vấn pháp lý chuyên nghiệp.

## Kiến Trúc

```text
User
  -> Gradio frontend
  -> RAG Engine API
  -> Gemini query profiler
  -> BM25 + Qdrant vector search
  -> law hint expansion
  -> metadata rescoring
  -> PhoRanker reranking
  -> legal context expansion
  -> Gemini answer generation
  -> answer + sources
```

Các thành phần retrieval chính:

- BM25 dùng `pyvi` + `rank-bm25`.
- Vector search dùng Qdrant + `SentenceTransformer`.
- Reranker mặc định dùng `itdainb/PhoRanker`.
- Query profile và answer generation dùng Gemini API.
- Context sau rerank được mở rộng theo cùng điều luật và điều luật được tham chiếu.

## Cấu Trúc Repo

```text
Vietnamese_Legal_Chatbot/
├── data_pipeline/
│   ├── data/
│   │   ├── raw/          # văn bản luật gốc .doc/.docx
│   │   ├── cleaned/      # text đã làm sạch
│   │   ├── processed/    # chunks JSON
│   │   └── indexes/      # BM25 artifacts
│   ├── dags/             # Airflow DAG + crawler tùy chọn
│   ├── ingestion/        # cleaner + legal chunker
│   └── scripts/
│       ├── ingest_pdf.py # tên legacy, hiện xử lý .doc/.docx
│       └── build_index.py
├── rag_engine/
│   ├── api/              # FastAPI endpoints
│   ├── core/             # settings + logger
│   ├── eval/             # eval_questions.json
│   ├── generation/       # prompt builder + Gemini generator
│   ├── retrieval/        # query profile, hybrid search, reranker, rescoring
│   ├── scripts/          # build/evaluate eval set
│   ├── services/         # ChatService orchestration
│   └── tests/
├── frontend_app/
│   ├── assets/
│   ├── core/
│   ├── main.py
│   └── run.py
├── shared/               # BM25Store, VectorStore, config helpers
├── .env.example
├── docker-compose.yaml   # Airflow optional stack
└── README.md
```

## Knowledge Base

Văn bản luật gốc nằm trong `data_pipeline/data/raw`. Hiện pipeline chủ yếu xử lý `.doc` và `.docx`; nếu bổ sung file mới, cần chạy lại ingestion và build index.

Knowledge base hiện đã mở rộng qua nhiều nhóm luật, ví dụ:

- Dân sự, hình sự, tố tụng dân sự, tố tụng hình sự.
- Đất đai, nhà ở, kinh doanh bất động sản, xây dựng.
- Lao động, bảo hiểm xã hội, bảo hiểm y tế.
- Thuế, thương mại, doanh nghiệp, chứng khoán, tổ chức tín dụng.
- Giao thông đường bộ, xử lý vi phạm hành chính.
- Hôn nhân gia đình, căn cước, giao dịch điện tử, an ninh mạng, sở hữu trí tuệ.

## Cài Đặt

Yêu cầu khuyến nghị:

- Python 3.10 hoặc 3.11.
- Qdrant server hoặc Qdrant Cloud.
- Gemini API key.
- GPU là tùy chọn, nhưng giúp embedding/rerank nhanh hơn.

Tạo môi trường:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Cài dependency cho toàn bộ workspace:

```powershell
pip install -r data_pipeline\requirements.txt
pip install -r rag_engine\requirements.txt
pip install -r frontend_app\requirements.txt
```

Nếu chỉ chạy một phần cụ thể, có thể cài riêng requirement của phần đó.

## Cấu Hình

Tạo file `.env` từ `.env.example`:

```powershell
Copy-Item .env.example .env
```

Các biến quan trọng:

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY

EMBEDDING_MODEL_NAME=huyydangg/DEk21_hcmute_embedding
RERANKER_MODEL_NAME=itdainb/PhoRanker
GENERATOR_MODEL_NAME=models/gemini-flash-latest
QUERY_PROFILE_MODEL_NAME=models/gemini-flash-latest

QDRANT_COLLECTION=legal_documents
QDRANT_HOST=localhost
QDRANT_PORT=6333

TOP_K_BM25=15
TOP_K_VECTOR=15
TOP_K_RERANK=5
MAX_CONTEXT_CHUNKS=32
MAX_CONTEXT_CHARS=24000
```

Qdrant có ba chế độ cấu hình:

- Local server: dùng `QDRANT_HOST` + `QDRANT_PORT`.
- Qdrant Cloud: dùng `QDRANT_URL` + `QDRANT_API_KEY`.
- Local embedded path: dùng `QDRANT_PATH`.

Mặc định hệ thống Qdrant chạy tại `localhost:6333`.

## Chuẩn Bị Index

Nếu dùng Qdrant local bằng Docker:

```powershell
docker run -p 6333:6333 qdrant/qdrant
```

Từ root repo, đặt `PYTHONPATH` để các package nội bộ được import ổn định:

```powershell
$env:PYTHONPATH = (Get-Location).Path
```

Chạy ingestion:

```powershell
python data_pipeline\scripts\ingest_pdf.py
```

Build BM25 và Qdrant vector index:

```powershell
python data_pipeline\scripts\build_index.py
```

Sau khi thêm/sửa văn bản trong `data_pipeline/data/raw`, cần chạy lại cả hai lệnh trên.

## Chạy RAG Engine

```powershell
$env:PYTHONPATH = (Get-Location).Path
python rag_engine\run.py
```

API mặc định:

- Health check: `http://127.0.0.1:8000/health`
- Swagger docs: `http://127.0.0.1:8000/docs`
- Chat endpoint: `POST http://127.0.0.1:8000/api/chat`

Test nhanh bằng PowerShell:

```powershell
$body = @{ question = "Theo Luật Đất đai, người sử dụng đất bao gồm những đối tượng nào?" } | ConvertTo-Json
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/chat" `
  -ContentType "application/json; charset=utf-8" `
  -Body $body
```

## Chạy Frontend

Mở terminal thứ hai, vẫn từ root repo:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m frontend_app.run
```

Frontend mặc định chạy tại:

```text
http://127.0.0.1:7860
```

Frontend dùng SQLite mặc định tại `frontend_app/app_data.sqlite3` để lưu tài khoản và lịch sử hội thoại. Có thể đổi bằng biến `DATABASE_URL`.

## Evaluation

Tạo lại bộ câu hỏi đánh giá từ processed chunks:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python rag_engine\scripts\build_eval_questions.py
```

Chạy retrieval/legal metrics, bỏ qua sinh câu trả lời:

```powershell
python rag_engine\scripts\evaluate_ragas.py --no-generation --skip-ragas
```

Chạy evaluation đầy đủ với Gemini/RAGAS:

```powershell
python rag_engine\scripts\evaluate_ragas.py
```

Kết quả được lưu tại:

```text
rag_engine/eval/ragas_runs/<timestamp>/
```

Các artifact chính:

- `summary.json`: tổng hợp metric.
- `legal_metrics.json`: metric pháp lý nội bộ.
- `legal_details.csv`: chi tiết từng câu.
- `samples.jsonl`: sample đầy đủ kèm trace.
- `ragas_summary.json`, `ragas_rows.csv`: có khi chạy RAGAS thành công.

## Airflow Tùy Chọn

`docker-compose.yaml` hiện phục vụ stack Airflow + Postgres cho ETL tự động. Stack này không chạy RAG API/frontend.

Chạy Airflow:

```powershell
docker compose up airflow-init
docker compose up airflow-webserver airflow-scheduler
```

Web UI mặc định:

```text
http://localhost:8080
```

DAG chính: `vietnam_legal_rag_etl_pipeline`

Crawler dùng tài khoản Thư Viện Pháp Luật qua:

```env
TVPL_USERNAME=YOUR_TVPL_ACCOUNT
TVPL_PASSWORD=YOUR_PASSWORD
```

Không commit credential thật vào repo.

## Luồng Xử Lý Câu Hỏi

1. `QueryProfiler` gọi Gemini để phân tích intent, keywords, luật/điều/khoản được nhắc đến và query mở rộng.
2. `hybrid_search` lấy ứng viên từ BM25 và vector search, hợp nhất bằng Reciprocal Rank Fusion.
3. `law_hint_expander` bổ sung candidate theo tín hiệu luật/điều từ query profile và catalog trong Qdrant.
4. `metadata_rescorer` tăng điểm các chunk khớp metadata pháp lý.
5. `Reranker` dùng PhoRanker chọn top chunks.
6. `expand_legal_context` lấy thêm chunk cùng điều và điều được tham chiếu.
7. `ChatService` giới hạn context theo `MAX_CONTEXT_CHUNKS` và `MAX_CONTEXT_CHARS`.
8. `AnswerGenerator` gọi Gemini để trả lời theo prompt pháp lý bắt buộc có trích dẫn, căn cứ và kết luận.

Nếu Gemini query profile hoặc generation bị quá tải/lỗi provider, API trả payload có `error_code` và thông báo thân thiện để frontend hiển thị.

## Test

Repo đang có test đơn vị cho chunker, query profile, refusal policy, law hint expander và giới hạn context:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m pytest data_pipeline\tests rag_engine\tests
```

Nếu chưa cài `pytest`:

```powershell
pip install pytest
```

## Troubleshooting

Qdrant connection failed:

- Kiểm tra Qdrant đã chạy ở `localhost:6333`.
- Nếu dùng cloud, kiểm tra `QDRANT_URL`, `QDRANT_API_KEY`.
- Sau khi đổi collection/model embedding, chạy lại `build_index.py`.

Missing Gemini API key:

- Kiểm tra `.env` có `GEMINI_API_KEY`.
- Query profile và generation đều cần key này nếu bật generation.

Frontend không gọi được API:

- Kiểm tra `RAG_ENGINE_BASE_URL`.
- Đảm bảo `rag_engine\run.py` đang chạy.

Model tải chậm hoặc hết RAM:

- Lần đầu chạy embedding/reranker có thể tải model từ Hugging Face.
- GPU không bắt buộc, nhưng CPU sẽ chậm hơn đáng kể.

Kết quả vẫn dùng knowledge base cũ:

- Chạy lại ingestion.
- Chạy lại build index.
- Restart RAG engine để load BM25 mới và Qdrant collection mới.
