# Vietnamese Legal Chatbot

## Overview
Vietnamese Legal Chatbot is a production-minded academic project for Vietnamese legal question answering.  
The system uses legal-aware document chunking, hybrid retrieval, reranking, and Gemini API generation.

## Main Features
- Legal-aware chunking by Điều / Khoản / Điểm
- Hybrid retrieval: BM25 + vector search
- Metadata-aware rescoring
- Cross-encoder reranking
- Gemini API answer generation
- Mini evaluation pipeline for retrieval quality

## Project Structure

Vietnamese_Legal_Chatbot/
│
├── app/
│   ├── api/
│   │   └── main.py
│   ├── core/
│   │   ├── config.py
│   │   └── logging.py
│   ├── ingestion/
│   │   ├── cleaner.py
│   │   └── legal_chunker.py
│   ├── retrieval/
│   │   ├── bm25_indexer.py
│   │   ├── vector_indexer.py
│   │   ├── hybrid_search.py
│   │   └── reranker.py
│   ├── generation/
│   │   ├── prompt_builder.py
│   │   └── generator.py
│   ├── schemas/
│   │   └── chat.py
│   └── services/
│       └── chat_service.py
│
├── data/
│   ├── raw/
│   ├── cleaned/
│   └── processed/
│
├── scripts/
│   ├── build_index.py
│   └── ingest_pdf.py
│
├── tests/
│   └── test_chunker.py
│
├── .env.example
├── requirements.txt
├── README.md
└── run.py

## How to Run

### 1. Create virtual environment
python -m venv venv

### 2. Install dependencies
pip install -r requirements.txt

### 2.1 Configure environment
Copy `.env.example` to `.env` and set:
- `GEMINI_API_KEY=your_api_key`
- Optional: `GENERATOR_MODEL_NAME` (default `gemini-2.0-flash`)

### 3. Put legal PDFs into:
data/raw/

### 4. Run ingestion
python -m scripts.ingest_pdf

### 5. Build indexes
python -m scripts.build_index

### 6. Start API
python run.py

### 7. Open docs
http://127.0.0.1:8000/docs

## Evaluation
Run retrieval evaluation with:
python -m scripts.evaluate

## Current Results


## Limitations


## Future Work
