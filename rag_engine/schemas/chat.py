from typing import Any, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User legal question")


class SourceItem(BaseModel):
    so_hieu: Optional[str] = None
    loai_van_ban: Optional[str] = None
    dieu: Optional[str] = None
    khoan: Optional[str] = None
    diem: Optional[str] = None
    trich_doan: Optional[str] = None


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceItem]
    retrieved_count: int
    reranked_count: int
    latency_ms: Optional[float] = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None
    source: Optional[str] = None
    rerank_score: Optional[float] = None