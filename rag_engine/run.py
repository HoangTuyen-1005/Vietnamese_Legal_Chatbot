from __future__ import annotations

import uvicorn

from rag_engine.core.config import get_settings


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "rag_engine.api.main:app",
        host=settings.RAG_ENGINE_HOST,
        port=settings.RAG_ENGINE_PORT,
        reload=True,
    )

