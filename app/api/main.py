from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import get_settings
from app.core.logger import setup_logger
from app.schemas.chat import ChatRequest, ChatResponse
from app.retrieval.bm25_store import BM25Store
from app.retrieval.vector_store import VectorStore
from app.retrieval.reranker import Reranker
from app.generation.generator import AnswerGenerator
from app.services.chat_service import ChatService


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger = setup_logger()

    logger.info("Initializing application components...")

    bm25_store = BM25Store()
    bm25_store.load(
        index_path=settings.BM25_INDEX_PATH,
        docs_path=settings.BM25_DOCS_PATH,
    )

    vector_store = VectorStore(settings=settings)
    reranker = Reranker(model_name=settings.RERANKER_MODEL_NAME)
    generator = None
    if settings.ENABLE_GENERATION:
        logger.info(
            "Generation enabled | provider=gemini | model=%s | max_new_tokens=%s",
            settings.GENERATOR_MODEL_NAME,
            settings.MAX_NEW_TOKENS,
        )
        generator = AnswerGenerator(
            model_name=settings.GENERATOR_MODEL_NAME,
            api_key=settings.GEMINI_API_KEY,
            temperature=settings.GEMINI_TEMPERATURE,
        )
    else:
        logger.info("ENABLE_GENERATION=False. Chat API will return retrieval-only responses.")

    chat_service = ChatService(
        bm25_store=bm25_store,
        vector_store=vector_store,
        reranker=reranker,
        generator=generator,
        settings=settings,
        logger=logger,
    )

    app.state.chat_service = chat_service
    app.state.logger = logger
    logger.info("Application initialized successfully.")

    yield

    logger.info("Shutting down application...")


settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    debug=settings.DEBUG,
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"message": "Vietnamese Legal Chatbot API is running."}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = app.state.chat_service.answer_question(request.question)
    return ChatResponse(**result)
