import json
from pathlib import Path

from data_pipeline.core.config import get_settings
from shared.bm25_store import BM25Store
from shared.vector_store import VectorStore


def describe_qdrant_target(settings) -> str:
    if getattr(settings, "QDRANT_URL", None):
        return f"url={settings.QDRANT_URL}"
    if getattr(settings, "QDRANT_PATH", None):
        return f"path={settings.QDRANT_PATH}"
    return f"host={settings.QDRANT_HOST}:{settings.QDRANT_PORT}"


def load_processed_chunks(processed_dir: str) -> list[dict]:
    all_chunks = []
    for file_path in Path(processed_dir).glob("*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            all_chunks.extend(chunks)
    return all_chunks


def build_all_indexes(chunks: list[dict]) -> None:
    settings = get_settings()

    bm25_store = BM25Store()
    bm25_store.build(chunks)
    bm25_store.save(
        index_path=settings.BM25_INDEX_PATH,
        docs_path=settings.BM25_DOCS_PATH,
    )

    vector_store = VectorStore(settings=settings)
    print(
        f"Indexing vectors to Qdrant ({describe_qdrant_target(settings)}) | "
        f"collection={settings.QDRANT_COLLECTION}"
    )
    vector_store.recreate_collection()
    vector_store.index_chunks(chunks)

    if vector_store.client is not None:
        try:
            info = vector_store.client.get_collection(settings.QDRANT_COLLECTION)
            print(
                f"Qdrant collection '{settings.QDRANT_COLLECTION}' now has "
                f"{info.points_count} points."
            )
        except Exception as exc:
            print(f"Could not fetch Qdrant collection stats: {exc}")

    print("Index building completed.")

if __name__ == "__main__":
    settings = get_settings()
    chunks = load_processed_chunks(settings.PROCESSED_DATA_DIR)
    print(f"Loaded {len(chunks)} chunks.")
    build_all_indexes(chunks)
