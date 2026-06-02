from pathlib import Path

from data_pipeline.core.config import get_settings
from data_pipeline.ingestion.cleaner import extract_text_from_doc, clean_legal_text, save_cleaned_text
from data_pipeline.ingestion.legal_chunker import chunk_legal_document, save_chunks_to_json


def run_ingestion(input_dir: str, cleaned_dir: str, processed_dir: str) -> None:
    input_path = Path(input_dir)
    cleaned_path = Path(cleaned_dir)
    processed_path = Path(processed_dir)

    cleaned_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)

    doc_files = list(input_path.glob("*.doc")) + list(input_path.glob("*.docx"))

    for doc_file in doc_files:
        print(f"Processing: {doc_file.name}")

        raw_text = extract_text_from_doc(str(doc_file))
        cleaned_text = clean_legal_text(raw_text)

        cleaned_file = cleaned_path / f"{doc_file.stem}.txt"
        save_cleaned_text(cleaned_text, str(cleaned_file))

        chunks = chunk_legal_document(
            cleaned_text,
            file_name=doc_file.name,
        )

        processed_file = processed_path / f"{doc_file.stem}.json"
        save_chunks_to_json(chunks, str(processed_file))

        print(f"Saved cleaned text to: {cleaned_file}")
        print(f"Saved chunks to: {processed_file}")


if __name__ == "__main__":
    settings = get_settings()
    run_ingestion(
        input_dir=settings.RAW_DATA_DIR,
        cleaned_dir=settings.CLEANED_DATA_DIR,
        processed_dir=settings.PROCESSED_DATA_DIR,
    )
