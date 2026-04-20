from data_pipeline.ingestion.legal_chunker import chunk_legal_document


def test_chunker_returns_list():
    sample_text = """
    LUẬT GIAO THÔNG ĐƯỜNG BỘ

    Điều 1. Phạm vi điều chỉnh
    Luật này quy định về giao thông đường bộ.

    Điều 2. Đối tượng áp dụng
    Luật này áp dụng đối với cơ quan, tổ chức, cá nhân.
    """.strip()

    chunks = chunk_legal_document(sample_text)
    assert isinstance(chunks, list)


def test_chunker_not_empty():
    sample_text = """
    Điều 1. Phạm vi điều chỉnh
    Luật này quy định về giao thông đường bộ.
    """.strip()

    chunks = chunk_legal_document(sample_text)
    assert len(chunks) > 0


def test_chunk_has_required_keys():
    sample_text = """
    Điều 1. Phạm vi điều chỉnh
    Luật này quy định về giao thông đường bộ.
    """.strip()

    chunks = chunk_legal_document(sample_text)
    first = chunks[0]

    assert "chunk_id" in first
    assert "content" in first
    assert "metadata" in first
