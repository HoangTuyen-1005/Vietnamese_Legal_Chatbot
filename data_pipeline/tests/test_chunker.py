from data_pipeline.ingestion.legal_chunker import chunk_legal_document
from data_pipeline.ingestion.legal_chunker import extract_document_metadata


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


def test_metadata_prefers_law_code_from_original_law_filename():
    text = """
    LUẬT AN NINH MẠNG
    Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam đã được sửa đổi, bổ sung một số điều theo Nghị quyết số 203/2025/QH15;
    Quốc hội ban hành Luật An ninh mạng.
    """.strip()

    metadata = extract_document_metadata(
        text,
        file_name="Luat-An-ninh-mang-2025-so-116-2025-QH15-666020.docx",
    )

    assert metadata["so_hieu"] == "116/2025/QH15"


def test_metadata_normalizes_spaced_law_code_from_consolidated_text():
    text = """
    LUẬT PHÒNG, CHỐNG MUA BÁN NGƯỜI
    Luật Phòng, chống mua bán người số 53 /2024/QH15 ngày 28 tháng 11 năm 2024 của Quốc hội.
    """.strip()

    metadata = extract_document_metadata(
        text,
        file_name="Van-ban-hop-nhat-110-VBHN-VPQH-2026-Luat-Phong-chong-mua-ban-nguoi-708150.docx",
    )

    assert metadata["so_hieu"] == "53/2024/QH15"
