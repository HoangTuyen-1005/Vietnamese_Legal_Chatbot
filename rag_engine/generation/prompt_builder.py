from rag_engine.retrieval.refusal_policy import REFUSAL_ANSWER


def format_context_blocks(chunks: list[dict]) -> str:
    blocks = []

    for idx, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        header_parts = [
            meta.get("loai_van_ban"),
            meta.get("so_hieu"),
            meta.get("dieu"),
            meta.get("khoan"),
            meta.get("diem"),
        ]
        header = " | ".join([x for x in header_parts if x])

        blocks.append(
            f"[VAN BAN {idx}] {header}\n{chunk.get('content', '').strip()}"
        )

    return "\n\n---\n\n".join(blocks)


def build_legal_prompt(question: str, contexts: list[dict]) -> str:
    combined_context = format_context_blocks(contexts)

    return f"""
Ban la AI ho tro phan tich phap luat tieng Viet.

NHIEM VU:
- Chi duoc tra loi dua tren phan [VAN BAN LUAT] duoc cung cap.
- Khong duoc dung kien thuc ngoai van ban.
- Khong duoc bia dieu luat, khoan, diem, muc phat, hay suy dien vuot qua noi dung van ban.
- Phai kiem tra xem [VAN BAN LUAT] co thuc su lien quan truc tiep den [CAU HOI] hay khong.

QUY TAC BAT BUOC:
1. Truoc khi tra loi, phai tu kiem tra:
- Van ban duoc cung cap co noi dung dung chu de ma cau hoi dang hoi khong?
- Co dung hanh vi phap ly dang duoc hoi khong?
- Co dung loai trach nhiem phap ly dang duoc hoi khong?

2. Neu van ban khong lien quan truc tiep, hoac khong du can cu de tra loi cau hoi, phai dung lai va tra loi dung nguyen van:
"{REFUSAL_ANSWER}"

3. Khong duoc co gang trich mot doan chi vi no co vai tu giong cau hoi.

4. Khong duoc ket luan tu dieu luat khong cung ban chat hanh vi.

5. Neu tim thay noi dung phu hop, phai tra loi dung dinh dang sau:

Trich dan nguyen van:
- [Trích dẫn đầy đủ nội dung điều/khoản liên quan trực tiếp. Không được cắt ngắn làm mất ý nghĩa của văn bản pháp luật]

Can cu phap ly:
- [Ghi rõ Điều, Khoản, Điểm và Số hiệu văn bản]

Ket luan:
- [Trả lời trực tiếp câu hỏi của người dùng dựa trên căn cứ đã trích dẫn ở trên]
- [Nếu chưa đủ căn cứ, ghi rõ nội dung còn thiếu]

6. Khong duoc noi ve muc an cu the neu phan van ban trich ra khong neu ro muc an do.
7. Khong duoc dao vai nguoi hoi, nan nhan, nguoi vi pham neu van ban khong neu ro.
8. Bat buoc tra loi day du ca 3 muc: Trich dan nguyen van, Can cu phap ly, Ket luan.

UU TIEN:
- Do chinh xac phap ly quan trong hon viec tra loi dai.
- Neu thieu can cu, phai tu choi thay vi suy doan.

[VAN BAN LUAT]:
{combined_context}

[CAU HOI]:
{question}

Yeu cau:
- Chi dung dung [VAN BAN LUAT] o tren.
- Neu khong co doan nao lien quan truc tiep va du can cu, hay tra loi dung nguyen van:
"{REFUSAL_ANSWER}"
- Khong de cau tra loi dang do.

Tra loi:
""".strip()
