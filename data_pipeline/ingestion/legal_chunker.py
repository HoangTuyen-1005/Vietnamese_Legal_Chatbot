from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


RE_PHAN = re.compile(r"(?im)^\s*Phần\s+(?:thứ\s+\w+|[IVXLCDM0-9]+)\b.*$")
RE_TIEU_MUC = re.compile(r"(?im)^\s*Tiểu mục\s+[IVXLCDM0-9]+\b.*$")
RE_CHUONG = re.compile(
    r"(?im)^[ \t]*(Chương\s+[IVXLCDM0-9]+)(?:[ \t]*(?:[.\-:][ \t]*|[ \t]+)([^\n]+))?[ \t]*$"
)
RE_MUC = re.compile(
    r"(?im)^[ \t]*(Mục\s+[IVXLCDM0-9]+)(?:[ \t]*(?:[.\-:][ \t]*|[ \t]+)([^\n]+))?[ \t]*$"
)
RE_DIEU = re.compile(
    r"(?im)^[ \t]*(Điều\s+\d+[A-Za-z]?)[ \t]*(?:[.\-:][ \t]*)?([^\n]*)[ \t]*$"
)
RE_KHOAN = re.compile(r"(?im)^\s*(\d+)\.\s+")
RE_DIEM = re.compile(r"(?im)^\s*([a-zA-ZđĐ])\)\s+")

RE_SO_HIEU = re.compile(
    r"(?im)\b(\d{1,4}/(?:\d{4}/)?[A-ZĐ0-9\-]+(?:/[A-Z0-9\-]+)?)\b"
)
RE_LOAI_VAN_BAN = re.compile(
    r"(?im)\b(Luật|Bộ luật|Nghị định|Thông tư|Thông tư liên tịch|Nghị quyết|Quyết định|Pháp lệnh|Hiến pháp)\b"
)
RE_NGAY_BAN_HANH = re.compile(
    r"(?im)(?:ngày|ban hành ngày)\s+((?:\d{1,2}[/-]\d{1,2}[/-]\d{4})|(?:\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}))"
)


@dataclass
class ChunkRecord:
    chunk_id: str
    source_file: str
    document_name: str
    so_hieu: Optional[str]
    loai_van_ban: Optional[str]
    ngay_ban_hanh: Optional[str]
    chuong: Optional[str] = None
    ten_chuong: Optional[str] = None
    muc: Optional[str] = None
    ten_muc: Optional[str] = None
    dieu: Optional[str] = None
    ten_dieu: Optional[str] = None
    khoan: Optional[str] = None
    diem: Optional[str] = None
    cap_chunk: str = ""
    parent_path: List[str] = field(default_factory=list)
    text: str = ""


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_strip(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def normalize_doc_type(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    lookup = {
        "luật": "Luật",
        "bộ luật": "Bộ luật",
        "nghị định": "Nghị định",
        "thông tư": "Thông tư",
        "thông tư liên tịch": "Thông tư liên tịch",
        "nghị quyết": "Nghị quyết",
        "quyết định": "Quyết định",
        "pháp lệnh": "Pháp lệnh",
        "hiến pháp": "Hiến pháp",
    }
    key = value.strip().lower()
    return lookup.get(key, value.strip())


def is_structure_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    return bool(
        RE_PHAN.match(s)
        or RE_TIEU_MUC.match(s)
        or RE_CHUONG.match(s)
        or RE_MUC.match(s)
        or RE_DIEU.match(s)
        or RE_KHOAN.match(s)
        or RE_DIEM.match(s)
    )


def uppercase_ratio(line: str) -> float:
    letters = [ch for ch in line if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(ch.isupper() for ch in letters) / len(letters)


def is_probable_title_line(line: str, strict_upper: bool) -> bool:
    s = line.strip()
    if not s or is_structure_line(s):
        return False

    if strict_upper:
        return uppercase_ratio(s) >= 0.72 and len(s) <= 180

    if len(s) > 130:
        return False
    if re.search(r"[.;:,]$", s):
        return False

    words = s.split()
    if len(words) > 20:
        return False

    return s[0].isupper()


def extract_title_after_marker(
    section_text: str,
    strict_upper: bool,
    max_lines: int = 2,
) -> Tuple[Optional[str], str]:
    lines = [ln.rstrip() for ln in section_text.splitlines()]
    if len(lines) <= 1:
        return None, section_text.strip()

    idx = 1
    consumed_until = 1
    title_lines: List[str] = []

    while idx < len(lines) and len(title_lines) < max_lines:
        raw = lines[idx]
        s = raw.strip()

        if not s:
            idx += 1
            consumed_until = idx
            continue

        if is_structure_line(s):
            break

        if not is_probable_title_line(s, strict_upper=strict_upper):
            break

        title_lines.append(s)
        idx += 1
        consumed_until = idx

    if not title_lines:
        return None, "\n".join(lines).strip()

    kept_lines = [lines[0]] + lines[consumed_until:]
    return " ".join(title_lines).strip(), "\n".join(kept_lines).strip()


def ensure_dieu_title_in_text(
    dieu_text: str,
    dieu_label: Optional[str],
    ten_dieu: Optional[str],
) -> str:
    if not dieu_text or not dieu_label or not ten_dieu:
        return dieu_text

    lines = dieu_text.splitlines()
    if not lines:
        return dieu_text

    first = lines[0].strip()
    first_norm = re.sub(r"\s+", " ", first).rstrip(" .:")
    label_norm = re.sub(r"\s+", " ", dieu_label).strip()

    if first_norm.lower().startswith(label_norm.lower()):
        if ten_dieu.lower() not in first.lower():
            lines[0] = f"{dieu_label}. {ten_dieu}"
            return "\n".join(lines).strip()

    return dieu_text


def infer_doc_type_from_filename(file_name: str) -> Optional[str]:
    lower = file_name.lower()
    if "bo_luat" in lower:
        return "Bộ luật"
    if "nghi_dinh" in lower or "_nd_" in lower:
        return "Nghị định"
    if "thong_tu" in lower or "_tt_" in lower:
        return "Thông tư"
    if "nghi_quyet" in lower or "_nq_" in lower:
        return "Nghị quyết"
    if "quyet_dinh" in lower or "_qd_" in lower:
        return "Quyết định"
    if "luat" in lower:
        return "Luật"
    return None


def extract_document_metadata(text: str, file_name: str = "document.txt") -> Dict[str, Optional[str]]:
    header_zone = text[:5000]

    so_hieu_match = RE_SO_HIEU.search(header_zone)
    loai_match = RE_LOAI_VAN_BAN.search(header_zone)
    ngay_match = RE_NGAY_BAN_HANH.search(header_zone)

    ngay_ban_hanh = safe_strip(ngay_match.group(1)) if ngay_match else None
    loai_van_ban = normalize_doc_type(loai_match.group(1) if loai_match else None)

    if not loai_van_ban:
        loai_van_ban = infer_doc_type_from_filename(file_name)

    return {
        "document_name": Path(file_name).stem,
        "so_hieu": so_hieu_match.group(1) if so_hieu_match else None,
        "loai_van_ban": loai_van_ban,
        "ngay_ban_hanh": ngay_ban_hanh,
    }


def split_by_pattern(text: str, pattern: re.Pattern) -> List[Tuple[str, str, int, int]]:
    matches = list(pattern.finditer(text))
    if not matches:
        return []

    sections: List[Tuple[str, str, int, int]] = []

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        raw_section = text[start:end].strip()

        marker = match.group(1).strip()
        title = safe_strip(match.group(2)) if match.lastindex and match.lastindex >= 2 else None

        if title:
            if pattern is RE_DIEU:
                heading = f"{marker}. {title}"
            else:
                heading = f"{marker} {title}"
        else:
            heading = marker

        sections.append((heading.strip(), raw_section, start, end))

    return sections


def split_diem_in_khoan(khoan_text: str) -> List[Dict[str, str]]:
    diem_matches = list(RE_DIEM.finditer(khoan_text))
    if not diem_matches:
        return []

    intro = khoan_text[:diem_matches[0].start()].strip()
    diem_sections: List[Dict[str, str]] = []

    for i, m in enumerate(diem_matches):
        start = m.start()
        end = diem_matches[i + 1].start() if i + 1 < len(diem_matches) else len(khoan_text)
        section_text = khoan_text[start:end].strip()

        if intro:
            section_text = f"{intro}\n{section_text}"

        diem_sections.append({
            "diem": m.group(1).lower(),
            "text": section_text,
        })

    return diem_sections


def split_khoan_in_dieu(dieu_text: str) -> List[Dict[str, Any]]:
    matches = list(RE_KHOAN.finditer(dieu_text))
    if not matches:
        return []

    result: List[Dict[str, Any]] = []

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(dieu_text)
        section_text = dieu_text[start:end].strip()

        result.append({
            "khoan": m.group(1),
            "text": section_text,
        })

    return result


def parse_chuong_heading(heading: str) -> Tuple[Optional[str], Optional[str]]:
    m = RE_CHUONG.match(heading)
    if not m:
        return None, None
    return safe_strip(m.group(1)), safe_strip(m.group(2))


def parse_muc_heading(heading: str) -> Tuple[Optional[str], Optional[str]]:
    m = RE_MUC.match(heading)
    if not m:
        return None, None
    return safe_strip(m.group(1)), safe_strip(m.group(2))


def parse_dieu_heading(heading: str) -> Tuple[Optional[str], Optional[str]]:
    m = RE_DIEU.match(heading)
    if not m:
        return None, None
    return safe_strip(m.group(1)), safe_strip(m.group(2))


def chunk_document(text: str, file_name: str) -> List[ChunkRecord]:
    text = normalize_text(text)
    doc_meta = extract_document_metadata(text, file_name)

    records: List[ChunkRecord] = []
    chunk_counter = 0

    def next_id() -> str:
        nonlocal chunk_counter
        chunk_counter += 1
        return f"{doc_meta['document_name']}_{chunk_counter:06d}"

    chuong_sections = split_by_pattern(text, RE_CHUONG)
    pseudo_root = chuong_sections if chuong_sections else [("NO_CHUONG", text, 0, len(text))]

    for chuong_heading, chuong_text, _, _ in pseudo_root:
        if chuong_heading == "NO_CHUONG":
            chuong_label, ten_chuong = None, None
        else:
            chuong_label, ten_chuong = parse_chuong_heading(chuong_heading)
            if ten_chuong:
                continuation, chuong_text = extract_title_after_marker(
                    chuong_text, strict_upper=True, max_lines=2
                )
                if continuation:
                    ten_chuong = f"{ten_chuong} {continuation}".strip()
            else:
                inferred, chuong_text = extract_title_after_marker(
                    chuong_text, strict_upper=True, max_lines=2
                )
                ten_chuong = inferred

        muc_sections = split_by_pattern(chuong_text, RE_MUC)
        if not muc_sections:
            muc_sections = [("NO_MUC", chuong_text, 0, len(chuong_text))]

        for muc_heading, muc_text, _, _ in muc_sections:
            if muc_heading == "NO_MUC":
                muc_label, ten_muc = None, None
            else:
                muc_label, ten_muc = parse_muc_heading(muc_heading)
                if ten_muc:
                    continuation, muc_text = extract_title_after_marker(
                        muc_text, strict_upper=True, max_lines=2
                    )
                    if continuation:
                        ten_muc = f"{ten_muc} {continuation}".strip()
                else:
                    inferred, muc_text = extract_title_after_marker(
                        muc_text, strict_upper=True, max_lines=2
                    )
                    ten_muc = inferred

            dieu_sections = split_by_pattern(muc_text, RE_DIEU)
            if not dieu_sections:
                continue

            for dieu_heading, dieu_text, _, _ in dieu_sections:
                dieu_label, ten_dieu = parse_dieu_heading(dieu_heading)
                had_inline_title = bool(ten_dieu)

                if not ten_dieu:
                    inferred, dieu_text = extract_title_after_marker(
                        dieu_text, strict_upper=False, max_lines=1
                    )
                    ten_dieu = inferred

                if ten_dieu and not had_inline_title:
                    dieu_text = ensure_dieu_title_in_text(dieu_text, dieu_label, ten_dieu)

                khoan_sections = split_khoan_in_dieu(dieu_text)
                parent_path = [x for x in [chuong_label, muc_label, dieu_label] if x]

                if not khoan_sections:
                    records.append(
                        ChunkRecord(
                            chunk_id=next_id(),
                            source_file=file_name,
                            document_name=doc_meta["document_name"],
                            so_hieu=doc_meta["so_hieu"],
                            loai_van_ban=doc_meta["loai_van_ban"],
                            ngay_ban_hanh=doc_meta["ngay_ban_hanh"],
                            chuong=chuong_label,
                            ten_chuong=ten_chuong,
                            muc=muc_label,
                            ten_muc=ten_muc,
                            dieu=dieu_label,
                            ten_dieu=ten_dieu,
                            khoan=None,
                            diem=None,
                            cap_chunk="dieu",
                            parent_path=parent_path,
                            text=dieu_text.strip(),
                        )
                    )
                    continue

                for khoan_obj in khoan_sections:
                    khoan_label = f"Khoản {khoan_obj['khoan']}"
                    khoan_text = khoan_obj["text"].strip()
                    diem_sections = split_diem_in_khoan(khoan_text)

                    if not diem_sections:
                        records.append(
                            ChunkRecord(
                                chunk_id=next_id(),
                                source_file=file_name,
                                document_name=doc_meta["document_name"],
                                so_hieu=doc_meta["so_hieu"],
                                loai_van_ban=doc_meta["loai_van_ban"],
                                ngay_ban_hanh=doc_meta["ngay_ban_hanh"],
                                chuong=chuong_label,
                                ten_chuong=ten_chuong,
                                muc=muc_label,
                                ten_muc=ten_muc,
                                dieu=dieu_label,
                                ten_dieu=ten_dieu,
                                khoan=khoan_label,
                                diem=None,
                                cap_chunk="khoan",
                                parent_path=parent_path + [khoan_label],
                                text=khoan_text,
                            )
                        )
                        continue

                    for diem_obj in diem_sections:
                        diem_label = f"Điểm {diem_obj['diem']}"
                        records.append(
                            ChunkRecord(
                                chunk_id=next_id(),
                                source_file=file_name,
                                document_name=doc_meta["document_name"],
                                so_hieu=doc_meta["so_hieu"],
                                loai_van_ban=doc_meta["loai_van_ban"],
                                ngay_ban_hanh=doc_meta["ngay_ban_hanh"],
                                chuong=chuong_label,
                                ten_chuong=ten_chuong,
                                muc=muc_label,
                                ten_muc=ten_muc,
                                dieu=dieu_label,
                                ten_dieu=ten_dieu,
                                khoan=khoan_label,
                                diem=diem_label,
                                cap_chunk="diem",
                                parent_path=parent_path + [khoan_label, diem_label],
                                text=diem_obj["text"].strip(),
                            )
                        )

    return records


def _record_to_new_chunk(record: ChunkRecord) -> dict:
    data = asdict(record)
    text = data.pop("text", "").strip()

    metadata = {
        "source_file": data.get("source_file"),
        "document_name": data.get("document_name"),
        "so_hieu": data.get("so_hieu"),
        "loai_van_ban": data.get("loai_van_ban"),
        "ngay_ban_hanh": data.get("ngay_ban_hanh"),
        "chuong": data.get("chuong"),
        "ten_chuong": data.get("ten_chuong"),
        "muc": data.get("muc"),
        "ten_muc": data.get("ten_muc"),
        "dieu": data.get("dieu"),
        "ten_dieu": data.get("ten_dieu"),
        "khoan": data.get("khoan"),
        "diem": data.get("diem"),
        "cap_chunk": data.get("cap_chunk"),
        "parent_path": data.get("parent_path", []),
    }

    return {
        "chunk_id": data["chunk_id"],
        "content": text,
        "metadata": metadata,
    }


def chunk_legal_document(text: str, file_name: str = "document.txt") -> List[dict]:
    records = chunk_document(text, file_name)
    return [_record_to_new_chunk(record) for record in records]


def save_chunks_to_json(chunks: List[dict], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)