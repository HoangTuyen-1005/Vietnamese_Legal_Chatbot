from __future__ import annotations

import importlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


@dataclass
class CleanerConfig:
    margin_scan_lines: int = 6
    repeated_margin_min_pages: int = 3
    repeated_margin_ratio: float = 0.18
    trailing_footnote_scan_ratio: float = 0.55


def strip_accents(text: str) -> str:
    text = text.replace("Đ", "D").replace("đ", "d")
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def canonical_line(text: str) -> str:
    text = normalize_unicode(text).strip().lower()
    text = strip_accents(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\d+", "<num>", text)
    return text


def try_extract_with_fitz(pdf_path: Path) -> List[str]:
    fitz = importlib.import_module("fitz")
    pages: List[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pages.append(page.get_text("text") or "")
    return pages


def try_extract_with_pdfplumber(pdf_path: Path) -> List[str]:
    pdfplumber = importlib.import_module("pdfplumber")
    pages: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return pages


def try_extract_with_pypdf(pdf_path: Path) -> List[str]:
    pypdf = importlib.import_module("pypdf")
    reader = pypdf.PdfReader(str(pdf_path))
    return [(page.extract_text() or "") for page in reader.pages]


def extract_pdf_pages(pdf_path: Path) -> List[str]:
    backends = [
        ("fitz", try_extract_with_fitz),
        ("pdfplumber", try_extract_with_pdfplumber),
        ("pypdf", try_extract_with_pypdf),
    ]
    errors: List[str] = []

    for name, fn in backends:
        try:
            pages = fn(pdf_path)
            if pages:
                return pages
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    joined_errors = " | ".join(errors) if errors else "no backend available"
    raise RuntimeError(
        f"Could not extract PDF text for {pdf_path.name}. {joined_errors}"
    )


def split_non_empty_lines(page_text: str) -> List[str]:
    text = normalize_unicode(page_text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in text.split("\n")]
    return [ln for ln in lines if ln]


def find_repeated_margin_lines(
    pages_lines: Sequence[Sequence[str]],
    config: CleanerConfig,
) -> set[str]:
    top_counter: Counter[str] = Counter()
    bottom_counter: Counter[str] = Counter()

    for lines in pages_lines:
        if not lines:
            continue

        top = lines[: config.margin_scan_lines]
        bottom = lines[-config.margin_scan_lines :]

        for ln in top:
            top_counter[canonical_line(ln)] += 1
        for ln in bottom:
            bottom_counter[canonical_line(ln)] += 1

    page_count = max(len(pages_lines), 1)
    threshold = max(
        config.repeated_margin_min_pages,
        int(round(page_count * config.repeated_margin_ratio)),
    )

    repeated = {
        canon
        for canon, count in (top_counter | bottom_counter).items()
        if count >= threshold and len(canon) > 3
    }
    return repeated


def is_structure_heading(line: str) -> bool:
    normalized = strip_accents(line).lower().strip()
    if re.match(r"^(phan|chuong|muc)\s+[ivxlcdm0-9]+", normalized):
        return True
    if re.match(r"^dieu\s+\d+[a-z]?[.:]", normalized):
        return True
    if normalized.startswith("phu luc"):
        return True
    return False


def is_list_marker(line: str) -> bool:
    normalized = strip_accents(line).lower().strip()
    return bool(
        re.match(r"^\d+\.\s+", normalized)
        or re.match(r"^[a-z]\)\s+", normalized)
        or re.match(r"^[ivxlcdm]+\.\s+", normalized)
    )


def is_page_number_line(line: str) -> bool:
    normalized = strip_accents(line).lower().strip()
    return bool(
        re.match(r"^\d+$", normalized)
        or re.match(r"^trang\s+\d+(\s*/\s*\d+)?$", normalized)
        or re.match(r"^page\s+\d+(\s*/\s*\d+)?$", normalized)
    )


def is_common_boilerplate(line: str) -> bool:
    normalized = strip_accents(line).lower().strip()
    boilerplates = {
        "cong hoa xa hoi chu nghia viet nam",
        "doc lap - tu do - hanh phuc",
    }
    if normalized in boilerplates:
        return True
    if re.match(r"^[-_=]{3,}$", normalized):
        return True
    return False


def is_probable_agency_header(line: str, page_index: int, line_index: int) -> bool:
    if page_index != 0 or line_index > 14:
        return False

    normalized = strip_accents(line).lower().strip()
    if not normalized:
        return False

    keep_keywords = (
        "luat",
        "bo luat",
        "nghi dinh",
        "thong tu",
        "quyet dinh",
        "so:",
    )
    if any(keyword in normalized for keyword in keep_keywords):
        return False

    letters = [ch for ch in line if ch.isalpha()]
    if not letters:
        return False

    upper_ratio = sum(ch.isupper() for ch in letters) / len(letters)
    return upper_ratio >= 0.8 and len(normalized.split()) <= 8


def is_title_continuation(prev_line: str, curr_line: str) -> bool:
    prev_norm = strip_accents(prev_line).lower().strip()
    if prev_norm not in {
        "bo luat",
        "luat",
        "nghi dinh",
        "thong tu",
        "nghi quyet",
        "quyet dinh",
        "phap lenh",
        "hien phap",
    }:
        return False

    curr = curr_line.strip()
    if not curr:
        return False
    if is_structure_heading(curr) or is_list_marker(curr):
        return False

    letters = [ch for ch in curr if ch.isalpha()]
    if not letters:
        return False

    upper_ratio = sum(ch.isupper() for ch in letters) / len(letters)
    return upper_ratio >= 0.75 and len(curr) <= 100


def drop_noise_in_page(
    lines: Sequence[str],
    page_index: int,
    repeated_margin_lines: set[str],
    config: CleanerConfig,
) -> List[str]:
    cleaned: List[str] = []
    total = len(lines)

    for idx, original_line in enumerate(lines):
        line = original_line.strip()
        if not line:
            continue

        canon = canonical_line(line)
        in_margin = idx < config.margin_scan_lines or idx >= max(
            0, total - config.margin_scan_lines
        )

        if in_margin and canon in repeated_margin_lines:
            continue
        if is_page_number_line(line):
            continue
        if is_common_boilerplate(line):
            continue

        prev_line = lines[idx - 1] if idx > 0 else ""
        if is_probable_agency_header(line, page_index, idx) and not is_title_continuation(
            prev_line, line
        ):
            continue

        cleaned.append(line)

    return cleaned


def trim_trailing_footnote_appendix(
    lines: Sequence[str],
    config: CleanerConfig,
) -> List[str]:
    if len(lines) < 80:
        return list(lines)

    start = int(len(lines) * config.trailing_footnote_scan_ratio)
    footnote_head = re.compile(r"^\[\d+\]")

    for idx in range(start, max(start, len(lines) - 12)):
        window = lines[idx : idx + 12]
        foot_count = sum(1 for ln in window if footnote_head.match(ln.strip()))
        if foot_count >= 4:
            return list(lines[:idx])

    return list(lines)


def should_keep_newline(prev_line: str, curr_line: str) -> bool:
    if is_structure_heading(curr_line) or is_list_marker(curr_line):
        return True
    if is_structure_heading(prev_line):
        return True

    prev_stripped = prev_line.strip()
    curr_stripped = curr_line.strip()
    prev_norm = strip_accents(prev_stripped).lower()
    curr_norm = strip_accents(curr_stripped).lower()

    if is_title_continuation(prev_stripped, curr_stripped):
        return False

    if re.search(r"[.;:!?…]$", prev_stripped):
        return True
    if prev_stripped.endswith(("”", "\"")):
        return True

    prev_letters = [ch for ch in prev_stripped if ch.isalpha()]
    curr_letters = [ch for ch in curr_stripped if ch.isalpha()]

    if prev_letters:
        prev_upper_ratio = sum(ch.isupper() for ch in prev_letters) / len(prev_letters)
        if prev_upper_ratio >= 0.9 and len(prev_stripped) <= 80:
            return True

    if curr_letters:
        curr_upper_ratio = sum(ch.isupper() for ch in curr_letters) / len(curr_letters)
        if curr_upper_ratio >= 0.9 and len(curr_stripped) <= 120:
            return True

    if curr_norm.startswith(
        ("can cu ", "theo de nghi ", "quoc hoi ban hanh", "bo truong ", "chinh phu ban hanh")
    ):
        return True
    if "so:" in prev_norm:
        return True

    return False


def merge_wrapped_lines(lines: Sequence[str]) -> List[str]:
    merged: List[str] = []

    for raw in lines:
        line = re.sub(r"\s+", " ", raw.strip())
        if not line:
            if merged and merged[-1] != "":
                merged.append("")
            continue

        if not merged or merged[-1] == "":
            merged.append(line)
            continue

        prev = merged[-1]
        if should_keep_newline(prev, line):
            merged.append(line)
            continue

        if prev.endswith("-") and not prev.endswith(" -"):
            merged[-1] = prev[:-1] + line
        else:
            merged[-1] = prev + " " + line

    return merged


def clean_document_pages(
    page_texts: Sequence[str],
    config: CleanerConfig | None = None,
) -> str:
    config = config or CleanerConfig()
    pages_lines = [split_non_empty_lines(text) for text in page_texts]
    repeated_margin_lines = find_repeated_margin_lines(pages_lines, config)

    all_lines: List[str] = []
    for page_index, page_lines in enumerate(pages_lines):
        filtered = drop_noise_in_page(
            page_lines,
            page_index,
            repeated_margin_lines,
            config,
        )
        all_lines.extend(filtered)
        all_lines.append("")

    all_lines = trim_trailing_footnote_appendix(all_lines, config)
    merged = merge_wrapped_lines(all_lines)

    text = "\n".join(merged)
    text = re.sub(r"(?m)^\[\d+\].*$\n?", "", text)
    text = re.sub(r"\[(\d{1,4})\]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    marker = "xác thực văn bản hợp nhất"
    lowered = text.lower()
    marker_index = lowered.rfind(marker)
    if marker_index != -1 and marker_index > int(len(text) * 0.55):
        text = text[:marker_index].rstrip()

    return normalize_unicode(text).strip()


def extract_text_from_pdf(pdf_path: str) -> str:
    page_texts = extract_pdf_pages(Path(pdf_path))
    return clean_document_pages(page_texts)


def clean_legal_text(text: str) -> str:
    text = normalize_unicode(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def save_cleaned_text(text: str, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")