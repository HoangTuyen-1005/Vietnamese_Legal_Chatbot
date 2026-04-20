from __future__ import annotations

import re

import torch
from sentence_transformers import CrossEncoder

RE_DIEU_REFERENCE = re.compile(
    r"(?i)\b(?:điều|dieu)\s+(\d+[A-Za-z]?)\b"
)


def _parse_numbered_label(value: str | None) -> tuple[int, str]:
    if not value:
        return (10**9, "")

    text = str(value)
    match = re.search(r"(\d+)([A-Za-z]?)", text)
    if not match:
        return (10**9, text.lower())
    return (int(match.group(1)), match.group(2).lower())


def _parse_letter_label(value: str | None) -> tuple[int, str]:
    if not value:
        return (10**9, "")

    text = str(value).strip()
    match = re.search(r"([A-Za-z\u0111\u0110])\s*$", text)
    if not match:
        match = re.search(r"diem\s+([A-Za-z\u0111\u0110])", text, flags=re.IGNORECASE)
    if not match:
        return (10**9, text.lower())

    letter = match.group(1).lower()
    if letter == "\u0111":
        return (4, letter)
    if "a" <= letter <= "z":
        return (ord(letter) - ord("a"), letter)
    return (10**9, letter)


def _normalize_dieu_label(value: str) -> str | None:
    text = str(value or "").strip()
    match = re.search(r"(\d+)([A-Za-z]?)", text)
    if not match:
        return None
    number = int(match.group(1))
    suffix = match.group(2).upper()
    return f"Điều {number}{suffix}"


def _extract_referenced_dieu_labels(
    content: str,
    current_dieu: str | None,
    max_refs: int,
) -> list[str]:
    if max_refs <= 0:
        return []

    current_norm = _normalize_dieu_label(current_dieu or "")
    seen: set[str] = set()
    refs: list[str] = []

    for match in RE_DIEU_REFERENCE.finditer(content or ""):
        dieu_label = _normalize_dieu_label(match.group(1))
        if not dieu_label:
            continue
        if current_norm and dieu_label == current_norm:
            continue
        if dieu_label in seen:
            continue

        seen.add(dieu_label)
        refs.append(dieu_label)
        if len(refs) >= max_refs:
            break

    return refs


class Reranker:
    def __init__(self, model_name: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        automodel_args = {"torch_dtype": torch.float16} if device == "cuda" else None

        self.model = CrossEncoder(
            model_name,
            device=device,
            automodel_args=automodel_args,
        )

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        if not candidates:
            return []

        pairs = [[query, doc.get("content", "")] for doc in candidates]
        scores = self.model.predict(pairs)

        enriched = []
        for i, doc in enumerate(candidates):
            item = dict(doc)
            item["rerank_score"] = float(scores[i])
            enriched.append(item)

        enriched.sort(
            key=lambda x: x.get("rerank_score", float("-inf")),
            reverse=True,
        )
        return enriched[:top_k]

    def expand_legal_context(
        self,
        top_chunks: list[dict],
        vector_store,
        follow_referenced_dieu: bool = True,
        max_referenced_dieu_per_chunk: int = 3,
        max_referenced_dieu_total: int = 12,
    ) -> list[dict]:
        if not top_chunks:
            return []

        merged: dict[str, dict] = {}
        referenced_by_law: dict[str, set[str]] = {}

        for chunk in top_chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id:
                merged[chunk_id] = chunk

            meta = chunk.get("metadata", {})
            so_hieu = meta.get("so_hieu")
            dieu = meta.get("dieu")

            if not so_hieu or not dieu:
                continue

            related_chunks = vector_store.get_chunks_by_metadata(
                so_hieu=so_hieu,
                dieu=dieu,
            )

            for related in related_chunks:
                related_id = related.get("chunk_id")
                if related_id:
                    merged[related_id] = related

            if follow_referenced_dieu and max_referenced_dieu_per_chunk > 0:
                refs = _extract_referenced_dieu_labels(
                    content=chunk.get("content", ""),
                    current_dieu=dieu,
                    max_refs=max_referenced_dieu_per_chunk,
                )
                if refs:
                    referenced_by_law.setdefault(str(so_hieu), set()).update(refs)

        if (
            follow_referenced_dieu
            and max_referenced_dieu_total > 0
            and referenced_by_law
        ):
            fetched = 0
            for so_hieu, dieu_labels in referenced_by_law.items():
                if fetched >= max_referenced_dieu_total:
                    break

                ordered_dieu = sorted(dieu_labels, key=_parse_numbered_label)
                for dieu_label in ordered_dieu:
                    if fetched >= max_referenced_dieu_total:
                        break

                    related_chunks = vector_store.get_chunks_by_metadata(
                        so_hieu=so_hieu,
                        dieu=dieu_label,
                    )
                    for related in related_chunks:
                        related_id = related.get("chunk_id")
                        if related_id:
                            merged[related_id] = related
                    fetched += 1

        def sort_key(item: dict):
            meta = item.get("metadata", {})
            return (
                meta.get("so_hieu") or "",
                _parse_numbered_label(meta.get("dieu")),
                _parse_numbered_label(meta.get("khoan")),
                _parse_letter_label(meta.get("diem")),
                item.get("chunk_id") or "",
            )

        return sorted(merged.values(), key=sort_key)
