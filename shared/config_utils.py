from __future__ import annotations

from pathlib import Path


def find_env_file(start: Path | None = None) -> str:
    """
    Resolve the closest .env file from the provided start path and current cwd.
    Fallback to ".env" so Pydantic keeps default behavior.
    """
    candidates: list[Path] = []

    if start is not None:
        root = start.resolve()
        candidates.extend([root / ".env", *[parent / ".env" for parent in root.parents]])

    cwd = Path.cwd().resolve()
    candidates.extend([cwd / ".env", *[parent / ".env" for parent in cwd.parents]])

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)

        if candidate.exists():
            return key

    return ".env"


def normalize_legacy_path(value: str | None, legacy_map: dict[str, str]) -> str | None:
    if value is None:
        return None

    raw = str(value).strip()
    if not raw:
        return raw

    normalized = raw.replace("\\", "/")
    mapped = legacy_map.get(normalized.lower())
    if mapped:
        return mapped
    return raw

