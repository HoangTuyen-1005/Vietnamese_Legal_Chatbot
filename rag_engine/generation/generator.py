from __future__ import annotations

import os
from typing import Any

try:
    from google import genai
    from google.genai import types
except Exception as exc:  # pragma: no cover
    genai = None
    types = None
    _IMPORT_ERROR = exc
else:  # pragma: no cover
    _IMPORT_ERROR = None


class AnswerGenerator:
    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        temperature: float = 0.0,
    ):
        if genai is None or types is None:
            raise ImportError(
                "google-genai is not installed. Add it to requirements and reinstall dependencies."
            ) from _IMPORT_ERROR

        self.model_name = model_name
        self.temperature = float(temperature)
        self.api_key = (api_key or os.getenv("GEMINI_API_KEY") or "").strip()

        if not self.api_key:
            raise ValueError(
                "Missing GEMINI_API_KEY. Please set GEMINI_API_KEY in your .env file."
            )

        self.client = genai.Client(api_key=self.api_key)

    def _extract_text(self, response: Any) -> str:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        candidates = getattr(response, "candidates", None) or []
        parts_text: list[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            parts = getattr(content, "parts", None) or []
            for part in parts:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text.strip():
                    parts_text.append(part_text.strip())

        return "\n".join(parts_text).strip()

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0.")

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=max_new_tokens,
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_NONE",
                    ),
                ],
            ),
        )

        text = self._extract_text(response)
        if not text:
            raise RuntimeError("Gemini returned empty response content.")

        return text
