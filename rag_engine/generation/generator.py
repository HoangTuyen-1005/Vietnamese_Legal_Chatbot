from __future__ import annotations

import os
from typing import Any

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except Exception as exc:  # pragma: no cover
    genai = None
    GenerationConfig = None
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
        if genai is None or GenerationConfig is None:
            raise ImportError(
                "google-generativeai is not installed. Add it to requirements and reinstall dependencies."
            ) from _IMPORT_ERROR

        self.model_name = model_name
        self.temperature = float(temperature)
        self.api_key = (api_key or os.getenv("GEMINI_API_KEY") or "").strip()

        if not self.api_key:
            raise ValueError(
                "Missing GEMINI_API_KEY. Please set GEMINI_API_KEY in your .env file."
            )

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name=self.model_name)

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

        response = self.model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=max_new_tokens,
            ),
            safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
        )

        print(f"--- DEBUG INFO ---")
        print(f"Finish Reason: {response.candidates[0].finish_reason}")
        print(f"Safety Ratings: {response.candidates[0].safety_ratings}")
        print(f"Full Text Received: {response.text}")
        print(f"------------------")

        text = self._extract_text(response)
        if not text:
            raise RuntimeError("Gemini returned empty response content.")

        return text
