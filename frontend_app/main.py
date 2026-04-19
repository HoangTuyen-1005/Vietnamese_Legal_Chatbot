from __future__ import annotations

import json
from urllib import error, request

import gradio as gr

from frontend_app.core.config import FrontendSettings, get_settings


def _format_sources(sources: list[dict]) -> str:
    if not sources:
        return "Khong co nguon trich dan."

    blocks: list[str] = []
    for idx, source in enumerate(sources, start=1):
        header_parts = [
            source.get("loai_van_ban"),
            source.get("so_hieu"),
            source.get("dieu"),
            source.get("khoan"),
            source.get("diem"),
        ]
        header = " | ".join([part for part in header_parts if part])
        excerpt = (source.get("trich_doan") or "").strip()

        if header:
            blocks.append(f"**Nguon {idx}:** {header}\n\n> {excerpt}")
        else:
            blocks.append(f"**Nguon {idx}:**\n\n> {excerpt}")

    return "\n\n---\n\n".join(blocks)


def _chat_with_rag_engine(question: str) -> tuple[str, str]:
    settings = get_settings()
    normalized_question = (question or "").strip()

    if not normalized_question:
        return "Vui long nhap cau hoi.", ""

    body = json.dumps({"question": normalized_question}).encode("utf-8")
    endpoint = f"{settings.RAG_ENGINE_BASE_URL.rstrip('/')}/api/chat"
    req = request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=settings.REQUEST_TIMEOUT_SECONDS) as response:
            payload = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return (
            f"RAG engine tra ve loi HTTP {exc.code}.",
            f"```\n{detail}\n```",
        )
    except Exception as exc:
        return (
            "Khong the ket noi toi rag_engine. Hay kiem tra service RAG da chay hay chua.",
            str(exc),
        )

    try:
        result = json.loads(payload)
    except json.JSONDecodeError:
        return (
            "Phan hoi tu rag_engine khong phai JSON hop le.",
            f"```\n{payload}\n```",
        )

    answer = (result.get("answer") or "").strip()
    sources = result.get("sources") or []
    return answer or "(Khong co noi dung tra loi.)", _format_sources(sources)


def build_demo(settings: FrontendSettings | None = None) -> gr.Blocks:
    settings = settings or get_settings()

    with gr.Blocks(title=settings.APP_NAME) as demo:
        gr.Markdown(f"# {settings.APP_NAME}")
        gr.Markdown(
            f"Frontend nay goi den RAG Engine tai: `{settings.RAG_ENGINE_BASE_URL}`"
        )

        question_input = gr.Textbox(
            label="Cau hoi phap ly",
            placeholder="Vi du: Toi trom cap tai san duoc quy dinh tai dieu nao?",
            lines=4,
        )
        answer_output = gr.Textbox(
            label="Cau tra loi",
            lines=12,
            interactive=False,
        )
        gr.Markdown("### Nguon trich dan")
        sources_output = gr.Markdown(value="")

        submit_button = gr.Button("Gui cau hoi")
        clear_button = gr.Button("Xoa")

        submit_button.click(
            fn=_chat_with_rag_engine,
            inputs=question_input,
            outputs=[answer_output, sources_output],
        )
        question_input.submit(
            fn=_chat_with_rag_engine,
            inputs=question_input,
            outputs=[answer_output, sources_output],
        )
        clear_button.click(
            fn=lambda: ("", "", ""),
            inputs=[],
            outputs=[question_input, answer_output, sources_output],
        )

    return demo
