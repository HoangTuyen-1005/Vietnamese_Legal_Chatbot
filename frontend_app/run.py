import json
from urllib import request
import gradio as gr
from typing import Any
from frontend_app.core.config import get_settings

settings = get_settings()

# ==================== PHẦN 1: LOGIC FUNCTIONS (GIỮ NGUYÊN) ====================
def _format_sources(sources: list[dict]) -> str:
    if not sources:
        return "Không có nguồn trích dẫn."

    top_sources = sources[:5]
    blocks: list[str] = []

    for idx, source in enumerate(top_sources, start=1):
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
            blocks.append(f"**Nguồn {idx}:** {header}\n> {excerpt}")
        else:
            blocks.append(f"**Nguồn {idx}:**\n> {excerpt}")

    content = "\n\n".join(blocks)

    return f"""<details>
<summary>📚 <b>Xem {len(top_sources)} nguồn trích dẫn</b></summary>

{content}

</details>"""


def _get_rag_response(question: str) -> str:
    settings = get_settings()
    normalized_question = (question or "").strip()
    if not normalized_question:
        return "Vui lòng nhập câu hỏi."

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
    except Exception as exc:
        return f"Lỗi kết nối RAG Engine. Chi tiết: {str(exc)}"

    try:
        result = json.loads(payload)
    except json.JSONDecodeError:
        return "Phản hồi từ RAG Engine không hợp lệ."

    answer = (result.get("answer") or "").strip()
    sources = result.get("sources") or []

    final_response = answer or "(Không có nội dung trả lời.)"
    formatted_sources = _format_sources(sources)

    if formatted_sources != "Không có nguồn trích dẫn.":
        final_response += f"\n\n{formatted_sources}"

    return final_response


def _extract_text(msg_content):
    if isinstance(msg_content, str):
        return msg_content
    if isinstance(msg_content, list) and len(msg_content) > 0 and "text" in msg_content[0]:
        return msg_content[0]["text"]
    return str(msg_content)


def add_text(history, text):
    if not text:
        return history, ""
    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": "⏳ *Đang phân tích cơ sở dữ liệu luật...*"})
    return history, ""


def generate_bot_response(history):
    last_user_msg = history[-2]
    user_message = _extract_text(
        last_user_msg["content"] if isinstance(last_user_msg, dict) else last_user_msg.content
    )
    history[-1] = {"role": "assistant", "content": _get_rag_response(user_message)}
    return history


def update_sidebar(history, current_sidebar_list):
    if not history:
        return current_sidebar_list, "### LỊCH SỬ HỘI THOẠI\n*Chưa có đoạn chat nào.*"

    user_messages = [
        _extract_text(msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", ""))
        for msg in history
        if (msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")) == "user"
    ]

    if not user_messages:
        return current_sidebar_list, "### LỊCH SỬ HỘI THOẠI\n*Chưa có đoạn chat nào.*"

    latest_question = user_messages[-1]
    short_q = latest_question[:36] + "..." if len(latest_question) > 36 else latest_question

    if short_q not in current_sidebar_list:
        current_sidebar_list.insert(0, short_q)

    return (
        current_sidebar_list,
        "### LỊCH SỬ HỘI THOẠI\n\n" + "".join([f"🔹 {q}\n\n" for q in current_sidebar_list]),
    )


# ==================== PHẦN 2: CSS GIAO DIỆN MỚI ====================
CREATIVE_CSS = CREATIVE_CSS = """
:root {
    --bg: #0b0d12;
    --bg-elev: #11151c;
    --bg-elev-2: #151a22;
    --surface: rgba(255,255,255,0.03);
    --surface-2: rgba(255,255,255,0.05);
    --border: rgba(255,255,255,0.07);
    --text: #f5f7fb;
    --muted: #9aa4b2;
    --accent: #6ea8fe;
    --accent-2: #8ec5ff;
    --gold: #d4b06a;
    --user-bubble: #18202b;
    --bot-bubble: #121720;
    --sidebar-w: 300px;
    --radius-xl: 24px;
    --radius-lg: 18px;
    --radius-md: 14px;
    --shadow-lg: 0 24px 60px rgba(0, 0, 0, 0.35);
    --shadow-sm: 0 10px 28px rgba(0, 0, 0, 0.22);
}

html, body, .gradio-container {
    margin: 0 !important;
    padding: 0 !important;
    min-height: 100vh !important;
    background:
        radial-gradient(circle at top left, rgba(110,168,254,0.08), transparent 24%),
        radial-gradient(circle at bottom right, rgba(212,176,106,0.06), transparent 18%),
        linear-gradient(180deg, #080a0e 0%, #0c1016 100%) !important;
    color: var(--text) !important;
    font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}

.gradio-container * {
    box-sizing: border-box !important;
    border-color: transparent !important;
}

#app-shell {
    padding: 18px !important;
    gap: 18px !important;
}

/* LEFT SIDEBAR */
#sidebar {
    min-height: calc(100vh - 36px) !important;
    background: rgba(17, 21, 28, 0.9) !important;
    backdrop-filter: blur(14px) !important;
    border: 1px solid var(--border) !important;
    border-radius: 28px !important;
    box-shadow: var(--shadow-lg) !important;
    padding: 18px !important;
}

#brand-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02)) !important;
    border: 1px solid var(--border) !important;
    border-radius: 20px !important;
    padding: 18px !important;
    margin-bottom: 14px !important;
}

#brand-card .eyebrow {
    display: inline-flex !important;
    align-items: center !important;
    gap: 8px !important;
    padding: 6px 10px !important;
    border-radius: 999px !important;
    background: rgba(110,168,254,0.10) !important;
    color: var(--accent-2) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px !important;
}

#brand-card .title {
    margin-top: 12px !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
}

#brand-card .desc {
    margin-top: 6px !important;
    color: var(--muted) !important;
    font-size: 0.92rem !important;
    line-height: 1.45 !important;
}

#new-chat-btn {
    width: 100% !important;
    height: 46px !important;
    border: none !important;
    border-radius: 14px !important;
    background: linear-gradient(135deg, #6ea8fe, #8ec5ff) !important;
    color: #0b1220 !important;
    font-weight: 700 !important;
    box-shadow: 0 12px 28px rgba(110,168,254,0.24) !important;
    transition: transform .18s ease, box-shadow .18s ease !important;
}
#new-chat-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 16px 34px rgba(110,168,254,0.30) !important;
}

#history-text {
    margin-top: 14px !important;
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid var(--border) !important;
    border-radius: 20px !important;
    padding: 16px !important;
    min-height: 320px !important;
    color: var(--text) !important;
}

#history-text h3 {
    margin: 0 0 12px 0 !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    color: var(--gold) !important;
}

/* MAIN WORKSPACE */
#workspace {
    min-height: calc(100vh - 36px) !important;
}

#workspace-card {
    min-height: calc(100vh - 36px) !important;
    background: rgba(17, 21, 28, 0.78) !important;
    backdrop-filter: blur(14px) !important;
    border: 1px solid var(--border) !important;
    border-radius: 28px !important;
    box-shadow: var(--shadow-lg) !important;
    padding: 16px !important;
    display: flex !important;
    flex-direction: column !important;
}

#topbar {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 12px !important;
    padding: 8px 6px 14px 6px !important;
    border-bottom: 1px solid rgba(255,255,255,0.05) !important;
    margin-bottom: 12px !important;
}

#topbar .left {
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
}

#topbar .dot {
    width: 10px !important;
    height: 10px !important;
    border-radius: 999px !important;
    background: #22c55e !important;
    box-shadow: 0 0 0 4px rgba(34,197,94,0.14) !important;
}

#topbar .title {
    font-size: 1rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
}

#topbar .meta {
    color: var(--muted) !important;
    font-size: 0.85rem !important;
}

#topbar .pill {
    padding: 8px 12px !important;
    border-radius: 999px !important;
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
    font-size: 0.82rem !important;
}

/* CHAT AREA */
#chatbot-ui {
    flex: 1 1 auto !important;
    height: calc(100vh - 180px) !important;
    background:
        linear-gradient(180deg, rgba(255,255,255,0.015), rgba(255,255,255,0.02)) !important;
    border: 1px solid var(--border) !important;
    border-radius: 22px !important;
    padding: 14px !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.02) !important;
}

.message.user {
    background: linear-gradient(180deg, rgba(110,168,254,0.10), rgba(110,168,254,0.06)) !important;
    border: 1px solid rgba(110,168,254,0.16) !important;
    color: #f8fbff !important;
    border-radius: 18px !important;
    box-shadow: var(--shadow-sm) !important;
}

.message.bot {
    background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.015)) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    color: var(--text) !important;
    border-radius: 18px !important;
}

.message-avatar {
    width: 38px !important;
    height: 38px !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

/* CITATION DROPDOWN */
.message.bot details {
    margin-top: 16px !important;
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(212,176,106,0.16) !important;
    border-radius: 14px !important;
    padding: 12px 14px !important;
}

.message.bot details[open] {
    background: rgba(255,255,255,0.05) !important;
    border-color: rgba(212,176,106,0.24) !important;
}

.message.bot summary {
    cursor: pointer !important;
    color: var(--gold) !important;
    font-weight: 600 !important;
    outline: none !important;
}

.message.bot blockquote {
    margin-left: 0 !important;
    padding-left: 12px !important;
    border-left: 3px solid var(--gold) !important;
    color: #d7dbe2 !important;
}

/* COMPOSER */
#composer-wrap {
    margin-top: 14px !important;
    padding: 10px !important;
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
    align-items: end !important;
    box-shadow: var(--shadow-sm) !important;
}

#chat-input textarea {
    background: transparent !important;
    color: #ffffff !important;
    font-size: 1rem !important;
    padding: 12px 12px !important;
    line-height: 1.5 !important;
}

#chat-input textarea::placeholder {
    color: var(--muted) !important;
}

#submit-btn {
    height: 46px !important;
    min-width: 112px !important;
    border: none !important;
    border-radius: 14px !important;
    background: linear-gradient(135deg, #6ea8fe, #8ec5ff) !important;
    color: #08111f !important;
    font-weight: 700 !important;
    box-shadow: 0 12px 28px rgba(110,168,254,0.24) !important;
    transition: transform .18s ease, box-shadow .18s ease !important;
}
#submit-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 16px 34px rgba(110,168,254,0.30) !important;
}

/* EMPTY / TYPOGRAPHY CLEANUP */
#workspace-card p,
#workspace-card span,
#workspace-card div,
#sidebar p,
#sidebar span,
#sidebar div {
    -webkit-font-smoothing: antialiased !important;
    text-rendering: optimizeLegibility !important;
}

@media (max-width: 960px) {
    #app-shell {
        padding: 12px !important;
        gap: 12px !important;
    }

    #sidebar, #workspace-card {
        min-height: auto !important;
    }

    #chatbot-ui {
        height: 62vh !important;
    }

    #submit-btn {
        min-width: 88px !important;
    }
}
"""


def build_modern_demo(settings: Any) -> gr.Blocks:
    with gr.Blocks(title=settings.APP_NAME) as demo:
        sidebar_state = gr.State([])

        with gr.Row(elem_id="app-shell", equal_height=True):
            with gr.Column(scale=1, min_width=300, elem_id="sidebar"):
                gr.Markdown(
                    """
                    <div id="brand-card">
                        <div class="eyebrow">● Legal Assistant</div>
                        <div class="title">Chat pháp lý</div>
                        <div class="desc">Tra cứu và hỏi đáp với nguồn trích dẫn rõ ràng.</div>
                    </div>
                    """
                )

                new_chat_btn = gr.Button("Phiên mới", elem_id="new-chat-btn")

                sidebar_history = gr.Markdown(
                    "### Lịch sử\n*Chưa có đoạn chat nào.*",
                    elem_id="history-text"
                )

            with gr.Column(scale=4, elem_id="workspace"):
                with gr.Column(elem_id="workspace-card"):
                    gr.Markdown(
                        """
                        <div id="topbar">
                            <div class="left">
                                <div class="dot"></div>
                                <div>
                                    <div class="title">Tư vấn pháp lý thông minh</div>
                                    <div class="meta">Production chat workspace</div>
                                </div>
                            </div>
                            <div class="pill">RAG Engine</div>
                        </div>
                        """
                    )

                    chatbot = gr.Chatbot(
                        show_label=False,
                        elem_id="chatbot-ui",
                        avatar_images=(
                            "https://i.ibb.co/6P0xWzM/modern-user-avatar.png",
                            "https://i.ibb.co/v4bN8F4/legal-bot-avatar.png"
                        ),
                    )

                    with gr.Row(elem_id="composer-wrap"):
                        question_input = gr.Textbox(
                            show_label=False,
                            placeholder="Nhập câu hỏi pháp lý...",
                            lines=1,
                            max_lines=5,
                            scale=8,
                            container=False,
                            elem_id="chat-input",
                            autofocus=True,
                        )
                        submit_button = gr.Button("Gửi", scale=1, elem_id="submit-btn")

        question_input.submit(
            add_text,
            [chatbot, question_input],
            [chatbot, question_input],
            queue=False
        ).then(
            generate_bot_response,
            chatbot,
            chatbot
        ).then(
            update_sidebar,
            [chatbot, sidebar_state],
            [sidebar_state, sidebar_history]
        )

        submit_button.click(
            add_text,
            [chatbot, question_input],
            [chatbot, question_input],
            queue=False
        ).then(
            generate_bot_response,
            chatbot,
            chatbot
        ).then(
            update_sidebar,
            [chatbot, sidebar_state],
            [sidebar_state, sidebar_history]
        )

        new_chat_btn.click(
            lambda: ([], []),
            None,
            [chatbot, sidebar_state],
            queue=False
        ).then(
            lambda: "### Lịch sử\n*Chưa có đoạn chat nào.*",
            None,
            sidebar_history
        )

    return demo


# ==================== PHẦN 4: CHẠY APP ====================
if __name__ == "__main__":
    settings = get_settings()
    demo = build_modern_demo(settings=settings)
    theme = gr.themes.Base(
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
    )
    print(
        f"✨ Đang khởi chạy giao diện Legal AI tại "
        f"http://{settings.FRONTEND_HOST}:{settings.FRONTEND_PORT} ..."
    )
    demo.launch(
        server_name=settings.FRONTEND_HOST,
        server_port=settings.FRONTEND_PORT,
        share=settings.FRONTEND_SHARE,
        theme=theme,
        css=CREATIVE_CSS
    )