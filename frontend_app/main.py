import json
from urllib import error, request
import gradio as gr
from frontend_app.core.config import FrontendSettings, get_settings

def _format_sources(sources: list[dict]) -> str:
    if not sources:
        return "Không có nguồn trích dẫn."

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
            blocks.append(f"**Nguồn {idx}:** {header}\n\n> {excerpt}")
        else:
            blocks.append(f"**Nguồn {idx}:**\n\n> {excerpt}")

    return "\n\n---\n\n".join(blocks)


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
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return f"RAG engine trả về lỗi HTTP {exc.code}.\n```\n{detail}\n```"
    except Exception as exc:
        return f"Không thể kết nối tới RAG Engine. Hãy kiểm tra service đã chạy chưa.\nChi tiết: {str(exc)}"

    try:
        result = json.loads(payload)
        if not isinstance(result, dict):
            return f"Phản hồi từ RAG Engine không hợp lệ.\n```\n{payload}\n```"
    except json.JSONDecodeError:
        return f"Phản hồi từ RAG Engine không phải JSON.\n```\n{payload}\n```"

    answer = (result.get("answer") or "").strip()
    sources = result.get("sources") or []
    
    final_response = answer or "(Không có nội dung trả lời.)"
    formatted_sources = _format_sources(sources)
    if formatted_sources != "Không có nguồn trích dẫn.":
        final_response += f"\n\n---\n### 📚 Nguồn trích dẫn:\n{formatted_sources}"
        
    return final_response


def _extract_text(msg_content):
    """Hàm an toàn để trích xuất text đề phòng Gradio 6.0 trả về dạng block list multimodal"""
    if isinstance(msg_content, str):
        return msg_content
    if isinstance(msg_content, list) and len(msg_content) > 0:
        if isinstance(msg_content[0], dict) and "text" in msg_content[0]:
            return msg_content[0]["text"]
    return str(msg_content)


def add_text(history, text):
    if not text:
        return history, ""
    
    # Nâng cấp lên định dạng OpenAI Message
    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": "⏳ *Đang tra cứu cơ sở dữ liệu luật...*"})
    return history, ""


def generate_bot_response(history):
    # Lấy object tin nhắn user liền trước
    last_user_msg = history[-2]
    raw_content = last_user_msg["content"] if isinstance(last_user_msg, dict) else last_user_msg.content
    
    user_message = _extract_text(raw_content)
    bot_message = _get_rag_response(user_message)
    
    # Cập nhật kết quả vào object assistant cuối cùng
    history[-1] = {"role": "assistant", "content": bot_message}
    return history


def update_sidebar(history, current_sidebar_list):
    if not history:
        return current_sidebar_list, "### Lịch sử trò chuyện\n*Chưa có đoạn chat nào.*"
    
    # Tìm kiếm toàn bộ lịch sử để lấy các câu hỏi của user một cách an toàn
    user_messages = []
    for msg in history:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")
        if role == "user":
            raw_content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
            user_messages.append(_extract_text(raw_content))
            
    if not user_messages:
        return current_sidebar_list, "### Lịch sử trò chuyện\n*Chưa có đoạn chat nào.*"
        
    latest_question = user_messages[-1]
    short_q = latest_question[:30] + "..." if len(latest_question) > 30 else latest_question
    
    if short_q not in current_sidebar_list:
        current_sidebar_list.insert(0, short_q)
    
    markdown_str = "### Lịch sử trò chuyện\n\n"
    for q in current_sidebar_list:
        markdown_str += f"💬 {q}\n\n"
        
    return current_sidebar_list, markdown_str


def build_demo(settings: FrontendSettings | None = None) -> gr.Blocks:
    settings = settings or get_settings()

    with gr.Blocks(title="Vietnamese Legal Chatbot") as demo:
        sidebar_state = gr.State([])

        with gr.Row(equal_height=True):
            # ================= CỘT TRÁI: SIDEBAR =================
            with gr.Column(scale=1, elem_id="sidebar", min_width=260):
                gr.Markdown("## ⚖️ Workspace", elem_classes="text-center")
                new_chat_btn = gr.Button("➕ Đoạn chat mới", elem_id="new-chat-btn")
                
                gr.HTML("<hr style='border-color: #333; margin: 20px 0;'/>")
                
                sidebar_history = gr.Markdown("### Lịch sử trò chuyện\n*Chưa có đoạn chat nào.*", elem_id="history-text")

            # ================= CỘT PHẢI: KHU VỰC CHAT =================
            with gr.Column(scale=5, elem_id="main-chat-area"):
                gr.Markdown(f"## {settings.APP_NAME}", elem_id="app-title")
                gr.Markdown(f"Đang kết nối RAG Engine tại: `{settings.RAG_ENGINE_BASE_URL}`", elem_id="app-subtitle")

                chatbot = gr.Chatbot(
                    label="Trợ lý Luật Đất đai",
                    height=600,
                    show_label=False,
                    # ĐÃ ĐƯỢC XÓA BỎ: type="tuples"
                    elem_id="chatbot-ui",
                    avatar_images=(None, "https://cdn-icons-png.flaticon.com/512/6098/6098020.png"),
                )

                with gr.Row(elem_id="input-container"):
                    question_input = gr.Textbox(
                        show_label=False,
                        placeholder="Hỏi bất kỳ điều gì về Luật Đất đai...",
                        lines=1,
                        scale=8,
                        container=False,
                        elem_id="chat-input"
                    )
                    submit_button = gr.Button("Gửi 🚀", scale=1, elem_id="submit-btn")

        # ================= XỬ LÝ SỰ KIỆN =================
        question_input.submit(
            add_text, [chatbot, question_input], [chatbot, question_input], queue=False
        ).then(
            generate_bot_response, chatbot, chatbot
        ).then(
            update_sidebar, [chatbot, sidebar_state], [sidebar_state, sidebar_history]
        )

        submit_button.click(
            add_text, [chatbot, question_input], [chatbot, question_input], queue=False
        ).then(
            generate_bot_response, chatbot, chatbot
        ).then(
            update_sidebar, [chatbot, sidebar_state], [sidebar_state, sidebar_history]
        )

        new_chat_btn.click(
            lambda: ([], []), None, [chatbot, sidebar_state], queue=False
        ).then(
            lambda: "### Lịch sử trò chuyện\n*Chưa có đoạn chat nào.*", None, sidebar_history
        )

    return demo