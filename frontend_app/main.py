import base64
import json
import os
from datetime import datetime
from typing import Any
from urllib import error, request

import bcrypt
import gradio as gr
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, create_engine, event
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from .core.config import get_settings

settings = get_settings()
BOT_GREETING = "Xin chào tôi là chatbot pháp luật, tôi có thể giúp gì được cho bạn ?"

DATABASE_URL = settings.DATABASE_URL
engine_options = {}
if DATABASE_URL.startswith("sqlite"):
    engine_options["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **engine_options)
if DATABASE_URL.startswith("sqlite"):

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=MEMORY")
        cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=True)
    password_hash = Column(String(100))
    sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String(100))
    created_at = Column(DateTime, default=datetime.now)
    user = relationship("User", back_populates="sessions")
    messages = relationship(
        "ChatMessage",
        back_populates="chat_session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"))
    role = Column(String(20))
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    chat_session = relationship("ChatSession", back_populates="messages")


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    if DATABASE_URL.startswith("sqlite"):
        with engine.begin() as connection:
            columns = [row[1] for row in connection.exec_driver_sql("PRAGMA table_info(users)").fetchall()]
            if "email" not in columns:
                connection.exec_driver_sql("ALTER TABLE users ADD COLUMN email VARCHAR(255)")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


def _welcome_markup(username: str | None) -> str:
    display_name = username or "bạn"
    return f"""
    <div class="account-card">
        <div class="account-label">Đang đăng nhập</div>
        <div class="account-name">Xin chào, {display_name}</div>
    </div>
    """


def _default_avatar_svg(name: str) -> str:
    if "bot" in name:
        return """
        <svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96">
          <rect x="13" y="25" width="70" height="56" rx="19" fill="#58a6ff"/>
          <circle cx="35" cy="52" r="7" fill="#ffffff"/>
          <circle cx="61" cy="52" r="7" fill="#ffffff"/>
          <path d="M39 66h18" stroke="#ffffff" stroke-width="6" stroke-linecap="round"/>
          <path d="M48 13v13" stroke="#58a6ff" stroke-width="7" stroke-linecap="round"/>
          <circle cx="48" cy="11" r="8" fill="#58a6ff"/>
        </svg>
        """

    return """
    <svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96">
      <circle cx="48" cy="32" r="21" fill="#2f81f7"/>
      <path d="M16 88c6.4-23 22.9-33.5 32-33.5S73.6 65 80 88" fill="#2f81f7"/>
    </svg>
    """


def _avatar_data_uri(path: str) -> dict[str, str]:
    actual_path = path
    mime_type = "image/svg+xml" if path.lower().endswith(".svg") else "image/png"
    if not os.path.exists(actual_path):
        svg_path = os.path.splitext(path)[0] + ".svg"
        if os.path.exists(svg_path):
            actual_path = svg_path
            mime_type = "image/svg+xml"
        else:
            encoded = base64.b64encode(_default_avatar_svg(os.path.basename(path)).encode("utf-8")).decode("ascii")
            data_uri = f"data:image/svg+xml;base64,{encoded}"
            return {"path": data_uri, "url": data_uri}

    with open(actual_path, "rb") as avatar_file:
        encoded = base64.b64encode(avatar_file.read()).decode("ascii")
    data_uri = f"data:{mime_type};base64,{encoded}"
    return {"path": data_uri, "url": data_uri}


def login_user(username, password):
    clean_username = (username or "").strip()
    if not clean_username or not password:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            None,
            None,
            "Vui lòng nhập đầy đủ tên đăng nhập và mật khẩu.",
            _welcome_markup(None),
        )

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == clean_username).first()
        if user and verify_password(password, user.password_hash):
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                user.id,
                user.username,
                "",
                _welcome_markup(user.username),
            )
    finally:
        db.close()

    return (
        gr.update(visible=True),
        gr.update(visible=False),
        None,
        None,
        "Sai tên đăng nhập hoặc mật khẩu.",
        _welcome_markup(None),
    )


def register_user(username, email, password, confirm_password):
    clean_username = (username or "").strip()
    clean_email = (email or "").strip().lower()
    if not clean_username or not clean_email or not password or not confirm_password:
        return "Vui lòng nhập đầy đủ tên đăng nhập, email và mật khẩu."
    if "@" not in clean_email or "." not in clean_email:
        return "Email không hợp lệ."
    if password != confirm_password:
        return "Mật khẩu xác nhận không khớp."
    if len(password) < 6:
        return "Mật khẩu nên có ít nhất 6 ký tự."

    db = SessionLocal()
    try:
        if db.query(User).filter(User.username == clean_username).first():
            return "Tên đăng nhập đã tồn tại."
        if db.query(User).filter(User.email == clean_email).first():
            return "Email đã được sử dụng."

        new_user = User(username=clean_username, email=clean_email, password_hash=hash_password(password))
        db.add(new_user)
        db.commit()
        return "Đăng ký thành công. Hãy đăng nhập bằng tài khoản vừa tạo."
    finally:
        db.close()


def logout_user():
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        None,
        None,
        _default_chat_history(),
        None,
        gr.update(choices=[], value=None),
        _welcome_markup(None),
    )


def _default_chat_history():
    return [{"role": "assistant", "content": BOT_GREETING}]


def _with_default_greeting(history):
    history = history or []
    first_message = history[0] if history else None
    if (
        isinstance(first_message, dict)
        and first_message.get("role") == "assistant"
        and first_message.get("content") == BOT_GREETING
    ):
        return history
    return _default_chat_history() + history


def start_new_chat():
    return (_default_chat_history(), None, gr.update(value=None))


def _format_sources(sources: list[dict]) -> str:
    if not sources:
        return "Không có nguồn trích dẫn."

    top_sources = sources[:5]
    blocks: list[str] = []
    for idx, source in enumerate(top_sources, start=1):
        header = " | ".join(
            [
                part
                for part in [
                    source.get("loai_van_ban"),
                    source.get("so_hieu"),
                    source.get("dieu"),
                    source.get("khoan"),
                    source.get("diem"),
                ]
                if part
            ]
        )
        excerpt = (source.get("trich_doan") or "").strip()
        if header:
            blocks.append(f"**Nguồn {idx}:** {header}\n> {excerpt}")
        else:
            blocks.append(f"**Nguồn {idx}:**\n> {excerpt}")

    return (
        f"<details><summary><b>Xem {len(top_sources)} nguồn trích dẫn</b></summary>\n\n"
        + "\n\n".join(blocks)
        + "\n\n</details>"
    )


def _get_rag_response(question: str) -> str:
    def busy_message(detail: str = "") -> str:
        cleaned_detail = detail.strip()
        if cleaned_detail.startswith("<") and cleaned_detail.endswith(">"):
            cleaned_detail = cleaned_detail[1:-1]
        return f"Server đang bận, mong bạn quay lại sau. Chi tiết: <{cleaned_detail}>"

    normalized_question = (question or "").strip()
    if not normalized_question:
        return "Vui lòng nhập câu hỏi."

    body = json.dumps({"question": normalized_question}).encode("utf-8")
    endpoint = f"{settings.RAG_ENGINE_BASE_URL.rstrip('/')}/api/chat"
    req = request.Request(endpoint, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=settings.REQUEST_TIMEOUT_SECONDS) as response:
            payload = response.read().decode("utf-8")
    except error.HTTPError as exc:
        return busy_message(f"HTTP {exc.code}")
    except Exception as exc:
        return busy_message(str(exc))

    try:
        result = json.loads(payload)
        if not isinstance(result, dict):
            return busy_message("Phản hồi từ RAG Engine không hợp lệ")
    except json.JSONDecodeError as exc:
        return busy_message(str(exc))

    answer = (result.get("answer") or "").strip()
    formatted_sources = _format_sources(result.get("sources") or [])
    return answer + (f"\n\n{formatted_sources}" if formatted_sources != "Không có nguồn trích dẫn." else "")


def _extract_text(msg_content):
    if isinstance(msg_content, str):
        return msg_content
    if isinstance(msg_content, list) and len(msg_content) > 0 and "text" in msg_content[0]:
        return msg_content[0]["text"]
    return str(msg_content)


def add_user_message(history, text):
    clean_text = (text or "").strip()
    history = _with_default_greeting(history)
    if not clean_text:
        return history, ""

    history.append({"role": "user", "content": clean_text})
    history.append({"role": "assistant", "content": "*Đang phân tích cơ sở dữ liệu luật...*"})
    return history, ""


def generate_and_save_response(history, user_id, session_id):
    if not user_id:
        return history, session_id

    user_text = _extract_text(history[-2]["content"])
    bot_response = _get_rag_response(user_text)
    history[-1] = {"role": "assistant", "content": bot_response}

    db = SessionLocal()
    try:
        if not session_id:
            title = user_text[:30] + "..." if len(user_text) > 30 else user_text
            new_session = ChatSession(user_id=user_id, title=title)
            db.add(new_session)
            db.commit()
            session_id = new_session.id

        db.add(ChatMessage(session_id=session_id, role="user", content=user_text))
        db.add(ChatMessage(session_id=session_id, role="assistant", content=bot_response))
        db.commit()
    finally:
        db.close()

    return history, session_id


def load_user_sessions(user_id, selected_session_id=None):
    if not user_id:
        return gr.update(choices=[], value=None)

    db = SessionLocal()
    try:
        sessions = (
            db.query(ChatSession)
            .filter(ChatSession.user_id == user_id)
            .order_by(ChatSession.created_at.desc())
            .all()
        )
        choices = [(s.title, s.id) for s in sessions]
        session_ids = {s.id for s in sessions}
        selected_value = selected_session_id if selected_session_id in session_ids else None
        return gr.update(choices=choices, value=selected_value)
    finally:
        db.close()


def load_chat_history(session_id):
    if not session_id:
        return _default_chat_history()

    db = SessionLocal()
    try:
        messages = (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)
            .all()
        )
        return _with_default_greeting([{"role": msg.role, "content": msg.content} for msg in messages])
    finally:
        db.close()


def build_modern_demo(settings: Any) -> gr.Blocks:
    init_db()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    user_avatar = _avatar_data_uri(os.path.join(current_dir, "assets", "user_avatar.svg"))
    bot_avatar = _avatar_data_uri(os.path.join(current_dir, "assets", "bot_avatar.svg"))

    with gr.Blocks(title=settings.APP_NAME, elem_id="legal-app") as demo:
        current_user = gr.State(None)
        current_username = gr.State(None)
        current_session = gr.State(None)

        with gr.Column(visible=True, elem_id="auth-container") as auth_view:
            gr.Markdown(
                """
                <div id="auth-title">
                    <h1>Legal AI</h1>
                    <p>Đăng nhập để tiếp tục phiên tư vấn pháp lý của bạn.</p>
                </div>
                """
            )
            with gr.Tabs(elem_id="auth-tabs"):
                with gr.TabItem("Đăng nhập", elem_id="login-panel"):
                    l_user = gr.Textbox(label="Tên đăng nhập", elem_classes="auth-input")
                    l_pass = gr.Textbox(label="Mật khẩu", type="password", elem_classes="auth-input")
                    l_msg = gr.Markdown(elem_id="login-message")
                    l_btn = gr.Button("Đăng nhập", elem_classes="auth-btn")

                with gr.TabItem("Đăng ký", elem_id="register-panel"):
                    r_user = gr.Textbox(label="Tên đăng nhập", elem_classes="auth-input")
                    r_email = gr.Textbox(label="Email", elem_classes="auth-input")
                    r_pass = gr.Textbox(label="Mật khẩu", type="password", elem_classes="auth-input")
                    r_confirm = gr.Textbox(label="Xác nhận mật khẩu", type="password", elem_classes="auth-input")
                    r_msg = gr.Markdown(elem_id="register-message")
                    r_btn = gr.Button("Tạo tài khoản", elem_classes="auth-btn")

        with gr.Row(visible=False, elem_id="app-shell", equal_height=True) as app_view:
            with gr.Column(scale=1, min_width=300, elem_id="sidebar"):
                gr.Markdown(
                    """
                    <div id="brand-card">
                        <h2>Legal AI</h2>
                        <p>Trợ lý pháp lý tiếng Việt</p>
                    </div>
                    """
                )

                new_chat_btn = gr.Button("+ Phiên chat mới", elem_id="new-chat-btn")

                with gr.Column(elem_id="history-section"):
                    gr.Markdown("### Lịch sử hội thoại", elem_id="history-title")
                    session_list = gr.Radio(
                        label="",
                        choices=[],
                        interactive=True,
                        show_label=False,
                        elem_id="history-list",
                    )

                with gr.Column(elem_id="account-section"):
                    welcome_text = gr.Markdown(_welcome_markup(None), elem_id="welcome-text")
                    logout_btn = gr.Button("Đăng xuất", variant="stop", size="sm", elem_id="logout-btn")

            with gr.Column(scale=4, elem_id="workspace"):
                gr.Markdown(
                    """
                    <div id="chat-header">
                        <div>
                            <h1>Tư vấn pháp lý</h1>
                            <p>Đặt câu hỏi rõ ràng để nhận câu trả lời kèm nguồn trích dẫn khi có.</p>
                        </div>
                    </div>
                    """
                )

                chatbot = gr.Chatbot(
                    value=_default_chat_history(),
                    show_label=False,
                    elem_id="chatbot-ui",
                    height="100%",
                    layout="bubble",
                    avatar_images=(user_avatar, bot_avatar),
                    placeholder="<div class='chat-placeholder'>Bắt đầu bằng một câu hỏi pháp lý của bạn.</div>",
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

        r_btn.click(register_user, inputs=[r_user, r_email, r_pass, r_confirm], outputs=r_msg)
        l_btn.click(
            login_user,
            inputs=[l_user, l_pass],
            outputs=[auth_view, app_view, current_user, current_username, l_msg, welcome_text],
        ).then(load_user_sessions, inputs=[current_user, current_session], outputs=session_list)

        logout_btn.click(
            logout_user,
            outputs=[
                auth_view,
                app_view,
                current_user,
                current_username,
                chatbot,
                current_session,
                session_list,
                welcome_text,
            ],
        )

        def process_chat(history, text, uid, sid):
            if not (text or "").strip():
                yield _with_default_greeting(history), sid, gr.update(value="")
                return

            hist, _ = add_user_message(history, text)
            yield hist, sid, gr.update(value="")

            final_hist, new_sid = generate_and_save_response(hist, uid, sid)
            yield final_hist, new_sid, gr.update()

        for event in [question_input.submit, submit_button.click]:
            event(
                process_chat,
                inputs=[chatbot, question_input, current_user, current_session],
                outputs=[chatbot, current_session, question_input],
            ).then(load_user_sessions, inputs=[current_user, current_session], outputs=session_list)

        new_chat_btn.click(start_new_chat, outputs=[chatbot, current_session, session_list])

        session_list.change(load_chat_history, inputs=[session_list], outputs=[chatbot]).then(
            lambda x: x, inputs=[session_list], outputs=[current_session]
        )

    return demo


def build_demo(settings: Any | None = None) -> gr.Blocks:
    return build_modern_demo(settings or get_settings())
