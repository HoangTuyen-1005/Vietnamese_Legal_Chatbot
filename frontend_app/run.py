import gradio as gr
from frontend_app.main import build_demo
from frontend_app.core.config import get_settings

# CSS siêu xịn xò (ChatGPT-style Dark Mode)
custom_css = """
/* Reset tổng thể */
body, .gradio-container { background-color: #212121 !important; }

/* Cột Sidebar */
#sidebar {
    background-color: #171717 !important;
    padding: 20px 15px !important;
    height: 100vh !important;
    border-right: 1px solid #2f2f2f !important;
}

/* Nút New Chat */
#new-chat-btn {
    background-color: transparent !important;
    border: 1px solid #4a4a4a !important;
    color: white !important;
    justify-content: flex-start !important;
    padding-left: 15px !important;
    border-radius: 8px !important;
}
#new-chat-btn:hover { background-color: #2f2f2f !important; }

/* Text Lịch sử chat */
#history-text h3 { color: #888 !important; font-size: 0.9em !important; text-transform: uppercase; }
#history-text { color: #ececec !important; }

/* Khu vực Chat Chính */
#main-chat-area { padding: 20px 40px !important; }
#app-title { text-align: center; color: white !important; margin-bottom: 5px !important; }
#app-subtitle { text-align: center; color: #888 !important; font-size: 0.9em !important; }

/* Ẩn viền Chatbot và làm trong suốt background */
#chatbot-ui { 
    background-color: transparent !important; 
    border: none !important; 
    box-shadow: none !important; 
}
#chatbot-ui .message-wrap { border: none !important; }

/* Bong bóng chat */
.message.user {
    background-color: #2f2f2f !important;
    border-radius: 18px 18px 0px 18px !important;
    padding: 12px 18px !important;
}
.message.bot {
    background-color: transparent !important;
    border-radius: 18px 18px 18px 0px !important;
    padding: 12px 18px !important;
}

/* Khung nhập liệu nguyên khối (Pill shape) */
#input-container {
    background-color: #2f2f2f !important;
    border-radius: 25px !important;
    padding: 8px 15px !important;
    align-items: center !important;
    margin-top: 15px !important;
    border: 1px solid #4a4a4a !important;
}
#chat-input textarea {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: white !important;
    font-size: 1em !important;
}
#chat-input textarea:focus { border: none !important; box-shadow: none !important; }

/* Nút Gửi */
#submit-btn {
    background-color: white !important;
    color: black !important;
    border-radius: 20px !important;
    font-weight: bold !important;
    border: none !important;
    height: 40px !important;
}
#submit-btn:hover { background-color: #e5e5e5 !important; }
"""

# Theme tối giản
theme = gr.themes.Monochrome(
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_text_color="#ececec",
    block_border_width="0px"
)

if __name__ == "__main__":
    settings = get_settings()
    demo = build_demo(settings=settings)
    
    # Launch kèm theme và css
    demo.launch(
        theme=theme,
        css=custom_css,
        server_name=settings.FRONTEND_HOST,
        server_port=settings.FRONTEND_PORT,
        share=settings.FRONTEND_SHARE,
    )