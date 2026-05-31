import os
import socket

import gradio as gr

from .core.config import get_settings
from .main import build_modern_demo


def _find_available_port(host: str, preferred_port: int, attempts: int = 20) -> int:
    bind_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    for port in range(preferred_port, preferred_port + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex((bind_host, port)) != 0:
                return port
    raise OSError(f"Khong tim thay cong trong tu {preferred_port} den {preferred_port + attempts - 1}.")


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _build_theme():
    return gr.themes.Ocean(
        primary_hue=gr.themes.colors.teal,
        secondary_hue=gr.themes.colors.cyan,
        neutral_hue=gr.themes.colors.slate,
        spacing_size=gr.themes.sizes.spacing_md,
        radius_size=gr.themes.sizes.radius_lg,
        text_size=gr.themes.sizes.text_md,
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "Consolas", "monospace"],
    ).set(
        body_background_fill="linear-gradient(135deg, #eef7f8 0%, #f8fbff 45%, #eef4fb 100%)",
        body_background_fill_dark="linear-gradient(135deg, #071116 0%, #0c1620 52%, #101827 100%)",
        block_background_fill="#ffffff",
        block_background_fill_dark="#111a22",
        block_border_color="#d4e3ea",
        block_border_color_dark="#273745",
        input_background_fill="#f3f8fa",
        input_background_fill_dark="#17212b",
        input_border_color="#cddde5",
        input_border_color_dark="#304151",
        button_primary_background_fill="#0f8b8d",
        button_primary_background_fill_hover="#0a6f72",
        button_primary_background_fill_dark="#2bb8b3",
        button_primary_background_fill_hover_dark="#5ed8d1",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#061114",
    )


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(current_dir, "assets")
    gradio_temp_dir = os.path.join(current_dir, ".gradio_temp")
    os.environ["GRADIO_TEMP_DIR"] = gradio_temp_dir
    os.makedirs(gradio_temp_dir, exist_ok=True)

    settings = get_settings()
    demo = build_modern_demo(settings=settings)

    css_path = os.path.join(current_dir, "style.css")

    with open(css_path, "r", encoding="utf-8") as f:
        creative_css = f.read()

    theme = _build_theme()
    server_name = os.getenv("GRADIO_SERVER_NAME", settings.FRONTEND_HOST)
    preferred_port = int(os.getenv("GRADIO_SERVER_PORT", settings.FRONTEND_PORT))
    server_port = _find_available_port(server_name, preferred_port)
    display_host = "127.0.0.1" if server_name in {"0.0.0.0", "::"} else server_name
    enable_share = _env_flag("GRADIO_SHARE", default=settings.FRONTEND_SHARE)

    _, local_url, share_url = demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=enable_share,
        theme=theme,
        css=creative_css,
        allowed_paths=[assets_dir, gradio_temp_dir],
        quiet=True,
        prevent_thread_lock=True,
    )
    local_url = local_url.replace("0.0.0.0", display_host)
    print(f"Running on local URL:  {local_url}", flush=True)
    if enable_share and share_url:
        print(f"Running on public URL: {share_url}", flush=True)
    elif enable_share:
        print("Running on public URL: Chua tao duoc share link.", flush=True)
    else:
        print("Running on public URL: Da tat bang GRADIO_SHARE=0.", flush=True)
    demo.block_thread()


if __name__ == "__main__":
    main()
