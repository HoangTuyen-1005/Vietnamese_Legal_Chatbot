from frontend_app.core.config import get_settings
from frontend_app.main import build_demo


if __name__ == "__main__":
    settings = get_settings()
    demo = build_demo(settings=settings)
    demo.launch(
        server_name=settings.FRONTEND_HOST,
        server_port=settings.FRONTEND_PORT,
        share=settings.FRONTEND_SHARE,
    )
