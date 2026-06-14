"""
FastAPI application.
Mounts the built React frontend as static files,
exposes GET /health, and registers the WebSocket route at /ws.
Initialises camera, detector, memory, brain, and TTS on startup.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from agent.brain import IRISBrain
from agent.context import ConversationContext
from api.handlers import WebSocketHandler
from config import settings
from core.camera import CameraCapture
from core.detector import ObjectDetector
from core.memory import VisualMemory
from voice.stt import SpeechToText
from voice.tts import TextToSpeech

logger = logging.getLogger(__name__)

# ── Singleton services ────────────────────────────────────────────────────────
# These are shared across all WebSocket clients. They are started on app
# startup and stopped on shutdown via the lifespan context manager.

camera   = CameraCapture()
detector = ObjectDetector(camera)
memory   = VisualMemory()   # uses session_id='default'; overridable per-client later
tts      = TextToSpeech()
stt_service: SpeechToText | None = None    # lazy — requires mic permission
brain    = IRISBrain(
    context=ConversationContext(),   # global context (shared across all clients)
    memory=memory,
)

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start all IRIS background services on startup; stop them on shutdown."""
    global stt_service
    logger.info("IRIS startup — initialising services …")

    # Visual memory (async SQLite)
    await memory.open()

    # Camera + YOLO detector (background threads)
    camera.start()
    detector.start()

    # Claude client
    brain.init()

    # TTS engine
    await tts.init()

    # STT — microphone (may fail if no mic available; non-fatal)
    def _on_transcript(text: str) -> None:
        import asyncio
        # Forward transcribed text into the active async event loop
        # The handler loop will pick it up via the WebSocket audio pipeline
        logger.info("Mic transcript (global): %r", text)

    try:
        stt_service = SpeechToText(callback=_on_transcript)
        stt_service.start()
    except Exception as exc:
        logger.warning("STT could not start (no mic?): %s", exc)
        stt_service = None

    logger.info("IRIS is ready.")
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("IRIS shutdown — stopping services …")

    if stt_service is not None:
        stt_service.stop()

    detector.stop()
    camera.stop()
    await memory.close()
    await tts.close()
    await brain.close()

    logger.info("IRIS stopped cleanly.")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance.

    Returns:
        Configured :class:`fastapi.FastAPI` application with routes and
        static file serving registered.
    """
    app = FastAPI(
        title="IRIS — Intelligent Real-time Interactive Sensing",
        version="1.0.0",
        description="Real-time multimodal AI agent with vision, voice, and memory.",
        lifespan=lifespan,
    )

    # ── REST routes ───────────────────────────────────────────────────────────

    @app.get("/health", tags=["system"])
    async def health() -> dict:
        """Simple liveness check — returns service status."""
        return {
            "status": "ok",
            "camera_running": camera.is_running(),
            "detector_running": detector.is_running(),
            "camera_fps": camera.get_fps(),
            "model": settings.vlm_model,
        }

    # ── WebSocket route ───────────────────────────────────────────────────────

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        """Handle a single client WebSocket session.

        Creates a :class:`~api.handlers.WebSocketHandler` per connection so
        each client gets an isolated session with its own detection push loop.
        """
        handler = WebSocketHandler(
            websocket=websocket,
            detector=detector,
            memory=memory,
            brain=brain,
            tts=tts,
            stt=stt_service,
            camera=camera,
        )
        await handler.run()

    # ── Static frontend ───────────────────────────────────────────────────────
    static_dir = Path(__file__).parent.parent / "static"

    if static_dir.exists():
        # Serve the built React app under /app; root / serves index.html
        app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

        @app.get("/", include_in_schema=False)
        async def serve_index() -> FileResponse:
            return FileResponse(static_dir / "index.html")

        @app.get("/{full_path:path}", include_in_schema=False)
        async def serve_spa(full_path: str) -> FileResponse:
            """Catch-all that returns index.html for client-side routing."""
            target = static_dir / full_path
            if target.is_file():
                return FileResponse(target)
            return FileResponse(static_dir / "index.html")
    else:
        logger.warning(
            "Static frontend not found at '%s'. "
            "Run 'cd frontend && npm run build' to build the React app.",
            static_dir,
        )

    return app


# Module-level app instance (imported by main.py and Uvicorn)
app = create_app()
