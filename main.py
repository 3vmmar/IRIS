"""IRIS entrypoint — starts the FastAPI server and all background threads."""

from __future__ import annotations

import logging
import sys

import uvicorn

from config import settings

# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

logger = logging.getLogger("iris.main")


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    """Start the Uvicorn server.

    All service initialisation (camera, detector, memory, brain, TTS) is handled
    by the FastAPI lifespan context manager in ``api/server.py``.
    Uvicorn is run in single-worker mode so the background threads and asyncio
    event loop share one process — no IPC overhead.
    """
    logger.info("Starting IRIS on port %d …", settings.port)

    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=False,           # Reload would restart background threads — keep off
        workers=1,              # Single worker: camera/detector threads are process-local
        ws_ping_interval=20,    # Keep WebSocket connections alive
        ws_ping_timeout=30,
    )


if __name__ == "__main__":
    main()
