"""
FastAPI application entrypoint.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2

from config import settings
from api.handlers import IRISHandler
from core.camera import CameraCapture
from core.detector import ObjectDetector
from core.memory import VisualMemory
from agent.brain import Brain
from voice.stt import SpeechToText
from voice.tts import TextToSpeech

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for singletons."""
    # Singletons
    app.state.camera = CameraCapture()
    app.state.detector = ObjectDetector(app.state.camera)
    app.state.memory = VisualMemory()
    app.state.stt = SpeechToText()
    app.state.tts = TextToSpeech()
    app.state.brain = None  # type: ignore
    
    logger.info("Starting IRIS services...")
    app.state.camera.start()
    app.state.detector.start()
    await app.state.memory.open()
    
    try:
        app.state.brain = Brain()
    except Exception as e:
        logger.error("Failed to initialize Brain: %s", e)
        
    try:
        app.state.stt.load()
    except Exception as e:
        logger.warning("Failed to load STT (optional): %s", e)
        
    try:
        app.state.tts.load()
    except Exception as e:
        logger.warning("Failed to load TTS (optional): %s", e)
        
    logger.info("IRIS ready.")
    
    yield
    
    logger.info("Stopping IRIS services...")
    app.state.detector.stop()
    app.state.camera.stop()
    await app.state.memory.close()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "camera": app.state.camera.is_running(),
        "detector": app.state.detector.is_running(),
        "stt": app.state.stt.is_loaded(),
        "tts": app.state.tts.is_loaded(),
    }

@app.get("/video_feed", tags=["system"])
async def video_feed():
    """Stream the raw camera feed as MJPEG."""
    import asyncio

    async def _frame_generator():
        while True:
            frame = app.state.camera.get_frame()
            if frame is not None:
                # Encode frame as JPEG
                loop = asyncio.get_event_loop()
                ret, buffer = await loop.run_in_executor(None, cv2.imencode, '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            await asyncio.sleep(0.033) # roughly 30 FPS

    return StreamingResponse(_frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time interaction."""
    await websocket.accept()
    handler = IRISHandler(
        ws=websocket,
        camera=websocket.app.state.camera,
        detector=websocket.app.state.detector,
        memory=websocket.app.state.memory,
        brain=websocket.app.state.brain,
        stt=websocket.app.state.stt,
        tts=websocket.app.state.tts
    )
    try:
        await handler.handle()
    except WebSocketDisconnect:
        pass

static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")
    
    # Custom handler for index to serve index.html directly from static/
    # FastAPI static mount handles / automatically if index.html is present,
    # but we can mount the whole static dir explicitly if needed.
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
else:
    logger.warning("Static directory not found. UI will not be served.")

    @app.get("/")
    async def index():
        """Fallback index when no static UI exists."""
        return JSONResponse({"message": "IRIS backend running. Build the React frontend to serve the UI."})
