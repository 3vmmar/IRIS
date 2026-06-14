"""Pydantic v2 settings — loads all IRIS configuration from .env at startup.

Every other module imports the singleton:
    from config import settings
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",          # silently ignore unrecognised .env keys
    )

    # ── Anthropic / VLM ───────────────────────────────────────────────────────
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key — required at query time, not at import time.",
    )
    vlm_model: str = Field(
        default="claude-haiku-4-5",
        description="Claude model to use: 'claude-haiku-4-5' (fast) or 'claude-sonnet-4-6' (detailed).",
    )

    # ── Camera & detection ────────────────────────────────────────────────────
    camera_index: int = Field(
        default=0,
        description="OpenCV device index for the camera (0 = default webcam).",
    )
    frame_skip: int = Field(
        default=3,
        description="Run YOLO26 inference every Nth frame to balance FPS vs. CPU load.",
    )
    yolo_model: str = Field(
        default="yolo11s.pt",
        description="YOLO model variant: yolo11n | yolo11s | yolo11m | yolo11l | yolo11x.",
    )
    yolo_confidence: float = Field(
        default=0.4,
        description="Minimum YOLO detection confidence to pass a result downstream.",
    )

    # ── Voice — speech-to-text ────────────────────────────────────────────────
    whisper_model: str = Field(
        default="base",
        description="faster-whisper model size: tiny | base | small | medium.",
    )

    # ── Voice — text-to-speech ────────────────────────────────────────────────
    tts_engine: str = Field(
        default="edge-tts",
        description="TTS backend to use: 'edge-tts' (free, Microsoft neural) or 'elevenlabs' (API).",
    )
    elevenlabs_api_key: str = Field(
        default="",
        description="ElevenLabs API key — only required when tts_engine='elevenlabs'.",
    )
    elevenlabs_voice_id: str = Field(
        default="",
        description="ElevenLabs voice ID to use for synthesis.",
    )

    # ── Visual memory ─────────────────────────────────────────────────────────
    memory_min_confidence: float = Field(
        default=0.6,
        description="Minimum detection confidence required to write a sighting to SQLite.",
    )
    memory_streak_frames: int = Field(
        default=3,
        description="Consecutive frames a label must appear before its sighting is persisted.",
    )
    memory_window_minutes: int = Field(
        default=60,
        description="Deduplication window: skip writes if same class+zone was logged within N minutes.",
    )
    memory_db_path: str = Field(
        default="iris_memory.db",
        description="File path for the SQLite visual memory database.",
    )

    # ── Conversation context ──────────────────────────────────────────────────
    context_window_turns: int = Field(
        default=10,
        description="Number of (user, assistant) turn pairs to keep in the rolling context buffer.",
    )

    # ── Server ────────────────────────────────────────────────────────────────
    port: int = Field(
        default=8000,
        description="Port for the Uvicorn / FastAPI server.",
    )
    log_level: str = Field(
        default="info",
        description="Uvicorn log level: debug | info | warning | error | critical.",
    )


# Module-level singleton — import with: from config import settings
settings = Settings()
