<div align="center">

#  IRIS

### Intelligent Real-time Interactive Sensing

*A real-time multimodal AI agent that sees, listens, remembers, and responds.*

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-WebSocket-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![YOLO26](https://img.shields.io/badge/YOLO26-Object_Detection-purple?style=flat-square)](https://ultralytics.com/)
[![Claude](https://img.shields.io/badge/Claude-Vision_LLM-CC785C?style=flat-square&logo=anthropic&logoColor=white)](https://anthropic.com/)
[![Whisper](https://img.shields.io/badge/Whisper-STT-orange?style=flat-square)](https://github.com/openai/whisper)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## Overview

IRIS is a real-time multimodal AI agent that connects your camera and microphone to a vision-language pipeline. Point the camera at any environment, speak a question, and IRIS sees the frame, reasons about it, and responds in both text and voice.

```
You:   "What's on my desk?"
IRIS:  "I can see a laptop, a phone, a water bottle, and two pens.
        The monitor is displaying a code editor."

You:   "Where did I leave my keys?"
IRIS:  "I last saw your keys in the bottom-left zone, about 8 minutes ago."

You:   "Is anything unusual?"
IRIS:  "There's a cable hanging near the edge of the desk that wasn't there earlier."
```

IRIS is not a demo wrapper around a single API call. It is a fully async, multi-threaded perception system — the same architecture pattern used in production visual AI agents.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          INPUT LAYER                            │
│                                                                 │
│        Camera (OpenCV · 30fps)      Microphone (silero-VAD)     │
└────────────────┬────────────────────────────┬───────────────────┘
                 │  async frame deque          │  audio chunks
┌────────────────▼────────────────────────────▼───────────────────┐
│                         VISION LAYER                            │
│                                                                 │
│   YOLO26 detector (~10fps)   Scene diff     faster-whisper STT  │
└────────────────┬────────────────────────────┬───────────────────┘
                 │  detections + frame          │  transcribed query
┌────────────────▼────────────────────────────▼───────────────────┐
│                          BRAIN LAYER                            │
│                                                                 │
│    Claude (claude-haiku-4-5 · claude-sonnet-4-6) via Anthropic API     │
│    Visual Memory (SQLite · 3×3 zone grid · async writes)        │
│    Conversation Context (rolling N-turn window)                 │
└────────────────┬────────────────────────────┬───────────────────┘
                 │  streamed tokens             │  memory updates
┌────────────────▼────────────────────────────▼───────────────────┐
│                         OUTPUT LAYER                            │
│                                                                 │
│    Text stream (WebSocket)      TTS (Coqui · ElevenLabs)        │
│    BBox overlay (Canvas)        Memory log panel                │
└────────────────┬────────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────────┐
│              FastAPI + WebSocket Server  ·  HTML/JS Frontend    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Features

| Feature | Description |
|---|---|
| **Real-time detection** | YOLO26 — the latest Ultralytics release — running at ~10fps with bounding boxes drawn live on canvas |
| **Visual Q&A** | Ask anything about what the camera sees; Claude reasons over the frame and detection context |
| **Voice input** | Speak queries; silero-VAD detects start/end of speech before triggering Whisper |
| **Voice output** | AI responds in synthesized speech via Coqui TTS (local) or ElevenLabs (API) |
| **Visual memory** | IRIS remembers where it last saw objects using a 3×3 spatial zone grid, persisted to SQLite |
| **Streaming responses** | Tokens stream into the chat panel in real time — no waiting for a full response |
| **Conversation context** | Full rolling window of prior turns injected into every Claude call |
| **Scene change detection** | Automatically triggers a new description when the environment changes significantly |

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Object detection | **YOLO26** (Ultralytics) | Latest YOLO — unifies detection, segmentation, pose, and OBB in one model |
| Vision-language | **Claude claude-haiku-4-5 / claude-sonnet-4-6** (Anthropic) | Best-in-class vision reasoning with streaming support |
| Speech-to-text | **faster-whisper** + silero-VAD | CTranslate2-optimized, <400ms latency, runs fully local |
| Text-to-speech | **Coqui TTS** / ElevenLabs | Local offline voice or premium API voice quality |
| Backend | **FastAPI** + WebSocket | Async, production-grade, minimal overhead |
| Memory | **aiosqlite** + SQLite | Zero-config, async writes, no external database |
| Frontend | Vanilla HTML/JS | No build step, instant iteration, single file |

---

## Project Structure

```
iris/
├── core/
│   ├── camera.py          # Frame capture thread → thread-safe deque
│   ├── detector.py        # YOLO26 inference thread → cached detection results
│   ├── scene.py           # Scene change detection (IoU-delta on label sets)
│   └── memory.py          # SQLite visual memory with 3×3 zone grid
├── voice/
│   ├── stt.py             # faster-whisper + silero-VAD
│   └── tts.py             # Coqui TTS / ElevenLabs abstraction
├── agent/
│   ├── brain.py           # Anthropic API call manager, frame encoding, token streaming
│   ├── prompts.py         # System prompt templates and builder functions
│   └── context.py         # Rolling conversation buffer
├── api/
│   ├── server.py          # FastAPI app — REST + WebSocket + static frontend
│   └── handlers.py        # Per-client WebSocket protocol handler
├── frontend/
│   ├── index.html         # Single-page UI
│   ├── camera.js          # MediaStream → canvas + bbox overlay
│   ├── audio.js           # MediaRecorder → WebSocket audio streaming
│   └── chat.js            # Streaming token renderer + memory log
├── config.py              # Pydantic settings (loaded from .env)
├── main.py                # Entrypoint
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Visual Memory

Objects are stored on a 3×3 spatial grid (9 named zones). A write only occurs when:

- Detection confidence ≥ 0.60
- Same object present for ≥ 3 consecutive frames
- No existing record for the same class + zone within the last 60 minutes

```sql
CREATE TABLE sightings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    object      TEXT    NOT NULL,
    zone        TEXT    NOT NULL,
    x_rel       REAL,
    y_rel       REAL,
    confidence  REAL,
    seen_at     TEXT    NOT NULL,
    session_id  TEXT
);
```

Memory context is injected into every Claude prompt so IRIS can answer *"where did I leave X?"* without the object being in frame.

---

## WebSocket Protocol

```jsonc
// Client → Server
{ "type": "text_query",       "text": "Where is my phone?" }
{ "type": "audio_chunk",      "data": "<base64 PCM>"       }
{ "type": "request_snapshot"                               }

// Server → Client
{ "type": "detections",    "boxes": [...], "frame_id": 42  }
{ "type": "text_token",    "token": " can"                 }
{ "type": "text_done",     "full": "I can see..."          }
{ "type": "audio_chunk",   "data": "<base64 WAV>"          }
{ "type": "memory_update", "object": "keys", "zone": "bottom-left" }
{ "type": "error",         "message": "..."                }
```

---

## Setup

```bash
git clone https://github.com/3vmmar/iris.git
cd iris

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env        # Fill in your Anthropic API key
```

```bash
python main.py
# → Open http://localhost:8000
```

### Docker

```bash
docker compose up --build
```

---

## Environment Variables

```env
# .env.example

ANTHROPIC_API_KEY=sk-ant-...      # Required
VLM_MODEL=claude-haiku-4-5          # claude-haiku-4-5 (fast) | claude-sonnet-4-6 (detailed)

CAMERA_INDEX=0
FRAME_SKIP=3                      # Run YOLO26 every Nth frame
YOLO_MODEL=yolo26s.pt             # yolo26n | yolo26s | yolo26m | yolo26l | yolo26x

WHISPER_MODEL=base                # tiny | base | small | medium
TTS_ENGINE=coqui                  # coqui | elevenlabs
ELEVENLABS_API_KEY=               # Optional — only if TTS_ENGINE=elevenlabs
ELEVENLABS_VOICE_ID=              # Optional

MEMORY_WINDOW_MINUTES=60
CONTEXT_WINDOW_TURNS=10
PORT=8000
```

---

## Author

**Ammar**  — Computer Science & AI, Zewail City of Science and Technology  
AI Engineer · [github.com/3vmmar](https://github.com/3vmmar)

---

<div align="center">
<sub>IRIS demonstrates real-world AI engineering: async multi-threaded pipelines, vision-language models, local inference, and voice interaction — the same stack used in production agentic systems.</sub>
</div>