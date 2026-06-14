# IRIS — Intelligent Real-time Interactive Sensing

> A multimodal AI agent that sees through your camera, listens through your microphone, remembers what it observes, and responds in natural language and voice.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-WebSocket-009688?style=flat&logo=fastapi&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object_Detection-purple?style=flat)
![Whisper](https://img.shields.io/badge/Whisper-STT-orange?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## What IRIS Does

IRIS is a real-time multimodal AI agent that connects your camera and microphone to a vision-language pipeline. You speak a question, IRIS sees the frame, reasons about it, and responds in voice and text.

```
You:   "Where did I leave my keys?"
IRIS:  "I last saw your keys on the left side of the desk, about 12 minutes ago."

You:   "What's on my desk right now?"
IRIS:  "I can see a laptop, a water bottle, a phone, and two pens. The monitor
        is on and appears to be showing a code editor."

You:   "Is anything out of place?"
IRIS:  "There's a cable hanging near the edge of the desk that wasn't there earlier."
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                          │
│   Camera (OpenCV 30fps)          Microphone (silero-VAD)    │
└──────────────┬───────────────────────────┬──────────────────┘
               │ async frame queue          │ audio chunks
┌──────────────▼───────────────────────────▼──────────────────┐
│                        VISION LAYER                         │
│   YOLOv8 detector (~6fps)    Scene diff    Whisper STT      │
└──────────────┬───────────────────────────┬──────────────────┘
               │ detections + frame         │ transcribed text
┌──────────────▼───────────────────────────▼──────────────────┐
│                         BRAIN LAYER                         │
│   VLM (GPT-4o-mini / Qwen2.5-VL)  +  Visual Memory (SQLite)│
│   Conversation context (rolling window)                     │
└──────────────┬───────────────────────────┬──────────────────┘
               │ text tokens (streaming)    │ memory updates
┌──────────────▼───────────────────────────▼──────────────────┐
│                        OUTPUT LAYER                         │
│   Text stream (WebSocket)         TTS (Coqui / ElevenLabs)  │
│   BBox overlay (Canvas)           Memory log panel          │
└──────────────┬────────────────────────────────────────────  ┘
               │
┌──────────────▼──────────────────────────────────────────────┐
│              FastAPI + WebSocket Server                     │
│              HTML/JS Frontend (single page)                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Features

| Feature | Status | Description |
|---|---|---|
| Real-time object detection | ✅ | YOLOv8s running every 5th frame with bbox overlay on canvas |
| Visual Q&A | ✅ | Ask anything about what the camera sees |
| Voice input | ✅ | Speak queries — VAD detects start/end automatically |
| Voice output | ✅ | AI responds in synthesized speech |
| Visual memory | ✅ | IRIS remembers where it last saw objects with timestamps |
| Streaming responses | ✅ | Tokens stream in real-time, no 2-second blank wait |
| Conversation context | ✅ | Full rolling window of prior turns injected into each query |
| Web UI | ✅ | Single-page interface: live camera, bbox overlay, chat panel, memory log |

---

## Project Structure

```
iris/
├── core/
│   ├── camera.py          # Background thread: frame capture → deque
│   ├── detector.py        # Background thread: YOLOv8 inference loop
│   ├── scene.py           # Scene change detection (triggers VLM calls)
│   └── memory.py          # SQLite async CRUD — visual memory store
├── voice/
│   ├── stt.py             # faster-whisper wrapper + silero-VAD
│   └── tts.py             # Coqui TTS / ElevenLabs abstraction
├── agent/
│   ├── brain.py           # VLM call manager + prompt builder
│   ├── prompts.py         # System prompt templates
│   └── context.py         # Rolling conversation buffer
├── api/
│   ├── server.py          # FastAPI app — REST + WebSocket endpoints
│   └── handlers.py        # WebSocket message protocol handler
├── frontend/
│   ├── index.html         # Single-page UI
│   ├── camera.js          # MediaStream → canvas pipeline
│   ├── audio.js           # MediaRecorder → WebSocket audio
│   └── chat.js            # Chat panel + streaming text renderer
├── config.py              # Pydantic settings (model, frame rate, etc.)
├── main.py                # Entrypoint: starts all threads + server
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## WebSocket Protocol

```jsonc
// Client → Server
{ "type": "audio_chunk",      "data": "<base64>" }
{ "type": "text_query",       "text": "Where is my phone?" }
{ "type": "request_snapshot"                       }

// Server → Client
{ "type": "detections",  "boxes": [...], "frame_id": 42 }
{ "type": "text_token",  "token": " can"              }  // streaming
{ "type": "text_done",   "full": "I can see..."        }
{ "type": "audio_chunk", "data": "<base64 wav>"        }
{ "type": "memory_update","object": "keys", "zone": "left", "ts": "..." }
{ "type": "error",       "message": "..."              }
```

---

## Visual Memory

Objects are stored in a 3×3 spatial grid (9 zones) relative to the frame. Each detection write requires:
- Confidence ≥ 0.6
- Object present for ≥ 3 consecutive frames
- Same class + zone within 30 minutes → update record, don't duplicate

```sql
CREATE TABLE sightings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    object      TEXT    NOT NULL,
    zone        TEXT    NOT NULL,   -- e.g. "bottom-right"
    x_rel       REAL,               -- 0.0–1.0 fractional position
    y_rel       REAL,
    confidence  REAL,
    seen_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
    session_id  TEXT
);
```

Memory is injected into every VLM prompt as structured context so IRIS can answer "where did I leave X?" without the object being in frame.

---

## Tech Stack

| Component | Choice | Reason |
|---|---|---|
| Object detection | `ultralytics` YOLOv8s | Best speed/accuracy tradeoff, CPU-friendly |
| Vision-language | GPT-4o-mini (API) or Qwen2.5-VL-7B (local) | GPT-4o-mini for dev speed; swap to local for demo |
| Speech-to-text | `faster-whisper` base | CTranslate2-optimized, <400ms latency |
| Voice activity | `silero-vad` | Accurate, fast, local |
| Text-to-speech | `TTS` (Coqui) | Open-source, offline, natural voice |
| Backend | `FastAPI` + WebSocket | Async, production-grade |
| Memory | `aiosqlite` + SQLite | Zero-config, async writes |
| Frontend | Vanilla HTML/JS | No build step, instant iteration |

---

## Setup

```bash
git clone https://github.com/3vmmar/iris.git
cd iris

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Copy and fill in your API key (if using GPT-4o-mini)
cp .env.example .env
```

```bash
# Run
python main.py

# Open browser
http://localhost:8000
```

### Docker

```bash
docker compose up --build
```

---

## Environment Variables

```env
# .env.example
OPENAI_API_KEY=sk-...          # Required if VLM_BACKEND=openai
VLM_BACKEND=openai             # openai | local
LOCAL_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
CAMERA_INDEX=0
FRAME_SKIP=5                   # Run YOLO every Nth frame
YOLO_MODEL=yolov8s.pt
WHISPER_MODEL=base
TTS_ENGINE=coqui               # coqui | elevenlabs
ELEVENLABS_API_KEY=            # Optional
MEMORY_WINDOW_MINUTES=60
CONTEXT_WINDOW_TURNS=10
PORT=8000
```

---

## Roadmap

- [x] Week 1 — The Eyes: camera pipeline, YOLO detection, WebSocket frame stream, basic VLM Q&A
- [ ] Week 2 — The Brain: voice input, conversation memory, context injection, memory retrieval routing
- [ ] Week 3 — The Face: TTS output, polished web UI, memory log panel, Docker, demo video

---

## Author

**Ammar** — Computer Science & AI, Zewail City of Science and Technology  
AI Engineer · [GitHub](https://github.com/3vmmar)

---

*IRIS is a portfolio project demonstrating real-time multimodal AI engineering: async pipelines, vision-language models, local inference, and voice interaction — the same stack used in production AI agent systems.*