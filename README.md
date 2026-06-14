<div align="center">

<br />

# IRIS

**Intelligent Real-time Interactive Sensing**

A real-time multimodal AI agent that sees through your camera,<br/>listens through your microphone, remembers what it observes, and responds in voice and text.

<br/>

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-WebSocket-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![YOLO26](https://img.shields.io/badge/YOLO26-Detection-7C3AED?style=flat-square)](https://ultralytics.com)
[![Claude](https://img.shields.io/badge/Claude-Vision_LLM-CC785C?style=flat-square&logo=anthropic&logoColor=white)](https://anthropic.com)
[![Tailwind](https://img.shields.io/badge/Tailwind-CSS-38BDF8?style=flat-square&logo=tailwindcss&logoColor=white)](https://tailwindcss.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

<br/>

</div>

---

## What is IRIS?

IRIS is not a demo wrapper around a single API call. It is a fully async, multi-threaded visual perception system — the same architectural pattern used in production AI agents.

Point the camera at any environment, ask a question out loud, and IRIS sees the frame, reasons over it using Claude, and responds in both text and synthesized voice. It also maintains a persistent visual memory so it can answer questions about objects it saw minutes ago, even if they are no longer in frame.

```
You   →  "What's on my desk right now?"
IRIS  →  "I can see a laptop, a phone, a water bottle, and two pens.
           The monitor is showing a code editor."

You   →  "Where did I leave my keys?"
IRIS  →  "I last saw your keys in the bottom-left zone, about 8 minutes ago."

You   →  "Is anything unusual?"
IRIS  →  "There's a cable hanging near the edge of the desk
           that wasn't there in my last observation."
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                            INPUT LAYER                               │
│                                                                      │
│        Camera  (OpenCV · 30 fps)       Microphone  (silero-VAD)      │
└───────────────────┬──────────────────────────┬───────────────────────┘
                    │  thread-safe frame deque  │  raw audio chunks
┌───────────────────▼──────────────────────────▼───────────────────────┐
│                            VISION LAYER                              │
│                                                                      │
│    YOLO26  (~10 fps · 5 tasks unified)     faster-whisper  STT       │
│    Scene change detector  (IoU-delta)                                │
└───────────────────┬──────────────────────────┬───────────────────────┘
                    │  detections + raw frame   │  transcribed query
┌───────────────────▼──────────────────────────▼───────────────────────┐
│                            BRAIN LAYER                               │
│                                                                      │
│    Claude  (claude-haiku-4-5 · claude-sonnet-4-6)  via Anthropic API         │
│    Visual Memory  (SQLite · 3×3 zone grid · async writes)            │
│    Conversation Context  (rolling N-turn window)                     │
└───────────────────┬──────────────────────────┬───────────────────────┘
                    │  streamed tokens          │  memory updates
┌───────────────────▼──────────────────────────▼───────────────────────┐
│                            OUTPUT LAYER                              │
│                                                                      │
│    Text stream  (WebSocket)       TTS  (Coqui · ElevenLabs)          │
│    BBox overlay  (Canvas)         Memory log sidebar                 │
└───────────────────┬──────────────────────────────────────────────────┘
                    │
┌───────────────────▼──────────────────────────────────────────────────┐
│         FastAPI + WebSocket backend  ·  React + Tailwind frontend    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Features

| | Feature | Detail |
|---|---|---|
| 🎯 | **Real-time detection** | YOLO26 — unifies detection, segmentation, pose, classification, and OBB in one model |
| 🧠 | **Visual Q&A** | Claude reasons over the live frame and detection context to answer natural language queries |
| 🎤 | **Voice input** | silero-VAD detects speech boundaries, faster-whisper transcribes locally in <400ms |
| 🔊 | **Voice output** | Coqui TTS (offline) or ElevenLabs (API) synthesizes every IRIS response |
| 🗃️ | **Visual memory** | Objects persisted to SQLite with 3×3 zone grid, timestamps, and confidence thresholds |
| ⚡ | **Streaming responses** | Tokens stream into the UI in real time — zero wait for full response |
| 💬 | **Conversation context** | Full N-turn history injected into every Claude call |
| 🔄 | **Scene change detection** | Automatically flags significant environment changes |

---

## Tech Stack

| Layer | Technology | Reason |
|---|---|---|
| Object detection | **YOLO26** (Ultralytics, Sep 2025) | Latest YOLO — first to unify 5 vision tasks natively |
| Vision-language model | **Claude claude-haiku-4-5 / claude-sonnet-4-6** (Anthropic) | Best reasoning + native vision + streaming support |
| Speech-to-text | **faster-whisper** + silero-VAD | CTranslate2-optimized, runs fully local, <400ms latency |
| Text-to-speech | **Coqui TTS** / ElevenLabs | Local offline voice or premium API voice quality |
| Backend | **FastAPI** + WebSocket | Async, production-grade, minimal overhead |
| Memory | **aiosqlite** + SQLite | Zero-config persistent store, async write-safe |
| Frontend | **React 18** + **Vite** | Modern component model, instant HMR, zero config |
| Styling | **Tailwind CSS** | Utility-first, dark theme, no runtime overhead |
| State | **Zustand** | Lightweight global store, no boilerplate |

---

## Project Structure

```
IRIS/
├── core/
│   ├── camera.py            # Frame capture thread → thread-safe deque
│   ├── detector.py          # YOLO26 inference thread → cached detection results
│   ├── scene.py             # Scene change detection (IoU-delta on label sets)
│   └── memory.py            # SQLite visual memory with 3×3 zone grid
├── voice/
│   ├── stt.py               # faster-whisper + silero-VAD
│   └── tts.py               # Coqui TTS / ElevenLabs abstraction
├── agent/
│   ├── brain.py             # Claude API manager — frame encoding + token streaming
│   ├── prompts.py           # System prompt templates and builder functions
│   └── context.py           # Rolling conversation buffer
├── api/
│   ├── server.py            # FastAPI app — mounts React build + WebSocket
│   └── handlers.py          # Per-client WebSocket protocol handler
├── frontend/
│   ├── src/
│   │   ├── App.jsx                       # Root layout component
│   │   ├── components/
│   │   │   ├── CameraFeed.jsx            # Live video + canvas bbox overlay
│   │   │   ├── DetectionOverlay.jsx      # YOLO26 bbox + zone label renderer
│   │   │   ├── ChatPanel.jsx             # Streaming chat with IRIS
│   │   │   ├── VoiceButton.jsx           # Push-to-talk / VAD trigger
│   │   │   ├── MemoryLog.jsx             # Visual memory sidebar
│   │   │   └── StatusBar.jsx             # Connection status + FPS counter
│   │   ├── hooks/
│   │   │   ├── useWebSocket.js           # WS connection + message routing
│   │   │   ├── useCamera.js              # MediaStream → canvas pipeline
│   │   │   └── useAudio.js               # MediaRecorder + audio streaming
│   │   ├── store/
│   │   │   └── useStore.js               # Zustand global state
│   │   └── styles/
│   │       └── index.css                 # Tailwind base + CSS custom properties
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
├── config.py                # Pydantic settings (loaded from .env)
├── main.py                  # Entrypoint
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Visual Memory

Objects are stored on a **3×3 spatial zone grid**. A write only occurs when:

- Detection confidence ≥ **0.60**
- Same object present for **≥ 3 consecutive frames**
- No existing record for the same class + zone within the **memory window** (default: 60 min)

```sql
CREATE TABLE sightings (
    id          INTEGER  PRIMARY KEY AUTOINCREMENT,
    object      TEXT     NOT NULL,
    zone        TEXT     NOT NULL,
    x_rel       REAL,
    y_rel       REAL,
    confidence  REAL,
    seen_at     TEXT     NOT NULL,
    session_id  TEXT
);
```

Memory is injected into every Claude prompt as structured context so IRIS can answer *"where did I leave X?"* without the object being in frame.

---

## WebSocket Protocol

```jsonc
// Client → Server
{ "type": "text_query",       "text": "Where is my phone?"  }
{ "type": "audio_chunk",      "data": "<base64 PCM>"        }
{ "type": "request_snapshot"                                 }

// Server → Client
{ "type": "detections",    "boxes": [...], "frame_id": 42   }
{ "type": "text_token",    "token": " can"                  }
{ "type": "text_done",     "full": "I can see..."           }
{ "type": "audio_chunk",   "data": "<base64 WAV>"           }
{ "type": "memory_update", "object": "keys", "zone": "bottom-left" }
{ "type": "error",         "message": "..."                 }
```

---

## Setup

### Backend

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # Fill in your Anthropic API key
python main.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev        # Dev server at http://localhost:5173 (proxies API to :8000)
```

### Production build

```bash
cd frontend && npm run build    # Outputs to ../static — served by FastAPI
```

### Docker

```bash
docker compose up --build
# → http://localhost:8000
```

---

## Environment Variables

```env
# .env.example

ANTHROPIC_API_KEY=sk-ant-...        # Required
VLM_MODEL=claude-haiku-4-5            # claude-haiku-4-5 (fast) | claude-sonnet-4-6 (deep)

CAMERA_INDEX=0
FRAME_SKIP=3                        # Run YOLO26 every Nth frame
YOLO_MODEL=yolo26s.pt               # yolo26n | yolo26s | yolo26m | yolo26l | yolo26x

WHISPER_MODEL=base                  # tiny | base | small | medium
TTS_ENGINE=coqui                    # coqui | elevenlabs
ELEVENLABS_API_KEY=
ELEVENLABS_VOICE_ID=

MEMORY_WINDOW_MINUTES=60
CONTEXT_WINDOW_TURNS=10
PORT=8000
```

---

## Author

**Ammar**  · Data Science & AI, Zewail City of Science and Technology  
AI Engineer · [github.com/3vmmar](https://github.com/3vmmar)

---

<div align="center">
<sub>
IRIS demonstrates production AI engineering patterns — async multi-threaded pipelines,<br/>
vision-language models, local inference, real-time WebSocket streaming, and modern React UI.
</sub>
</div>