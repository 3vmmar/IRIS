"""
The per-client WebSocket protocol handler.
"""

from __future__ import annotations

import asyncio
import base64
import functools
import json
import logging
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect

from config import settings
from agent.brain import Brain
from core.camera import CameraCapture
from core.detector import ObjectDetector
from core.memory import VisualMemory
from core.scene import SceneChangeDetector
from voice.stt import SpeechToText
from voice.tts import TextToSpeech

logger = logging.getLogger(__name__)


class IRISHandler:
    """Manages a single WebSocket client session.

    Handles real-time detection push loop and incoming message processing
    (text queries, audio chunks, snapshot requests).
    """

    def __init__(
        self,
        ws: WebSocket,
        camera: CameraCapture,
        detector: ObjectDetector,
        memory: VisualMemory,
        brain: Brain,
        stt: SpeechToText,
        tts: TextToSpeech,
    ) -> None:
        self._ws = ws
        self._camera = camera
        self._detector = detector
        self._memory = memory
        self._brain = brain
        self._stt = stt
        self._tts = tts
        self._scene_detector = SceneChangeDetector()
        self._last_frame_id: int = -1

    async def handle(self) -> None:
        """Main entry point. Starts detection push and message processing concurrently."""
        push_task = asyncio.create_task(self._detection_push_loop())
        msg_task = asyncio.create_task(self._message_loop())

        try:
            await asyncio.gather(push_task, msg_task)
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected.")
        except Exception as e:
            logger.error("Error in handler: %s", e)
        finally:
            push_task.cancel()
            msg_task.cancel()

    async def _detection_push_loop(self) -> None:
        """Continuously pushes detection frames to the client."""
        while True:
            try:
                result = self._detector.get_latest()
                if result is not None and result.frame_id != self._last_frame_id:
                    # Construct message
                    boxes = [d.to_dict() for d in result.detections]
                    await self._send({
                        "type": "detections",
                        "boxes": boxes,
                        "frame_id": result.frame_id,
                        "width": result.frame_width,
                        "height": result.frame_height,
                        "inference_ms": result.inference_ms,
                        "time_since_last_analysis": self._scene_detector.time_since_last_trigger(),
                    })
                    
                    self._last_frame_id = result.frame_id

                    # Memory observation
                    for d in result.detections:
                        is_new = await self._memory.observe(
                            label=d.label,
                            zone=d.zone,
                            x_rel=d.x_rel,
                            y_rel=d.y_rel,
                            confidence=d.confidence
                        )
                        if is_new:
                            await self._send({
                                "type": "memory_update",
                                "object": d.label,
                                "zone": d.zone
                            })

                    # Scene change detection
                    if self._scene_detector.has_changed(result):
                        asyncio.create_task(self._auto_describe())

            except Exception as e:
                logger.error("Detection push loop error: %s", e)

            await asyncio.sleep(0.15)

    async def _auto_describe(self) -> None:
        """Generates an automatic description of a new scene."""
        try:
            frame = self._camera.get_frame()
            result = self._detector.get_latest()
            detections = [d.to_dict() for d in result.detections] if result else []
            memory_context = await self._memory.recent_context()

            full_tokens = []
            async for token in self._brain.describe_scene(frame, detections, memory_context):
                await self._send({"type": "text_token", "token": token})
                full_tokens.append(token)
            
            full_text = "".join(full_tokens)
            await self._send({"type": "text_done", "full": full_text})

            if self._tts.is_loaded():
                loop = asyncio.get_running_loop()
                audio_bytes = await loop.run_in_executor(None, self._tts.synthesize, full_text)
                if audio_bytes:
                    await self._send({
                        "type": "audio_chunk",
                        "data": base64.b64encode(audio_bytes).decode("utf-8")
                    })
        except Exception as e:
            logger.error("Auto describe error: %s", e)
            await self._send({"type": "error", "message": str(e)})

    async def _message_loop(self) -> None:
        """Continuously reads incoming client messages."""
        while True:
            raw = await self._ws.receive_text()
            try:
                msg = json.loads(raw)
                msg_type = msg.get("type")

                if msg_type == "text_query":
                    asyncio.create_task(self._handle_text_query(msg.get("text", "")))
                elif msg_type == "audio_chunk":
                    asyncio.create_task(self._handle_audio_chunk(msg.get("data", "")))
                elif msg_type == "request_snapshot":
                    asyncio.create_task(self._handle_snapshot())
                else:
                    await self._send({"type": "error", "message": f"Unknown message type {msg_type}"})
            except json.JSONDecodeError:
                await self._send({"type": "error", "message": "Invalid JSON"})
            except Exception as e:
                logger.error("Message loop error: %s", e)

    async def _handle_text_query(self, query: str) -> None:
        """Handles a direct text query from the user."""
        try:
            frame = self._camera.get_frame()
            result = self._detector.get_latest()
            detections = [d.to_dict() for d in result.detections] if result else []
            memory_context = await self._memory.recent_context()

            full_tokens = []
            async for token in self._brain.respond_stream(query, frame, detections, memory_context):
                await self._send({"type": "text_token", "token": token})
                full_tokens.append(token)
            
            full_text = "".join(full_tokens)
            await self._send({"type": "text_done", "full": full_text})

            if self._tts.is_loaded():
                loop = asyncio.get_running_loop()
                audio_bytes = await loop.run_in_executor(None, self._tts.synthesize, full_text)
                if audio_bytes:
                    await self._send({
                        "type": "audio_chunk",
                        "data": base64.b64encode(audio_bytes).decode("utf-8")
                    })
        except Exception as e:
            logger.error("Handle text query error: %s", e)
            await self._send({"type": "error", "message": str(e)})

    async def _handle_audio_chunk(self, data: str) -> None:
        """Handles an incoming audio chunk (base64)."""
        if not self._stt.is_loaded():
            await self._send({"type": "error", "message": "STT model not loaded"})
            return

        try:
            audio_bytes = base64.b64decode(data)
            loop = asyncio.get_running_loop()
            transcription = await loop.run_in_executor(None, functools.partial(self._stt.transcribe_bytes, audio_bytes, 16000))
            
            if transcription:
                await self._handle_text_query(transcription)
        except Exception as e:
            logger.error("Handle audio chunk error: %s", e)
            await self._send({"type": "error", "message": str(e)})

    async def _handle_snapshot(self) -> None:
        """Forces an immediate scene description."""
        self._scene_detector.force_reset()
        await self._auto_describe()

    async def _send(self, payload: dict) -> None:
        """Helper to send JSON securely."""
        try:
            await self._ws.send_json(payload)
        except Exception:
            pass
