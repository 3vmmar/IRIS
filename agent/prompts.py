"""
System prompt templates and builder functions.
Covers: scene Q&A, object location grounding, memory context injection,
and multi-turn conversation formatting for the Anthropic Messages API.
"""

from __future__ import annotations

import base64
from typing import Optional

import cv2
import numpy as np

from core.detector import DetectionResult

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are IRIS — Intelligent Real-time Interactive Sensing.

You are an AI agent with real-time access to a camera feed. You can see what the camera sees, \
remember where objects were located in the past, and reason about the environment to answer \
questions conversationally.

Guidelines:
- Answer concisely and directly. Avoid filler phrases like "Certainly!" or "Great question!".
- When describing object locations, use the spatial zone vocabulary: \
  top-left, top, top-right, left, center, right, \
  bottom-left, bottom, bottom-right.
- When answering "where is X?", check the memory context first if the object isn't in the current frame.
- If you are uncertain about something you see, say so — do not hallucinate.
- Keep responses to 1–3 sentences unless the user asks for detail.
- For yes/no questions, lead with the answer, then add one sentence of supporting detail.
"""

# ── Detection context formatter ────────────────────────────────────────────────

def format_detections(result: Optional[DetectionResult]) -> str:
    """Format a DetectionResult into a compact natural-language detection list.

    This string is injected into the user turn prompt so Claude knows exactly
    what YOLO26 saw at the moment of the query — without requiring Claude to
    perform its own object detection from scratch.

    Args:
        result: The latest :class:`~core.detector.DetectionResult`, or ``None``
                if the detector hasn't produced output yet.

    Returns:
        A formatted multi-line string, e.g.::

            [Current detections — frame 42]
            • person       in center         (conf 0.91)
            • laptop       in bottom-left    (conf 0.87)
            • cup          in top-right      (conf 0.74)

        Returns ``"[No detections available]"`` when result is None or empty.
    """
    if result is None or not result.detections:
        return "[No detections available]"

    lines = [f"[Current detections — frame {result.frame_id}]"]
    for det in sorted(result.detections, key=lambda d: d.confidence, reverse=True):
        lines.append(
            f"• {det.label:<12} in {det.zone:<16} (conf {det.confidence:.2f})"
        )
    return "\n".join(lines)


# ── Frame encoder ──────────────────────────────────────────────────────────────

def encode_frame_base64(frame: np.ndarray, quality: int = 70) -> str:
    """Encode a BGR OpenCV frame as a base64 JPEG string for the Anthropic vision API.

    Args:
        frame:   BGR image array (HxWx3, uint8).
        quality: JPEG quality 1–100. 70 balances token cost vs. image clarity.
                 Lower values reduce latency and Anthropic API cost.

    Returns:
        Base64-encoded string of the JPEG-compressed image.

    Raises:
        ValueError: If the frame cannot be JPEG-encoded by OpenCV.
    """
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    ret, buf = cv2.imencode(".jpg", frame, encode_params)
    if not ret:
        raise ValueError("OpenCV failed to JPEG-encode the camera frame.")
    return base64.standard_b64encode(buf.tobytes()).decode("utf-8")


# ── Full prompt builder ────────────────────────────────────────────────────────

def build_user_message(
    query: str,
    frame: Optional[np.ndarray],
    detection_result: Optional[DetectionResult],
    memory_context: str,
) -> list[dict]:
    """Build the ``content`` list for a user turn in the Anthropic Messages API.

    Constructs a multi-part content block that includes:
    1. The JPEG-encoded camera frame (vision input).
    2. The YOLO26 detection summary (structured object list).
    3. The visual memory context (recent sightings from SQLite).
    4. The user's natural-language query.

    Args:
        query:            The user's text or transcribed voice query.
        frame:            Latest camera frame (BGR). Pass ``None`` if unavailable.
        detection_result: Latest YOLO26 result. Pass ``None`` if unavailable.
        memory_context:   Pre-formatted string from
                          :meth:`~core.memory.VisualMemory.get_context_string`.
                          Pass empty string if memory is empty.

    Returns:
        A list of Anthropic content blocks (``image`` + ``text``) suitable for
        the ``"user"`` role in the Messages API ``messages`` parameter.
    """
    content: list[dict] = []

    # ── 1. Camera frame (vision) ──────────────────────────────────────────────
    if frame is not None:
        try:
            b64 = encode_frame_base64(frame)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64,
                },
            })
        except ValueError as exc:
            # Don't crash — just omit the image and let Claude work from text context
            pass

    # ── 2. Text block: detections + memory + query ────────────────────────────
    text_parts: list[str] = []

    detection_str = format_detections(detection_result)
    text_parts.append(detection_str)

    if memory_context:
        text_parts.append(memory_context)

    text_parts.append(f"User query: {query}")

    content.append({
        "type": "text",
        "text": "\n\n".join(text_parts),
    })

    return content
