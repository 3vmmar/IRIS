"""
VLM call manager — coordinates frame encoding, prompt construction,
context injection, and Claude API streaming.

Every user query and automatic scene description flows through this module.

Imports allowed: config.settings, agent.prompts, agent.context,
                 anthropic, numpy, PIL.Image, base64, io, time, logging.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from typing import AsyncIterator, Optional

import anthropic
import numpy as np
from PIL import Image

from agent.context import ConversationContext
from agent import prompts
from config import settings

logger = logging.getLogger(__name__)


class Brain:
    """Manages all Claude API interactions for IRIS.

    Owns a :class:`~agent.context.ConversationContext` and an
    ``AsyncAnthropic`` client. Provides two streaming entry points:

    * :meth:`respond_stream` — user-initiated queries (added to context)
    * :meth:`describe_scene` — automatic scene descriptions (not in context)

    Usage::

        brain = Brain()

        async for token in brain.respond_stream(
            query="What's on my desk?",
            frame=frame_bgr,
            detections=[{"label": "laptop", "confidence": 0.91, "zone": "center"}],
            memory_context="laptop: center zone, just now",
        ):
            await ws.send_json({"type": "text_token", "token": token})
    """

    _MAX_TOKENS = 512

    def __init__(self) -> None:
        """Initialise the Anthropic client and conversation context.

        Raises:
            RuntimeError: If ``ANTHROPIC_API_KEY`` is empty or unset.
        """
        if not settings.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to .env before starting IRIS."
            )
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._context = ConversationContext()
        logger.info("Brain initialised (model=%s).", settings.vlm_model)

    # ─────────────────────────── frame encoding ───────────────────────────────

    @staticmethod
    def _encode_frame(frame: np.ndarray) -> str:
        """Convert an OpenCV BGR frame to a base64-encoded JPEG string.

        Args:
            frame: BGR image array (H×W×3, uint8) from OpenCV.

        Returns:
            Raw base64 string of the JPEG-encoded image (not a data URL).
        """
        # BGR → RGB
        rgb = frame[:, :, ::-1]
        img = Image.fromarray(rgb)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.standard_b64encode(buf.getvalue()).decode("utf-8")

    # ─────────────────────────── user queries ─────────────────────────────────

    async def respond_stream(
        self,
        query: str,
        frame: Optional[np.ndarray],
        detections: list[dict],
        memory_context: str,
    ) -> AsyncIterator[str]:
        """Stream a Claude response for a user query, then commit to context.

        Builds the full structured prompt (detections + memory + image +
        history), calls the Anthropic streaming API, and yields each text
        token as it arrives. The completed turn is committed to conversation
        context after all tokens have been yielded.

        Args:
            query:          The user's text or transcribed voice query.
            frame:          Current camera frame (BGR ndarray) or ``None``.
            detections:     Detection dicts from the latest inference pass.
            memory_context: Formatted string from
                            :meth:`~core.memory.VisualMemory.recent_context`.

        Yields:
            Individual text-delta strings from the Claude streaming API.

        Raises:
            RuntimeError: On API authentication or network failure.
        """
        parts: list[str] = []
        async for token in self._stream_impl(
            query=query,
            frame=frame,
            detections=detections,
            memory_context=memory_context,
            model=settings.vlm_model,
        ):
            parts.append(token)
            yield token

        # Commit completed turn
        full_response = "".join(parts)
        if full_response:
            self._context.add_turn(query, full_response)
            logger.info("Turn committed (%d chars).", len(full_response))

    # ─────────────────────────── scene descriptions ───────────────────────────

    async def describe_scene(
        self,
        frame: Optional[np.ndarray],
        detections: list[dict],
        memory_context: str,
    ) -> AsyncIterator[str]:
        """Stream an automatic scene description using the stronger model.

        Uses ``claude-sonnet-4-6`` regardless of ``settings.vlm_model`` since
        scene descriptions warrant deeper analysis. Does **not** add the
        result to conversation context (this is an automatic trigger, not
        a user-initiated turn).

        Args:
            frame:          Current camera frame or ``None``.
            detections:     Detection dicts from the latest inference pass.
            memory_context: Formatted memory context string.

        Yields:
            Individual text-delta strings.
        """
        async for token in self._stream_impl(
            query="Describe what you see in detail.",
            frame=frame,
            detections=detections,
            memory_context=memory_context,
            model="claude-sonnet-4-6",
        ):
            yield token

    # ─────────────────────────── context management ───────────────────────────

    def clear_context(self) -> None:
        """Clear all stored conversation turns.

        Call this when starting a new session so prior context does not
        leak into the new conversation.
        """
        self._context.clear()
        logger.info("Conversation context cleared.")

    # ─────────────────────────── internal streaming ───────────────────────────

    async def _stream_impl(
        self,
        query: str,
        frame: Optional[np.ndarray],
        detections: list[dict],
        memory_context: str,
        model: str,
    ) -> AsyncIterator[str]:
        """Core streaming implementation shared by respond_stream and describe_scene.

        Args:
            query:          User or system query string.
            frame:          Camera frame or None.
            detections:     Detection dicts.
            memory_context: Memory context string.
            model:          Anthropic model identifier to use.

        Yields:
            Text-delta tokens from the Claude streaming response.
        """
        # ── Build text content ────────────────────────────────────────────────
        detections_text = prompts.format_detections(detections)
        user_message    = prompts.build_user_message(query, detections_text, memory_context)

        # ── Build Anthropic content block ─────────────────────────────────────
        user_content: list[dict] = []

        if frame is not None:
            try:
                loop = asyncio.get_event_loop()
                b64 = await loop.run_in_executor(None, self._encode_frame, frame)
                user_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64,
                    },
                })
            except Exception as exc:
                logger.warning("Frame encoding failed — omitting image: %s", exc)

        user_content.append({"type": "text", "text": user_message})

        # ── Assemble messages ─────────────────────────────────────────────────
        context_turns = self._context.to_messages(prompts.SYSTEM_PROMPT)
        messages = prompts.build_messages(context_turns, user_content)

        logger.info(
            "Claude request → model=%s, turns=%d, has_image=%s",
            model,
            self._context.turn_count(),
            frame is not None,
        )

        # ── Stream from Anthropic ─────────────────────────────────────────────
        t0 = time.monotonic()

        try:
            async with self._client.messages.stream(
                model=model,
                max_tokens=self._MAX_TOKENS,
                system=prompts.SYSTEM_PROMPT,
                messages=messages,
            ) as stream:
                async for text_chunk in stream.text_stream:
                    yield text_chunk

        except anthropic.AuthenticationError:
            logger.error("Anthropic authentication failed — check ANTHROPIC_API_KEY.")
            yield "[Error: invalid API key]"
            return
        except anthropic.RateLimitError:
            logger.warning("Anthropic rate limit hit.")
            yield "[Error: rate limit — please wait a moment]"
            return
        except anthropic.APIError as exc:
            logger.error("Anthropic API error: %s", exc)
            yield f"[Error: {exc}]"
            return

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        logger.debug("Claude response streamed in %.0f ms.", elapsed_ms)
