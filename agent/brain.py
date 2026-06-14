"""
Claude API call manager (Anthropic SDK).
Encodes the current frame as base64 JPEG, builds the full structured prompt,
and streams response tokens back to the WebSocket handler.
Uses claude-haiku-4-5 for real-time queries and claude-sonnet-4-6 for deep analysis.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Optional

import anthropic
import numpy as np

from agent.context import ConversationContext
from agent.prompts import SYSTEM_PROMPT, build_user_message
from config import settings
from core.detector import DetectionResult
from core.memory import VisualMemory

logger = logging.getLogger(__name__)



class IRISBrain:
    """Manages all Claude API interactions for IRIS.

    Responsibilities:
    - Validates the Anthropic API key at query time (not at import time).
    - Builds the full structured prompt (frame + detections + memory + history).
    - Streams response tokens via the Anthropic streaming Messages API.
    - Appends each completed (user, assistant) pair to the context buffer.
    - Supports model switching between claude-haiku-4-5 (fast) and
      claude-sonnet-4-6 (detailed) per-query.

    Usage::

        brain = IRISBrain(context=ctx, memory=memory)

        async for token in brain.query("What do you see?", frame, detections):
            ws.send_json({"type": "text_token", "token": token})
    """

    # Maximum tokens to generate per response
    _MAX_TOKENS = 1024

    def __init__(
        self,
        context: ConversationContext,
        memory: VisualMemory,
    ) -> None:
        """
        Args:
            context: Shared :class:`~agent.context.ConversationContext` instance
                     that accumulates conversation history across queries.
            memory:  Initialised :class:`~core.memory.VisualMemory` instance
                     for retrieving the formatted memory context string.
        """
        self._context = context
        self._memory = memory
        self._client: Optional[anthropic.AsyncAnthropic] = None

    # ─────────────────────────── lifecycle ────────────────────────────────────

    def init(self) -> None:
        """Instantiate the Anthropic async client.

        Raises:
            RuntimeError: If ``ANTHROPIC_API_KEY`` is empty.
        """
        if not settings.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to .env before starting IRIS."
            )
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        logger.info("IRISBrain initialised (model=%s).", settings.vlm_model)

    async def close(self) -> None:
        """Close the Anthropic HTTP client."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    # ─────────────────────────── public API ───────────────────────────────────

    async def query(
        self,
        user_text: str,
        frame: Optional[np.ndarray] = None,
        detection_result: Optional[DetectionResult] = None,
        model: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream a Claude response token-by-token for the given query.

        Builds the full prompt (vision + detections + memory + history),
        calls the Anthropic streaming Messages API, and yields each text
        token as it arrives. Appends the completed turn to context when done.

        Args:
            user_text:        The user's query (text or transcribed voice).
            frame:            Current camera frame (BGR ndarray) or None.
            detection_result: Latest YOLO26 result or None.
            model:            Override the model for this call. Defaults to
                              ``settings.vlm_model``. Pass
                              ``"claude-sonnet-4-6"`` for deep-analysis queries.

        Yields:
            Individual text token strings as they stream from the API.

        Raises:
            RuntimeError: If ``init()`` was not called, or on API failure.
        """
        if self._client is None:
            raise RuntimeError("IRISBrain.init() has not been called.")

        active_model = model or settings.vlm_model

        # ── Build memory context ──────────────────────────────────────────────
        try:
            memory_context = await self._memory.recent_context()
        except Exception as exc:
            logger.warning("Memory context retrieval failed: %s", exc)
            memory_context = ""

        # ── Build message content for this turn ───────────────────────────────
        user_content = build_user_message(
            query=user_text,
            frame=frame,
            detection_result=detection_result,
            memory_context=memory_context,
        )

        # ── Inject conversation history ───────────────────────────────────────
        history = self._context.to_messages(SYSTEM_PROMPT)
        messages = history + [{"role": "user", "content": user_content}]

        # ── Stream from Anthropic API ─────────────────────────────────────────
        full_response_parts: list[str] = []

        logger.info(
            "Claude query → model=%s, turns_in_context=%d, has_frame=%s",
            active_model,
            self._context.turn_count(),
            frame is not None,
        )

        try:
            async with self._client.messages.stream(
                model=active_model,
                max_tokens=self._MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=messages,
            ) as stream:
                async for text_chunk in stream.text_stream:
                    full_response_parts.append(text_chunk)
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

        # ── Save completed turn to context ────────────────────────────────────
        full_response = "".join(full_response_parts)
        self._context.add_turn(user_text, full_response)
        logger.info("Claude response complete (%d chars).", len(full_response))

    async def describe_scene(
        self,
        frame: Optional[np.ndarray],
        detection_result: Optional[DetectionResult],
    ) -> AsyncIterator[str]:
        """Generate an unsolicited scene description when scene change is detected.

        Uses the fast ``claude-haiku-4-5`` model regardless of the configured
        default, since scene descriptions are triggered automatically and need
        to be low-latency.

        Args:
            frame:            Current camera frame or None.
            detection_result: Latest YOLO26 result or None.

        Yields:
            Individual text token strings.
        """
        prompt = (
            "The scene has changed. Briefly describe what you now observe in 1–2 sentences. "
            "Focus on new or moved objects."
        )
        async for token in self.query(
            user_text=prompt,
            frame=frame,
            detection_result=detection_result,
            model="claude-haiku-4-5",
        ):
            yield token
