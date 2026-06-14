"""
Ephemeral conversation memory — keeps the last N (user, assistant) turn pairs
and formats them as Anthropic API message history.

Allowed imports: config.settings, standard library only.
"""

from __future__ import annotations

from typing import Optional

from config import settings


class ConversationContext:
    """Rolling buffer of conversation turns for the Anthropic Messages API.

    Keeps at most ``settings.context_window_turns`` (user, assistant) pairs.
    Older turns are automatically evicted from the front of the list when the
    limit is exceeded.

    The current query is **not** stored here — ``brain.py`` appends it
    directly to the messages list returned by :meth:`to_messages` before
    sending to the API.

    Usage::

        ctx = ConversationContext()

        # After a completed query/response cycle:
        ctx.add_turn(user="What is on my desk?",
                     assistant="I can see a laptop and a coffee cup.")

        # Before the next Claude call:
        messages = ctx.to_messages(system_prompt=SYSTEM_PROMPT)
        # → pass to anthropic client as messages=messages, system=system_prompt
    """

    def __init__(self) -> None:
        # List of (user_text, assistant_text) tuples, oldest first.
        self._turns: list[tuple[str, str]] = []

    # ─────────────────────────── public API ───────────────────────────────────

    def add_turn(self, user: str, assistant: str) -> None:
        """Append a completed (user, assistant) exchange to the buffer.

        If the buffer already contains ``settings.context_window_turns`` turns,
        the oldest turn is evicted from the front before appending.

        Args:
            user:      The user's query text (after transcription if voice input).
            assistant: The complete response text returned by Claude.
        """
        self._turns.append((user.strip(), assistant.strip()))
        while len(self._turns) > settings.context_window_turns:
            self._turns.pop(0)

    def to_messages(self, system_prompt: str) -> list[dict]:
        """Return stored turns as an Anthropic-compatible messages list.

        The ``system_prompt`` parameter is accepted for signature completeness
        and documentation clarity — it is **not** included in the returned
        list. Pass it separately to the Anthropic SDK as the top-level
        ``system`` parameter.

        The returned list contains only ``user`` / ``assistant`` role dicts,
        which the SDK requires to alternate starting with ``"user"``. Because
        ``add_turn`` always stores complete pairs, this invariant is guaranteed.

        Args:
            system_prompt: The IRIS system prompt string. Not included in the
                           return value — only here so callers have a single
                           canonical call site for the full message context.

        Returns:
            A list of ``{"role": str, "content": str}`` dicts in Anthropic
            message format, oldest turn first::

                [
                    {"role": "user",      "content": "What is on my desk?"},
                    {"role": "assistant", "content": "I can see a laptop…"},
                    ...
                ]

            Returns an empty list if no turns have been stored yet.
        """
        messages: list[dict] = []
        for user_text, assistant_text in self._turns:
            messages.append({"role": "user",      "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})
        return messages

    def last_assistant_response(self) -> Optional[str]:
        """Return the most recent assistant response text.

        Returns:
            The assistant string from the newest stored turn, or ``None`` if
            no turns have been added yet.
        """
        if not self._turns:
            return None
        return self._turns[-1][1]

    def turn_count(self) -> int:
        """Return the number of (user, assistant) pairs currently stored.

        Returns:
            Integer count of stored turns (0 when the buffer is empty).
        """
        return len(self._turns)

    def clear(self) -> None:
        """Empty the turn buffer.

        Call this when starting a new conversation session so historical
        context from a previous session does not leak into the new one.
        """
        self._turns.clear()
