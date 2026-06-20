"""
System prompt, detection formatting, and message construction for the
Anthropic Messages API.

Pure functions and string constants — no classes, no I/O, no API calls.

Imports allowed: standard library only. No IRIS modules.
"""

from __future__ import annotations


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are IRIS — Intelligent Real-time Interactive Sensing.

You are an AI agent with continuous access to a live camera feed. You can see \
what the camera sees right now, and you have a persistent visual memory that \
records which objects appeared, where, and when — even after they leave the frame.

Behaviour rules:
- Keep responses under 3 sentences unless the user explicitly asks for detail. \
Your responses are spoken aloud via TTS, so brevity is essential.
- When describing where an object is, always reference its spatial zone \
(e.g. "in the bottom-right zone", "on the left side").
- The valid zones are: top-left, top, top-right, left, center, right, \
bottom-left, bottom, bottom-right.
- Never fabricate detections. If an object is not visible in the current \
frame and not in your memory, say so honestly.
- When the user asks "where is X?", check the memory context first — the \
object may have been seen recently even if it is not visible right now.
- Maintain a calm, natural conversational tone. Do not be robotic, and \
do not be over-enthusiastic. Speak as a helpful assistant who can see.\
"""


# ── Detection formatter ───────────────────────────────────────────────────────

def format_detections(detections: list[dict]) -> str:
    """Format a list of detection dicts into a readable text block.

    Each dict is expected to have at least ``label``, ``confidence``,
    and ``zone`` keys (as returned by :meth:`Detection.to_dict`).

    Args:
        detections: List of detection dicts. May be empty.

    Returns:
        A formatted string listing each detected object with its label,
        confidence percentage, and zone. Returns a "no objects" notice
        when the list is empty.

    Example::

        person — 92% confidence — center zone
        laptop — 87% confidence — bottom-right zone
    """
    if not detections:
        return "No objects detected in current frame."

    lines: list[str] = []
    for det in detections:
        label = det.get("label", "unknown")
        conf  = det.get("confidence", 0.0)
        zone  = det.get("zone", "unknown")
        lines.append(f"{label} — {conf * 100:.0f}% confidence — {zone} zone")
    return "\n".join(lines)


# ── User message builder ──────────────────────────────────────────────────────

def build_user_message(
    query: str,
    detections_text: str,
    memory_context: str,
) -> str:
    """Combine detections, memory, and user query into a single structured message.

    Each section is clearly labelled so Claude can distinguish the
    machine-generated context from the human question.

    Args:
        query:           The user's natural-language question or command.
        detections_text: Formatted string from :func:`format_detections`.
        memory_context:  Formatted string from
                         :meth:`~core.memory.VisualMemory.recent_context`.

    Returns:
        A single string with labelled sections, ready to be placed in an
        Anthropic ``text`` content block.
    """
    sections: list[str] = []

    sections.append(f"[Current Frame Detections]\n{detections_text}")

    if memory_context:
        sections.append(f"[Visual Memory]\n{memory_context}")

    sections.append(f"[User Query]\n{query}")

    return "\n\n".join(sections)


# ── Messages list builder ─────────────────────────────────────────────────────

def build_messages(
    context_turns: list[dict],
    user_content: list[dict],
) -> list[dict]:
    """Assemble the full ``messages`` list for the Anthropic Messages API.

    Prepends the conversation history (prior turns) and appends the current
    user turn as the final message.

    Args:
        context_turns: Prior ``user`` / ``assistant`` message dicts from
                       :meth:`ConversationContext.to_messages`. May be empty.
        user_content:  The current user turn as a list of Anthropic content
                       items (text blocks + optional image blocks).

    Returns:
        Complete ``messages`` list ready to pass to
        ``client.messages.stream(messages=...)``.
    """
    messages = list(context_turns)        # shallow copy — don't mutate the original
    messages.append({"role": "user", "content": user_content})
    return messages
