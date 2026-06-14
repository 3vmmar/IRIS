"""
Scene change detector — gates VLM calls by comparing consecutive
YOLO detection label sets using Jaccard distance.

Allowed imports: core.detector.DetectionResult, standard library only.
"""

from __future__ import annotations

import time
from typing import Optional

from core.detector import DetectionResult


def _jaccard_distance(a: frozenset[str], b: frozenset[str]) -> float:
    """Compute the Jaccard distance between two label sets.

    Jaccard distance = 1 - (|A ∩ B| / |A ∪ B|)

    Edge cases:
    - Both empty  → 0.0  (identical empty scenes, no change)
    - One empty   → 1.0  (completely different, always a change)

    Args:
        a: Previous label set.
        b: Current label set.

    Returns:
        Float in [0.0, 1.0]. Higher means more different.
    """
    union = a | b
    if not union:
        return 0.0          # both empty → no change
    intersection = a & b
    return 1.0 - len(intersection) / len(union)


class SceneChangeDetector:
    """Gates VLM calls by detecting meaningful changes in the observed scene.

    Compares the label set of consecutive :class:`~core.detector.DetectionResult`
    objects using Jaccard distance. A change is declared only when:

    1. Enough time has elapsed since the previous trigger
       (``min_interval_seconds``).
    2. The label-set distance exceeds ``label_change_threshold``.

    This prevents the VLM brain from being called on every frame while still
    reacting quickly when something genuinely new enters or leaves the scene.

    This class is **not** thread-safe — call it from a single thread only
    (either the detector thread or the WebSocket handler).

    Example::

        scene = SceneChangeDetector(label_change_threshold=0.35,
                                    min_interval_seconds=3.0)

        # In the detection loop:
        result = detector.get_latest()
        if scene.has_changed(result):
            asyncio.create_task(brain.describe_scene(frame, result))
    """

    def __init__(
        self,
        label_change_threshold: float = 0.35,
        min_interval_seconds: float = 3.0,
    ) -> None:
        """
        Args:
            label_change_threshold: Jaccard distance that must be exceeded for a
                                    scene change to be declared. 0.35 means at
                                    least 35 % of the combined label vocabulary
                                    must differ between consecutive frames.
            min_interval_seconds:   Minimum wall-clock seconds between two
                                    consecutive triggers. Prevents rapid-fire VLM
                                    calls when the scene is noisy or transitional.
        """
        self._threshold:       float = label_change_threshold
        self._min_interval:    float = min_interval_seconds
        self._prev_labels:     frozenset[str] = frozenset()
        self._last_trigger_ts: float = 0.0   # monotonic timestamp; 0 = never triggered

    # ─────────────────────────── public API ───────────────────────────────────

    def has_changed(self, result: Optional[DetectionResult]) -> bool:
        """Evaluate whether the current detection result represents a scene change.

        Returns ``True`` if **all three** conditions are met:

        1. *result* is not ``None``.
        2. At least ``min_interval_seconds`` have elapsed since the last trigger.
        3. The Jaccard distance between the previous and current label sets
           exceeds ``label_change_threshold``.

        State is updated **only** when ``True`` is returned:
        - The previous label set is replaced with the current one.
        - The last-trigger timestamp is recorded.

        When ``False`` is returned, no internal state changes — the next call
        will still compare against the same previous label set.

        Args:
            result: The latest :class:`~core.detector.DetectionResult`, or
                    ``None`` if the detector has not yet produced output.

        Returns:
            ``True`` if a meaningful scene change has been detected and the
            cooldown interval has elapsed; ``False`` otherwise.
        """
        # ── Condition 1: valid result ─────────────────────────────────────────
        if result is None:
            return False

        # ── Condition 2: cooldown elapsed ─────────────────────────────────────
        now = time.monotonic()
        if now - self._last_trigger_ts < self._min_interval:
            return False

        # ── Condition 3: label-set distance above threshold ───────────────────
        current_labels = frozenset(d.label for d in result.detections)
        distance = _jaccard_distance(self._prev_labels, current_labels)

        if distance <= self._threshold:
            return False

        # All conditions met — update state and signal change
        self._prev_labels     = current_labels
        self._last_trigger_ts = now
        return True

    def force_reset(self) -> None:
        """Clear all internal state as if the detector had just started.

        Use this when the user explicitly requests a snapshot, starts a new
        session, or navigates to a completely different environment so that
        the very next detection frame can immediately trigger a VLM call.
        """
        self._prev_labels     = frozenset()
        self._last_trigger_ts = 0.0

    def time_since_last_trigger(self) -> float:
        """Return the number of seconds elapsed since the last scene-change trigger.

        Returns:
            Seconds since the last trigger as a float. Returns ``float("inf")``
            if ``has_changed`` has never returned ``True`` (i.e. no trigger has
            ever fired).
        """
        if self._last_trigger_ts == 0.0:
            return float("inf")
        return time.monotonic() - self._last_trigger_ts
