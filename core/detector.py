"""
Background inference thread that runs YOLO object detection on the latest
camera frame every FRAME_SKIP frames and caches structured results.

Allowed imports: config.settings, core.camera.CameraCapture,
                 standard library, ultralytics, numpy.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

from config import settings
from core.camera import CameraCapture

logger = logging.getLogger(__name__)


# ── Zone logic ────────────────────────────────────────────────────────────────

# 3×3 lookup table indexed by [row_idx][col_idx]
_ZONE_GRID: list[list[str]] = [
    ["top-left",    "top",    "top-right"],    # row 0 — top third
    ["left",        "center", "right"],         # row 1 — middle third
    ["bottom-left", "bottom", "bottom-right"], # row 2 — bottom third
]


def compute_zone(x_rel: float, y_rel: float) -> str:
    """Map a normalised (x_rel, y_rel) centroid to a 3×3 zone label.

    Divides the frame into a 3×3 grid and returns the named zone for
    the given fractional centroid position.

    The function is safe at the boundary values 0.0 and 1.0:
    ``int(1.0 * 3) == 3`` is clamped to 2 before lookup.

    Args:
        x_rel: Horizontal centre of the bbox as a fraction of frame width
               (0.0 = left edge, 1.0 = right edge).
        y_rel: Vertical centre of the bbox as a fraction of frame height
               (0.0 = top edge, 1.0 = bottom edge).

    Returns:
        One of the 9 zone label strings::

            "top-left"     "top"      "top-right"
            "left"         "center"   "right"
            "bottom-left"  "bottom"   "bottom-right"
    """
    col_idx = min(int(x_rel * 3), 2)   # clamp: handles x_rel == 1.0
    row_idx = min(int(y_rel * 3), 2)   # clamp: handles y_rel == 1.0
    return _ZONE_GRID[row_idx][col_idx]


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """A single detected object from one YOLO inference pass.

    Attributes:
        label:      YOLO class name (e.g. ``"person"``, ``"cup"``).
        confidence: Detection confidence score in the range [0.0, 1.0].
        x1:         Left pixel coordinate of the bounding box.
        y1:         Top pixel coordinate of the bounding box.
        x2:         Right pixel coordinate of the bounding box.
        y2:         Bottom pixel coordinate of the bounding box.
        zone:       Named 3×3 grid zone for the bbox centroid
                    (see :func:`compute_zone`).
        x_rel:      Bbox centre X as a fraction of frame width (0.0–1.0).
        y_rel:      Bbox centre Y as a fraction of frame height (0.0–1.0).
    """
    label:      str
    confidence: float
    x1:         int
    y1:         int
    x2:         int
    y2:         int
    zone:       str
    x_rel:      float
    y_rel:      float

    def to_dict(self) -> dict:
        """Return all fields as a JSON-serialisable dict.

        This is the canonical wire format sent to the frontend over WebSocket
        inside the ``detections`` message payload.

        Returns:
            Dict with keys: ``label``, ``confidence``, ``x1``, ``y1``,
            ``x2``, ``y2``, ``zone``, ``x_rel``, ``y_rel``.
        """
        return {
            "label":      self.label,
            "confidence": round(self.confidence, 4),
            "x1":         self.x1,
            "y1":         self.y1,
            "x2":         self.x2,
            "y2":         self.y2,
            "zone":       self.zone,
            "x_rel":      round(self.x_rel, 4),
            "y_rel":      round(self.y_rel, 4),
        }


@dataclass
class DetectionResult:
    """The complete output of one YOLO inference pass.

    Attributes:
        frame_id:     Monotonically increasing integer, incremented on every
                      inference call regardless of whether any objects were found.
        detections:   All :class:`Detection` objects that passed the confidence
                      threshold (may be an empty list).
        frame_width:  Width of the frame that was inferenced, in pixels.
        frame_height: Height of the frame that was inferenced, in pixels.
        inference_ms: Wall-clock time that YOLO inference took, in milliseconds.
    """
    frame_id:     int
    detections:   List[Detection]
    frame_width:  int
    frame_height: int
    inference_ms: float


# ── Detector class ────────────────────────────────────────────────────────────

class ObjectDetector:
    """Runs YOLO object detection in a dedicated daemon thread.

    Pulls the latest frame from a :class:`~core.camera.CameraCapture` instance,
    runs YOLO inference every ``settings.frame_skip`` frames, and caches the
    structured :class:`DetectionResult` so any module can read it instantly
    without blocking or waiting for the next inference pass.

    Usage::

        cam = CameraCapture()
        cam.start()

        detector = ObjectDetector(cam)
        detector.start()

        result: DetectionResult | None = detector.get_latest()

        detector.stop()
        cam.stop()
    """

    def __init__(self, camera: CameraCapture) -> None:
        """
        Args:
            camera: A started :class:`~core.camera.CameraCapture` instance.
                    ``get_frame()`` is called on each iteration of the
                    inference loop.
        """
        self._camera = camera
        self._model: Optional[YOLO] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._latest: Optional[DetectionResult] = None
        self._lock = threading.Lock()

        self._frame_counter: int = 0   # total iterations (including skipped)
        self._inference_id:  int = 0   # incremented only on actual inference runs

    # ─────────────────────────── public API ───────────────────────────────────

    def start(self) -> None:
        """Load the YOLO model and start the background inference thread.

        The model is loaded **synchronously** in the calling thread before
        the background thread starts, so ``get_latest()`` will never block
        due to model loading after ``start()`` returns.

        Logs the model path and number of detectable classes at INFO level.

        Raises:
            RuntimeError: If the model file specified in ``settings.yolo_model``
                          cannot be loaded by Ultralytics.
        """
        logger.info("Loading YOLO model: '%s' …", settings.yolo_model)
        try:
            self._model = YOLO(settings.yolo_model)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load YOLO model '{settings.yolo_model}': {exc}"
            ) from exc

        num_classes = len(self._model.names) if self._model.names else 0
        logger.info(
            "YOLO model loaded — path='%s', classes=%d.",
            settings.yolo_model, num_classes,
        )

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._inference_loop,
            name="iris-yolo-detector",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Object detector thread started (frame_skip=%d, confidence_threshold=%.2f).",
            settings.frame_skip,
            settings.yolo_confidence,
        )

    def stop(self) -> None:
        """Signal the inference thread to exit and join with a 3-second timeout.

        The YOLO model is *not* explicitly unloaded — garbage collection will
        reclaim it after all references drop.
        """
        logger.info("Stopping object detector …")
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                logger.warning(
                    "Detector thread did not exit within 3 s — abandoning."
                )
        logger.info("Object detector stopped.")

    def get_latest(self) -> Optional[DetectionResult]:
        """Return the most recent cached inference result.

        Thread-safe and non-blocking. Returns ``None`` if inference has not
        produced a result yet (e.g. called immediately after ``start()``
        before the first frame arrives).

        Returns:
            The latest :class:`DetectionResult`, or ``None``.
        """
        with self._lock:
            return self._latest

    def is_running(self) -> bool:
        """Return ``True`` if the inference thread is currently alive.

        Returns:
            ``True`` while the background daemon thread is running.
        """
        return self._thread is not None and self._thread.is_alive()

    # ─────────────────────────── inference loop ───────────────────────────────

    def _inference_loop(self) -> None:
        """Main loop executed by the background daemon thread.

        On every iteration:
        1. Increment the frame counter.
        2. Skip inference if ``frame_counter % settings.frame_skip != 0``.
        3. Call ``camera.get_frame()``; if ``None``, sleep 50 ms and continue.
        4. Run YOLO with ``verbose=False``.
        5. Filter detections below ``settings.yolo_confidence``.
        6. Compute zone and relative coordinates for each passing detection.
        7. Record inference time in milliseconds.
        8. Store the result under the lock.
        9. Sleep 10 ms to prevent the thread from pegging the CPU at 100%.
        """
        assert self._model is not None, "start() must be called before the loop runs."

        while not self._stop_event.is_set():
            self._frame_counter += 1

            # ── Frame-skip gate ───────────────────────────────────────────────
            if self._frame_counter % settings.frame_skip != 0:
                time.sleep(0.010)
                continue

            # ── Acquire frame ─────────────────────────────────────────────────
            frame: Optional[np.ndarray] = self._camera.get_frame()
            if frame is None:
                time.sleep(0.050)
                continue

            frame_h, frame_w = frame.shape[:2]

            # ── Run inference ─────────────────────────────────────────────────
            t0 = time.perf_counter()
            try:
                yolo_results = self._model(
                    frame,
                    conf=settings.yolo_confidence,
                    verbose=False,
                )
            except Exception as exc:
                logger.warning(
                    "YOLO inference error on frame %d: %s",
                    self._frame_counter, exc,
                )
                time.sleep(0.010)
                continue

            inference_ms = (time.perf_counter() - t0) * 1000.0

            # ── Parse detections ──────────────────────────────────────────────
            detections: list[Detection] = []

            for result in yolo_results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf < settings.yolo_confidence:
                        continue   # redundant safety guard

                    cls_id  = int(box.cls[0])
                    label   = self._model.names.get(cls_id, str(cls_id))
                    fx1, fy1, fx2, fy2 = (float(v) for v in box.xyxy[0])

                    # Integer pixel coordinates
                    ix1 = int(fx1)
                    iy1 = int(fy1)
                    ix2 = int(fx2)
                    iy2 = int(fy2)

                    # Normalised centroid
                    x_rel = ((fx1 + fx2) / 2.0) / frame_w
                    y_rel = ((fy1 + fy2) / 2.0) / frame_h

                    # Clamp to [0, 1] in case of floating-point overflow at edges
                    x_rel = max(0.0, min(1.0, x_rel))
                    y_rel = max(0.0, min(1.0, y_rel))

                    zone = compute_zone(x_rel, y_rel)

                    detections.append(Detection(
                        label=label,
                        confidence=conf,
                        x1=ix1, y1=iy1,
                        x2=ix2, y2=iy2,
                        zone=zone,
                        x_rel=x_rel,
                        y_rel=y_rel,
                    ))

            # ── Cache result ──────────────────────────────────────────────────
            self._inference_id += 1
            result_obj = DetectionResult(
                frame_id=self._inference_id,
                detections=detections,
                frame_width=frame_w,
                frame_height=frame_h,
                inference_ms=round(inference_ms, 2),
            )
            with self._lock:
                self._latest = result_obj

            logger.debug(
                "Inference #%d: %d detection(s) in %.1f ms.",
                self._inference_id, len(detections), inference_ms,
            )

            time.sleep(0.010)   # yield CPU between inference passes
