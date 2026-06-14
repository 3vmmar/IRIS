"""
Background thread that reads frames from the camera at 30fps
and exposes the latest frame through a thread-safe deque(maxlen=1).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

import cv2
import numpy as np

from config import settings

logger = logging.getLogger(__name__)


class CameraCapture:
    """Continuously captures frames from a camera device in a dedicated daemon thread.

    Design contract:
    - One background thread owns the OpenCV VideoCapture object exclusively.
    - All other modules call `get_frame()` — it is non-blocking and always returns
      the most recent frame (or None if capture hasn't started yet).
    - A deque(maxlen=1) is used as the buffer so the latest frame always overwrites
      any previous one; no queue backlog can accumulate.
    - The thread is safe to stop and restart.

    Usage:
        cam = CameraCapture()
        cam.start()
        frame = cam.get_frame()   # np.ndarray | None
        cam.stop()
    """

    # Number of frame timestamps to keep for rolling FPS calculation
    _FPS_WINDOW: int = 30

    def __init__(self) -> None:
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # deque(maxlen=1) — holds at most one frame; writes overwrite stale data
        self._buffer: deque[np.ndarray] = deque(maxlen=1)
        self._lock = threading.Lock()

        # Rolling timestamps of the last _FPS_WINDOW successful reads
        self._frame_times: deque[float] = deque(maxlen=self._FPS_WINDOW)

    # ─────────────────────────────── public API ───────────────────────────────

    def start(self) -> None:
        """Open the camera and start the background capture thread.

        Reads `settings.camera_index` to select the device. Sets capture
        resolution to 1280×720 and requests 30 FPS via OpenCV CAP_PROP
        properties (best-effort — hardware may not honour them).

        Raises:
            RuntimeError: If the camera device cannot be opened.
        """
        logger.info("Opening camera at index %d …", settings.camera_index)
        cap = cv2.VideoCapture(settings.camera_index, cv2.CAP_ANY)

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {settings.camera_index}. "
                "Check that the device is connected and not in use by another process."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        self._cap = cap
        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._capture_loop,
            name="iris-camera-capture",
            daemon=True,   # exits automatically if the main process dies
        )
        self._thread.start()
        logger.info(
            "Camera capture thread started (device %d, target 1280×720 @ 30fps).",
            settings.camera_index,
        )

    def stop(self) -> None:
        """Signal the capture thread to exit and release the camera device.

        Joins the thread with a 2-second timeout. If the thread does not exit
        within that window it is abandoned (daemon=True means it won't block
        process shutdown).
        """
        logger.info("Stopping camera capture …")
        self._stop_event.set()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning(
                    "Camera thread did not exit within 2 s — abandoning it."
                )

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        logger.info("Camera capture stopped.")

    def get_frame(self) -> np.ndarray | None:
        """Return a copy of the latest captured frame, or None if not yet available.

        This method is non-blocking. The caller receives an independent copy of
        the frame array so downstream processing cannot corrupt the buffer.

        Returns:
            np.ndarray: The most recent BGR frame (HxWx3 uint8), or None.
        """
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer[0].copy()

    def is_running(self) -> bool:
        """Return True if the capture thread is currently alive.

        Returns:
            bool: Thread liveness state.
        """
        return self._thread is not None and self._thread.is_alive()

    def get_fps(self) -> float:
        """Return the measured capture frame rate as a rolling average.

        Computes FPS over the last `_FPS_WINDOW` (30) frame timestamps.
        Returns 0.0 if fewer than two timestamps have been collected yet.

        Returns:
            float: Frames per second, rounded to one decimal place.
        """
        with self._lock:
            timestamps = list(self._frame_times)

        if len(timestamps) < 2:
            return 0.0

        elapsed = timestamps[-1] - timestamps[0]
        if elapsed <= 0.0:
            return 0.0

        fps = (len(timestamps) - 1) / elapsed
        return round(fps, 1)

    # ────────────────────────────── internal ──────────────────────────────────

    def _capture_loop(self) -> None:
        """Main loop executed by the background thread.

        Reads frames as fast as the device allows (targeting 30 fps), stores the
        latest into the deque, and updates the rolling timestamp list for FPS
        measurement.

        On read failure: skips the frame, sleeps 50 ms, and retries. After 10
        consecutive failures a warning is emitted so operators can detect a
        disconnected or failing camera without crashing the whole agent.
        """
        consecutive_failures: int = 0

        while not self._stop_event.is_set():
            if self._cap is None:
                break

            ret, frame = self._cap.read()

            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures > 10:
                    logger.warning(
                        "Camera read has failed %d consecutive times — "
                        "device may be disconnected (index %d).",
                        consecutive_failures,
                        settings.camera_index,
                    )
                time.sleep(0.05)
                continue

            # Successful read — reset failure counter and update buffer + timestamps
            consecutive_failures = 0
            now = time.monotonic()

            with self._lock:
                self._buffer.append(frame)
                self._frame_times.append(now)
