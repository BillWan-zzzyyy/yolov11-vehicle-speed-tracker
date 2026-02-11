import os
import queue
import threading
from datetime import datetime

import cv2 as cv


class AsyncVideoRecorder:
    """Asynchronously writes frames to disk to reduce main-loop blocking."""

    def __init__(
        self,
        output_dir,
        frame_size,
        fps,
        filename_prefix="tracked_video",
        codec="mp4v",
        queue_size=120,
        drop_frames=True,
        enabled=True,
    ):
        self.enabled = enabled
        self.drop_frames = drop_frames
        self.frame_size = tuple(frame_size)
        self.frames_written = 0
        self.frames_dropped = 0
        self.output_path = None
        self._queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._stop_event = threading.Event()
        self._thread = None
        self._writer = None

        if not self.enabled:
            return

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.mp4")
        fourcc = cv.VideoWriter_fourcc(*codec)
        self._writer = cv.VideoWriter(self.output_path, fourcc, float(fps), self.frame_size)

        if not self._writer.isOpened():
            self.enabled = False
            self.output_path = None
            self._writer.release()
            self._writer = None
            return

        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    def _writer_loop(self):
        while True:
            if self._stop_event.is_set() and self._queue.empty():
                break

            try:
                frame = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                if self._writer is not None:
                    self._writer.write(frame)
                    self.frames_written += 1
            finally:
                self._queue.task_done()

    def write(self, frame):
        """Queue one frame for asynchronous writing without blocking."""
        if not self.enabled or self._writer is None:
            return False

        if frame is None:
            return False

        h, w = frame.shape[:2]
        target_w, target_h = self.frame_size
        if (w, h) != (target_w, target_h):
            frame = cv.resize(frame, (target_w, target_h))

        try:
            self._queue.put_nowait(frame.copy())
            return True
        except queue.Full:
            if self.drop_frames:
                self.frames_dropped += 1
                return False

            try:
                self._queue.put(frame.copy(), timeout=0.05)
                return True
            except queue.Full:
                self.frames_dropped += 1
                return False

    def close(self):
        """Flush pending frames and release writer resources."""
        if not self.enabled:
            return {
                "enabled": False,
                "output_path": None,
                "frames_written": 0,
                "frames_dropped": 0,
            }

        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=5.0)

        if self._writer is not None:
            self._writer.release()
            self._writer = None

        return {
            "enabled": True,
            "output_path": self.output_path,
            "frames_written": self.frames_written,
            "frames_dropped": self.frames_dropped,
        }
