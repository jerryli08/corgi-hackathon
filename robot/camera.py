"""Wrist camera. Grabs continuously on a thread so readers always get the newest
frame instead of a stale buffered one."""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from robot.config import CAMERA_INDEX, FRAME_H, FRAME_W, JPEG_QUALITY, STREAM_FPS

_ENCODE = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]


class CameraBase:
    def frame(self) -> np.ndarray:
        raise NotImplementedError

    def jpeg(self) -> bytes:
        ok, buf = cv2.imencode(".jpg", self.frame(), _ENCODE)
        if not ok:
            raise RuntimeError("jpeg encode failed")
        return buf.tobytes()

    def mjpeg(self, fps: int = STREAM_FPS):
        period = 1.0 / max(1, fps)
        while True:
            try:
                payload = self.jpeg()
            except Exception:
                time.sleep(period)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
            time.sleep(period)

    def close(self) -> None:
        pass


class MockCamera(CameraBase):
    def frame(self) -> np.ndarray:
        from robot.world import WORLD

        return WORLD.render()


class UsbCamera(CameraBase):
    def __init__(self, index: int = CAMERA_INDEX) -> None:
        self._cap = cv2.VideoCapture(index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self._cap.isOpened():
            raise RuntimeError(f"camera {index} would not open")
        self._latest: np.ndarray | None = None
        self._lock = threading.Lock()
        self._closed = threading.Event()
        threading.Thread(target=self._grab_loop, daemon=True).start()
        for _ in range(50):
            if self._latest is not None:
                return
            time.sleep(0.05)
        self.close()
        raise RuntimeError(f"camera {index} opened but produced no frames")

    def _grab_loop(self) -> None:
        while not self._closed.is_set():
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._latest = frame
            else:
                time.sleep(0.01)

    def frame(self) -> np.ndarray:
        with self._lock:
            if self._latest is None:
                raise RuntimeError("no frame yet")
            return self._latest.copy()

    def close(self) -> None:
        self._closed.set()
        self._cap.release()


def make_camera(mock: bool, index: int = CAMERA_INDEX) -> CameraBase:
    return MockCamera() if mock else UsbCamera(index)
