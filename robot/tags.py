"""AprilTag detection for the return leg.

On hardware this uses pupil-apriltags. Against the simulator there is no real tag, so
the caller falls back to the color backend looking for HOME_TAG_LABEL.
"""

from __future__ import annotations

import cv2
import numpy as np

_detector = None
_unavailable = False


def _get_detector():
    global _detector, _unavailable
    if _unavailable:
        return None
    if _detector is None:
        try:
            from pupil_apriltags import Detector  # type: ignore

            _detector = Detector(families="tag36h11", nthreads=2, quad_decimate=2.0)
        except Exception:
            # No wheels on very new Pythons. Degrading to color is the whole point of
            # having a fallback, so do not make noise about it on every frame.
            _unavailable = True
            return None
    return _detector


def find_tag(frame: np.ndarray, tag_id: int) -> tuple[float, float, float, float] | None:
    """Normalized bbox of the requested tag, or None."""
    detector = _get_detector()
    if detector is None:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    for det in detector.detect(gray):
        if det.tag_id != tag_id:
            continue
        xs = [p[0] for p in det.corners]
        ys = [p[1] for p in det.corners]
        return (min(xs) / w, min(ys) / h, max(xs) / w, max(ys) / h)
    return None
