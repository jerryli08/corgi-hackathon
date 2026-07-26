"""A crude simulated robot, so the whole stack runs on a laptop with nothing plugged in.

This is not a physics engine. It exists so the visual-servo loop, the skill state
machine and the web UI can be built and debugged without hardware, and so grasp
failures can be reproduced on demand instead of waited for.
"""

from __future__ import annotations

import math
import random
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

from robot.config import CAMERA_FOV_DEG, FRAME_H, FRAME_W

CAM_HEIGHT_M = 0.18
# The wrist camera at the SCAN pose looks slightly downward. Without this an object
# close enough to grasp falls below the frame, its bbox shrinks, and the servo loop
# reads "still far away" and drives straight over it.
CAM_PITCH_DEG = 25.0
GRASP_MIN_M, GRASP_MAX_M = 0.16, 0.28
GRASP_MAX_BEARING_DEG = 10.0


@dataclass
class SimObject:
    label: str
    color: tuple[int, int, int]  # BGR
    x: float
    y: float
    radius_m: float = 0.035
    # Height over width when rendered. Groceries are blobs; a standing person is not,
    # and the aspect guard in the come skill only means something if the simulator
    # actually draws them differently.
    aspect: float = 1.0
    held: bool = False


@dataclass
class SimWorld:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    holding: str | None = None
    # What is aboard. Basket items travel with the robot and leave the world.
    basket: list[str] = field(default_factory=list)
    # Set >0 to make grasps fail on purpose and exercise the ask-for-help path.
    grasp_failure_rate: float = 0.0
    objects: list[SimObject] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def reset(self) -> None:
        with self._lock:
            self.x = self.y = self.theta = 0.0
            self.holding = None
            self.basket = []
            self.objects = [
                SimObject("strawberries", (60, 60, 220), 1.30, 0.28),
                SimObject("banana", (60, 210, 235), 1.15, -0.22),
                SimObject("granola bar", (70, 120, 160), 1.55, 0.02),
                SimObject("water bottle", (200, 190, 120), 1.40, -0.45),
                # stands in for the AprilTag on the customer's table
                SimObject("home_tag", (200, 60, 200), -0.35, 0.0, radius_m=0.08),
                # the person who texted. Behind and to the left, so `come` has to
                # actually turn around and look rather than finding them straight ahead.
                SimObject("person", (200, 120, 70), -1.60, 1.60, radius_m=0.16, aspect=2.2),
            ]

    # -- motion -----------------------------------------------------------
    def drive(self, linear: float, angular: float, dt: float) -> None:
        with self._lock:
            self.theta += angular * dt
            self.x += linear * math.cos(self.theta) * dt
            self.y += linear * math.sin(self.theta) * dt

    def relative(self, obj: SimObject) -> tuple[float, float]:
        """(distance_m, bearing_rad) of an object in the robot frame."""
        dx, dy = obj.x - self.x, obj.y - self.y
        dist = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx) - self.theta
        bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
        return dist, bearing

    # -- manipulation -----------------------------------------------------
    def try_grasp(self) -> bool:
        """True if something graspable is in the sweet spot and we got lucky."""
        with self._lock:
            if self.holding is not None:
                return False
            for obj in self.objects:
                if obj.held or obj.label in ("home_tag", "person"):
                    continue
                dist, bearing = self.relative(obj)
                in_zone = GRASP_MIN_M <= dist <= GRASP_MAX_M
                aligned = abs(math.degrees(bearing)) <= GRASP_MAX_BEARING_DEG
                if in_zone and aligned:
                    if random.random() < self.grasp_failure_rate:
                        return False
                    obj.held = True
                    self.holding = obj.label
                    return True
            return False

    def release(self) -> None:
        with self._lock:
            if self.holding is None:
                return
            for obj in self.objects:
                # Only what is actually in the jaws. Basket items are flagged held too,
                # and an unrelated gripper-open must not tip them onto the floor.
                if obj.held and obj.label == self.holding:
                    obj.held = False
                    # dropped just in front of wherever the robot is now
                    obj.x = self.x + 0.22 * math.cos(self.theta)
                    obj.y = self.y + 0.22 * math.sin(self.theta)
            self.holding = None

    def stow(self) -> bool:
        """Move whatever is in the jaws into the basket, where it rides along."""
        with self._lock:
            if self.holding is None:
                return False
            self.basket.append(self.holding)
            self.holding = None
            return True

    def unstow(self) -> bool:
        """Take the oldest basket item back into the jaws, first in first out."""
        with self._lock:
            if self.holding is not None or not self.basket:
                return False
            label = self.basket.pop(0)
            for obj in self.objects:
                if obj.label == label:
                    obj.held = True
                    break
            self.holding = label
            return True

    # -- rendering --------------------------------------------------------
    def render(self) -> np.ndarray:
        """A synthetic wrist-camera view. Colored blobs are enough for the servo loop."""
        frame = np.full((FRAME_H, FRAME_W, 3), 70, dtype=np.uint8)
        horizon = FRAME_H // 2
        frame[:horizon, :] = (120, 100, 90)

        half_fov = math.radians(CAMERA_FOV_DEG / 2)
        focal_px = (FRAME_W / 2) / math.tan(half_fov)

        with self._lock:
            visible = []
            for obj in self.objects:
                if obj.held:
                    continue
                dist, bearing = self.relative(obj)
                if dist < 0.05 or abs(bearing) > half_fov:
                    continue
                visible.append((dist, obj, bearing))
            visible.sort(key=lambda v: v[0], reverse=True)  # painter's: far first

            for dist, obj, bearing in visible:
                # Robot frame is x-forward, y-left, so a positive bearing (object to the
                # robot's left) must land on the LEFT of the image. Hence the minus.
                px = int(FRAME_W / 2 - math.tan(bearing) * focal_px)
                below_axis = math.atan2(CAM_HEIGHT_M, dist) - math.radians(CAM_PITCH_DEG)
                py = int(horizon + math.tan(below_axis) * focal_px)
                r = max(2, int((obj.radius_m / dist) * focal_px))
                axes = (r, max(2, int(r * obj.aspect)))
                cv2.ellipse(frame, (px, py), axes, 0, 0, 360, obj.color, -1)
                cv2.ellipse(frame, (px, py), axes, 0, 0, 360, (30, 30, 30), 1)

            banner = []
            if self.holding:
                banner.append(f"holding: {self.holding}")
            if self.basket:
                banner.append(f"basket: {', '.join(self.basket)}")
            if banner:
                cv2.rectangle(frame, (0, FRAME_H - 60), (FRAME_W, FRAME_H), (40, 40, 40), -1)
                for i, line in enumerate(banner):
                    cv2.putText(
                        frame,
                        line,
                        (12, FRAME_H - 36 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (230, 230, 230),
                        1,
                        cv2.LINE_AA,
                    )
        return frame

    def snapshot(self) -> dict:
        """Ground truth. When the servo loop converges somewhere wrong, this tells you
        immediately whether vision or control is the one lying to you."""
        with self._lock:
            objects = []
            for o in self.objects:
                dist, bearing = self.relative(o)
                objects.append(
                    {
                        "label": o.label,
                        "distance_m": round(dist, 3),
                        "bearing_deg": round(math.degrees(bearing), 1),
                        "held": o.held,
                    }
                )
            return {
                "robot": {
                    "x": round(self.x, 3),
                    "y": round(self.y, 3),
                    "theta": round(self.theta, 3),
                },
                "holding": self.holding,
                "basket": list(self.basket),
                "objects": objects,
            }


WORLD = SimWorld()
WORLD.reset()
