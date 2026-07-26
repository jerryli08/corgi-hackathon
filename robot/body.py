"""One async facade over the three pieces of hardware.

The drive base talks blocking serial and the arm sleeps its way through a joint
interpolation, so every hardware call goes out to a worker thread. The skill loop
above this line gets to be plain async code that never blocks the event loop, which
is what keeps the camera stream and the web UI responsive while the robot is moving.
"""

from __future__ import annotations

import asyncio

import numpy as np

from robot.arm import ArmBase, make_arm
from robot.camera import CameraBase, make_camera
from robot.config import ARM_ENABLED, CAMERA_ENABLED, MOCK, STEP_MS
from robot.drive import DriveController, make_drive


class Body:
    def __init__(
        self,
        drive: DriveController,
        arm: ArmBase,
        camera: CameraBase | None = None,
    ) -> None:
        self.drive_base = drive
        self.arm = arm
        self.camera = camera

    # -- camera -----------------------------------------------------------
    @property
    def has_camera(self) -> bool:
        return self.camera is not None

    async def jpeg(self) -> bytes:
        if self.camera is None:
            raise RuntimeError("no camera")
        return await asyncio.to_thread(self.camera.jpeg)

    async def frame(self) -> np.ndarray:
        if self.camera is None:
            raise RuntimeError("no camera")
        return await asyncio.to_thread(self.camera.frame)

    # -- arm --------------------------------------------------------------
    async def pose(self, name: str, ms: int = 800, reach: str = "mid") -> list[float]:
        return await asyncio.to_thread(self.arm.go_to_pose, name, ms, reach)

    async def joints(self, positions: list[float], ms: int = 800) -> list[float]:
        return await asyncio.to_thread(self.arm.go_to_joints, positions, ms)

    async def gripper(self, state: str) -> bool:
        return await asyncio.to_thread(self.arm.set_gripper, state)

    # -- base -------------------------------------------------------------
    async def drive(self, linear: float = 0.0, angular: float = 0.0, ms: int = STEP_MS) -> None:
        await asyncio.to_thread(self.drive_base.drive_velocity, linear, angular, ms)

    async def drive_us(self, left_us: int, right_us: int, ms: int | None = None) -> str:
        return await asyncio.to_thread(self.drive_base.drive_us, left_us, right_us, ms)

    async def stop(self) -> None:
        await asyncio.to_thread(self.drive_base.stop)

    async def estop(self) -> None:
        await self.stop()
        await asyncio.to_thread(self.arm.relax)

    # -- lifecycle --------------------------------------------------------
    def close(self) -> None:
        try:
            self.drive_base.close()
        finally:
            if self.camera is not None:
                self.camera.close()


def make_body(mock: bool = MOCK) -> tuple[Body, list[str]]:
    """Build a Body from whatever hardware is actually present.

    Returns the body plus a list of human-readable notes about anything that could not
    be brought up. Nothing here raises: a missing camera or arm should degrade the demo,
    not stop the server from booting.
    """
    notes: list[str] = []

    drive = make_drive(mock)
    if not mock:
        try:
            port = drive.connect()
            notes.append(f"drive connected on {port}")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"drive not connected: {exc}")

    arm = make_arm(mock=mock, enabled=ARM_ENABLED or mock)
    if not arm.present:
        notes.append("no arm (drive-only mode: the robot escorts the item instead of lifting it)")

    camera: CameraBase | None = None
    if CAMERA_ENABLED:
        try:
            camera = make_camera(mock)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"camera unavailable: {exc}")
    else:
        notes.append("camera disabled")

    return Body(drive=drive, arm=arm, camera=camera), notes
