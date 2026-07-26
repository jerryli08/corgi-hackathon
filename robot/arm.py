"""SO-101 arm: joint-space keyframe playback. No IK, on purpose.

Three implementations behind one interface:

    NullArm     no arm bolted on. The robot drives to the item and presents it, and
                the skill loop still runs end to end. This is the default.
    MockArm     simulated arm, grasps resolve against the simulator.
    FeetechArm  STS3215 servos on a half-duplex TTL bus.
"""

from __future__ import annotations

import threading
import time

from robot.config import (
    CENTER_TICKS,
    GRIPPER_CLOSED_DEG,
    GRIPPER_EMPTY_THRESHOLD_DEG,
    GRIPPER_OPEN_DEG,
    JOINT_SIGN,
    MOVE_HZ,
    SERVO_BAUD,
    SERVO_IDS,
    SERVO_PORT,
    TICKS_PER_DEG,
)
from robot.poses import resolve


class ArmBase:
    present = True

    def __init__(self) -> None:
        self._positions = resolve("HOME")
        self._moving = False
        self._gripper_command = "open"
        self._lock = threading.Lock()

    @property
    def positions(self) -> list[float]:
        return list(self._positions)

    @property
    def moving(self) -> bool:
        return self._moving

    @property
    def holding(self) -> bool:
        """Commanded shut but stopped short. Position alone is not enough: an open
        gripper also sits well above the threshold."""
        return (
            self._gripper_command == "close"
            and self._positions[5] > GRIPPER_EMPTY_THRESHOLD_DEG
        )

    def go_to_pose(self, name: str, ms: int = 800, reach: str = "mid") -> list[float]:
        return self.go_to_joints(resolve(name, reach), ms)

    def go_to_joints(self, target: list[float], ms: int = 800) -> list[float]:
        """Linear interpolation in joint space. Blocking, and that is fine --
        every caller wants the move finished before it does anything else."""
        with self._lock:
            self._moving = True
            start = list(self._positions)
            steps = max(1, int((ms / 1000.0) * MOVE_HZ))
            period = (ms / 1000.0) / steps
            try:
                for i in range(1, steps + 1):
                    a = i / steps
                    frame = [s + (t - s) * a for s, t in zip(start, target, strict=True)]
                    self._write(frame)
                    self._positions = frame
                    time.sleep(period)
            finally:
                self._moving = False
            return list(self._positions)

    def set_gripper(self, state: str) -> bool:
        """Returns closed_on_object. This is the whole grasp-failure detector."""
        self._gripper_command = state
        target = list(self._positions)
        target[5] = GRIPPER_OPEN_DEG if state == "open" else GRIPPER_CLOSED_DEG
        self.go_to_joints(target, ms=500)
        if state == "open":
            return False
        return self._gripper_blocked()

    # -- subclass hooks ---------------------------------------------------
    def _write(self, degrees: list[float]) -> None:
        raise NotImplementedError

    def _gripper_blocked(self) -> bool:
        raise NotImplementedError

    def relax(self) -> None:
        pass


class NullArm(ArmBase):
    """No arm attached. Poses are no-ops and every close 'succeeds', so the drive and
    perception half of the demo can run on its own without the skill loop concluding
    that it failed to pick anything up."""

    present = False

    @property
    def holding(self) -> bool:
        return False

    def go_to_joints(self, target: list[float], ms: int = 800) -> list[float]:
        return list(self._positions)

    def set_gripper(self, state: str) -> bool:
        self._gripper_command = state
        return state == "close"

    def _write(self, degrees: list[float]) -> None:
        pass

    def _gripper_blocked(self) -> bool:
        return True


class MockArm(ArmBase):
    def _write(self, degrees: list[float]) -> None:
        pass

    def _gripper_blocked(self) -> bool:
        from robot.world import WORLD

        got_it = WORLD.try_grasp()
        # A real gripper stops short when it is holding something; mimic that so the
        # reported position matches what the hardware would say.
        if got_it:
            self._positions[5] = GRIPPER_EMPTY_THRESHOLD_DEG + 3.0
        return got_it

    def set_gripper(self, state: str) -> bool:
        closed_on_object = super().set_gripper(state)
        if state == "open":
            from robot.world import WORLD

            WORLD.release()
        return closed_on_object


class FeetechArm(ArmBase):
    """STS3215 servos on a half-duplex TTL bus."""

    def __init__(self) -> None:
        super().__init__()
        from scservo_sdk import PortHandler, sms_sts  # type: ignore

        self._port = PortHandler(SERVO_PORT)
        if not self._port.openPort():
            raise RuntimeError(f"could not open servo port {SERVO_PORT}")
        self._port.setBaudRate(SERVO_BAUD)
        self._bus = sms_sts(self._port)
        self._positions = self._read_all()

    @staticmethod
    def _deg_to_ticks(deg: float, sign: int) -> int:
        return int(CENTER_TICKS + sign * deg * TICKS_PER_DEG)

    @staticmethod
    def _ticks_to_deg(ticks: int, sign: int) -> float:
        return sign * (ticks - CENTER_TICKS) / TICKS_PER_DEG

    def _read_all(self) -> list[float]:
        out = []
        for sid, sign in zip(SERVO_IDS, JOINT_SIGN, strict=True):
            ticks, _, _ = self._bus.ReadPos(sid)
            out.append(self._ticks_to_deg(ticks, sign))
        return out

    def _write(self, degrees: list[float]) -> None:
        for sid, sign, deg in zip(SERVO_IDS, JOINT_SIGN, degrees, strict=True):
            self._bus.WritePosEx(sid, self._deg_to_ticks(deg, sign), 2400, 50)

    def _gripper_blocked(self) -> bool:
        ticks, _, _ = self._bus.ReadPos(SERVO_IDS[5])
        actual = self._ticks_to_deg(ticks, JOINT_SIGN[5])
        self._positions[5] = actual
        return actual > GRIPPER_EMPTY_THRESHOLD_DEG

    def relax(self) -> None:
        for sid in SERVO_IDS:
            self._bus.write1ByteTxRx(sid, 40, 0)  # torque enable off


def make_arm(mock: bool, enabled: bool) -> ArmBase:
    if not enabled:
        return NullArm()
    return MockArm() if mock else FeetechArm()
