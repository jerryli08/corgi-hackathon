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
    JOINT_NAMES,
    JOINT_SIGN,
    MOVE_HZ,
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

    # -- the basket ------------------------------------------------------
    # Deliberately not set_gripper("open"): that means "let go of this where we are",
    # which on the simulator drops the object on the floor. These two mean "it is in
    # the basket now" and "take it back out", which is a different thing entirely.
    def stow(self) -> None:
        """Open the jaws while parked over the basket."""
        self._gripper_command = "open"
        target = list(self._positions)
        target[5] = GRIPPER_OPEN_DEG
        self.go_to_joints(target, ms=500)

    def unstow(self) -> bool:
        """Close the jaws on something lifted back out of the basket.

        Unlike a grasp off the table there is nothing to miss -- this same arm put the
        item in the basket -- so this does not consult the grip detector. It does report
        the position a real gripper would, stopped short on the object, so that
        /api/arm/state stays truthful while the robot is presenting it.
        """
        self._gripper_command = "close"
        target = list(self._positions)
        target[5] = GRIPPER_CLOSED_DEG
        self.go_to_joints(target, ms=500)
        self._positions[5] = GRIPPER_EMPTY_THRESHOLD_DEG + 3.0
        return True

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

    def stow(self) -> None:
        self._gripper_command = "open"

    def unstow(self) -> bool:
        self._gripper_command = "close"
        return True

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

    def stow(self) -> None:
        from robot.world import WORLD

        super().stow()
        WORLD.stow()

    def unstow(self) -> bool:
        from robot.world import WORLD

        if not WORLD.unstow():
            return False
        return super().unstow()


class FeetechArm(ArmBase):
    """STS3215 servos on a half-duplex TTL bus, driven through LeRobot's FeetechMotorsBus.

    We talk to the bus in raw ticks (normalize=False) and keep our own degree<->tick
    map (CENTER_TICKS + sign * deg * TICKS_PER_DEG), so the pose keyframes in poses.py
    stay meaningful without depending on a LeRobot calibration file.
    """

    def __init__(self) -> None:
        super().__init__()
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus

        # RANGE_M100_100 vs DEGREES only matters for normalized reads/writes, which we
        # never use -- every access below passes normalize=False.
        self._motors = {
            name: Motor(sid, "sts3215", MotorNormMode.RANGE_M100_100)
            for name, sid in zip(JOINT_NAMES, SERVO_IDS, strict=True)
        }
        self._bus = FeetechMotorsBus(port=SERVO_PORT, motors=self._motors)
        self._bus.connect(handshake=False)
        self._soften()
        self._bus.enable_torque()
        self._positions = self._read_all()

    def _soften(self) -> None:
        """Drop the stock P gain so the arm holds without buzzing.

        Feetech default P is 32; LeRobot's SO-101 config uses 16. We go a touch softer
        (12) because this arm is bolted to a walker frame that amplifies shake. Must run
        with torque off -- EEPROM/RAM writes that need an unlocked bus.
        """
        from lerobot.motors.feetech import OperatingMode

        with self._bus.torque_disabled():
            self._bus.configure_motors()
            for name in JOINT_NAMES:
                self._bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
                self._bus.write("P_Coefficient", name, 12)
                self._bus.write("I_Coefficient", name, 0)
                self._bus.write("D_Coefficient", name, 32)

    @staticmethod
    def _deg_to_ticks(deg: float, sign: int) -> int:
        return int(CENTER_TICKS + sign * deg * TICKS_PER_DEG)

    @staticmethod
    def _ticks_to_deg(ticks: int, sign: int) -> float:
        return sign * (ticks - CENTER_TICKS) / TICKS_PER_DEG

    def _read_all(self) -> list[float]:
        out = []
        for name, sign in zip(JOINT_NAMES, JOINT_SIGN, strict=True):
            ticks = self._bus.read("Present_Position", name, normalize=False)
            out.append(self._ticks_to_deg(int(ticks), sign))
        return out

    def _write(self, degrees: list[float]) -> None:
        goal = {
            name: self._deg_to_ticks(deg, sign)
            for name, sign, deg in zip(JOINT_NAMES, JOINT_SIGN, degrees, strict=True)
        }
        self._bus.sync_write("Goal_Position", goal, normalize=False)

    def _gripper_blocked(self) -> bool:
        gripper = JOINT_NAMES[5]
        ticks = self._bus.read("Present_Position", gripper, normalize=False)
        actual = self._ticks_to_deg(int(ticks), JOINT_SIGN[5])
        self._positions[5] = actual
        return actual > GRIPPER_EMPTY_THRESHOLD_DEG

    def relax(self) -> None:
        self._bus.disable_torque()


def make_arm(mock: bool, enabled: bool) -> ArmBase:
    if not enabled:
        return NullArm()
    return MockArm() if mock else FeetechArm()
