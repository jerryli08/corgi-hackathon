"""Walker mode against the simulator, with the dead-man as the point of the exercise.

These tests run a real mock body, a real EventBus and the real loop task, and they
assert on `robot.world.WORLD` rather than on the commands the mode issued. That
distinction is the whole value here: a test that only checks `body.drive` was called
proves the mode talks, not that the base stops. The dead-man test in particular
samples the simulated robot's position at two later instants and requires it to be
identical, which is the closest a laptop gets to "the wheels are not turning".

Two facts set every wait in this file. The loop issues one command per WALK_STEP_MS,
so nothing observable happens in less than a step. And each of those commands stays
live in the drive watchdog for WALK_STEP_MS + WATCHDOG_GRACE_MS, so after the last
instruction the simulated base coasts for that long before it is fair to call it
stopped. Nothing sleeps for a round number of seconds.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest
import pytest_asyncio

from robot.body import Body, make_body
from robot.config import (
    WALK_ANGULAR,
    WALK_DEADMAN_MS,
    WALK_LINEAR,
    WALK_STEP_MS,
    WATCHDOG_GRACE_MS,
)
from robot.events import EventBus
from robot.poses import resolve
from robot.walker import WalkerMode
from robot.world import WORLD

STEP_S = WALK_STEP_MS / 1000.0
DEADMAN_S = WALK_DEADMAN_MS / 1000.0
# How long the simulated base can still be rolling after the mode falls silent: the
# last short command plus the grace the host watchdog allows before it stops the wheels.
COAST_S = STEP_S + WATCHDOG_GRACE_MS / 1000.0
# Enough that a slow thread hop cannot be mistaken for a base that failed to stop.
SLACK_S = 0.3

# The arm interpolation is real, just not the 900ms the mode asks for: thirteen tests
# of two 900ms moves each is most of this file's wall-clock budget and proves nothing.
POSE_MS = 40

# Position tolerance for "did not move at all", in metres. Float noise only.
STILL_M = 1e-9


@dataclass
class Rig:
    """Everything a test needs, plus the log of what was commanded."""

    walker: WalkerMode
    body: Body
    bus: EventBus
    commands: list[tuple[float, float]] = field(default_factory=list)

    def phases(self) -> list[str]:
        return [e["phase"] for e in self.bus.recent(limit=200) if e["type"] == "phase"]


@pytest_asyncio.fixture
async def rig(world):
    """A live WalkerMode over the mock body. `world` guarantees a reset simulator."""
    body, _notes = make_body(mock=True)
    bus = EventBus()
    commands: list[tuple[float, float]] = []

    real_pose = body.arm.go_to_pose

    async def quick_pose(name: str, ms: int = POSE_MS, reach: str = "mid") -> list[float]:
        return await asyncio.to_thread(real_pose, name, POSE_MS, reach)

    real_drive = body.drive

    async def recording_drive(
        linear: float = 0.0, angular: float = 0.0, ms: int = WALK_STEP_MS
    ) -> None:
        commands.append((linear, angular))
        await real_drive(linear, angular, ms=ms)

    body.pose = quick_pose  # type: ignore[method-assign]
    body.drive = recording_drive  # type: ignore[method-assign]

    walker = WalkerMode(body=body, bus=bus)
    yield Rig(walker=walker, body=body, bus=bus, commands=commands)

    await walker.stop(reason="test over")
    # Otherwise every test leaves a watchdog thread behind that still owns the world.
    body.close()


# --------------------------------------------------------------------------
# entering the mode
# --------------------------------------------------------------------------
async def test_start_raises_the_handhold_and_leaves_the_wheels_stopped(rig: Rig) -> None:
    assert await rig.walker.start() is True

    assert rig.walker.active is True
    assert rig.walker.moving is False
    assert rig.body.arm.positions == pytest.approx(resolve("HANDLE"))
    assert "WALKING" in rig.phases()

    # Entering the mode is not permission to move: nothing until someone nudges.
    await asyncio.sleep(2 * STEP_S)
    assert rig.commands == []
    assert WORLD.x == pytest.approx(0.0, abs=STILL_M)
    assert WORLD.theta == pytest.approx(0.0, abs=STILL_M)


async def test_nudge_before_start_is_refused(rig: Rig) -> None:
    assert await rig.walker.nudge("forward") is False
    assert rig.walker.active is False
    assert rig.commands == []
    assert WORLD.x == pytest.approx(0.0, abs=STILL_M)


# --------------------------------------------------------------------------
# moving
# --------------------------------------------------------------------------
async def test_nudge_forward_actually_moves_the_robot(rig: Rig) -> None:
    await rig.walker.start()
    assert await rig.walker.nudge("forward") is True

    await asyncio.sleep(3 * STEP_S)

    assert WORLD.x > 0.01
    assert rig.walker.moving is True


async def test_repeated_nudges_keep_it_moving(rig: Rig) -> None:
    await rig.walker.start()

    # Longer than one dead-man window, so this can only pass if the arriving
    # instructions are what is keeping the base alive.
    elapsed = 0.0
    while elapsed < DEADMAN_S * 1.4:
        assert await rig.walker.nudge("forward") is True
        await asyncio.sleep(STEP_S / 2)
        elapsed += STEP_S / 2

    assert rig.walker.moving is True
    assert WORLD.x > 0.05


async def test_nudge_left_turns_without_travelling(rig: Rig) -> None:
    await rig.walker.start()
    await rig.walker.nudge("left")

    await asyncio.sleep(3 * STEP_S)

    assert WORLD.theta > 0.1  # counter-clockwise, per DIRECTIONS
    assert abs(WORLD.x) < 0.005
    assert abs(WORLD.y) < 0.005


async def test_unknown_direction_is_refused_and_moves_nothing(rig: Rig) -> None:
    await rig.walker.start()

    assert await rig.walker.nudge("sideways") is False
    await asyncio.sleep(2 * STEP_S)

    assert rig.commands == []
    assert WORLD.x == pytest.approx(0.0, abs=STILL_M)
    assert WORLD.theta == pytest.approx(0.0, abs=STILL_M)
    assert rig.walker.state()["direction"] is None


async def test_commanded_speed_never_exceeds_the_walking_limits(rig: Rig) -> None:
    await rig.walker.start()
    await rig.walker.nudge("forward")
    await asyncio.sleep(2 * STEP_S)
    await rig.walker.nudge("left")
    await asyncio.sleep(2 * STEP_S)

    assert rig.commands, "the loop issued nothing to check"
    for linear, angular in rig.commands:
        assert abs(linear) <= WALK_LINEAR
        assert abs(angular) <= WALK_ANGULAR
    assert (WALK_LINEAR, 0.0) in rig.commands


# --------------------------------------------------------------------------
# the dead-man
# --------------------------------------------------------------------------
async def test_the_base_stops_when_the_nudges_stop(rig: Rig) -> None:
    """The one that matters. One nudge, then silence, and the base has to stop itself."""
    await rig.walker.start()
    await rig.walker.nudge("forward")

    await asyncio.sleep(DEADMAN_S + COAST_S + SLACK_S)
    settled = WORLD.x
    assert settled > 0.01, "it never moved, so this proves nothing about stopping"

    await asyncio.sleep(3 * STEP_S)
    assert WORLD.x == pytest.approx(settled, abs=STILL_M)

    assert rig.walker.moving is False
    # A dead-man stop is a hold, not an exit: the arm stays up and the mode stays live.
    assert rig.walker.active is True
    assert "HOLDING" in rig.phases()
    assert rig.walker.state()["deadman_ms_left"] == 0


async def test_hold_stops_the_wheels_but_stays_in_the_mode(rig: Rig) -> None:
    await rig.walker.start()
    await rig.walker.nudge("forward")
    await asyncio.sleep(2 * STEP_S)

    await rig.walker.hold()
    assert rig.walker.active is True
    assert rig.walker.moving is False

    await asyncio.sleep(0.05)  # let any watchdog tick already in flight land
    settled = WORLD.x
    await asyncio.sleep(2 * STEP_S)
    assert WORLD.x == pytest.approx(settled, abs=STILL_M)
    assert rig.walker.state()["direction"] is None
    assert "HOLDING" in rig.phases()


# --------------------------------------------------------------------------
# leaving the mode
# --------------------------------------------------------------------------
async def test_stop_clears_active_and_folds_the_arm_home(rig: Rig) -> None:
    await rig.walker.start()
    await rig.walker.nudge("forward")
    await asyncio.sleep(STEP_S)

    await rig.walker.stop()

    assert rig.walker.active is False
    assert rig.walker.moving is False
    assert rig.body.arm.positions == pytest.approx(resolve("HOME"))
    assert "STANDING_BY" in rig.phases()

    await asyncio.sleep(0.05)
    settled = WORLD.x
    await asyncio.sleep(2 * STEP_S)
    assert WORLD.x == pytest.approx(settled, abs=STILL_M)


async def test_stop_twice_and_stop_without_start_are_both_safe(rig: Rig) -> None:
    await rig.walker.stop()  # never started
    assert rig.walker.active is False
    # Nothing was ours to move, so the arm is still where it powered on.
    assert rig.body.arm.positions == pytest.approx(resolve("HOME"))

    await rig.walker.start()
    await rig.walker.stop()
    await rig.walker.stop()

    assert rig.walker.active is False
    assert rig.body.arm.positions == pytest.approx(resolve("HOME"))


# --------------------------------------------------------------------------
# what the API reads
# --------------------------------------------------------------------------
async def test_state_reports_the_documented_fields(rig: Rig) -> None:
    idle = rig.walker.state()
    assert set(idle) == {
        "active",
        "moving",
        "direction",
        "linear",
        "angular",
        "deadman_ms_left",
        "session_s",
        "reason",
    }
    assert idle["active"] is False
    assert idle["direction"] is None
    assert idle["deadman_ms_left"] == 0

    await rig.walker.start()
    await rig.walker.nudge("forward")
    state = rig.walker.state()

    assert state["active"] is True
    assert state["moving"] is True
    assert state["direction"] == "forward"
    assert state["linear"] == pytest.approx(WALK_LINEAR)
    assert state["angular"] == pytest.approx(0.0)
    assert 0 < state["deadman_ms_left"] <= WALK_DEADMAN_MS
    assert isinstance(state["deadman_ms_left"], int)
    assert state["reason"] == "requested"

    await rig.walker.stop(reason="done")
    assert rig.walker.state()["reason"] == "done"
