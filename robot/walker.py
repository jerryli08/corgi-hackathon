"""Walker mode: a paced escort, and the dead-man switch that makes it acceptable.

The robot travels at walking speed beside someone with the arm held up as a handhold
reference -- something at a known height to steady a hand against. It walks with the
person. It is not a mobility aid, it does not take anyone's weight, and it weighs a
couple of kilos: two hobby servos on 7.4V.

The load-bearing idea is the dead-man. This base is open loop -- no encoders, no force
sensing, no way for the code to notice that the person beside it has stopped walking.
The one thing that makes moving it next to someone unsteady defensible is that the
wheels turn ONLY while fresh instructions keep arriving. `nudge()` sets a direction and
pushes a deadline out; the loop re-issues one short velocity command per WALK_STEP_MS
for as long as that deadline is in the future; silence stops the base within
WALK_DEADMAN_MS. Every layer below agrees: the host watchdog in drive.py stops the
wheels when the next command is late, and the firmware stops them if this process dies.

This is a long-running interactive mode rather than a one-shot skill, which is why it
lives here instead of in `Skills`.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from typing import TYPE_CHECKING

from robot.config import (
    WALK_ANGULAR,
    WALK_DEADMAN_MS,
    WALK_LINEAR,
    WALK_MAX_S,
    WALK_RECHECK_STEPS,
    WALK_STEP_MS,
)
from robot.events import EventBus, phrase, progress

if TYPE_CHECKING:  # skills imports nothing from here, but keep the runtime graph flat
    from robot.skills import Skills

# The four things a person can ask for, as (linear m/s, angular rad/s). Positive
# angular is counter-clockwise, so "left" is positive -- see velocity_to_us in drive.py.
# Walking speeds only: MAX_LINEAR has no business in this module.
DIRECTIONS: dict[str, tuple[float, float]] = {
    "forward": (WALK_LINEAR, 0.0),
    "back": (-WALK_LINEAR, 0.0),
    "left": (0.0, WALK_ANGULAR),
    "right": (0.0, -WALK_ANGULAR),
}

LOST_SIGHT = "I've lost sight of you, I'll wait here"


class WalkerMode:
    """Paced escort. The robot travels at walking speed beside the person with the arm
    held out as a handhold reference. It is NOT load-bearing and no string in here may
    suggest otherwise.

    Dead-man operated: the wheels turn only while instructions keep arriving. One
    missed instruction and the base stops within WALK_DEADMAN_MS. That is what makes an
    open-loop hobby-servo base acceptable next to someone unsteady.
    """

    def __init__(self, body, bus: EventBus, skills: Skills | None = None) -> None:
        self.body = body
        self.bus = bus
        self.skills = skills
        self.last_error = ""

        self._active = False
        self._session_id = ""
        self._direction: str | None = None
        self._cmd: tuple[float, float] = (0.0, 0.0)
        self._deadline = 0.0  # monotonic; the dead-man
        self._started_at = 0.0
        self._steps = 0
        self._reason = ""
        self._phase = ""
        self._loop: asyncio.Task | None = None

    # -- what the rest of the app asks --------------------------------------
    @property
    def active(self) -> bool:
        return self._active

    @property
    def moving(self) -> bool:
        return (
            self._active
            and self._direction is not None
            and time.monotonic() < self._deadline
        )

    def state(self) -> dict:
        now = time.monotonic()
        left_s = max(0.0, self._deadline - now) if self._active else 0.0
        return {
            "active": self._active,
            "moving": self.moving,
            "direction": self._direction,
            "linear": self._cmd[0],
            "angular": self._cmd[1],
            "deadman_ms_left": int(left_s * 1000),
            "session_s": round(now - self._started_at, 1) if self._active else 0.0,
            # Why we are walking while active; why we stopped once we are not.
            "reason": self._reason,
        }

    # -- the mode -----------------------------------------------------------
    async def start(self, *, reason: str = "requested") -> bool:
        """Arm to HANDLE, wheels stopped, emit WALKING. Idempotent: starting while
        active just extends the session. Refuses (returns False) if a skill is running.
        """
        if self._skill_running():
            return False

        if self._active:
            # A second start is a fresh instruction like any other: it buys more
            # session time, and more dead-man time if a direction is already set.
            self._started_at = time.monotonic()
            self._deadline = max(self._deadline, time.monotonic() + WALK_DEADMAN_MS / 1000.0)
            return True

        self._active = True
        self._session_id = f"walk_{uuid.uuid4().hex[:6]}"
        self._reason = reason
        self._started_at = time.monotonic()
        self._steps = 0
        self._direction = None
        self._cmd = (0.0, 0.0)
        self._deadline = 0.0  # nothing moves until someone nudges

        await self._stop_wheels()
        await self._park_arm("HANDLE")
        self._emit("WALKING")
        self._loop = asyncio.create_task(self._run())
        return True

    async def nudge(self, direction: str, ms: int | None = None) -> bool:
        """direction in ("forward", "back", "left", "right"). Extends the dead-man
        deadline by WALK_DEADMAN_MS and sets the current command. Returns False if not
        active or the direction is unknown. Speeds come from WALK_LINEAR / WALK_ANGULAR
        only -- walker mode never uses MAX_LINEAR.

        It does not drive. The loop does, so that there is exactly one place where a
        command can be issued and exactly one place that checks the deadline first.
        """
        if not self._active:
            return False
        cmd = DIRECTIONS.get(direction)
        if cmd is None:
            return False

        # A caller may ask for less dead-man time than the default but never for more:
        # nobody gets to buy their way out of the one safety property this mode has.
        hold_ms = WALK_DEADMAN_MS if ms is None else max(1, min(int(ms), WALK_DEADMAN_MS))
        self._direction = direction
        self._cmd = cmd
        self._deadline = time.monotonic() + hold_ms / 1000.0
        return True

    async def hold(self) -> None:
        """Stop the wheels, stay in the mode, arm stays out. Emits HOLDING."""
        await self._hold()

    async def stop(self, *, reason: str = "done") -> None:
        """Leave the mode: stop the wheels, arm to HOME, emit STANDING_BY. Safe to call
        when not active, twice in a row, and from inside an exception handler.
        """
        was_active = self._active
        self._active = False
        self._direction = None
        self._cmd = (0.0, 0.0)
        self._deadline = 0.0
        self._reason = reason

        loop, self._loop = self._loop, None
        # The timeout and crash paths call stop() from inside the loop task itself, and
        # a task cannot cancel-and-await itself. Everyone else gets the real cancel.
        if loop is not None and loop is not asyncio.current_task():
            loop.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await loop

        if not was_active:
            return  # not our wheels and not our arm to move

        await self._stop_wheels()
        await self._park_arm("HOME")
        self._emit("STANDING_BY")

    # -- the loop -----------------------------------------------------------
    async def _run(self) -> None:
        step_s = WALK_STEP_MS / 1000.0
        try:
            while self._active:
                await asyncio.sleep(step_s)
                now = time.monotonic()

                if now - self._started_at >= WALK_MAX_S:
                    await self.stop(reason="timed out")
                    return

                if self._direction is None:
                    continue

                if now >= self._deadline:
                    # The dead-man fired: no fresh instruction arrived in time.
                    await self._hold()
                    continue

                # One short command per step. Never one long command with a sleep
                # through it: drive_velocity's `ms` is what arms the host watchdog, so
                # a long ms means the watchdog will not stop the wheels until that long
                # deadline expires -- the person could let go and the base would keep
                # rolling. And a loop asleep inside its own command is checking neither
                # the dead-man nor whether the person is still there.
                self._steps += 1
                linear, angular = self._cmd
                self._emit("WALKING")
                await self.body.drive(linear, angular, ms=WALK_STEP_MS)

                if self._steps % WALK_RECHECK_STEPS == 0 and not await self._still_there():
                    # No hunting: someone may have a hand on the robot, and driving off
                    # to find them is the one thing that would pull them over.
                    await self._hold(LOST_SIGHT)
        except asyncio.CancelledError:
            # No awaiting on the way out. A second cancellation could interrupt the
            # thread hop and leave the wheels turning, so stop them from here.
            self._stop_wheels_now()
            raise
        # A crashed walker loop must never leave the base driving.
        except Exception as exc:
            self.last_error = repr(exc)
            await self.stop(reason="stopped after an error")

    async def _hold(self, human_text: str | None = None) -> None:
        self._direction = None
        self._cmd = (0.0, 0.0)
        self._deadline = 0.0
        await self._stop_wheels()
        self._emit("HOLDING", human_text)

    async def _still_there(self) -> bool:
        """Is the person still in front of us?

        `skills._locate_person` is the person detector defined in SPEC section 7; it is
        looked up with getattr so this module works against a Skills build that does not
        have it yet, in which case there is nothing to check and we keep going.
        """
        locate = getattr(self.skills, "_locate_person", None)
        if locate is None:
            return True
        try:
            return await locate() is not None
        except Exception:
            # A camera or provider failure is not evidence that the person is still
            # beside us, and wheels stopped is the only state defensible without eyes.
            return False

    # -- hardware, all of it best-effort ------------------------------------
    def _skill_running(self) -> bool:
        task = getattr(self.skills, "_current", None)
        return task is not None and not task.done()

    async def _stop_wheels(self) -> None:
        try:
            await self.body.stop()
        except Exception as exc:
            self.last_error = repr(exc)
            self._stop_wheels_now()

    def _stop_wheels_now(self) -> None:
        """Stop without awaiting anything: one short write on the calling thread.

        Used on the cancellation path. Even if this fails, every command this mode
        issues lasts only WALK_STEP_MS, so the host watchdog and then the firmware
        failsafe stop the wheels on their own shortly after.
        """
        try:
            self.body.drive_base.stop()
        except Exception as exc:
            self.last_error = repr(exc)

    async def _park_arm(self, pose: str) -> None:
        # A missing or relaxed arm is a drive-only escort, not a failure: the person
        # still gets a robot keeping pace with them.
        try:
            await self.body.pose(pose, ms=900)
        except Exception as exc:
            self.last_error = repr(exc)

    # -- events -------------------------------------------------------------
    def _emit(self, phase: str, human_text: str | None = None) -> None:
        # The loop re-enters WALKING every step. Emitting each time would have the robot
        # announce "walking with you" four times a second, the same way the servo loop
        # would announce "lining up" -- see Skills._set.
        if phase == self._phase:
            return
        self._phase = phase
        self.bus.emit(
            {
                "type": "phase",
                "task_id": self._session_id,
                "order_id": None,
                "label": "",
                "phase": phase,
                "human_text": human_text or phrase(phase),
                "progress": progress(phase),
                "ok": True,
            }
        )
