"""The skill state machine.

The load-bearing idea: the base is the positioner and the arm is a stamp. We never
estimate the object's 3D pose. We drive until it appears at one calibrated spot in the
image, then replay one canned grasp that is known to work from exactly there.

Everything is async and only one skill runs at a time. Skills report progress on the
event bus rather than returning it, because the customer is watching a web page and
the order queue is waiting on the same run.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field

from robot.config import (
    DELIVER_SWEET_SPOT_H,
    HOME_TAG_ID,
    HOME_TAG_LABEL,
    K_ANGULAR,
    K_LINEAR,
    MAX_SEARCH_STEPS,
    MAX_SERVO_STEPS,
    REACH_BUCKETS,
    SEARCH_STEP_RAD,
    STEP_MS,
    SWEET_SPOT_H,
    SWEET_SPOT_X,
    TOL_H,
    TOL_X,
)
from robot.events import EventBus, phrase, progress
from robot.tags import find_tag
from robot.vision import Detection


def reach_for(bbox_height: float) -> str:
    for threshold, bucket in REACH_BUCKETS:
        if bbox_height >= threshold:
            return bucket
    return "mid"


@dataclass
class Task:
    id: str
    label: str = ""
    order_id: str | None = None
    phase: str = "PENDING"
    done: bool = False
    ok: bool = False
    detail: str = ""
    _finished: asyncio.Event = field(default_factory=asyncio.Event)

    async def wait(self) -> bool:
        await self._finished.wait()
        return self.ok

    def as_dict(self) -> dict:
        return {
            "task_id": self.id,
            "label": self.label,
            "order_id": self.order_id,
            "phase": self.phase,
            "done": self.done,
            "ok": self.ok,
            "detail": self.detail,
        }


class Skills:
    def __init__(self, body, vision, bus: EventBus) -> None:
        self.body = body
        self.vision = vision
        self.bus = bus
        self.carrying: str | None = None
        self.phase = "IDLE"
        self.last_error = ""
        self.tasks: dict[str, Task] = {}
        # Base commands issued while approaching, so we can retrace to put things back.
        self._breadcrumbs: list[tuple[float, float, int]] = []
        self._busy = asyncio.Lock()
        self._current: asyncio.Task | None = None

    # -- plumbing ---------------------------------------------------------
    def _set(self, task: Task, phase: str, **fmt) -> None:
        # The servo loop re-enters ALIGNING on every step. Emitting each time would make
        # the robot say "lining up" a dozen times in a row, which sounds broken.
        if task.phase == phase:
            return
        self.phase = task.phase = phase
        self.bus.emit(
            {
                "type": "phase",
                "task_id": task.id,
                "order_id": task.order_id,
                "label": task.label,
                "phase": phase,
                "human_text": phrase(phase, **{"label": task.label or "it", **fmt}),
                "progress": progress(phase),
                "ok": phase not in ("FAILED", "NEEDS_HELP"),
            }
        )

    def start(self, coro_factory, label: str = "", order_id: str | None = None) -> Task:
        task = Task(id=f"t_{uuid.uuid4().hex[:6]}", label=label, order_id=order_id)
        self.tasks[task.id] = task
        self._current = asyncio.create_task(self._run(task, coro_factory))
        return task

    async def _run(self, task: Task, coro_factory) -> None:
        async with self._busy:
            try:
                task.ok = await coro_factory(task)
            except asyncio.CancelledError:
                task.ok = False
                task.detail = "cancelled"
                self._set(task, "CANCELLED")
                raise
            except Exception as exc:  # a crashed skill must never leave the base driving
                self.last_error = repr(exc)
                task.ok = False
                task.detail = repr(exc)
                self._set(task, "FAILED")
            finally:
                try:
                    await self.body.stop()
                except Exception:
                    pass
                task.done = True
                task._finished.set()
                self.phase = "IDLE"

    async def cancel_current(self) -> None:
        if self._current and not self._current.done():
            self._current.cancel()
            try:
                await self._current
            except (asyncio.CancelledError, Exception):
                pass

    async def _drive(self, linear: float, angular: float, ms: int, record: bool) -> None:
        await self.body.drive(linear=linear, angular=angular, ms=ms)
        if record:
            self._breadcrumbs.append((linear, angular, ms))
        await asyncio.sleep(ms / 1000.0)

    # -- perception -------------------------------------------------------
    async def scene(self) -> list[Detection]:
        return await self.vision.scene(await self.body.frame())

    async def locate(self, query: str) -> Detection | None:
        return await self.vision.locate(await self.body.frame(), query)

    async def _locate_home(self) -> Detection | None:
        frame = await self.body.frame()
        bbox = find_tag(frame, HOME_TAG_ID)
        if bbox:
            return Detection(HOME_TAG_LABEL, bbox, 1.0)
        return await self.vision.locate(frame, HOME_TAG_LABEL)

    # -- the two motion primitives ----------------------------------------
    async def _search(self, task: Task, locator, label: str) -> Detection | None:
        """Rotate in place, looking between steps, until the target appears."""
        self._set(task, "SEARCHING", label=label)
        await self.body.pose("SCAN", ms=700)
        for _ in range(MAX_SEARCH_STEPS):
            det = await locator()
            if det:
                return det
            await self._drive(0.0, SEARCH_STEP_RAD / (STEP_MS / 1000.0), STEP_MS, record=True)
        return None

    async def _servo_to(
        self, task: Task, locator, label: str, target_h: float
    ) -> Detection | None:
        """Move-then-look. A VLM round-trip is far too slow for a continuous loop, so we
        take one short step per observation. It converges in a handful of steps and it
        reads as the robot checking its work rather than lagging."""
        self._set(task, "APPROACHING", label=label)
        last_err_x = 0.0
        for _ in range(MAX_SERVO_STEPS):
            det = await locator()
            if det is None:
                # Lost it mid-approach. Sweep back toward whichever side it was last on
                # rather than always turning the same way, which walks away from it.
                direction = -1.0 if last_err_x > 0 else 1.0
                await self._drive(
                    0.0, direction * SEARCH_STEP_RAD / 2 / (STEP_MS / 1000.0), STEP_MS, True
                )
                continue

            err_x = det.cx - SWEET_SPOT_X
            err_h = target_h - det.height
            last_err_x = err_x
            if abs(err_x) <= TOL_X and abs(err_h) <= TOL_H:
                await self.body.stop()
                return det

            if abs(err_x) > TOL_X:
                self._set(task, "ALIGNING", label=label)
                await self._drive(0.0, -K_ANGULAR * err_x, STEP_MS, record=True)
            else:
                await self._drive(K_LINEAR * err_h, 0.0, STEP_MS, record=True)

        # Never converged. Grasping from here would be a blind swipe at a stale
        # detection, which on hardware means the arm hits the table.
        return None

    # -- skills -----------------------------------------------------------
    # Each returns a Task immediately; callers await task.wait() or watch the bus.
    def fetch(self, label: str, order_id: str | None = None) -> Task:
        async def run(task: Task) -> bool:
            self._breadcrumbs.clear()

            async def locator():
                return await self.locate(label)

            det = await self._search(task, locator, label)
            if det is None:
                self._set(task, "FAILED", label=label)
                return False

            det = await self._servo_to(task, locator, label, SWEET_SPOT_H)
            if det is None:
                self._set(task, "FAILED", label=label)
                return False

            reach = reach_for(det.height)
            for attempt in ("first", "retry"):
                self._set(task, "GRASPING", label=label)
                await self.body.pose("PRE_GRASP", ms=900, reach=reach)
                await self.body.pose("DESCEND", ms=700, reach=reach)
                got_it = await self.body.gripper("close")
                self._set(task, "VERIFYING", label=label)
                await self.body.pose("LIFT", ms=700)
                if got_it:
                    self.carrying = label
                    await self.body.pose("CARRY", ms=700)
                    self._set(task, "DONE", label=label)
                    return True
                await self.body.gripper("open")
                if attempt == "first":
                    # one silent retry a little further forward before asking for help
                    reach = "far" if reach != "far" else "mid"
                    await self._drive(0.06, 0.0, 300, record=True)

            self._set(task, "NEEDS_HELP", label=label)
            return False

        return self.start(run, label=label, order_id=order_id)

    def deliver(self, order_id: str | None = None) -> Task:
        label = self.carrying or "it"

        async def run(task: Task) -> bool:
            self._set(task, "RETURNING")
            det = await self._search(task, self._locate_home, "way back")
            if det is None:
                self._set(task, "FAILED", label="my way back")
                return False
            await self._servo_to(task, self._locate_home, "way back", DELIVER_SWEET_SPOT_H)
            self._set(task, "PRESENTING")
            await self.body.pose("PRESENT", ms=900)
            await self.body.gripper("open")
            self.carrying = None
            await asyncio.sleep(0.8)
            await self.body.pose("HOME", ms=800)
            self._set(task, "DONE")
            return True

        return self.start(run, label=label, order_id=order_id)

    def return_item(self, order_id: str | None = None) -> Task:
        """Retrace the approach and put the object back roughly where it came from."""
        label = self.carrying or "it"

        async def run(task: Task) -> bool:
            self._set(task, "REPLACING", label=label)
            for linear, angular, ms in reversed(self._breadcrumbs):
                await self._drive(-linear, -angular, ms, record=False)
            await self.body.pose("PRE_GRASP", ms=800)
            await self.body.pose("DESCEND", ms=700)
            await self.body.gripper("open")
            await self.body.pose("SCAN", ms=700)
            self.carrying = None
            self._breadcrumbs.clear()
            self._set(task, "DONE", label=label)
            return True

        return self.start(run, label=label, order_id=order_id)

    # -- composite: what an order actually is ------------------------------
    async def fetch_and_deliver(self, label: str, order_id: str | None = None) -> tuple[bool, str]:
        """Go get it, bring it back. Returns (ok, message) for the order queue."""
        fetch_task = self.fetch(label, order_id=order_id)
        if not await fetch_task.wait():
            if fetch_task.phase == "NEEDS_HELP":
                return False, phrase("NEEDS_HELP", label=label)
            return False, fetch_task.detail or phrase("FAILED", label=label)

        deliver_task = self.deliver(order_id=order_id)
        if not await deliver_task.wait():
            return False, deliver_task.detail or "picked it up but couldn't find my way back"
        return True, f"delivered the {label}"

    def state(self) -> dict:
        return {
            "phase": self.phase,
            "carrying": self.carrying,
            "last_error": self.last_error,
            "vision_backend": getattr(self.vision, "name", "unknown"),
        }
