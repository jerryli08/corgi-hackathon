"""Subprocess adapter for the two-camera pill-bottle mission FSM."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from robot.config import (
    SINGLECAM_MISSION_ITEM_KEYWORDS,
    SINGLECAM_MISSION_PYTHON,
    SINGLECAM_MISSION_SCRIPT,
    SINGLECAM_MISSION_TIMEOUT_S,
)

if TYPE_CHECKING:
    from robot.events import EventBus

ROOT = Path(__file__).resolve().parent.parent
EVENT_PREFIX = "CORGI_EVENT "

# Camera-specific states remain visible under /api/state. Existing public phases are
# emitted on the bus so the order UI and Photon milestone messages keep one vocabulary.
PUBLIC_PHASES = {
    "PATH_FOLLOWING": "SEARCHING",
    "TARGET_CENTERING": "ALIGNING",
    "TARGET_LOCKED": "VERIFYING",
    "ARM_PICK_PLACEHOLDER": "GRASPING",
    "WINCH_UP": "STOWING",
    "BASKET_DROP_PLACEHOLDER": "STOWING",
    "WINCH_DOWN": "STOWING",
    "RESUMING_PATH": "RETURNING",
    "COMPLETE": "DONE",
    "FAILED": "FAILED",
}


def should_use_singlecam_mission(item: str) -> bool:
    text = item.strip().lower()
    return any(keyword.lower() in text for keyword in SINGLECAM_MISSION_ITEM_KEYWORDS)


class SinglecamMissionRunner:
    def __init__(self) -> None:
        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None
        self._timeout_task: asyncio.Task | None = None
        self._bus: EventBus | None = None
        self._order_id: str | None = None
        self.item = ""
        self.phase = "IDLE"
        self.human_text = "standing by"
        self.progress = 0.0
        self.last_error = ""
        self.last_exit_code: int | None = None

    @property
    def active(self) -> bool:
        return self._process is not None and self._process.returncode is None

    def state(self) -> dict:
        script = (ROOT / SINGLECAM_MISSION_SCRIPT).resolve()
        return {
            "active": self.active,
            "pid": self._process.pid if self._process is not None else None,
            "item": self.item,
            "phase": self.phase,
            "human_text": self.human_text,
            "progress": self.progress,
            "last_error": self.last_error,
            "last_exit_code": self.last_exit_code,
            "payload_enabled": os.getenv("CORGI_SINGLECAM_PAYLOAD_ENABLED", "").lower()
            in ("1", "true", "yes", "on"),
            "script": str(script),
            "python": SINGLECAM_MISSION_PYTHON or sys.executable,
        }

    async def start(
        self,
        item: str,
        *,
        order_id: str | None = None,
        bus: EventBus | None = None,
    ) -> tuple[bool, str]:
        """Launch without waiting; used by the manual API endpoint."""
        return await self._launch(item, order_id=order_id, bus=bus)

    async def run(
        self,
        item: str,
        *,
        order_id: str,
        bus: EventBus,
    ) -> tuple[bool, str]:
        """Launch and wait for a real FSM terminal state; used by OrderService."""
        ok, message = await self._launch(item, order_id=order_id, bus=bus)
        if not ok:
            return False, message

        assert self._process is not None
        return_code = await self._process.wait()
        if self._reader_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
        self.last_exit_code = return_code

        if return_code == 0 and self.phase == "COMPLETE":
            return True, f"camera mission for {item} completed"
        if not self.last_error:
            self.last_error = (
                f"camera mission exited with code {return_code} during {self.phase}"
            )
        return False, self.last_error

    async def _launch(
        self,
        item: str,
        *,
        order_id: str | None,
        bus: EventBus | None,
    ) -> tuple[bool, str]:
        if self.active:
            return False, "singlecam mission is already running"

        script = (ROOT / SINGLECAM_MISSION_SCRIPT).resolve()
        if not script.exists():
            self.last_error = f"missing mission script: {script}"
            return False, self.last_error

        self.item = item
        self.phase = "STARTING"
        self.human_text = f"starting camera mission for {item}"
        self.progress = 0.02
        self.last_error = ""
        self.last_exit_code = None
        self._bus = bus
        self._order_id = order_id

        python = SINGLECAM_MISSION_PYTHON or sys.executable
        env = os.environ.copy()
        env["CORGI_SINGLECAM_ITEM"] = item
        try:
            self._process = await asyncio.create_subprocess_exec(
                python,
                str(script),
                "--mission",
                item,
                cwd=str(script.parent),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )
        except Exception as exc:
            self.last_error = repr(exc)
            self._process = None
            return False, f"could not start singlecam mission: {exc}"

        self._reader_task = asyncio.create_task(self._read_output())
        if SINGLECAM_MISSION_TIMEOUT_S > 0:
            self._timeout_task = asyncio.create_task(
                self._stop_after_timeout(SINGLECAM_MISSION_TIMEOUT_S)
            )
        return True, f"started camera mission for {item}"

    async def _read_output(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        while line_bytes := await process.stdout.readline():
            line = line_bytes.decode("utf-8", errors="replace").rstrip()
            if not line.startswith(EVENT_PREFIX):
                print(f"[singlecam] {line}")
                continue
            try:
                event = json.loads(line[len(EVENT_PREFIX) :])
                self._accept_event(event)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                self.last_error = f"invalid singlecam event: {exc}"

    def _accept_event(self, event: dict) -> None:
        phase = str(event.get("phase") or "")
        if not phase:
            return
        self.phase = phase
        self.human_text = str(event.get("human_text") or phase.lower())
        self.progress = max(self.progress, float(event.get("progress") or 0.0))
        if phase == "FAILED":
            self.last_error = self.human_text

        if self._bus is not None:
            self._bus.emit(
                {
                    "type": "phase",
                    "order_id": self._order_id,
                    "label": self.item,
                    "phase": PUBLIC_PHASES.get(phase, phase),
                    "mission_phase": phase,
                    "human_text": self.human_text,
                    "progress": self.progress,
                    "ok": phase != "FAILED",
                }
            )

    async def stop(self) -> None:
        process = self._process
        if process is None or process.returncode is not None:
            return
        try:
            if os.name == "nt":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.send_signal(signal.SIGINT)
        except (AttributeError, ProcessLookupError, OSError):
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=3.0)
        except TimeoutError:
            process.kill()
            await process.wait()
        finally:
            self.last_exit_code = process.returncode
            self.phase = "CANCELLED"
            self.human_text = "camera mission stopped"
            if (
                self._timeout_task is not None
                and self._timeout_task is not asyncio.current_task()
            ):
                self._timeout_task.cancel()

    async def _stop_after_timeout(self, seconds: float) -> None:
        await asyncio.sleep(seconds)
        if self.active:
            self.last_error = f"camera mission timed out after {seconds:g} seconds"
            await self.stop()


singlecam_mission = SinglecamMissionRunner()
