"""Phase events.

`human_text` is what the robot tells the customer, so edit it like dialogue, not like
log lines. Short, competent, no exclamation marks. It is pre-written rather than
generated so the narration stays in sync with the robot's actual motion.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

PHASE_TEXT: dict[str, str] = {
    "QUEUED": "got it, you're in the queue",
    "SEARCHING": "looking for the {label}",
    "APPROACHING": "found it, driving over",
    "ALIGNING": "lining up",
    "GRASPING": "picking it up",
    "VERIFYING": "checking my grip",
    "NEEDS_HELP": "I can't quite get a grip on the {label} — could you nudge it toward me?",
    "RETURNING": "heading back",
    "PRESENTING": "here you go",
    "REPLACING": "putting the {label} back",
    "DONE": "done",
    "FAILED": "I couldn't find the {label}",
    "CANCELLED": "cancelled",
}

# Rough progress for the customer-facing bar. Not a real estimate, and nobody expects
# one -- it just has to move in the right direction and never go backwards.
PHASE_PROGRESS: dict[str, float] = {
    "QUEUED": 0.02,
    "SEARCHING": 0.15,
    "APPROACHING": 0.35,
    "ALIGNING": 0.5,
    "GRASPING": 0.62,
    "VERIFYING": 0.7,
    "NEEDS_HELP": 0.7,
    "RETURNING": 0.85,
    "PRESENTING": 0.95,
    "REPLACING": 0.9,
    "DONE": 1.0,
    "FAILED": 1.0,
    "CANCELLED": 1.0,
}


def phrase(phase: str, **kwargs) -> str:
    return PHASE_TEXT.get(phase, phase.lower()).format(**{"label": "it", **kwargs})


def progress(phase: str) -> float:
    return PHASE_PROGRESS.get(phase, 0.0)


@dataclass
class EventBus:
    history_limit: int = 200
    _subscribers: set[asyncio.Queue] = field(default_factory=set)
    _history: list[dict] = field(default_factory=list)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def emit(self, event: dict) -> None:
        event.setdefault("at", time.time())
        self._history.append(event)
        del self._history[: max(0, len(self._history) - self.history_limit)]
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # A slow consumer must never stall the robot.
                pass

    def recent(self, limit: int = 40) -> list[dict]:
        return self._history[-limit:]
