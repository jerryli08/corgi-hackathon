"""The order queue: one customer request at a time, front to back.

Orders are the customer's view of the robot and phases are the robot's view of itself.
This module owns the first and subscribes to the second, so the web page can show
"looking for the granola bar" without knowing that a skill state machine exists.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum

from robot.events import EventBus, phrase, progress


class OrderStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Order:
    id: str
    item: str
    status: OrderStatus = OrderStatus.QUEUED
    message: str = ""
    phase: str = "QUEUED"
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "item": self.item,
            "status": self.status.value,
            "message": self.message,
            "phase": self.phase,
            "progress": round(self.progress, 3),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# Returns (ok, message). Raising is allowed too; the queue turns it into a failure.
FulfillFn = Callable[[Order], Awaitable[tuple[bool, str]]]


async def stub_fulfill(order: Order) -> tuple[bool, str]:
    """Used when there is no camera, so the UI is still demoable on a bare laptop."""
    await asyncio.sleep(2.0)
    return True, f"pretended to fetch the {order.item} (no camera attached)"


class OrderService:
    def __init__(self, fulfill: FulfillFn | None = None, bus: EventBus | None = None) -> None:
        self._orders: dict[str, Order] = {}
        self._fulfill = fulfill or stub_fulfill
        self._bus = bus
        self._wake = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        self._running = False

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._tasks.append(asyncio.create_task(self._loop()))
        if self._bus is not None:
            self._tasks.append(asyncio.create_task(self._follow_phases()))

    async def stop(self) -> None:
        self._running = False
        self._wake.set()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks.clear()

    def set_fulfill(self, fulfill: FulfillFn) -> None:
        self._fulfill = fulfill

    # -- queue ------------------------------------------------------------
    def create(self, item: str) -> Order:
        item = item.strip()
        if not item:
            raise ValueError("item is required")
        order = Order(id=uuid.uuid4().hex[:8], item=item)
        order.message = phrase("QUEUED")
        order.progress = progress("QUEUED")
        self._orders[order.id] = order
        self._wake.set()
        return order

    def get(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)

    def list(self) -> list[Order]:
        return sorted(self._orders.values(), key=lambda o: o.created_at, reverse=True)

    def cancel(self, order_id: str) -> Order | None:
        order = self._orders.get(order_id)
        if order is None or order.status is not OrderStatus.QUEUED:
            return None
        self._update(order, OrderStatus.CANCELLED, "CANCELLED", "cancelled before it started")
        return order

    @property
    def current(self) -> Order | None:
        return next((o for o in self._orders.values() if o.status is OrderStatus.RUNNING), None)

    # -- internals --------------------------------------------------------
    def _update(
        self,
        order: Order,
        status: OrderStatus | None = None,
        phase: str | None = None,
        message: str | None = None,
    ) -> None:
        if status is not None:
            order.status = status
        if phase is not None:
            order.phase = phase
            order.progress = max(order.progress, progress(phase))
        if message is not None:
            order.message = message
        order.updated_at = time.time()

    def _next_queued(self) -> Order | None:
        queued = [o for o in self._orders.values() if o.status is OrderStatus.QUEUED]
        return min(queued, key=lambda o: o.created_at, default=None)

    async def _loop(self) -> None:
        while self._running:
            order = self._next_queued()
            if order is None:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    pass
                continue

            self._update(order, OrderStatus.RUNNING, "QUEUED", "robot is on it")
            try:
                ok, message = await self._fulfill(order)
            except asyncio.CancelledError:
                self._update(order, OrderStatus.CANCELLED, "CANCELLED", "server shutting down")
                raise
            except Exception as exc:  # noqa: BLE001 -- the customer should see why
                self._update(order, OrderStatus.FAILED, "FAILED", str(exc))
                continue
            # NEEDS_HELP survives: the order failed, but the message is an actionable
            # request rather than a dead end, and the UI shows it differently.
            if ok:
                phase = "DONE"
            elif order.phase == "NEEDS_HELP":
                phase = "NEEDS_HELP"
            else:
                phase = "FAILED"
            self._update(order, OrderStatus.DONE if ok else OrderStatus.FAILED, phase, message)

    async def _follow_phases(self) -> None:
        """Mirror robot phases onto whichever order is running, so the customer sees
        the same story the ops console does."""
        assert self._bus is not None
        queue = self._bus.subscribe()
        try:
            while True:
                event = await queue.get()
                if event.get("type") != "phase":
                    continue
                order = self.get(event["order_id"]) if event.get("order_id") else self.current
                if order is None or order.status is not OrderStatus.RUNNING:
                    continue
                # DONE/FAILED belong to a single skill, not the whole order -- the queue
                # decides when the order itself is finished.
                if event["phase"] in ("DONE", "FAILED"):
                    continue
                self._update(order, phase=event["phase"], message=event.get("human_text", ""))
        finally:
            self._bus.unsubscribe(queue)
