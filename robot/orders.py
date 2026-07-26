"""Order queue + stub fulfillment (vision/arm wired in later)."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


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
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


FulfillFn = Callable[[Order], None]


class OrderService:
    def __init__(self, fulfill: FulfillFn | None = None) -> None:
        self._orders: dict[str, Order] = {}
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._fulfill = fulfill or self._default_fulfill
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._worker.start()

    def stop(self) -> None:
        with self._cv:
            self._running = False
            self._cv.notify_all()

    def create(self, item: str) -> Order:
        item = item.strip()
        if not item:
            raise ValueError("item is required")
        order = Order(id=uuid.uuid4().hex[:8], item=item)
        with self._cv:
            self._orders[order.id] = order
            self._cv.notify()
        return order

    def get(self, order_id: str) -> Order | None:
        with self._lock:
            return self._orders.get(order_id)

    def list(self) -> list[Order]:
        with self._lock:
            return sorted(self._orders.values(), key=lambda o: o.created_at, reverse=True)

    def _set(self, order: Order, status: OrderStatus, message: str = "") -> None:
        order.status = status
        order.message = message
        order.updated_at = time.time()

    def _loop(self) -> None:
        while True:
            with self._cv:
                while self._running and not any(
                    o.status == OrderStatus.QUEUED for o in self._orders.values()
                ):
                    self._cv.wait(timeout=0.5)
                if not self._running:
                    return
                order = next(
                    (o for o in self._orders.values() if o.status == OrderStatus.QUEUED),
                    None,
                )
                if order is None:
                    continue
                self._set(order, OrderStatus.RUNNING, "Robot is working on it")

            try:
                self._fulfill(order)
                with self._lock:
                    self._set(order, OrderStatus.DONE, "Item retrieved (stub)")
            except Exception as exc:  # noqa: BLE001 — surface to UI
                with self._lock:
                    self._set(order, OrderStatus.FAILED, str(exc))

    @staticmethod
    def _default_fulfill(order: Order) -> None:
        # Placeholder until vision + arm + drive path are connected.
        time.sleep(2.0)
