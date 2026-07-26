"""Wireless order API, customer UI and ops console for the Corgi grocery robot.

One process on the Mac holds everything: the serial link to the drive base, the camera,
the arm, perception, the skill state machine and the web server. Split it up later if
it ever needs to be; today the whole robot is `python -m robot.server`.
"""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from robot.body import make_body
from robot.config import HOST, MOCK, PORT, STEP_MS, VISION_BACKEND
from robot.drive import list_serial_ports
from robot.events import EventBus
from robot.orders import Order, OrderService, stub_fulfill
from robot.skills import Skills
from robot.vision import make_vision

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"

bus = EventBus()
body, BOOT_NOTES = make_body(MOCK)
vision = make_vision() if body.has_camera else None
skills = Skills(body=body, vision=vision, bus=bus) if vision else None
orders = OrderService(bus=bus)


async def fulfill(order: Order) -> tuple[bool, str]:
    assert skills is not None
    return await skills.fetch_and_deliver(order.item, order_id=order.id)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if skills is not None:
        orders.set_fulfill(fulfill)
    else:
        BOOT_NOTES.append("no perception: orders are simulated until a camera is attached")
        orders.set_fulfill(stub_fulfill)
    orders.start()

    print(f"[corgi] mock={MOCK} vision={VISION_BACKEND} arm={body.arm.present}")
    for note in BOOT_NOTES:
        print(f"[corgi] {note}")

    yield

    await orders.stop()
    if skills is not None:
        await skills.cancel_current()
    with contextlib.suppress(Exception):
        await body.estop()
    if vision is not None:
        with contextlib.suppress(Exception):
            await vision.aclose()
    body.close()


app = FastAPI(title="Corgi Grocery Robot", lifespan=lifespan)


# --------------------------------------------------------------------------
# schemas
# --------------------------------------------------------------------------
class OrderIn(BaseModel):
    item: str = Field(min_length=1, max_length=80)


class OrderOut(BaseModel):
    id: str
    item: str
    status: str
    message: str
    phase: str
    progress: float


class DriveIn(BaseModel):
    left_us: int = Field(ge=1000, le=2000)
    right_us: int = Field(ge=1000, le=2000)
    ms: int | None = Field(default=None, ge=1, le=5000)


class VelocityIn(BaseModel):
    linear: float = 0.0
    angular: float = 0.0
    ms: int = Field(default=STEP_MS, ge=1, le=5000)


class PoseIn(BaseModel):
    name: str
    ms: int = Field(default=800, ge=1, le=10000)
    reach: str = "mid"


class GripperIn(BaseModel):
    state: str


class LocateIn(BaseModel):
    query: str = Field(min_length=1, max_length=80)


class FetchIn(BaseModel):
    label: str = Field(min_length=1, max_length=80)


def _order_out(order: Order) -> OrderOut:
    return OrderOut(
        id=order.id,
        item=order.item,
        status=order.status.value,
        message=order.message,
        phase=order.phase,
        progress=round(order.progress, 3),
    )


def _require_skills() -> Skills:
    if skills is None:
        raise HTTPException(503, "perception unavailable: no camera attached")
    return skills


def _require_drive():
    if not body.drive_base.connected:
        raise HTTPException(503, "drive not connected")
    return body.drive_base


# --------------------------------------------------------------------------
# health and state
# --------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "ok": True,
        "mock": MOCK,
        "drive_connected": body.drive_base.connected,
        "arm_present": body.arm.present,
        "camera_present": body.has_camera,
        "vision_backend": VISION_BACKEND if vision else None,
        "notes": BOOT_NOTES,
        "serial_ports": [{"device": d, "description": desc} for d, desc in list_serial_ports()],
    }


@app.get("/api/state")
def state():
    current = orders.current
    return {
        "robot": skills.state() if skills else {"phase": "IDLE", "carrying": None},
        "current_order": current.as_dict() if current else None,
        "queued": sum(1 for o in orders.list() if o.status.value == "queued"),
    }


# --------------------------------------------------------------------------
# orders
# --------------------------------------------------------------------------
@app.post("/api/orders", response_model=OrderOut)
def create_order(payload: OrderIn):
    try:
        return _order_out(orders.create(payload.item))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/orders", response_model=list[OrderOut])
def list_orders():
    return [_order_out(o) for o in orders.list()]


@app.get("/api/orders/{order_id}", response_model=OrderOut)
def get_order(order_id: str):
    order = orders.get(order_id)
    if not order:
        raise HTTPException(404, "order not found")
    return _order_out(order)


@app.post("/api/orders/{order_id}/cancel", response_model=OrderOut)
def cancel_order(order_id: str):
    if orders.get(order_id) is None:
        raise HTTPException(404, "order not found")
    order = orders.cancel(order_id)
    if order is None:
        raise HTTPException(409, "order already started")
    return _order_out(order)


# --------------------------------------------------------------------------
# camera
# --------------------------------------------------------------------------
@app.get("/api/camera/frame.jpg")
async def camera_frame():
    if not body.has_camera:
        raise HTTPException(503, "no camera attached")
    return Response(await body.jpeg(), media_type="image/jpeg")


@app.get("/api/camera/stream.mjpg")
def camera_stream():
    if not body.has_camera:
        raise HTTPException(503, "no camera attached")
    assert body.camera is not None
    return StreamingResponse(
        body.camera.mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


# --------------------------------------------------------------------------
# skills
# --------------------------------------------------------------------------
@app.post("/api/skills/scene")
async def skills_scene():
    dets = await _require_skills().scene()
    return {"objects": [d.as_dict() for d in dets]}


@app.post("/api/skills/locate")
async def skills_locate(payload: LocateIn):
    det = await _require_skills().locate(payload.query)
    if det is None:
        return {"found": False, "bbox": None, "est_distance_m": None}
    return {
        "found": True,
        "bbox": list(det.bbox),
        # Crude but monotonic: bbox height is the only depth cue a single camera has.
        "est_distance_m": round(0.09 / max(det.height, 1e-3), 2),
    }


@app.post("/api/skills/fetch")
async def skills_fetch(payload: FetchIn):
    return _require_skills().fetch(payload.label).as_dict()


@app.post("/api/skills/deliver")
async def skills_deliver():
    return _require_skills().deliver().as_dict()


@app.post("/api/skills/return_item")
async def skills_return_item():
    return _require_skills().return_item().as_dict()


@app.get("/api/tasks/{task_id}")
async def task_status(task_id: str):
    task = _require_skills().tasks.get(task_id)
    if task is None:
        raise HTTPException(404, "no such task")
    return task.as_dict()


# --------------------------------------------------------------------------
# direct hardware control (manual driving, arm calibration, e-stop)
# --------------------------------------------------------------------------
@app.post("/api/drive/stop")
async def drive_stop():
    return {"reply": await asyncio.to_thread(_require_drive().stop)}


@app.post("/api/drive/cmd")
async def drive_cmd(payload: DriveIn):
    _require_drive()
    return {"reply": await body.drive_us(payload.left_us, payload.right_us, payload.ms)}


@app.post("/api/drive/velocity")
async def drive_velocity(payload: VelocityIn):
    _require_drive()
    await body.drive(linear=payload.linear, angular=payload.angular, ms=payload.ms)
    return {"ok": True}


@app.post("/api/arm/pose")
async def arm_pose(payload: PoseIn):
    try:
        positions = await body.pose(payload.name, payload.ms, payload.reach)
    except KeyError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, "positions": positions}


@app.post("/api/arm/gripper")
async def arm_gripper(payload: GripperIn):
    if payload.state not in ("open", "close"):
        raise HTTPException(400, "state must be 'open' or 'close'")
    return {"ok": True, "closed_on_object": await body.gripper(payload.state)}


@app.get("/api/arm/state")
def arm_state():
    return {
        "present": body.arm.present,
        "positions": body.arm.positions,
        "moving": body.arm.moving,
        "gripper_closed_on_object": body.arm.holding,
    }


@app.post("/api/estop")
async def estop():
    if skills is not None:
        await skills.cancel_current()
    await body.estop()
    bus.emit({"type": "estop", "phase": "CANCELLED", "human_text": "stopped", "ok": False})
    return {"ok": True}


# --------------------------------------------------------------------------
# events
# --------------------------------------------------------------------------
@app.get("/api/events/recent")
def recent_events(limit: int = 40):
    return {"events": bus.recent(limit)}


@app.websocket("/api/events")
async def events(ws: WebSocket):
    await ws.accept()
    queue = bus.subscribe()
    try:
        for event in bus.recent(20):
            await ws.send_json(event)
        while True:
            await ws.send_json(await queue.get())
    except (WebSocketDisconnect, asyncio.CancelledError, RuntimeError):
        pass
    finally:
        bus.unsubscribe(queue)


# --------------------------------------------------------------------------
# simulator ground truth
# --------------------------------------------------------------------------
@app.get("/api/debug/world")
def debug_world():
    """When the servo loop converges somewhere wrong, this says immediately whether
    vision or control is the one lying to you."""
    if not MOCK:
        raise HTTPException(400, "only available with CORGI_MOCK=1")
    from robot.world import WORLD

    return WORLD.snapshot()


@app.post("/api/debug/reset")
def debug_reset(grasp_failure_rate: float | None = None):
    if not MOCK:
        raise HTTPException(400, "only available with CORGI_MOCK=1")
    from robot.world import WORLD

    WORLD.reset()
    if grasp_failure_rate is not None:
        WORLD.grasp_failure_rate = max(0.0, min(1.0, grasp_failure_rate))
    if skills is not None:
        skills.carrying = None
    return {"ok": True, "grasp_failure_rate": WORLD.grasp_failure_rate}


# --------------------------------------------------------------------------
# pages
# --------------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/ops")
def ops():
    return FileResponse(WEB_DIR / "ops.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def main() -> None:
    import uvicorn

    uvicorn.run("robot.server:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()
