"""Wireless order API + customer UI for the Corgi grocery robot."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from robot.drive import DriveBase, DriveConfig, list_serial_ports
from robot.orders import OrderService

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"

drive = DriveBase(DriveConfig(port=os.environ.get("CORGI_DRIVE_PORT")))
orders = OrderService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    orders.start()
    # Drive connect is optional at boot — website still works for demos.
    try:
        port = drive.connect()
        print(f"[corgi] drive connected on {port}")
    except Exception as exc:  # noqa: BLE001
        print(f"[corgi] drive not connected yet: {exc}")
    yield
    drive.close()
    orders.stop()


app = FastAPI(title="Corgi Grocery Robot", lifespan=lifespan)


class OrderIn(BaseModel):
    item: str = Field(min_length=1, max_length=80)


class OrderOut(BaseModel):
    id: str
    item: str
    status: str
    message: str


class DriveIn(BaseModel):
    left_us: int = Field(ge=1000, le=2000)
    right_us: int = Field(ge=1000, le=2000)


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "drive_connected": drive.connected,
        "serial_ports": [{"device": d, "description": desc} for d, desc in list_serial_ports()],
    }


@app.post("/api/orders", response_model=OrderOut)
def create_order(body: OrderIn):
    order = orders.create(body.item)
    return OrderOut(
        id=order.id, item=order.item, status=order.status.value, message=order.message
    )


@app.get("/api/orders", response_model=list[OrderOut])
def list_orders():
    return [
        OrderOut(id=o.id, item=o.item, status=o.status.value, message=o.message)
        for o in orders.list()
    ]


@app.get("/api/orders/{order_id}", response_model=OrderOut)
def get_order(order_id: str):
    order = orders.get(order_id)
    if not order:
        raise HTTPException(404, "order not found")
    return OrderOut(
        id=order.id, item=order.item, status=order.status.value, message=order.message
    )


@app.post("/api/drive/stop")
def drive_stop():
    if not drive.connected:
        raise HTTPException(503, "drive not connected")
    return {"reply": drive.stop()}


@app.post("/api/drive/cmd")
def drive_cmd(body: DriveIn):
    if not drive.connected:
        raise HTTPException(503, "drive not connected")
    return {"reply": drive.drive_us(body.left_us, body.right_us)}


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def main() -> None:
    import uvicorn

    host = os.environ.get("CORGI_HOST", "0.0.0.0")
    port = int(os.environ.get("CORGI_PORT", "8000"))
    uvicorn.run("robot.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
