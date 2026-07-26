#!/usr/bin/env python3
"""Record the text-sim demo for the README.

Opens a fake Messages thread beside the robot's eye view, sends one simulated
text, and writes docs/assets/text-sim-demo.mp4 while the mock robot fetches
and delivers.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import httpx
import numpy as np

BASE = os.getenv("CORGI_BASE_URL", "http://localhost:8010")
OUT = Path(__file__).resolve().parent.parent / "docs" / "assets" / "text-sim-demo.mp4"
FPS = 8
W, H = 960, 540
PHONE_W = 360
TEXT = "bring me my water bottle"


def _blank(color=(18, 16, 14)) -> np.ndarray:
    return np.full((H, W, 3), color, dtype=np.uint8)


def _put(img, text, org, scale=0.55, color=(230, 230, 230), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _wrap(text: str, width: int = 28) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if len(trial) <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines or [""]


def draw_phone(frame: np.ndarray, messages: list[tuple[str, str]], phase: str) -> None:
    panel = frame[:, :PHONE_W]
    panel[:] = (32, 28, 24)
    cv2.rectangle(panel, (12, 12), (PHONE_W - 12, H - 12), (70, 62, 54), 1)
    _put(panel, "Messages", (28, 48), 0.7, (245, 240, 232), 2)
    _put(panel, "CORGI", (28, 78), 0.5, (140, 190, 170), 1)
    y = 110
    for who, text in messages[-8:]:
        lines = _wrap(text, 26)
        bubble_h = 18 + 22 * len(lines)
        if who == "you":
            x0 = 70
            color = (70, 120, 95)
            tcolor = (245, 250, 245)
        else:
            x0 = 24
            color = (55, 50, 46)
            tcolor = (230, 225, 218)
        cv2.rectangle(panel, (x0, y), (PHONE_W - 24, y + bubble_h), color, -1)
        for i, line in enumerate(lines):
            _put(panel, line, (x0 + 12, y + 22 + i * 22), 0.48, tcolor, 1)
        y += bubble_h + 12
        if y > H - 80:
            break
    if phase:
        _put(panel, phase[:34], (28, H - 28), 0.45, (160, 150, 140), 1)


def draw_cam(frame: np.ndarray, jpeg: bytes | None, status: str) -> None:
    panel = frame[:, PHONE_W:]
    panel[:] = (12, 12, 14)
    _put(panel, "simulation backend", (20, 36), 0.55, (120, 180, 160), 1)
    _put(panel, status[:48], (20, 62), 0.5, (210, 210, 210), 1)
    if jpeg:
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        cam = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if cam is not None:
            avail_w = W - PHONE_W - 40
            avail_h = H - 100
            scale = min(avail_w / cam.shape[1], avail_h / cam.shape[0])
            nw, nh = int(cam.shape[1] * scale), int(cam.shape[0] * scale)
            cam = cv2.resize(cam, (nw, nh))
            x0 = PHONE_W + 20 + (avail_w - nw) // 2
            y0 = 80 + (avail_h - nh) // 2
            frame[y0 : y0 + nh, x0 : x0 + nw] = cam


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT), fourcc, FPS, (W, H))
    if not writer.isOpened():
        raise SystemExit(f"could not open writer for {OUT}")

    messages: list[tuple[str, str]] = []
    frames: list[np.ndarray] = []

    with httpx.Client(timeout=20.0) as client:
        client.post(f"{BASE}/api/debug/reset", params={"grasp_failure_rate": 0.0})

        # Title beat
        for _ in range(FPS * 2):
            frame = _blank()
            draw_phone(frame, messages, "standing by")
            draw_cam(frame, None, "waiting for a text")
            frames.append(frame)

        # User types / sends
        messages.append(("you", TEXT))
        for _ in range(FPS):
            frame = _blank()
            draw_phone(frame, messages, "sending…")
            draw_cam(frame, None, "inbound text")
            frames.append(frame)

        client.post(f"{BASE}/api/imessage/simulate", json={"text": TEXT}).raise_for_status()
        # Pull early robot reply if any
        time.sleep(0.5)

        def pull_outbound() -> None:
            log = client.get(f"{BASE}/api/imessage/log").json()
            for sent in log.get("outbound", []):
                text = sent.get("text") or ""
                if text and text not in seen_out:
                    messages.append(("robot", text))
                    seen_out.add(text)

        seen_out: set[str] = set()
        pull_outbound()

        # Follow the run
        deadline = time.monotonic() + 90
        last_phase = ""
        while time.monotonic() < deadline:
            state = client.get(f"{BASE}/api/state").json()
            current = state.get("current_order") or {}
            phase = current.get("phase") or state.get("robot", {}).get("phase") or "IDLE"
            message = current.get("message") or phase
            status = f"{phase}  ·  {message}"

            pull_outbound()

            jpeg = None
            try:
                jpeg = client.get(f"{BASE}/api/camera/frame.jpg").content
            except Exception:
                pass

            frame = _blank()
            draw_phone(frame, messages, message)
            draw_cam(frame, jpeg, status)
            frames.append(frame)
            last_phase = phase

            if current.get("status") in ("done", "failed", "cancelled"):
                # hold the finish
                for _ in range(FPS * 3):
                    frames.append(frame.copy())
                break
            if not current and last_phase in ("IDLE", "DONE"):
                # order may already be finished between polls
                orders = client.get(f"{BASE}/api/orders").json()
                if orders and orders[0].get("status") in ("done", "failed"):
                    for _ in range(FPS * 3):
                        frames.append(frame.copy())
                    break
            time.sleep(1.0 / FPS)

    for frame in frames:
        writer.write(frame)
    writer.release()

    # Remux to H.264 for GitHub / browser playback
    h264 = OUT.with_name("text-sim-demo.h264.mp4")
    import subprocess

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(OUT),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(h264),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    h264.replace(OUT)
    print(f"wrote {OUT} ({len(frames)} frames, {len(frames) / FPS:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
