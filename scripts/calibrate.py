#!/usr/bin/env python3
"""Sweet-spot calibration -- the highest-value ten minutes of the build.

Place an object where the canned grasp succeeds. Run this. Write the printed cx and
height into CORGI_SWEET_SPOT_X and CORGI_SWEET_SPOT_H, then write them on tape and
stick it on the robot.

    python scripts/calibrate.py strawberries
"""

from __future__ import annotations

import os
import sys
import time

import cv2
import httpx
import numpy as np

BASE = os.getenv("CORGI_BASE_URL", "http://localhost:8000")


def main(query: str) -> int:
    print(f"watching for {query!r}. move the object until the grasp works, then note these.")
    print("ctrl-c to stop.\n")

    with httpx.Client() as client:
        while True:
            try:
                raw = client.get(f"{BASE}/api/camera/frame.jpg", timeout=5).content
                frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                found = client.post(
                    f"{BASE}/api/skills/locate", json={"query": query}, timeout=15
                ).json()
            except Exception as exc:
                print(f"  ...{exc}")
                time.sleep(1)
                continue

            if not found.get("found"):
                print("  not visible")
            else:
                x0, y0, x1, y1 = found["bbox"]
                cx, height = (x0 + x1) / 2, y1 - y0
                print(
                    f"  cx={cx:.3f}  height={height:.3f}  "
                    f"est_dist={found.get('est_distance_m')}m"
                )
                h, w = frame.shape[:2]
                cv2.rectangle(
                    frame, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), (0, 255, 0), 2
                )
                cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 200, 255), 1)

            try:
                cv2.imshow("calibrate", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return 0
            except cv2.error:
                pass  # headless: the printed numbers are the point anyway
            time.sleep(0.4)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "strawberries"))
    except KeyboardInterrupt:
        sys.exit(0)
