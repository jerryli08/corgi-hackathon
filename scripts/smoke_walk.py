#!/usr/bin/env python3
"""Prove the dead-man, against the running server.

    python scripts/smoke_walk.py

Walker mode moves an open-loop base next to someone unsteady on their feet. The only
thing that makes that acceptable is that the wheels turn ONLY while instructions keep
arriving. This script is the test of that claim: it nudges for a couple of seconds, then
deliberately stops asking, and fails if the robot is still moving afterwards.

Run it against mock (CORGI_MOCK=1). On hardware, stand clear and keep a hand near the
power switch -- the whole point of the run is to command the base and then stop talking.
"""

from __future__ import annotations

import os
import sys
import time

import httpx

BASE = os.getenv("CORGI_BASE_URL", "http://localhost:8000")
NUDGE_FOR_S = 2.0
NUDGE_EVERY_S = 0.2


def pose(client: httpx.Client) -> tuple[float, float]:
    world = client.get(f"{BASE}/api/debug/world", timeout=5).json()
    robot = world["robot"]
    return robot["x"], robot["y"]


def moved(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def main() -> int:
    with httpx.Client() as client:
        try:
            health = client.get(f"{BASE}/api/health", timeout=5).json()
        except Exception as exc:
            print(f"server not reachable at {BASE}: {exc}")
            return 1
        if not health["mock"]:
            print("warning: server is NOT in mock mode, this will move real hardware\n")

        deadman_ms = 1200
        client.post(f"{BASE}/api/debug/reset", timeout=5)

        # --- come to the person ------------------------------------------
        print("come: asking the robot to come over")
        task = client.post(f"{BASE}/api/skills/come", timeout=15).json()
        deadline = time.monotonic() + 120
        phase = ""
        while time.monotonic() < deadline:
            status = client.get(f"{BASE}/api/tasks/{task['task_id']}", timeout=10).json()
            if status["phase"] != phase:
                phase = status["phase"]
                print(f"  {phase}")
            if status["done"]:
                break
            time.sleep(0.3)
        else:
            print("  timed out waiting to arrive")
            return 1
        if not status["ok"]:
            print(f"  FAILED to reach the person: {status['detail']}")
            return 1

        # --- walker mode --------------------------------------------------
        print("\nwalker: starting")
        started = client.post(f"{BASE}/api/walker/start", timeout=10).json()
        if not started["ok"]:
            print(f"  refused to start: {started['state']}")
            return 1

        before = pose(client)
        print(f"walker: nudging forward for {NUDGE_FOR_S}s")
        until = time.monotonic() + NUDGE_FOR_S
        while time.monotonic() < until:
            client.post(
                f"{BASE}/api/walker/nudge", json={"direction": "forward"}, timeout=5
            )
            time.sleep(NUDGE_EVERY_S)
        during = pose(client)
        travelled = moved(before, during)
        print(f"  moved {travelled:.3f} m while being nudged")
        if travelled < 0.02:
            print("  FAILED: nudging did not move the base at all")
            client.post(f"{BASE}/api/walker/stop", timeout=5)
            return 1

        # --- the actual test: stop asking ---------------------------------
        print(f"\nwalker: silence. the base must stop within {deadman_ms}ms on its own")
        time.sleep(deadman_ms / 1000.0 + 0.6)
        settled = pose(client)
        time.sleep(0.8)
        after = pose(client)
        drift = moved(settled, after)
        print(f"  moved {drift:.4f} m in the 0.8s after it should have stopped")

        state = client.get(f"{BASE}/api/walker/state", timeout=5).json()
        print(f"  walker state: active={state['active']} moving={state['moving']}")

        client.post(f"{BASE}/api/walker/stop", timeout=5)

        if drift > 0.005:
            print("\nFAILED: the base kept moving after the nudges stopped")
            return 1
        if state["moving"]:
            print("\nFAILED: walker still reports itself as moving")
            return 1

        print("\ndead-man holds: no instructions, no movement")
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
