#!/usr/bin/env python3
"""End-to-end smoke test against the running server.

    python scripts/smoke.py                 # one order, printing every phase
    python scripts/smoke.py 20              # 20 orders, prints a real success rate
    python scripts/smoke.py 20 --fail 0.3   # same, with 30% of grasps sabotaged
    python scripts/smoke.py --via text      # go in through iMessage, not the API

The multi-run mode is the one that matters. It is how you find out your actual success
rate instead of extrapolating from the last run you happened to watch. In mock mode the
--fail flag exercises the ask-for-help path before you ever see it on hardware.

`--via text` is the one that tests what a resident actually does: it posts a plain
English sentence to the webhook path, and checks that a reply came back. Everything
after that -- the router, the queue, the skills -- is the same code either way.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import httpx

BASE = os.getenv("CORGI_BASE_URL", "http://localhost:8000")
TARGET = "strawberries"
TERMINAL = {"done", "failed", "cancelled"}


def truth(client: httpx.Client, label: str) -> str:
    try:
        world = client.get(f"{BASE}/api/debug/world", timeout=5).json()
    except Exception:
        return ""
    for obj in world.get("objects", []):
        if obj["label"] == label:
            return f"d={obj['distance_m']}m bearing={obj['bearing_deg']}deg held={obj['held']}"
    return ""


def order_by_text(client: httpx.Client, target: str, verbose: bool) -> dict | None:
    """Ask in plain English, the way the resident does, and find the order it created."""
    sent = client.post(
        f"{BASE}/api/imessage/simulate",
        json={"text": f"can you bring me my {target} please"},
        timeout=20,
    ).json()
    if verbose:
        intent = sent.get("intent") or {}
        route = intent.get("route") or {}
        print(f"  routed as {intent.get('kind')!r} item={intent.get('item')!r} "
              f"via {route.get('backend')}/{route.get('tier')}")
        print(f"  replied  {sent.get('reply')!r}")

    if not sent.get("reply"):
        print("  no reply was composed for that text")
        return None

    log = client.get(f"{BASE}/api/imessage/log", timeout=10).json()
    if not any(m["ok"] for m in log.get("outbound", [])):
        print("  nothing was actually sent back to the resident")
        return None

    orders = client.get(f"{BASE}/api/orders", timeout=10).json()
    if not orders:
        print("  the text did not create an order")
        return None
    return orders[0]


def one(client: httpx.Client, target: str, fail_rate: float, verbose: bool, via: str) -> bool:
    client.post(
        f"{BASE}/api/debug/reset", params={"grasp_failure_rate": fail_rate}, timeout=5
    )

    if via == "text":
        order = order_by_text(client, target, verbose)
        if order is None:
            return False
    else:
        order = client.post(f"{BASE}/api/orders", json={"item": target}, timeout=15).json()

    last_phase = None
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        order = client.get(f"{BASE}/api/orders/{order['id']}", timeout=10).json()
        if verbose and order["phase"] != last_phase:
            last_phase = order["phase"]
            print(f"  {order['phase']:<12} {order['message']:<44} {truth(client, target)}")
        if order["status"] in TERMINAL:
            return order["status"] == "done"
        time.sleep(0.3)
    print("  timed out")
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="?", type=int, default=1)
    parser.add_argument("--item", default=TARGET)
    parser.add_argument("--fail", type=float, default=0.0, help="grasp failure rate, 0..1")
    parser.add_argument(
        "--via",
        choices=("api", "text"),
        default="api",
        help="api posts an order directly; text goes in through iMessage",
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("runs must be at least 1")

    with httpx.Client() as client:
        try:
            health = client.get(f"{BASE}/api/health", timeout=5).json()
        except Exception as exc:
            print(f"server not reachable at {BASE}: {exc}")
            return 1
        if not health["mock"]:
            print("warning: server is NOT in mock mode, this will move real hardware\n")

        wins = 0
        for i in range(args.runs):
            print(f"run {i + 1}/{args.runs}  (via {args.via})")
            ok = one(client, args.item, args.fail, args.runs == 1, args.via)
            wins += ok
            print(f"  -> {'ok' if ok else 'FAILED'}")

    print(f"\n{wins}/{args.runs} succeeded ({100 * wins // args.runs}%)")
    return 0 if wins == args.runs else 1


if __name__ == "__main__":
    sys.exit(main())
