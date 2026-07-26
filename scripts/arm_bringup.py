#!/usr/bin/env python3
"""SO-101 arm bring-up: read joints, jog them, and replay named poses -- safely.

This is the tool you use with the arm on the bench before the skill loop ever runs.
Nothing moves until you enable torque, the first move home is deliberately slow, and
every jog is a small relative step so a typo can't fling a joint across the table.

    python scripts/arm_bringup.py                 # uses CORGI_SERVO_PORT, else COM7

Commands (type at the prompt):
    r                 read + print all joint positions
    t                 toggle torque (off = you can move it by hand)
    home              slowly interpolate to the HOME pose
    pose <NAME>       go to a named pose (SCAN, PRE_GRASP, DESCEND, LIFT, CARRY, ...)
    poses             list the named poses
    j <idx> <deg>     jog joint <idx> (0-5) by <deg> degrees, relative
    g open|close      open/close the gripper (close reports if it grabbed something)
    speed <ms>        set move duration in ms for pose/home moves (default 1500)
    q                 relax torque and quit
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot.arm import FeetechArm  # noqa: E402
from robot.config import JOINT_NAMES, SERVO_IDS  # noqa: E402
from robot.poses import POSES, resolve  # noqa: E402


def _print_positions(arm: FeetechArm) -> None:
    live = arm._read_all()  # noqa: SLF001 -- bring-up tool, intentionally low-level
    arm._positions = list(live)  # keep the in-memory start pose honest for the next move
    print("  idx  joint            id   degrees")
    for i, (name, sid, deg) in enumerate(zip(JOINT_NAMES, SERVO_IDS, live, strict=True)):
        print(f"  [{i}]  {name:15} {sid:>3}   {deg:+7.1f}")


def main() -> int:
    port = os.getenv("CORGI_SERVO_PORT", "COM7")
    os.environ.setdefault("CORGI_SERVO_PORT", port)
    print(f"connecting to SO-101 on {port} ...")

    arm = FeetechArm()
    torque = True  # FeetechArm enables torque on connect
    move_ms = 1500
    print("connected. torque is ON. current pose:\n")
    _print_positions(arm)
    print(
        "\nTip: 't' to relax and hand-pose it, 'home' to fold up slowly, 'q' to quit.\n"
        "The first 'home' from a random pose is the big one -- keep a hand near the arm."
    )

    while True:
        try:
            raw = input("arm> ").strip()
        except (EOFError, KeyboardInterrupt):
            raw = "q"
        if not raw:
            continue
        parts = raw.split()
        cmd = parts[0].lower()

        try:
            if cmd in ("q", "quit", "exit"):
                break
            elif cmd == "r":
                _print_positions(arm)
            elif cmd == "t":
                if torque:
                    arm.relax()
                    torque = False
                    print("torque OFF -- you can move the arm by hand. 't' to hold again.")
                else:
                    arm._bus.enable_torque()  # noqa: SLF001
                    arm._positions = arm._read_all()  # noqa: SLF001
                    torque = True
                    print("torque ON -- holding current pose.")
            elif cmd == "speed" and len(parts) == 2:
                move_ms = max(200, int(parts[1]))
                print(f"move duration set to {move_ms} ms")
            elif cmd == "home":
                if not torque:
                    print("torque is OFF -- enable it first with 't'.")
                    continue
                print(f"moving to HOME over {move_ms} ms ...")
                arm.go_to_pose("HOME", ms=move_ms)
                _print_positions(arm)
            elif cmd == "poses":
                print("  " + ", ".join(sorted(POSES)))
            elif cmd == "pose" and len(parts) == 2:
                name = parts[1].upper()
                if name not in POSES:
                    print(f"unknown pose {name!r}. have: {', '.join(sorted(POSES))}")
                    continue
                if not torque:
                    print("torque is OFF -- enable it first with 't'.")
                    continue
                print(f"moving to {name} over {move_ms} ms ...")
                arm.go_to_pose(name, ms=move_ms)
                _print_positions(arm)
            elif cmd == "j" and len(parts) == 3:
                if not torque:
                    print("torque is OFF -- enable it first with 't'.")
                    continue
                idx, delta = int(parts[1]), float(parts[2])
                if not 0 <= idx <= 5:
                    print("joint index must be 0-5")
                    continue
                target = arm.positions
                target[idx] += delta
                print(f"jogging {JOINT_NAMES[idx]} by {delta:+.1f} deg -> {target[idx]:+.1f}")
                arm.go_to_joints(target, ms=max(400, int(abs(delta) * 25)))
            elif cmd == "g" and len(parts) == 2 and parts[1] in ("open", "close"):
                if not torque:
                    print("torque is OFF -- enable it first with 't'.")
                    continue
                got = arm.set_gripper(parts[1])
                if parts[1] == "close":
                    print("closed on something" if got else "closed on nothing (empty jaws)")
                else:
                    print("opened")
            else:
                print("commands: r | t | home | pose <NAME> | poses | j <idx> <deg> | "
                      "g open|close | speed <ms> | q")
        except Exception as exc:  # bench tool: report and keep the prompt alive
            print(f"  ! {type(exc).__name__}: {exc}")

    print("relaxing torque and disconnecting ...")
    try:
        arm.relax()
        arm._bus.disconnect(disable_torque=True)  # noqa: SLF001
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {exc}")
    print("bye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
