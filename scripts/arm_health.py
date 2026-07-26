#!/usr/bin/env python3
"""Quick SO-101 health check: voltage, torque, and a hold you can feel."""

from __future__ import annotations

import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

from robot.config import JOINT_NAMES, SERVO_IDS, SERVO_PORT

REGS = [
    "Torque_Enable",
    "Operating_Mode",
    "Present_Position",
    "Goal_Position",
    "Present_Voltage",
    "Present_Temperature",
    "Present_Current",
    "Present_Speed",
    "Moving",
]


def main() -> None:
    motors = {
        name: Motor(sid, "sts3215", MotorNormMode.RANGE_M100_100)
        for name, sid in zip(JOINT_NAMES, SERVO_IDS, strict=True)
    }
    bus = FeetechMotorsBus(port=SERVO_PORT, motors=motors)
    print(f"connecting {SERVO_PORT} ...")
    bus.connect(handshake=False)

    print("\n=== servo health ===")
    for name in JOINT_NAMES:
        print(name)
        for reg in REGS:
            try:
                val = bus.read(reg, name, normalize=False)
            except Exception as exc:  # noqa: BLE001
                val = f"ERR {type(exc).__name__}: {exc}"
            print(f"  {reg}: {val}")
        print()

    print("=== torque OFF ===")
    bus.disable_torque()
    time.sleep(0.3)
    for name in JOINT_NAMES:
        print(f"  {name}: Torque_Enable={bus.read('Torque_Enable', name, normalize=False)}")

    print("\n=== torque ON + hold current pose ===")
    for name in JOINT_NAMES:
        pos = int(bus.read("Present_Position", name, normalize=False))
        bus.write("Goal_Position", name, pos, normalize=False)
    bus.enable_torque()
    time.sleep(0.3)
    for name in JOINT_NAMES:
        print(f"  {name}: Torque_Enable={bus.read('Torque_Enable', name, normalize=False)}")

    print("\nHolding 8s — gently push a joint. It should feel stiff if torque is real.")
    time.sleep(8)

    print("\n=== after hold ===")
    for name in JOINT_NAMES:
        volt = bus.read("Present_Voltage", name, normalize=False)
        temp = bus.read("Present_Temperature", name, normalize=False)
        torq = bus.read("Torque_Enable", name, normalize=False)
        pos = bus.read("Present_Position", name, normalize=False)
        print(f"{name:15} torq={torq} volt={volt} temp={temp} pos={pos}")

    bus.disable_torque()
    bus.disconnect(disable_torque=True)
    print("\ndone — torque OFF")


if __name__ == "__main__":
    main()
