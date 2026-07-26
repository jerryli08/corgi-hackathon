#!/usr/bin/env python3
"""List USB serial ports and cameras visible to the Mac brain."""

from __future__ import annotations

import subprocess
import sys

from serial.tools import list_ports


def main() -> int:
    print("=== Serial ports ===")
    ports = list(list_ports.comports())
    if not ports:
        print("(none)")
    for p in ports:
        print(f"{p.device:30}  {p.description}  [{p.manufacturer or 'n/a'}]")

    print("\n=== Cameras (system_profiler) ===")
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPCameraDataType"], text=True, stderr=subprocess.DEVNULL
        )
        print(out.strip() or "(none)")
    except Exception as exc:  # noqa: BLE001
        print(f"could not query cameras: {exc}")

    print("\nTip: plug Arduino, SO-101 bus, and C920 into the USB hub, then re-run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
