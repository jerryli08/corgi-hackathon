"""Start the Corgi server with the two-camera mission owning the Arduino."""

from __future__ import annotations

import os
import sys

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# This launcher intentionally forces the server-side hardware objects off. The
# singlecam subprocess is the sole owner of the cameras and Arduino connection.
FORCED_SETTINGS = {
    "CORGI_MOCK": "1",
    "CORGI_CAMERA_ENABLED": "0",
    "CORGI_ARM_ENABLED": "0",
    "CORGI_SINGLECAM_MISSION_ENABLED": "1",
    "CORGI_SINGLECAM_PAYLOAD_ENABLED": "0",
    "CORGI_SINGLECAM_SERVO_ENABLED": "1",
}

# Per-computer values in .env or the shell still win for these settings.
DEFAULTS = {
    "CORGI_SINGLECAM_ARDUINO_PORT": "COM5",
    "CORGI_HOST": "127.0.0.1",
    "CORGI_PORT": "8000",
}

os.environ.update(FORCED_SETTINGS)
for name, value in DEFAULTS.items():
    os.environ.setdefault(name, value)


def main() -> int:
    try:
        import uvicorn
    except ImportError:
        print(
            "ERROR: Missing server dependencies. Activate the project virtual "
            "environment and run: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    print("Starting Corgi two-camera robot server")
    print(f"Arduino: {os.environ['CORGI_SINGLECAM_ARDUINO_PORT']}")
    print("Arm/payload sequence: disabled (safe dry-run)")
    print("Server: http://127.0.0.1:8000")
    uvicorn.run(
        "robot.server:app",
        host=os.environ["CORGI_HOST"],
        port=int(os.environ["CORGI_PORT"]),
        reload=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
