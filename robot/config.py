"""Every tunable number lives here. Nothing else should hardcode a constant.

The two that matter most are SWEET_SPOT_X and SWEET_SPOT_H: where an object has to
appear in the camera frame for the canned grasp to work. Calibrate them with
`python scripts/calibrate.py <item>`, then write them on tape and stick it on the robot.
"""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is a convenience, not a requirement
    pass


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _num(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


# --- Mode -----------------------------------------------------------------
# MOCK=1 runs the whole stack against a simulated robot and a synthetic camera,
# so the server, the skills and the web UI all work with nothing plugged in.
MOCK = _flag("CORGI_MOCK")

HOST = os.getenv("CORGI_HOST", "0.0.0.0")
PORT = _int("CORGI_PORT", 8000)

# --- Drive base (Arduino + 2x continuous-rotation servos) -----------------
DRIVE_PORT = os.getenv("CORGI_DRIVE_PORT") or None
DRIVE_BAUD = _int("CORGI_DRIVE_BAUD", 115200)

STOP_US = 1500
MIN_US, MAX_US = 1000, 2000
# Pulse offset from STOP_US at full commanded speed. Deliberately well short of the
# 500us the servos allow: the demo should read as deliberate, not sluggish or frantic.
FULL_SCALE_US = _int("CORGI_FULL_SCALE_US", 300)
DEFAULT_SPEED_US = _int("CORGI_DEFAULT_SPEED_US", 120)

WHEEL_BASE_M = _num("CORGI_WHEEL_BASE_M", 0.22)
MAX_LINEAR = _num("CORGI_MAX_LINEAR", 0.25)  # m/s
MAX_ANGULAR = _num("CORGI_MAX_ANGULAR", 1.2)  # rad/s

# If no new velocity arrives within (ms + grace), the host stops the wheels itself.
WATCHDOG_GRACE_MS = _int("CORGI_WATCHDOG_GRACE_MS", 200)
# And if the host stops talking altogether, the Arduino stops the wheels on its own.
# Two independent watchdogs, deliberately: the second one covers the host dying.
FIRMWARE_WATCHDOG_MS = _int("CORGI_FIRMWARE_WATCHDOG_MS", 1500)

# --- Arm (SO-101, Feetech STS3215 on a half-duplex TTL bus) ---------------
# Off by default: the base robot drives to the item and presents it. Turn it on
# once the arm is bolted down and calibrated.
ARM_ENABLED = _flag("CORGI_ARM_ENABLED")
SERVO_PORT = os.getenv("CORGI_SERVO_PORT", "/dev/tty.usbmodem58FA0960681")
SERVO_BAUD = _int("CORGI_SERVO_BAUD", 1_000_000)

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
SERVO_IDS = [1, 2, 3, 4, 5, 6]

# STS3215: 4096 ticks over 360 degrees.
TICKS_PER_DEG = 4096.0 / 360.0
CENTER_TICKS = 2048

# Per-joint sign flip, set during assembly. If a joint moves the wrong way, flip it
# here rather than negating angles at the call site.
JOINT_SIGN = [1, 1, 1, 1, 1, 1]

MOVE_HZ = 50  # interpolation rate for joint-space moves

GRIPPER_OPEN_DEG = 35.0
GRIPPER_CLOSED_DEG = 0.0
# If the gripper travels past this on a close, the jaws are empty. Calibrate by closing
# on nothing, then on your smallest object, and picking a value between them. This is
# the entire grasp-failure detector and it costs nothing.
GRIPPER_EMPTY_THRESHOLD_DEG = _num("CORGI_GRIPPER_EMPTY_DEG", 4.0)

# --- Camera ---------------------------------------------------------------
CAMERA_ENABLED = _flag("CORGI_CAMERA_ENABLED", "1")
CAMERA_INDEX = _int("CORGI_CAMERA_INDEX", 0)
FRAME_W, FRAME_H = 640, 480
JPEG_QUALITY = 60
CAMERA_FOV_DEG = 60.0
STREAM_FPS = _int("CORGI_STREAM_FPS", 12)

# --- Vision ---------------------------------------------------------------
# "color" needs no network and works against the simulator. "vlm" is the demo path.
VISION_BACKEND = os.getenv("CORGI_VISION_BACKEND", "color")
VLM_PROVIDER = os.getenv("CORGI_VLM_PROVIDER", "gemini")  # gemini | openai | anthropic
VLM_MODEL = os.getenv("CORGI_VLM_MODEL", "")  # blank = provider default
VLM_MAX_EDGE = 512  # downscale before upload; roughly halves round-trip latency
VLM_TIMEOUT_S = _num("CORGI_VLM_TIMEOUT_S", 8.0)

# --- The sweet spot: where an object must appear for the canned grasp to work ---
SWEET_SPOT_X = _num("CORGI_SWEET_SPOT_X", 0.50)  # normalized image x
SWEET_SPOT_H = _num("CORGI_SWEET_SPOT_H", 0.34)  # normalized bbox height
TOL_X = _num("CORGI_TOL_X", 0.035)
TOL_H = _num("CORGI_TOL_H", 0.03)

# Proportional gains for the move-then-look loop.
K_ANGULAR = _num("CORGI_K_ANGULAR", 1.6)
K_LINEAR = _num("CORGI_K_LINEAR", 1.2)
STEP_MS = _int("CORGI_STEP_MS", 300)
MAX_SERVO_STEPS = _int("CORGI_MAX_SERVO_STEPS", 40)

# Search: rotate this much per look until the target appears.
SEARCH_STEP_RAD = _num("CORGI_SEARCH_STEP_RAD", 0.35)
MAX_SEARCH_STEPS = _int("CORGI_MAX_SEARCH_STEPS", 24)

# bbox height -> arm reach bucket. Bigger box means closer object means shorter reach.
REACH_BUCKETS = [(0.40, "near"), (0.30, "mid"), (0.0, "far")]

# --- Delivery -------------------------------------------------------------
HOME_TAG_ID = _int("CORGI_HOME_TAG_ID", 0)
HOME_TAG_LABEL = "home_tag"
DELIVER_SWEET_SPOT_H = _num("CORGI_DELIVER_SWEET_SPOT_H", 0.45)
