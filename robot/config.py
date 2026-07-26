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


def _csv(name: str, default: str = "") -> list[str]:
    return [part.strip() for part in os.getenv(name, default).split(",") if part.strip()]


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

# --- Messaging (Photon / iMessage) ----------------------------------------
# log         = print and record, no network. The default, and what the tests use.
# photon      = Photon Spectrum, through the bridge in corgi/ (outbound only; inbound
#               arrives on the relay endpoint below, pushed by that same process).
# applescript = the Messages app on this Mac via osascript. No account, no keys,
#               genuinely iMessage; the fallback when the bridge is not running.
MESSAGING_BACKEND = os.getenv("CORGI_MESSAGING_BACKEND", "log")
PHOTON_BRIDGE_URL = os.getenv("CORGI_PHOTON_BRIDGE_URL", "http://127.0.0.1:8787")
PHOTON_BRIDGE_TIMEOUT_S = _num("CORGI_PHOTON_BRIDGE_TIMEOUT_S", 6.0)
# Shared secret the corgi/ bridge sends on every relayed inbound text, so the endpoint
# is not wide open to anything else running on the same machine. Blank skips the check
# -- the bridge only binds loopback anyway, so this is defense in depth, not the lock.
BRIDGE_SECRET = os.getenv("CORGI_BRIDGE_SECRET", "")
# Spectrum's webhook signing secret. It is shown once, when the webhook is registered.
# Only needed if you register a public Photon Cloud webhook instead of running the
# corgi/ bridge -- the two are alternatives, not both at once.
PHOTON_WEBHOOK_SECRET = os.getenv("CORGI_PHOTON_WEBHOOK_SECRET", "")
# Refuse unsigned webhooks. Only turn this off to poke at the endpoint by hand.
PHOTON_REQUIRE_SIGNATURE = _flag("CORGI_PHOTON_REQUIRE_SIGNATURE", "1")
PHOTON_WEBHOOK_TOLERANCE_S = _int("CORGI_PHOTON_WEBHOOK_TOLERANCE_S", 300)

# Blank = answer anyone. Otherwise an allowlist of E.164 numbers; anyone else gets one
# polite refusal and is then ignored.
ALLOWED_SENDERS = _csv("CORGI_ALLOWED_SENDERS")
# Photon's default quota is 5000 messages per server per day. We are nowhere near it,
# but an elderly user must never get nine texts about one water bottle, so cap it hard.
DAILY_MESSAGE_BUDGET = _int("CORGI_DAILY_MESSAGE_BUDGET", 200)
TEXT_MIN_GAP_S = _num("CORGI_TEXT_MIN_GAP_S", 6.0)
# Let the browser's phone simulator inject messages. This is the whole unplugged demo.
ALLOW_SIMULATED_TEXTS = _flag("CORGI_ALLOW_SIMULATED_TEXTS", "1")

# --- Router (Merge Gateway) -----------------------------------------------
# keyword = deterministic, offline, and the fallback for every merge failure.
# merge   = Merge Gateway, which is itself routing across providers underneath us.
ROUTER_BACKEND = os.getenv("CORGI_ROUTER_BACKEND", "keyword")
MERGE_BASE_URL = os.getenv("CORGI_MERGE_BASE_URL", "https://api-gateway.merge.dev/v1")
# responses = the official merge_gateway SDK's native /responses call, which reports
#             back which vendor actually served each request. The default.
# openai    = the OpenAI-compatible shim at {base}/openai/chat/completions, plain REST,
#             no SDK dependency -- the fallback for anyone who does not want the package.
MERGE_API = os.getenv("CORGI_MERGE_API", "responses")
MERGE_API_KEY = os.getenv("MERGE_API_KEY", "")
# Two tiers. A short command ("water please") never needs more than the fast one.
# These are catalogue ids and they move: `python scripts/check_router.py` prints what
# your key can actually reach, rather than making you guess.
ROUTER_FAST_MODEL = os.getenv("CORGI_ROUTER_FAST_MODEL", "anthropic/claude-haiku-4-5-20251001")
ROUTER_DEEP_MODEL = os.getenv("CORGI_ROUTER_DEEP_MODEL", "anthropic/claude-sonnet-5")
ROUTER_TIMEOUT_S = _num("CORGI_ROUTER_TIMEOUT_S", 8.0)
# Below this the fast model's answer is not trusted and the deep model gets a turn.
ROUTER_CONFIDENCE_FLOOR = _num("CORGI_ROUTER_CONFIDENCE_FLOOR", 0.65)
ROUTER_MAX_CHARS = _int("CORGI_ROUTER_MAX_CHARS", 600)

# --- Coming to a person ---------------------------------------------------
# Stop further out than a grasp: this ends up beside someone, not against them.
COME_SWEET_SPOT_H = _num("CORGI_COME_SWEET_SPOT_H", 0.62)
COME_TOL_H = _num("CORGI_COME_TOL_H", 0.06)
PERSON_LABEL = os.getenv("CORGI_PERSON_LABEL", "person")
# A standing person's bbox is taller than it is wide. A chair's is not. The cheapest
# possible guard against a VLM confidently boxing the furniture.
PERSON_MIN_ASPECT = _num("CORGI_PERSON_MIN_ASPECT", 1.15)

# --- Walking alongside someone -------------------------------------------
# NOT a mobility aid and NOT load-bearing: the base weighs a couple of kilos and runs
# on hobby servos. This is a paced escort with a handhold reference at a known height.
# Half speed, because the point is to be predictable next to someone unsteady.
WALK_LINEAR = _num("CORGI_WALK_LINEAR", 0.12)  # m/s
WALK_ANGULAR = _num("CORGI_WALK_ANGULAR", 0.5)  # rad/s
WALK_STEP_MS = _int("CORGI_WALK_STEP_MS", 250)
# Dead-man: the wheels move only while instructions keep arriving. Miss one and stop.
# This is what makes an open-loop hobby-servo base acceptable next to a person.
WALK_DEADMAN_MS = _int("CORGI_WALK_DEADMAN_MS", 1200)
# And the mode itself times out, so a forgotten session does not idle with the arm out.
WALK_MAX_S = _num("CORGI_WALK_MAX_S", 300.0)
# Pause and re-look for the person every this many steps.
WALK_RECHECK_STEPS = _int("CORGI_WALK_RECHECK_STEPS", 12)

# --- Basket ---------------------------------------------------------------
# What the robot can carry aboard at once. The arm stows into it and lifts back out.
BASKET_CAPACITY = _int("CORGI_BASKET_CAPACITY", 3)

# --- Singlecam pill-bottle mission ---------------------------------------
# Optional bridge to the Depth Anything + SegFormer experiment. It uses the same
# Python executable that runs this server unless CORGI_SINGLECAM_MISSION_PYTHON is set.
SINGLECAM_MISSION_ENABLED = _flag("CORGI_SINGLECAM_MISSION_ENABLED")
SINGLECAM_MISSION_ITEM_KEYWORDS = _csv(
    "CORGI_SINGLECAM_MISSION_ITEM_KEYWORDS", "pill bottle,pills,medicine"
)
SINGLECAM_MISSION_SCRIPT = os.getenv(
    "CORGI_SINGLECAM_MISSION_SCRIPT", "experiments/singlecam_depth_pathing/main.py"
)
SINGLECAM_MISSION_PYTHON = os.getenv("CORGI_SINGLECAM_MISSION_PYTHON", "")
SINGLECAM_MISSION_TIMEOUT_S = _num("CORGI_SINGLECAM_MISSION_TIMEOUT_S", 0.0)
