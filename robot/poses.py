"""Named joint-space keyframes, in degrees, in JOINT_NAMES order.

This is the deliberate substitute for inverse kinematics. Recalibrate these by
hand-posing the arm and reading back `GET /api/arm/state` -- do not compute them.
"""

from robot.config import GRIPPER_CLOSED_DEG, GRIPPER_OPEN_DEG

_O = GRIPPER_OPEN_DEG
_C = GRIPPER_CLOSED_DEG

POSES = {
    # folded, safe to power on in
    "HOME":      [0.0, -80.0,  95.0,  40.0, 0.0, _O],
    # camera looking forward, arm out of frame -- the driving/searching pose
    "SCAN":      [0.0, -55.0,  80.0,  10.0, 0.0, _O],
    # above the object, jaws open, camera looking down at it
    "PRE_GRASP": [0.0, -20.0,  55.0,  55.0, 0.0, _O],
    # lowered onto the object
    "DESCEND":   [0.0,   0.0,  40.0,  50.0, 0.0, _O],
    "LIFT":      [0.0, -35.0,  70.0,  55.0, 0.0, _C],
    # tucked in tight while driving, keeps the centre of gravity back
    "CARRY":     [0.0, -75.0,  95.0,  60.0, 0.0, _C],
    # held out toward the customer
    "PRESENT":   [0.0, -30.0,  50.0,  20.0, 0.0, _C],
}

# The single parameter that replaces IK: how far forward the arm extends.
# Applied to shoulder_lift and elbow_flex only, on grasp poses only.
REACH_OFFSETS = {
    "near": (-6.0, +8.0),
    "mid":  (0.0, 0.0),
    "far":  (+7.0, -9.0),
}

_REACH_APPLIES_TO = {"PRE_GRASP", "DESCEND"}


def resolve(name: str, reach: str = "mid") -> list[float]:
    if name not in POSES:
        raise KeyError(f"unknown pose {name!r}, have {sorted(POSES)}")
    pose = list(POSES[name])
    if name in _REACH_APPLIES_TO:
        d_shoulder, d_elbow = REACH_OFFSETS.get(reach, (0.0, 0.0))
        pose[1] += d_shoulder
        pose[2] += d_elbow
    return pose
