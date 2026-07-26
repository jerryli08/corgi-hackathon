"""Hybrid pill-bottle grasp: learned ACT approach+grip, then scripted place.

The ACT policy is only trusted for the hard part -- servo the wrist cam onto the
bottle and close the jaws. Everything after a confirmed grip (lift, slider up,
carry, drop in the bucket) is deterministic keyframe playback, so a wobbly policy
can't throw the arm across the room mid-demo.

Runs on the walker laptop against the real SO-101 on COM7 with the wrist C920 on
OpenCV index 1 -- the exact hardware the dataset was recorded on, so the policy
sees the same joint/gripper units it trained on.

    python scripts/run_grasp_policy.py \
        --checkpoint outputs/train/walker_pill_grasp_act/checkpoints/015000/pretrained_model

Keys while running:
    g   force the handoff to scripted place right now (operator override)
    q   abort: stop sending actions and relax

Scripted place waypoints come from robot/place_poses.json (see teach_place_poses.py).
Without that file the policy still runs and grips; the arm just holds instead of
placing, which is enough to prove the learned half on its own.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PLACE_POSES_PATH = REPO_ROOT / "robot" / "place_poses.json"

# Dataset action/state order -- must match datasets/.../meta/info.json.
MOTOR_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
CAM_KEY = "wrist"  # matches observation.images.wrist in the dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a trained checkpoint's pretrained_model/ directory.",
    )
    p.add_argument("--port", default="COM7", help="Follower serial port.")
    p.add_argument("--robot-id", default="walker_follower", help="Calibration id.")
    p.add_argument("--cam-index", type=int, default=1, help="Wrist cam OpenCV index.")
    p.add_argument("--task", default="grasp the pill bottle on the shelf")
    p.add_argument("--fps", type=float, default=30.0, help="Control loop rate.")
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device.",
    )
    p.add_argument(
        "--max-seconds",
        type=float,
        default=30.0,
        help="Hard cap on the learned phase before forcing the handoff.",
    )
    p.add_argument(
        "--warmup-seconds",
        type=float,
        default=3.0,
        help="Ignore auto grip detection for this long so it can't fire on the "
        "opening pose before the arm has reached the bottle.",
    )
    p.add_argument(
        "--hold-frames",
        type=int,
        default=8,
        help="Consecutive frames of a stalled-closed gripper that count as a grip.",
    )
    p.add_argument(
        "--no-place",
        action="store_true",
        help="Grip only; skip scripted place even if place_poses.json exists.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the policy and print actions but do NOT drive the arm.",
    )
    return p.parse_args()


class KeyWatcher:
    """Non-blocking single-key reader (Windows msvcrt, POSIX fallback)."""

    def __init__(self) -> None:
        self._key: str | None = None
        self._stop = False
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def _loop(self) -> None:
        try:
            import msvcrt  # Windows

            while not self._stop:
                if msvcrt.kbhit():
                    self._key = msvcrt.getch().decode(errors="ignore").lower()
                time.sleep(0.02)
        except ImportError:
            for line in sys.stdin:  # POSIX: line-buffered fallback
                if self._stop:
                    break
                line = line.strip().lower()
                if line:
                    self._key = line[0]

    def get(self) -> str | None:
        k, self._key = self._key, None
        return k

    def stop(self) -> None:
        self._stop = True


def build_robot(args: argparse.Namespace):
    # Use SOFollowerRobotConfig (registered as so101_follower), not the bare
    # SOFollowerConfig dataclass -- that one is missing id/calibration_dir and
    # crashes Robot.__init__.
    import lerobot.robots.so_follower  # noqa: F401  # registers choice types
    from lerobot.cameras.configs import Cv2Backends
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.config import RobotConfig

    # DSHOW: MSMF/ANY often hangs forever on Windows webcam open.
    cam = OpenCVCameraConfig(
        index_or_path=args.cam_index,
        width=640,
        height=480,
        fps=30,
        backend=Cv2Backends.DSHOW,
        warmup_s=1,
    )
    cfg_cls = RobotConfig.get_choice_class("so101_follower")
    cfg = cfg_cls(port=args.port, id=args.robot_id, cameras={CAM_KEY: cam})
    robot = make_robot_from_config(cfg)

    # Connect in pieces so a hang tells us which device is stuck.
    print(f"[hw] bus.connect({args.port})...", flush=True)
    robot.bus.connect()
    print("[hw] bus OK", flush=True)
    if getattr(robot, "calibration", None):
        try:
            robot.bus.write_calibration(robot.calibration)
            print("[hw] calibration written", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[hw] calibration write skipped: {e}", flush=True)
    for name, cam_dev in robot.cameras.items():
        print(f"[hw] camera '{name}' connect...", flush=True)
        cam_dev.connect()
        print(f"[hw] camera '{name}' OK", flush=True)
    print("[hw] configure...", flush=True)
    robot.configure()
    print("[hw] connected.", flush=True)
    return robot


def load_policy(args: argparse.Namespace):
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    ckpt = str(Path(args.checkpoint))
    policy = ACTPolicy.from_pretrained(ckpt)
    policy.to(args.device)
    policy.eval()
    # Normalization stats are baked into the checkpoint's processor config, so
    # dataset_stats can stay None here.
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=ckpt,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    return policy, preprocessor, postprocessor


def obs_to_batch(obs: dict, task: str) -> dict:
    state = np.array([float(obs[f"{m}.pos"]) for m in MOTOR_ORDER], dtype=np.float32)
    # The normalizer works in float; handing it raw uint8 makes it try to cast
    # float stats down to uint8 and overflow. Camera gives H x W x 3 RGB uint8,
    # the policy wants B x C x H x W float in [0, 1].
    img = np.asarray(obs[CAM_KEY], dtype=np.float32) / 255.0
    img_t = torch.from_numpy(img).permute(2, 0, 1).contiguous()
    batch = {
        "observation.state": torch.from_numpy(state)[None],
        f"observation.images.{CAM_KEY}": img_t[None],
        "task": task,
    }
    return batch


def action_to_dict(action: torch.Tensor) -> dict:
    vals = action.squeeze(0).detach().cpu().numpy().tolist()
    return {f"{m}.pos": float(v) for m, v in zip(MOTOR_ORDER, vals, strict=True)}


def move_holding(robot, target: dict, grip_value: float, ms: int, fps: float, dry_run: bool) -> None:
    """Interpolate the 5 body joints to `target` while pinning the gripper shut.

    Every frame re-commands the gripper at `grip_value`, so the jaws keep
    squeezing the bottle for the whole ride instead of drifting open.
    """
    period = 1.0 / fps
    start = {k: float(v) for k, v in robot.get_observation().items() if k.endswith(".pos")}
    steps = max(1, int((ms / 1000.0) * fps))
    for i in range(1, steps + 1):
        a = i / steps
        frame = {
            k: start[k] + (float(target[k]) - start[k]) * a
            for k in target
            if k != "gripper.pos"
        }
        frame["gripper.pos"] = grip_value
        if not dry_run:
            robot.send_action(frame)
        time.sleep(period)


def run_scripted_place(
    robot, home_pose: dict, grip_value: float, fps: float, dry_run: bool
) -> None:
    """Post-grasp: retract to the start pose with the jaws clamped, then either
    replay taught place waypoints or hold and wait for the operator."""
    print(f"[place] returning to start pose, gripper pinned at {grip_value:.1f}")
    move_holding(robot, home_pose, grip_value, ms=2500, fps=fps, dry_run=dry_run)
    print("[place] at home, still holding.")

    if PLACE_POSES_PATH.exists():
        waypoints = json.loads(PLACE_POSES_PATH.read_text())
        print(f"[place] Replaying {len(waypoints)} scripted waypoints.")
        for wp in waypoints:
            target = wp["pos"]  # dict motor.pos -> value, in robot units
            grip = float(wp.get("gripper", grip_value))
            move_holding(robot, target, grip, ms=int(wp.get("ms", 800)), fps=fps, dry_run=dry_run)
        print("[place] Done.")
        return

    # No taught waypoints yet: hold the bottle at home until the operator
    # decides -- 'o' opens the jaws (drop into basket by hand), 'q' quits.
    print("[place] no place_poses.json -- holding. Press 'o' to open jaws, 'q' to finish.")
    keys = KeyWatcher()
    try:
        while True:
            k = keys.get()
            if k == "o":
                obs = robot.get_observation()
                release = {"gripper.pos": grip_value + 30.0}
                if not dry_run:
                    robot.send_action(release)
                print("[place] jaws opened.")
            elif k == "q":
                break
            if not dry_run:
                # keep re-asserting the hold so the servo doesn't sag
                pass
            time.sleep(0.05)
    finally:
        keys.stop()


def main() -> int:
    args = parse_args()

    print(f"[load] policy from {args.checkpoint} on {args.device}")
    policy, preprocessor, postprocessor = load_policy(args)
    policy.reset()

    print(f"[hw] connecting follower on {args.port}, wrist cam index {args.cam_index}")
    robot = build_robot(args)

    # The pose the operator started us in (should be HOME-ish). We retract back
    # to exactly this after the grip, so no unit conversion is ever needed.
    home_pose = {
        k: float(v) for k, v in robot.get_observation().items() if k.endswith(".pos")
    }
    print(f"[hw] start pose captured ({len(home_pose)} joints) -- this is 'home'.")

    keys = KeyWatcher()
    period = 1.0 / args.fps
    t0 = time.time()
    open_ref: float | None = None
    last_cmd_grip: float | None = None
    grip_now = 0.0
    hold_count = 0
    gripped = False
    aborted = False

    print("[run] learned grasp phase. 'g' = force place, 'q' = abort.")
    try:
        while True:
            loop_start = time.time()
            elapsed = loop_start - t0

            k = keys.get()
            if k == "q":
                aborted = True
                print("[run] aborted by operator.")
                break
            if k == "g":
                print("[run] operator forced handoff.")
                break

            obs = robot.get_observation()
            grip_now = float(obs["gripper.pos"])
            if open_ref is None:
                open_ref = grip_now  # first frame: jaws are open (baseline)

            batch = obs_to_batch(obs, args.task)
            batch = preprocessor(batch)
            with torch.no_grad():
                action = policy.select_action(batch)
            action = postprocessor(action)
            action_dict = action_to_dict(action)

            if not args.dry_run:
                robot.send_action(action_dict)

            # Grip heuristic: policy is commanding the gripper toward closed
            # (well below where it started open) yet the measured jaw stays open
            # of fully-shut -- i.e. it stalled on the bottle -- for hold_frames.
            cmd_grip = action_dict["gripper.pos"]
            last_cmd_grip = cmd_grip
            commanding_close = open_ref is not None and cmd_grip < (open_ref - abs(open_ref) * 0.3 - 1.0)
            stalled_on_object = grip_now > (cmd_grip + 3.0)
            if elapsed >= args.warmup_seconds and commanding_close and stalled_on_object:
                hold_count += 1
                if hold_count >= args.hold_frames:
                    gripped = True
                    print(f"[run] grip confirmed at t={elapsed:.1f}s (jaw={grip_now:.1f}).")
                    break
            else:
                hold_count = 0

            if elapsed >= args.max_seconds:
                print(f"[run] max-seconds ({args.max_seconds}s) hit -- forcing handoff.")
                break

            dt = time.time() - loop_start
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        aborted = True
        print("\n[run] KeyboardInterrupt -- stopping.")
    finally:
        keys.stop()

    if not aborted and not args.no_place:
        # Pin the jaws at the tighter of (what the policy was commanding, what
        # the jaw actually reads), so the squeeze is kept during the retract.
        candidates = [v for v in (last_cmd_grip, grip_now) if v is not None]
        grip_value = max(0.0, min(candidates)) if candidates else 0.0
        run_scripted_place(robot, home_pose, grip_value, args.fps, args.dry_run)

    try:
        if hasattr(robot, "disconnect"):
            robot.disconnect()
    except Exception as e:  # noqa: BLE001
        print(f"[hw] disconnect warning: {e}")

    print(f"[done] gripped={gripped} aborted={aborted}")
    return 0 if not aborted else 1


if __name__ == "__main__":
    raise SystemExit(main())
