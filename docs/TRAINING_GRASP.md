# Training the Walker Grasp Policy

Hand this to whoever has the **NVIDIA PC**. Recording happens on the laptop with the arms; training happens on the GPU box.

## Goal (hybrid architecture)

We only **learn** the hard part: see the pill bottle on a shelf and firmly grip it.

Everything after a confirmed grip is **scripted** (already working on hardware):

```
[LEARNED / ACT]   wrist-cam view → approach → close until firm grip
        ↓
[SCRIPT]          keep gripper closed (torque hold)
        ↓
[SCRIPT]          arm → HOME / LIFT tuck
        ↓
[SCRIPT]          Feetech linear slider UP
        ↓
[SCRIPT]          STOW over bucket → open jaws → done
```

**Do not** put slider lift or bucket drop in the training dataset. Short grasp-only demos train better and faster.

### Why shelf helps
Bottle sits at a roughly fixed height with a clear approach. Vary bottle position left/right/depth on the shelf a bit across episodes so the policy generalizes, but keep height consistent.

---

## Hardware (recording laptop)

| Piece | Plug into recording laptop |
|---|---|
| Follower SO-101 (on walker) | USB (CH343) + DC power |
| Leader SO-101 | USB (CH343) + DC power |
| Wrist camera on the arm | USB |

Not needed for training demos: walker Logitech, drive base, NVIDIA PC.

This laptop has **no NVIDIA GPU** (AMD only). Inference of a small ACT policy can run on CPU later; **training must run on the NVIDIA PC**.

Known ports from bring-up (may change after unplug/replug):

- Follower often appears as `COM7` (CH343 `VID:PID=1A86:55D3`)
- Leader will be a **second** CH343 COM port
- Find ports anytime: `lerobot-find-port`

Quit any `scripts/arm_bringup.py` session (`q`) before calibrating/recording — it holds the serial port.

---

## Software (both machines)

Recording laptop already has:

- `lerobot` 0.6.0
- CLI: `lerobot-calibrate`, `lerobot-teleoperate`, `lerobot-record`, `lerobot-train`, …

On the NVIDIA PC:

```powershell
# Python 3.10+ recommended
pip install "lerobot[feetech]"
# Install CUDA-enabled PyTorch matching the GPU driver, e.g.:
# https://pytorch.org/get-started/locally/
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

You want `True` and a real GPU name before training.

---

## Part A — Record demos (laptop with arms)

### 1. Calibrate (once per arm / after mechanical changes)

```powershell
lerobot-calibrate --robot.type=so101_follower --robot.port=COM7 --robot.id=walker_follower
lerobot-calibrate --teleop.type=so101_leader --teleop.port=COMx --teleop.id=walker_leader
```

Replace `COMx` with the leader port from `lerobot-find-port`. Follow the on-screen prompts (center, sweep ranges).

### 2. Teleop smoke test

```powershell
lerobot-teleoperate `
  --robot.type=so101_follower --robot.port=COM7 --robot.id=walker_follower `
  --teleop.type=so101_leader --teleop.port=COMx --teleop.id=walker_leader
```

Move the leader; the follower should mirror. Ctrl+C when done.

### 3. Confirm wrist camera index

```powershell
lerobot-find-cameras
```

Or OpenCV probe. Use the index that shows the **wrist view** (shelf + bottle), **not** the laptop webcam.

### 4. Record grasp-only episodes

~**40 episodes** is a solid start for a shelf pill-bottle grasp. Each episode:

1. Place bottle on shelf in wrist-cam view (vary XY a little)
2. Start episode
3. Teleop: approach → align → close until firm
4. **End episode immediately** once gripped (do not lift / slider / bucket)
5. Open jaws, reset bottle, repeat

Example (edit ports, camera index, and HF username):

```powershell
lerobot-record `
  --robot.type=so101_follower `
  --robot.port=COM7 `
  --robot.id=walker_follower `
  --robot.cameras="{ wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30} }" `
  --teleop.type=so101_leader `
  --teleop.port=COMx `
  --teleop.id=walker_leader `
  --dataset.repo_id=YOUR_HF_USER/walker_pill_grasp `
  --dataset.num_episodes=40 `
  --dataset.single_task="grasp the pill bottle on the shelf"
```

If you prefer not to use Hugging Face during recording, record to a local dataset path (see current `lerobot-record --help` for `dataset.root` / local options) and **copy the folder** to the NVIDIA PC with a USB drive.

### Dataset quality checklist
- Bottle always visible in wrist cam before motion
- Successful firm grips only (delete / skip failed episodes)
- No post-grasp lift in the episodes
- Similar lighting to the demo environment

---

## Part B — Train ACT (NVIDIA PC)

### 1. Get the dataset onto the GPU box
- Clone/pull from Hugging Face, **or**
- Copy the local LeRobot dataset folder via USB

### 2. Train

```powershell
lerobot-train `
  --dataset.repo_id=YOUR_HF_USER/walker_pill_grasp `
  --policy.type=act `
  --output_dir=outputs/train/walker_pill_grasp_act `
  --job_name=walker_pill_grasp_act `
  --policy.device=cuda `
  --batch_size=8 `
  --steps=100000
```

Tune `batch_size` to VRAM (drop to 4 if OOM). Checkpoint often under:

`outputs/train/walker_pill_grasp_act/checkpoints/last/pretrained_model`

Expect on the order of **1–3 hours** on a modern NVIDIA GPU for this small task — not days.

### 3. Ship the checkpoint back
Copy the `pretrained_model` (or whole `checkpoints/last`) folder back to the recording/walker laptop.

---

## Part C — Run the policy (walker laptop)

Policy drives the follower; **no leader** needed at runtime:

```powershell
lerobot-record `
  --robot.type=so101_follower `
  --robot.port=COM7 `
  --robot.id=walker_follower `
  --robot.cameras="{ wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30} }" `
  --policy.path=PATH/TO/pretrained_model `
  --dataset.repo_id=YOUR_HF_USER/walker_pill_grasp_eval `
  --dataset.num_episodes=10 `
  --dataset.single_task="grasp the pill bottle on the shelf"
```

(Exact flags can vary slightly by lerobot 0.6.x — if something errors, run `lerobot-record --help` and mirror the record-time camera/robot ids.)

**Success criterion for handoff to script:** gripper close reports holding (jaws stopped short of empty), same idea as `g close` → “closed on something” in `scripts/arm_bringup.py`.

Then the Corgi skill loop should:

1. Freeze / hold gripper closed  
2. Go to canned `LIFT` / `HOME`  
3. Raise linear slider  
4. `STOW` → open into bucket  

(Canned shelf/floor poses already live in `robot/poses.py` as a fallback: `PRE_GRASP`, `DESCEND`, `LIFT`.)

---

## Division of labor

| Person | Machine | Job |
|---|---|---|
| Arm / walker team | Laptop + arms + wrist cam | Calibrate, teleop, record ~40 grasp demos, run policy + scripted place |
| NVIDIA friend | GPU PC | Install CUDA torch + lerobot, train ACT, send checkpoint back |

---

## Fallback (demo insurance)

If the policy is flaky at the event, the hand-taught canned sequence still works for a fixed bottle pose:

`pose PRE_GRASP` → `pose DESCEND` → `g close` → `pose LIFT` → (slider) → stow/open

Proven on hardware 2026-07-26.

---

## Quick command cheat sheet

```powershell
lerobot-find-port
lerobot-find-cameras
lerobot-calibrate --robot.type=so101_follower --robot.port=COM7 --robot.id=walker_follower
lerobot-calibrate --teleop.type=so101_leader --teleop.port=COMx --teleop.id=walker_leader
lerobot-teleoperate --robot.type=so101_follower --robot.port=COM7 --robot.id=walker_follower --teleop.type=so101_leader --teleop.port=COMx --teleop.id=walker_leader
lerobot-train --policy.type=act --policy.device=cuda --dataset.repo_id=... --output_dir=... --steps=100000
```

When in doubt: `lerobot-record --help` / `lerobot-train --help` for the exact 0.6.0 flag names on that machine.
