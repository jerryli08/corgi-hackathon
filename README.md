# Corgi

A small grocery robot you order from on your phone. You type an item into a web page,
the robot looks around for it, drives over, picks it up, brings it back, and tells you
what it is doing the whole way.

One process on a Mac holds the entire robot: the serial link to the drive base, the
camera, the arm, perception, the skill state machine and the web server.

```
web/            what the customer sees, plus the ops console
robot/server.py HTTP + WebSocket API, the only entry point
robot/orders.py the order queue
robot/skills.py search -> approach -> align -> grasp -> verify -> return
robot/vision.py perception: HSV blobs offline, a VLM for the demo
robot/body.py   async facade over drive + arm + camera
robot/world.py  a simulated robot, so all of the above runs unplugged
firmware/       the Arduino sketch the drive base runs
```

## Run it with no hardware

```bash
pip install -r requirements.txt
CORGI_MOCK=1 ./scripts/run_server.sh
```

Then open <http://localhost:8000> and order something. The simulator starts with
`strawberries`, `banana`, `granola bar` and `water bottle` in front of the robot.

<http://localhost:8000/ops> is the view you actually want open while building: live
camera, phase log, manual jog and skill buttons, e-stop, and the simulator's ground
truth. When the servo loop converges somewhere wrong, the ground-truth panel tells you
immediately whether vision or control is the one lying to you.

```bash
python scripts/smoke.py            # one order, printing every phase
python scripts/smoke.py 20         # 20 orders, prints your real success rate
python scripts/smoke.py 20 --fail 0.3   # sabotage 30% of grasps on purpose
```

`smoke.py 20` is the one that matters. It is how you find out your actual success rate
instead of extrapolating from the last run you happened to watch.

## Going to hardware

1. Flash `firmware/drivebase/drivebase.ino` to the Arduino. `python
   scripts/check_devices.py` lists the serial ports and cameras the Mac can see.
2. Start without `CORGI_MOCK`. The drive base is autodetected; a missing camera or arm
   degrades the demo rather than stopping the server, and `/api/health` says what came
   up and what didn't.
3. Calibrate the two numbers that matter: `python scripts/calibrate.py strawberries`,
   then set `CORGI_SWEET_SPOT_X` and `CORGI_SWEET_SPOT_H` in `.env`.
4. Bolt on the arm and set `CORGI_ARM_ENABLED=1`. Without it the robot runs in
   drive-only mode: it finds the item and escorts you to it instead of lifting it.
5. Switch perception on for the demo: `CORGI_VISION_BACKEND=vlm` plus the matching
   API key. The color backend stays as the fallback for when the Wi-Fi is bad.

Copy `.env.example` to `.env` for the full list of knobs. Every tunable in the codebase
lives in `robot/config.py` and nothing else hardcodes a constant.

## How it decides where to stop

The base is the positioner and the arm is a stamp. Nothing ever estimates the object's
3D pose. The robot drives until the object appears at one calibrated spot in the image
— `SWEET_SPOT_X` across, `SWEET_SPOT_H` tall — and then replays one canned grasp that
is known to work from exactly there. `SWEET_SPOT_H` is the only depth cue a single
camera gets, and it is enough.

Grasp failure is detected for free: after closing, if the gripper servo travelled past
`GRIPPER_EMPTY_DEG` the jaws are empty. The robot retries once a little further
forward, then asks you to nudge the object toward it.

Every velocity command carries a duration and two independent watchdogs enforce it. The
host stops the wheels if the next command is late; the Arduino stops them if the host
stops talking altogether.

## API

| Method | Path | What |
|---|---|---|
| GET | `/api/health` | what hardware came up, and why anything didn't |
| GET | `/api/state` | robot phase, what it is carrying, the current order |
| POST | `/api/orders` | `{item}` — queue a fetch-and-deliver |
| GET | `/api/orders` · `/api/orders/{id}` | queue and single-order status |
| POST | `/api/orders/{id}/cancel` | only while still queued |
| WS | `/api/events` | phase stream, `human_text` is spoken verbatim |
| GET | `/api/camera/frame.jpg` · `/api/camera/stream.mjpg` | what the robot sees |
| POST | `/api/skills/scene` · `/locate` · `/fetch` · `/deliver` · `/return_item` | skills, direct |
| GET | `/api/tasks/{id}` | `{phase, done, ok, detail}` |
| POST | `/api/drive/velocity` · `/api/drive/cmd` · `/api/drive/stop` | manual driving |
| POST | `/api/arm/pose` · `/api/arm/gripper` | manual arm, for calibration |
| POST | `/api/estop` | stop everything, relax the arm |
| GET | `/api/debug/world` · POST `/api/debug/reset` | simulator ground truth, mock only |

Bounding boxes are always normalized `[x0, y0, x1, y1]` in `0..1`, origin top-left.

Phases: `SEARCHING` · `APPROACHING` · `ALIGNING` · `GRASPING` · `VERIFYING` ·
`NEEDS_HELP` · `RETURNING` · `PRESENTING` · `REPLACING` · `DONE` · `FAILED`.
