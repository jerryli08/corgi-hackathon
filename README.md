# Corgi

A small robot you text. It lives with someone who does not want to get up, and it
answers plain English sent from the Messages app they already use.

Two things it does:

```
"can you bring me my water bottle"   -> finds it, picks it up, puts it in its basket,
                                        drives back, and hands it over
"come here" / "walk with me"         -> drives to you, then keeps pace beside you with
                                        the arm held out as something to steady a hand on
```

One process on a Mac holds the whole robot: the serial link to the drive base, the
camera, the arm, perception, the skill state machine, the iMessage transport, the router
that reads the messages, and the web server.

```
web/              the resident's page (a phone) and the ops console
corgi/            the Photon/Spectrum bridge -- its own scaffolded project, gitignored
robot/server.py   HTTP + WebSocket API, the only entry point
robot/concierge.py a text arrives -> an intent -> the robot does something -> a reply
robot/brain.py    Merge Gateway as an LLM router: free text -> one typed intent
robot/messaging.py iMessage in and out, and the rate limiter between
robot/orders.py   the order queue
robot/skills.py   search -> approach -> align -> grasp -> verify -> stow -> return
robot/walker.py   walking alongside someone, dead-man operated
robot/vision.py   perception: HSV blobs offline, a VLM for the demo
robot/world.py    a simulated robot, so all of the above runs unplugged
firmware/         the Arduino sketch the drive base runs
```

## What walker mode is, and what it is not

The base is two continuous-rotation hobby servos on a 7.4V battery. It weighs a couple
of kilos.

**It cannot take a person's weight and nothing here pretends otherwise.** Walker mode is
a paced escort: the robot travels at walking speed beside someone and holds the arm up so
there is something at a known height to rest a hand against. It is not a mobility aid,
not a medical device, and not load-bearing. The strings in the product say "walk with
you" and "steady yourself", never "lean on".

It is also dead-man operated. The wheels turn only while instructions keep arriving — let
go of the button and the base stops within `CORGI_WALK_DEADMAN_MS`. That is the property
that makes an open-loop base acceptable next to someone unsteady, and
`python scripts/smoke_walk.py` is the test that it actually holds.

The texted-`help` path is honest in the same way: it stops the robot and says it cannot
call anyone, because it cannot.

## Run it with no hardware

```bash
pip install -r requirements.txt
CORGI_MOCK=1 ./scripts/run_server.sh
```

Open <http://localhost:8000> and text it. There is no phone in the loop — the page posts
to the same handler a real message hits — so the whole thing demos on a laptop with
nothing plugged in and no API keys. Out of the box it uses the keyword router and prints
outgoing texts to the console.

Try, in order:

```
bring me my water bottle          the full fetch: find, grasp, stow, return, hand over
come here                         drives to the person and stops about a metre short
walk with me                      comes over, then the walk buttons do something
stop                              cancels whatever is running
get me the thing on the counter   asks which one, and queues nothing
```

<http://localhost:8000/ops> is the view to keep open while building: live camera, phase
log, the router's decision for every message (which model, which tier, how long, whether
it escalated), the message log including the texts the robot chose *not* to send, walker
state with the dead-man countdown, manual jog, e-stop, and the simulator's ground truth.
When the servo loop converges somewhere wrong, the ground-truth panel says immediately
whether vision or control is the one lying to you.

```bash
python scripts/smoke.py                 # one order, printing every phase
python scripts/smoke.py 20              # 20 orders, prints your real success rate
python scripts/smoke.py 20 --fail 0.3   # sabotage 30% of grasps on purpose
python scripts/smoke.py --via text      # go in through iMessage instead of the API
python scripts/smoke_walk.py            # prove the dead-man actually stops the wheels
python scripts/check_router.py          # what the router can reach, and what it decides
pytest                                  # the whole suite, no hardware, no network
```

`smoke.py 20` is the one that matters. It is how you find out your actual success rate
instead of extrapolating from the last run you happened to watch.

## Texting it for real (Photon)

The bridge is its own scaffolded project, not part of this repo (`corgi/` is
gitignored, the same way `.venv/` is). It holds the live connection to Spectrum Cloud
and does both directions in one small process — no webhook, no ngrok:

```
phone <-> Photon <-> corgi/ (bun) <-> FastAPI  (both directions, one process each way)
```

Scaffold it once, from the repo root:

```bash
bun create spectrum-project@latest corgi --projectId <your-project-id> --providers imessage --yes
```

That authenticates to your Photon account, writes `corgi/.env` with the real
`PROJECT_ID`/`PROJECT_SECRET`, and installs `spectrum-ts`. `corgi/src/index.ts` is
already wired to this robot: it reads Spectrum's own inbound stream and posts each text
to `POST /api/imessage/relay` on the Python server, and it runs a small HTTP server on
`:8787` that `POST /send` — which is what `PhotonMessenger` in `robot/messaging.py`
calls whenever the robot wants to reply.

Run it alongside the robot:

```bash
cd corgi && bun start
```

Then set `CORGI_MESSAGING_BACKEND=photon` on the robot. Two things worth knowing:
recipients must be phone numbers (Apple ID emails are not supported), and
`CORGI_BRIDGE_SECRET`, if you set it on both sides, is a second lock on the relay
endpoint — the bridge only ever binds loopback, so it is defense in depth, not the door.

`POST /api/imessage/webhook` on the Python side still exists if you would rather
register a public Photon Cloud webhook (needs ngrok and the signing secret,
`CORGI_PHOTON_WEBHOOK_SECRET`) instead of running this bridge — the two are
alternatives, not both at once.

If the bridge will not start, `CORGI_MESSAGING_BACKEND=applescript` sends through the
Messages app signed in on this Mac — no account, no keys, genuinely iMessage. It cannot
receive, so inbound still needs the bridge, the webhook, or the browser page.

## The router (Merge Gateway)

Turning "can you get me my pills, oh and the remote" into a typed intent is the one part
of this that wants a model. Merge Gateway is the router, through the official SDK
(`pip install merge-gateway-python`, already in requirements.txt):

```bash
MERGE_API_KEY=... CORGI_ROUTER_BACKEND=merge python scripts/check_router.py
```

That prints what your key can actually serve, flags whether the model ids in
`robot/config.py` are among them, then routes ten real messages and shows the decision
for each — which model, which vendor actually served it, how long, whether it escalated.

If the `merge_gateway` package is ever missing, `make_router()` degrades to the keyword
router with a boot note rather than failing to start — the same as a missing API key.
There is also a plain-REST OpenAI-compatible path (`CORGI_MERGE_API=openai`) for anyone
who would rather not add the SDK dependency at all.

Two tiers, because most messages are two words: the fast model reads everything, and only
a low-confidence or unparseable answer costs a call to the deep one. Every failure —
timeout, 401, malformed JSON, an intent that is not in the list — degrades to the keyword
router rather than dropping the person's message. And after any model answer, the
keywords get a second look at `stop` and `help`: missing a "stop" from someone who needs
the robot to stop is the worst thing this system can do, so a model that misses one gets
overruled.

Set `CORGI_ROUTER_BACKEND=keyword` (the default) and the whole thing runs offline.

## Going to hardware

1. Flash `firmware/drivebase/drivebase.ino` to the Arduino. `python
   scripts/check_devices.py` lists the serial ports and cameras the Mac can see.
2. Start without `CORGI_MOCK`. The drive base is autodetected; a missing camera or arm
   degrades the demo rather than stopping the server, and `/api/health` says what came
   up and what didn't.
3. Calibrate the two numbers that matter: `python scripts/calibrate.py water bottle`,
   then set `CORGI_SWEET_SPOT_X` and `CORGI_SWEET_SPOT_H` in `.env`.
4. Bolt on the arm and set `CORGI_ARM_ENABLED=1`. Without it the robot runs in
   drive-only mode: it finds the item and escorts you to it instead of lifting it, and
   the rest of the run — the phases, the delivery leg, what the person is told — is
   identical.
5. Hand-pose the four new keyframes in `robot/poses.py` (`STOW`, `STOW_OPEN`, `UNSTOW`,
   `HANDLE`) and read them back off `GET /api/arm/state`. Do not compute them.
6. Switch perception on for the demo: `CORGI_VISION_BACKEND=vlm` plus the matching API
   key. The colour backend stays as the fallback for when the Wi-Fi is bad — note that
   it finds a person by looking for a blue shirt, so retune `person` in
   `HSV_PROFILES` to what they are actually wearing, or run the VLM.

Copy `.env.example` to `.env` for the full list of knobs. Every tunable in the codebase
lives in `robot/config.py` and nothing else hardcodes a constant.

## How it decides where to stop

The base is the positioner and the arm is a stamp. Nothing ever estimates the object's
3D pose. The robot drives until the object appears at one calibrated spot in the image
— `SWEET_SPOT_X` across, `SWEET_SPOT_H` tall — and then replays one canned grasp that
is known to work from exactly there. `SWEET_SPOT_H` is the only depth cue a single
camera gets, and it is enough.

Coming to a person is the same loop with a nearer target and a looser tolerance
(`COME_SWEET_SPOT_H`, `COME_TOL_H`): there is no canned motion waiting at the end, so it
only has to end up beside someone. Asked to find a person, a VLM will happily box an
armchair, so a detection is rejected unless it is taller than it is wide
(`CORGI_PERSON_MIN_ASPECT`) — one comparison, and it rules out driving up to the sofa.

Grasp failure is detected for free: after closing, if the gripper servo travelled past
`GRIPPER_EMPTY_DEG` the jaws are empty. The robot retries once a little further
forward, then asks you to nudge the object toward it.

The item then goes in the basket rather than being carried in the jaws the whole way
home, and comes back out only once the robot has arrived — an item held out for the
return leg is one collision away from being on the floor.

Every velocity command carries a duration and two independent watchdogs enforce it. The
host stops the wheels if the next command is late; the Arduino stops them if the host
stops talking altogether. Walker mode re-issues its command every `WALK_STEP_MS` for
exactly this reason: one long command with a sleep would disarm both of them.

## How many texts one errand sends

Two. An elderly person must not get nine texts about one water bottle, so only
milestones text: the acknowledgement, and then arrival, delivery, a request for help, or
a failure. `SEARCHING`, `APPROACHING`, `ALIGNING`, `GRASPING`, `VERIFYING`, `STOWING`,
`RETURNING` and `UNSTOWING` are all silent — they show on the web page and in the ops
console, which is enough. On top of that the outbox dedupes by milestone, enforces a
quiet gap, and holds a daily cap; every text it decides *not* to send is recorded with
the reason, so the ops console shows the robot choosing to stay quiet.

## API

| Method | Path | What |
|---|---|---|
| GET | `/api/health` | what hardware came up, and why anything didn't |
| GET | `/api/state` | robot phase, basket, current order, walker state |
| POST | `/api/imessage/webhook` | inbound texts from a public Photon Cloud webhook; signature checked |
| POST | `/api/imessage/relay` | inbound texts pushed by the corgi/ bridge process |
| POST | `/api/imessage/simulate` | `{text}` — the same path with no phone in the loop |
| GET | `/api/imessage/log` | messages in, messages out, and the ones held back |
| GET | `/api/contacts` | who has texted, and what is remembered about them |
| POST | `/api/router/preview` | `{text}` — route it and report the decision, change nothing |
| POST | `/api/orders` | `{item}` — queue a fetch-and-deliver |
| GET | `/api/orders` · `/api/orders/{id}` | queue and single-order status |
| POST | `/api/orders/{id}/cancel` | only while still queued |
| WS | `/api/events` | phase stream, `human_text` is spoken verbatim |
| GET | `/api/events/recent` | the last N phase events |
| GET | `/api/camera/frame.jpg` · `/api/camera/stream.mjpg` | what the robot sees |
| POST | `/api/skills/scene` · `/locate` · `/fetch` · `/come` · `/deliver` · `/return_item` | skills, direct |
| GET | `/api/tasks/{id}` | `{phase, done, ok, detail}` |
| POST | `/api/walker/start` · `/nudge` · `/hold` · `/stop` | walking alongside someone |
| GET | `/api/walker/state` | active, direction, dead-man ms remaining, session length |
| POST | `/api/drive/velocity` · `/api/drive/cmd` · `/api/drive/stop` | manual driving |
| POST | `/api/arm/pose` · `/api/arm/gripper` · GET `/api/arm/state` | manual arm, for calibration |
| POST | `/api/estop` | stop everything: walker, skill, wheels, and relax the arm |
| GET | `/api/debug/world` · POST `/api/debug/reset` | simulator ground truth, mock only |

Bounding boxes are always normalized `[x0, y0, x1, y1]` in `0..1`, origin top-left.

Intents: `fetch` · `come` · `walk` · `stop` · `status` · `help` · `chat`.

Phases: `QUEUED` · `SEARCHING` · `APPROACHING` · `ALIGNING` · `GRASPING` · `VERIFYING` ·
`STOWING` · `NEEDS_HELP` · `RETURNING` · `UNSTOWING` · `PRESENTING` · `REPLACING` ·
`CALLED` · `COMING` · `ARRIVED` · `WALKING` · `HOLDING` · `STANDING_BY` · `DONE` ·
`FAILED` · `CANCELLED`.
