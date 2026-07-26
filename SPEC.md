# Corgi — a robot you text

**The pivot.** Corgi was a grocery robot you ordered from on a web page. It is now an
in-home helper for someone who does not want to get up. You text it in plain English from
the Messages app you already use, and it does one of two things:

1. **"can you bring me my water bottle"** — it finds the item, picks it up, puts it in its
   basket, drives back to you, and hands it over.
2. **"come here"** / **"walk with me to the kitchen"** — it drives to you and then paces
   alongside you as a walking companion at your speed, with the arm held out as a
   handhold reference.

Two sponsor APIs carry the new surface:

* **Photon** (Spectrum) is the iMessage transport. Inbound texts arrive as signed
  webhooks; replies go back to the same conversation.
* **Merge Gateway** is the LLM router. It turns free text into one typed intent, choosing
  a cheap fast model for the 90% of messages that are two words and escalating to a
  stronger model only when the fast one is unsure.

Everything still runs as one process on a Mac, and everything still runs with nothing
plugged in (`CORGI_MOCK=1`).

---

## What this document is

This is the implementation contract. Every signature below is normative: if you are
implementing one module, the other modules will call yours exactly as written here. Do not
change a public name, argument order, or return type without changing this file.

Non-negotiable house rules, inherited from the existing codebase:

* Every tunable lives in `robot/config.py`. Nothing else hardcodes a constant.
* Two backends behind one interface, chosen by config, with the offline one as the
  default — the way `vision.py`, `arm.py` and `drive.py` already do it. The whole demo
  must work with no API keys and no network.
* Nothing raises on the way up. A missing key, a dead sidecar, a 500 from a router: note
  it in `BOOT_NOTES` / degrade to the offline backend, do not stop the server booting.
* `human_text` is dialogue, not log lines. Short, plain, competent, no exclamation marks,
  no emoji, no "I'd be happy to". An 80-year-old is reading it.
* Hardware calls go out to a worker thread via `body`. The skill loop never blocks the
  event loop.

## Honesty constraints — read before writing walker mode

The base is two continuous-rotation hobby servos on 7.4V. **It cannot bear a person's
weight and the code must never imply that it can.** Walker mode is a *paced escort*: the
robot travels at walking speed beside the person and holds the arm out as a handhold
*reference* so they have something at a known height to steady a hand against. It is not a
mobility aid, not a medical device, and not load-bearing.

Concretely, this means:

* No string anywhere — text, UI, README, docstring — may say "lean on", "support your
  weight", "hold you up", or "walker" without qualification. Use "walk with you", "keep
  pace", "steady yourself".
* Walker mode is dead-man operated. The wheels move only while a fresh instruction keeps
  arriving. Silence stops the base within `WALK_DEADMAN_MS`.
* The texted-`help` path must say what it actually did. It did not call anyone. It must not
  imply it did.

---

## 1. `robot/config.py` — additions

Append these blocks. Keep the existing style (`_flag`, `_num`, `_int`, and a comment
saying why the number is what it is). Also add a `_csv` helper:

```python
def _csv(name: str, default: str = "") -> list[str]:
    return [p.strip() for p in os.getenv(name, default).split(",") if p.strip()]
```

```python
# --- Messaging (Photon / iMessage) ----------------------------------------
# log        = print and record, no network. The default, and what the tests use.
# photon     = Photon Spectrum, via the Node sidecar in bridge/ (outbound only;
#              inbound arrives on the webhook below).
# applescript = local Messages.app on this Mac via osascript. No account, no keys,
#              genuinely iMessage; the fallback when the sidecar is not running.
MESSAGING_BACKEND = os.getenv("CORGI_MESSAGING_BACKEND", "log")
PHOTON_BRIDGE_URL = os.getenv("CORGI_PHOTON_BRIDGE_URL", "http://127.0.0.1:8787")
PHOTON_BRIDGE_TIMEOUT_S = _num("CORGI_PHOTON_BRIDGE_TIMEOUT_S", 6.0)
# Spectrum webhook signing secret, shown once when the webhook is registered.
PHOTON_WEBHOOK_SECRET = os.getenv("CORGI_PHOTON_WEBHOOK_SECRET", "")
# Refuse unsigned webhooks. Only turn this off to poke at the endpoint by hand.
PHOTON_REQUIRE_SIGNATURE = _flag("CORGI_PHOTON_REQUIRE_SIGNATURE", "1")
PHOTON_WEBHOOK_TOLERANCE_S = _int("CORGI_PHOTON_WEBHOOK_TOLERANCE_S", 300)

# Blank = answer anyone. Otherwise an allowlist of E.164 numbers; anything else gets
# one polite refusal and is then ignored.
ALLOWED_SENDERS = _csv("CORGI_ALLOWED_SENDERS")
# Photon's default quota is 5000 messages/server/day. We are nowhere near it, but an
# elderly user must never get nine texts about one water bottle, so cap it hard.
DAILY_MESSAGE_BUDGET = _int("CORGI_DAILY_MESSAGE_BUDGET", 200)
TEXT_MIN_GAP_S = _num("CORGI_TEXT_MIN_GAP_S", 6.0)
# Allow the browser phone simulator to inject messages. On by default in mock.
ALLOW_SIMULATED_TEXTS = _flag("CORGI_ALLOW_SIMULATED_TEXTS", "1")

# --- Router (Merge Gateway) -----------------------------------------------
# keyword = deterministic, offline, and the fallback for every merge failure.
# merge   = Merge Gateway, which is itself routing across providers underneath us.
ROUTER_BACKEND = os.getenv("CORGI_ROUTER_BACKEND", "keyword")
MERGE_BASE_URL = os.getenv("CORGI_MERGE_BASE_URL", "https://api-gateway.merge.dev/v1")
# responses = Merge's native API, which reports back which vendor and tier served the
#             call -- that is the interesting part, so it is the default.
# openai    = the OpenAI-compatible shim at {base}/openai/chat/completions.
MERGE_API = os.getenv("CORGI_MERGE_API", "responses")
MERGE_API_KEY = os.getenv("MERGE_API_KEY", "")
# Two tiers. Short commands ("water please") never need more than the fast one.
ROUTER_FAST_MODEL = os.getenv("CORGI_ROUTER_FAST_MODEL", "anthropic/claude-haiku-4-5-20251001")
ROUTER_DEEP_MODEL = os.getenv("CORGI_ROUTER_DEEP_MODEL", "anthropic/claude-sonnet-5")
ROUTER_FAST_TIER = os.getenv("CORGI_ROUTER_FAST_TIER", "flex")      # merge service_tier
ROUTER_DEEP_TIER = os.getenv("CORGI_ROUTER_DEEP_TIER", "standard")
ROUTER_TIMEOUT_S = _num("CORGI_ROUTER_TIMEOUT_S", 8.0)
# Below this the fast model's answer is not trusted and the deep model gets a turn.
ROUTER_CONFIDENCE_FLOOR = _num("CORGI_ROUTER_CONFIDENCE_FLOOR", 0.65)
ROUTER_MAX_CHARS = _int("CORGI_ROUTER_MAX_CHARS", 600)

# --- Coming to a person ---------------------------------------------------
# Stop further out than a grasp: this ends up next to someone, not against them.
COME_SWEET_SPOT_H = _num("CORGI_COME_SWEET_SPOT_H", 0.62)
COME_TOL_H = _num("CORGI_COME_TOL_H", 0.06)
PERSON_LABEL = os.getenv("CORGI_PERSON_LABEL", "person")
# A person's bbox is taller than it is wide. A chair's is not. Cheapest possible
# guard against a VLM confidently boxing the furniture.
PERSON_MIN_ASPECT = _num("CORGI_PERSON_MIN_ASPECT", 1.15)

# --- Walking alongside someone -------------------------------------------
# NOT a mobility aid and NOT load-bearing: the base weighs a couple of kilos. This is
# a paced escort with a handhold reference at a known height. Half speed, because the
# point is to be predictable next to someone unsteady.
WALK_LINEAR = _num("CORGI_WALK_LINEAR", 0.12)          # m/s
WALK_ANGULAR = _num("CORGI_WALK_ANGULAR", 0.5)         # rad/s
WALK_STEP_MS = _int("CORGI_WALK_STEP_MS", 250)
# Dead-man: the wheels move only while instructions keep arriving. Miss one and stop.
WALK_DEADMAN_MS = _int("CORGI_WALK_DEADMAN_MS", 1200)
# And the whole mode times out, so a forgotten session does not idle with the arm out.
WALK_MAX_S = _num("CORGI_WALK_MAX_S", 300.0)
# Pause after this many steps and re-look for the person before continuing.
WALK_RECHECK_STEPS = _int("CORGI_WALK_RECHECK_STEPS", 12)

# --- Basket ---------------------------------------------------------------
# What the robot can carry aboard at once. The arm stows into it and lifts back out.
BASKET_CAPACITY = _int("CORGI_BASKET_CAPACITY", 3)
```

`ROUTER_FAST_MODEL` / `ROUTER_DEEP_MODEL` are guesses at Merge's catalogue ids and must be
overridable without a code change — `scripts/check_router.py` (below) lists what the key
can actually reach.

---

## 2. `robot/messaging.py` — new

Transport only. It knows nothing about robots.

```python
@dataclass
class InboundMessage:
    id: str
    sender: str                  # E.164 where available, else the opaque Spectrum id
    space_id: str                # the conversation to reply into
    text: str
    platform: str = "iMessage"
    at: float = field(default_factory=time.time)
    simulated: bool = False

    def as_dict(self) -> dict: ...


@dataclass
class SentMessage:
    space_id: str
    text: str
    at: float
    backend: str
    ok: bool
    detail: str = ""

    def as_dict(self) -> dict: ...
```

### Senders

```python
class Messenger:
    name: str
    async def send(self, space_id: str, text: str, *, to: str | None = None) -> None
        """Raise on failure. The Outbox decides what a failure means."""
    async def aclose(self) -> None: ...


class LogMessenger(Messenger):
    """name = "log". Prints "[imessage -> {space_id}] {text}" and returns."""


class PhotonMessenger(Messenger):
    """name = "photon". POST {PHOTON_BRIDGE_URL}/send with
    {"spaceId": space_id, "text": text, "to": to} and a JSON body reply
    {"ok": true}. Non-2xx or ok=false raises RuntimeError with the body attached.
    Holds one httpx.AsyncClient with PHOTON_BRIDGE_TIMEOUT_S."""


class AppleScriptMessenger(Messenger):
    """name = "applescript". Sends through the Messages app on this Mac:

        osascript -e 'tell application "Messages"
            send <text> to buddy <to> of (1st service whose service type = iMessage)
        end tell'

    Run it with asyncio.create_subprocess_exec and NEVER build the script by string
    interpolation -- pass the text and the recipient as separate argv entries bound to
    `on run argv` so a message containing a quote cannot become AppleScript. Requires a
    phone number: raise ValueError if `to` is None. Non-zero exit raises RuntimeError
    with stderr."""


def make_messenger(backend: str = MESSAGING_BACKEND) -> tuple[Messenger, list[str]]:
    """Returns the messenger plus boot notes. Unknown backend, or `photon` with no
    bridge URL, degrades to LogMessenger with a note rather than raising."""
```

### Webhook

```python
class SignatureError(Exception): ...


def verify_spectrum_signature(
    raw_body: bytes,
    *,
    timestamp: str | None,
    signature: str | None,
    secret: str,
    tolerance_s: int = PHOTON_WEBHOOK_TOLERANCE_S,
    now: float | None = None,
) -> None:
    """Spectrum signs v0. Raises SignatureError on any problem.

    expected = "v0=" + hmac_sha256(secret, f"v0:{timestamp}:{raw_body.decode()}").hexdigest()

    Compare with hmac.compare_digest. Reject a missing header, a non-integer timestamp,
    a timestamp more than tolerance_s away from now (replay), and a mismatch. The
    headers are X-Spectrum-Timestamp and X-Spectrum-Signature."""


def parse_spectrum_webhook(payload: dict) -> InboundMessage | None:
    """Pull an inbound text out of a Spectrum `messages` envelope:

        {"event": "messages",
         "space":   {"id": "...", "platform": "iMessage", "type": "dm", "phone": "+1..."},
         "message": {"id": "spc-msg-...", "platform": "iMessage", "direction": "inbound",
                     "timestamp": "2026-07-25T18:03:11Z",
                     "sender": {"id": "+15551234567", "platform": "iMessage"},
                     "space":  {"id": "...", "platform": "iMessage", "type": "dm",
                                "phone": "+1..."},
                     "content": {"type": "text", "text": "bring me my water bottle"}}}

    Return None -- not an error -- for anything we do not act on: direction != "inbound",
    content.type != "text", empty text, or event != "messages". Tolerate a missing
    top-level `space` by falling back to message.space. Never raise on a shape you did
    not expect; return None."""
```

### Outbox

The one place that decides whether a text is actually worth sending.

```python
class Outbox:
    """Rate limit, dedupe, and a daily cap in front of a Messenger.

    An elderly user must not get a text per phase transition. Callers pass a `key` for
    anything that could fire more than once (usually f"{order_id}:{phase}"); the same
    key never sends twice.
    """

    def __init__(self, messenger: Messenger, *, min_gap_s=TEXT_MIN_GAP_S,
                 daily_budget=DAILY_MESSAGE_BUDGET, history_limit: int = 200) -> None

    async def send(self, space_id: str, text: str, *, to: str | None = None,
                   key: str | None = None, urgent: bool = False) -> bool:
        """False means deliberately dropped (duplicate key, inside min_gap_s, over
        budget) or the send failed; True means handed to the transport. `urgent=True`
        bypasses min_gap_s but not the daily budget or the dedupe key -- arrival,
        failure and NEEDS_HELP are urgent, progress chatter is not.

        Every attempt appends a SentMessage to history, including the dropped ones with
        ok=False and detail saying why, because the ops console needs to show that the
        robot chose to stay quiet."""

    def recent(self, limit: int = 40) -> list[dict]
    def stats(self) -> dict   # {"backend", "sent", "dropped", "budget", "remaining"}
    async def aclose(self) -> None
```

The daily counter resets when `time.time()` crosses into a new UTC day. Track it as a
`(day_ordinal, count)` pair, not a background timer.

---

## 3. `robot/brain.py` — new

Free text in, one typed intent out. This is where Merge lives.

```python
INTENT_KINDS = ("fetch", "come", "walk", "stop", "status", "help", "chat")


@dataclass
class RouteInfo:
    """How the decision got made. The ops console shows this verbatim, because
    'which model handled this request' is the entire point of a router."""
    backend: str            # "keyword" | "merge"
    tier: str               # "fast" | "deep" | "none"
    model: str = ""         # what we asked for
    served_by: str = ""     # what Merge says actually served it (model or vendor)
    service_tier: str = ""  # what Merge says it billed
    latency_ms: int = 0
    escalated: bool = False
    fell_back: bool = False # merge was asked for and did not answer
    note: str = ""
    def as_dict(self) -> dict: ...


@dataclass
class Intent:
    kind: str                        # one of INTENT_KINDS; never anything else
    item: str | None = None          # fetch only, lowercased, stripped
    also: list[str] = field(default_factory=list)  # extra requests found in one text
    reply: str = ""                  # a clarifying question, or the chat answer
    confidence: float = 0.0
    needs_clarification: bool = False
    raw: str = ""
    route: RouteInfo = field(default_factory=lambda: RouteInfo("keyword", "none"))
    def as_dict(self) -> dict: ...


@dataclass
class RouterContext:
    """What the router is allowed to know about the robot right now."""
    busy: bool = False
    phase: str = "IDLE"
    carrying: str | None = None
    basket: list[str] = field(default_factory=list)
    known_items: list[str] = field(default_factory=list)  # what vision can name
    last_item: str | None = None       # for "another one" / "same again"
    walking: bool = False
    def as_prompt_block(self) -> str: ...
```

### Backends

```python
class Router:
    name: str
    async def route(self, text: str, ctx: RouterContext | None = None) -> Intent
    async def aclose(self) -> None


class KeywordRouter(Router):
    """name = "keyword". Deterministic, offline, and the fallback for every merge
    failure, so it has to be genuinely decent, not a stub.

    Order matters -- check stop first, it is the one that must never be misrouted:

      stop    "stop", "wait", "hold on", "never mind", "nevermind", "cancel", "stay",
              "that's enough", "quit", "halt"
      help    "help" as its own word, "emergency", "i've fallen", "i fell",
              "i need help", "call someone", "911"
      walk    "walk with me", "walk me", "let's walk", "take me to", "help me walk",
              "steady", "i want to go to"
      come    "come", "come here", "over here", "where are you", "i need you"
      status  "what are you doing", "status", "how long", "did you get", "are you there",
              "hows it going", "how's it going"
      fetch   an imperative verb near a noun: "bring", "get", "grab", "fetch", "hand",
              "pass", "i want", "i'd like", "can you get", "need", "where is my"
      chat    everything else

    Item extraction for fetch: strip a leading polite prefix ("can you", "could you",
    "please", "would you mind", "hey corgi"), strip the verb, strip a trailing
    politeness ("please", "thanks", "thank you", "for me", "if you don't mind"), strip
    leading articles and possessives ("the", "a", "my", "some"). "another one" / "the
    same" / "same again" resolves to ctx.last_item. Confidence 0.9 for a keyword hit,
    0.5 for a bare-noun message with no verb ("water bottle"), 0.2 for chat.

    Two requests in one text: split on " and ", "&", ",", "then". First becomes `item`,
    the rest go in `also` (deduped, max 3)."""


class MergeRouter(Router):
    """name = "merge". One tight prompt, JSON out, two tiers.

    route() does:
      1. Truncate text to ROUTER_MAX_CHARS.
      2. Ask ROUTER_FAST_MODEL at ROUTER_FAST_TIER.
      3. If the reply will not parse, or kind is not in INTENT_KINDS, or
         confidence < ROUTER_CONFIDENCE_FLOOR, ask ROUTER_DEEP_MODEL at
         ROUTER_DEEP_TIER and set route.escalated = True.
      4. On any exception, timeout, or still-unparseable deep reply: delegate to a
         KeywordRouter held as self._fallback, set route.fell_back = True, keep
         route.backend = "merge", and put the error in route.note. NEVER raise out of
         route(): a router outage must degrade to keywords, not drop the person's text.
      5. `stop` and `help` are safety-critical. After any LLM answer, run the
         KeywordRouter too; if it says stop or help and the model did not, take the
         keyword answer and note the override. Missing a "stop" is worse than a
         needless one.

    Transport, MERGE_API == "responses" (the default):
        POST {MERGE_BASE_URL}/responses
        Authorization: Bearer {MERGE_API_KEY}
        {"model": <tier model>,
         "service_tier": <tier>, "service_tier_fallback": true,
         "input": [{"type": "message", "role": "system", "content": SYSTEM_PROMPT},
                   {"type": "message", "role": "user",   "content": <text + ctx block>}]}

      Merge reports back `model`, `vendor` and `service_tier`; copy them into
      RouteInfo.served_by / service_tier. Extract the text tolerantly, in this order:
        - response["output_text"] if a non-empty string
        - the first non-empty {"type": "output_text"|"text", "text": ...} found by
          walking output[*].content[*]
        - response["choices"][0]["message"]["content"]
      Raise ValueError if none of them yield text -- step 4 catches it.

    Transport, MERGE_API == "openai":
        POST {MERGE_BASE_URL}/openai/chat/completions with the ordinary OpenAI body
        ({"model", "messages", "temperature": 0,
          "response_format": {"type": "json_object"}}), read
        choices[0].message.content, and take served_by from response["model"].

    Both paths reuse one httpx.AsyncClient at ROUTER_TIMEOUT_S."""


def make_router(backend: str = ROUTER_BACKEND) -> tuple[Router, list[str]]:
    """`merge` with no MERGE_API_KEY degrades to KeywordRouter with a boot note."""
```

### The prompt

One tight prompt, in the module, as a constant. It must:

* State the fixed intent list and that `intent` must be exactly one of them.
* Demand JSON only: `{"intent","item","also","reply","confidence","needs_clarification"}`.
* Say the writer is texting an elderly person: `reply` is at most two short sentences,
  plain words, no emoji, no exclamation marks, no "I'd be happy to".
* Say `reply` is only for `chat` and for `needs_clarification`; leave it empty otherwise,
  because every other reply is a pre-written template that stays in sync with the motion.
* Carry `ctx.as_prompt_block()`: what the robot is doing, what is in the basket, what
  vision can name, the last item.
* Give four worked examples, including one rambling two-request message and one where the
  right answer is `needs_clarification` ("the thing on the counter").

---

## 4. `robot/concierge.py` — new

The glue: a text arrives, an intent comes back, the robot does something, the person gets
told. This is the only module that knows about both `brain` and `skills`.

```python
@dataclass
class Contact:
    phone: str
    space_id: str
    first_seen: float
    last_seen: float
    messages: int = 0
    last_item: str | None = None
    last_intent: str | None = None
    pending_clarification: str | None = None   # the text we asked a question about
    allowed: bool = True
    def as_dict(self) -> dict: ...
```

Remember only what makes the next message better: the conversation id, the last item, the
last intent, and one outstanding clarification. Do **not** accumulate a transcript, and do
not keep anything a person said about their health.

```python
class Concierge:
    def __init__(self, *, router: Router, outbox: Outbox, orders: OrderService,
                 skills: Skills | None, walker: WalkerMode | None, bus: EventBus) -> None

    async def handle(self, msg: InboundMessage) -> dict:
        """The whole inbound path. Returns a dict for the API and the ops log:
        {"intent": {...}, "action": str, "reply": str, "sent": bool}.

        Steps:
          1. Allowlist. If ALLOWED_SENDERS is non-empty and msg.sender is not in it, send
             REFUSE_UNKNOWN once (key f"refuse:{sender}") and return action="refused".
          2. Upsert the Contact; bump messages; last_seen.
          3. If contact.pending_clarification is set, prepend it to the text so
             "the water bottle" answers "which one?" — then clear it.
          4. router.route(text, ctx built from skills/orders/walker state).
          5. Dispatch (below). Every branch composes its reply from a template in this
             module, except `chat` and `needs_clarification`, which use intent.reply
             (falling back to a template when it is empty).
          6. outbox.send(..., key=..., urgent=...) and return.

        Must never raise: wrap dispatch in try/except, text FAILED_GENERIC, and log."""

    async def follow_phases(self) -> None:
        """Long-running. Subscribes to the bus and texts MILESTONES ONLY."""

    def contacts(self) -> list[dict]
    def recent(self) -> list[dict]        # the last 40 handled messages, for ops
```

### Dispatch table

| intent   | what it does                                                                                          | reply template                                     | urgent |
|----------|-------------------------------------------------------------------------------------------------------|----------------------------------------------------|--------|
| `fetch`  | `orders.create(item)`. If `also` is non-empty, queue those too and say how many.                      | `ACCEPT_FETCH`                                     | no     |
| `come`   | `skills.come()`                                                                                        | `ACCEPT_COME`                                      | no     |
| `walk`   | `skills.come()`, then on success `walker.start()`                                                     | `ACCEPT_WALK`                                      | no     |
| `stop`   | `walker.stop()`, `skills.cancel_current()`, cancel queued orders                                       | `ACK_STOP`                                         | yes    |
| `status` | read `skills.state()` + `orders.current`                                                                | `STATUS_*`                                         | yes    |
| `help`   | `walker.stop()` + `skills.cancel_current()` + emit a `help_requested` event                            | `ACK_HELP`                                         | yes    |
| `chat`   | nothing                                                                                                | `intent.reply` or `CHAT_FALLBACK`                  | no     |

Refusals and edge cases:

* **Busy.** A `fetch` while an order is running still queues — that is what a queue is for
  — but the reply says so: `ACCEPT_FETCH_QUEUED`.
* **Basket full.** `len(skills.basket) >= BASKET_CAPACITY` → `REFUSE_BASKET_FULL`, no order.
* **No perception.** `skills is None` → `REFUSE_NO_CAMERA`.
* **Walking already.** `come`/`fetch` while `walker.active` → `REFUSE_WALKING`.
* **Clarification.** `intent.needs_clarification` → store the text in
  `contact.pending_clarification` and ask; no order is created.

### The strings

Module constants. These are the voice of the product; write them as speech.

```python
ACCEPT_FETCH        = "Okay. Going to get the {item} now."
ACCEPT_FETCH_QUEUED = "Okay, the {item} is next after the {current}."
ACCEPT_FETCH_MULTI  = "Okay. Getting the {item} first, then {rest}."
ACCEPT_COME         = "On my way to you."
ACCEPT_WALK         = "On my way. I'll walk with you when I get there."
ACK_STOP            = "Stopped."
ACK_HELP            = ("I've stopped and I'm staying put. I can't call anyone for you — "
                       "if this is an emergency, please call 911 or press your alert button.")
STATUS_IDLE         = "Nothing on right now. Text me what you need."
STATUS_BUSY         = "Still working on the {item}. {phase}"
STATUS_WALKING      = "Walking with you. Say stop when you want to finish."
STATUS_CARRYING     = "I have the {item} with me."
ASK_WHICH           = "Which one do you mean?"
DELIVERED           = "Here's the {item}."
ARRIVED             = "I'm here."
NEEDS_HELP          = "I can't get a grip on the {item}. Could you nudge it toward me?"
CANT_FIND           = "I couldn't find the {item}. Is it somewhere else?"
FAILED_GENERIC      = "Something went wrong on my end. Nothing is moving."
REFUSE_UNKNOWN      = "This robot only answers the person it's set up for."
REFUSE_BASKET_FULL  = "My basket is full. Let me bring you what I have first."
REFUSE_NO_CAMERA    = "I can't see anything right now, so I can't go get things."
REFUSE_WALKING      = "I'm walking with you at the moment. Say stop first."
CHAT_FALLBACK       = "I can bring you something, or come to you. What do you need?"
```

### Which phases text, and which stay silent

`follow_phases` subscribes to the bus and texts on exactly these, keyed
`f"{order_id or task_id}:{phase}"`:

| phase        | text          | urgent |
|--------------|---------------|--------|
| `PRESENTING` | `DELIVERED`   | yes    |
| `ARRIVED`    | `ARRIVED`     | yes    |
| `NEEDS_HELP` | `NEEDS_HELP`  | yes    |
| `FAILED`     | `CANT_FIND`   | yes    |
| `WALKING`    | `STATUS_WALKING` | no  |

Everything else — `SEARCHING`, `APPROACHING`, `ALIGNING`, `GRASPING`, `VERIFYING`,
`STOWING`, `RETURNING`, `UNSTOWING` — is **silent**. It shows in the web UI and the ops
console, and that is enough. One fetch produces at most two texts: the acknowledgement and
the outcome.

`follow_phases` needs the space to reply into. Keep a `_route: dict[str, tuple[str, str]]`
mapping order_id/task_id → (space_id, phone), populated in `handle`, and fall back to the
most recent contact when a phase event has no id we recognise.

---

## 5. `robot/poses.py` — additions

New keyframes, in `JOINT_NAMES` order, degrees. Recalibrate by hand-posing; do not compute.

```python
# folded back over the deck, above the basket, still holding
"STOW":      [0.0, -95.0, 115.0,  70.0, 0.0, _C],
# same place, jaws open -- the item drops the last centimetre into the basket
"STOW_OPEN": [0.0, -95.0, 115.0,  70.0, 0.0, _O],
# reaching down into the basket to take something back out
"UNSTOW":    [0.0, -70.0, 100.0,  85.0, 0.0, _O],
# held up and locked: something at a known height to steady a hand against while
# walking. A handhold reference, NOT a support -- see the honesty constraints.
"HANDLE":    [0.0, -95.0,  35.0, -10.0, 0.0, _C],
```

`_REACH_APPLIES_TO` stays `{"PRE_GRASP", "DESCEND"}`. Reach offsets must not apply to any
of the four above.

---

## 6. `robot/events.py` — additions

New phases in both tables. `PHASE_PROGRESS` must be monotonic along each flow because
`orders.py` does `max(order.progress, progress(phase))` — a missing entry means 0.0 and the
bar appears to stall.

```python
"CALLED":      "on my way to you",
"COMING":      "coming to you",
"ARRIVED":     "I'm here",
"STOWING":     "putting the {label} in my basket",
"UNSTOWING":   "getting the {label} out of my basket",
"WALKING":     "walking with you",
"HOLDING":     "standing still",
"STANDING_BY": "standing by",
```

Progress: `CALLED` 0.05, `COMING` 0.4, `ARRIVED` 1.0, `STOWING` 0.75, `UNSTOWING` 0.9,
`WALKING` 0.5, `HOLDING` 0.5, `STANDING_BY` 0.0.

Note that the fetch flow now goes `... VERIFYING (0.7) → STOWING (0.75) → RETURNING (0.85)
→ UNSTOWING (0.9) → PRESENTING (0.95) → DONE (1.0)`. Check every number stays ascending.

---

## 7. `robot/skills.py` — changes

### Basket

`Skills` gains `self.basket: list[str] = []`. `carrying` keeps its meaning: what is in the
jaws *right now*. The basket is what is aboard.

`fetch()` gains a stow step after `VERIFYING` succeeds:

```
GRASPING → VERIFYING → STOWING → DONE
```

`STOWING`: `pose("STOW")`, `pose("STOW_OPEN")`, `gripper("open")`, append `label` to
`self.basket`, `carrying = None`, `pose("SCAN")`. With a `NullArm` this is all no-ops, so
guard on `self.body.arm.present`: without an arm, keep the existing escort behaviour and
record the item in the basket anyway so the rest of the flow is identical.

`deliver()` gains the mirror before `PRESENTING`:

```
RETURNING → UNSTOWING → PRESENTING → DONE
```

`UNSTOWING`: `pose("UNSTOW")`, `gripper("close")`, `pose("PRESENT")` — then the existing
`gripper("open")` hands it over. Pop the item from `self.basket` and set `carrying` across
the two steps so `/api/state` is truthful mid-move.

`state()` gains `"basket": list(self.basket)` and `"walking": bool`.

### `come`

```python
def come(self, order_id: str | None = None) -> Task:
    """Find the person and stop a comfortable distance short of them.

    SEARCHING(person) -> COMING -> ARRIVED, or FAILED if the person is never found.
    Uses _servo_to with target_h = COME_SWEET_SPOT_H and COME_TOL_H, so it stops
    further out than a grasp. Arm sits at SCAN the whole way.
    """
```

Person detection wraps `locate` with the aspect guard:

```python
async def _locate_person(self) -> Detection | None:
    det = await self.vision.locate(await self.body.frame(), PERSON_LABEL)
    if det is None:
        return None
    width = det.bbox[2] - det.bbox[0]
    if det.height < width * PERSON_MIN_ASPECT:
        return None          # too wide to be a standing person: probably furniture
    return det
```

`_servo_to` currently hardcodes `TOL_H`. Give it a `tol_h: float | None = None`
parameter defaulting to `TOL_H` so `come` can be looser, and pass `tol_x=TOL_X` likewise.
Do not change the existing call sites' behaviour.

### Everything else

Leave the search/servo/grasp core alone except where the audit says it is broken. The
confirmed audit fixes land in the same edit.

---

## 8. `robot/walker.py` — new

A long-running interactive mode, not a one-shot skill, so it does not live in `Skills`.

```python
class WalkerMode:
    """Paced escort. The robot travels at walking speed beside the person with the arm
    held out as a handhold reference. It is NOT load-bearing and no string in here may
    suggest otherwise.

    Dead-man operated: the wheels turn only while instructions keep arriving. One
    missed instruction and the base stops within WALK_DEADMAN_MS. That is what makes an
    open-loop hobby-servo base acceptable next to someone unsteady.
    """

    def __init__(self, body, bus: EventBus, skills: Skills | None = None) -> None

    @property
    def active(self) -> bool
    @property
    def moving(self) -> bool

    async def start(self, *, reason: str = "requested") -> bool:
        """Arm to HANDLE, wheels stopped, emit WALKING. Idempotent: starting while
        active just extends the session. Refuses (returns False) if a skill is running."""

    async def nudge(self, direction: str, ms: int | None = None) -> bool:
        """direction in ("forward", "back", "left", "right"). Extends the dead-man
        deadline by WALK_DEADMAN_MS and sets the current command. Returns False if not
        active or the direction is unknown. Speeds come from WALK_LINEAR / WALK_ANGULAR
        only -- walker mode never uses MAX_LINEAR."""

    async def hold(self) -> None:
        """Stop the wheels, stay in the mode, arm stays out. Emits HOLDING."""

    async def stop(self, *, reason: str = "done") -> None:
        """Leave the mode: stop the wheels, arm to HOME, emit STANDING_BY. Safe to call
        when not active."""

    def state(self) -> dict:
        # {"active", "moving", "direction", "linear", "angular",
        #  "deadman_ms_left", "session_s", "reason"}
```

The loop, started by `start()` and cancelled by `stop()`:

* Every `WALK_STEP_MS`, if the dead-man deadline is in the future and a direction is set,
  issue `body.drive(linear, angular, ms=WALK_STEP_MS)`. Re-issuing every step is what keeps
  both the host watchdog and the Arduino's 1500ms failsafe armed — never send one long
  command and sleep.
* When the deadline passes: `body.stop()`, clear the direction, emit `HOLDING` once.
* After `WALK_MAX_S`: `await self.stop(reason="timed out")`.
* Every `WALK_RECHECK_STEPS` steps, if `skills` is available, re-look for the person. Lost
  → `hold()` and emit `HOLDING` with `human_text` "I've lost sight of you, I'll wait here".
  Do **not** hunt for them while someone may be holding on.
* Wrap the body in try/except: any exception stops the wheels and exits the mode.

`body.estop()` must end walker mode. Wire that in `server.py`, not here.

---

## 9. `robot/world.py` — simulator changes

* `SimWorld` gains `basket: list[str]` and a `person` object at a fixed spot
  (`SimObject("person", (120, 160, 210), -0.30, 0.55, radius_m=0.16)` — radius large
  enough that its bbox in the render is tall, so the aspect guard passes).
  `reset()` seeds it along`home_tag`.
* `stow()` — the currently held object leaves the world and joins `basket`.
  `unstow(label)` — the reverse, back into the jaws.
* `render()` shows `basket: a, b` under the existing `holding:` banner.
* `snapshot()` gains `"basket"`. Take `_lock` in `snapshot()` (it currently reads without
  it) and compute `relative()` once per object rather than twice.
* Keep `home_tag`: the return leg still needs it.

The person is a *tall* blob, so `ColorVision` needs an HSV profile for `"person"` matching
its BGR colour, otherwise `come` can never work in mock. Add it to `HSV_PROFILES` in
`vision.py` and verify the hue by hand from the BGR triple.

---

## 10. `robot/server.py` — changes

Module scope gains, after `orders`:

```python
messenger, MESSAGING_NOTES = make_messenger()
outbox = Outbox(messenger)
router, ROUTER_NOTES = make_router()
walker = WalkerMode(body=body, bus=bus, skills=skills)
concierge = Concierge(router=router, outbox=outbox, orders=orders,
                      skills=skills, walker=walker, bus=bus)
BOOT_NOTES.extend(MESSAGING_NOTES + ROUTER_NOTES)
```

`lifespan` additionally starts `concierge.follow_phases()` as a task, keeps a strong
reference to it, and on shutdown cancels it and awaits `outbox.aclose()`,
`router.aclose()`, `walker.stop()`.

### New endpoints

| Method | Path | Body / result |
|---|---|---|
| POST | `/api/imessage/webhook` | Raw Spectrum envelope. Verify the signature against `PHOTON_WEBHOOK_SECRET`, 401 on `SignatureError`. Parse; a `None` parse returns `{"ignored": true}` with **200** (a webhook we do not act on is not an error, and Spectrum will retry a non-2xx). Then `await concierge.handle(msg)`. |
| POST | `/api/imessage/simulate` | `{"text": str, "from": str = "+15550000000", "space_id": str = "sim"}`. 403 unless `ALLOW_SIMULATED_TEXTS`. Same handler, `simulated=True`. This is what the browser phone and the tests use. |
| GET | `/api/imessage/log` | `{"inbound": [...], "outbound": outbox.recent(), "stats": outbox.stats()}` |
| GET | `/api/contacts` | `concierge.contacts()` |
| POST | `/api/router/preview` | `{"text": str}` → `intent.as_dict()`. **No side effects** — it routes and returns, nothing is dispatched. The single best demo of Merge in the whole app. |
| POST | `/api/skills/come` | → `Task.as_dict()` |
| POST | `/api/walker/start` | `{}` → `{"ok": bool, "state": ...}` |
| POST | `/api/walker/nudge` | `{"direction": str, "ms": int \| None}` → `{"ok", "state"}` |
| POST | `/api/walker/hold` | → `{"ok", "state"}` |
| POST | `/api/walker/stop` | → `{"ok", "state"}` |
| GET | `/api/walker/state` | `walker.state()` |

Signature verification needs the **raw** body, so the webhook handler takes
`request: Request` and calls `await request.body()` before parsing JSON. Do not use a
pydantic model for it.

`/api/estop` additionally `await walker.stop(reason="estop")` **before** cancelling skills.

`/api/health` gains `"messaging": outbox.stats()`, `"router": {"backend": router.name}`,
`"walker": walker.state()`, and `"basket": skills.basket if skills else []`.

`/api/state` gains `"walker": walker.state()`.

---

## 11. `bridge/` — new, Node sidecar for outbound Photon sends

Photon's Spectrum SDK is TypeScript and there is no documented REST send endpoint, so
outbound goes through a ~60-line sidecar. Inbound does **not** need it: that is the webhook.

* `bridge/photon-bridge.mjs` — `spectrum-ts` client built from `SPECTRUM_PROJECT_ID` /
  `SPECTRUM_PROJECT_SECRET`, plus a `node:http` server on `PORT` (default 8787) exposing:
  * `POST /send` `{spaceId, text, to}` → `{ok: true}`; resolve the space by id, else by
    phone, and `await space.send(text)`.
  * `GET /health` → `{ok, project, spaces}`.
  Log every send on one line. Exit non-zero with a clear message if the credentials are
  missing.
* `bridge/package.json` — `{"type": "module"}`, dependency `spectrum-ts`, `"start"` script.
* `bridge/README.md` — how to get a projectId/secret, register the webhook with
  `curl -u "$PROJECT_ID:$PROJECT_SECRET" -X POST
  https://spectrum.photon.codes/projects/$PROJECT_ID/webhooks/ -d '{"webhookUrl":"..."}'`,
  where the `signingSecret` in that response goes (`CORGI_PHOTON_WEBHOOK_SECRET`), and the
  ngrok line for exposing port 8000. Note that Apple ID emails are not supported —
  recipients must be phone numbers.

`node_modules/` is already gitignored.

---

## 12. `web/` — changes

### `web/index.html` + `app.js` + `styles.css` — the resident page

This is now a **phone**. The demo is: type a text, watch the robot move, watch the reply
come back.

* A message thread, styled as iMessage bubbles: outgoing right, incoming left. Populated
  from `GET /api/imessage/log` on load, then live from the WebSocket and the send response.
* A single text input + send button that POSTs `/api/imessage/simulate`.
* Below it: the live camera, the current phase in large plain type, and the basket contents.
* Walker controls: four large hold-to-move buttons that repeatedly POST `/api/walker/nudge`
  while held (`pointerdown` → interval at ~`WALK_STEP_MS`, `pointerup`/`pointerleave`/
  `blur` → stop). Releasing must stop the repeat — that is the dead-man in the UI. A
  prominent stop button.
* Minimum 18px body text, ≥44px touch targets, visible focus rings, `aria-live="polite"`
  on the phase line and the thread. A caption under the walker controls stating in plain
  words that the robot walks with you and is not something to lean on.
* Keep it text-first: no user-supplied string may be interpolated into `innerHTML`. Build
  message bubbles with `createElement` + `textContent`.

### `web/ops.html` + `ops.js` + `ops.css` — the ops console

Add, keeping the existing dense monospace look:

* **Router panel** — the last N decisions: text, intent, item, confidence, backend, tier,
  requested model, `served_by`, `service_tier`, latency, and `escalated`/`fell_back` flags.
  Plus a preview box that POSTs `/api/router/preview` so you can prove the routing on stage
  without moving the robot.
* **Messaging panel** — inbound and outbound with the drop reasons visible, and
  `outbox.stats()`.
* **Walker panel** — active, direction, dead-man ms remaining, session seconds, plus
  start/hold/stop and the four nudges.
* **Basket** — what is aboard.
* Fix the `innerHTML` injection the audit finds in `line()`: build nodes and use
  `textContent` for anything server-supplied.

---

## 13. Tests — new `tests/`

`pytest` + `pytest-asyncio`, added to `requirements.txt` under a dev comment. Every test
must pass with no network and no hardware: set `CORGI_MOCK=1` in a session fixture before
importing `robot.*`.

* `tests/conftest.py` — sets the env, builds an `httpx.ASGITransport` client against
  `robot.server.app`, and a `reset_world` fixture.
* `tests/test_router_keyword.py` — a table of ~40 real messages → expected intent and item,
  including every misspelling and two-request case in the spec. Must cover: `stop` never
  losing to `fetch`, `help` never losing to `chat`, "another one" resolving against
  `ctx.last_item`.
* `tests/test_router_merge.py` — `MergeRouter` against a stubbed `httpx` transport:
  the happy path for both `MERGE_API` modes, a low-confidence answer escalating to the deep
  model, a malformed reply falling back to keywords with `fell_back=True`, a timeout doing
  the same, and the keyword safety override taking `stop` back from the model.
* `tests/test_messaging.py` — `verify_spectrum_signature` accepting a correctly signed body
  and rejecting a bad signature, a stale timestamp, and a missing header;
  `parse_spectrum_webhook` on a real envelope, on an outbound message, on a reaction, and
  on junk; `Outbox` dedupe by key, `min_gap_s`, the daily cap, and `urgent` bypassing the
  gap but not the cap.
* `tests/test_concierge.py` — with a `LogMessenger` and the mock robot: a fetch text creates
  an order and sends exactly one reply; a `stop` text cancels; an unknown sender gets one
  refusal and then silence; a clarification round-trip; a full fetch produces exactly two
  texts.
* `tests/test_walker.py` — `nudge` moves and the dead-man stops it without further nudges;
  `stop` parks the arm; `estop` ends the mode; speeds never exceed `WALK_LINEAR`.
* `tests/test_skills_basket.py` — a mock fetch ends with the item in `basket` and
  `carrying is None`; deliver empties it; `BASKET_CAPACITY` is enforced by the concierge.
* `tests/test_api.py` — `/api/health` shape, the webhook 401/200/`ignored` cases,
  `/api/router/preview` having no side effects, `/api/walker/*` round-trip.

---

## 14. Scripts

* `scripts/smoke.py` — keep the existing order path; add `--via text` which drives the run
  through `POST /api/imessage/simulate` instead of `POST /api/orders` and asserts a reply
  came back on `/api/imessage/log`. Fix the `runs=0` division.
* `scripts/smoke_walk.py` — new: come to the person, enter walker mode, nudge forward for
  two seconds, stop nudging, assert the base stopped on its own within
  `WALK_DEADMAN_MS + slack`, then `stop`. This is the test that proves the dead-man.
* `scripts/check_router.py` — new: `GET {MERGE_BASE_URL}/models` with the key and print
  what is reachable, then route a handful of sample texts through the configured router and
  print the intent plus the `RouteInfo`. This is how you find out the real model ids
  instead of guessing at them.
* `scripts/check_devices.py` — unchanged.

## 15. Docs

`README.md` is rewritten around the new product: what it is, the two things you can text
it, the honest statement of what walker mode is and is not, the unplugged quickstart, the
Photon setup (webhook + sidecar), the Merge setup (key + `CORGI_ROUTER_BACKEND=merge` +
`check_router.py`), the updated API table, the updated phase list, and the hardware notes
that are still true. `.env.example` grows every new `CORGI_*` with the same default as
`config.py` — the audit checks this, so keep them in sync.
