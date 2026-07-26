"""The concierge: the one module that knows about both the router and the robot.

A text arrives, `brain` turns it into one typed intent, something moves, and the person
is told. `brain` knows nothing about wheels and `skills` knows nothing about phones, so
this is the seam between them -- and keeping it the *only* seam is what stops the
robot's voice from ending up scattered across six files. Every sentence the person
reads is a constant in this module.

The load-bearing idea is restraint. One errand is worth exactly two texts: "going to
get it" and "here it is". The phase stream fires a dozen times for a single fetch, so
`follow_phases` filters it down to five milestones and keys each one on the order plus
the phase, because a servo loop re-entering a phase must not be able to text twice.
Everything else shows on the web page and in the ops console, and that is enough.

The second idea is that nothing in here is allowed to fail loudly. A router that times
out, an order service that rejects an item, a messenger that hangs: the person still
gets a reply, or at worst nothing happens quietly, and the process stays up.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from robot.brain import Intent, KeywordRouter, Router, RouterContext
from robot.config import (
    ALLOWED_SENDERS,
    BASKET_CAPACITY,
    PERSON_LABEL,
    ROUTER_MAX_CHARS,
    SINGLECAM_MISSION_ENABLED,
    SINGLECAM_MISSION_ITEM_KEYWORDS,
)
from robot.events import EventBus
from robot.messaging import InboundMessage, Outbox
from robot.orders import OrderService, OrderStatus
from robot.skills import Skills
from robot.vision import HSV_PROFILES, NOT_FETCHABLE
from robot.walker import WalkerMode

# --------------------------------------------------------------------------
# the strings -- the voice of the product. Write them as speech, not as log lines.
# --------------------------------------------------------------------------
ACCEPT_FETCH = "Okay. Going to get the {item} now."
ACCEPT_FETCH_QUEUED = "Okay, the {item} is next after the {current}."
ACCEPT_FETCH_MULTI = "Okay. Getting the {item} first, then {rest}."
ACCEPT_COME = "On my way to you."
ACCEPT_WALK = "On my way. I'll walk with you when I get there."
ACK_STOP = "Stopped."
ACK_HELP = (
    "I've stopped and I'm staying put. I can't call anyone for you — "
    "if this is an emergency, please call 911 or press your alert button."
)
STATUS_IDLE = "Nothing on right now. Text me what you need."
STATUS_BUSY = "Still working on the {item}. {phase}"
STATUS_WALKING = "Walking with you. Say stop when you want to finish."
STATUS_CARRYING = "I have the {item} with me."
ASK_WHICH = "Which one do you mean?"
DELIVERED = "Here's the {item}."
ARRIVED = "I'm here."
NEEDS_HELP = "I can't get a grip on the {item}. Could you nudge it toward me?"
CANT_FIND = "I couldn't find the {item}. Is it somewhere else?"
FAILED_GENERIC = "Something went wrong on my end. Nothing is moving."
REFUSE_UNKNOWN = "This robot only answers the person it's set up for."
REFUSE_BASKET_FULL = "My basket is full. Let me bring you what I have first."
REFUSE_NO_CAMERA = "I can't see anything right now, so I can't go get things."
REFUSE_WALKING = "I'm walking with you at the moment. Say stop first."
CHAT_FALLBACK = "I can bring you something, or come to you. What do you need?"

# The five phases worth a text, as phase -> (template, urgent). Everything else --
# SEARCHING, APPROACHING, ALIGNING, GRASPING, VERIFYING, STOWING, RETURNING,
# UNSTOWING -- is deliberately silent: it is progress, not news.
MILESTONES: dict[str, tuple[str, bool]] = {
    "PRESENTING": (DELIVERED, True),
    "ARRIVED": (ARRIVED, True),
    "NEEDS_HELP": (NEEDS_HELP, True),
    "FAILED": (CANT_FIND, True),
    "WALKING": (STATUS_WALKING, False),
}

# The colour backend's profile table is the only list of names the robot can actually
# find, so it doubles as the hint the router gets about what is fetchable here. With
# the VLM backend it is a hint and nothing more, which is all `known_items` ever is.
KNOWN_ITEMS: tuple[str, ...] = tuple(k for k in HSV_PROFILES if k not in NOT_FETCHABLE)

# One conversation, a handful of errands. Both maps are bounded anyway, because an
# unrecognised sender still gets a Contact and a webhook can arrive as often as the
# internet likes.
ROUTE_LIMIT = 64
CONTACT_LIMIT = 32
RECENT_LIMIT = 40


def _singlecam_can_fetch(item: str) -> bool:
    if not SINGLECAM_MISSION_ENABLED:
        return False
    text = item.strip().lower()
    return any(keyword.lower() in text for keyword in SINGLECAM_MISSION_ITEM_KEYWORDS)


@dataclass
class Contact:
    """Only what makes the next message better: where to reply, what was last asked
    for, and one outstanding question. No transcript, and nothing anybody said about
    their health."""

    phone: str
    space_id: str
    first_seen: float
    last_seen: float
    messages: int = 0
    last_item: str | None = None
    last_intent: str | None = None
    pending_clarification: str | None = None  # the text we asked a question about
    allowed: bool = True

    def as_dict(self) -> dict:
        return {
            "phone": self.phone,
            "space_id": self.space_id,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "messages": self.messages,
            "last_item": self.last_item,
            "last_intent": self.last_intent,
            "pending_clarification": self.pending_clarification,
            "allowed": self.allowed,
        }


@dataclass
class _Outcome:
    """What one dispatch decided: the ops log's word for it, and the text to send."""

    action: str
    reply: str = ""
    urgent: bool = False
    key: str | None = None


def _sentence(text: str) -> str:
    """Phase text is written as a fragment ("looking for the banana"). Inside a status
    reply it is the second half of something someone reads aloud, so it gets a capital
    and a full stop."""
    clean = (text or "").strip()
    if not clean:
        return ""
    if clean[-1] not in ".?":
        clean += "."
    return clean[0].upper() + clean[1:]


def _listed(names: list[str]) -> str:
    """"the banana", "the banana and the water bottle", "a, b and c"."""
    labels = [f"the {n}" for n in names]
    if len(labels) <= 1:
        return labels[0] if labels else ""
    return f"{', '.join(labels[:-1])} and {labels[-1]}"


class Concierge:
    def __init__(
        self,
        *,
        router: Router,
        outbox: Outbox,
        orders: OrderService,
        skills: Skills | None,
        walker: WalkerMode | None,
        bus: EventBus,
    ) -> None:
        self.router = router
        self.outbox = outbox
        self.orders = orders
        self.skills = skills
        self.walker = walker
        self.bus = bus
        self.last_error = ""

        self._contacts: dict[str, Contact] = {}
        # order_id / task_id -> (space_id, phone). This is what lets a phase event find
        # its way back to a conversation half a minute after the text that started it.
        self._route: dict[str, tuple[str, str]] = {}
        self._recent: list[dict] = []
        self._last_contact: Contact | None = None
        # Deliberately serialized. Two texts arriving together must not both read
        # "the basket has room" and both queue an order.
        self._lock = asyncio.Lock()
        # A "walk" waits for the drive to finish before the arm goes out, and that wait
        # cannot happen inside handle(): the webhook would sit open for the whole drive.
        self._background: set[asyncio.Task] = set()
        # Last resort under a router that raises. A "stop" has to work even then.
        self._fallback = KeywordRouter()

    # -- the inbound path ---------------------------------------------------
    async def handle(self, msg: InboundMessage) -> dict:
        """The whole inbound path: allowlist, intent, action, reply.

        Never raises. Whatever goes wrong downstream, the caller gets a dict and the
        person gets either a sentence or silence.
        """
        async with self._lock:
            try:
                return await self._handle(msg)
            except Exception as exc:
                # Belt and braces: _handle already guards dispatch. This catches the
                # bookkeeping around it, which must not be able to 500 a webhook.
                self.last_error = repr(exc)
                return {"intent": None, "action": "error", "reply": "", "sent": False}

    async def _handle(self, msg: InboundMessage) -> dict:
        contact = self._upsert(msg)
        if not contact.allowed:
            # Keyed on the sender, so an unknown number gets one refusal ever and is
            # then ignored however many times it writes.
            sent = await self.outbox.send(
                msg.space_id, REFUSE_UNKNOWN, to=msg.sender, key=f"refuse:{msg.sender}"
            )
            return self._log(msg, None, _Outcome("refused", REFUSE_UNKNOWN), sent)

        # An outstanding question is answered by the very next message and by nothing
        # else: prepend it, then clear it whatever the answer turns out to be, so an
        # hour-old "which one?" cannot contaminate tonight's text.
        text = msg.text
        if contact.pending_clarification:
            text = f"{contact.pending_clarification} {text}".strip()
        contact.pending_clarification = None

        intent = await self._route_text(text, contact)
        contact.last_intent = intent.kind

        try:
            outcome = await self._dispatch(intent, contact, msg)
        except Exception as exc:
            self.last_error = repr(exc)
            outcome = _Outcome("failed", FAILED_GENERIC, urgent=True)

        sent = False
        if outcome.reply:
            sent = await self.outbox.send(
                msg.space_id,
                outcome.reply,
                to=contact.phone,
                key=outcome.key,
                urgent=outcome.urgent,
            )
        return self._log(msg, intent, outcome, sent)

    async def _route_text(self, text: str, contact: Contact) -> Intent:
        """One intent, whatever happens. Both backends promise not to raise; this is
        here for the day one of them breaks that promise mid-demo."""
        ctx = self._context(contact)
        try:
            return await self.router.route(text, ctx)
        except Exception as exc:
            self.last_error = repr(exc)
            intent = await self._fallback.route(text, ctx)
            intent.route.fell_back = True
            intent.route.note = f"router raised, keywords answered: {exc!r}"[:160]
            return intent

    # -- dispatch -----------------------------------------------------------
    async def _dispatch(self, intent: Intent, contact: Contact, msg: InboundMessage) -> _Outcome:
        # stop and help come first and are never held up by anything, including a
        # question we were waiting on an answer to.
        if intent.kind == "stop":
            await self._halt()
            return _Outcome("stopped", ACK_STOP, urgent=True)

        if intent.kind == "help":
            await self._halt()
            self.bus.emit(
                {
                    "type": "help_requested",
                    "space_id": msg.space_id,
                    "phone": contact.phone,
                    "text": msg.text,
                    # Said out loud in the event too, so no consumer can grow a habit of
                    # assuming somebody was called. Nobody was.
                    "human_text": "asked for help by text; nobody has been contacted",
                }
            )
            return _Outcome("help", ACK_HELP, urgent=True)

        if intent.kind == "status":
            return _Outcome("status", self._status_reply(), urgent=True)

        if intent.needs_clarification:
            # The question is about this text, so this text is what the answer gets
            # prepended to. Clipped, because a rambling message plus its follow-up must
            # not grow past what the router will read.
            contact.pending_clarification = intent.raw[:ROUTER_MAX_CHARS] or msg.text
            return _Outcome("asked", intent.reply or ASK_WHICH)

        if intent.kind == "fetch":
            return self._fetch(intent, contact, msg)

        if intent.kind in ("come", "walk"):
            return self._come(intent.kind, msg)

        return _Outcome("chat", intent.reply or CHAT_FALLBACK)

    def _fetch(self, intent: Intent, contact: Contact, msg: InboundMessage) -> _Outcome:
        item = (intent.item or "").strip()
        if self.skills is None and not _singlecam_can_fetch(item):
            return _Outcome("refused", REFUSE_NO_CAMERA)
        if self.walker is not None and self.walker.active:
            return _Outcome("refused", REFUSE_WALKING)
        if self.skills is not None and len(self.skills.basket) >= BASKET_CAPACITY:
            return _Outcome("refused", REFUSE_BASKET_FULL)

        if not item:
            contact.pending_clarification = intent.raw[:ROUTER_MAX_CHARS] or msg.text
            return _Outcome("asked", ASK_WHICH)

        # Read before creating: `current` is the order already running, which is what
        # the person is waiting on and therefore what the reply has to name.
        current = self.orders.current
        try:
            order = self.orders.create(item)
        except Exception as exc:
            self.last_error = repr(exc)
            return _Outcome("failed", FAILED_GENERIC, urgent=True)
        self._remember(order.id, msg)

        rest: list[str] = []
        for extra in intent.also:
            name = (extra or "").strip()
            if not name or name == item or name in rest:
                continue
            try:
                self._remember(self.orders.create(name).id, msg)
            except Exception as exc:
                self.last_error = repr(exc)
                continue
            rest.append(name)

        contact.last_item = item
        if rest:
            reply = ACCEPT_FETCH_MULTI.format(item=item, rest=_listed(rest))
        elif current is not None:
            reply = ACCEPT_FETCH_QUEUED.format(item=item, current=current.item)
        else:
            reply = ACCEPT_FETCH.format(item=item)
        return _Outcome("queued", reply, key=f"{order.id}:accept")

    def _come(self, kind: str, msg: InboundMessage) -> _Outcome:
        if self.skills is None:
            # Both of these end with "drive to the person", and without eyes there is no
            # finding them. Saying so is better than setting off across the room.
            return _Outcome("refused", REFUSE_NO_CAMERA)
        if kind == "come" and self.walker is not None and self.walker.active:
            return _Outcome("refused", REFUSE_WALKING)

        # `come` is added to Skills by SPEC section 7. getattr keeps this module
        # importable and the rest of it working against a build that predates it.
        come = getattr(self.skills, "come", None)
        if come is None:
            return _Outcome("failed", FAILED_GENERIC, urgent=True)

        task = come()
        self._remember(task.id, msg)
        if kind == "walk" and self.walker is not None:
            self._spawn(self._walk_when_there(task))
            return _Outcome("walking", ACCEPT_WALK, key=f"{task.id}:accept")
        return _Outcome("coming", ACCEPT_COME, key=f"{task.id}:accept")

    async def _walk_when_there(self, task) -> None:
        """The arm only goes out once the robot is actually beside the person: a
        handhold reference held out across the room is just something to walk into."""
        try:
            if await task.wait() and self.walker is not None:
                await self.walker.start(reason="asked to walk with")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.last_error = repr(exc)

    async def _halt(self) -> None:
        """Everything the robot is doing, stopped, in the order that matters: wheels
        and arm first, then the queue that would start it all again."""
        if self.walker is not None:
            try:
                await self.walker.stop(reason="asked to stop")
            except Exception as exc:
                self.last_error = repr(exc)
        if self.skills is not None:
            try:
                await self.skills.cancel_current()
            except Exception as exc:
                self.last_error = repr(exc)
        if SINGLECAM_MISSION_ENABLED:
            try:
                from robot.singlecam_mission import singlecam_mission

                await singlecam_mission.stop()
            except Exception as exc:
                self.last_error = repr(exc)
        for order in self.orders.list():
            if order.status is OrderStatus.QUEUED:
                self.orders.cancel(order.id)

    def _status_reply(self) -> str:
        if self.walker is not None and self.walker.active:
            return STATUS_WALKING

        current = self.orders.current
        if current is not None:
            return STATUS_BUSY.format(item=current.item, phase=_sentence(current.message))

        if self.skills is not None:
            aboard = self.skills.carrying or (self.skills.basket[0] if self.skills.basket else None)
            if aboard:
                return STATUS_CARRYING.format(item=aboard)
        return STATUS_IDLE

    # -- milestones ---------------------------------------------------------
    async def follow_phases(self) -> None:
        """Long-running. Subscribes to the bus and texts milestones only."""
        queue = self.bus.subscribe()
        try:
            while True:
                event = await queue.get()
                try:
                    await self._on_event(event)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # A malformed event is not worth losing the follower over: it would
                    # cost every remaining milestone of the demo.
                    self.last_error = repr(exc)
        finally:
            self.bus.unsubscribe(queue)

    async def _on_event(self, event: dict) -> None:
        if event.get("type") != "phase":
            return
        milestone = MILESTONES.get(str(event.get("phase") or ""))
        if milestone is None:
            return

        template, urgent = milestone
        ident = str(event.get("order_id") or event.get("task_id") or "")
        target = self._route.get(ident)
        if target is None and self._last_contact is not None:
            # A phase from something nobody texted about -- the ops console pressed a
            # button. The person still deserves to be told, so it goes to whoever last
            # spoke to the robot.
            target = (self._last_contact.space_id, self._last_contact.phone)
        if target is None:
            return

        space_id, phone = target
        label = str(event.get("label") or "").strip() or "it"
        if label == PERSON_LABEL:
            label = "you"
        await self.outbox.send(
            space_id,
            template.format(item=label),
            to=phone,
            # The one thing this key has to prevent: a servo loop re-entering a phase
            # and texting the same milestone twice.
            key=f"{ident}:{event['phase']}",
            urgent=urgent,
        )

    # -- what the API reads -------------------------------------------------
    def contacts(self) -> list[dict]:
        return [c.as_dict() for c in sorted(self._contacts.values(), key=lambda c: -c.last_seen)]

    def recent(self) -> list[dict]:
        return list(self._recent)

    # -- internals ----------------------------------------------------------
    def _upsert(self, msg: InboundMessage) -> Contact:
        now = time.time()
        contact = self._contacts.get(msg.sender)
        if contact is None:
            contact = Contact(
                phone=msg.sender,
                space_id=msg.space_id,
                first_seen=now,
                last_seen=now,
                allowed=not ALLOWED_SENDERS or msg.sender in ALLOWED_SENDERS,
            )
            self._contacts[msg.sender] = contact
            self._prune(self._contacts, CONTACT_LIMIT)
        contact.space_id = msg.space_id or contact.space_id
        contact.last_seen = now
        contact.messages += 1
        if contact.allowed:
            # An unknown sender is never the fallback for a milestone text.
            self._last_contact = contact
        return contact

    def _remember(self, ident: str, msg: InboundMessage) -> None:
        self._route[ident] = (msg.space_id, msg.sender)
        self._prune(self._route, ROUTE_LIMIT)

    @staticmethod
    def _prune(table: dict, limit: int) -> None:
        # Insertion-ordered, so the oldest entries are simply the first ones.
        for key in list(table)[: max(0, len(table) - limit)]:
            table.pop(key, None)

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        # A task nobody holds is a task the garbage collector may take mid-drive.
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    def _context(self, contact: Contact) -> RouterContext:
        state = self.skills.state() if self.skills is not None else {}
        current = self.orders.current
        phase = str(state.get("phase") or "IDLE")
        return RouterContext(
            busy=current is not None or phase not in ("IDLE", "DONE"),
            phase=phase,
            carrying=state.get("carrying"),
            basket=list(state.get("basket") or []),
            known_items=list(KNOWN_ITEMS),
            last_item=contact.last_item or (current.item if current else None),
            walking=bool(self.walker is not None and self.walker.active),
        )

    def _log(
        self, msg: InboundMessage, intent: Intent | None, outcome: _Outcome, sent: bool
    ) -> dict:
        entry = {
            **msg.as_dict(),
            "intent": intent.kind if intent else None,
            "item": intent.item if intent else None,
            "action": outcome.action,
            "reply": outcome.reply,
            "sent": sent,
            "route": intent.route.as_dict() if intent else None,
        }
        self._recent.append(entry)
        del self._recent[: max(0, len(self._recent) - RECENT_LIMIT)]
        return {
            "intent": intent.as_dict() if intent else None,
            "action": outcome.action,
            "reply": outcome.reply,
            "sent": sent,
        }
