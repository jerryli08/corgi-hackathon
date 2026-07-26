"""The concierge over the real robot: a text in, an order and one reply out.

Nothing here is a double except the transport. The router is the real KeywordRouter, the
queue is the real OrderService, the skills run against the mock body and the simulated
world, and walker mode is the real thing. What is counted is what the person's phone
would have shown, read off `Outbox.recent()`, because "the robot did the right thing but
texted about it four times" is the failure this module exists to prevent.

The load-bearing assertion in the file is `test_one_whole_fetch_produces_exactly_two_texts`.
It runs a complete errand -- search, approach, grasp, stow, drive back, hand over, a dozen
phase events -- and requires exactly two texts to have left the building.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import pytest
import pytest_asyncio

from robot import concierge as concierge_module
from robot.body import make_body
from robot.brain import KeywordRouter
from robot.concierge import (
    ACCEPT_FETCH,
    ACCEPT_FETCH_MULTI,
    ACK_HELP,
    ACK_STOP,
    ASK_WHICH,
    DELIVERED,
    REFUSE_BASKET_FULL,
    REFUSE_UNKNOWN,
    STATUS_IDLE,
    Concierge,
)
from robot.config import BASKET_CAPACITY
from robot.events import EventBus
from robot.messaging import InboundMessage, LogMessenger, Outbox
from robot.orders import Order, OrderService, OrderStatus
from robot.skills import Skills
from robot.vision import ColorVision
from robot.walker import WalkerMode

RESIDENT = "+15550000000"
STRANGER = "+15559999999"
SPACE = "sim"

# The arm interpolation is real, just not the 800ms the skills ask for: a full fetch
# does eight pose moves and the wall clock is the scarcest thing in this file.
POSE_MS = 40
# A whole errand in the simulator takes something like twenty seconds. This is the
# ceiling on that, not an expectation.
ERRAND_TIMEOUT_S = 90.0


@dataclass
class Rig:
    """A live concierge over the whole robot, plus the tasks it needs cleaning up."""

    concierge: Concierge
    orders: OrderService
    skills: Skills
    walker: WalkerMode
    outbox: Outbox
    bus: EventBus
    background: list[asyncio.Task] = field(default_factory=list)
    _n: int = 0

    async def text(self, body: str, sender: str = RESIDENT) -> dict:
        """One inbound message, exactly as the webhook or the simulator would build it."""
        self._n += 1
        return await self.concierge.handle(
            InboundMessage(
                id=f"m{self._n}",
                sender=sender,
                space_id=SPACE,
                text=body,
                simulated=True,
            )
        )

    def texts(self) -> list[str]:
        """What actually reached the phone, in order."""
        return [m["text"] for m in self.outbox.recent(limit=200) if m["ok"]]

    def follow(self) -> None:
        self.background.append(asyncio.create_task(self.concierge.follow_phases()))

    def run_queue(self, fulfill: Callable[[Order], Awaitable[tuple[bool, str]]]) -> None:
        self.orders.set_fulfill(fulfill)
        self.orders.start()

    def events(self, kind: str) -> list[dict]:
        return [e for e in self.bus.recent(limit=200) if e.get("type") == kind]


@pytest_asyncio.fixture
async def rig(world):
    """`world` guarantees a reset simulator with no random grasp failures."""
    body, _notes = make_body(mock=True)
    real_pose = body.arm.go_to_pose

    async def quick_pose(name: str, ms: int = POSE_MS, reach: str = "mid") -> list[float]:
        return await asyncio.to_thread(real_pose, name, POSE_MS, reach)

    body.pose = quick_pose  # type: ignore[method-assign]

    bus = EventBus()
    skills = Skills(body=body, vision=ColorVision(), bus=bus)
    orders = OrderService(bus=bus)
    walker = WalkerMode(body=body, bus=bus, skills=skills)
    # min_gap_s=0: the quiet gap belongs to test_messaging.py, and leaving it on here
    # would drop the second text of a two-message exchange and prove nothing.
    outbox = Outbox(LogMessenger(), min_gap_s=0.0)
    concierge = Concierge(
        router=KeywordRouter(),
        outbox=outbox,
        orders=orders,
        skills=skills,
        walker=walker,
        bus=bus,
    )

    rig = Rig(
        concierge=concierge,
        orders=orders,
        skills=skills,
        walker=walker,
        outbox=outbox,
        bus=bus,
    )
    yield rig

    for task in rig.background:
        task.cancel()
    for task in rig.background:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
    await walker.stop(reason="test over")
    await skills.cancel_current()
    await orders.stop()
    # Otherwise every test leaves a watchdog thread behind that still owns the world.
    body.close()


async def wait_until(predicate, limit_s: float = 5.0) -> bool:
    """Poll rather than sleep a fixed amount: the simulator's timings are not promises."""
    deadline = asyncio.get_running_loop().time() + limit_s
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.05)
    return False


async def slow_fulfill(order: Order) -> tuple[bool, str]:
    """A fetch that starts and never finishes, so `orders.current` stays put while the
    test asks the robot what it is doing. Cancelled by `orders.stop()`."""
    await asyncio.sleep(ERRAND_TIMEOUT_S)
    return True, "never gets here"


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------
async def test_a_fetch_text_makes_one_order_and_sends_one_reply(rig: Rig) -> None:
    result = await rig.text("can you bring me my water bottle")

    orders = rig.orders.list()
    assert len(orders) == 1
    assert orders[0].item == "water bottle"

    assert rig.texts() == [ACCEPT_FETCH.format(item="water bottle")]
    assert result["action"] == "queued"
    assert result["sent"] is True
    assert result["intent"]["kind"] == "fetch"
    assert result["intent"]["item"] == "water bottle"


async def test_two_requests_in_one_text_queue_both_and_the_reply_names_both(rig: Rig) -> None:
    result = await rig.text("bring me the water bottle and a banana")

    assert sorted(o.item for o in rig.orders.list()) == ["banana", "water bottle"]

    reply = result["reply"]
    assert reply == ACCEPT_FETCH_MULTI.format(item="water bottle", rest="the banana")
    assert "water bottle" in reply and "banana" in reply
    assert rig.texts() == [reply]


async def test_a_full_basket_refuses_and_creates_nothing(rig: Rig) -> None:
    rig.skills.basket = ["banana"] * BASKET_CAPACITY

    result = await rig.text("bring me the water bottle")

    assert result["reply"] == REFUSE_BASKET_FULL
    assert result["action"] == "refused"
    assert rig.orders.list() == []
    assert rig.texts() == [REFUSE_BASKET_FULL]


# --------------------------------------------------------------------------
# the clarification round trip
# --------------------------------------------------------------------------
async def test_an_ambiguous_text_asks_which_one_and_the_answer_completes_it(rig: Rig) -> None:
    first = await rig.text("can you get me that")

    assert first["reply"] == ASK_WHICH
    assert first["action"] == "asked"
    assert rig.orders.list() == [], "a question must never create an order"

    second = await rig.text("the water bottle")

    assert [o.item for o in rig.orders.list()] == ["water bottle"]
    assert second["reply"] == ACCEPT_FETCH.format(item="water bottle")
    assert rig.texts() == [ASK_WHICH, ACCEPT_FETCH.format(item="water bottle")]
    # And the question is spent: it must not be glued onto tonight's message too.
    assert rig.concierge.contacts()[0]["pending_clarification"] is None


async def test_a_stale_question_cannot_contaminate_a_later_message(rig: Rig) -> None:
    """An unanswered question dies with the next message, whatever that message is."""
    await rig.text("can you get me that")
    await rig.text("what are you doing")

    assert rig.concierge.contacts()[0]["pending_clarification"] is None

    await rig.text("bring me the banana")

    assert [o.item for o in rig.orders.list()] == ["banana"]


# --------------------------------------------------------------------------
# stop and help
# --------------------------------------------------------------------------
async def test_stop_cancels_the_running_skill_and_the_queue(rig: Rig) -> None:
    running = rig.skills.fetch("banana")
    rig.orders.create("water bottle")
    rig.orders.create("granola bar")
    assert await wait_until(lambda: rig.skills.phase != "IDLE")

    result = await rig.text("stop")

    assert result["reply"] == ACK_STOP
    assert running.done is True
    assert running.ok is False
    assert [o.status for o in rig.orders.list()] == [OrderStatus.CANCELLED] * 2
    assert rig.texts() == [ACK_STOP]


async def test_a_fall_stops_everything_and_promises_nothing_it_did_not_do(rig: Rig) -> None:
    running = rig.skills.fetch("banana")
    rig.orders.create("water bottle")
    assert await wait_until(lambda: rig.skills.phase != "IDLE")

    result = await rig.text("i've fallen")

    assert result["reply"] == ACK_HELP
    assert rig.texts() == [ACK_HELP]
    assert running.done is True
    assert rig.orders.list()[0].status is OrderStatus.CANCELLED
    assert rig.walker.active is False

    assert len(rig.events("help_requested")) == 1
    assert rig.events("help_requested")[0]["phone"] == RESIDENT

    # The honesty constraint, asserted rather than trusted: nobody was called and the
    # text must not suggest otherwise.
    said = result["reply"].lower()
    assert "i can't call anyone for you" in said
    for lie in ("i have called", "i've called", "contacted", "notified", "on the way", "alerted"):
        assert lie not in said


# --------------------------------------------------------------------------
# the allowlist
# --------------------------------------------------------------------------
async def test_an_unknown_sender_gets_one_refusal_and_then_silence(rig: Rig, monkeypatch) -> None:
    # Patched where the module reads it: concierge imported the name at import time.
    monkeypatch.setattr(concierge_module, "ALLOWED_SENDERS", [RESIDENT])

    first = await rig.text("bring me the water bottle", sender=STRANGER)
    second = await rig.text("hello? bring me the water bottle", sender=STRANGER)

    assert first["action"] == "refused"
    assert second["action"] == "refused"
    assert rig.texts() == [REFUSE_UNKNOWN], "the second message must be answered with silence"
    assert rig.orders.list() == []

    # And the person it is set up for is unaffected.
    await rig.text("bring me the water bottle")
    assert [o.item for o in rig.orders.list()] == ["water bottle"]
    assert rig.texts()[-1] == ACCEPT_FETCH.format(item="water bottle")


# --------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------
async def test_status_reads_differently_when_idle_and_when_working(rig: Rig) -> None:
    idle = await rig.text("what are you doing")
    assert idle["reply"] == STATUS_IDLE

    rig.run_queue(slow_fulfill)
    await rig.text("bring me the banana")
    assert await wait_until(lambda: rig.orders.current is not None)

    busy = await rig.text("what are you doing")

    assert busy["reply"] != idle["reply"]
    assert busy["reply"].startswith("Still working on the banana.")


# --------------------------------------------------------------------------
# milestones -- the reason this module exists
# --------------------------------------------------------------------------
async def test_one_whole_fetch_produces_exactly_two_texts(rig: Rig) -> None:
    """A complete errand: the acknowledgement and the outcome, and nothing between."""
    rig.follow()

    async def fulfill(order: Order) -> tuple[bool, str]:
        return await rig.skills.fetch_and_deliver(order.item, order_id=order.id)

    rig.run_queue(fulfill)

    await rig.text("can you bring me the granola bar")
    order = rig.orders.list()[0]

    assert await wait_until(
        lambda: order.status in (OrderStatus.DONE, OrderStatus.FAILED),
        limit_s=ERRAND_TIMEOUT_S,
    ), "the simulated errand never finished"
    assert order.status is OrderStatus.DONE, order.message
    # The PRESENTING text is emitted from inside the skill, so give the follower its turn.
    assert await wait_until(lambda: len(rig.texts()) >= 2)
    await asyncio.sleep(0.2)

    assert rig.texts() == [
        ACCEPT_FETCH.format(item="granola bar"),
        DELIVERED.format(item="granola bar"),
    ]

    # Every silent phase really did happen; they were filtered, not skipped.
    phases = {e["phase"] for e in rig.events("phase")}
    assert {"SEARCHING", "GRASPING", "STOWING", "RETURNING", "PRESENTING"} <= phases


async def test_progress_phases_are_never_texted(rig: Rig) -> None:
    """The same filter, without waiting twenty seconds for a robot to prove it."""
    rig.follow()
    inbound = InboundMessage(id="m0", sender=RESIDENT, space_id=SPACE, text="")
    rig.concierge._remember("o-1", inbound)

    for phase in ("SEARCHING", "APPROACHING", "ALIGNING", "GRASPING", "VERIFYING", "STOWING"):
        rig.bus.emit(
            {
                "type": "phase",
                "task_id": "t-1",
                "order_id": "o-1",
                "label": "banana",
                "phase": phase,
                "human_text": "",
                "progress": 0.5,
                "ok": True,
            }
        )
    await asyncio.sleep(0.2)
    assert rig.texts() == []

    # And a milestone re-emitted mid-servo still only texts once.
    for _ in range(3):
        rig.bus.emit(
            {
                "type": "phase",
                "task_id": "t-1",
                "order_id": "o-1",
                "label": "banana",
                "phase": "PRESENTING",
                "human_text": "",
                "progress": 0.95,
                "ok": True,
            }
        )
    assert await wait_until(lambda: len(rig.texts()) == 1)
    await asyncio.sleep(0.2)
    assert rig.texts() == [DELIVERED.format(item="banana")]


# --------------------------------------------------------------------------
# nothing raises on the way up
# --------------------------------------------------------------------------
async def test_a_router_that_explodes_still_gets_the_person_an_answer(rig: Rig) -> None:
    async def boom(text: str, ctx=None):
        raise RuntimeError("gateway on fire")

    rig.concierge.router.route = boom  # type: ignore[method-assign]

    result = await rig.text("stop")

    # The keywords are the last line of defence, and a stop is the one that has to work.
    assert result["intent"]["kind"] == "stop"
    assert result["reply"] == ACK_STOP
    assert rig.texts() == [ACK_STOP]


async def test_an_order_service_that_rejects_the_item_does_not_take_the_process_down(
    rig: Rig,
) -> None:
    def refuse(item: str) -> Order:
        raise ValueError("item is required")

    rig.orders.create = refuse  # type: ignore[method-assign]

    result = await rig.text("bring me the water bottle")

    assert result["action"] == "failed"
    assert result["reply"] == concierge_module.FAILED_GENERIC


async def test_a_messenger_that_fails_is_recorded_and_survived(rig: Rig) -> None:
    async def dead(space_id: str, text: str, *, to: str | None = None) -> None:
        raise RuntimeError("bridge is down")

    rig.outbox.messenger.send = dead  # type: ignore[method-assign]

    result = await rig.text("bring me the water bottle")

    assert result["sent"] is False
    assert rig.texts() == []
    assert [o.item for o in rig.orders.list()] == ["water bottle"], "the robot still goes"
    assert "bridge is down" in rig.outbox.recent()[-1]["detail"]


# --------------------------------------------------------------------------
# what the ops console reads
# --------------------------------------------------------------------------
async def test_contacts_and_recent_are_json_and_hold_no_transcript(rig: Rig) -> None:
    await rig.text("bring me the water bottle")
    await rig.text("what are you doing")

    contacts = rig.concierge.contacts()
    assert len(contacts) == 1
    assert contacts[0]["phone"] == RESIDENT
    assert contacts[0]["messages"] == 2
    assert contacts[0]["last_item"] == "water bottle"
    assert contacts[0]["last_intent"] == "status"

    recent = rig.concierge.recent()
    assert [e["intent"] for e in recent] == ["fetch", "status"]
    assert recent[0]["text"] == "bring me the water bottle"
    assert recent[0]["route"]["backend"] == "keyword"


async def test_walking_refuses_a_fetch_until_the_person_says_stop(rig: Rig) -> None:
    assert await rig.walker.start() is True

    refused = await rig.text("bring me the water bottle")
    assert refused["action"] == "refused"
    assert refused["reply"] == concierge_module.REFUSE_WALKING
    assert rig.orders.list() == []

    stopped = await rig.text("stop")
    assert stopped["reply"] == ACK_STOP
    assert rig.walker.active is False

    accepted = await rig.text("bring me the water bottle")
    assert accepted["action"] == "queued"


async def test_status_while_walking_says_so(rig: Rig) -> None:
    await rig.walker.start()
    result = await rig.text("what are you doing")
    assert result["reply"] == concierge_module.STATUS_WALKING


@pytest.mark.parametrize(
    "text",
    ["bring me the water bottle", "come here", "walk with me to the kitchen", "are you a dog"],
)
async def test_handle_never_raises_without_perception(text: str, rig: Rig) -> None:
    """A laptop with no camera is a supported configuration, not a crash."""
    rig.concierge.skills = None

    result = await rig.text(text)

    assert isinstance(result["reply"], str)
    assert result["action"] in ("refused", "chat")
