"""The Merge backend against a stubbed gateway.

No test here touches the network: `MergeRouter` is built with an `httpx.MockTransport`, so
the gateway is a function that records what was asked and hands back whatever reply the
test wants to see -- a good one, a reasoning block in front of the text, an unsure answer,
an intent that does not exist, garbage, a timeout, a 401.

Two properties are load-bearing and each has its own test. First, route() never raises: a
provider outage has to come out as a keyword answer with fell_back set, because dropping
an elderly person's text is not an option the system has. Second, the keywords get the
last word on stop and help, so a model that answers "chat" to the word "stop" is overruled
and the override is written into RouteInfo for the ops console to show.
"""

from __future__ import annotations

import json

import httpx
import pytest
import pytest_asyncio

import robot.brain as brain
from robot.brain import INTENT_KINDS, MergeRouter, RouterContext
from robot.config import (
    MERGE_BASE_URL,
    ROUTER_CONFIDENCE_FLOOR,
    ROUTER_DEEP_MODEL,
    ROUTER_DEEP_TIER,
    ROUTER_FAST_MODEL,
    ROUTER_FAST_TIER,
)

FETCH_TEXT = "can you bring me my water bottle please"
CTX = RouterContext(known_items=["water bottle", "banana"], last_item="banana")


def reply_json(**over: object) -> str:
    """One well-formed router reply, as the model would write it."""
    payload = {
        "intent": "fetch",
        "item": "water bottle",
        "also": [],
        "reply": "",
        "confidence": 0.97,
        "needs_clarification": False,
    }
    payload.update(over)
    return json.dumps(payload)


def responses_reply(
    text: str, *, model: str = "anthropic/claude-haiku-4-5", service_tier: str = "flex"
) -> httpx.Response:
    """A /responses body in the shape Merge documents: a message block with output_text."""
    return httpx.Response(
        200,
        json={
            "model": model,
            "service_tier": service_tier,
            "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
        },
    )


def openai_reply(text: str, *, model: str = "openai/gpt-4o-mini") -> httpx.Response:
    return httpx.Response(
        200, json={"model": model, "choices": [{"message": {"role": "assistant", "content": text}}]}
    )


class Gateway:
    """A stubbed Merge. Replies are served in order; the last one repeats, so a test that
    wants both tiers to fail only has to pass one bad reply."""

    def __init__(self, *replies: httpx.Response | Exception) -> None:
        self._replies = list(replies)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self._replies[min(len(self.requests), len(self._replies)) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply

    @property
    def calls(self) -> int:
        return len(self.requests)

    def body(self, index: int = 0) -> dict:
        return json.loads(self.requests[index].content)

    @property
    def models(self) -> list[str]:
        return [self.body(i)["model"] for i in range(self.calls)]


@pytest_asyncio.fixture
async def build():
    """Builds routers on a stub gateway and closes them, so no client leaks a socket."""
    built: list[MergeRouter] = []

    def make(gateway: Gateway) -> MergeRouter:
        router = MergeRouter(transport=httpx.MockTransport(gateway))
        built.append(router)
        return router

    yield make
    for router in built:
        await router.aclose()


@pytest.fixture(autouse=True)
def responses_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """MERGE_API is read from the module namespace at call time, so pin it per test."""
    monkeypatch.setattr(brain, "MERGE_API", "responses")
    monkeypatch.setattr(brain, "MERGE_API_KEY", "test-key")


# --------------------------------------------------------------------------
# happy paths
# --------------------------------------------------------------------------
async def test_responses_happy_path(build) -> None:
    gateway = Gateway(responses_reply(reply_json(), model="anthropic/claude-haiku-4-5"))
    intent = await build(gateway).route(FETCH_TEXT, CTX)

    assert intent.kind == "fetch"
    assert intent.item == "water bottle"
    assert intent.confidence == pytest.approx(0.97)
    assert intent.raw == FETCH_TEXT
    assert intent.reply == ""  # the concierge owns the wording for a fetch

    route = intent.route
    assert (route.backend, route.tier) == ("merge", "fast")
    assert route.model == ROUTER_FAST_MODEL
    assert route.served_by == "anthropic/claude-haiku-4-5"
    assert route.service_tier == "flex"
    assert route.escalated is False
    assert route.fell_back is False
    assert route.note == ""

    assert gateway.calls == 1
    request = gateway.requests[0]
    assert str(request.url) == f"{MERGE_BASE_URL}/responses"
    assert request.headers["authorization"] == "Bearer test-key"
    body = gateway.body()
    assert body["model"] == ROUTER_FAST_MODEL
    assert body["service_tier"] == ROUTER_FAST_TIER
    assert body["service_tier_fallback"] is True
    assert [message["role"] for message in body["input"]] == ["system", "user"]
    # The robot's own state travels with the message, or "another one" cannot resolve.
    assert FETCH_TEXT in body["input"][1]["content"]
    assert "last item asked for: banana" in body["input"][1]["content"]


async def test_openai_happy_path(build, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(brain, "MERGE_API", "openai")
    gateway = Gateway(openai_reply(reply_json(item="banana"), model="openai/gpt-4o-mini"))
    intent = await build(gateway).route("bring me the banana", CTX)

    assert (intent.kind, intent.item) == ("fetch", "banana")
    assert intent.route.backend == "merge"
    assert intent.route.model == ROUTER_FAST_MODEL
    assert intent.route.served_by == "openai/gpt-4o-mini"
    assert intent.route.fell_back is False

    assert str(gateway.requests[0].url) == f"{MERGE_BASE_URL}/openai/chat/completions"
    body = gateway.body()
    assert body["temperature"] == 0
    assert body["response_format"] == {"type": "json_object"}
    assert [message["role"] for message in body["messages"]] == ["system", "user"]


async def test_reasoning_block_before_the_text_still_parses(build) -> None:
    """A thinking model puts its reasoning first. Reading output[0] blindly would take
    the reasoning as the answer and fall back on every single call."""
    gateway = Gateway(
        httpx.Response(
            200,
            json={
                "model": "anthropic/claude-sonnet-5",
                "service_tier": "standard",
                "output": [
                    {
                        "type": "reasoning",
                        "content": [{"type": "reasoning_text", "text": "she wants the bottle"}],
                    },
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": reply_json()}],
                    },
                ],
            },
        )
    )
    intent = await build(gateway).route(FETCH_TEXT, CTX)
    assert (intent.kind, intent.item) == ("fetch", "water bottle")
    assert intent.route.fell_back is False
    assert intent.route.served_by == "anthropic/claude-sonnet-5"


async def test_chat_reply_comes_through_tidied(build) -> None:
    gateway = Gateway(
        responses_reply(
            reply_json(
                intent="chat",
                item=None,
                reply="No, I am a robot with wheels and a basket!",
                confidence=0.9,
            )
        )
    )
    intent = await build(gateway).route("are you a real dog", CTX)
    assert intent.kind == "chat"
    assert intent.reply == "No, I am a robot with wheels and a basket."
    assert "!" not in intent.reply


# --------------------------------------------------------------------------
# escalation
# --------------------------------------------------------------------------
async def test_low_confidence_escalates_to_the_deep_model(build) -> None:
    unsure = responses_reply(reply_json(confidence=ROUTER_CONFIDENCE_FLOOR - 0.1))
    sure = responses_reply(
        reply_json(confidence=0.95), model="anthropic/claude-sonnet-5", service_tier="standard"
    )
    gateway = Gateway(unsure, sure)
    intent = await build(gateway).route(FETCH_TEXT, CTX)

    assert gateway.calls == 2, "an unsure fast answer must be asked again, once"
    assert gateway.models == [ROUTER_FAST_MODEL, ROUTER_DEEP_MODEL]
    assert gateway.body(1)["service_tier"] == ROUTER_DEEP_TIER

    assert (intent.kind, intent.item) == ("fetch", "water bottle")
    assert intent.confidence == pytest.approx(0.95)
    assert intent.route.escalated is True
    assert intent.route.fell_back is False
    assert intent.route.tier == "deep"
    assert intent.route.model == ROUTER_DEEP_MODEL
    assert intent.route.served_by == "anthropic/claude-sonnet-5"
    assert intent.route.service_tier == "standard"
    assert "escalated" in intent.route.note


async def test_a_confident_answer_costs_one_call(build) -> None:
    gateway = Gateway(responses_reply(reply_json(confidence=ROUTER_CONFIDENCE_FLOOR + 0.01)))
    intent = await build(gateway).route(FETCH_TEXT, CTX)
    assert gateway.calls == 1
    assert intent.route.escalated is False
    assert intent.route.tier == "fast"


async def test_unknown_intent_escalates_then_falls_back(build) -> None:
    """"dance" is not an intent this robot has. A stronger model usually fixes that, so
    it is asked once; when it repeats itself the keywords answer."""
    gateway = Gateway(responses_reply(reply_json(intent="dance")))
    intent = await build(gateway).route(FETCH_TEXT, CTX)

    assert gateway.calls == 2
    assert gateway.models == [ROUTER_FAST_MODEL, ROUTER_DEEP_MODEL]
    assert intent.kind in INTENT_KINDS
    assert (intent.kind, intent.item) == ("fetch", "water bottle")
    assert intent.route.backend == "merge"
    assert intent.route.fell_back is True
    assert "dance" in intent.route.note


# --------------------------------------------------------------------------
# every way the gateway can let us down
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("name", "reply"),
    (
        ("body is not json", httpx.Response(200, text="<html>gateway error</html>")),
        ("no json in the text", responses_reply("sorry, I cannot help with that")),
        ("text is not an object", responses_reply("[1, 2, 3]")),
        ("no text anywhere", httpx.Response(200, json={"model": "m", "output": []})),
        ("truncated json", responses_reply('{"intent": "fetch", "item": "water bot')),
        ("null intent", responses_reply(reply_json(intent=None))),
    ),
)
async def test_malformed_reply_falls_back_to_keywords(build, name: str, reply) -> None:
    gateway = Gateway(reply)
    intent = await build(gateway).route(FETCH_TEXT, CTX)

    assert (intent.kind, intent.item) == ("fetch", "water bottle"), name
    assert intent.route.backend == "merge"
    assert intent.route.tier == "none"
    assert intent.route.fell_back is True
    assert intent.route.note.startswith("keywords answered instead")
    assert intent.confidence > 0
    # A reply that will not parse is worth one stronger opinion before giving up.
    assert gateway.calls == 2, name


async def test_timeout_falls_back_to_keywords(build) -> None:
    gateway = Gateway(httpx.TimeoutException("the gateway took too long"))
    intent = await build(gateway).route("bring me the banana", CTX)

    assert (intent.kind, intent.item) == ("fetch", "banana")
    assert intent.route.fell_back is True
    assert intent.route.backend == "merge"
    assert "Timeout" in intent.route.note
    # A timeout is an outage, not a bad answer, so there is nothing to escalate to.
    assert gateway.calls == 1


async def test_connect_error_falls_back_to_keywords(build) -> None:
    gateway = Gateway(httpx.ConnectError("no route to the gateway"))
    intent = await build(gateway).route("come here", CTX)
    assert intent.kind == "come"
    assert intent.route.fell_back is True


async def test_401_falls_back_to_keywords(build) -> None:
    """A wrong or expired key must not take the robot down with it."""
    gateway = Gateway(httpx.Response(401, json={"error": "invalid api key"}))
    intent = await build(gateway).route(FETCH_TEXT, CTX)

    assert (intent.kind, intent.item) == ("fetch", "water bottle")
    assert intent.route.fell_back is True
    assert "401" in intent.route.note
    assert gateway.calls == 1


async def test_500_falls_back_to_keywords(build) -> None:
    gateway = Gateway(httpx.Response(500, text="upstream on fire"))
    intent = await build(gateway).route("what are you doing", CTX)
    assert intent.kind == "status"
    assert intent.route.fell_back is True


async def test_a_dead_gateway_still_hears_stop(build) -> None:
    """The worst case: no model, and the person wants the robot to stop."""
    gateway = Gateway(httpx.TimeoutException("gone"))
    intent = await build(gateway).route("stop", CTX)
    assert intent.kind == "stop"
    assert intent.route.fell_back is True


# --------------------------------------------------------------------------
# the safety override
# --------------------------------------------------------------------------
@pytest.mark.parametrize("text", ("stop", "STOP", "never mind", "that's enough"))
async def test_keywords_take_stop_back_from_the_model(build, text: str) -> None:
    gateway = Gateway(
        responses_reply(
            reply_json(intent="chat", item=None, reply="All right.", confidence=0.99)
        )
    )
    intent = await build(gateway).route(text, CTX)

    assert intent.kind == "stop", "a model that mishears stop must not be believed"
    assert intent.item is None
    assert intent.raw == text
    # The console has to be able to say the model was overruled and why.
    assert "keyword override" in intent.route.note
    assert "model said chat" in intent.route.note
    assert intent.route.backend == "merge"
    assert intent.route.fell_back is False


async def test_keywords_take_help_back_from_the_model(build) -> None:
    gateway = Gateway(responses_reply(reply_json(intent="status", item=None, confidence=0.99)))
    intent = await build(gateway).route("ive fallen and i cant get up", CTX)
    assert intent.kind == "help"
    assert "keyword override" in intent.route.note


async def test_no_override_when_the_model_agrees(build) -> None:
    gateway = Gateway(responses_reply(reply_json(intent="stop", item=None, confidence=0.99)))
    intent = await build(gateway).route("stop please", CTX)
    assert intent.kind == "stop"
    assert intent.route.note == ""


async def test_a_fetch_is_left_alone(build) -> None:
    """The override only ever fires for stop and help; a good fetch keeps the model's
    item, not the keyword router's guess."""
    gateway = Gateway(responses_reply(reply_json(item="water bottle", also=["banana"])))
    intent = await build(gateway).route("the bottle and the banana please", CTX)
    assert (intent.kind, intent.item, intent.also) == ("fetch", "water bottle", ["banana"])
    assert intent.route.note == ""


# --------------------------------------------------------------------------
# the shape of what comes out
# --------------------------------------------------------------------------
async def test_long_text_is_clipped_before_it_is_sent(build) -> None:
    gateway = Gateway(responses_reply(reply_json()))
    rambling = "bring me the water bottle " + ("and talk to me about the garden " * 60)
    intent = await build(gateway).route(rambling, CTX)

    sent = gateway.body()["input"][1]["content"]
    assert len(sent) < len(rambling)
    # The full text is kept for the log even though the model saw less of it.
    assert intent.raw == rambling


async def test_needs_clarification_carries_the_question_and_no_item(build) -> None:
    gateway = Gateway(
        responses_reply(
            reply_json(
                item="thing",
                reply="Which one do you mean?",
                confidence=0.9,
                needs_clarification=True,
            )
        )
    )
    intent = await build(gateway).route("get me the thing on the counter", CTX)
    assert intent.needs_clarification is True
    assert intent.item is None
    assert intent.reply == "Which one do you mean?"


async def test_as_dict_is_serialisable(build) -> None:
    gateway = Gateway(responses_reply(reply_json()))
    intent = await build(gateway).route(FETCH_TEXT, CTX)
    data = intent.as_dict()
    assert json.loads(json.dumps(data)) == data
    assert data["route"]["backend"] == "merge"
    assert data["kind"] in INTENT_KINDS
