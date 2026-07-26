"""The Merge backend, against fakes standing in for the real dependency.

The native "responses" path goes through the official `merge_gateway` SDK
(`pip install merge-gateway-python`), whose client is synchronous with no seam of its
own for a custom transport -- so `MergeRouter` takes a `client=` object instead, and the
tests here hand it a small stand-in with the same `.responses.create(...)` shape,
returning real `merge_gateway.types.Response` objects (built with `Response.model_validate`,
so a wrong field name in a test fails loudly, the same as it would against the network).
No test touches the network.

The "openai" shim stays on plain httpx (no SDK involved for that path), so it keeps the
`httpx.MockTransport` seam it always had.

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
from merge_gateway import APIError, AuthenticationError
from merge_gateway.types import Response

import robot.brain as brain
from robot.brain import INTENT_KINDS, MergeRouter, RouterContext
from robot.config import (
    MERGE_BASE_URL,
    ROUTER_CONFIDENCE_FLOOR,
    ROUTER_DEEP_MODEL,
    ROUTER_FAST_MODEL,
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
    text: str,
    *,
    model: str = "anthropic/claude-haiku-4-5",
    vendor: str | None = None,
    selected_tier: int | None = None,
) -> Response:
    """A real merge_gateway Response, validated the same way the SDK builds one from a
    real HTTP body -- a wrong field name here fails the test, not the demo."""
    payload: dict = {
        "model": model,
        "vendor": vendor,
        "output": [{"content": [{"type": "text", "text": text}]}],
    }
    if selected_tier is not None:
        payload["routing"] = {"selected_tier": selected_tier, "vendor_used": vendor}
    return Response.model_validate(payload)


def openai_reply(text: str, *, model: str = "openai/gpt-4o-mini") -> httpx.Response:
    return httpx.Response(
        200, json={"model": model, "choices": [{"message": {"role": "assistant", "content": text}}]}
    )


class FakeResponses:
    """Stands in for `client.responses`. Replies are served in order; the last one
    repeats, so a test that wants both tiers to fail only has to pass one bad reply."""

    def __init__(self, *replies: Response | Exception) -> None:
        self._replies = list(replies)
        self.calls: list[dict] = []

    def create(self, **kwargs: object) -> Response:
        self.calls.append(kwargs)
        reply = self._replies[min(len(self.calls), len(self._replies)) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeClient:
    """Stands in for `merge_gateway.MergeGateway` -- same public shape
    (`.responses.create(...)`, `.close()`), no network underneath."""

    def __init__(self, *replies: Response | Exception) -> None:
        self.responses = FakeResponses(*replies)
        self.closed = False

    def close(self) -> None:
        self.closed = True

    @property
    def calls(self) -> int:
        return len(self.responses.calls)

    @property
    def models(self) -> list[str]:
        return [call["model"] for call in self.responses.calls]

    def kwargs(self, index: int = 0) -> dict:
        return self.responses.calls[index]


class Gateway:
    """A stubbed Merge, for the openai-compatible REST shim only."""

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


@pytest_asyncio.fixture
async def build():
    """Builds routers on a fake SDK client and closes them, so nothing leaks."""
    built: list[MergeRouter] = []

    def make(client: FakeClient) -> MergeRouter:
        router = MergeRouter(client=client)
        built.append(router)
        return router

    yield make
    for router in built:
        await router.aclose()


@pytest_asyncio.fixture
async def build_openai():
    """Builds routers on a stubbed httpx transport, for the OpenAI-compatible shim."""
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
    client = FakeClient(responses_reply(reply_json(), model="anthropic/claude-haiku-4-5"))
    intent = await build(client).route(FETCH_TEXT, CTX)

    assert intent.kind == "fetch"
    assert intent.item == "water bottle"
    assert intent.confidence == pytest.approx(0.97)
    assert intent.raw == FETCH_TEXT
    assert intent.reply == ""  # the concierge owns the wording for a fetch

    route = intent.route
    assert (route.backend, route.tier) == ("merge", "fast")
    assert route.model == ROUTER_FAST_MODEL
    assert route.served_by == "anthropic/claude-haiku-4-5"
    # No routing policy is configured on the account in this test, so Merge's own
    # routing metadata has nothing to report -- that is the honest default, not a bug.
    assert route.service_tier == ""
    assert route.escalated is False
    assert route.fell_back is False
    assert route.note == ""

    assert client.calls == 1
    kwargs = client.kwargs()
    assert kwargs["model"] == ROUTER_FAST_MODEL
    assert kwargs["include_routing_metadata"] is True
    assert "service_tier" not in kwargs  # not a real parameter -- must never be sent
    assert [message["role"] for message in kwargs["input"]] == ["system", "user"]
    # The robot's own state travels with the message, or "another one" cannot resolve.
    assert FETCH_TEXT in kwargs["input"][1]["content"]
    assert "last item asked for: banana" in kwargs["input"][1]["content"]


async def test_routing_metadata_is_reported_when_present(build) -> None:
    """When a routing policy IS configured, Merge's own tier and vendor choice show up
    in RouteInfo -- the ops console is the reason this router exists at all."""
    client = FakeClient(
        responses_reply(
            reply_json(), model="anthropic/claude-haiku-4-5", vendor="anthropic", selected_tier=2
        )
    )
    intent = await build(client).route(FETCH_TEXT, CTX)
    assert "anthropic" in intent.route.service_tier
    assert "tier 2" in intent.route.service_tier


async def test_openai_happy_path(build_openai, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(brain, "MERGE_API", "openai")
    gateway = Gateway(openai_reply(reply_json(item="banana"), model="openai/gpt-4o-mini"))
    intent = await build_openai(gateway).route("bring me the banana", CTX)

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
    """A thinking model puts its reasoning first, in the same message. Reading
    content[0] blindly would take the reasoning as the answer and fall back every time."""
    response = Response.model_validate(
        {
            "model": "anthropic/claude-sonnet-5",
            "output": [
                {
                    "content": [
                        {"type": "thinking", "thinking": "she wants the bottle"},
                        {"type": "text", "text": reply_json()},
                    ]
                }
            ],
        }
    )
    intent = await build(FakeClient(response)).route(FETCH_TEXT, CTX)
    assert (intent.kind, intent.item) == ("fetch", "water bottle")
    assert intent.route.fell_back is False
    assert intent.route.served_by == "anthropic/claude-sonnet-5"


async def test_chat_reply_comes_through_tidied(build) -> None:
    client = FakeClient(
        responses_reply(
            reply_json(
                intent="chat",
                item=None,
                reply="No, I am a robot with wheels and a basket!",
                confidence=0.9,
            )
        )
    )
    intent = await build(client).route("are you a real dog", CTX)
    assert intent.kind == "chat"
    assert intent.reply == "No, I am a robot with wheels and a basket."
    assert "!" not in intent.reply


# --------------------------------------------------------------------------
# escalation
# --------------------------------------------------------------------------
async def test_low_confidence_escalates_to_the_deep_model(build) -> None:
    unsure = responses_reply(reply_json(confidence=ROUTER_CONFIDENCE_FLOOR - 0.1))
    sure = responses_reply(reply_json(confidence=0.95), model="anthropic/claude-sonnet-5")
    client = FakeClient(unsure, sure)
    intent = await build(client).route(FETCH_TEXT, CTX)

    assert client.calls == 2, "an unsure fast answer must be asked again, once"
    assert client.models == [ROUTER_FAST_MODEL, ROUTER_DEEP_MODEL]

    assert (intent.kind, intent.item) == ("fetch", "water bottle")
    assert intent.confidence == pytest.approx(0.95)
    assert intent.route.escalated is True
    assert intent.route.fell_back is False
    assert intent.route.tier == "deep"
    assert intent.route.model == ROUTER_DEEP_MODEL
    assert intent.route.served_by == "anthropic/claude-sonnet-5"
    assert "escalated" in intent.route.note


async def test_a_confident_answer_costs_one_call(build) -> None:
    client = FakeClient(responses_reply(reply_json(confidence=ROUTER_CONFIDENCE_FLOOR + 0.01)))
    intent = await build(client).route(FETCH_TEXT, CTX)
    assert client.calls == 1
    assert intent.route.escalated is False
    assert intent.route.tier == "fast"


async def test_unknown_intent_escalates_then_falls_back(build) -> None:
    """"dance" is not an intent this robot has. A stronger model usually fixes that, so
    it is asked once; when it repeats itself the keywords answer."""
    client = FakeClient(responses_reply(reply_json(intent="dance")))
    intent = await build(client).route(FETCH_TEXT, CTX)

    assert client.calls == 2
    assert client.models == [ROUTER_FAST_MODEL, ROUTER_DEEP_MODEL]
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
        ("body is not json", ValueError("Expecting value: line 1 column 1 (char 0)")),
        ("no json in the text", responses_reply("sorry, I cannot help with that")),
        ("text is not an object", responses_reply("[1, 2, 3]")),
        ("no text anywhere", Response.model_validate({"model": "m", "output": []})),
        ("truncated json", responses_reply('{"intent": "fetch", "item": "water bot')),
        ("null intent", responses_reply(reply_json(intent=None))),
    ),
)
async def test_malformed_reply_falls_back_to_keywords(build, name: str, reply) -> None:
    client = FakeClient(reply)
    intent = await build(client).route(FETCH_TEXT, CTX)

    assert (intent.kind, intent.item) == ("fetch", "water bottle"), name
    assert intent.route.backend == "merge"
    assert intent.route.tier == "none"
    assert intent.route.fell_back is True
    assert intent.route.note.startswith("keywords answered instead")
    assert intent.confidence > 0
    # A reply that will not parse is worth one stronger opinion before giving up.
    assert client.calls == 2, name


async def test_timeout_falls_back_to_keywords(build) -> None:
    client = FakeClient(httpx.TimeoutException("the gateway took too long"))
    intent = await build(client).route("bring me the banana", CTX)

    assert (intent.kind, intent.item) == ("fetch", "banana")
    assert intent.route.fell_back is True
    assert intent.route.backend == "merge"
    assert "Timeout" in intent.route.note
    # A timeout is an outage, not a bad answer, so there is nothing to escalate to.
    assert client.calls == 1


async def test_connect_error_falls_back_to_keywords(build) -> None:
    client = FakeClient(httpx.ConnectError("no route to the gateway"))
    intent = await build(client).route("come here", CTX)
    assert intent.kind == "come"
    assert intent.route.fell_back is True


async def test_401_falls_back_to_keywords(build) -> None:
    """A wrong or expired key must not take the robot down with it."""
    client = FakeClient(AuthenticationError(body={"error": "invalid api key"}))
    intent = await build(client).route(FETCH_TEXT, CTX)

    assert (intent.kind, intent.item) == ("fetch", "water bottle")
    assert intent.route.fell_back is True
    assert "401" in intent.route.note or "Invalid API key" in intent.route.note
    assert client.calls == 1


async def test_500_falls_back_to_keywords(build) -> None:
    client = FakeClient(APIError("upstream on fire", status_code=500))
    intent = await build(client).route("what are you doing", CTX)
    assert intent.kind == "status"
    assert intent.route.fell_back is True


async def test_a_dead_gateway_still_hears_stop(build) -> None:
    """The worst case: no model, and the person wants the robot to stop."""
    client = FakeClient(httpx.TimeoutException("gone"))
    intent = await build(client).route("stop", CTX)
    assert intent.kind == "stop"
    assert intent.route.fell_back is True


async def test_the_package_missing_falls_back_to_keywords() -> None:
    """MergeRouter built with no client at all (the package genuinely is not
    installed) must degrade exactly like any other unusable reply."""
    router = MergeRouter(client=None)
    try:
        intent = await router.route("stop", CTX)
        assert intent.kind == "stop"
        assert intent.route.fell_back is True
        assert "not installed" in intent.route.note
    finally:
        await router.aclose()


# --------------------------------------------------------------------------
# the safety override
# --------------------------------------------------------------------------
@pytest.mark.parametrize("text", ("stop", "STOP", "never mind", "that's enough"))
async def test_keywords_take_stop_back_from_the_model(build, text: str) -> None:
    client = FakeClient(
        responses_reply(reply_json(intent="chat", item=None, reply="All right.", confidence=0.99))
    )
    intent = await build(client).route(text, CTX)

    assert intent.kind == "stop", "a model that mishears stop must not be believed"
    assert intent.item is None
    assert intent.raw == text
    # The console has to be able to say the model was overruled and why.
    assert "keyword override" in intent.route.note
    assert "model said chat" in intent.route.note
    assert intent.route.backend == "merge"
    assert intent.route.fell_back is False


async def test_keywords_take_help_back_from_the_model(build) -> None:
    client = FakeClient(responses_reply(reply_json(intent="status", item=None, confidence=0.99)))
    intent = await build(client).route("ive fallen and i cant get up", CTX)
    assert intent.kind == "help"
    assert "keyword override" in intent.route.note


async def test_no_override_when_the_model_agrees(build) -> None:
    client = FakeClient(responses_reply(reply_json(intent="stop", item=None, confidence=0.99)))
    intent = await build(client).route("stop please", CTX)
    assert intent.kind == "stop"
    assert intent.route.note == ""


async def test_a_fetch_is_left_alone(build) -> None:
    """The override only ever fires for stop and help; a good fetch keeps the model's
    item, not the keyword router's guess."""
    client = FakeClient(responses_reply(reply_json(item="water bottle", also=["banana"])))
    intent = await build(client).route("the bottle and the banana please", CTX)
    assert (intent.kind, intent.item, intent.also) == ("fetch", "water bottle", ["banana"])
    assert intent.route.note == ""


# --------------------------------------------------------------------------
# the shape of what comes out
# --------------------------------------------------------------------------
async def test_long_text_is_clipped_before_it_is_sent(build) -> None:
    client = FakeClient(responses_reply(reply_json()))
    rambling = "bring me the water bottle " + ("and talk to me about the garden " * 60)
    intent = await build(client).route(rambling, CTX)

    sent = client.kwargs()["input"][1]["content"]
    assert len(sent) < len(rambling)
    # The full text is kept for the log even though the model saw less of it.
    assert intent.raw == rambling


async def test_needs_clarification_carries_the_question_and_no_item(build) -> None:
    client = FakeClient(
        responses_reply(
            reply_json(
                item="thing",
                reply="Which one do you mean?",
                confidence=0.9,
                needs_clarification=True,
            )
        )
    )
    intent = await build(client).route("get me the thing on the counter", CTX)
    assert intent.needs_clarification is True
    assert intent.item is None
    assert intent.reply == "Which one do you mean?"


async def test_as_dict_is_serialisable(build) -> None:
    client = FakeClient(responses_reply(reply_json()))
    intent = await build(client).route(FETCH_TEXT, CTX)
    data = intent.as_dict()
    assert json.loads(json.dumps(data)) == data
    assert data["route"]["backend"] == "merge"
    assert data["kind"] in INTENT_KINDS


# --------------------------------------------------------------------------
# lifecycle
# --------------------------------------------------------------------------
async def test_aclose_closes_the_sdk_client() -> None:
    client = FakeClient(responses_reply(reply_json()))
    router = MergeRouter(client=client)
    await router.route(FETCH_TEXT, CTX)
    await router.aclose()
    assert client.closed is True
