"""The offline router, against real messages.

The keyword router is what answers when there is no key, no network, or a provider having
a bad day, so this file is the record of what the robot understands with nothing plugged
in. The table is the test: every row is a message somebody actually texts -- misspelled,
unpunctuated, shouted, wrapped in three layers of politeness -- and the expected intent
and item. Add a row when a real message goes wrong, and never loosen one to make the
suite green: a row that stops passing means someone changed what the robot hears.

The safety rows are the point of the whole exercise. Every stop-ish phrasing must land on
stop and every fall must land on help, while "that was helpful" must not.
"""

from __future__ import annotations

import pytest

from robot.brain import (
    CONF_BARE_NOUN,
    CONF_KEYWORD,
    INTENT_KINDS,
    KeywordRouter,
    RouterContext,
)

# (message, expected kind, expected item). item is None for everything but fetch, and for
# a fetch whose object cannot be named -- those rows are checked again further down.
CASES: tuple[tuple[str, str, str | None], ...] = (
    # --- fetch, the ordinary way people ask ---------------------------------
    ("can you bring me my water bottle please", "fetch", "water bottle"),
    ("Corgi, could you get the granola bar?", "fetch", "granola bar"),
    ("i'd like some strawberries thanks", "fetch", "strawberries"),
    ("i need my phone", "fetch", "phone"),
    ("grab my sweater", "fetch", "sweater"),
    ("pass me the tissues", "fetch", "tissues"),
    ("hand me the phone charger", "fetch", "phone charger"),
    ("pick up my newspaper", "fetch", "newspaper"),
    ("could i have my book", "fetch", "book"),
    ("i'll have the apple", "fetch", "apple"),
    ("do you have my hearing aid", "fetch", "hearing aid"),
    ("i could use a glass of water", "fetch", "glass of water"),
    # No question mark, no apostrophe: this is what the messages actually look like.
    ("wheres my water bottle", "fetch", "water bottle"),
    ("wheres the remote", "fetch", "remote"),
    # "walker" is not a request to go walking.
    ("wheres my walker", "fetch", "walker"),
    # --- fetch, misspelled --------------------------------------------------
    ("cud you bring me my watter bottle", "fetch", "watter bottle"),
    ("plz bring the banana", "fetch", "banana"),
    ("id like my glasses", "fetch", "glasses"),
    # --- fetch, shouted -----------------------------------------------------
    ("BRING THE WATER BOTTLE", "fetch", "water bottle"),
    ("WHERE IS MY REMOTE", "fetch", "remote"),
    ("PLEASE BRING ME MY WATER", "fetch", "water"),
    # --- fetch, buried in politeness and rambling ---------------------------
    ("could you please bring me the tv remote thank you very much", "fetch", "tv remote"),
    ("sorry to bother you but could you fetch my glasses", "fetch", "glasses"),
    ("i was thinking about lunch so could you get the granola bar", "fetch", "granola bar"),
    ("im cold can you bring the blanket", "fetch", "blanket"),
    ("hey corgi could you please get me the water bottle when you get a chance", "fetch",
     "water bottle"),
    # --- fetch, no verb at all: a name and nothing else ---------------------
    ("banana", "fetch", "banana"),
    ("the tv remote", "fetch", "tv remote"),
    ("my reading glasses", "fetch", "reading glasses"),
    ("some crackers please", "fetch", "crackers"),
    # --- fetch, two things in one message -----------------------------------
    ("bring me the banana and the water bottle", "fetch", "banana"),
    ("can you get my pills and a glass of water", "fetch", "pills"),
    ("bring the water then the banana", "fetch", "water"),
    # --- fetch with nothing nameable in it ----------------------------------
    ("can you get the thing on the counter", "fetch", None),
    ("get me that thing over there", "fetch", None),
    # --- stop ---------------------------------------------------------------
    ("STOP", "stop", None),
    ("stop please", "stop", None),
    ("wait", "stop", None),
    ("hold on a second", "stop", None),
    ("never mind", "stop", None),
    ("nevermind", "stop", None),
    ("that's enough", "stop", None),
    ("actually never mind", "stop", None),
    ("forget it", "stop", None),
    ("cancel that", "stop", None),
    ("stay there", "stop", None),
    # --- help ---------------------------------------------------------------
    ("help", "help", None),
    ("I NEED HELP", "help", None),
    ("ive fallen in the kitchen", "help", None),
    ("i fell", "help", None),
    ("i cant get up", "help", None),
    ("call my daughter", "help", None),
    ("call 911", "help", None),
    ("help me up", "help", None),
    # --- walk ---------------------------------------------------------------
    ("walk with me", "walk", None),
    ("lets go for a walk", "walk", None),
    ("take me to the bathroom", "walk", None),
    ("i want to go to the kitchen", "walk", None),
    # "help me walk" is a walk request with the word help in it, not a fall.
    ("help me walk to the kitchen", "walk", None),
    # --- come ---------------------------------------------------------------
    ("come here", "come", None),
    ("over here", "come", None),
    ("where are u", "come", None),
    ("i need you", "come", None),
    # --- status -------------------------------------------------------------
    ("what are you doing", "status", None),
    ("how long will you be", "status", None),
    ("did you find the remote", "status", None),
    ("are you there", "status", None),
    ("hows it going", "status", None),
    ("any luck", "status", None),
    # --- chat ---------------------------------------------------------------
    ("hello", "chat", None),
    ("good morning", "chat", None),
    ("thank you", "chat", None),
    ("are you a real dog", "chat", None),
    ("that was helpful", "chat", None),
    ("i love you", "chat", None),
    ("im cold", "chat", None),
    ("whats the weather", "chat", None),
)

# What the camera can name. Present so the table exercises the same path the running
# robot takes; no row depends on last_item, which "another one" gets its own test for.
KNOWN = ["water bottle", "banana", "remote", "granola bar", "apple"]


@pytest.fixture
def router() -> KeywordRouter:
    return KeywordRouter()


@pytest.fixture
def ctx() -> RouterContext:
    return RouterContext(known_items=list(KNOWN))


@pytest.mark.parametrize(("text", "kind", "item"), CASES, ids=[row[0] for row in CASES])
async def test_table(router: KeywordRouter, ctx: RouterContext, text, kind, item) -> None:
    intent = await router.route(text, ctx)
    assert intent.kind == kind, f"{text!r} routed to {intent.kind}, expected {kind}"
    assert intent.item == item, f"{text!r} gave item {intent.item!r}, expected {item!r}"
    # A kind outside the list would crash the skill dispatcher downstream.
    assert intent.kind in INTENT_KINDS
    assert intent.raw == text
    # The concierge owns every word the person reads, so the router never writes one.
    assert intent.reply == ""


def test_table_is_a_real_table() -> None:
    assert len(CASES) >= 40
    assert len({row[0] for row in CASES}) == len(CASES), "duplicate message in the table"
    assert {row[1] for row in CASES} == set(INTENT_KINDS), "the table misses an intent"


# --------------------------------------------------------------------------
# stop, which must never be misheard
# --------------------------------------------------------------------------
STOP_PHRASINGS = (
    "stop",
    "STOP",
    "stop!",
    "stop it",
    "stop please",
    "please stop",
    "corgi stop",
    "wait",
    "wait a minute",
    "hold on",
    "hold on a second",
    "never mind",
    "nevermind",
    "never mind then",
    "cancel",
    "cancel that",
    "cancel the water",
    "that's enough",
    "thats enough",
    "that is enough",
    "no thats enough for now",
    "stay",
    "stay there",
    "quit",
    "halt",
    "forget it",
    # The word stop wins even when a fetch is wrapped around it.
    "stop, i dont want the banana after all",
    "actually never mind the water bottle",
)


@pytest.mark.parametrize("text", STOP_PHRASINGS)
async def test_stop_never_loses(router: KeywordRouter, ctx: RouterContext, text: str) -> None:
    intent = await router.route(text, ctx)
    assert intent.kind == "stop"
    assert intent.item is None
    assert intent.confidence == CONF_KEYWORD


# --------------------------------------------------------------------------
# help, and the words that only look like it
# --------------------------------------------------------------------------
HELP_PHRASINGS = (
    "help",
    "HELP",
    "help!",
    "help me",
    "i need help",
    "i need help in here",
    "ive fallen",
    "i've fallen",
    "ive fallen and i cant get up",
    "i fell",
    "i had a fall",
    "i cant get up",
    "emergency",
    "call someone",
    "call my daughter",
    "call an ambulance",
    "911",
)

NOT_HELP = (
    "that was helpful",
    "you were very helpful thank you",
    "youre helpful",
)


@pytest.mark.parametrize("text", HELP_PHRASINGS)
async def test_help_never_loses(router: KeywordRouter, ctx: RouterContext, text: str) -> None:
    intent = await router.route(text, ctx)
    assert intent.kind == "help"
    assert intent.confidence == CONF_KEYWORD


@pytest.mark.parametrize("text", NOT_HELP)
async def test_helpful_is_not_help(router: KeywordRouter, ctx: RouterContext, text: str) -> None:
    intent = await router.route(text, ctx)
    assert intent.kind != "help"


# --------------------------------------------------------------------------
# "another one"
# --------------------------------------------------------------------------
ANAPHORS = ("another one", "another one please", "another", "the same", "same again", "one more")


@pytest.mark.parametrize("text", ANAPHORS)
async def test_anaphor_resolves_against_last_item(router: KeywordRouter, text: str) -> None:
    intent = await router.route(text, RouterContext(known_items=list(KNOWN), last_item="banana"))
    assert intent.kind == "fetch"
    assert intent.item == "banana"
    assert intent.needs_clarification is False
    # A matched phrase, not a guess, so it scores like a verb hit.
    assert intent.confidence == CONF_KEYWORD


@pytest.mark.parametrize("text", ANAPHORS)
async def test_anaphor_with_no_last_item_asks(router: KeywordRouter, text: str) -> None:
    """Nothing to resolve against, so it stays a fetch and the concierge asks which
    thing. Guessing an item here would send the robot after the wrong object."""
    intent = await router.route(text, RouterContext(known_items=list(KNOWN)))
    assert intent.kind == "fetch"
    assert intent.item is None
    assert intent.needs_clarification is True
    assert intent.confidence == CONF_BARE_NOUN


async def test_named_item_beats_the_anaphor(router: KeywordRouter) -> None:
    """"another banana" names its own object and must not be hijacked by last_item."""
    ctx = RouterContext(known_items=list(KNOWN), last_item="water bottle")
    intent = await router.route("bring me another banana", ctx)
    assert (intent.kind, intent.item) == ("fetch", "banana")


# --------------------------------------------------------------------------
# confidence, clarification, and two requests in one message
# --------------------------------------------------------------------------
async def test_bare_noun_is_a_fetch_but_a_less_certain_one(
    router: KeywordRouter, ctx: RouterContext
) -> None:
    bare = await router.route("banana", ctx)
    verb = await router.route("bring me the banana", ctx)
    assert bare.kind == verb.kind == "fetch"
    assert bare.item == verb.item == "banana"
    assert bare.confidence == CONF_BARE_NOUN
    assert verb.confidence == CONF_KEYWORD
    assert bare.confidence < verb.confidence


async def test_bare_noun_works_for_an_item_the_camera_never_heard_of(
    router: KeywordRouter, ctx: RouterContext
) -> None:
    intent = await router.route("watter bottle", ctx)
    assert (intent.kind, intent.item) == ("fetch", "watter bottle")
    assert intent.confidence == CONF_BARE_NOUN


@pytest.mark.parametrize(
    ("text", "item", "also"),
    (
        ("bring me the water bottle and a banana", "water bottle", ["banana"]),
        ("bring me my glasses and the remote and a banana", "glasses", ["remote", "banana"]),
        ("can you get my pills and a glass of water", "pills", ["glass of water"]),
        ("bring the water then the banana", "water", ["banana"]),
        ("the banana & the apple", "banana", ["apple"]),
        ("bring the remote, the banana", "remote", ["banana"]),
    ),
)
async def test_two_requests_fill_also(
    router: KeywordRouter, ctx: RouterContext, text: str, item: str, also: list[str]
) -> None:
    intent = await router.route(text, ctx)
    assert intent.kind == "fetch"
    assert intent.item == item
    assert intent.also == also


async def test_also_is_capped_and_deduped(router: KeywordRouter, ctx: RouterContext) -> None:
    intent = await router.route(
        "bring the banana and the apple and the remote and the banana and a spoon and a fork",
        ctx,
    )
    assert intent.item == "banana"
    assert len(intent.also) <= 3
    assert "banana" not in intent.also


@pytest.mark.parametrize(
    "text",
    (
        "can you get the thing on the counter",
        "get me that thing over there",
        "bring me that",
        "get me something",
        "i think it might be time for my pills could you get them",
    ),
)
async def test_unnameable_object_asks_instead_of_guessing(
    router: KeywordRouter, ctx: RouterContext, text: str
) -> None:
    intent = await router.route(text, ctx)
    assert intent.kind == "fetch"
    assert intent.item is None
    assert intent.needs_clarification is True


async def test_empty_message_is_chat(router: KeywordRouter, ctx: RouterContext) -> None:
    """A dropped attachment arrives as an empty body. It must not become a fetch."""
    for text in ("", "   ", "?"):
        intent = await router.route(text, ctx)
        assert intent.kind == "chat"
        assert intent.item is None


async def test_route_needs_no_context(router: KeywordRouter) -> None:
    intent = await router.route("bring me the banana")
    assert (intent.kind, intent.item) == ("fetch", "banana")
    assert intent.route.backend == "keyword"
    assert intent.route.tier == "none"
    assert intent.route.fell_back is False
    assert intent.as_dict()["route"]["backend"] == "keyword"
