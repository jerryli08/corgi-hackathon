"""Free text in, one typed intent out. Two backends behind one interface.

    keyword -- a table of word-boundary patterns. No network, no key, and the default,
               so the demo works unplugged. It is also the fallback under every merge
               failure, which is why it is a real router and not a stub.
    merge   -- Merge Gateway. One tight prompt, JSON out, two tiers: a cheap model for
               the two-word messages that are 90% of the traffic, escalating to a
               stronger one only when the cheap answer is unsure.

The load-bearing idea is that `stop` and `help` are safety-critical and the language
model is the least trustworthy part of the system. So the keyword router runs on every
message even when a model answered, and if the keywords hear "stop" or "help" and the
model did not, the keywords win and the override is recorded in RouteInfo. Missing a
"stop" from someone who needs the robot to stop is the worst failure this system has.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import ValidationError

from robot.config import (
    MERGE_API,
    MERGE_API_KEY,
    MERGE_BASE_URL,
    ROUTER_BACKEND,
    ROUTER_CONFIDENCE_FLOOR,
    ROUTER_DEEP_MODEL,
    ROUTER_FAST_MODEL,
    ROUTER_MAX_CHARS,
    ROUTER_TIMEOUT_S,
)

try:
    # The official SDK (pip install merge-gateway-python) backs the native "responses"
    # path. It is optional in the sense that make_router() degrades to the keyword
    # router if it is missing -- the "openai" shim below needs no SDK at all -- but it
    # is a normal entry in requirements.txt, not a hardware-only extra.
    from merge_gateway import MergeGateway
    from merge_gateway.types import Response as _MergeResponse
except ImportError:  # pragma: no cover - exercised by not installing the package
    MergeGateway = None  # type: ignore[assignment,misc]
    _MergeResponse = Any  # type: ignore[assignment,misc]

INTENT_KINDS = ("fetch", "come", "walk", "stop", "status", "help", "chat")

# Confidence the keyword router reports. A pattern hit is a near-certainty; a bare noun
# ("banana") is a guess that the person wants it fetched; chat is the shrug.
CONF_KEYWORD = 0.9
CONF_BARE_NOUN = 0.5
CONF_CHAT = 0.2

MAX_ALSO = 3


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
@dataclass
class RouteInfo:
    """How the decision got made. The ops console shows this verbatim, because
    'which model handled this request' is the entire point of a router."""

    backend: str
    tier: str  # ours: "fast" or "deep" -- which of our two models was asked
    model: str = ""
    served_by: str = ""
    # Merge's own routing metadata, when there is any to report: which vendor executed
    # the call and which of Merge's routing tiers it picked, if a routing policy is
    # configured. Blank when we are not using Merge's own routing (the normal case,
    # since `tier` above is our client-side fast/deep choice of model, not theirs).
    service_tier: str = ""
    latency_ms: int = 0
    escalated: bool = False
    fell_back: bool = False
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "backend": self.backend,
            "tier": self.tier,
            "model": self.model,
            "served_by": self.served_by,
            "service_tier": self.service_tier,
            "latency_ms": self.latency_ms,
            "escalated": self.escalated,
            "fell_back": self.fell_back,
            "note": self.note,
        }


@dataclass
class Intent:
    kind: str
    item: str | None = None
    also: list[str] = field(default_factory=list)
    reply: str = ""
    confidence: float = 0.0
    needs_clarification: bool = False
    raw: str = ""
    route: RouteInfo = field(default_factory=lambda: RouteInfo("keyword", "none"))

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "item": self.item,
            "also": list(self.also),
            "reply": self.reply,
            "confidence": round(self.confidence, 3),
            "needs_clarification": self.needs_clarification,
            "raw": self.raw,
            "route": self.route.as_dict(),
        }


@dataclass
class RouterContext:
    """What the router is allowed to know about the robot right now."""

    busy: bool = False
    phase: str = "IDLE"
    carrying: str | None = None
    basket: list[str] = field(default_factory=list)
    known_items: list[str] = field(default_factory=list)
    last_item: str | None = None
    walking: bool = False

    def as_prompt_block(self) -> str:
        return "\n".join(
            [
                "Robot right now:",
                f"- doing: {'busy' if self.busy else 'idle'}, phase {self.phase}",
                f"- in the jaws: {self.carrying or 'nothing'}",
                f"- in the basket: {', '.join(self.basket) or 'nothing'}",
                f"- items the camera can name: {', '.join(self.known_items) or 'unknown'}",
                f"- last item asked for: {self.last_item or 'none'}",
                f"- walking with the person: {'yes' if self.walking else 'no'}",
            ]
        )


# --------------------------------------------------------------------------
# keyword backend
# --------------------------------------------------------------------------
# Precedence is fixed and it is the whole design: stop, help, walk, come, status,
# fetch, chat. Everything matches on word boundaries, so "stop" does not fire on
# "stopwatch" and "help" does not fire on "helpful".
_STOP_PATTERNS = (
    r"\bstop\b",
    r"\bwait\b",
    r"\bhold on\b",
    r"\bnever ?mind\b",
    r"\bcancel\b",
    r"\bstay\b",
    r"\b(?:that'?s|that is|thats) enough\b",
    r"\bquit\b",
    r"\bhalt\b",
    r"\bforget it\b",
)

_HELP_PATTERNS = (
    r"\bhelp\b",
    r"\bemergency\b",
    r"\bfallen\b",
    r"\bi fell\b",
    r"\bi (?:had|have had) a fall\b",
    r"\bi need help\b",
    r"\bcall (?:someone|somebody|for help|911|an ambulance|my daughter|my son)\b",
    r"\b911\b",
    r"\bi can'?t get up\b",
)

# One carve-out to the precedence: "help me walk" is the word "help" attached to a walk
# request, and answering it with "I've stopped and I'm staying put" would be wrong.
# Anything else containing "help" -- including "help me up" -- stays a help.
_HELP_IS_WALK = re.compile(r"\bhelp me (?:walk|to walk|get to|down|along)\b")

_WALK_PATTERNS = (
    r"\bwalk with me\b",
    r"\bwalk me\b",
    r"\b(?:let'?s|lets) (?:go )?(?:for a )?walk\b",
    r"\btake me to\b",
    r"\bhelp me (?:walk|to walk|get to)\b",
    r"\bsteady\b",
    r"\bi want to go to\b",
    r"\bi'?d like to go to\b",
    r"\bwalk (?:to|down|around|beside|alongside)\b",
    r"\bgo for a walk\b",
)

_COME_PATTERNS = (
    r"\bcome\b",
    r"\bover here\b",
    r"\bwhere are (?:you|u)\b",
    r"\bwhere'?re you\b",
    r"\bwhere r u\b",
    r"\bi need you\b",
    r"\bin here\b",
)

_STATUS_PATTERNS = (
    r"\bwhat are you doing\b",
    r"\bwhat'?re you doing\b",
    r"\bwhat are you up to\b",
    r"\bstatus\b",
    r"\bhow long\b",
    r"\bdid you (?:get|find)\b",
    r"\bare you there\b",
    r"\bhow'?s it going\b",
    r"\bhow'?s it coming\b",
    r"\bwhat'?s going on\b",
    r"\bwhat is going on\b",
    r"\bany luck\b",
)

# The imperative verbs, and the phrases people use instead of one. This regex both
# detects a fetch and marks where the item starts: everything after the match.
_FETCH_VERB = re.compile(
    r"\b(?:"
    r"i'?d like|i would like|i'?d love|i want|i need|i could use|i'?ll have|"
    r"(?:can|could|would|will) (?:you|i) (?:please )?"
    r"(?:bring|get|grab|fetch|hand|pass|give|find|have)|"
    r"bring|get me|get|grab|fetch|hand|pass|pick up|find|"
    r"where'?s my|where is my|where'?s the|where is the|where did i put|where'?d i put|"
    r"do you have|have you got|need"
    r")\b"
)

# "another one" and friends only resolve against ctx.last_item when the whole message is
# the anaphor. "bring me another banana" names its own item and must not be hijacked.
_ANAPHOR = re.compile(
    r"^(?:the )?(?:another(?: one)?|same(?: one| thing)?(?: again)?|"
    r"same as (?:before|last time)|one more|usual|do that again)$"
)

# Politeness the person wraps a request in. Stripped in a loop, so "hey corgi could you
# please" comes off one layer at a time.
_ADDRESS_PREFIX = re.compile(r"^(?:hey|hi|hello|ok|okay|yo)?\s*corgi\b[\s,.!]*")
_POLITE_PREFIX = re.compile(
    r"^(?:please|can you|could you|would you|will you|would you mind|do you mind|"
    r"i wonder if you could|if you could|if you don'?t mind|when you get a chance|"
    r"hey|hi|hello|ok|okay|so|um|uh|maybe|actually|just)\b[\s,]*"
)
_POLITE_SUFFIX = re.compile(
    r"[\s,]*\b(?:please|thanks|thank you|thank you very much|thanks a lot|for me|"
    r"if you don'?t mind|when you get a chance|when you can|would you|will you|"
    r"ok|okay|cheers|ta|now|sometime|at some point)$"
)
# Articles, possessives and the leftover object pronouns after the verb.
_LEADING_JUNK = re.compile(
    r"^(?:me|us|to me|for me|the|a|an|my|mine|our|some|any|that|those|these|this|"
    r"another|one of the|couple of|bit of|of|more)\b[\s,]*"
)
_SEPARATORS = re.compile(r"\s+and\s+|\s+plus\s+|\s+then\s+|\s*&\s*|\s*,\s*")

# A message that is only a noun phrase is almost always a fetch, but not if it is one of
# these -- someone saying hello is not asking for a hello.
_SMALL_TALK = {
    "hi",
    "hello",
    "hey",
    "morning",
    "good morning",
    "good afternoon",
    "good evening",
    "good night",
    "goodnight",
    "night",
    "bye",
    "goodbye",
    "yes",
    "yeah",
    "yep",
    "no",
    "nope",
    "sure",
    "alright",
    "fine",
    "nothing",
    "sorry",
    "love you",
    "i love you",
    "good dog",
    "good boy",
    "who are you",
    "what are you",
}

# Words that name nothing. The person has to be asked which thing they mean.
_VAGUE = {
    "it",
    "that",
    "this",
    "one",
    "them",
    "something",
    "anything",
    "everything",
    "thing",
    "things",
    "stuff",
    "the thing",
    "that thing",
    "the other one",
}
# A head noun this vague names nothing however much follows it: "the thing on the
# counter" is a question, not an order.
_VAGUE_HEADS = {"thing", "things", "stuff", "something", "anything", "everything", "one"}

# A message that opens with one of these, or contains a copula, is a sentence rather
# than the name of an object, so the bare-noun rule must not claim it.
_NOT_A_NOUN_OPENER = {
    "what",
    "who",
    "why",
    "how",
    "when",
    "whose",
    "where",
    "is",
    "are",
    "was",
    "do",
    "does",
    "did",
    "should",
    "shall",
    "am",
    "i",
    "you",
    "we",
    "they",
    "he",
    "she",
    "there",
    "let",
    "lets",
    # The contractions of the same words, spelled with and without the apostrophe,
    # because people text both ways. Without these, "im cold" reads as a noun phrase
    # and comes back as a request to fetch an object called "im cold".
    "i'm",
    "im",
    "i've",
    "ive",
    "i'll",
    "ill",
    "what's",
    "whats",
    "who's",
    "whos",
    "how's",
    "hows",
    "there's",
    "theres",
    "it's",
    "its",
    "you're",
    "youre",
    "we're",
    "were",
    "they're",
    "theyre",
}
_SENTENCE_WORDS = {
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "am",
    "not",
    "isnt",
    "aren't",
    "isn't",
    "dont",
    "don't",
    "doesnt",
    "cant",
    "can't",
}

# Item extraction, worked examples. Add a row here when a real message goes wrong; the
# rules below are only worth what this table says they are.
#
#   "can you bring me my water bottle please"    -> water bottle
#   "Corgi, could you get the granola bar?"      -> granola bar
#   "i'd like some strawberries thanks"          -> strawberries
#   "banana"                                     -> banana        (no verb, conf 0.5)
#   "bring me the water bottle and a banana"     -> water bottle  + also ["banana"]
#   "another one please"                         -> ctx.last_item
#   "wheres my water bottle"                     -> water bottle
#   "can you get the thing on the counter"       -> needs_clarification
#   "hello"                                      -> chat, no item


def _clean(text: str) -> str:
    """Lowercase, straighten curly apostrophes, drop end punctuation, collapse spaces."""
    out = (text or "").lower().replace("’", "'").replace("‘", "'")
    out = re.sub(r"\s+", " ", out).strip()
    return out.strip(" .!?;:\"'")


def _strip_edges(text: str) -> str:
    out = _ADDRESS_PREFIX.sub("", text, count=1).strip()
    for _ in range(4):
        stripped = _POLITE_PREFIX.sub("", out, count=1).strip()
        if stripped == out:
            break
        out = stripped
    for _ in range(4):
        stripped = _POLITE_SUFFIX.sub("", out, count=1).strip()
        if stripped == out:
            break
        out = stripped
    return out.strip(" ,.!?")


def _strip_junk(part: str) -> str:
    out = part.strip()
    for _ in range(4):
        stripped = _LEADING_JUNK.sub("", out, count=1).strip()
        if stripped == out:
            break
        out = stripped
    for _ in range(3):
        stripped = _POLITE_SUFFIX.sub("", out, count=1).strip()
        if stripped == out:
            break
        out = stripped
    return out.strip(" ,.!?")


def _is_vague(name: str) -> bool:
    return name in _VAGUE or name.split()[0] in _VAGUE_HEADS


def _items(text: str, ctx: RouterContext) -> tuple[str | None, list[str]]:
    """(first item, the rest). Returns (None, []) when nothing nameable is left."""
    body = _strip_edges(_clean(text))
    if _ANAPHOR.match(body):
        return (ctx.last_item, []) if ctx.last_item else (None, [])

    match = _FETCH_VERB.search(body)
    if match:
        body = body[match.end() :]

    found: list[str] = []
    for part in _SEPARATORS.split(body):
        name = _strip_junk(part or "")
        if name and name not in found and not _is_vague(name):
            found.append(name)
    if not found:
        return None, []
    return found[0], found[1 : 1 + MAX_ALSO]


def _looks_like_bare_noun(text: str) -> bool:
    """A short noun phrase with no verb: "banana", "my water bottle"."""
    body = _strip_junk(_strip_edges(_clean(text)))
    if not body or body in _SMALL_TALK or _is_vague(body):
        return False
    words = body.split()
    if len(words) > 4 or words[0] in _NOT_A_NOUN_OPENER:
        return False
    if any(word in _SENTENCE_WORDS for word in words):
        return False
    return bool(re.fullmatch(r"[a-z][a-z' -]*", body))


def _any(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


class Router:
    """One interface, two backends. Neither one is allowed to raise at the caller."""

    name = "router"

    async def route(self, text: str, ctx: RouterContext | None = None) -> Intent:
        raise NotImplementedError

    async def aclose(self) -> None:
        pass


class KeywordRouter(Router):
    """Deterministic, offline, and the fallback for every merge failure.

    `reply` is always left empty: the concierge owns the wording, so there is exactly one
    place where the robot's voice lives.
    """

    name = "keyword"

    async def route(self, text: str, ctx: RouterContext | None = None) -> Intent:
        started = time.monotonic()
        ctx = ctx or RouterContext()
        body = _clean(text)
        kind, item, also, needs_clarification, confidence = self._classify(body, ctx)
        return Intent(
            kind=kind,
            item=item,
            also=also,
            confidence=confidence,
            needs_clarification=needs_clarification,
            raw=text,
            route=RouteInfo(
                backend="keyword",
                tier="none",
                latency_ms=int((time.monotonic() - started) * 1000),
            ),
        )

    def _classify(
        self, body: str, ctx: RouterContext
    ) -> tuple[str, str | None, list[str], bool, float]:
        if _any(_STOP_PATTERNS, body):
            return "stop", None, [], False, CONF_KEYWORD
        if _any(_HELP_PATTERNS, body) and not _HELP_IS_WALK.search(body):
            return "help", None, [], False, CONF_KEYWORD
        if _any(_WALK_PATTERNS, body):
            return "walk", None, [], False, CONF_KEYWORD
        if _any(_COME_PATTERNS, body):
            return "come", None, [], False, CONF_KEYWORD
        if _any(_STATUS_PATTERNS, body):
            return "status", None, [], False, CONF_KEYWORD

        has_verb = bool(_FETCH_VERB.search(body))
        # "another one" is a matched phrase, not a guess, so it scores like a verb hit.
        anaphor = bool(_ANAPHOR.match(_strip_edges(body)))
        named = any(label and label in body for label in ctx.known_items)
        if has_verb or anaphor or named or _looks_like_bare_noun(body):
            item, also = _items(body, ctx)
            confidence = CONF_KEYWORD if has_verb or anaphor else CONF_BARE_NOUN
            if item is None:
                # A request with no nameable object: "get me that thing over there".
                return "fetch", None, [], True, min(confidence, CONF_BARE_NOUN)
            return "fetch", item, also, False, confidence

        return "chat", None, [], False, CONF_CHAT


# --------------------------------------------------------------------------
# merge backend
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """You are the intent router for Corgi, a small home helper robot. The \
person texting you is elderly and does not want to get up. Turn their message into one \
typed intent.

Reply with JSON only. No prose, no code fence. Exactly these keys:
{"intent": str, "item": str|null, "also": [str], "reply": str, "confidence": number,
 "needs_clarification": bool}

"intent" must be exactly one of: fetch, come, walk, stop, status, help, chat.
  fetch  - bring them an object. Put the object in "item", lowercase, no article
           ("water bottle", not "my Water Bottle"). Extra objects in the same message go
           in "also", at most three.
  come   - drive to where they are.
  walk   - walk with them. The robot paces alongside and holds its arm out as something
           at a known height to steady a hand against. It cannot take anyone's weight.
  stop   - stop what you are doing. Also "wait", "hold on", "never mind", "cancel".
  status - what are you doing, how long, did you get it.
  help   - they have fallen, they are frightened, they ask for help or for someone to be
           called. When a message could be help, answer help.
  chat   - anything else.

"confidence" is 0 to 1: how sure you are of the intent, not of the item.
"needs_clarification" is true when you cannot tell which object they mean ("the thing on
the counter"). Then leave "item" null and ask the question in "reply".

"reply" is only for chat and for needs_clarification. Leave it as "" for every other
intent, because those replies are pre-written templates that stay in sync with what the
robot is actually doing.

When you do write "reply": at most two short sentences, plain words an 80-year-old reads
without effort. No emoji. No exclamation marks. Never "I'd be happy to". Never promise
anything the robot has not done.

A block describing what the robot is doing right now follows the message. Use it to
resolve "another one" and "the same again" against the last item.

Examples.

Message: can you bring me my water bottle please
{"intent":"fetch","item":"water bottle","also":[],"reply":"","confidence":0.97,
 "needs_clarification":false}

Message: hi corgi i was going to make lunch so could you bring the granola bar over and \
then the banana as well when you have a minute
{"intent":"fetch","item":"granola bar","also":["banana"],"reply":"","confidence":0.9,
 "needs_clarification":false}

Message: get me the thing on the counter
{"intent":"fetch","item":null,"also":[],"reply":"Which one do you mean?",
 "confidence":0.45,"needs_clarification":true}

Message: are you a real dog
{"intent":"chat","item":null,"also":[],"reply":"No, I am a robot with wheels and a \
basket. I can bring you something if you like.","confidence":0.9,
 "needs_clarification":false}"""


class _ReplyProblem(Exception):
    """The provider answered, but not with an intent we can use. Escalate, do not
    fall back: a stronger model usually gets the JSON right."""


def _response_text(data: dict) -> str:
    """Pull the assistant text out of a /responses body without trusting its shape.

    Merge's exact output shape is not something we can verify from here, and a reasoning
    block arriving before the text block must not break the read, so walk defensively.
    """
    direct = data.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct

    for block in data.get("output") or []:
        if not isinstance(block, dict):
            continue
        for part in block.get("content") or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") in ("output_text", "text"):
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    return text

    choices = data.get("choices") or []
    if choices and isinstance(choices[0], dict):
        content = (choices[0].get("message") or {}).get("content")
        if isinstance(content, str) and content.strip():
            return content

    raise _ReplyProblem("no text in the router reply")


def _json_body(r: httpx.Response) -> dict:
    """A body that is not JSON is a reply that will not parse, so it escalates like any
    other unusable reply instead of being treated as an outage."""
    try:
        data = r.json()
    except ValueError as exc:
        raise _ReplyProblem(f"reply body was not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise _ReplyProblem("reply body was not a JSON object")
    return data


def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise _ReplyProblem(f"no JSON in the router reply: {text[:120]}")
    try:
        data = json.loads(match.group(0))
    except ValueError as exc:
        raise _ReplyProblem(f"unparseable JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise _ReplyProblem("router reply was not a JSON object")
    return data


def _clip(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


# A model reply is the only text in the system that nobody wrote by hand, and it goes
# straight to an 80-year-old, so it gets the same house rules as everything else.
_EMOJI = re.compile(r"[\U0001f000-\U0001faff←-⇿☀-➿️]")
_BANNED_TONE = re.compile(r"i'?d be happy to|as an ai|i am an ai|no problem at all|absolutely")


def _tidy_reply(reply: str) -> str:
    """Returns "" when the reply is unusable, and the concierge template takes over."""
    text = _EMOJI.sub("", reply).replace("!", ".")
    text = re.sub(r"\s+", " ", text).strip()
    if not text or _BANNED_TONE.search(text.lower()):
        return ""
    sentences = re.split(r"(?<=[.?])\s+", text)
    return " ".join(sentences[:2]).strip()


def _intent_from_json(data: dict, raw: str, route: RouteInfo) -> Intent:
    kind = str(data.get("intent") or "").strip().lower()
    if kind not in INTENT_KINDS:
        raise _ReplyProblem(f"intent {kind!r} is not one of {INTENT_KINDS}")

    item = data.get("item")
    item = str(item).strip().lower() or None if isinstance(item, str) else None

    also: list[str] = []
    for entry in data.get("also") or []:
        name = str(entry).strip().lower()
        if name and name != item and name not in also:
            also.append(name)

    needs_clarification = bool(data.get("needs_clarification"))
    # The templates in the concierge stay in sync with the motion, so a model reply is
    # only ever allowed to speak for chat and for a question we have to ask.
    reply = ""
    if kind == "chat" or needs_clarification:
        reply = _tidy_reply(str(data.get("reply") or ""))

    return Intent(
        kind=kind,
        item=None if needs_clarification else item,
        also=also[:MAX_ALSO],
        reply=reply,
        confidence=_clip(data.get("confidence")),
        needs_clarification=needs_clarification,
        raw=raw,
        route=route,
    )


def _reason(exc: BaseException) -> str:
    detail = str(exc).strip().replace("\n", " ") or repr(exc)
    if isinstance(exc, _ReplyProblem):
        return detail[:160]
    return f"{type(exc).__name__}: {detail}"[:160]


def _response_text_from_sdk(response: _MergeResponse) -> str:
    """Pull the assistant text out of a merge_gateway Response.

    A reasoning ("thinking") block can arrive in the same message before the text
    block, so this walks every block of every output message rather than trusting
    output[0].content[0] to be the answer.
    """
    for out in response.output:
        for block in out.content:
            if getattr(block, "type", None) == "text":
                text = getattr(block, "text", "")
                if isinstance(text, str) and text.strip():
                    return text
    raise _ReplyProblem("no text in the router reply")


def _routing_label(response: _MergeResponse) -> str:
    """Whatever Merge's own routing metadata reports, if there is any. Blank unless a
    routing policy is configured on the dashboard -- we do our own fast/deep choice of
    model client-side, so this is usually empty, and that is not a problem."""
    routing = getattr(response, "routing", None)
    if routing is None:
        return ""
    bits = []
    if routing.vendor_used:
        bits.append(routing.vendor_used)
    if routing.selected_tier is not None:
        bits.append(f"tier {routing.selected_tier}")
    return " ".join(bits)


_UNSET = object()  # distinguishes "client not passed" (build the real one) from
# "client=None" (the tests' way of saying the package is genuinely not installed)


class MergeRouter(Router):
    """Merge Gateway: the official SDK (`pip install merge-gateway-python`) for the
    native /responses path, plain REST for the OpenAI-compatible shim.

    Two tiers, and a keyword router held alongside for two jobs: answering when Merge
    does not, and checking every model answer for a "stop" or a "help" the model missed.
    """

    name = "merge"

    def __init__(
        self,
        *,
        client: MergeGateway | None = _UNSET,  # type: ignore[assignment]
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        # The SDK's client is synchronous (a plain httpx.Client under the hood, with no
        # seam of its own for a custom transport), so every call to it goes through
        # asyncio.to_thread rather than blocking the event loop. `client` is the tests'
        # seam for that path: a stand-in with the same responses.create(...) shape,
        # never a real network call. `transport` is the equivalent seam for the
        # OpenAI-compatible shim, which stays on plain async httpx.
        if client is not _UNSET:
            self._client = client
        elif MergeGateway is not None:
            self._client = MergeGateway(
                api_key=MERGE_API_KEY, base_url=MERGE_BASE_URL, timeout=ROUTER_TIMEOUT_S
            )
        else:
            self._client = None
        self._http = httpx.AsyncClient(timeout=ROUTER_TIMEOUT_S, transport=transport)
        self._fallback = KeywordRouter()

    async def route(self, text: str, ctx: RouterContext | None = None) -> Intent:
        ctx = ctx or RouterContext()
        clipped = (text or "").strip()[:ROUTER_MAX_CHARS]
        started = time.monotonic()

        try:
            reason = ""
            try:
                intent = await self._ask(ROUTER_FAST_MODEL, "fast", clipped, ctx)
                if intent.confidence < ROUTER_CONFIDENCE_FLOOR:
                    reason = f"fast tier confidence {intent.confidence:.2f}"
            except _ReplyProblem as problem:
                reason = str(problem)

            if reason:
                # A second, stronger opinion. If this one is unusable too, the except
                # below hands the message to the keywords.
                intent = await self._ask(ROUTER_DEEP_MODEL, "deep", clipped, ctx)
                intent.route.escalated = True
                intent.route.note = f"escalated: {reason}"
        except Exception as exc:
            return await self._degrade(text, ctx, _reason(exc), started)

        intent.raw = text
        intent.route.latency_ms = int((time.monotonic() - started) * 1000)
        try:
            return await self._safety_check(intent, text, ctx)
        except Exception as exc:
            problem = f"safety check failed: {_reason(exc)}"
            intent.route.note = f"{intent.route.note}; {problem}" if intent.route.note else problem
            return intent

    async def _safety_check(self, intent: Intent, text: str, ctx: RouterContext) -> Intent:
        """Give the keywords the last word on stop and help. A needless stop costs the
        person one more text; a missed one is the worst thing this robot can do."""
        keyword = await self._fallback.route(text, ctx)
        if keyword.kind not in ("stop", "help") or keyword.kind == intent.kind:
            return intent

        note = f"keyword override: model said {intent.kind}, keywords said {keyword.kind}"
        keyword.route = RouteInfo(
            backend=intent.route.backend,
            tier=intent.route.tier,
            model=intent.route.model,
            served_by=intent.route.served_by,
            service_tier=intent.route.service_tier,
            latency_ms=intent.route.latency_ms,
            escalated=intent.route.escalated,
            fell_back=intent.route.fell_back,
            note=f"{intent.route.note}; {note}" if intent.route.note else note,
        )
        keyword.raw = text
        return keyword

    async def _degrade(
        self, text: str, ctx: RouterContext, reason: str, started: float
    ) -> Intent:
        intent = await self._fallback.route(text, ctx)
        intent.route = RouteInfo(
            backend="merge",
            tier="none",
            fell_back=True,
            latency_ms=int((time.monotonic() - started) * 1000),
            note=f"keywords answered instead: {reason}",
        )
        return intent

    async def _ask(self, model: str, tier: str, text: str, ctx: RouterContext) -> Intent:
        user = f"{text}\n\n{ctx.as_prompt_block()}"
        if MERGE_API == "openai":
            reply, served_by, routing_label = await self._ask_openai(model, user)
        else:
            reply, served_by, routing_label = await self._ask_responses(model, user)

        route = RouteInfo(
            backend="merge",
            tier=tier,
            model=model,
            served_by=served_by,
            service_tier=routing_label,
        )
        return _intent_from_json(_parse_json(reply), text, route)

    async def _ask_responses(self, model: str, user: str) -> tuple[str, str, str]:
        if self._client is None:
            raise _ReplyProblem("the merge_gateway package is not installed")

        try:
            response = await asyncio.to_thread(
                self._client.responses.create,
                model=model,
                input=[
                    {"type": "message", "role": "system", "content": SYSTEM_PROMPT},
                    {"type": "message", "role": "user", "content": user},
                ],
                include_routing_metadata=True,
            )
        except (ValueError, ValidationError) as exc:
            # A malformed-but-200 body is the model's own bad answer, not a gateway
            # outage -- one retry at the deep model usually fixes it, same as any other
            # unusable reply. The SDK's own HTTP-status errors (401/404/429/5xx) and raw
            # httpx transport errors (timeout, connect failure) are deliberately NOT
            # caught here: those are outages, and route() sends them straight to the
            # keyword fallback without spending a second call on a gateway that is not
            # answering.
            raise _ReplyProblem(f"gateway sent an unusable body: {exc}") from exc

        # Which vendor actually served the call is the interesting part of a router, so
        # keep whatever Merge admits to.
        return _response_text_from_sdk(response), response.model or "", _routing_label(response)

    async def _ask_openai(self, model: str, user: str) -> tuple[str, str, str]:
        r = await self._http.post(
            f"{MERGE_BASE_URL}/openai/chat/completions",
            headers={"Authorization": f"Bearer {MERGE_API_KEY}"},
            json={
                "model": model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
            },
        )
        r.raise_for_status()
        data = _json_body(r)
        return _response_text(data), str(data.get("model") or ""), ""

    async def aclose(self) -> None:
        await self._http.aclose()
        if self._client is not None:
            await asyncio.to_thread(self._client.close)
        await self._fallback.aclose()


def make_router(backend: str = ROUTER_BACKEND) -> tuple[Router, list[str]]:
    notes: list[str] = []
    if backend == "merge":
        if not MERGE_API_KEY:
            notes.append("router: MERGE_API_KEY is not set, using the keyword router")
        elif MERGE_API != "openai" and MergeGateway is None:
            notes.append(
                "router: the merge_gateway package is not installed "
                "(pip install merge-gateway-python), using the keyword router"
            )
        else:
            return MergeRouter(), notes
    elif backend not in ("keyword", ""):
        notes.append(f"router: unknown backend {backend!r}, using the keyword router")
    return KeywordRouter(), notes
