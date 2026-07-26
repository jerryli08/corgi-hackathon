"""The text-message transport. It knows nothing about robots.

Three things live here and nothing else:

* **Senders** -- one interface, three backends chosen by config, with the offline one
  as the default: print to the console, POST to the Node sidecar that owns Photon's
  Spectrum SDK, or drive the Messages app on this Mac through osascript. Every send
  path raises on failure; deciding what a failure means is not the transport's job.
* **The webhook front door** -- signature verification that says out loud *which*
  check failed, and a parser that returns None rather than raising on any shape it did
  not expect. Both halves are hostile-input code: the body is whatever the internet
  posted, and the item name inside it is under the sender's control.
* **The Outbox** -- the one place that decides whether a text is worth sending. This is
  the load-bearing idea. An 80-year-old must not get nine texts about one water bottle,
  so a dedupe key, a quiet gap and a daily cap sit in front of the transport, and a
  message the robot chose not to send is recorded exactly like one it did.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime

import httpx

from robot.config import (
    DAILY_MESSAGE_BUDGET,
    MESSAGING_BACKEND,
    PHOTON_BRIDGE_TIMEOUT_S,
    PHOTON_BRIDGE_URL,
    PHOTON_WEBHOOK_TOLERANCE_S,
    TEXT_MIN_GAP_S,
)

# Spectrum sends these two on every webhook delivery. server.py reads them off the
# request, so they live here next to the code that checks them.
HEADER_TIMESTAMP = "X-Spectrum-Timestamp"
HEADER_SIGNATURE = "X-Spectrum-Signature"

SECONDS_PER_DAY = 86400


# --------------------------------------------------------------------------
# messages
# --------------------------------------------------------------------------
@dataclass
class InboundMessage:
    id: str
    sender: str  # E.164 where available, else the opaque Spectrum id
    space_id: str  # the conversation to reply into
    text: str
    platform: str = "iMessage"
    at: float = field(default_factory=time.time)
    simulated: bool = False

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "sender": self.sender,
            "space_id": self.space_id,
            "text": self.text,
            "platform": self.platform,
            "at": self.at,
            "simulated": self.simulated,
        }


@dataclass
class SentMessage:
    space_id: str
    text: str
    at: float
    backend: str
    ok: bool
    detail: str = ""

    def as_dict(self) -> dict:
        return {
            "space_id": self.space_id,
            "text": self.text,
            "at": self.at,
            "backend": self.backend,
            "ok": self.ok,
            "detail": self.detail,
        }


# --------------------------------------------------------------------------
# senders
# --------------------------------------------------------------------------
class Messenger:
    """One outbound channel. Subclasses only have to put text somewhere."""

    name = "none"

    async def send(self, space_id: str, text: str, *, to: str | None = None) -> None:
        """Raise on failure. The Outbox decides what a failure means."""
        raise NotImplementedError

    async def aclose(self) -> None:
        pass


class LogMessenger(Messenger):
    """Prints and returns. The default, what the tests use, and the honest fallback:
    the demo still shows the exact text the person would have received."""

    name = "log"

    async def send(self, space_id: str, text: str, *, to: str | None = None) -> None:
        print(f"[imessage -> {space_id}] {text}")


class PhotonMessenger(Messenger):
    """Outbound through the corgi/ bridge process, which holds the live Spectrum
    connection (`bun create spectrum-project@latest corgi --providers imessage`).

    Only outbound goes this way. Inbound arrives on POST /api/imessage/relay, pushed by
    that same process as it reads Spectrum's own message stream, so the bridge being
    down costs replies, not commands.
    """

    name = "photon"

    def __init__(
        self,
        base_url: str = PHOTON_BRIDGE_URL,
        timeout_s: float = PHOTON_BRIDGE_TIMEOUT_S,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=timeout_s)

    async def send(self, space_id: str, text: str, *, to: str | None = None) -> None:
        r = await self._http.post(
            f"{self.base_url}/send",
            json={"spaceId": space_id, "text": text, "to": to},
        )
        # Whatever the sidecar said is the only debugging information there is, so it
        # travels with the exception rather than being swallowed here.
        body = r.text[:300].strip()
        if r.status_code // 100 != 2:
            raise RuntimeError(f"photon bridge HTTP {r.status_code}: {body}")
        try:
            ok = bool(r.json().get("ok"))
        except Exception:
            ok = False
        if not ok:
            raise RuntimeError(f"photon bridge did not confirm the send: {body}")

    async def aclose(self) -> None:
        await self._http.aclose()


def _is_browser_phone_space(space_id: str) -> bool:
    """The web phone and the tests reply into `sim`, which is not a Spectrum space."""
    return not space_id or space_id == "sim" or space_id.startswith("sim:")


class BrowserPhoneMessenger(Messenger):
    """Photon for real iMessage conversations; console for the browser phone.

    Without this, `CORGI_MESSAGING_BACKEND=photon` makes every simulated text 404 on
    the bridge (no conversation named `sim`). The web UI only shows successful sends,
    so texting looks dead even when the skill already started.
    """

    def __init__(self, remote: Messenger, local: Messenger | None = None) -> None:
        self.remote = remote
        self.local = local or LogMessenger()
        self.name = remote.name

    async def send(self, space_id: str, text: str, *, to: str | None = None) -> None:
        if _is_browser_phone_space(space_id):
            await self.local.send(space_id, text, to=to)
            return
        await self.remote.send(space_id, text, to=to)

    async def aclose(self) -> None:
        await self.remote.aclose()
        await self.local.aclose()


# The recipient and the message text arrive as argv, never as script source. An item
# name comes from whoever is texting us: interpolating it into this string would let
# `" & (do shell script "...")` run as AppleScript. `on run argv` makes that impossible
# -- osascript hands the values to the compiled script as data.
_APPLESCRIPT_SEND = """
on run argv
    set targetPhone to item 1 of argv
    set messageText to item 2 of argv
    tell application "Messages"
        set targetService to 1st service whose service type = iMessage
        send messageText to buddy targetPhone of targetService
    end tell
end run
"""


class AppleScriptMessenger(Messenger):
    """The Messages app on this Mac. No account, no keys, genuinely iMessage.

    This is the fallback when the sidecar is not running, and it needs a phone number:
    Messages resolves a buddy, not a Spectrum conversation id.
    """

    name = "applescript"

    async def send(self, space_id: str, text: str, *, to: str | None = None) -> None:
        if not to:
            raise ValueError("applescript needs a phone number to send to")

        proc = await asyncio.create_subprocess_exec(
            "osascript",
            "-e",
            _APPLESCRIPT_SEND,
            to,
            text,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            # Messages can sit on a permissions dialog forever, and the Outbox is
            # serialized behind this call, so cap it with the messaging timeout.
            _, err = await asyncio.wait_for(proc.communicate(), timeout=PHOTON_BRIDGE_TIMEOUT_S)
        except TimeoutError:
            proc.kill()
            with contextlib.suppress(Exception):
                await proc.communicate()
            raise RuntimeError(
                f"osascript did not finish within {PHOTON_BRIDGE_TIMEOUT_S:.0f}s "
                "(is Messages waiting on a dialog?)"
            ) from None

        if proc.returncode != 0:
            detail = err.decode("utf-8", "replace").strip() or "no stderr"
            raise RuntimeError(f"osascript exited {proc.returncode}: {detail}")


def make_messenger(backend: str = MESSAGING_BACKEND) -> tuple[Messenger, list[str]]:
    """Build the configured sender, plus notes about anything that had to degrade.

    Nothing here raises. A messaging backend that cannot come up must cost the demo its
    texts, not its boot.
    """
    notes: list[str] = []
    choice = (backend or "").strip().lower()

    if choice == "photon":
        if not PHOTON_BRIDGE_URL.strip():
            notes.append("messaging: no photon bridge URL, printing texts to the console instead")
        else:
            try:
                # Browser-phone texts (`space_id=sim`) stay local; real Spectrum spaces
                # still go through the bridge.
                return BrowserPhoneMessenger(PhotonMessenger()), notes
            except Exception as exc:
                notes.append(f"messaging: photon bridge unavailable ({exc}), printing instead")
    elif choice == "applescript":
        notes.append("messaging: sending through Messages on this Mac (phone numbers only)")
        return AppleScriptMessenger(), notes
    elif choice not in ("", "log"):
        notes.append(f"messaging: unknown backend {backend!r}, printing texts to the console")

    return LogMessenger(), notes


# --------------------------------------------------------------------------
# webhook
# --------------------------------------------------------------------------
class SignatureError(Exception):
    """Why a webhook was rejected. The message is the whole point: a webhook that
    silently 401s at three in the morning is unfixable."""


def spectrum_signature(secret: str, timestamp: str, raw_body: bytes) -> str:
    """The v0 signature Spectrum puts in X-Spectrum-Signature.

    Signed over bytes rather than a decoded str: the payload is whatever was posted,
    and a body that is not valid UTF-8 must fail the comparison, not the decode.
    """
    payload = b"v0:" + timestamp.encode("utf-8", "surrogateescape") + b":" + raw_body
    digest = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return f"v0={digest}"


def verify_spectrum_signature(
    raw_body: bytes,
    *,
    timestamp: str | None,
    signature: str | None,
    secret: str,
    tolerance_s: int = PHOTON_WEBHOOK_TOLERANCE_S,
    now: float | None = None,
) -> None:
    """Raise SignatureError unless this body really came from Spectrum, recently.

    Every rejection names the check that failed, because the alternative is staring at
    a 401 with no idea whether the secret is wrong, the clock is wrong, or ngrok is
    rewriting the body.
    """
    if not secret:
        # Not a pass. An unset secret means the webhook was never registered properly,
        # and treating that as "no signing configured" would accept anyone's POST.
        raise SignatureError(
            "no webhook signing secret is configured (CORGI_PHOTON_WEBHOOK_SECRET)"
        )
    if not timestamp:
        raise SignatureError(f"missing {HEADER_TIMESTAMP} header")
    if not signature:
        raise SignatureError(f"missing {HEADER_SIGNATURE} header")

    try:
        sent_at = int(str(timestamp).strip())
    except (TypeError, ValueError):
        raise SignatureError(f"{HEADER_TIMESTAMP} is not an integer: {timestamp!r}") from None

    drift = abs((time.time() if now is None else now) - sent_at)
    if drift > tolerance_s:
        raise SignatureError(
            f"{HEADER_TIMESTAMP} is {drift:.0f}s from now, tolerance is {tolerance_s}s "
            "(replay, or this Mac's clock is off)"
        )

    expected = spectrum_signature(secret, str(timestamp).strip(), raw_body)
    if not hmac.compare_digest(expected, str(signature).strip()):
        raise SignatureError("signature does not match the body (wrong signing secret?)")


def _sub(obj: object, key: str) -> dict:
    """A nested dict, or an empty one. Spectrum sends null for absent objects and the
    simulator sends strings, so a `.get()` chain has to survive both."""
    value = obj.get(key) if isinstance(obj, dict) else None
    return value if isinstance(value, dict) else {}


def _text(obj: dict, key: str) -> str:
    value = obj.get(key)
    return value.strip() if isinstance(value, str) else ""


def _parse_at(stamp: str) -> float:
    try:
        return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()
    except Exception:
        return time.time()


def parse_spectrum_webhook(payload: dict) -> InboundMessage | None:
    """Pull one inbound text out of a Spectrum `messages` envelope, or return None.

    None is the answer for everything we do not act on -- an outbound copy, a reaction,
    an empty body, a different event, a completely unrelated JSON document. It is never
    an error, because Spectrum retries anything that is not a 2xx and a webhook we
    ignore would then arrive forever.
    """
    try:
        if not isinstance(payload, dict) or payload.get("event") != "messages":
            return None

        message = _sub(payload, "message")
        if message.get("direction") != "inbound":
            return None

        content = _sub(message, "content")
        if content.get("type") != "text":
            return None
        text = _text(content, "text")
        if not text:
            return None

        # The top-level space is the authoritative conversation; message.space is the
        # copy carried on the message and is all we get on some deliveries.
        space = _sub(payload, "space") or _sub(message, "space")
        message_space = _sub(message, "space")

        # A phone number is what a human recognises in the ops console, so prefer it
        # over the opaque Spectrum id.
        sender = (
            _text(_sub(message, "sender"), "id")
            or _text(space, "phone")
            or _text(message_space, "phone")
        )
        space_id = _text(space, "id") or _text(message_space, "id")
        if not space_id and not sender:
            return None  # nothing to reply into and nobody to reply to

        return InboundMessage(
            id=_text(message, "id") or f"spc-{uuid.uuid4().hex[:12]}",
            sender=sender or space_id,
            space_id=space_id or sender,
            text=text,
            platform=_text(message, "platform") or _text(space, "platform") or "iMessage",
            at=_parse_at(_text(message, "timestamp")),
        )
    except Exception:
        return None


# --------------------------------------------------------------------------
# outbox
# --------------------------------------------------------------------------
class Outbox:
    """Rate limit, dedupe, and a daily cap in front of a Messenger.

    An elderly user must not get a text per phase transition. Callers pass a `key` for
    anything that could fire more than once (usually f"{order_id}:{phase}"); the same
    key never sends twice.
    """

    def __init__(
        self,
        messenger: Messenger,
        *,
        min_gap_s: float = TEXT_MIN_GAP_S,
        daily_budget: int = DAILY_MESSAGE_BUDGET,
        history_limit: int = 200,
    ) -> None:
        self.messenger = messenger
        self.min_gap_s = min_gap_s
        self.daily_budget = daily_budget
        self.history_limit = history_limit
        self._keys: set[str] = set()
        self._history: list[SentMessage] = []
        self._last_ok_at = 0.0
        self._sent = 0
        self._dropped = 0
        # (UTC day ordinal, texts sent that day). Recomputed on read, so there is no
        # timer to forget to cancel and no midnight task that has to still be alive.
        self._day = (0, 0)
        # One text at a time: the gap and the budget are only honest if two callers
        # cannot both pass the checks before either send has happened.
        self._lock = asyncio.Lock()

    async def send(
        self,
        space_id: str,
        text: str,
        *,
        to: str | None = None,
        key: str | None = None,
        urgent: bool = False,
    ) -> bool:
        """True means handed to the transport, False means it was not sent.

        The order of the three gates is deliberate: a duplicate is dropped even when
        urgent, the daily cap holds even when urgent, and only the quiet gap yields to
        urgency. Arrival, failure and a request for help are urgent; progress is not.
        """
        async with self._lock:
            now = time.time()

            if key and key in self._keys:
                return self._record(space_id, text, now, False, f"duplicate of {key}")

            used = self._today(now)
            if used >= self.daily_budget:
                return self._record(
                    space_id, text, now, False, f"over the daily budget of {self.daily_budget}"
                )

            if not urgent and self.min_gap_s > 0 and (now - self._last_ok_at) < self.min_gap_s:
                return self._record(
                    space_id, text, now, False, f"inside the {self.min_gap_s:.0f}s quiet gap"
                )

            try:
                await self.messenger.send(space_id, text, to=to)
            except Exception as exc:
                return self._record(space_id, text, now, False, f"send failed: {exc}")

            self._last_ok_at = time.time()
            self._day = (self._day[0], used + 1)
            self._sent += 1
            # The key is burnt only once the text actually left, so a milestone lost to
            # a dead sidecar can still be delivered by the next attempt.
            if key:
                self._keys.add(key)
            return self._record(space_id, text, now, True, "")

    def recent(self, limit: int = 40) -> list[dict]:
        return [m.as_dict() for m in self._history[-limit:]]

    def stats(self) -> dict:
        return {
            "backend": self.messenger.name,
            "sent": self._sent,
            "dropped": self._dropped,
            "budget": self.daily_budget,
            "remaining": max(0, self.daily_budget - self._today(time.time())),
        }

    async def aclose(self) -> None:
        await self.messenger.aclose()

    # -- internals --------------------------------------------------------
    def _today(self, now: float) -> int:
        """Texts sent so far in the current UTC day, rolling the counter if it moved."""
        day = int(now // SECONDS_PER_DAY)
        if day != self._day[0]:
            self._day = (day, 0)
        return self._day[1]

    def _record(self, space_id: str, text: str, at: float, ok: bool, detail: str) -> bool:
        # Dropped texts are history too: the ops console has to be able to show that the
        # robot chose to stay quiet, otherwise a missing text looks like a crash.
        if not ok:
            self._dropped += 1
        self._history.append(
            SentMessage(
                space_id=space_id,
                text=text,
                at=at,
                backend=self.messenger.name,
                ok=ok,
                detail=detail,
            )
        )
        del self._history[: max(0, len(self._history) - self.history_limit)]
        return ok
