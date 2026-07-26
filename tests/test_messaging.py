"""Unit tests for the text transport.

Two things are worth stating up front, because they are why these tests are worth
running. First, the signature is re-derived here with hmac/hashlib from the formula in
the spec rather than by calling the module's own helper: a test that signs with the code
it is checking proves only that the code agrees with itself. Second, every parser case
asserts a `None`, not an exception -- the webhook body is whatever the internet posted,
and the whole design is that a shape we did not expect costs one ignored text and
nothing else.

Nothing here touches a network, a subprocess or a clock it does not control.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from datetime import UTC, datetime

import pytest

from robot.messaging import (
    HEADER_SIGNATURE,
    HEADER_TIMESTAMP,
    InboundMessage,
    LogMessenger,
    Messenger,
    Outbox,
    SignatureError,
    make_messenger,
    parse_spectrum_webhook,
    verify_spectrum_signature,
)

SECRET = "shh-this-is-the-webhook-secret"
BODY = b'{"event":"messages","message":{"content":{"text":"bring me my water bottle"}}}'
NOW = 1_800_000_000.0


def sign(secret: str, timestamp: str, raw_body: bytes) -> str:
    """The v0 signature, written out from the spec so the test is an independent check."""
    payload = b"v0:" + timestamp.encode() + b":" + raw_body
    return "v0=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


# --------------------------------------------------------------------------
# signature verification
# --------------------------------------------------------------------------
def test_correctly_signed_body_passes() -> None:
    stamp = str(int(NOW))
    verify_spectrum_signature(
        BODY,
        timestamp=stamp,
        signature=sign(SECRET, stamp, BODY),
        secret=SECRET,
        now=NOW,
    )


def test_tampered_body_fails() -> None:
    stamp = str(int(NOW))
    signature = sign(SECRET, stamp, BODY)
    tampered = BODY.replace(b"water bottle", b"whisky bottle")

    with pytest.raises(SignatureError) as caught:
        verify_spectrum_signature(
            tampered, timestamp=stamp, signature=signature, secret=SECRET, now=NOW
        )
    assert "signature" in str(caught.value)


def test_wrong_secret_fails() -> None:
    stamp = str(int(NOW))
    with pytest.raises(SignatureError):
        verify_spectrum_signature(
            BODY,
            timestamp=stamp,
            signature=sign("not-the-secret", stamp, BODY),
            secret=SECRET,
            now=NOW,
        )


def test_stale_timestamp_fails() -> None:
    # Correctly signed, just old: this is the replay case, and it must not pass merely
    # because the HMAC checks out.
    stamp = str(int(NOW) - 400)
    with pytest.raises(SignatureError) as caught:
        verify_spectrum_signature(
            BODY,
            timestamp=stamp,
            signature=sign(SECRET, stamp, BODY),
            secret=SECRET,
            tolerance_s=300,
            now=NOW,
        )
    assert HEADER_TIMESTAMP in str(caught.value)


def test_future_timestamp_fails() -> None:
    stamp = str(int(NOW) + 400)
    with pytest.raises(SignatureError):
        verify_spectrum_signature(
            BODY,
            timestamp=stamp,
            signature=sign(SECRET, stamp, BODY),
            secret=SECRET,
            tolerance_s=300,
            now=NOW,
        )


def test_missing_signature_header_fails() -> None:
    stamp = str(int(NOW))
    with pytest.raises(SignatureError) as caught:
        verify_spectrum_signature(
            BODY, timestamp=stamp, signature=None, secret=SECRET, now=NOW
        )
    assert HEADER_SIGNATURE in str(caught.value)


def test_missing_timestamp_header_fails() -> None:
    with pytest.raises(SignatureError) as caught:
        verify_spectrum_signature(
            BODY,
            timestamp=None,
            signature=sign(SECRET, str(int(NOW)), BODY),
            secret=SECRET,
            now=NOW,
        )
    assert HEADER_TIMESTAMP in str(caught.value)


def test_empty_secret_raises_rather_than_waving_it_through() -> None:
    stamp = str(int(NOW))
    with pytest.raises(SignatureError) as caught:
        verify_spectrum_signature(
            BODY, timestamp=stamp, signature=sign("", stamp, BODY), secret="", now=NOW
        )
    assert "secret" in str(caught.value)


def test_non_integer_timestamp_fails() -> None:
    with pytest.raises(SignatureError):
        verify_spectrum_signature(
            BODY,
            timestamp="last tuesday",
            signature=sign(SECRET, "last tuesday", BODY),
            secret=SECRET,
            now=NOW,
        )


# --------------------------------------------------------------------------
# webhook parsing
# --------------------------------------------------------------------------
SPACE = {
    "id": "spc-space-9f2",
    "platform": "iMessage",
    "type": "dm",
    "phone": "+15551234567",
}


def envelope(**message_overrides: object) -> dict:
    """The envelope from the spec, with the message half overridable per test."""
    message: dict = {
        "id": "spc-msg-4471",
        "platform": "iMessage",
        "direction": "inbound",
        "timestamp": "2026-07-25T18:03:11Z",
        "sender": {"id": "+15551234567", "platform": "iMessage"},
        "space": dict(SPACE),
        "content": {"type": "text", "text": "bring me my water bottle"},
    }
    message.update(message_overrides)
    return {"event": "messages", "space": dict(SPACE), "message": message}


def test_inbound_text_parses_into_every_field() -> None:
    msg = parse_spectrum_webhook(envelope())

    assert isinstance(msg, InboundMessage)
    assert msg.as_dict() == {
        "id": "spc-msg-4471",
        "sender": "+15551234567",
        "space_id": "spc-space-9f2",
        "text": "bring me my water bottle",
        "platform": "iMessage",
        "at": datetime(2026, 7, 25, 18, 3, 11, tzinfo=UTC).timestamp(),
        "simulated": False,
    }


def test_outbound_message_is_ignored() -> None:
    assert parse_spectrum_webhook(envelope(direction="outbound")) is None


def test_reaction_is_ignored() -> None:
    payload = envelope(content={"type": "reaction", "reaction": "love", "text": "loved"})
    assert parse_spectrum_webhook(payload) is None


def test_attachment_is_ignored() -> None:
    payload = envelope(
        content={"type": "attachment", "url": "https://example.invalid/cat.heic"}
    )
    assert parse_spectrum_webhook(payload) is None


def test_empty_text_is_ignored() -> None:
    assert parse_spectrum_webhook(envelope(content={"type": "text", "text": "   "})) is None


def test_empty_dict_is_ignored() -> None:
    assert parse_spectrum_webhook({}) is None


def test_wrong_event_is_ignored() -> None:
    payload = envelope()
    payload["event"] = "spaces"
    assert parse_spectrum_webhook(payload) is None


@pytest.mark.parametrize(
    "payload",
    [
        {"event": "messages", "message": "bring me my water bottle"},
        {"event": "messages", "message": {"direction": "inbound", "content": "hello"}},
        {"event": "messages", "space": "spc-space-9f2", "message": {"direction": "inbound"}},
        {"event": "messages", "message": {"direction": "inbound", "sender": "+15551234567"}},
    ],
)
def test_a_string_where_a_dict_belongs_returns_none(payload: dict) -> None:
    # The simulator and hand-rolled curl calls both produce these. None, not a 500,
    # because Spectrum retries anything that is not a 2xx.
    assert parse_spectrum_webhook(payload) is None


def test_space_id_falls_back_to_the_copy_on_the_message() -> None:
    payload = envelope()
    del payload["space"]

    msg = parse_spectrum_webhook(payload)
    assert msg is not None
    assert msg.space_id == "spc-space-9f2"
    assert msg.sender == "+15551234567"


# --------------------------------------------------------------------------
# outbox
# --------------------------------------------------------------------------
class FakeMessenger(Messenger):
    """Records what reached the transport, which is the only thing the Outbox controls."""

    name = "fake"

    def __init__(self) -> None:
        self.sends: list[tuple[str, str, str | None]] = []
        self.closed = False

    async def send(self, space_id: str, text: str, *, to: str | None = None) -> None:
        self.sends.append((space_id, text, to))

    async def aclose(self) -> None:
        self.closed = True


class BrokenMessenger(Messenger):
    """A dead sidecar. Every send raises, exactly as the Messenger contract requires."""

    name = "broken"

    def __init__(self) -> None:
        self.attempts = 0

    async def send(self, space_id: str, text: str, *, to: str | None = None) -> None:
        self.attempts += 1
        raise RuntimeError("photon bridge HTTP 500: sidecar is not running")


async def test_same_key_sends_once() -> None:
    fake = FakeMessenger()
    box = Outbox(fake, min_gap_s=0)

    first = await box.send("spc-1", "found it, driving over", key="ord-7:APPROACHING")
    second = await box.send("spc-1", "found it, driving over", key="ord-7:APPROACHING")

    assert (first, second) == (True, False)
    assert len(fake.sends) == 1
    assert fake.sends[0] == ("spc-1", "found it, driving over", None)
    assert "duplicate" in box.recent()[-1]["detail"]


async def test_second_send_inside_the_quiet_gap_is_dropped() -> None:
    fake = FakeMessenger()
    box = Outbox(fake, min_gap_s=60)

    assert await box.send("spc-1", "on my way to you") is True
    assert await box.send("spc-1", "lining up") is False
    assert len(fake.sends) == 1


async def test_urgent_bypasses_the_quiet_gap() -> None:
    fake = FakeMessenger()
    box = Outbox(fake, min_gap_s=60)

    assert await box.send("spc-1", "looking for the water bottle") is True
    assert await box.send("spc-1", "I'm here", urgent=True) is True
    assert len(fake.sends) == 2


async def test_urgent_does_not_bypass_the_daily_budget() -> None:
    fake = FakeMessenger()
    box = Outbox(fake, min_gap_s=0, daily_budget=1)

    assert await box.send("spc-1", "on my way to you") is True
    assert await box.send("spc-1", "I'm here", urgent=True) is False
    assert len(fake.sends) == 1
    assert "budget" in box.recent()[-1]["detail"]


async def test_urgent_does_not_bypass_the_dedupe_key() -> None:
    fake = FakeMessenger()
    box = Outbox(fake, min_gap_s=0)

    assert await box.send("spc-1", "I'm here", key="ord-7:ARRIVED", urgent=True) is True
    assert await box.send("spc-1", "I'm here", key="ord-7:ARRIVED", urgent=True) is False
    assert len(fake.sends) == 1


async def test_a_dropped_send_is_in_the_history_with_a_reason() -> None:
    fake = FakeMessenger()
    box = Outbox(fake, min_gap_s=60)

    await box.send("spc-1", "on my way to you")
    await box.send("spc-1", "lining up")

    history = box.recent()
    assert len(history) == 2
    assert history[0]["ok"] is True
    assert history[0]["detail"] == ""
    dropped = history[1]
    assert dropped["ok"] is False
    assert dropped["text"] == "lining up"
    assert dropped["backend"] == "fake"
    assert "quiet gap" in dropped["detail"]


async def test_stats_counts_sent_and_dropped() -> None:
    fake = FakeMessenger()
    box = Outbox(fake, min_gap_s=0, daily_budget=2)

    await box.send("spc-1", "on my way to you")
    await box.send("spc-1", "I'm here")
    await box.send("spc-1", "done")  # over budget

    assert box.stats() == {
        "backend": "fake",
        "sent": 2,
        "dropped": 1,
        "budget": 2,
        "remaining": 0,
    }


async def test_recent_respects_its_limit() -> None:
    box = Outbox(FakeMessenger(), min_gap_s=0)
    for i in range(5):
        await box.send("spc-1", f"text {i}")

    tail = box.recent(2)
    assert [m["text"] for m in tail] == ["text 3", "text 4"]


async def test_a_messenger_that_raises_is_recorded_not_propagated() -> None:
    broken = BrokenMessenger()
    box = Outbox(broken, min_gap_s=0)

    assert await box.send("spc-1", "here you go", key="ord-7:PRESENTING") is False
    assert broken.attempts == 1

    dropped = box.recent()[-1]
    assert dropped["ok"] is False
    assert "sidecar is not running" in dropped["detail"]
    assert box.stats()["sent"] == 0

    # The key is only burnt on a real send, so the milestone survives a dead sidecar.
    assert await box.send("spc-1", "here you go", key="ord-7:PRESENTING") is False
    assert broken.attempts == 2


async def test_aclose_closes_the_messenger() -> None:
    fake = FakeMessenger()
    await Outbox(fake).aclose()
    assert fake.closed is True


async def test_the_day_counter_rolls_over(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeMessenger()
    box = Outbox(fake, min_gap_s=0, daily_budget=1)
    day = 86_400.0

    monkeypatch.setattr(time, "time", lambda: 10.0 * day + 100.0)
    assert await box.send("spc-1", "on my way to you") is True
    assert await box.send("spc-1", "I'm here") is False

    # A new UTC day is a fresh budget, and it has to come from the clock rather than
    # from a timer that nobody is left alive to fire.
    monkeypatch.setattr(time, "time", lambda: 11.0 * day + 100.0)
    assert await box.send("spc-1", "on my way to you") is True
    assert box.stats()["remaining"] == 0


# --------------------------------------------------------------------------
# backend selection
# --------------------------------------------------------------------------
def test_unknown_backend_degrades_to_log_with_a_note() -> None:
    messenger, notes = make_messenger("carrier-pigeon")

    assert isinstance(messenger, LogMessenger)
    assert messenger.name == "log"
    assert any("carrier-pigeon" in note for note in notes)


def test_log_backend_is_quiet() -> None:
    messenger, notes = make_messenger("log")
    assert isinstance(messenger, LogMessenger)
    assert notes == []


def test_photon_without_a_bridge_url_degrades_to_log(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("robot.messaging.PHOTON_BRIDGE_URL", "")

    messenger, notes = make_messenger("photon")
    assert isinstance(messenger, LogMessenger)
    assert notes and "photon" in notes[0]


@pytest.mark.asyncio
async def test_photon_backend_keeps_browser_phone_replies_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The web phone uses space_id=sim. That is not a Spectrum conversation, so it must
    never be POSTed to the Photon bridge — otherwise every simulated text 404s and the
    UI looks like texting does nothing."""
    from robot.messaging import BrowserPhoneMessenger, PhotonMessenger

    monkeypatch.setattr("robot.messaging.PHOTON_BRIDGE_URL", "http://127.0.0.1:8787")
    messenger, notes = make_messenger("photon")
    assert notes == []
    assert isinstance(messenger, BrowserPhoneMessenger)
    assert isinstance(messenger.remote, PhotonMessenger)

    called: list[tuple[str, str]] = []

    async def track(space_id: str, text: str, *, to: str | None = None) -> None:
        called.append((space_id, text))

    messenger.remote.send = track  # type: ignore[method-assign]
    await messenger.send("sim", "On my way to you.")
    assert called == [], "browser-phone replies must not hit the Photon bridge"

    await messenger.send("any;-;+15551234567", "On my way to you.")
    assert called == [("any;-;+15551234567", "On my way to you.")]
