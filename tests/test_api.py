"""The HTTP surface, against the real app with its lifespan running.

Importing robot.server builds the whole robot at module scope, so these tests are also
the only thing that proves the singletons and the lifespan hooks come up in the right
order -- which is exactly the kind of breakage that only shows up on a real boot.
"""

from __future__ import annotations

import json
import time

from robot.config import PHOTON_WEBHOOK_SECRET
from robot.messaging import HEADER_SIGNATURE, HEADER_TIMESTAMP, spectrum_signature


def envelope(text: str = "bring me my water bottle") -> dict:
    """A realistic Spectrum inbound-text webhook body."""
    return {
        "event": "messages",
        "space": {"id": "spc-1", "platform": "iMessage", "type": "dm", "phone": "+15551234567"},
        "message": {
            "id": "spc-msg-1",
            "platform": "iMessage",
            "direction": "inbound",
            "timestamp": "2026-07-25T18:03:11Z",
            "sender": {"id": "+15551234567", "platform": "iMessage"},
            "space": {"id": "spc-1", "platform": "iMessage", "type": "dm"},
            "content": {"type": "text", "text": text},
        },
    }


def signed(payload: dict) -> tuple[bytes, dict[str, str]]:
    raw = json.dumps(payload).encode()
    ts = str(int(time.time()))
    return raw, {
        HEADER_TIMESTAMP: ts,
        HEADER_SIGNATURE: spectrum_signature(PHOTON_WEBHOOK_SECRET, ts, raw),
        "content-type": "application/json",
    }


# -- health and state -----------------------------------------------------
async def test_health_reports_every_subsystem(app_client):
    health = (await app_client.get("/api/health")).json()

    assert health["ok"] is True
    assert health["mock"] is True
    # The pivot's subsystems have to be visible here or there is no way to tell, on
    # stage, whether the robot is texting through Photon or printing to a terminal.
    assert health["messaging"]["backend"] == "log"
    assert health["router"]["backend"] == "keyword"
    assert health["walker"]["active"] is False
    assert health["basket"] == []


async def test_state_reports_the_basket_and_the_walker(app_client):
    state = (await app_client.get("/api/state")).json()

    assert state["robot"]["basket"] == []
    assert state["walker"]["active"] is False
    assert state["queued"] == 0


# -- the webhook ----------------------------------------------------------
async def test_webhook_rejects_an_unsigned_body(app_client):
    res = await app_client.post("/api/imessage/webhook", json=envelope())
    assert res.status_code == 401
    # The message has to name the check that failed -- a 401 that does not say whether
    # the secret, the clock or the header is wrong is unfixable at three in the morning.
    detail = res.json()["detail"].lower()
    assert HEADER_TIMESTAMP.lower() in detail or HEADER_SIGNATURE.lower() in detail


async def test_webhook_rejects_a_tampered_body(app_client):
    raw, headers = signed(envelope())
    res = await app_client.post(
        "/api/imessage/webhook", content=raw.replace(b"water", b"vodka."), headers=headers
    )
    assert res.status_code == 401


async def test_webhook_accepts_a_signed_text(app_client, world):
    raw, headers = signed(envelope())
    res = await app_client.post("/api/imessage/webhook", content=raw, headers=headers)

    assert res.status_code == 200
    body = res.json()
    assert body.get("ignored") is not True
    assert body["intent"]["kind"] == "fetch"
    assert body["intent"]["item"] == "water bottle"


async def test_webhook_ignores_what_it_does_not_act_on(app_client):
    """A reaction or an outbound echo is not an error. Spectrum retries a non-2xx, and
    there is nothing here worth retrying."""
    for payload in (
        {"event": "messages", "message": {"direction": "outbound", "content": {"type": "text"}}},
        {"event": "messages", "message": {"direction": "inbound", "content": {"type": "reaction"}}},
        {"event": "something_else"},
        {},
    ):
        raw, headers = signed(payload)
        res = await app_client.post("/api/imessage/webhook", content=raw, headers=headers)
        assert res.status_code == 200, payload
        assert res.json()["ignored"] is True, payload


async def test_webhook_survives_a_body_that_is_not_json(app_client):
    ts = str(int(time.time()))
    raw = b"this is not json"
    headers = {
        HEADER_TIMESTAMP: ts,
        HEADER_SIGNATURE: spectrum_signature(PHOTON_WEBHOOK_SECRET, ts, raw),
    }
    res = await app_client.post("/api/imessage/webhook", content=raw, headers=headers)
    assert res.status_code == 200
    assert res.json()["ignored"] is True


# -- the simulated phone --------------------------------------------------
async def test_simulate_runs_the_same_path_as_a_real_text(app_client, world):
    res = await app_client.post("/api/imessage/simulate", json={"text": "come here please"})

    assert res.status_code == 200
    body = res.json()
    assert body["intent"]["kind"] == "come"
    assert body["reply"]


async def test_the_log_shows_both_sides(app_client, world):
    await app_client.post("/api/imessage/simulate", json={"text": "hello there"})
    log = (await app_client.get("/api/imessage/log")).json()

    assert "inbound" in log and "outbound" in log
    assert log["stats"]["backend"] == "log"
    # Whatever the robot said back is on the record, sent or deliberately withheld.
    assert log["outbound"]


# -- the corgi/ bridge relay -----------------------------------------------
async def test_relay_runs_the_same_path_as_a_real_text(app_client, world):
    """This is what the corgi/ bridge process posts once it holds the live Spectrum
    connection -- a real text, not a simulated one."""
    res = await app_client.post(
        "/api/imessage/relay",
        json={"space_id": "spc-real-1", "sender": "+15551234567", "text": "come here"},
    )

    assert res.status_code == 200
    body = res.json()
    assert body["intent"]["kind"] == "come"
    assert body["reply"]


async def test_relay_requires_no_secret_when_none_is_configured(app_client):
    # CORGI_BRIDGE_SECRET is unset in the test environment (see conftest.py), so a
    # relay with no header at all must still be accepted.
    res = await app_client.post(
        "/api/imessage/relay",
        json={"space_id": "spc-real-2", "sender": "+15551234567", "text": "status"},
    )
    assert res.status_code == 200


# -- the router preview ---------------------------------------------------
async def test_preview_reports_a_decision(app_client):
    res = await app_client.post(
        "/api/router/preview", json={"text": "could you get the granola bar"}
    )

    assert res.status_code == 200
    intent = res.json()
    assert intent["kind"] == "fetch"
    assert intent["item"] == "granola bar"
    assert intent["route"]["backend"] == "keyword"


async def test_preview_has_no_side_effects(app_client, world):
    before = len((await app_client.get("/api/orders")).json())
    sent_before = len((await app_client.get("/api/imessage/log")).json()["outbound"])

    await app_client.post("/api/router/preview", json={"text": "bring me my water bottle"})

    assert len((await app_client.get("/api/orders")).json()) == before
    assert len((await app_client.get("/api/imessage/log")).json()["outbound"]) == sent_before


# -- walker ---------------------------------------------------------------
async def test_walker_round_trip(app_client):
    try:
        started = (await app_client.post("/api/walker/start")).json()
        assert started["ok"] is True
        assert started["state"]["active"] is True
        # Nothing moves until someone asks it to.
        assert started["state"]["moving"] is False

        nudged = (await app_client.post("/api/walker/nudge", json={"direction": "forward"})).json()
        assert nudged["ok"] is True
        assert nudged["state"]["moving"] is True
        assert nudged["state"]["deadman_ms_left"] > 0

        held = (await app_client.post("/api/walker/hold")).json()
        assert held["state"]["active"] is True
        assert held["state"]["moving"] is False
    finally:
        stopped = (await app_client.post("/api/walker/stop")).json()
        assert stopped["state"]["active"] is False


async def test_walker_refuses_an_unknown_direction(app_client):
    try:
        await app_client.post("/api/walker/start")
        res = await app_client.post("/api/walker/nudge", json={"direction": "sideways"})
        assert res.json()["ok"] is False
    finally:
        await app_client.post("/api/walker/stop")


async def test_nudging_a_stopped_walker_does_nothing(app_client):
    res = await app_client.post("/api/walker/nudge", json={"direction": "forward"})
    assert res.json()["ok"] is False
    assert res.json()["state"]["active"] is False


async def test_estop_ends_walker_mode(app_client):
    await app_client.post("/api/walker/start")
    await app_client.post("/api/walker/nudge", json={"direction": "forward"})

    assert (await app_client.post("/api/estop")).json()["ok"] is True

    assert (await app_client.get("/api/walker/state")).json()["active"] is False


# -- skills ---------------------------------------------------------------
async def test_come_starts_a_task(app_client, world):
    res = await app_client.post("/api/skills/come")
    assert res.status_code == 200
    task = res.json()
    assert task["task_id"].startswith("t_")

    status = (await app_client.get(f"/api/tasks/{task['task_id']}")).json()
    assert status["task_id"] == task["task_id"]
    await app_client.post("/api/estop")


async def test_unknown_task_is_a_404(app_client):
    assert (await app_client.get("/api/tasks/t_nope")).status_code == 404


# -- pages ----------------------------------------------------------------
async def test_the_pages_and_their_assets_are_reachable(app_client):
    """The HTML asks for /static/*, so a mount that does not line up is a blank page."""
    for path in ("/", "/ops", "/static/app.js", "/static/styles.css", "/static/ops.js"):
        res = await app_client.get(path)
        assert res.status_code == 200, path
