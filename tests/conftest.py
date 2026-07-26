"""Test wiring.

Everything here has to work with no hardware and no network, so the environment is
pinned before any `robot.*` module is imported -- `robot.config` reads os.environ at
import time and there is exactly one chance to get that right.
"""

from __future__ import annotations

import os

# Must happen before the first `import robot.*` anywhere in the test run.
os.environ.update(
    {
        "CORGI_MOCK": "1",
        "CORGI_ARM_ENABLED": "1",  # exercise the real grasp/stow path, not NullArm
        "CORGI_CAMERA_ENABLED": "1",
        "CORGI_VISION_BACKEND": "color",
        "CORGI_MESSAGING_BACKEND": "log",
        "CORGI_ROUTER_BACKEND": "keyword",
        "CORGI_ALLOW_SIMULATED_TEXTS": "1",
        "CORGI_ALLOWED_SENDERS": "",
        # Set so the webhook's happy path is reachable in tests. An unset secret is a
        # configuration error that rejects everything, which is also worth testing.
        "CORGI_PHOTON_WEBHOOK_SECRET": "test-secret-0123456789abcdef",
        # Tests assert on individual sends; the anti-spam gap would eat them.
        "CORGI_TEXT_MIN_GAP_S": "0",
        # Keep the skill loops short so a full fetch fits in a test.
        "CORGI_MAX_SEARCH_STEPS": "24",
        "CORGI_MAX_SERVO_STEPS": "40",
    }
)

import httpx
import pytest
import pytest_asyncio


@pytest.fixture
def world():
    """The simulator's ground truth, reset before each test that touches it."""
    from robot.world import WORLD

    WORLD.reset()
    WORLD.grasp_failure_rate = 0.0
    yield WORLD
    WORLD.reset()


@pytest_asyncio.fixture
async def app_client():
    """An httpx client speaking ASGI directly to the server, lifespan included.

    Importing robot.server builds the whole robot at module scope, so this is also the
    only place that proves the singletons come up in the right order.
    """
    from robot.server import app

    transport = httpx.ASGITransport(app=app)
    # The lifespan hooks own the order queue and the phase follower; without them
    # nothing gets fulfilled and half the endpoints answer from a cold cache.
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
        app.router.lifespan_context(app),
    ):
        yield client
