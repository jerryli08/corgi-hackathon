"""The basket: an item goes in on the way back, and comes out only once we arrive.

These run the real skill state machine against the simulator, so they are slower than
the rest of the suite and they are the ones that would actually catch a regression in
the stow/unstow handover between skills.py, arm.py and world.py.
"""

from __future__ import annotations

import pytest

from robot.body import make_body
from robot.events import EventBus
from robot.poses import resolve
from robot.skills import Skills
from robot.vision import HSV_PROFILES, NOT_FETCHABLE, ColorVision, _largest_blob
from robot.world import WORLD


@pytest.fixture
def rig(world):
    """A mock robot with a real arm, a real colour backend and a real event bus."""
    body, _ = make_body(mock=True)
    skills = Skills(body=body, vision=ColorVision(), bus=EventBus())
    assert body.arm.present, "these tests need CORGI_ARM_ENABLED=1 from conftest"
    yield skills, body
    body.close()


def phases_of(skills: Skills) -> list[str]:
    """Record every phase transition a skill goes through."""
    seen: list[str] = []
    original = skills._set

    def spy(task, phase, **fmt):
        if task.phase != phase:
            seen.append(phase)
        return original(task, phase, **fmt)

    skills._set = spy
    return seen


async def test_fetch_stows_into_the_basket(rig):
    skills, _ = rig
    seen = phases_of(skills)

    assert await skills.fetch("strawberries").wait() is True

    assert skills.basket == ["strawberries"]
    # The jaws are empty afterwards: the point of the basket is that the arm is free.
    assert skills.carrying is None
    assert "STOWING" in seen
    # And the simulator agrees, rather than the bookkeeping having drifted from it.
    assert WORLD.basket == ["strawberries"]
    assert WORLD.holding is None


async def test_a_stowed_item_is_not_left_on_the_floor(rig):
    """A basket item must survive an unrelated gripper-open."""
    skills, body = rig
    assert await skills.fetch("strawberries").wait() is True

    await body.gripper("open")

    assert WORLD.basket == ["strawberries"]
    dropped = [o for o in WORLD.objects if o.label == "strawberries" and not o.held]
    assert not dropped, "the basket item was released onto the floor"


async def test_deliver_takes_it_back_out_and_hands_it_over(rig):
    skills, _ = rig
    assert await skills.fetch("strawberries").wait() is True
    seen = phases_of(skills)

    assert await skills.deliver().wait() is True

    assert skills.basket == []
    assert skills.carrying is None
    assert WORLD.basket == []
    # Unstowing happens on arrival, not before the return leg.
    assert seen.index("UNSTOWING") > seen.index("RETURNING")
    assert seen.index("UNSTOWING") < seen.index("PRESENTING")


async def test_two_fetches_ride_together(rig):
    skills, _ = rig
    assert await skills.fetch("strawberries").wait() is True
    assert await skills.fetch("banana").wait() is True

    assert skills.basket == ["strawberries", "banana"]
    assert WORLD.basket == ["strawberries", "banana"]

    # First in, first out: deliver hands over the one that was asked for first.
    assert await skills.deliver().wait() is True
    assert skills.basket == ["banana"]


async def test_state_reports_the_basket(rig):
    skills, _ = rig
    assert skills.state()["basket"] == []
    assert await skills.fetch("banana").wait() is True
    assert skills.state()["basket"] == ["banana"]


# -- coming to a person ---------------------------------------------------
async def test_come_stops_short_of_the_person(rig):
    skills, _ = rig
    seen = phases_of(skills)

    assert await skills.come().wait() is True

    person = next(o for o in WORLD.snapshot()["objects"] if o["label"] == "person")
    # Beside them, not against them, and not still across the room.
    assert 0.5 <= person["distance_m"] <= 1.6, person
    assert seen[0] == "CALLED"
    assert seen[-1] == "ARRIVED"
    assert "COMING" in seen


async def test_the_person_is_not_mistaken_for_a_grocery(rig):
    """The person blob must not answer to any fetchable label, or `come` and `fetch`
    would fight over the same detection."""
    _, body = rig
    WORLD.theta = 2.356  # face the person
    frame = await body.frame()

    assert _largest_blob(frame, HSV_PROFILES["person"]) is not None

    for label, profile in HSV_PROFILES.items():
        if label in NOT_FETCHABLE:
            continue
        assert _largest_blob(frame, profile) is None, f"{label} also matches the person"


async def test_furniture_shaped_detections_are_rejected(rig, monkeypatch):
    """A wide, short box is a chair, and driving up to it is the failure to avoid."""
    skills, _ = rig
    from robot.vision import Detection

    async def wide_box(_frame, _query):
        return Detection("person", (0.2, 0.45, 0.8, 0.55), 0.9)  # 0.6 wide, 0.1 tall

    monkeypatch.setattr(skills.vision, "locate", wide_box)
    assert await skills._locate_person() is None

    async def tall_box(_frame, _query):
        return Detection("person", (0.4, 0.1, 0.6, 0.9), 0.9)  # 0.2 wide, 0.8 tall

    monkeypatch.setattr(skills.vision, "locate", tall_box)
    assert await skills._locate_person() is not None


async def test_a_missing_arm_still_records_the_item(world):
    """Drive-only mode escorts the item, and the rest of the run is identical."""
    from robot.arm import NullArm

    body, _ = make_body(mock=True)
    # Pinned directly rather than through the environment: robot.config reads os.environ
    # at import time, so setting CORGI_ARM_ENABLED here would do nothing.
    body.arm = NullArm()
    skills = Skills(body=body, vision=ColorVision(), bus=EventBus())
    try:
        assert await skills.fetch("water bottle").wait() is True
        assert skills.basket == ["water bottle"]
        assert skills.carrying is None
    finally:
        body.close()


def test_handle_pose_is_not_reach_adjusted():
    """Reach offsets exist for grasping. Applying one to the handhold would move the
    thing someone is resting a hand on."""
    for reach in ("near", "mid", "far"):
        assert resolve("HANDLE", reach) == resolve("HANDLE")
        assert resolve("STOW", reach) == resolve("STOW")
