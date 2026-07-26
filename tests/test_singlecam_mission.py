from __future__ import annotations

from robot.events import EventBus
from robot.singlecam_mission import SinglecamMissionRunner, should_use_singlecam_mission


def test_pill_bottle_routes_to_singlecam() -> None:
    assert should_use_singlecam_mission("my pill bottle")
    assert should_use_singlecam_mission("medicine")
    assert not should_use_singlecam_mission("water bottle")


def test_camera_fsm_event_updates_state_and_public_phase() -> None:
    bus = EventBus()
    runner = SinglecamMissionRunner()
    runner.item = "pill bottle"
    runner._order_id = "order-1"
    runner._bus = bus
    runner._accept_event(
        {
            "phase": "TARGET_CENTERING",
            "human_text": "found the pill bottle; centering it",
            "progress": 0.45,
        }
    )
    assert runner.phase == "TARGET_CENTERING"
    assert runner.progress == 0.45
    event = bus.recent()[-1]
    assert event["phase"] == "ALIGNING"
    assert event["mission_phase"] == "TARGET_CENTERING"
    assert event["order_id"] == "order-1"


def test_progress_never_moves_backwards() -> None:
    runner = SinglecamMissionRunner()
    runner.progress = 0.8
    runner._accept_event({"phase": "WINCH_DOWN", "progress": 0.2})
    assert runner.progress == 0.8


def test_failed_event_records_error() -> None:
    runner = SinglecamMissionRunner()
    runner._accept_event(
        {"phase": "FAILED", "human_text": "camera disconnected", "progress": 1.0}
    )
    assert runner.last_error == "camera disconnected"
