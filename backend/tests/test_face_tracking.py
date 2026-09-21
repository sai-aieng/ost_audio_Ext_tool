"""Deterministic face-selection tests without real model inference."""

from utils.windows_runtime import prepare_native_runtime

# Match main.py startup before importing NumPy/native dependencies on Windows.
prepare_native_runtime()

import numpy as np
import pytest

from faces.tracking import Observation, Tracker, body_movement, clamp_box, rank_presenters

CONFIG = {
    "min_pixel_change": 0.035, "movement_threshold": 0.025,
    "min_visible_seconds": 1.0, "min_moving_fraction": 0.10, "max_presenters": 3,
}
IDENTITY = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)


def person(offset=0):
    points = np.zeros((33, 3))
    points[:, 2] = 1
    for index, xy in {0: (50, 10), 11: (30, 40), 12: (70, 40),
                      13: (25, 60), 14: (75, 60), 15: (20, 85),
                      16: (80, 85), 23: (35, 100), 24: (65, 100)}.items():
        points[index, :2] = np.array(xy) + [offset, 0]
    return Observation(points, [10 + offset, 0, 90 + offset, 110])


def test_camera_translation_is_not_body_movement():
    transform = IDENTITY.copy()
    transform[0, 2] = 10
    assert body_movement(person(), person(10), transform) == pytest.approx(0)


def test_hand_motion_has_nonzero_score():
    moved = person()
    moved.points[15, 0] += 20
    assert body_movement(person(), moved, IDENTITY) > CONFIG["movement_threshold"]


def test_jitter_without_pixel_evidence_is_not_a_presenter():
    tracker = Tracker()
    for timestamp in [0, 0.5, 1, 1.5]:
        track, observation = tracker.associate([person()], timestamp)[0]
        track.observe(observation, timestamp, timestamp + 0.5, 0.2, 0, CONFIG)
        track.best = {"quality_score": 10}
    assert rank_presenters(tracker.tracks, CONFIG) == []


def test_visible_time_and_movement_select_presenter_not_static_photo():
    tracker = Tracker()
    for timestamp in [0, 0.5, 1, 1.5]:
        for track, observation in tracker.associate([person(), person(200)], timestamp):
            moving = observation.center[0] < 100
            track.observe(observation, timestamp, timestamp + 0.5,
                          0.2 if moving else 0, 0.2 if moving else 0, CONFIG)
            track.best = {"quality_score": 10}
    selected = rank_presenters(tracker.tracks, CONFIG)
    assert len(selected) == 1
    assert selected[0].visible_seconds == 2
    assert selected[0].intervals == [[0, 2]]


def test_scene_cut_and_occlusion_do_not_assert_identity():
    tracker = Tracker()
    first, obs = tracker.associate([person()], 0)[0]
    first.observe(obs, 0, 0.5, None, 0, CONFIG)
    tracker.reset_scene()
    second, _ = tracker.associate([person()], 1)[0]
    assert second.track_id != first.track_id
    third, _ = tracker.associate([person()], 10)[0]
    assert third.track_id != second.track_id


def test_box_clipping_preserves_exclusive_pixel_edges():
    assert clamp_box([-2.2, 5.1, 103, 60.3], 100, 60) == [0, 5, 100, 60]
