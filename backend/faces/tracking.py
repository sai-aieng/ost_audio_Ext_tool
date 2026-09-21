"""Conservative within-scene tracking and body-movement scoring."""

from dataclasses import dataclass, field

import numpy as np

BODY = np.array([11, 12, 13, 14, 15, 16, 23, 24])


def clamp_box(box, width, height):
    x1, y1, x2, y2 = box
    return [max(0, min(width, int(np.floor(x1)))),
            max(0, min(height, int(np.floor(y1)))),
            max(0, min(width, int(np.ceil(x2)))),
            max(0, min(height, int(np.ceil(y2))))]


def iou(a, b):
    area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = max(0, a[2] - a[0]) * max(0, a[3] - a[1]) + max(0, b[2] - b[0]) * max(0, b[3] - b[1]) - area
    return area / union if union > 0 else 0.0


@dataclass
class Observation:
    points: np.ndarray  # 33 rows: original-frame x, y, visibility/presence
    box: list[int]

    @property
    def center(self):
        return self.points[[11, 12], :2].mean(axis=0)

    @property
    def scale(self):
        return max(12.0, float(np.linalg.norm(self.points[11, :2] - self.points[12, :2])))


def body_movement(previous, current, affine):
    """Compare reliable body joints after compensating global camera motion."""
    valid = (previous.points[BODY, 2] >= 0.5) & (current.points[BODY, 2] >= 0.5)
    if valid.sum() < 4:
        return None
    old_xy = previous.points[BODY[valid], :2]
    aligned = old_xy @ affine[:, :2].T + affine[:, 2]
    distances = np.linalg.norm(current.points[BODY[valid], :2] - aligned, axis=1)
    # Mean allows hand/arm motion to contribute when the torso is still.
    return min(1.0, float(np.mean(distances)) / current.scale)


@dataclass
class Track:
    track_id: str
    last: Observation
    last_time: float
    intervals: list[list[float]] = field(default_factory=list)
    samples: int = 0
    comparisons: int = 0
    moving_comparisons: int = 0
    movement_sum: float = 0.0
    best: dict | None = None
    best_jpeg: bytes | None = None

    @property
    def visible_seconds(self):
        return sum(end - start for start, end in self.intervals)

    @property
    def moving_fraction(self):
        return self.moving_comparisons / max(1, self.comparisons)

    def observe(self, observation, timestamp, interval_end, movement, pixel_change, config):
        if self.intervals and timestamp <= self.intervals[-1][1] + 1e-6:
            self.intervals[-1][1] = max(self.intervals[-1][1], interval_end)
        else:
            self.intervals.append([timestamp, interval_end])
        if movement is not None:
            self.comparisons += 1
            # Require image evidence too: stationary model jitter is not motion.
            if pixel_change >= config["min_pixel_change"]:
                self.movement_sum += movement
                if movement >= config["movement_threshold"]:
                    self.moving_comparisons += 1
        self.samples += 1
        self.last = observation
        self.last_time = timestamp

    def eligible(self, config):
        return (self.best is not None
                and self.visible_seconds >= config["min_visible_seconds"]
                and self.moving_comparisons >= 4
                and self.moving_fraction >= config["min_moving_fraction"])


class Tracker:
    def __init__(self):
        self.tracks = []
        self.active = []

    def reset_scene(self):
        self.active = []

    def associate(self, observations, timestamp, max_gap=1.25):
        self.active = [t for t in self.active if timestamp - t.last_time <= max_gap]
        pairs = []
        for index, observation in enumerate(observations):
            for track in self.active:
                ratio = observation.scale / track.last.scale
                distance = float(np.linalg.norm(observation.center - track.last.center)) / max(observation.scale, track.last.scale)
                overlap = iou(observation.box, track.last.box)
                if 0.5 <= ratio <= 2 and distance < 0.9 and overlap > 0.1:
                    pairs.append((distance + 1 - overlap, index, track))
        used_observations, used_tracks, matches = set(), set(), {}
        for _, index, track in sorted(pairs, key=lambda item: item[0]):
            if index not in used_observations and track.track_id not in used_tracks:
                matches[index] = track
                used_observations.add(index)
                used_tracks.add(track.track_id)
        for index, observation in enumerate(observations):
            if index not in matches:
                if len(self.tracks) >= 2000:
                    raise RuntimeError("Too many presenter tracks; use a shorter video.")
                track = Track(f"track-{len(self.tracks) + 1:04d}", observation, timestamp)
                self.tracks.append(track)
                self.active.append(track)
                matches[index] = track
        return [(matches[index], observation) for index, observation in enumerate(observations)]


def rank_presenters(tracks, config):
    return sorted((t for t in tracks if t.eligible(config)),
                  key=lambda t: (t.visible_seconds, t.moving_fraction, t.best["quality_score"]),
                  reverse=True)[:config["max_presenters"]]
