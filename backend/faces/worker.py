"""Run only face/pose models in a child process using the current project Python."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from utils.windows_runtime import prepare_native_runtime

prepare_native_runtime()

import av
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from faces.tracking import BODY, Observation, Tracker, body_movement, clamp_box, rank_presenters
from faces.exporter import export_tracks
from faces.json_io import write_json, write_progress


def open_models(root, config):
    base = python.BaseOptions
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=base(model_asset_path=str(root / "pose_landmarker_lite.task"),
                          delegate=base.Delegate.CPU),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=config["max_poses"],
        min_pose_detection_confidence=0.4, min_pose_presence_confidence=0.4,
        output_segmentation_masks=False,
    ))
    try:
        face = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
            base_options=base(model_asset_path=str(root / "blaze_face_short_range.tflite"),
                              delegate=base.Delegate.CPU),
            running_mode=vision.RunningMode.IMAGE, min_detection_confidence=0.5,
        ))
    except Exception:
        pose.close()
        raise
    return pose, face


def image(frame):
    return mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))


def detect_bodies(frame, detector, config):
    height, width = frame.shape[:2]
    windows = [(0, 0, width, height)]
    if config["scan_tiles"]:
        # Overlapping halves improve recall of small inset presenters; not a
        # hard-coded right-hand narrator region.
        windows += [(0, 0, int(width * 0.6), height),
                    (int(width * 0.4), 0, width, height)]
    observations = []
    for x1, y1, x2, y2 in windows:
        roi = frame[y1:y2, x1:x2]
        scale = min(1.0, 960 / max(roi.shape[:2]))
        resized = cv2.resize(roi, None, fx=scale, fy=scale) if scale < 1 else roi
        result = detector.detect(image(resized))
        for landmarks in result.pose_landmarks:
            points = np.array([[p.x * (x2 - x1) + x1, p.y * (y2 - y1) + y1,
                                min(p.visibility or 0, p.presence or 0)] for p in landmarks])
            if min(points[11, 2], points[12, 2]) < 0.5:
                continue
            reliable = points[np.r_[0, BODY]]
            reliable = reliable[reliable[:, 2] >= 0.5, :2]
            if len(reliable) < 4:
                continue
            lower, upper = reliable.min(axis=0), reliable.max(axis=0)
            margin = max(8, float(np.linalg.norm(points[11, :2] - points[12, :2])) * 0.25)
            box = clamp_box([*(lower - margin), *(upper + margin)], width, height)
            obs = Observation(points, box)
            duplicate = any(float(np.linalg.norm(obs.center - other.center)) <
                            0.35 * max(obs.scale, other.scale) for other in observations)
            if not duplicate and box[2] > box[0] and box[3] > box[1]:
                observations.append(obs)
    return observations


def camera_alignment(previous, current):
    """Estimate global affine motion; reject widespread scene changes."""
    identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    if previous is None:
        return identity, False, current
    scale = min(1.0, 480 / current.shape[1])
    a = cv2.resize(previous, None, fx=scale, fy=scale)
    b = cv2.resize(current, (a.shape[1], a.shape[0]))
    points = cv2.goodFeaturesToTrack(a, maxCorners=160, qualityLevel=0.02, minDistance=10)
    affine = identity.copy()
    if points is not None and len(points) >= 8:
        matched, valid, _ = cv2.calcOpticalFlowPyrLK(a, b, points, None)
        if matched is not None and valid is not None:
            ok = valid.ravel() == 1
            if ok.sum() >= 8:
                estimate, inliers = cv2.estimateAffinePartial2D(
                    points[ok], matched[ok], method=cv2.RANSAC, ransacReprojThreshold=2.5)
                if estimate is not None and inliers is not None and inliers.mean() > 0.5:
                    zoom = float(np.linalg.norm(estimate[0, :2]))
                    if 0.8 < zoom < 1.25:
                        affine = estimate.astype(np.float32)
                        affine[:, 2] /= scale
    aligned = cv2.warpAffine(previous, affine, (current.shape[1], current.shape[0]),
                             borderMode=cv2.BORDER_REPLICATE)
    changed = np.abs(current.astype(np.float32) - aligned.astype(np.float32))
    cut = float(np.mean(changed > 35)) > 0.55
    return affine, cut, aligned


def pixel_change(aligned, gray, observation):
    x1, y1, x2, y2 = observation.box
    # Start at the shoulders to avoid counting mouth-only motion as body motion.
    y1 = max(y1, min(y2, int(observation.center[1])))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    a = cv2.GaussianBlur(aligned[y1:y2, x1:x2], (5, 5), 0)
    b = cv2.GaussianBlur(gray[y1:y2, x1:x2], (5, 5), 0)
    return float(np.mean(cv2.absdiff(a, b) > 18))


def face_crop(frame, observation, detector, timestamp):
    if observation.points[0, 2] < 0.4:
        return None
    height, width = frame.shape[:2]
    nose = observation.points[0, :2]
    radius = max(20, observation.scale)
    rx1, ry1, rx2, ry2 = clamp_box(
        [nose[0] - radius, nose[1] - radius, nose[0] + radius, nose[1] + radius], width, height)
    roi = frame[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return None
    detections = detector.detect(image(roi)).detections
    candidates = []
    for detection in detections:
        box = detection.bounding_box
        mapped = clamp_box([rx1 + box.origin_x, ry1 + box.origin_y,
                            rx1 + box.origin_x + box.width, ry1 + box.origin_y + box.height],
                           width, height)
        x1, y1, x2, y2 = mapped
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        if min(x2 - x1, y2 - y1) < 12 or np.linalg.norm(center - nose) > radius * 0.65:
            continue
        confidence = float(detection.categories[0].score)
        sharpness = float(cv2.Laplacian(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY),
                                        cv2.CV_64F).var())
        if sharpness < 8:
            continue
        quality = confidence * np.sqrt((x2 - x1) * (y2 - y1)) * np.log1p(sharpness)
        # Add 30% of the longer face dimension on every side; clip to frame.
        pad = max(x2 - x1, y2 - y1) * 0.30
        crop_box = clamp_box([x1 - pad, y1 - pad, x2 + pad, y2 + pad], width, height)
        cx1, cy1, cx2, cy2 = crop_box
        ok, encoded = cv2.imencode(".jpg", frame[cy1:cy2, cx1:cx2], [cv2.IMWRITE_JPEG_QUALITY, 95])
        if ok:
            candidates.append(({
                "timestamp_sec": round(timestamp, 6), "bbox_xyxy": mapped,
                "crop_bbox_xyxy": crop_box, "source_frame_width": width,
                "source_frame_height": height, "confidence": round(confidence, 4),
                "quality_score": round(float(quality), 4),
            }, encoded.tobytes()))
    return max(candidates, key=lambda candidate: candidate[0]["quality_score"]) if candidates else None


def add_face_only_observations(frame, detector, observations, config):
    """Find faces even when Pose cannot see a body. Do not invent body evidence."""
    height, width = frame.shape[:2]
    windows = [(0, 0, width, height)]
    if config["scan_tiles"]:
        windows += [(0, 0, int(width * 0.6), height),
                    (int(width * 0.4), 0, width, height)]
    for x1, y1, x2, y2 in windows:
        for detection in detector.detect(image(frame[y1:y2, x1:x2])).detections:
            box = detection.bounding_box
            fw, fh = box.width, box.height
            center = np.array([x1 + box.origin_x + fw / 2, y1 + box.origin_y + fh / 2])
            if min(fw, fh) < 12:
                continue
            if any(np.linalg.norm(o.points[0, :2] - center) < max(fw, fh) for o in observations):
                continue
            points = np.zeros((33, 3))
            points[0] = [*center, 1]
            # Geometry for spatial association only; confidence stays zero so
            # these synthetic shoulders cannot contribute to body movement.
            points[11, :2] = center + [-fw, fh]
            points[12, :2] = center + [fw, fh]
            bounds = clamp_box([center[0] - fw, center[1] - fh,
                                center[0] + fw, center[1] + 3 * fh], width, height)
            observations.append(Observation(points, bounds))
    return observations


def run(video, output, config):
    cv2.setNumThreads(2)
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    write_progress(output / "progress.json", {"stage": "Loading face models", "progress_pct": 1})
    root = Path(config["model_root"])
    if not root.is_absolute():
        root = BACKEND / root
    pose, face = open_models(root, config)
    tracker, previous = Tracker(), None
    samples, decoded, next_sample, last_time = 0, 0, 0.0, 0.0
    step = 1 / config["sample_rate_fps"]
    scene_cuts = 0
    try:
        with av.open(str(video)) as container:
            if not container.streams.video:
                raise ValueError("No video stream found")
            stream = container.streams.video[0]
            stream.thread_count = 2
            duration = float(stream.duration * stream.time_base) if stream.duration else float(container.duration or 0) / av.time_base
            origin = float(stream.start_time * stream.time_base) if stream.start_time is not None else None
            for frame in container.decode(stream):
                decoded += 1
                if frame.pts is None:
                    raise ValueError("Video has missing frame timestamps; remux it before face extraction")
                frame_time = float(frame.pts * frame.time_base)
                if origin is None:
                    origin = frame_time
                timestamp = max(0.0, frame_time - origin)
                last_time = timestamp
                if timestamp + 1e-6 < next_sample:
                    continue
                next_sample = timestamp + step
                bgr = frame.to_ndarray(format="bgr24")
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                if previous is not None and previous.shape != gray.shape:
                    previous = None
                    tracker.reset_scene()
                affine, cut, aligned = camera_alignment(previous, gray)
                if cut:
                    scene_cuts += 1
                    tracker.reset_scene()
                observations = detect_bodies(bgr, pose, config)
                observations = add_face_only_observations(bgr, face, observations, config)
                for track, observation in tracker.associate(observations, timestamp, max_gap=step * 2.5):
                    movement = None
                    if track.samples and previous is not None and not cut:
                        # Do not compare landmarks across a missed sampled frame.
                        if timestamp - track.last_time <= step * 1.5:
                            movement = body_movement(track.last, observation, affine)
                    end = min(timestamp + step, duration) if duration > 0 else timestamp + step
                    track.observe(observation, timestamp, max(timestamp, end), movement,
                                  pixel_change(aligned, gray, observation), config)
                    if not cut and (track.best is None or track.samples % 2 == 0):
                        candidate = face_crop(bgr, observation, face, timestamp)
                        if candidate and (track.best is None or candidate[0]["quality_score"] > track.best["quality_score"]):
                            track.best, track.best_jpeg = candidate
                previous = gray
                samples += 1
                write_progress(output / "progress.json", {
                    "stage": "Tracking body movement", "frames_processed": samples,
                    "tracks_detected": len(tracker.tracks),
                    "progress_pct": min(95.0, 5 + 90 * timestamp / duration) if duration > 0 else 5,
                    "video_timestamp_sec": round(timestamp, 3),
                })
            if samples == 0:
                raise ValueError("No decodable video frames")
            actual_end = min(duration, last_time + float(1 / stream.average_rate)) if duration > 0 and stream.average_rate else last_time + step
            for track in tracker.tracks:
                track.intervals = [[a, min(b, actual_end)] for a, b in track.intervals if a < actual_end]
    finally:
        face.close()
        pose.close()
    exported = export_tracks(tracker.tracks, output, config)
    payload = {
        **exported, "sample_rate_fps": config["sample_rate_fps"],
        "frames_sampled": samples, "frames_decoded": decoded, "tracks_detected": len(tracker.tracks),
        "scene_cuts": scene_cuts, "model": "MediaPipe Pose Landmarker Lite + BlazeFace",
        "inference_duration_sec": round(perf_counter() - started, 3),
        "warnings": [
            "Body movement selects likely presenters, not confirmed narrators.",
            "Track labels are not identities; the same person may have multiple tracks after cuts or occlusion.",
            "Visible time is a sampled estimate, not frame-exact; crop timestamps use decoded frame PTS.",
            "Static folder means no qualifying body-movement evidence; uncertain tracks are marked in classification_evidence.",
            "One best usable crop per detected track is saved, not every video frame; detection can miss faces.",
        ],
    }
    if not exported["narrators"]:
        payload["warnings"].append("No face track met the movement and crop-quality criteria.")
    write_json(output / "faces.json", payload)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run(args.video, args.output, json.loads(args.config.read_text(encoding="utf-8")))
