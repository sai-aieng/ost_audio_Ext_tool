"""Export every usable face track into a timestamped, categorized collection."""

from datetime import datetime, timezone

from faces.duplicates import unique_tracks


def export_tracks(tracks, output, config):
    stamp = datetime.now(timezone.utc).strftime("output_%Y-%m-%d_%H-%M-%S_%f_UTC")
    for category in ("narrator", "static"):
        (output / stamp / category).mkdir(parents=True, exist_ok=False)
    ranked = sorted((t for t in tracks if t.best is not None and t.best_jpeg is not None),
                    key=lambda t: (t.visible_seconds, t.moving_fraction, t.best["quality_score"]),
                    reverse=True)
    ranked, duplicates = unique_tracks(ranked, config)
    faces = []
    for track in ranked:
        category = "narrator" if track.eligible(config) else "static"
        relative = f"{stamp}/{category}/{track.track_id}.jpg"
        (output / relative).write_bytes(track.best_jpeg)
        faces.append({
            **track.best, "track_id": track.track_id, "category": category,
            "face_image": relative,
            "duplicate_track_ids": duplicates[track.track_id],
            "visible_time_sec": round(track.visible_seconds, 3),
            "appearance_intervals": [{"start_sec": round(a, 6), "end_sec": round(b, 6)}
                                     for a, b in track.intervals],
            "sampled_frames": track.samples,
            "moving_fraction": round(track.moving_fraction, 4),
            "movement_score": round(track.movement_sum / max(1, track.comparisons), 4),
            "classification_evidence": "body_movement" if category == "narrator" else
                ("low_body_movement" if track.comparisons >= 2 else "insufficient_body_evidence"),
            "selection_reason": "Likely presenter from repeated body movement." if category == "narrator"
                else "No qualifying body-movement evidence; not proof of a static photo.",
        })
    narrators = [f for f in faces if f["category"] == "narrator"]
    for rank, item in enumerate(narrators, 1):
        item.update(rank=rank, label=f"Presenter track {rank}")
    return {
        "output_folder": stamp, "faces": faces, "narrators": narrators,
        "duplicate_photos_removed": sum(map(len, duplicates.values())),
        "static_faces": [f for f in faces if f["category"] == "static"],
        # Preserve the old top-N response over the unique narrator photos.
        "presenters": narrators[:config["max_presenters"]],
    }
