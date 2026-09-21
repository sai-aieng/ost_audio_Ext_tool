from types import SimpleNamespace
import re

from faces.exporter import export_tracks


def track(number, narrator):
    return SimpleNamespace(
        track_id=f"track-{number:04d}", best={"quality_score": 5, "timestamp_sec": 1.5},
        best_jpeg=f"test-jpeg-{number}".encode(), visible_seconds=3, moving_fraction=0.5 if narrator else 0,
        intervals=[[0, 3]], samples=6, comparisons=5, movement_sum=0.5 if narrator else 0,
        eligible=lambda config: narrator,
    )


def test_all_crops_saved_in_two_timestamped_folders(tmp_path):
    result = export_tracks([track(1, True), track(2, False), track(3, True)],
                           tmp_path, {"max_presenters": 1})
    assert re.fullmatch(r"output_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_\d{6}_UTC", result["output_folder"])
    assert len(result["faces"]) == 3
    assert len(result["narrators"]) == 2
    assert len(result["presenters"]) == 1
    assert len(result["static_faces"]) == 1
    for face in result["faces"]:
        assert (tmp_path / face["face_image"]).read_bytes() == f"test-jpeg-{int(face['track_id'].split('-')[1])}".encode()
        assert face["category"] in face["face_image"]


def test_empty_output_still_has_both_folders(tmp_path):
    result = export_tracks([], tmp_path, {"max_presenters": 3})
    assert (tmp_path / result["output_folder"] / "narrator").is_dir()
    assert (tmp_path / result["output_folder"] / "static").is_dir()


def test_uncertain_static_is_labeled(tmp_path):
    item = track(1, False)
    item.comparisons = 0
    result = export_tracks([item], tmp_path, {"max_presenters": 3})
    assert result["static_faces"][0]["classification_evidence"] == "insufficient_body_evidence"


def photo(seed, brightness=0, quality=95):
    from io import BytesIO
    import numpy as np
    from PIL import Image
    pixels = np.random.default_rng(seed).integers(30, 220, (64, 64, 3), dtype=np.uint8)
    pixels = np.clip(pixels.astype(int) + brightness, 0, 255).astype(np.uint8)
    stream = BytesIO()
    Image.fromarray(pixels).save(stream, format="JPEG", quality=quality)
    return stream.getvalue()


def test_identical_photos_across_categories_keep_narrator(tmp_path):
    static, narrator = track(1, False), track(2, True)
    static.best_jpeg = narrator.best_jpeg = photo(1)
    static.best["quality_score"] = 100
    result = export_tracks([static, narrator], tmp_path, {"max_presenters": 3})
    assert len(result["faces"]) == 1
    assert result["narrators"][0]["track_id"] == narrator.track_id
    assert result["narrators"][0]["duplicate_track_ids"] == [static.track_id]
    assert result["static_faces"] == []
    assert result["duplicate_photos_removed"] == 1
    assert len(list(tmp_path.rglob("*.jpg"))) == 1


def test_near_duplicate_keeps_better_crop_but_distinct_photo_remains(tmp_path):
    first, better, different = track(1, True), track(2, True), track(3, True)
    first.best_jpeg = photo(7, quality=94)
    better.best_jpeg = photo(7, brightness=2)
    better.best["quality_score"] = 20
    different.best_jpeg = photo(99)
    result = export_tracks([first, better, different], tmp_path, {"max_presenters": 3})
    assert {f["track_id"] for f in result["faces"]} == {better.track_id, different.track_id}
    assert result["duplicate_photos_removed"] == 1
    assert result["faces"][0]["duplicate_track_ids"] == [first.track_id]


def test_duplicate_check_is_local_to_each_extraction(tmp_path):
    item = track(1, True)
    item.best_jpeg = photo(2)
    for name in ("first", "second"):
        result = export_tracks([item], tmp_path / name, {"max_presenters": 3})
        assert len(result["faces"]) == 1
        assert result["duplicate_photos_removed"] == 0
