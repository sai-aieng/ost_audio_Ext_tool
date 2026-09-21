"""Verify adapter settings and complete consumption of lazy transcription."""

from types import SimpleNamespace

from pipeline.faster_whisper_service import FasterWhisperService


def test_faster_whisper_preserves_options_and_consumes_segments(monkeypatch):
    consumed = []

    class FakeModel:
        def transcribe(self, path, **options):
            assert options["beam_size"] == 1
            assert options["best_of"] == 1
            assert options["vad_filter"] is False
            assert options["without_timestamps"] is False
            assert options["condition_on_previous_text"] is True
            assert options["language"] == "en"
            assert options["temperature"] == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

            def segments():
                consumed.append(True)
                yield SimpleNamespace(start=1.25, end=2.5, text=" Hello ")
                yield SimpleNamespace(start=2.5, end=3.0, text=" ")
                yield SimpleNamespace(start=3.5, end=4.75, text="World")

            return segments(), SimpleNamespace(language="en")

    service = FasterWhisperService()
    monkeypatch.setattr(service, "get_model", lambda config: FakeModel())
    language, segments = service.transcribe("test.mp4", {"language": "en"})
    assert consumed == [True]
    assert language == "en"
    assert [(s.start_sec, s.end_sec, s.text) for s in segments] == [
        (1.25, 2.5, "Hello"), (3.5, 4.75, "World"),
    ]
