from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from yt_sub_hub.config import get_settings
from yt_sub_hub.library import LibraryStore
from yt_sub_hub.models import Analysis, Transcript, TranscriptSegment, VideoMetadata


def _normalized_frame(video_id: str = "abc123", timestamp: float = 135) -> dict:
    return {
        "video_id": video_id,
        "timestamp_seconds": timestamp,
        "frame_path": f"library/videos/{video_id}/visual/frames/000135.jpg",
        "model": "vision-test-model",
        "screen_type": "candlestick_chart",
        "summary": "A candlestick chart with support and volume.",
        "visible_objects": ["candlestick chart", "volume bars", "support zone"],
        "chart_elements": {
            "ticker": "S&P500",
            "timeframe": "unknown",
            "patterns": ["breakout", "pullback"],
            "levels": ["support", "resistance"],
            "indicators": ["volume"],
        },
        "explanation": "The frame shows price attempting to hold above support.",
        "transcript_context": {"start": 115, "end": 155, "text": "context text"},
        "uncertainties": ["timeframe is not readable"],
        "confidence": 0.82,
    }


def test_get_settings_loads_visual_config(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "YT_SUB_HUB_ENABLE_VISUAL_ANALYSIS=true",
                "YT_SUB_HUB_VISION_MODEL=llava-test",
                "YT_SUB_HUB_FRAME_IMAGE_FORMAT=jpg",
                "YT_SUB_HUB_FRAME_CONTEXT_SECONDS=25",
                "YT_SUB_HUB_VISUAL_MAX_OUTPUT_TOKENS=999",
                "YT_SUB_HUB_VISUAL_BEST_EFFORT=false",
            ]
        ),
        encoding="utf-8",
    )

    settings = get_settings(env_file=env_file)

    assert settings.enable_visual_analysis is True
    assert settings.vision_model == "llava-test"
    assert settings.frame_image_format == "jpg"
    assert settings.frame_context_seconds == 25
    assert settings.visual_max_output_tokens == 999
    assert settings.visual_best_effort is False


def test_default_vision_model_is_empty(tmp_path: Path):
    settings = get_settings(env_file=tmp_path / "missing.env")

    assert settings.vision_model == ""


def test_library_visual_paths_are_stable(tmp_path: Path):
    store = LibraryStore(tmp_path)
    store.ensure_layout()

    paths = store.visual_artifact_paths("abc123", "000135")

    assert paths.video_cache == tmp_path / "assets" / "video" / "abc123.mp4"
    assert paths.visual_dir == tmp_path / "videos" / "abc123" / "visual"
    assert paths.frame == tmp_path / "videos" / "abc123" / "visual" / "frames" / "000135.jpg"
    assert paths.raw == tmp_path / "videos" / "abc123" / "visual" / "frame_000135.raw.json"
    assert paths.normalized == tmp_path / "videos" / "abc123" / "visual" / "frame_000135.normalized.json"
    assert paths.summary == tmp_path / "videos" / "abc123" / "visual" / "visual_summary.json"
    assert paths.note == tmp_path / "videos" / "abc123" / "visual" / "visual_note.md"


def test_cli_analyze_frame_uses_visual_service(monkeypatch):
    import yt_sub_hub.cli as cli

    class FakeService:
        def __init__(self):
            self.calls = []

        def analyze_frame(self, video_id, timestamp_seconds, question=None, force=False):
            self.calls.append((video_id, timestamp_seconds, question, force))
            return _normalized_frame(video_id, timestamp_seconds)

        def visual_note_path(self, video_id):
            return Path("library") / "videos" / video_id / "visual" / "visual_note.md"

        def normalized_path(self, video_id, timestamp_seconds):
            return Path("library") / "videos" / video_id / "visual" / "frame_000135.normalized.json"

    fake = FakeService()
    monkeypatch.setattr(cli, "_visual_service", lambda: fake)

    result = CliRunner().invoke(
        cli.app,
        ["analyze-frame", "abc123", "--at", "135", "--question", "What is shown?", "--force"],
    )

    assert result.exit_code == 0
    assert fake.calls == [("abc123", 135.0, "What is shown?", True)]
    assert "visual_frame_analyzed: abc123 @ 135s" in result.output
    assert "screen_type: candlestick_chart" in result.output
    assert "saved: library/videos/abc123/visual/frame_000135.normalized.json" in result.output
    assert "note: library/videos/abc123/visual/visual_note.md" in result.output


def test_api_analyze_frame_endpoint_returns_normalized_json(monkeypatch):
    from fastapi.testclient import TestClient

    import yt_sub_hub.api as api_module
    from yt_sub_hub.api import create_app

    class FakeService:
        def __init__(self, *args, **kwargs):
            pass

        def analyze_frame(self, video_id, timestamp_seconds, question=None, force=False):
            assert video_id == "abc123"
            assert timestamp_seconds == 135
            assert question == "What is shown?"
            assert force is False
            return _normalized_frame(video_id, timestamp_seconds)

    monkeypatch.setattr(api_module, "VisualAnalysisService", FakeService)

    response = TestClient(create_app()).post(
        "/videos/abc123/frames/analyze",
        json={"timestamp_seconds": 135, "question": "What is shown?", "force": False},
    )

    assert response.status_code == 200
    assert response.json()["screen_type"] == "candlestick_chart"


def test_api_analyze_frame_invalid_body_returns_422():
    from fastapi.testclient import TestClient

    from yt_sub_hub.api import create_app

    response = TestClient(create_app()).post("/videos/abc123/frames/analyze", json={})

    assert response.status_code == 422


def test_visual_summary_is_idempotently_updated(tmp_path: Path):
    store = LibraryStore(tmp_path)
    frame = _normalized_frame("abc123", 135)

    first = store.save_visual_frame("abc123", "000135", raw={"raw": True}, normalized=frame)
    second = store.save_visual_frame("abc123", "000135", raw={"raw": "updated"}, normalized=frame)

    summary = json.loads(first.summary.read_text(encoding="utf-8"))
    assert first.summary == second.summary
    assert len(summary["frames"]) == 1
    assert summary["frames"][0]["timestamp_seconds"] == 135


def test_rebuild_graph_merges_existing_visual_summary(tmp_path: Path):
    from yt_sub_hub.pipeline import VideoProcessingPipeline

    store = LibraryStore(tmp_path)
    metadata = VideoMetadata(video_id="abc123", canonical_url="https://youtu.be/abc123", title="Talk")
    transcript = Transcript(video_id="abc123", source="youtube", segments=[TranscriptSegment(0, 10, "Hello")])
    analysis = Analysis(
        language="en",
        cleaned_title="Talk",
        abstract="Abstract",
        tags=[],
        outline=[],
        sections=[],
        concepts=[],
        claims=[],
        limitations=[],
        open_questions=[],
        related_topics=[],
    )
    store.save_video_artifacts(metadata, transcript, transcript, analysis, "note")
    store.save_visual_frame("abc123", "000135", raw={"raw": True}, normalized=_normalized_frame("abc123", 135))
    pipeline = VideoProcessingPipeline.for_test(tmp_path)

    assert pipeline.rebuild_graph(None, dry_run=True) == 1

    extraction = store.load_graph_extraction("abc123")
    assert any(node["label"] == "VisualFrame" for node in extraction["nodes"])
    assert any(rel["type"] == "HAS_VISUAL_FRAME" for rel in extraction["relationships"])
