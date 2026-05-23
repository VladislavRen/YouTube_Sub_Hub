from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from yt_sub_hub.config import get_settings
from yt_sub_hub.library import LibraryStore
from yt_sub_hub.models import Transcript, TranscriptSegment, VideoMetadata
from yt_sub_hub.visual import (
    VisualAnalysisError,
    VisualAnalysisService,
    analyze_visual_frame,
    build_ffmpeg_frame_command,
    build_ytdlp_video_command,
    download_video_with_ytdlp,
    extract_frame_with_ffmpeg,
    normalize_visual_analysis,
    timestamp_key,
    transcript_window,
)


def test_timestamp_key_uses_six_digit_seconds_and_millisecond_suffix():
    assert timestamp_key(135) == "000135"
    assert timestamp_key(135.4) == "000135_400"
    assert timestamp_key(0.045) == "000000_045"


def test_transcript_window_returns_context_text_with_clamped_start_and_end():
    transcript = Transcript(
        video_id="abc123",
        source="youtube",
        segments=[
            TranscriptSegment(start=90, duration=10, text="before window"),
            TranscriptSegment(start=110, duration=5, text="first context"),
            TranscriptSegment(start=133, duration=4, text="target context"),
            TranscriptSegment(start=156, duration=6, text="last context"),
            TranscriptSegment(start=170, duration=3, text="after window"),
        ],
    )

    window = transcript_window(transcript, at_seconds=135, context_seconds=25)

    assert window.start == 110
    assert window.end == 160
    assert window.text == "first context\ntarget context\nlast context"


def test_normalize_visual_analysis_fills_defaults_and_serializes():
    context = transcript_window(
        Transcript("abc123", "youtube", [TranscriptSegment(130, 5, "breakout context")]),
        at_seconds=135,
        context_seconds=10,
    )

    analysis = normalize_visual_analysis(
        {
            "screen_type": "candlestick_chart",
            "summary": "A chart shows price holding support.",
            "visible_objects": ["candles", 123, ""],
            "chart_elements": {"ticker": "SPY", "patterns": ["breakout"]},
            "confidence": "0.8",
        },
        video_id="abc123",
        timestamp_seconds=135.4,
        frame_path=Path("library/videos/abc123/visual/frames/000135_400.jpg"),
        model="vision-test",
        transcript_context=context,
    )

    assert analysis.timestamp_key == "000135_400"
    assert analysis.visible_objects == ["candles", "123"]
    assert analysis.chart_elements["ticker"] == "SPY"
    assert analysis.chart_elements["patterns"] == ["breakout"]
    assert analysis.chart_elements["levels"] == []
    assert analysis.confidence == 0.8
    assert analysis.to_dict()["transcript_context"] == {
        "start": 125,
        "end": 145,
        "text": "breakout context",
    }


def test_visual_analysis_error_includes_actionable_next_step():
    error = VisualAnalysisError("ffmpeg failed", action="Install ffmpeg and retry frame extraction.")

    assert str(error) == "ffmpeg failed Action: Install ffmpeg and retry frame extraction."
    assert error.action == "Install ffmpeg and retry frame extraction."


def test_frame_extraction_helpers_build_commands_and_use_runner(tmp_path: Path):
    calls = []
    video_path = tmp_path / "video.mp4"
    frame_path = tmp_path / "frames" / "000135.jpg"

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        frame_path.write_bytes(b"jpg")

    assert build_ffmpeg_frame_command(video_path, 135.4, frame_path) == [
        "ffmpeg",
        "-y",
        "-ss",
        "135.4",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        str(frame_path),
    ]
    assert extract_frame_with_ffmpeg(video_path, 135.4, frame_path, runner=runner) == frame_path
    assert calls == [
        (
            build_ffmpeg_frame_command(video_path, 135.4, frame_path),
            {"check": True, "capture_output": True, "text": True},
        )
    ]


def test_video_download_helper_builds_yt_dlp_command_and_uses_runner(tmp_path: Path):
    calls = []
    output_path = tmp_path / "assets" / "video" / "abc123.mp4"

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"mp4")

    command = build_ytdlp_video_command("https://youtu.be/abc123", output_path)

    assert command == [
        "yt-dlp",
        "-f",
        "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "-o",
        str(output_path),
        "https://youtu.be/abc123",
    ]
    assert download_video_with_ytdlp("https://youtu.be/abc123", output_path, runner=runner) == output_path
    assert calls == [(command, {"check": True, "capture_output": True, "text": True})]


def test_analyze_visual_frame_uses_cache_unless_force_and_accepts_injected_callbacks(tmp_path: Path):
    calls = []
    cached = {"video_id": "abc123", "timestamp_seconds": 135, "screen_type": "cached"}

    class Store:
        def load_visual_frame(self, video_id, key):
            calls.append(("load", video_id, key))
            return cached

        def visual_artifact_paths(self, video_id, key):
            calls.append(("paths", video_id, key))
            return {"frame": tmp_path / f"{key}.jpg", "video_cache": tmp_path / "video.mp4"}

        def save_visual_frame(self, video_id, key, raw, normalized):
            calls.append(("save", video_id, key, raw, normalized["screen_type"]))
            return normalized

    def extractor(*args, **kwargs):
        raise AssertionError("extractor should not run when cache is valid")

    def provider(*args, **kwargs):
        raise AssertionError("provider should not run when cache is valid")

    assert analyze_visual_frame(
        "abc123",
        135,
        store=Store(),
        extractor=extractor,
        provider=provider,
    ) == cached

    raw = {"screen_type": "chart", "summary": "new"}

    def forced_extractor(**kwargs):
        calls.append(("extract", kwargs["video_id"], kwargs["timestamp_key"]))
        return tmp_path / "000135.jpg"

    def forced_provider(**kwargs):
        calls.append(("provider", kwargs["video_id"], kwargs["timestamp_seconds"]))
        return raw

    result = analyze_visual_frame(
        "abc123",
        135,
        store=Store(),
        extractor=forced_extractor,
        provider=forced_provider,
        force=True,
        model="vision-test",
    )

    assert result["screen_type"] == "chart"
    assert calls[-4:] == [
        ("paths", "abc123", "000135"),
        ("extract", "abc123", "000135"),
        ("provider", "abc123", 135),
        ("save", "abc123", "000135", raw, "chart"),
    ]


def test_visual_service_normalizes_provider_result_even_when_raw_wrapper_exists(tmp_path: Path):
    store = LibraryStore(tmp_path)
    store.ensure_layout()
    video_id = "abc123"
    video_dir = store.video_dir(video_id)
    video_dir.mkdir(parents=True, exist_ok=True)
    metadata = VideoMetadata(video_id=video_id, canonical_url="https://youtu.be/abc123", title="Market video")
    transcript = Transcript(video_id, "youtube", [TranscriptSegment(130, 10, "The speaker discusses a breakout.")])
    (video_dir / "metadata.json").write_text(json.dumps(metadata.to_dict()), encoding="utf-8")
    (video_dir / "transcript.normalized.json").write_text(json.dumps(transcript.to_dict()), encoding="utf-8")

    class Extractor:
        def extract_frame(self, metadata, output_path, timestamp_seconds, force=False):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"jpg")
            return output_path

    class Provider:
        last_raw_response = {
            "model": "gemma-vision",
            "prompt_version": "vision-frame-v1",
            "content": '{"screen_type":"candlestick_chart"}',
        }

        def analyze_frame(self, image_path, metadata, transcript_context, timestamp_seconds, question=None):
            return {
                "screen_type": "candlestick_chart",
                "summary": "A chart shows a bullish breakout.",
                "visible_objects": ["candles", "volume bars"],
                "chart_elements": {
                    "ticker": "S&P500",
                    "timeframe": "unknown",
                    "patterns": ["breakout"],
                    "levels": ["resistance"],
                    "indicators": ["volume"],
                },
                "explanation": "The frame shows price breaking above resistance.",
                "uncertainties": [],
                "confidence": 0.9,
            }

    settings = replace(get_settings(env_file=tmp_path / "missing.env"), library_dir=tmp_path, vision_model="gemma-vision")
    service = VisualAnalysisService(settings=settings, store=store, provider=Provider(), extractor=Extractor())

    result = service.analyze_frame(video_id, 135, force=True)

    assert result["screen_type"] == "candlestick_chart"
    assert result["summary"] == "A chart shows a bullish breakout."
    assert result["visible_objects"] == ["candles", "volume bars"]
    assert result["chart_elements"]["patterns"] == ["breakout"]
    paths = store.visual_artifact_paths(video_id, "000135")
    raw = json.loads(paths.raw.read_text(encoding="utf-8"))
    normalized = json.loads(paths.normalized.read_text(encoding="utf-8"))
    assert raw["prompt_version"] == "vision-frame-v1"
    assert "content" in raw
    assert normalized["screen_type"] == "candlestick_chart"
    assert normalized["summary"] == "A chart shows a bullish breakout."
    assert normalized["visible_objects"] == ["candles", "volume bars"]
    assert normalized["chart_elements"]["patterns"] == ["breakout"]


def test_visual_translation_saves_source_and_russian_normalized_summary_note(tmp_path: Path):
    store = LibraryStore(tmp_path)
    store.ensure_layout()
    video_id = "abc123"
    video_dir = store.video_dir(video_id)
    video_dir.mkdir(parents=True, exist_ok=True)
    metadata = VideoMetadata(video_id=video_id, canonical_url="https://youtu.be/abc123", title="Market video")
    transcript = Transcript(video_id, "youtube", [TranscriptSegment(130, 10, "El precio rompe resistencia.")])
    (video_dir / "metadata.json").write_text(json.dumps(metadata.to_dict()), encoding="utf-8")
    (video_dir / "transcript.normalized.json").write_text(json.dumps(transcript.to_dict()), encoding="utf-8")

    class Extractor:
        def extract_frame(self, metadata, output_path, timestamp_seconds, force=False):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"jpg")
            return output_path

    class Provider:
        last_raw_response = {"model": "gemma-vision", "content": "{}"}

        def analyze_frame(self, image_path, metadata, transcript_context, timestamp_seconds, question=None):
            return {
                "screen_type": "candlestick_chart",
                "summary": "A chart shows a bullish breakout.",
                "visible_objects": ["candles", "volume bars"],
                "chart_elements": {
                    "ticker": "S&P500",
                    "timeframe": "unknown",
                    "patterns": ["breakout"],
                    "levels": ["resistance"],
                    "indicators": ["volume"],
                },
                "explanation": "The frame shows price breaking above resistance.",
                "uncertainties": ["timeframe unclear"],
                "confidence": 0.9,
            }

    class Translator:
        def translate_visual_frame(self, frame, target_language):
            assert target_language == "ru"
            translated = dict(frame)
            translated["language"] = "ru"
            translated["translated_from"] = "en"
            translated["screen_type"] = "свечной график"
            translated["summary"] = "График показывает бычий пробой."
            translated["visible_objects"] = ["свечи", "столбцы объема"]
            translated["chart_elements"] = {
                **frame["chart_elements"],
                "patterns": ["пробой"],
                "levels": ["сопротивление"],
                "indicators": ["объем"],
            }
            translated["explanation"] = "Цена пробивает сопротивление."
            translated["uncertainties"] = ["таймфрейм неясен"]
            return translated

    settings = replace(
        get_settings(env_file=tmp_path / "missing.env"),
        library_dir=tmp_path,
        vision_model="gemma-vision",
        enable_translation=True,
        target_language="ru",
    )
    service = VisualAnalysisService(
        settings=settings,
        store=store,
        provider=Provider(),
        extractor=Extractor(),
        translation_provider=Translator(),
    )

    result = service.analyze_frame(video_id, 135, force=True)

    paths = store.visual_artifact_paths(video_id, "000135")
    source = json.loads(paths.normalized_source.read_text(encoding="utf-8"))
    normalized = json.loads(paths.normalized.read_text(encoding="utf-8"))
    summary = json.loads(paths.summary.read_text(encoding="utf-8"))
    note = paths.note.read_text(encoding="utf-8")
    assert source["summary"] == "A chart shows a bullish breakout."
    assert result["summary"] == "График показывает бычий пробой."
    assert normalized["summary"] == "График показывает бычий пробой."
    assert normalized["chart_elements"]["ticker"] == "S&P500"
    assert summary["frames"][0]["summary"] == "График показывает бычий пробой."
    assert "График показывает бычий пробой." in note


def test_visual_translation_merges_russian_summary_into_graph(tmp_path: Path):
    store = LibraryStore(tmp_path)
    store.ensure_layout()
    video_id = "abc123"
    video_dir = store.video_dir(video_id)
    video_dir.mkdir(parents=True, exist_ok=True)
    metadata = VideoMetadata(video_id=video_id, canonical_url="https://youtu.be/abc123", title="Market video")
    transcript = Transcript(video_id, "youtube", [TranscriptSegment(130, 10, "El precio rompe resistencia.")])
    (video_dir / "metadata.json").write_text(json.dumps(metadata.to_dict()), encoding="utf-8")
    (video_dir / "transcript.normalized.json").write_text(json.dumps(transcript.to_dict()), encoding="utf-8")
    store.save_graph_extraction(
        video_id,
        {
            "schema_version": "graph-extraction-v1",
            "video_id": video_id,
            "nodes": [{"id": f"video:{video_id}", "label": "Video", "video_id": video_id}],
            "relationships": [],
            "quality_report": {},
        },
    )

    class Extractor:
        def extract_frame(self, metadata, output_path, timestamp_seconds, force=False):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"jpg")
            return output_path

    class Provider:
        last_raw_response = {"model": "gemma-vision", "content": "{}"}

        def analyze_frame(self, image_path, metadata, transcript_context, timestamp_seconds, question=None):
            return {"screen_type": "chart", "summary": "English summary.", "confidence": 0.8}

    class Translator:
        def translate_visual_frame(self, frame, target_language):
            translated = dict(frame)
            translated["language"] = target_language
            translated["translated_from"] = "en"
            translated["summary"] = "Русское визуальное наблюдение."
            return translated

    settings = replace(
        get_settings(env_file=tmp_path / "missing.env"),
        library_dir=tmp_path,
        vision_model="gemma-vision",
        enable_translation=True,
        target_language="ru",
    )
    service = VisualAnalysisService(
        settings=settings,
        store=store,
        provider=Provider(),
        extractor=Extractor(),
        translation_provider=Translator(),
    )

    service.analyze_frame(video_id, 135, force=True)

    extraction = store.load_graph_extraction(video_id)
    visual_observations = [node for node in extraction["nodes"] if node["label"] == "VisualObservation"]
    assert any(node["text"] == "Русское визуальное наблюдение." for node in visual_observations)


def test_analyze_visual_frame_wraps_failures_in_visual_analysis_error():
    def failing_provider(**kwargs):
        raise RuntimeError("model offline")

    with pytest.raises(VisualAnalysisError, match="model offline.*Check the visual analysis provider"):
        analyze_visual_frame(
            "abc123",
            135,
            store=None,
            extractor=lambda **kwargs: Path("frame.jpg"),
            provider=failing_provider,
        )
