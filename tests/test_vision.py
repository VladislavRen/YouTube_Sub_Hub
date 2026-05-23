from __future__ import annotations

import base64
import json
import sys
import types
from pathlib import Path

import pytest

from yt_sub_hub.models import VideoMetadata
from yt_sub_hub.vision import (
    LMStudioVisionProvider,
    VisionProvider,
    parse_visual_frame_analysis_response,
)


def _metadata(video_id: str = "abc123") -> VideoMetadata:
    return VideoMetadata(
        video_id=video_id,
        canonical_url=f"https://youtu.be/{video_id}",
        title="Market structure lesson",
        channel="Trading Lab",
    )


def _provider(model: str = "llava-test") -> LMStudioVisionProvider:
    return LMStudioVisionProvider(
        base_url="http://localhost:1234/v1",
        api_key="key",
        model=model,
        max_tokens=777,
        timeout_seconds=12,
    )


def _vision_payload(**overrides):
    payload = {
        "screen_type": "candlestick_chart",
        "summary": "A candlestick chart is shown with volume and a marked support zone.",
        "visible_objects": ["candlestick chart", "volume bars", "support zone"],
        "chart_elements": {
            "ticker": "S&P500",
            "timeframe": "unknown",
            "patterns": ["pullback"],
            "levels": ["support"],
            "indicators": ["volume"],
        },
        "explanation": "The price is testing support while volume bars remain visible.",
        "uncertainties": ["The timeframe is not readable."],
        "confidence": 0.82,
    }
    payload.update(overrides)
    return payload


def test_vision_provider_protocol_is_runtime_checkable():
    provider = _provider()

    assert isinstance(provider, VisionProvider)


def test_empty_vision_model_id_raises_exact_validation_error(tmp_path: Path):
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"fake-jpeg")

    with pytest.raises(ValueError, match="^Vision model id is required for frame analysis\\.$"):
        _provider(model="").analyze_frame(
            image_path,
            _metadata(),
            {"start": 115, "end": 155, "text": "The speaker discusses support."},
            135,
        )


def test_analyze_frame_sends_image_data_url_json_schema_question_and_max_tokens(monkeypatch, tmp_path: Path):
    image_path = tmp_path / "frame.jpg"
    image_bytes = b"fake-jpeg-bytes"
    image_path.write_bytes(image_bytes)
    calls = []
    clients = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            message = type("Message", (), {"content": json.dumps(_vision_payload())})
            choice = type("Choice", (), {"message": message})
            return type("Response", (), {"choices": [choice]})

    class FakeOpenAI:
        def __init__(self, **kwargs):
            clients.append(kwargs)
            self.chat = type("Chat", (), {"completions": FakeCompletions()})

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(Client=FakeClient))

    transcript_context = {"start": 115, "end": 155, "text": "The speaker discusses support."}
    result = _provider().analyze_frame(
        image_path,
        _metadata(),
        transcript_context,
        135,
        question="What trading setup is visible?",
    )

    assert clients[0]["base_url"] == "http://localhost:1234/v1"
    assert clients[0]["api_key"] == "key"
    assert clients[0]["timeout"] == 12
    assert calls[0]["model"] == "llava-test"
    assert calls[0]["max_tokens"] == 777
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[0]["response_format"]["json_schema"]["schema"]["required"] == [
        "screen_type",
        "summary",
        "visible_objects",
        "chart_elements",
        "explanation",
        "uncertainties",
        "confidence",
    ]

    user_content = calls[0]["messages"][1]["content"]
    assert user_content[0]["type"] == "text"
    assert "Video id: abc123" in user_content[0]["text"]
    assert "Question: What trading setup is visible?" in user_content[0]["text"]
    assert "The speaker discusses support." in user_content[0]["text"]
    assert user_content[1] == {
        "type": "image_url",
        "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii"),
        },
    }

    assert result["video_id"] == "abc123"
    assert result["timestamp_seconds"] == 135.0
    assert result["frame_path"] == str(image_path)
    assert result["model"] == "llava-test"
    assert result["transcript_context"] == transcript_context
    assert result["screen_type"] == "candlestick_chart"


def test_parse_visual_frame_analysis_response_normalizes_schema_fields(tmp_path: Path):
    image_path = tmp_path / "frame.png"
    payload = _vision_payload(
        visible_objects=["chart", 42],
        chart_elements={"ticker": "BTCUSDT", "patterns": ["breakout", 7]},
        uncertainties="small text is blurry",
        confidence="0.91",
        ignored_field="ignored",
    )

    parsed = parse_visual_frame_analysis_response(
        "```json\n" + json.dumps(payload) + "\n```",
        video_id="video-1",
        timestamp_seconds=12.5,
        frame_path=image_path,
        model="vision-model",
        transcript_context={"text": "context"},
    )

    assert parsed == {
        "video_id": "video-1",
        "timestamp_seconds": 12.5,
        "frame_path": str(image_path),
        "model": "vision-model",
        "screen_type": "candlestick_chart",
        "summary": "A candlestick chart is shown with volume and a marked support zone.",
        "visible_objects": ["chart", "42"],
        "chart_elements": {"ticker": "BTCUSDT", "patterns": ["breakout", "7"]},
        "explanation": "The price is testing support while volume bars remain visible.",
        "transcript_context": {"text": "context"},
        "uncertainties": ["small text is blurry"],
        "confidence": 0.91,
    }
