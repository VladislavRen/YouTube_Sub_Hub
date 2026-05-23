from __future__ import annotations

import json
import sys
import types

import pytest

from yt_sub_hub.models import Analysis, Claim, Concept, NoteSection, Transcript, TranscriptSegment
from yt_sub_hub.translation import (
    LMStudioTranslationProvider,
    TranslationError,
    TranslationProvider,
    build_analysis_translation_response_format,
    build_transcript_translation_response_format,
)


class FakeTranslationProvider(TranslationProvider):
    def __init__(self):
        self.text_calls = []

    def translate_text(self, text: str, source_language: str | None, target_language: str) -> str:
        self.text_calls.append((text, source_language, target_language))
        return f"ru:{text}"


def _analysis() -> Analysis:
    return Analysis(
        language="es",
        cleaned_title="Titulo",
        abstract="Resumen",
        tags=["mercado"],
        outline=["Idea"],
        sections=[NoteSection(heading="Seccion", summary="Texto", timestamp_seconds=42, bullets=["Punto"])],
        concepts=[Concept(name="Soporte", definition="Zona")],
        claims=[Claim(text="El soporte importa.", evidence="Se dice soporte.")],
        limitations=["Limite"],
        open_questions=["Pregunta"],
        related_topics=["Trading"],
        prompt_version="scientific-note-v1",
        model="analysis-model",
    )


def test_translate_transcript_preserves_timestamps_and_segment_count():
    provider = FakeTranslationProvider()
    transcript = Transcript(
        video_id="abc",
        source="whisper",
        language="es",
        segments=[
            TranscriptSegment(start=0, duration=4.2, text="hola"),
            TranscriptSegment(start=4.2, duration=1.5, text="mundo"),
        ],
    )

    translated = provider.translate_transcript(transcript, "ru")

    assert translated.video_id == "abc"
    assert translated.source == "whisper"
    assert translated.language == "ru"
    assert [segment.text for segment in translated.segments] == ["ru:hola", "ru:mundo"]
    assert [(segment.start, segment.duration) for segment in translated.segments] == [(0, 4.2), (4.2, 1.5)]
    assert len(translated.segments) == len(transcript.segments)


def test_translate_analysis_preserves_timestamps_and_model_metadata():
    provider = FakeTranslationProvider()

    translated = provider.translate_analysis(_analysis(), "ru")

    assert translated.language == "ru"
    assert translated.cleaned_title == "ru:Titulo"
    assert translated.sections[0].timestamp_seconds == 42
    assert translated.sections[0].heading == "ru:Seccion"
    assert translated.concepts[0].name == "ru:Soporte"
    assert translated.claims[0].text == "ru:El soporte importa."
    assert translated.prompt_version == "scientific-note-v1"
    assert translated.model == "analysis-model"


def test_translate_visual_frame_preserves_tickers_and_metadata(monkeypatch):
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            payload = {
                "screen_type": "свечной график",
                "summary": "График показывает пробой.",
                "visible_objects": ["свечи"],
                "chart_elements": {
                    "ticker": "ПЕРЕВОДИТЬ НЕЛЬЗЯ",
                    "timeframe": "тоже нельзя",
                    "patterns": ["пробой"],
                    "levels": ["сопротивление"],
                    "indicators": ["объем"],
                },
                "explanation": "Цена пробивает уровень.",
                "uncertainties": [],
            }
            message = type("Message", (), {"content": json.dumps(payload)})
            choice = type("Choice", (), {"message": message})
            return type("Response", (), {"choices": [choice]})

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(Client=lambda **kwargs: object()))
    provider = LMStudioTranslationProvider("http://localhost:1234/v1", "key", "gemma")
    frame = {
        "video_id": "abc",
        "timestamp_seconds": 135,
        "timestamp_key": "000135",
        "frame_path": "frame.jpg",
        "model": "vision-model",
        "screen_type": "candlestick_chart",
        "summary": "A chart shows a breakout.",
        "visible_objects": ["candles"],
        "chart_elements": {
            "ticker": "S&P500",
            "timeframe": "1H",
            "patterns": ["breakout"],
            "levels": ["resistance"],
            "indicators": ["volume"],
        },
        "explanation": "Price breaks resistance.",
        "uncertainties": [],
    }

    translated = provider.translate_visual_frame(frame, "ru")

    assert translated["video_id"] == "abc"
    assert translated["timestamp_seconds"] == 135
    assert translated["frame_path"] == "frame.jpg"
    assert translated["model"] == "vision-model"
    assert translated["chart_elements"]["ticker"] == "S&P500"
    assert translated["chart_elements"]["timeframe"] == "1H"
    assert translated["chart_elements"]["patterns"] == ["пробой"]


def test_translation_skips_model_when_source_language_already_target():
    provider = FakeTranslationProvider()
    transcript = Transcript(
        video_id="abc",
        source="youtube",
        language="ru",
        segments=[TranscriptSegment(start=0, duration=1, text="Привет")],
    )

    translated = provider.translate_transcript(transcript, "ru")

    assert translated.language == "ru"
    assert translated.segments[0].text == "Привет"
    assert provider.text_calls == []


def test_lmstudio_translation_uses_json_schema(monkeypatch):
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            payload = {"segments": [{"index": 0, "text": "Привет"}]}
            message = type("Message", (), {"content": json.dumps(payload)})
            choice = type("Choice", (), {"message": message})
            return type("Response", (), {"choices": [choice]})

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(Client=FakeClient))
    provider = LMStudioTranslationProvider("http://localhost:1234/v1", "key", "gemma", max_chars=100)
    transcript = Transcript("abc", "youtube", [TranscriptSegment(0, 1, "Hola")], language="es")

    translated = provider.translate_transcript(transcript, "ru")

    assert translated.segments[0].text == "Привет"
    assert calls[0]["model"] == "gemma"
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[0]["response_format"]["json_schema"]["schema"]["required"] == ["segments"]


def test_lmstudio_translation_records_usage_and_retry_telemetry(monkeypatch):
    calls = []

    class FakeUsage:
        prompt_tokens = 11
        completion_tokens = 7
        total_tokens = 18

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError("temporary 502")
            payload = {"segments": [{"index": 0, "text": "Привет"}]}
            message = type("Message", (), {"content": json.dumps(payload)})
            choice = type("Choice", (), {"message": message})
            return type("Response", (), {"choices": [choice], "usage": FakeUsage()})

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})

    monkeypatch.setattr("yt_sub_hub.translation.time.sleep", lambda seconds: None)
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(Client=lambda **kwargs: object()))
    provider = LMStudioTranslationProvider("http://localhost:1234/v1", "key", "gemma", retries=1)
    transcript = Transcript("abc", "youtube", [TranscriptSegment(0, 1, "Hola")], language="es")

    provider.translate_transcript(transcript, "ru")

    assert provider.last_attempt_count == 2
    assert provider.retry_count == 1
    assert provider.translation_usage == [{"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}]


def test_lmstudio_translation_ignores_returned_timestamps_and_preserves_source(monkeypatch):
    class FakeCompletions:
        def create(self, **kwargs):
            payload = {
                "segments": [
                    {
                        "index": 0,
                        "start": 0.01,
                        "duration": 0.99,
                        "text": "Привет",
                    }
                ]
            }
            message = type("Message", (), {"content": json.dumps(payload)})
            choice = type("Choice", (), {"message": message})
            return type("Response", (), {"choices": [choice]})

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(Client=lambda **kwargs: object()))
    provider = LMStudioTranslationProvider("http://localhost:1234/v1", "key", "gemma", max_chars=100)
    transcript = Transcript("abc", "youtube", [TranscriptSegment(0.0, 1.25, "Hola")], language="es")

    translated = provider.translate_transcript(transcript, "ru")

    assert translated.segments[0].text == "Привет"
    assert translated.segments[0].start == 0.0
    assert translated.segments[0].duration == 1.25


def test_lmstudio_analysis_translation_uses_small_text_payloads(monkeypatch):
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            content = kwargs["messages"][1]["content"]
            assert '"sections"' not in content
            data = json.loads(content.split("Input JSON:\n", 1)[1])
            payload = {
                "items": [
                    {
                        "path": item["path"],
                        "text": "Русский перевод сохраняет смысл поля.",
                    }
                    for item in data["items"]
                ]
            }
            message = type("Message", (), {"content": json.dumps(payload)})
            choice = type("Choice", (), {"message": message})
            return type("Response", (), {"choices": [choice]})

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(Client=lambda **kwargs: object()))
    provider = LMStudioTranslationProvider("http://localhost:1234/v1", "key", "gemma", max_chars=100)

    translated = provider.translate_analysis(_analysis(), "ru")

    assert translated.language == "ru"
    assert translated.cleaned_title == "Русский перевод сохраняет смысл поля."
    assert translated.sections[0].timestamp_seconds == 42
    assert translated.concepts[0].name == "Русский перевод сохраняет смысл поля."
    assert calls
    assert all(call["response_format"]["json_schema"]["name"] == "analysis_text_translation" for call in calls)


def test_lmstudio_analysis_translation_splits_invalid_batch_and_preserves_cardinality(monkeypatch):
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            content = kwargs["messages"][1]["content"]
            calls.append(content)
            data = json.loads(content.split("Input JSON:\n", 1)[1])
            items = data["items"]
            if len(items) > 1:
                payload = {"items": []}
            else:
                payload = {
                    "items": [
                        {
                            "path": items[0]["path"],
                            "text": "Русский перевод сохраняет смысл исходного поля.",
                        }
                    ]
                }
            message = type("Message", (), {"content": json.dumps(payload)})
            choice = type("Choice", (), {"message": message})
            return type("Response", (), {"choices": [choice]})

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(Client=lambda **kwargs: object()))
    provider = LMStudioTranslationProvider("http://localhost:1234/v1", "key", "gemma", max_chars=10)

    translated = provider.translate_analysis(_analysis(), "ru")

    assert translated.language == "ru"
    assert len(translated.sections) == 1
    assert len(translated.concepts) == 1
    assert len(translated.claims) == 1
    assert translated.sections[0].timestamp_seconds == 42
    assert translated.claims[0].evidence == "Русский перевод сохраняет смысл исходного поля."
    assert len(calls) > 1


def test_lmstudio_schema_builders_are_strict_json_schema():
    transcript_schema = build_transcript_translation_response_format()
    analysis_schema = build_analysis_translation_response_format()

    assert transcript_schema["type"] == "json_schema"
    assert transcript_schema["json_schema"]["strict"] is True
    assert analysis_schema["type"] == "json_schema"
    assert analysis_schema["json_schema"]["strict"] is True


def test_lmstudio_translation_rejects_segment_count_mismatch(monkeypatch):
    class FakeCompletions:
        def create(self, **kwargs):
            message = type("Message", (), {"content": json.dumps({"segments": []})})
            choice = type("Choice", (), {"message": message})
            return type("Response", (), {"choices": [choice]})

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(Client=lambda **kwargs: object()))
    provider = LMStudioTranslationProvider("http://localhost:1234/v1", "key", "gemma")
    transcript = Transcript("abc", "youtube", [TranscriptSegment(0, 1, "Hola")], language="es")

    with pytest.raises(TranslationError, match="segment count"):
        provider.translate_transcript(transcript, "ru")


def test_lmstudio_translation_splits_batch_when_segment_count_mismatches(monkeypatch):
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            content = kwargs["messages"][1]["content"]
            calls.append(content)
            data = json.loads(content.split("Input JSON:\n", 1)[1])
            segments = data["segments"]
            if len(segments) > 1:
                payload = {"segments": []}
            else:
                payload = {"segments": [{"index": segments[0]["index"], "text": f"ru:{segments[0]['text']}"}]}
            message = type("Message", (), {"content": json.dumps(payload)})
            choice = type("Choice", (), {"message": message})
            return type("Response", (), {"choices": [choice]})

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(Client=lambda **kwargs: object()))
    provider = LMStudioTranslationProvider("http://localhost:1234/v1", "key", "gemma", max_chars=100)
    transcript = Transcript(
        "abc",
        "youtube",
        [TranscriptSegment(0, 1, "Hola"), TranscriptSegment(1, 1, "Mundo")],
        language="es",
    )

    translated = provider.translate_transcript(transcript, "ru")

    assert [segment.text for segment in translated.segments] == ["ru:Hola", "ru:Mundo"]
    assert len(calls) == 3


def test_lmstudio_transcript_translation_enforces_total_timeout(monkeypatch):
    clock = {"now": 0.0}

    def monotonic():
        return clock["now"]

    def translate_payload(self, payload, target_language, response_format, task):
        clock["now"] += 2.0
        segment = payload["segments"][0]
        return {"segments": [{"index": segment["index"], "text": f"ru:{segment['text']}"}]}

    monkeypatch.setattr("yt_sub_hub.translation.time.monotonic", monotonic)
    monkeypatch.setattr(LMStudioTranslationProvider, "_translate_payload", translate_payload)
    provider = LMStudioTranslationProvider("http://localhost:1234/v1", "key", "gemma", max_chars=25, timeout_seconds=1)
    transcript = Transcript(
        "abc",
        "youtube",
        [TranscriptSegment(0, 1, "Hola"), TranscriptSegment(1, 1, "Mundo")],
        language="es",
    )

    with pytest.raises(TranslationError, match="timed out"):
        provider.translate_transcript(transcript, "ru")
