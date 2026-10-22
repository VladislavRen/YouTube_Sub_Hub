import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from yt_sub_hub.whisper_backends import (
    OpenAIWhisperBackend,
    WhisperCppBackend,
    build_whisper_backend,
    _time_value,
    whisper_resource_lock,
)


def test_openai_whisper_backend_loads_cpu_device_and_passes_fp16_false(tmp_path, monkeypatch):
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"audio")
    captured = {}

    class FakeModel:
        def transcribe(self, path, **kwargs):
            captured["transcribe_path"] = path
            captured["transcribe_kwargs"] = kwargs
            return {
                "language": "en",
                "segments": [{"start": 0.0, "end": 1.5, "text": " Hello  world "}],
            }

    fake_whisper = types.SimpleNamespace(
        load_model=lambda model_name, device=None: captured.update({"model": model_name, "device": device}) or FakeModel()
    )
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)

    bundle = OpenAIWhisperBackend(model_name="base", device="cpu", fp16=False).transcribe(audio_path, video_id="abc123")

    assert captured["model"] == "base"
    assert captured["device"] == "cpu"
    assert captured["transcribe_path"] == str(audio_path)
    assert captured["transcribe_kwargs"]["fp16"] is False
    assert bundle.raw_payload["language"] == "en"
    assert bundle.normalized.source == "whisper"
    assert bundle.normalized.segments[0].duration == 1.5
    assert bundle.normalized.segments[0].text == "Hello world"


def test_openai_whisper_backend_loads_mps_device(tmp_path, monkeypatch):
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"audio")
    captured = {}

    class FakeModel:
        def transcribe(self, path, **kwargs):
            return {"segments": []}

    fake_whisper = types.SimpleNamespace(
        load_model=lambda model_name, device=None: captured.update({"device": device}) or FakeModel()
    )
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)

    OpenAIWhisperBackend(model_name="base", device="mps", fp16=False).transcribe(audio_path, video_id="abc123")

    assert captured["device"] == "mps"


def test_whisper_cpp_backend_builds_subprocess_args_without_shell(tmp_path):
    backend = WhisperCppBackend(
        bin_path="/opt/whisper.cpp/whisper-cli",
        model_path="/models/ggml-base.bin",
        threads=4,
        extra_args="--language en --no-timestamps",
    )

    args = backend.build_args(tmp_path / "audio.mp3", tmp_path / "out")

    assert args == [
        "/opt/whisper.cpp/whisper-cli",
        "-m",
        "/models/ggml-base.bin",
        "-f",
        str(tmp_path / "audio.mp3"),
        "-t",
        "4",
        "--language",
        "en",
        "--no-timestamps",
        "-oj",
        "-of",
        str(tmp_path / "out"),
    ]


def test_whisper_cpp_json_output_normalizes_into_transcript_schema(tmp_path, monkeypatch):
    bin_path = tmp_path / "whisper-cli"
    model_path = tmp_path / "ggml-base.bin"
    audio_path = tmp_path / "audio.mp3"
    bin_path.write_text("#!/bin/sh\n", encoding="utf-8")
    bin_path.chmod(0o755)
    model_path.write_bytes(b"model")
    audio_path.write_bytes(b"audio")

    def fake_run(args, **kwargs):
        output_prefix = Path(args[args.index("-of") + 1])
        output_prefix.with_suffix(".json").write_text(
            json.dumps(
                {
                    "language": "en",
                    "transcription": [
                        {"offsets": {"from": 1000, "to": 2500}, "text": " First line "},
                        {"timestamps": {"from": "00:00:02.500", "to": "00:00:04.000"}, "text": "Second   line"},
                        {"timestamps": {"from": "00:00:02,500", "to": "00:00:04,000"}, "text": " Пример текста "},
                    ],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    bundle = WhisperCppBackend(str(bin_path), str(model_path)).transcribe(audio_path, video_id="abc123")

    assert bundle.source == "whisper"
    assert bundle.raw_payload["language"] == "en"
    assert bundle.normalized.language == "en"
    assert [segment.text for segment in bundle.normalized.segments] == ["First line", "Second line", "Пример текста"]
    assert bundle.normalized.segments[0].start == 1.0
    assert bundle.normalized.segments[0].duration == 1.5
    assert bundle.normalized.segments[1].start == 2.5
    assert bundle.normalized.segments[1].duration == 1.5
    assert bundle.normalized.segments[2].start == 2.5
    assert bundle.normalized.segments[2].duration == 1.5


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("00:00:02,500", 2.5),
        ("00:02,500", 2.5),
        ("2,500", 2.5),
        ("00:00:02.500", 2.5),
        ("00:02.500", 2.5),
        ("2.5", 2.5),
        (2500, 2.5),
        (2.5, 2.5),
    ],
)
def test_time_value_normalizes_whisper_cpp_timestamp_formats(value, expected):
    assert _time_value(value) == expected


def test_time_value_rejects_invalid_non_empty_timestamp():
    with pytest.raises(ValueError, match="Invalid timestamp value: 'not-a-timestamp'"):
        _time_value("not-a-timestamp")


def test_whisper_cpp_empty_stdout_fallback_fails_instead_of_returning_empty_transcript(tmp_path, monkeypatch):
    bin_path = tmp_path / "whisper-cli"
    model_path = tmp_path / "ggml-base.bin"
    audio_path = tmp_path / "audio.mp3"
    bin_path.write_text("#!/bin/sh\n", encoding="utf-8")
    bin_path.chmod(0o755)
    model_path.write_bytes(b"model")
    audio_path.write_bytes(b"audio")

    monkeypatch.setattr(
        "subprocess.run",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 0, stdout="", stderr=""),
    )

    try:
        WhisperCppBackend(str(bin_path), str(model_path)).transcribe(audio_path, video_id="abc123")
    except RuntimeError as exc:
        assert "JSON output unavailable" in str(exc)
    else:
        raise AssertionError("empty whisper.cpp stdout fallback must fail")


def test_whisper_cpp_reads_language_from_native_result_object(tmp_path, monkeypatch):
    def fake_run(args, **kwargs):
        Path(args[args.index("-of") + 1]).with_suffix(".json").write_text(json.dumps({
            "result": {"language": "en"},
            "transcription": [{"offsets": {"from": 0, "to": 1000}, "text": "Hello"}],
        }))
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    bundle = WhisperCppBackend("whisper-cli", "base.bin").transcribe(tmp_path / "audio.wav")
    assert bundle.normalized.language == "en"


def test_whisper_resource_lock_distinguishes_cpu_and_accelerated_backends():
    assert whisper_resource_lock("openai-whisper-cpu") == "whisper_cpu"
    assert whisper_resource_lock("openai-whisper-mps") == "whisper_accelerated"
    assert whisper_resource_lock("whisper-cpp-metal") == "whisper_accelerated"


def test_build_whisper_cpp_backend_uses_discovered_binary_and_managed_model(tmp_path, monkeypatch):
    from yt_sub_hub.config import Settings

    binary = tmp_path / "whisper-cli"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    model = tmp_path / "assets" / "whisper.cpp" / "models" / "ggml-base.bin"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model")
    monkeypatch.setattr("shutil.which", lambda name: str(binary) if name == "whisper-cli" else None)
    settings = Settings(
        library_dir=tmp_path,
        lm_studio_base_url="http://localhost:1234/v1",
        lm_studio_api_key="lm-studio",
        qwen_model="qwen",
        gemma_model="gemma",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        keep_audio=True,
        enable_graph=False,
        graph_best_effort=True,
        lm_studio_timeout_seconds=120,
        max_prompt_chars=3500,
        ytdlp_cookies_from_browser=None,
        ytdlp_cookies_file=None,
        ytdlp_proxy=None,
        ytdlp_js_runtime=None,
        ytdlp_socket_timeout=60,
        ytdlp_retries=10,
        whisper_backend="whisper-cpp-metal",
    )

    backend = build_whisper_backend(settings)

    assert isinstance(backend, WhisperCppBackend)
    assert backend.bin_path == str(binary)
    assert backend.model_path == str(model)
