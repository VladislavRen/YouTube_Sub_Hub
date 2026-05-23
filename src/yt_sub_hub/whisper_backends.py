from __future__ import annotations

import json
import re
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from yt_sub_hub.config import Settings
from yt_sub_hub.models import Transcript, TranscriptBundle, TranscriptSegment
from yt_sub_hub.whisper_cpp_setup import resolve_whisper_cpp_binary, resolve_whisper_cpp_model


class WhisperBackend(Protocol):
    name: str
    model: str
    device: str

    def transcribe(self, audio_path: Path, *, video_id: str = "") -> TranscriptBundle: ...


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _segments_from_openai(result: dict[str, Any]) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    for item in result.get("segments", []):
        start = float(item.get("start", 0.0))
        end = float(item.get("end", start))
        raw_text = str(item.get("text", ""))
        text = _clean_text(raw_text)
        if text:
            segments.append(
                TranscriptSegment(
                    start=start,
                    duration=max(0.0, end - start),
                    text=text,
                    raw_text=raw_text,
                )
            )
    return segments


@dataclass(frozen=True)
class OpenAIWhisperBackend:
    model_name: str = "base"
    device: str = "cpu"
    fp16: bool = False

    @property
    def name(self) -> str:
        return f"openai-whisper-{self.device}"

    @property
    def model(self) -> str:
        return self.model_name

    def transcribe(self, audio_path: Path, *, video_id: str = "") -> TranscriptBundle:
        try:
            import whisper
        except ImportError as exc:
            raise RuntimeError("openai-whisper is not installed") from exc

        model = whisper.load_model(self.model_name, device=self.device)
        result = model.transcribe(str(audio_path), fp16=self.fp16)
        if not isinstance(result, dict):
            raise RuntimeError("openai-whisper returned an unexpected result")
        normalized = Transcript(
            video_id=video_id,
            source="whisper",
            language=result.get("language"),
            segments=_segments_from_openai(result),
        )
        return TranscriptBundle(video_id=video_id, source="whisper", raw_payload=result, normalized=normalized)


@dataclass(frozen=True)
class WhisperCppBackend:
    bin_path: str
    model_path: str
    threads: int | None = None
    extra_args: str = ""

    name: str = "whisper-cpp-metal"
    device: str = "metal"

    @property
    def model(self) -> str:
        return self.model_path

    def build_args(self, audio_path: Path, output_prefix: Path) -> list[str]:
        args = [
            self.bin_path,
            "-m",
            self.model_path,
            "-f",
            str(audio_path),
        ]
        if self.threads is not None:
            args.extend(["-t", str(self.threads)])
        if self.extra_args:
            args.extend(shlex.split(self.extra_args))
        args.extend(["-oj", "-of", str(output_prefix)])
        return args

    def transcribe(self, audio_path: Path, *, video_id: str = "") -> TranscriptBundle:
        with tempfile.TemporaryDirectory(prefix="yt-sub-hub-whisper-cpp-") as tmp:
            output_prefix = Path(tmp) / "transcript"
            args = self.build_args(audio_path, output_prefix)
            completed = subprocess.run(args, capture_output=True, text=True, check=False)
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "").strip()
                raise RuntimeError(f"whisper.cpp failed with exit code {completed.returncode}: {detail}")
            json_path = output_prefix.with_suffix(".json")
            if json_path.exists():
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            else:
                text = (completed.stdout or "").strip()
                if not text:
                    raise RuntimeError("whisper.cpp JSON output unavailable and stdout was empty")
                payload = {"warning": "whisper.cpp JSON output unavailable; normalized from stdout", "text": text}
        normalized = Transcript(
            video_id=video_id,
            source="whisper",
            language=payload.get("language") if isinstance(payload, dict) else None,
            segments=_segments_from_whisper_cpp_payload(payload),
        )
        return TranscriptBundle(video_id=video_id, source="whisper", raw_payload=payload, normalized=normalized)


def _segments_from_whisper_cpp_payload(payload: Any) -> list[TranscriptSegment]:
    if not isinstance(payload, dict):
        return []
    if "text" in payload and "segments" not in payload and "transcription" not in payload:
        text = _clean_text(str(payload.get("text", "")))
        return [TranscriptSegment(start=0.0, duration=0.0, text=text, raw_text=str(payload.get("text", "")))] if text else []
    raw_segments = payload.get("segments") or payload.get("transcription") or []
    segments: list[TranscriptSegment] = []
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        start, end = _whisper_cpp_segment_bounds(item)
        raw_text = str(item.get("text", ""))
        text = _clean_text(raw_text)
        if text:
            segments.append(
                TranscriptSegment(
                    start=start,
                    duration=max(0.0, end - start),
                    text=text,
                    raw_text=raw_text,
                )
            )
    return segments


def _whisper_cpp_segment_bounds(item: dict[str, Any]) -> tuple[float, float]:
    if "start" in item or "end" in item:
        start = float(item.get("start", 0.0))
        return start, float(item.get("end", start))
    if "t0" in item or "t1" in item:
        start = float(item.get("t0", 0.0)) / 100.0
        return start, float(item.get("t1", item.get("t0", 0.0))) / 100.0
    offsets = item.get("offsets")
    if isinstance(offsets, dict):
        return _time_value(offsets.get("from"), numeric_milliseconds=True), _time_value(
            offsets.get("to"),
            numeric_milliseconds=True,
        )
    timestamps = item.get("timestamps")
    if isinstance(timestamps, dict):
        return _time_value(timestamps.get("from"), numeric_milliseconds=True), _time_value(
            timestamps.get("to"),
            numeric_milliseconds=True,
        )
    return 0.0, 0.0


def _time_value(value: Any, *, numeric_milliseconds: bool = False) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        number = float(value)
        if numeric_milliseconds:
            return number / 1000.0
        return number / 1000.0 if number > 10_000 or number == int(number) and number >= 1000 else number
    text = str(value).strip()
    if not text:
        return 0.0
    return _parse_timestamp_text(text)


def _parse_timestamp_text(text: str) -> float:
    original = text
    normalized = text.strip().replace(",", ".")
    parts = normalized.split(":")
    if len(parts) > 3 or any(part == "" for part in parts):
        raise ValueError(f"Invalid timestamp value: {original!r}")
    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"Invalid timestamp value: {original!r}") from exc
    if any(value < 0 for value in values):
        raise ValueError(f"Invalid timestamp value: {original!r}")
    if len(values) == 3:
        hours, minutes, seconds = values
        return hours * 3600 + minutes * 60 + seconds
    if len(values) == 2:
        minutes, seconds = values
        return minutes * 60 + seconds
    return values[0]


def build_whisper_backend(settings: Settings, model_name: str | None = None) -> WhisperBackend:
    model = model_name or settings.whisper_model
    if settings.whisper_backend == "openai-whisper-cpu":
        return OpenAIWhisperBackend(model_name=model, device="cpu", fp16=settings.whisper_fp16)
    if settings.whisper_backend == "openai-whisper-mps":
        return OpenAIWhisperBackend(model_name=model, device="mps", fp16=settings.whisper_fp16)
    if settings.whisper_backend == "whisper-cpp-metal":
        binary = resolve_whisper_cpp_binary(settings)
        cpp_model = resolve_whisper_cpp_model(settings, model)
        if not binary or not cpp_model:
            raise RuntimeError(
                "whisper-cpp-metal requires an executable whisper-cli on PATH or YT_SUB_HUB_WHISPER_CPP_BIN, "
                "and a model prepared by `yt-sub-hub setup-whisper-cpp --download-model` "
                "or YT_SUB_HUB_WHISPER_CPP_MODEL"
            )
        return WhisperCppBackend(
            bin_path=binary,
            model_path=cpp_model,
            threads=settings.whisper_cpp_threads,
            extra_args=settings.whisper_cpp_extra_args,
        )
    raise RuntimeError(f"Unsupported Whisper backend: {settings.whisper_backend}")


def whisper_resource_lock(backend: str) -> str:
    return "whisper_cpu" if backend == "openai-whisper-cpu" else "whisper_accelerated"
