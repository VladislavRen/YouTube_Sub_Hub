from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from yt_sub_hub.config import Settings, get_settings
from yt_sub_hub.whisper_backends import build_whisper_backend


def resolve_benchmark_audio(target: str, settings: Settings) -> Path:
    path = Path(target)
    if path.exists():
        return path
    audio_path = settings.library_dir / "assets" / "audio" / f"{target}.mp3"
    if audio_path.exists():
        return audio_path
    raise FileNotFoundError(f"audio not found: {target} or {audio_path}")


def run_whisper_benchmark(target: str, *, settings: Settings | None = None, seconds: int = 30) -> dict[str, Any]:
    settings = settings or get_settings()
    audio_path = resolve_benchmark_audio(target, settings)
    backend = build_whisper_backend(settings)
    with tempfile.TemporaryDirectory(prefix="yt-sub-hub-whisper-benchmark-") as tmp:
        clip_path = Path(tmp) / "clip.mp3"
        benchmark_path = _make_audio_clip(audio_path, clip_path, seconds)
        start = time.perf_counter()
        bundle = backend.transcribe(benchmark_path, video_id=Path(target).stem if Path(target).exists() else target)
        elapsed = time.perf_counter() - start
    text_sample = bundle.normalized.full_text().replace("\n", " ")[:240]
    return {
        "backend": getattr(backend, "name", settings.whisper_backend),
        "model": getattr(backend, "model", settings.whisper_model),
        "device": getattr(backend, "device", "unknown"),
        "elapsed_seconds": round(elapsed, 3),
        "language": bundle.normalized.language or "",
        "segments": len(bundle.normalized.segments),
        "sample": text_sample,
    }


def _make_audio_clip(audio_path: Path, clip_path: Path, seconds: int) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return audio_path
    completed = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(audio_path),
            "-t",
            str(seconds),
            "-vn",
            "-acodec",
            "copy",
            str(clip_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0 and clip_path.exists() and clip_path.stat().st_size > 0:
        return clip_path
    completed = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(audio_path),
            "-t",
            str(seconds),
            "-vn",
            str(clip_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"failed to create benchmark audio clip: {detail}")
    return clip_path
