from __future__ import annotations

import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from yt_sub_hub.config import Settings

WHISPER_CPP_BINARY_CANDIDATES = ("whisper-cli", "whisper-cpp")
WHISPER_CPP_MODEL_BASE_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"


@dataclass(frozen=True)
class WhisperCppSetupStatus:
    binary_path: str | None
    model_path: str | None
    model_ready: bool
    model_downloaded: bool = False
    message: str = ""


def resolve_whisper_cpp_binary(settings: Settings) -> str | None:
    if settings.whisper_cpp_bin:
        path = Path(settings.whisper_cpp_bin)
        return str(path) if _is_executable_file(path) else None
    for candidate in WHISPER_CPP_BINARY_CANDIDATES:
        discovered = shutil.which(candidate)
        if discovered:
            return discovered
    return None


def managed_whisper_cpp_model_path(settings: Settings, model_name: str | None = None) -> Path:
    filename = _whisper_cpp_model_filename(model_name or settings.whisper_model)
    return settings.library_dir / "assets" / "whisper.cpp" / "models" / filename


def resolve_whisper_cpp_model(settings: Settings, model_name: str | None = None) -> str | None:
    if settings.whisper_cpp_model:
        path = Path(settings.whisper_cpp_model)
        return str(path) if path.exists() and path.is_file() else None
    managed = managed_whisper_cpp_model_path(settings, model_name)
    return str(managed) if managed.exists() and managed.is_file() else None


def prepare_whisper_cpp(
    settings: Settings,
    *,
    model_name: str | None = None,
    download_model: bool = False,
) -> WhisperCppSetupStatus:
    binary_path = resolve_whisper_cpp_binary(settings)
    model_path = resolve_whisper_cpp_model(settings, model_name)
    downloaded = False
    if model_path is None:
        target = managed_whisper_cpp_model_path(settings, model_name)
        if download_model:
            target.parent.mkdir(parents=True, exist_ok=True)
            download_target = target.with_suffix(target.suffix + ".download")
            try:
                urllib.request.urlretrieve(whisper_cpp_model_url(model_name or settings.whisper_model), str(download_target))
                if not download_target.exists() or download_target.stat().st_size <= 0:
                    raise RuntimeError(f"downloaded whisper.cpp model is empty: {download_target}")
                download_target.replace(target)
            finally:
                download_target.unlink(missing_ok=True)
            model_path = str(target)
            downloaded = True
    model_ready = model_path is not None and Path(model_path).exists()
    messages = []
    if binary_path is None:
        messages.append("whisper.cpp binary not found; install whisper.cpp so `whisper-cli` is on PATH or set YT_SUB_HUB_WHISPER_CPP_BIN")
    if not model_ready:
        messages.append(
            "whisper.cpp model not found; run `yt-sub-hub setup-whisper-cpp --download-model` "
            "or set YT_SUB_HUB_WHISPER_CPP_MODEL"
        )
    return WhisperCppSetupStatus(
        binary_path=binary_path,
        model_path=model_path,
        model_ready=model_ready,
        model_downloaded=downloaded,
        message="; ".join(messages),
    )


def whisper_cpp_model_url(model_name: str) -> str:
    return f"{WHISPER_CPP_MODEL_BASE_URL}/{_whisper_cpp_model_filename(model_name)}"


def _whisper_cpp_model_filename(model_name: str) -> str:
    stripped = model_name.strip()
    if stripped.startswith("ggml-") and stripped.endswith(".bin"):
        return stripped
    return f"ggml-{stripped}.bin"


def _is_executable_file(path: Path) -> bool:
    return path.exists() and path.is_file() and bool(path.stat().st_mode & 0o111)
