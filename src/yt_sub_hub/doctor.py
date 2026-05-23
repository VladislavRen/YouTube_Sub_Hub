from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from yt_sub_hub.config import Settings, get_settings
from yt_sub_hub.ingestion import YouTubeAuthRequiredError, fetch_video_metadata
from yt_sub_hub.whisper_cpp_setup import resolve_whisper_cpp_binary, resolve_whisper_cpp_model


YOUTUBE_METADATA_PROBE_URL = "https://www.youtube.com/watch?v=o6hlVz6mQZY"
YOUTUBE_METADATA_TIMEOUT_MESSAGE = (
    "YouTube metadata probe timed out. Set YT_SUB_HUB_YTDLP_PROXY=http://127.0.0.1:12334 or check network route."
)


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class DoctorReport:
    checks: list[DoctorCheck]

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    @property
    def failed_names(self) -> list[str]:
        return [check.name for check in self.checks if not check.ok]


def _check_python() -> DoctorCheck:
    version = sys.version_info
    ok = version.major == 3 and version.minor == 11
    return DoctorCheck("python", ok, f"{version.major}.{version.minor}.{version.micro}")


def _check_import(package: str, module: str | None = None) -> DoctorCheck:
    module_name = module or package.replace("-", "_")
    ok = importlib.util.find_spec(module_name) is not None
    return DoctorCheck(f"dependency:{package}", ok, module_name if ok else "not installed")


def _check_ffmpeg() -> DoctorCheck:
    path = shutil.which("ffmpeg")
    return DoctorCheck("ffmpeg", path is not None, path or "missing")


def _lm_models(settings: Settings) -> tuple[set[str], str | None]:
    url = settings.lm_studio_base_url.rstrip("/") + "/models"
    try:
        import httpx

        response = httpx.get(url, timeout=10, trust_env=False)
        response.raise_for_status()
        payload = response.json()
        return {item["id"] for item in payload.get("data", [])}, None
    except Exception as exc:
        return set(), str(exc)


def _check_lm_model(model: str, models: set[str], error: str | None) -> DoctorCheck:
    if error:
        return DoctorCheck("lm_studio", False, error)
    return DoctorCheck(f"lm_studio_model:{model}", model in models, "available" if model in models else "missing")


def _check_lm_generation(settings: Settings, model: str) -> DoctorCheck:
    url = settings.lm_studio_base_url.rstrip("/") + "/chat/completions"
    try:
        import httpx

        response = httpx.post(
            url,
            timeout=min(settings.lm_studio_timeout_seconds, 120),
            headers={"Authorization": f"Bearer {settings.lm_studio_api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Return JSON: {\"ok\": true}"}],
                "temperature": 0,
                "max_tokens": 32,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "doctor_smoke",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["ok"],
                            "properties": {"ok": {"type": "boolean"}},
                        },
                    },
                },
            },
            trust_env=False,
        )
        if response.status_code >= 400:
            return DoctorCheck(f"lm_studio_generate:{model}", False, response.text)
        return DoctorCheck(f"lm_studio_generate:{model}", True, "ok")
    except Exception as exc:
        return DoctorCheck(f"lm_studio_generate:{model}", False, str(exc))


def _check_neo4j(settings: Settings) -> DoctorCheck:
    try:
        from yt_sub_hub.graph import Neo4jGraphStore

        graph = Neo4jGraphStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
        try:
            graph.verify_connectivity()
        finally:
            graph.close()
        return DoctorCheck("neo4j", True, settings.neo4j_uri)
    except Exception as exc:
        return DoctorCheck("neo4j", False, str(exc))


def _check_ytdlp_proxy(settings: Settings) -> DoctorCheck:
    return DoctorCheck("yt-dlp proxy", True, "configured" if settings.ytdlp_proxy else "not configured")


def _check_ytdlp_cookies_file(settings: Settings) -> DoctorCheck:
    if not settings.ytdlp_cookies_file:
        return DoctorCheck("yt-dlp cookies file", True, "not configured")
    path = Path(settings.ytdlp_cookies_file)
    try:
        with path.open("rb"):
            pass
    except OSError:
        return DoctorCheck(
            "yt-dlp cookies file",
            False,
            f"configured but not readable: {settings.ytdlp_cookies_file}",
        )
    return DoctorCheck("yt-dlp cookies file", True, "configured")


def _check_ytdlp_browser_cookies(settings: Settings) -> DoctorCheck:
    if settings.ytdlp_cookies_file:
        return DoctorCheck("yt-dlp browser cookies", True, "disabled")
    return DoctorCheck(
        "yt-dlp browser cookies",
        True,
        "configured" if settings.ytdlp_cookies_from_browser else "disabled",
    )


def _check_media_download(settings: Settings) -> DoctorCheck:
    return DoctorCheck("media download", True, "enabled" if settings.media_download_enabled else "disabled")


def _check_acceptance_mode(settings: Settings, *, strict: bool) -> DoctorCheck:
    if settings.acceptance_mode == "full_processing" and not settings.media_download_enabled:
        return DoctorCheck(
            "acceptance mode",
            not strict,
            "full_processing requires YT_SUB_HUB_MEDIA_DOWNLOAD=enabled",
        )
    if settings.acceptance_mode == "data_api_only" and settings.media_download_enabled:
        return DoctorCheck("acceptance mode", True, "data_api_only; media download enabled but ignored")
    return DoctorCheck("acceptance mode", True, settings.acceptance_mode)


def _check_torch() -> list[DoctorCheck]:
    try:
        import torch
    except ImportError:
        return [
            DoctorCheck("torch", False, "not installed"),
            DoctorCheck("torch mps", False, "unknown: torch not installed"),
            DoctorCheck("torch cuda", False, "unknown: torch not installed"),
        ]
    mps_available = bool(getattr(getattr(torch, "backends", None), "mps", None) and torch.backends.mps.is_available())
    cuda_available = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    return [
        DoctorCheck("torch", True, "installed"),
        DoctorCheck("torch mps", True, "available" if mps_available else "unavailable"),
        DoctorCheck("torch cuda", True, "available" if cuda_available else "unavailable"),
    ]


def _torch_mps_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(getattr(getattr(torch, "backends", None), "mps", None) and torch.backends.mps.is_available())


def _check_whisper_backend_selection(settings: Settings) -> DoctorCheck:
    return DoctorCheck(
        "whisper backend",
        True,
        f"{settings.whisper_backend} model={settings.whisper_model}",
    )


def _check_whisper_backend_validation(settings: Settings) -> DoctorCheck:
    if settings.whisper_backend == "openai-whisper-mps" and not _torch_mps_available():
        return DoctorCheck("whisper backend validation", False, "MPS is not available")
    return DoctorCheck("whisper backend validation", True, "ok")


def _check_whisper_cpp_binary(settings: Settings) -> DoctorCheck:
    if settings.whisper_backend != "whisper-cpp-metal":
        return DoctorCheck("whisper.cpp binary", True, "not selected")
    resolved = resolve_whisper_cpp_binary(settings)
    if resolved:
        return DoctorCheck("whisper.cpp binary", True, resolved)
    if not settings.whisper_cpp_bin:
        return DoctorCheck(
            "whisper.cpp binary",
            False,
            "missing: install whisper.cpp so `whisper-cli` is on PATH or set YT_SUB_HUB_WHISPER_CPP_BIN=/path/to/whisper-cli",
        )
    path = Path(settings.whisper_cpp_bin)
    if not path.exists() or not path.is_file() or not path.stat().st_mode & 0o111:
        return DoctorCheck("whisper.cpp binary", False, f"configured but not executable: {settings.whisper_cpp_bin}")
    return DoctorCheck("whisper.cpp binary", True, resolved)


def _check_whisper_cpp_model(settings: Settings) -> DoctorCheck:
    if settings.whisper_backend != "whisper-cpp-metal":
        return DoctorCheck("whisper.cpp model", True, "not selected")
    resolved = resolve_whisper_cpp_model(settings)
    if resolved:
        return DoctorCheck("whisper.cpp model", True, resolved)
    if not settings.whisper_cpp_model:
        return DoctorCheck(
            "whisper.cpp model",
            False,
            "missing: run `yt-sub-hub setup-whisper-cpp --download-model` or set YT_SUB_HUB_WHISPER_CPP_MODEL=/path/to/ggml-model.bin",
        )
    path = Path(settings.whisper_cpp_model)
    if not path.exists() or not path.is_file():
        return DoctorCheck("whisper.cpp model", False, f"configured but missing: {settings.whisper_cpp_model}")
    return DoctorCheck("whisper.cpp model", True, resolved)


def _check_whisper_cpp_probe(settings: Settings) -> DoctorCheck:
    if settings.whisper_backend != "whisper-cpp-metal":
        return DoctorCheck("whisper.cpp probe", True, "not selected")
    binary = _check_whisper_cpp_binary(settings)
    model = _check_whisper_cpp_model(settings)
    if not binary.ok or not model.ok:
        return DoctorCheck("whisper.cpp probe", False, "skipped until binary and model are configured")
    resolved_binary = resolve_whisper_cpp_binary(settings)
    if resolved_binary is None:
        return DoctorCheck("whisper.cpp probe", False, "skipped until binary is available")
    for flag in ["--help", "-h", "--version"]:
        try:
            completed = subprocess.run(
                [resolved_binary, flag],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            output = (completed.stdout or completed.stderr or "").strip()
            if output or completed.returncode == 0:
                return DoctorCheck("whisper.cpp probe", True, f"{flag} responded")
        except Exception:
            continue
    return DoctorCheck("whisper.cpp probe", False, "binary did not respond to help/version probes")


def _is_timeout_error(error: Exception) -> bool:
    message = str(error).lower()
    return isinstance(error, TimeoutError) or "timed out" in message or "timeout" in message


def _check_youtube_metadata_probe(settings: Settings) -> DoctorCheck:
    try:
        metadata = fetch_video_metadata(YOUTUBE_METADATA_PROBE_URL, settings=settings)
        return DoctorCheck("youtube_metadata_probe", True, f"{metadata.video_id} {metadata.title}")
    except YouTubeAuthRequiredError as exc:
        return DoctorCheck("youtube_metadata_probe", False, str(exc))
    except Exception as exc:
        if _is_timeout_error(exc):
            return DoctorCheck("youtube_metadata_probe", False, YOUTUBE_METADATA_TIMEOUT_MESSAGE)
        return DoctorCheck("youtube_metadata_probe", False, str(exc))


def run_doctor(settings: Settings | None = None, strict: bool = False) -> DoctorReport:
    settings = settings or get_settings()
    models, model_error = _lm_models(settings)
    model_ids = list(dict.fromkeys([settings.qwen_model, settings.gemma_model]))
    checks = [
        _check_python(),
        _check_ffmpeg(),
        _check_import("youtube-transcript-api", "youtube_transcript_api"),
        _check_import("yt-dlp", "yt_dlp"),
        _check_import("openai"),
        _check_import("neo4j"),
        _check_whisper_backend_selection(settings),
    ]
    checks.extend(_check_torch())
    checks.extend(
        [
            _check_ytdlp_proxy(settings),
            _check_ytdlp_cookies_file(settings),
            _check_ytdlp_browser_cookies(settings),
            _check_media_download(settings),
            _check_acceptance_mode(settings, strict=strict),
            _check_youtube_metadata_probe(settings),
        ]
    )
    checks.extend(_check_lm_model(model, models, model_error) for model in model_ids)
    if settings.enable_graph or strict:
        checks.append(_check_neo4j(settings))
    if strict:
        checks.extend(
            [
                _check_whisper_backend_validation(settings),
                _check_whisper_cpp_binary(settings),
                _check_whisper_cpp_model(settings),
                _check_whisper_cpp_probe(settings),
            ]
        )
        checks.extend(_check_lm_generation(settings, model) for model in model_ids)
    return DoctorReport(checks=checks)


def format_report(report: DoctorReport) -> str:
    lines: list[str] = []
    for check in report.checks:
        mark = "OK" if check.ok else "FAIL"
        lines.append(f"{mark} {check.name}: {check.detail}")
    return "\n".join(lines)
