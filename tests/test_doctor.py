from pathlib import Path
import sys
import types

import pytest

from yt_sub_hub.config import Settings
from yt_sub_hub.doctor import DoctorCheck, DoctorReport, format_report, run_doctor
from yt_sub_hub.ingestion import YouTubeAuthRequiredError
from yt_sub_hub.models import VideoMetadata


def doctor_settings(
    *,
    proxy: str | None = None,
    cookies_file: str | None = None,
    browser_cookies: str | None = None,
    media_download_enabled: bool = False,
    acceptance_mode: str = "data_api_only",
) -> Settings:
    return Settings(
        library_dir=Path("library"),
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
        ytdlp_cookies_from_browser=browser_cookies,
        ytdlp_cookies_file=cookies_file,
        ytdlp_proxy=proxy,
        ytdlp_js_runtime=None,
        ytdlp_socket_timeout=60,
        ytdlp_retries=10,
        media_download_enabled=media_download_enabled,
        acceptance_mode=acceptance_mode,
    )


def stub_base_doctor_checks(monkeypatch):
    monkeypatch.setattr("yt_sub_hub.doctor._check_python", lambda: DoctorCheck("python", True, "3.11.0"))
    monkeypatch.setattr("yt_sub_hub.doctor._check_ffmpeg", lambda: DoctorCheck("ffmpeg", True, "/usr/bin/ffmpeg"))
    monkeypatch.setattr("yt_sub_hub.doctor._check_import", lambda package, module=None: DoctorCheck(f"dependency:{package}", True, module or package))
    monkeypatch.setattr("yt_sub_hub.doctor._lm_models", lambda settings: ({"qwen", "gemma"}, None))
    monkeypatch.setattr("yt_sub_hub.doctor._check_neo4j", lambda settings: DoctorCheck("neo4j", True, "ok"))
    monkeypatch.setattr("yt_sub_hub.doctor._check_lm_generation", lambda settings, model: DoctorCheck(f"lm_studio_generate:{model}", True, "ok"))


def test_doctor_report_fails_when_required_model_is_missing():
    report = DoctorReport(
        checks=[
            DoctorCheck(name="lm_studio_model:qwen/qwen3.6-35b-a3b", ok=False, detail="missing"),
            DoctorCheck(name="ffmpeg", ok=True, detail="/usr/local/bin/ffmpeg"),
        ]
    )

    assert report.ok is False
    assert "lm_studio_model:qwen/qwen3.6-35b-a3b" in report.failed_names


@pytest.mark.parametrize("content,finish_reason,ok", [
    ('{"ok": true}', "stop", True),
    ("", "length", False),
    ('{"ok": true}', "length", False),
    ("thinking only", "stop", False),
    ('{"ok": false}', "stop", False),
    ('{"ok": 1}', "stop", False),
    ('[]', "stop", False),
])
def test_generation_probe_checks_json_answer(monkeypatch, content, finish_reason, ok):
    import httpx
    from yt_sub_hub.doctor import _check_lm_generation

    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: httpx.Response(200, json={
        "choices": [{"finish_reason": finish_reason, "message": {
            "content": content, "reasoning_content": "model reasoning is not the answer",
        }}],
    }))

    result = _check_lm_generation(doctor_settings(), "qwen")

    assert result.ok is ok
    if not ok:
        assert "JSON" in result.detail


def test_media_fallback_reports_missing_javascript_runtime(monkeypatch):
    from yt_sub_hub.doctor import _check_media_dependencies

    monkeypatch.setattr("yt_sub_hub.doctor.shutil.which", lambda name: None)
    checks = _check_media_dependencies(doctor_settings(media_download_enabled=True, acceptance_mode="full_processing"))
    assert any(not check.ok and "JavaScript" in check.name for check in checks)
    assert any("YT_SUB_HUB_YTDLP_JS_RUNTIME" in check.detail for check in checks)


def test_official_mode_does_not_require_javascript_runtime(monkeypatch):
    from yt_sub_hub.doctor import _check_media_dependencies

    monkeypatch.setattr("yt_sub_hub.doctor.shutil.which", lambda name: None)
    assert _check_media_dependencies(doctor_settings()) == []


def test_doctor_reports_ytdlp_proxy_cookie_settings_and_metadata_probe(tmp_path, monkeypatch):
    stub_base_doctor_checks(monkeypatch)
    cookies_file = tmp_path / "youtube-cookies.txt"
    cookies_file.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")

    def fetch_video_metadata(url, settings=None):
        assert settings.ytdlp_proxy == "http://127.0.0.1:12334"
        return VideoMetadata(video_id="o6hlVz6mQZY", canonical_url=url, title="Nasdaq — Torteando a los Permabearish")

    monkeypatch.setattr("yt_sub_hub.doctor.fetch_video_metadata", fetch_video_metadata)

    report = run_doctor(
        settings=doctor_settings(
            proxy="http://127.0.0.1:12334",
            cookies_file=str(cookies_file),
            browser_cookies="chrome",
        )
    )
    output = format_report(report)

    assert "OK yt-dlp proxy: configured" in output
    assert "OK yt-dlp cookies file: configured" in output
    assert "OK yt-dlp browser cookies: disabled" in output
    assert "OK youtube_metadata_probe: o6hlVz6mQZY Nasdaq — Torteando a los Permabearish" in output


def test_doctor_fails_when_configured_cookies_file_is_missing(monkeypatch):
    stub_base_doctor_checks(monkeypatch)
    monkeypatch.setattr(
        "yt_sub_hub.doctor.fetch_video_metadata",
        lambda url, settings=None: VideoMetadata(video_id="o6hlVz6mQZY", canonical_url=url, title="Title"),
    )

    output = format_report(run_doctor(settings=doctor_settings(cookies_file="/missing/youtube-cookies.txt")))

    assert "FAIL yt-dlp cookies file: configured but not readable: /missing/youtube-cookies.txt" in output


def test_doctor_reports_disabled_browser_cookies_by_default(monkeypatch):
    stub_base_doctor_checks(monkeypatch)
    monkeypatch.setattr(
        "yt_sub_hub.doctor.fetch_video_metadata",
        lambda url, settings=None: VideoMetadata(video_id="o6hlVz6mQZY", canonical_url=url, title="Title"),
    )

    output = format_report(run_doctor(settings=doctor_settings()))

    assert "OK yt-dlp proxy: not configured" in output
    assert "OK yt-dlp cookies file: not configured" in output
    assert "OK yt-dlp browser cookies: disabled" in output
    assert "OK media download: disabled" in output


def test_doctor_strict_fails_full_processing_when_media_download_disabled(monkeypatch):
    stub_base_doctor_checks(monkeypatch)
    monkeypatch.setattr(
        "yt_sub_hub.doctor.fetch_video_metadata",
        lambda url, settings=None: VideoMetadata(video_id="o6hlVz6mQZY", canonical_url=url, title="Title"),
    )

    report = run_doctor(settings=doctor_settings(acceptance_mode="full_processing", media_download_enabled=False), strict=True)
    output = format_report(report)

    assert report.ok is False
    assert "FAIL acceptance mode: full_processing requires YT_SUB_HUB_MEDIA_DOWNLOAD=enabled" in output


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            TimeoutError("timed out"),
            "YouTube metadata probe timed out. Set YT_SUB_HUB_YTDLP_PROXY=http://127.0.0.1:12334 or check network route.",
        ),
        (
            YouTubeAuthRequiredError(
                "Видео требует авторизованные cookies. Включите cookies явно через "
                "YT_SUB_HUB_YTDLP_COOKIES_FILE=/secure/path/youtube-cookies.txt. "
                "Browser cookies are disabled by default."
            ),
            "Видео требует авторизованные cookies. Включите cookies явно через "
            "YT_SUB_HUB_YTDLP_COOKIES_FILE=/secure/path/youtube-cookies.txt. "
            "Browser cookies are disabled by default.",
        ),
    ],
)
def test_doctor_strict_fails_metadata_probe_with_actionable_text(monkeypatch, error, expected):
    stub_base_doctor_checks(monkeypatch)

    def fail_metadata(url, settings=None):
        raise error

    monkeypatch.setattr("yt_sub_hub.doctor.fetch_video_metadata", fail_metadata)

    report = run_doctor(settings=doctor_settings(proxy="http://127.0.0.1:12334"), strict=True)
    output = format_report(report)

    assert report.ok is False
    assert "FAIL youtube_metadata_probe:" in output
    assert expected in output


def test_doctor_reports_selected_whisper_backend_and_torch_acceleration(monkeypatch):
    stub_base_doctor_checks(monkeypatch)
    monkeypatch.setattr(
        "yt_sub_hub.doctor.fetch_video_metadata",
        lambda url, settings=None: VideoMetadata(video_id="o6hlVz6mQZY", canonical_url=url, title="Title"),
    )
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    settings = doctor_settings()
    object.__setattr__(settings, "whisper_backend", "openai-whisper-cpu")
    object.__setattr__(settings, "whisper_model", "base")
    report = run_doctor(settings=settings)
    output = format_report(report)

    assert "OK whisper backend: openai-whisper-cpu model=base" in output
    assert "OK torch: installed" in output
    assert "OK torch mps: available" in output
    assert "OK torch cuda: unavailable" in output


def test_doctor_strict_fails_mps_backend_when_mps_unavailable(monkeypatch):
    stub_base_doctor_checks(monkeypatch)
    monkeypatch.setattr(
        "yt_sub_hub.doctor.fetch_video_metadata",
        lambda url, settings=None: VideoMetadata(video_id="o6hlVz6mQZY", canonical_url=url, title="Title"),
    )
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    settings = doctor_settings()
    object.__setattr__(settings, "whisper_backend", "openai-whisper-mps")
    object.__setattr__(settings, "whisper_model", "base")
    report = run_doctor(settings=settings, strict=True)
    output = format_report(report)

    assert report.ok is False
    assert "FAIL whisper backend validation: MPS is not available" in output


def test_doctor_strict_fails_whisper_cpp_backend_when_binary_or_model_missing(monkeypatch):
    stub_base_doctor_checks(monkeypatch)
    monkeypatch.setattr(
        "yt_sub_hub.doctor.fetch_video_metadata",
        lambda url, settings=None: VideoMetadata(video_id="o6hlVz6mQZY", canonical_url=url, title="Title"),
    )

    settings = doctor_settings()
    object.__setattr__(settings, "whisper_backend", "whisper-cpp-metal")
    object.__setattr__(settings, "whisper_cpp_bin", "/missing/whisper-cli")
    object.__setattr__(settings, "whisper_cpp_model", "/missing/ggml-base.bin")

    report = run_doctor(settings=settings, strict=True)
    output = format_report(report)

    assert report.ok is False
    assert "FAIL whisper.cpp binary: configured but not executable: /missing/whisper-cli" in output
    assert "FAIL whisper.cpp model: configured but missing: /missing/ggml-base.bin" in output


def test_doctor_accepts_managed_whisper_cpp_setup_without_manual_env(tmp_path, monkeypatch):
    stub_base_doctor_checks(monkeypatch)
    monkeypatch.setattr(
        "yt_sub_hub.doctor.fetch_video_metadata",
        lambda url, settings=None: VideoMetadata(video_id="o6hlVz6mQZY", canonical_url=url, title="Title"),
    )
    binary = tmp_path / "whisper-cli"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    model = tmp_path / "assets" / "whisper.cpp" / "models" / "ggml-base.bin"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model")
    monkeypatch.setattr("shutil.which", lambda name: str(binary) if name == "whisper-cli" else None)
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: types.SimpleNamespace(returncode=0, stdout="whisper.cpp help", stderr=""),
    )

    settings = doctor_settings()
    object.__setattr__(settings, "library_dir", tmp_path)
    object.__setattr__(settings, "whisper_backend", "whisper-cpp-metal")
    object.__setattr__(settings, "whisper_cpp_bin", None)
    object.__setattr__(settings, "whisper_cpp_model", None)

    report = run_doctor(settings=settings, strict=True)
    output = format_report(report)

    assert "OK whisper.cpp binary:" in output
    assert str(binary) in output
    assert "OK whisper.cpp model:" in output
    assert str(model) in output
    assert "OK whisper.cpp probe: --help responded" in output
