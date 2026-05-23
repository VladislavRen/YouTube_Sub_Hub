from pathlib import Path

from yt_sub_hub.config import get_settings, normalize_acceptance_mode, normalize_quality_mode


def test_get_settings_loads_dotenv_file(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "YT_SUB_HUB_LIBRARY_DIR=custom-library",
                "YT_SUB_HUB_QWEN_MODEL=qwen/qwen3.6-35b-a3b",
                "YT_SUB_HUB_GEMMA_MODEL=google/gemma-4-e4b",
                "YT_SUB_HUB_NEO4J_BROWSER_URL=http://neo4j.local/browser/",
                "YT_SUB_HUB_ENABLE_GRAPH=false",
                "YT_SUB_HUB_ENABLE_DEEP_GRAPH_LLM=true",
                "YT_SUB_HUB_DEEP_GRAPH_MIN_CONFIDENCE=0.72",
                "YT_SUB_HUB_DEEP_GRAPH_MAX_CHUNK_CHARS=1800",
                "YT_SUB_HUB_DEEP_GRAPH_MAX_OUTPUT_TOKENS=1234",
                "YT_SUB_HUB_DEEP_GRAPH_BEST_EFFORT=false",
                "YT_SUB_HUB_ENABLE_TRANSLATION=true",
                "YT_SUB_HUB_TARGET_LANGUAGE=es",
                "YT_SUB_HUB_TRANSLATION_MODEL=custom/translator",
                "YT_SUB_HUB_TRANSLATION_MAX_CHARS=1200",
                "YT_SUB_HUB_TRANSLATION_BEST_EFFORT=true",
                "YT_SUB_HUB_YTDLP_COOKIES_FROM_BROWSER=chrome",
                "YT_SUB_HUB_YTDLP_COOKIES_FILE=/secure/youtube-cookies.txt",
                "YT_SUB_HUB_YTDLP_PROXY=http://127.0.0.1:12334",
                "YT_SUB_HUB_YTDLP_JS_RUNTIME=node",
                "YT_SUB_HUB_YTDLP_SOCKET_TIMEOUT=90",
                "YT_SUB_HUB_YTDLP_RETRIES=12",
                "YT_SUB_HUB_YOUTUBE_API_TOKEN_FILE=/secure/token.json",
                "YT_SUB_HUB_MEDIA_DOWNLOAD=disabled",
            ]
        ),
        encoding="utf-8",
    )

    settings = get_settings(env_file=env_file)

    assert settings.library_dir == Path("custom-library")
    assert settings.qwen_model == "qwen/qwen3.6-35b-a3b"
    assert settings.gemma_model == "google/gemma-4-e4b"
    assert settings.neo4j_browser_url == "http://neo4j.local/browser/"
    assert settings.enable_graph is False
    assert settings.enable_deep_graph_llm is True
    assert settings.deep_graph_min_confidence == 0.72
    assert settings.deep_graph_max_chunk_chars == 1800
    assert settings.deep_graph_max_output_tokens == 1234
    assert settings.deep_graph_best_effort is False
    assert settings.enable_translation is True
    assert settings.target_language == "es"
    assert settings.translation_model == "custom/translator"
    assert settings.translation_max_chars == 1200
    assert settings.translation_best_effort is True
    assert settings.ytdlp_cookies_from_browser == "chrome"
    assert settings.ytdlp_cookies_file == "/secure/youtube-cookies.txt"
    assert settings.ytdlp_proxy == "http://127.0.0.1:12334"
    assert settings.ytdlp_js_runtime == "node"
    assert settings.ytdlp_socket_timeout == 90
    assert settings.ytdlp_retries == 12
    assert settings.youtube_api_token_file == "/secure/token.json"
    assert settings.media_download_enabled is False


def test_get_settings_loads_whisper_backend_config(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "YT_SUB_HUB_WHISPER_BACKEND=openai-whisper-mps",
                "YT_SUB_HUB_WHISPER_MODEL=small",
                "YT_SUB_HUB_WHISPER_FP16=false",
                "YT_SUB_HUB_WHISPER_CPP_BIN=/opt/whisper.cpp/whisper-cli",
                "YT_SUB_HUB_WHISPER_CPP_MODEL=/models/ggml-base.bin",
                "YT_SUB_HUB_WHISPER_CPP_THREADS=6",
                "YT_SUB_HUB_WHISPER_CPP_EXTRA_ARGS=--language en --print-progress",
            ]
        ),
        encoding="utf-8",
    )

    settings = get_settings(env_file=env_file)

    assert settings.whisper_backend == "openai-whisper-mps"
    assert settings.whisper_model == "small"
    assert settings.whisper_fp16 is False
    assert settings.whisper_cpp_bin == "/opt/whisper.cpp/whisper-cli"
    assert settings.whisper_cpp_model == "/models/ggml-base.bin"
    assert settings.whisper_cpp_threads == 6
    assert settings.whisper_cpp_extra_args == "--language en --print-progress"


def test_whisper_backend_defaults_are_safe_cpu(monkeypatch):
    for key in [
        "YT_SUB_HUB_WHISPER_BACKEND",
        "YT_SUB_HUB_WHISPER_MODEL",
        "YT_SUB_HUB_WHISPER_FP16",
        "YT_SUB_HUB_WHISPER_CPP_BIN",
        "YT_SUB_HUB_WHISPER_CPP_MODEL",
        "YT_SUB_HUB_WHISPER_CPP_THREADS",
        "YT_SUB_HUB_WHISPER_CPP_EXTRA_ARGS",
    ]:
        monkeypatch.delenv(key, raising=False)

    settings = get_settings(env_file=Path("/tmp/non-existent-yt-sub-hub.env"))

    assert settings.whisper_backend == "openai-whisper-cpu"
    assert settings.whisper_model == "base"
    assert settings.whisper_fp16 is False
    assert settings.whisper_cpp_bin is None
    assert settings.whisper_cpp_model is None
    assert settings.whisper_cpp_threads is None
    assert settings.whisper_cpp_extra_args == ""


def test_invalid_whisper_backend_is_rejected_with_clear_error(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("YT_SUB_HUB_WHISPER_BACKEND=fast-mystery\n", encoding="utf-8")

    try:
        get_settings(env_file=env_file)
    except ValueError as exc:
        assert "YT_SUB_HUB_WHISPER_BACKEND" in str(exc)
        assert "openai-whisper-cpu" in str(exc)
        assert "whisper-cpp-metal" in str(exc)
    else:
        raise AssertionError("invalid whisper backend should fail settings loading")


def test_default_prompt_budget_is_safe_for_gemma_context(monkeypatch):
    for key in ["YT_SUB_HUB_MAX_PROMPT_CHARS"]:
        monkeypatch.delenv(key, raising=False)

    env_file = Path("/tmp/non-existent-yt-sub-hub.env")
    settings = get_settings(env_file=env_file)

    assert settings.max_prompt_chars == 3500


def test_get_settings_loads_translation_config(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "YT_SUB_HUB_ENABLE_TRANSLATION=true",
                "YT_SUB_HUB_TARGET_LANGUAGE=ru",
                "YT_SUB_HUB_TRANSLATION_MODEL=google/gemma-4-e4b",
                "YT_SUB_HUB_TRANSLATION_MAX_CHARS=1800",
                "YT_SUB_HUB_TRANSLATION_CHUNK_MAX_SEGMENTS=12",
                "YT_SUB_HUB_TRANSLATION_CHUNK_MAX_CHARS=900",
                "YT_SUB_HUB_TRANSLATION_CHUNK_TIMEOUT_SECONDS=75",
                "YT_SUB_HUB_LM_STUDIO_TRANSLATION_CONCURRENCY=2",
                "YT_SUB_HUB_LM_STUDIO_BENCHMARK_TIMEOUT_SECONDS=45",
                "YT_SUB_HUB_LM_STUDIO_BENCHMARK_RUN_TIMEOUT_SECONDS=900",
                "YT_SUB_HUB_LM_STUDIO_BENCHMARK_SAMPLE_CHUNKS=6",
                "YT_SUB_HUB_TRANSLATION_BEST_EFFORT=true",
            ]
        ),
        encoding="utf-8",
    )

    settings = get_settings(env_file=env_file)

    assert settings.enable_translation is True
    assert settings.target_language == "ru"
    assert settings.translation_model == "google/gemma-4-e4b"
    assert settings.translation_max_chars == 1800
    assert settings.translation_chunk_max_segments == 12
    assert settings.translation_chunk_max_chars == 900
    assert settings.translation_chunk_timeout_seconds == 75
    assert settings.lm_studio_translation_concurrency == 2
    assert settings.lm_studio_benchmark_timeout_seconds == 45
    assert settings.lm_studio_benchmark_run_timeout_seconds == 900
    assert settings.lm_studio_benchmark_sample_chunks == 6
    assert settings.translation_best_effort is True


def test_translation_settings_defaults(monkeypatch):
    for key in [
        "YT_SUB_HUB_ENABLE_TRANSLATION",
        "YT_SUB_HUB_TARGET_LANGUAGE",
        "YT_SUB_HUB_TRANSLATION_MODEL",
        "YT_SUB_HUB_TRANSLATION_MAX_CHARS",
        "YT_SUB_HUB_TRANSLATION_CHUNK_MAX_SEGMENTS",
        "YT_SUB_HUB_TRANSLATION_CHUNK_MAX_CHARS",
        "YT_SUB_HUB_TRANSLATION_CHUNK_TIMEOUT_SECONDS",
        "YT_SUB_HUB_LM_STUDIO_TRANSLATION_CONCURRENCY",
        "YT_SUB_HUB_LM_STUDIO_BENCHMARK_TIMEOUT_SECONDS",
        "YT_SUB_HUB_LM_STUDIO_BENCHMARK_RUN_TIMEOUT_SECONDS",
        "YT_SUB_HUB_LM_STUDIO_BENCHMARK_SAMPLE_CHUNKS",
        "YT_SUB_HUB_TRANSLATION_BEST_EFFORT",
    ]:
        monkeypatch.delenv(key, raising=False)

    env_file = Path("/tmp/non-existent-yt-sub-hub.env")
    settings = get_settings(env_file=env_file)

    assert settings.enable_translation is False
    assert settings.target_language == "ru"
    assert settings.translation_model == "google/gemma-4-e4b"
    assert settings.translation_max_chars == 2500
    assert settings.translation_chunk_max_segments == 30
    assert settings.translation_chunk_max_chars == 2000
    assert settings.translation_chunk_timeout_seconds == 120
    assert settings.lm_studio_translation_concurrency == 1
    assert settings.lm_studio_benchmark_timeout_seconds == 120
    assert settings.lm_studio_benchmark_run_timeout_seconds == 0
    assert settings.lm_studio_benchmark_sample_chunks == 12
    assert settings.translation_best_effort is False


def test_media_download_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("YT_SUB_HUB_MEDIA_DOWNLOAD", raising=False)

    settings = get_settings(env_file=Path("/tmp/non-existent-yt-sub-hub.env"))

    assert settings.media_download_enabled is False


def test_acceptance_mode_defaults_from_media_download(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("YT_SUB_HUB_ACCEPTANCE_MODE", raising=False)

    disabled = tmp_path / "disabled.env"
    disabled.write_text("YT_SUB_HUB_MEDIA_DOWNLOAD=disabled\n", encoding="utf-8")
    enabled = tmp_path / "enabled.env"
    enabled.write_text("YT_SUB_HUB_MEDIA_DOWNLOAD=enabled\n", encoding="utf-8")

    assert get_settings(env_file=disabled).acceptance_mode == "data_api_only"
    assert get_settings(env_file=enabled).acceptance_mode == "full_processing"


def test_acceptance_mode_accepts_supported_values(tmp_path: Path):
    for mode in ["data_api_only", "full_processing"]:
        env_file = tmp_path / f"{mode}.env"
        env_file.write_text(f"YT_SUB_HUB_ACCEPTANCE_MODE={mode}\n", encoding="utf-8")

        assert get_settings(env_file=env_file).acceptance_mode == mode
        assert normalize_acceptance_mode(mode.upper()) == mode


def test_invalid_acceptance_mode_is_rejected_with_clear_error(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("YT_SUB_HUB_ACCEPTANCE_MODE=metadata_only\n", encoding="utf-8")

    try:
        get_settings(env_file=env_file)
    except ValueError as exc:
        assert str(exc) == "YT_SUB_HUB_ACCEPTANCE_MODE must be data_api_only or full_processing"
    else:
        raise AssertionError("invalid acceptance mode should fail settings loading")


def test_quality_mode_defaults_to_strict(monkeypatch):
    monkeypatch.delenv("YT_SUB_HUB_QUALITY_MODE", raising=False)

    settings = get_settings(env_file=Path("/tmp/non-existent-yt-sub-hub.env"))

    assert settings.quality_mode == "strict"


def test_quality_mode_accepts_supported_values(tmp_path: Path):
    for mode in ["strict", "best_effort", "draft"]:
        env_file = tmp_path / f"{mode}.env"
        env_file.write_text(f"YT_SUB_HUB_QUALITY_MODE={mode}\n", encoding="utf-8")

        assert get_settings(env_file=env_file).quality_mode == mode
        assert normalize_quality_mode(mode.upper()) == mode


def test_invalid_quality_mode_is_rejected_with_clear_error(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("YT_SUB_HUB_QUALITY_MODE=loose\n", encoding="utf-8")

    try:
        get_settings(env_file=env_file)
    except ValueError as exc:
        assert str(exc) == "YT_SUB_HUB_QUALITY_MODE must be strict, best_effort, or draft"
    else:
        raise AssertionError("invalid quality mode should fail settings loading")
