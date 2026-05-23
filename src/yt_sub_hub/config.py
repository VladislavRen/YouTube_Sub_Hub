from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

WHISPER_BACKENDS = {"openai-whisper-cpu", "openai-whisper-mps", "whisper-cpp-metal"}
QUALITY_MODES = {"strict", "best_effort", "draft"}
ACCEPTANCE_MODES = {"data_api_only", "full_processing"}


@dataclass(frozen=True)
class Settings:
    library_dir: Path
    lm_studio_base_url: str
    lm_studio_api_key: str
    qwen_model: str
    gemma_model: str
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    keep_audio: bool
    enable_graph: bool
    graph_best_effort: bool
    lm_studio_timeout_seconds: float
    max_prompt_chars: int
    ytdlp_cookies_from_browser: str | None
    ytdlp_js_runtime: str | None
    ytdlp_socket_timeout: float
    ytdlp_retries: int
    enable_visual_analysis: bool = True
    vision_model: str = ""
    frame_image_format: str = "jpg"
    frame_context_seconds: int = 20
    visual_max_output_tokens: int = 1200
    visual_best_effort: bool = True
    enable_deep_graph_llm: bool = False
    deep_graph_min_confidence: float = 0.55
    deep_graph_max_chunk_chars: int = 2500
    deep_graph_max_output_tokens: int = 1500
    deep_graph_best_effort: bool = True
    enable_translation: bool = False
    target_language: str = "ru"
    translation_model: str = "google/gemma-4-e4b"
    translation_max_chars: int = 2500
    translation_chunk_max_segments: int = 30
    translation_chunk_max_chars: int = 2000
    translation_chunk_timeout_seconds: float = 120
    lm_studio_translation_concurrency: int = 1
    youtube_data_api_concurrency: int = 2
    local_candidate_scoring_concurrency: int = 2
    llm_discovery_review_concurrency: int = 1
    neo4j_sync_concurrency: int = 1
    queue_mutation_concurrency: int = 1
    lm_studio_benchmark_timeout_seconds: float = 120
    lm_studio_benchmark_run_timeout_seconds: float = 0
    lm_studio_benchmark_sample_chunks: int = 12
    translation_best_effort: bool = False
    analysis_map_reduce_enabled: bool = True
    analysis_chunk_max_segments: int = 120
    analysis_chunk_max_chars: int = 4500
    analysis_chunk_timeout_seconds: float = 120
    analysis_reduce_timeout_seconds: float = 120
    analysis_direct_max_segments: int = 180
    ytdlp_cookies_file: str | None = None
    ytdlp_proxy: str | None = None
    whisper_backend: str = "openai-whisper-cpu"
    whisper_model: str = "base"
    whisper_fp16: bool = False
    whisper_cpp_bin: str | None = None
    whisper_cpp_model: str | None = None
    whisper_cpp_threads: int | None = None
    whisper_cpp_extra_args: str = ""
    runtime_stale_after_seconds: int = 1800
    youtube_api_token_file: str | None = "token.json"
    media_download_enabled: bool = False
    acceptance_mode: Literal["data_api_only", "full_processing"] | str = "data_api_only"
    neo4j_browser_url: str = "http://localhost:7474/browser/"
    quality_mode: Literal["strict", "best_effort", "draft"] | str = "strict"


def _bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _load_dotenv(env_file: Path) -> dict[str, str]:
    if not env_file.exists():
        return {}
    try:
        from dotenv import dotenv_values

        return {key: str(value) for key, value in dotenv_values(env_file).items() if value is not None}
    except ImportError:
        values: dict[str, str] = {}
        for line in env_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
        return values


def _env(name: str, default: str, values: Mapping[str, str]) -> str:
    return os.getenv(name) or values.get(name) or default


def _optional_int(value: str) -> int | None:
    stripped = value.strip()
    return int(stripped) if stripped else None


def _whisper_backend(value: str) -> str:
    backend = value.strip()
    if backend not in WHISPER_BACKENDS:
        allowed = ", ".join(sorted(WHISPER_BACKENDS))
        raise ValueError(f"Invalid YT_SUB_HUB_WHISPER_BACKEND={backend!r}. Allowed values: {allowed}")
    return backend


def _media_download_enabled(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"enabled", "enable", "true", "1", "yes", "on"}:
        return True
    if normalized in {"disabled", "disable", "false", "0", "no", "off"}:
        return False
    raise ValueError("Invalid YT_SUB_HUB_MEDIA_DOWNLOAD value. Allowed values: enabled, disabled")


def normalize_quality_mode(value: str | None) -> str:
    mode = (value or "strict").strip().lower()
    if mode not in QUALITY_MODES:
        raise ValueError("YT_SUB_HUB_QUALITY_MODE must be strict, best_effort, or draft")
    return mode


def normalize_acceptance_mode(value: str | None) -> str:
    mode = (value or "data_api_only").strip().lower()
    if mode not in ACCEPTANCE_MODES:
        raise ValueError("YT_SUB_HUB_ACCEPTANCE_MODE must be data_api_only or full_processing")
    return mode


def get_settings(env_file: Path | str | None = None) -> Settings:
    dotenv_path = Path(env_file) if env_file is not None else Path(".env")
    values = _load_dotenv(dotenv_path)
    media_download_enabled = _media_download_enabled(_env("YT_SUB_HUB_MEDIA_DOWNLOAD", "disabled", values))
    acceptance_mode_raw = _env("YT_SUB_HUB_ACCEPTANCE_MODE", "", values).strip()
    acceptance_mode = (
        normalize_acceptance_mode(acceptance_mode_raw)
        if acceptance_mode_raw
        else ("full_processing" if media_download_enabled else "data_api_only")
    )
    return Settings(
        library_dir=Path(_env("YT_SUB_HUB_LIBRARY_DIR", "library", values)),
        lm_studio_base_url=_env("YT_SUB_HUB_LM_STUDIO_BASE_URL", "http://localhost:1234/v1", values),
        lm_studio_api_key=_env("YT_SUB_HUB_LM_STUDIO_API_KEY", "lm-studio", values),
        qwen_model=_env("YT_SUB_HUB_QWEN_MODEL", "qwen/qwen3.6-35b-a3b", values),
        gemma_model=_env("YT_SUB_HUB_GEMMA_MODEL", "google/gemma-4-e4b", values),
        neo4j_uri=_env("YT_SUB_HUB_NEO4J_URI", "bolt://localhost:7687", values),
        neo4j_user=_env("YT_SUB_HUB_NEO4J_USER", "neo4j", values),
        neo4j_password=_env("YT_SUB_HUB_NEO4J_PASSWORD", "password", values),
        neo4j_browser_url=_env("YT_SUB_HUB_NEO4J_BROWSER_URL", "http://localhost:7474/browser/", values),
        keep_audio=_bool(_env("YT_SUB_HUB_KEEP_AUDIO", "true", values), True),
        enable_graph=_bool(_env("YT_SUB_HUB_ENABLE_GRAPH", "true", values), True),
        graph_best_effort=_bool(_env("YT_SUB_HUB_GRAPH_BEST_EFFORT", "true", values), True),
        enable_deep_graph_llm=_bool(_env("YT_SUB_HUB_ENABLE_DEEP_GRAPH_LLM", "false", values), False),
        deep_graph_min_confidence=float(_env("YT_SUB_HUB_DEEP_GRAPH_MIN_CONFIDENCE", "0.55", values)),
        deep_graph_max_chunk_chars=int(_env("YT_SUB_HUB_DEEP_GRAPH_MAX_CHUNK_CHARS", "2500", values)),
        deep_graph_max_output_tokens=int(_env("YT_SUB_HUB_DEEP_GRAPH_MAX_OUTPUT_TOKENS", "1500", values)),
        deep_graph_best_effort=_bool(_env("YT_SUB_HUB_DEEP_GRAPH_BEST_EFFORT", "true", values), True),
        lm_studio_timeout_seconds=float(_env("YT_SUB_HUB_LM_STUDIO_TIMEOUT_SECONDS", "120", values)),
        max_prompt_chars=int(_env("YT_SUB_HUB_MAX_PROMPT_CHARS", "3500", values)),
        ytdlp_cookies_from_browser=_env("YT_SUB_HUB_YTDLP_COOKIES_FROM_BROWSER", "", values) or None,
        ytdlp_cookies_file=_env("YT_SUB_HUB_YTDLP_COOKIES_FILE", "", values) or None,
        ytdlp_proxy=_env("YT_SUB_HUB_YTDLP_PROXY", "", values) or None,
        ytdlp_js_runtime=_env("YT_SUB_HUB_YTDLP_JS_RUNTIME", "", values) or None,
        ytdlp_socket_timeout=float(_env("YT_SUB_HUB_YTDLP_SOCKET_TIMEOUT", "60", values)),
        ytdlp_retries=int(_env("YT_SUB_HUB_YTDLP_RETRIES", "10", values)),
        enable_visual_analysis=_bool(_env("YT_SUB_HUB_ENABLE_VISUAL_ANALYSIS", "true", values), True),
        vision_model=_env("YT_SUB_HUB_VISION_MODEL", "", values),
        frame_image_format=_env("YT_SUB_HUB_FRAME_IMAGE_FORMAT", "jpg", values),
        frame_context_seconds=int(_env("YT_SUB_HUB_FRAME_CONTEXT_SECONDS", "20", values)),
        visual_max_output_tokens=int(_env("YT_SUB_HUB_VISUAL_MAX_OUTPUT_TOKENS", "1200", values)),
        visual_best_effort=_bool(_env("YT_SUB_HUB_VISUAL_BEST_EFFORT", "true", values), True),
        enable_translation=_bool(_env("YT_SUB_HUB_ENABLE_TRANSLATION", "false", values), False),
        target_language=_env("YT_SUB_HUB_TARGET_LANGUAGE", "ru", values),
        translation_model=_env("YT_SUB_HUB_TRANSLATION_MODEL", "google/gemma-4-e4b", values),
        translation_max_chars=int(_env("YT_SUB_HUB_TRANSLATION_MAX_CHARS", "2500", values)),
        translation_chunk_max_segments=int(_env("YT_SUB_HUB_TRANSLATION_CHUNK_MAX_SEGMENTS", "30", values)),
        translation_chunk_max_chars=int(_env("YT_SUB_HUB_TRANSLATION_CHUNK_MAX_CHARS", "2000", values)),
        translation_chunk_timeout_seconds=float(_env("YT_SUB_HUB_TRANSLATION_CHUNK_TIMEOUT_SECONDS", "120", values)),
        lm_studio_translation_concurrency=int(_env("YT_SUB_HUB_LM_STUDIO_TRANSLATION_CONCURRENCY", "1", values)),
        youtube_data_api_concurrency=int(_env("YT_SUB_HUB_YOUTUBE_DATA_API_CONCURRENCY", "2", values)),
        local_candidate_scoring_concurrency=int(_env("YT_SUB_HUB_LOCAL_CANDIDATE_SCORING_CONCURRENCY", "2", values)),
        llm_discovery_review_concurrency=int(_env("YT_SUB_HUB_LLM_DISCOVERY_REVIEW_CONCURRENCY", "1", values)),
        neo4j_sync_concurrency=int(_env("YT_SUB_HUB_NEO4J_SYNC_CONCURRENCY", "1", values)),
        queue_mutation_concurrency=int(_env("YT_SUB_HUB_QUEUE_MUTATION_CONCURRENCY", "1", values)),
        lm_studio_benchmark_timeout_seconds=float(_env("YT_SUB_HUB_LM_STUDIO_BENCHMARK_TIMEOUT_SECONDS", "120", values)),
        lm_studio_benchmark_run_timeout_seconds=float(_env("YT_SUB_HUB_LM_STUDIO_BENCHMARK_RUN_TIMEOUT_SECONDS", "0", values)),
        lm_studio_benchmark_sample_chunks=int(_env("YT_SUB_HUB_LM_STUDIO_BENCHMARK_SAMPLE_CHUNKS", "12", values)),
        translation_best_effort=_bool(_env("YT_SUB_HUB_TRANSLATION_BEST_EFFORT", "false", values), False),
        analysis_map_reduce_enabled=_bool(_env("YT_SUB_HUB_ANALYSIS_MAP_REDUCE_ENABLED", "true", values), True),
        analysis_chunk_max_segments=int(_env("YT_SUB_HUB_ANALYSIS_CHUNK_MAX_SEGMENTS", "120", values)),
        analysis_chunk_max_chars=int(_env("YT_SUB_HUB_ANALYSIS_CHUNK_MAX_CHARS", "4500", values)),
        analysis_chunk_timeout_seconds=float(_env("YT_SUB_HUB_ANALYSIS_CHUNK_TIMEOUT_SECONDS", "120", values)),
        analysis_reduce_timeout_seconds=float(_env("YT_SUB_HUB_ANALYSIS_REDUCE_TIMEOUT_SECONDS", "120", values)),
        analysis_direct_max_segments=int(_env("YT_SUB_HUB_ANALYSIS_DIRECT_MAX_SEGMENTS", "180", values)),
        whisper_backend=_whisper_backend(_env("YT_SUB_HUB_WHISPER_BACKEND", "openai-whisper-cpu", values)),
        whisper_model=_env("YT_SUB_HUB_WHISPER_MODEL", "base", values),
        whisper_fp16=_bool(_env("YT_SUB_HUB_WHISPER_FP16", "false", values), False),
        whisper_cpp_bin=_env("YT_SUB_HUB_WHISPER_CPP_BIN", "", values) or None,
        whisper_cpp_model=_env("YT_SUB_HUB_WHISPER_CPP_MODEL", "", values) or None,
        whisper_cpp_threads=_optional_int(_env("YT_SUB_HUB_WHISPER_CPP_THREADS", "", values)),
        whisper_cpp_extra_args=_env("YT_SUB_HUB_WHISPER_CPP_EXTRA_ARGS", "", values),
        runtime_stale_after_seconds=int(_env("YT_SUB_HUB_RUNTIME_STALE_AFTER_SECONDS", "1800", values)),
        youtube_api_token_file=_env("YT_SUB_HUB_YOUTUBE_API_TOKEN_FILE", "token.json", values) or None,
        media_download_enabled=media_download_enabled,
        acceptance_mode=acceptance_mode,
        quality_mode=normalize_quality_mode(_env("YT_SUB_HUB_QUALITY_MODE", "strict", values)),
    )
