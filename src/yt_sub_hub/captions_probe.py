from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from yt_sub_hub.channel import ChannelDataApiError, fetch_channel_listing
from yt_sub_hub.config import Settings, get_settings
from yt_sub_hub.ingestion import InvalidYouTubeUrl, parse_youtube_api_duration, parse_youtube_url
from yt_sub_hub.transcripts import (
    list_youtube_data_api_captions,
    probe_youtube_data_api_caption_download,
)
from yt_sub_hub.youtube_data_api import (
    YouTubeDataApiClientError,
    build_youtube_client,
    google_api_error_class,
    sanitize_google_api_message,
)

SCHEMA_VERSION = "captions_probe.v1"
BLOCKER_CAPTIONS_UNAVAILABLE_MEDIA_DISABLED = "captions_unavailable_media_download_disabled"
VIDEO_ID_LENGTH = 11


class CaptionsProbeError(ValueError):
    pass


@dataclass(frozen=True)
class TargetResolution:
    kind: str
    video_id: str | None = None
    channel_target: str | None = None


def probe_captions(
    target: str,
    *,
    limit: int | None = 3,
    settings: Settings | None = None,
    youtube_service: Any | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    resolved = _resolve_target(target)
    try:
        youtube = youtube_service or build_youtube_client(settings, persist_refresh=False)
    except YouTubeDataApiClientError as exc:
        if resolved.kind == "video":
            video = _client_error_video_payload(str(resolved.video_id), settings, exc)
            return _payload(target, "video", limit, settings, [video])
        raise CaptionsProbeError(sanitize_google_api_message(exc)) from exc
    if resolved.kind == "video":
        video = probe_video_captions(str(resolved.video_id), settings=settings, youtube_service=youtube)
        return _payload(target, "video", limit, settings, [video])
    return _probe_channel(str(resolved.channel_target), limit=limit, settings=settings, youtube_service=youtube)


def probe_video_captions(
    video_id: str,
    *,
    settings: Settings | None = None,
    youtube_service: Any | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    try:
        youtube = youtube_service or build_youtube_client(settings, persist_refresh=False)
    except YouTubeDataApiClientError as exc:
        return _client_error_video_payload(video_id, settings, exc)
    media_mode = _media_download_mode(settings)
    metadata = _probe_metadata(youtube, video_id)
    if metadata["status"] != "ok":
        decision = _decision_from_error_class(str(metadata.get("error_class") or "unavailable"))
        return _video_payload(
            video_id,
            metadata=metadata,
            captions=[],
            downloadable=False,
            download_error_class=metadata.get("error_class"),
            download_error_message=metadata.get("error_message"),
            selected_caption_language=None,
            decision=decision,
            media_download_mode=media_mode,
            acceptance_mode=settings.acceptance_mode,
        )

    try:
        raw_captions = list_youtube_data_api_captions(video_id, settings=settings, youtube_service=youtube)
    except Exception as exc:
        error_class = _classify_exception(exc)
        decision = _decision_from_error_class(error_class)
        if error_class == "forbidden":
            decision = "needs_whisper_fallback"
        return _video_payload(
            video_id,
            metadata=metadata,
            captions=[],
            downloadable=False,
            download_error_class=error_class,
            download_error_message=sanitize_google_api_message(exc),
            selected_caption_language=None,
            decision=decision,
            media_download_mode=media_mode,
            acceptance_mode=settings.acceptance_mode,
        )

    captions = [_caption_payload(item) for item in raw_captions]
    selected = _select_caption(raw_captions, settings.target_language, metadata)
    if selected is None:
        return _video_payload(
            video_id,
            metadata=metadata,
            captions=captions,
            downloadable=False,
            download_error_class=None,
            download_error_message=None,
            selected_caption_language=None,
            decision="needs_whisper_fallback",
            media_download_mode=media_mode,
            acceptance_mode=settings.acceptance_mode,
        )

    selected_id = selected.get("id")
    selected_language = ((selected.get("snippet") or {}).get("language") or None) if isinstance(selected, dict) else None
    try:
        download = probe_youtube_data_api_caption_download(
            video_id,
            str(selected_id),
            settings=settings,
            youtube_service=youtube,
        )
    except Exception as exc:
        error_class = _classify_exception(exc)
        decision = _decision_from_error_class(error_class)
        if error_class == "forbidden":
            decision = "needs_whisper_fallback"
        return _video_payload(
            video_id,
            metadata=metadata,
            captions=captions,
            downloadable=False,
            download_error_class=error_class,
            download_error_message=_redact_caption_id(sanitize_google_api_message(exc), str(selected_id)),
            selected_caption_language=selected_language,
            decision=decision,
            media_download_mode=media_mode,
            acceptance_mode=settings.acceptance_mode,
        )

    downloadable = bool(download.get("downloadable"))
    download_error_class = download.get("download_error_class")
    decision = "data_api_captions_ready" if downloadable else _decision_from_download_error(download_error_class)
    return _video_payload(
        video_id,
        metadata=metadata,
        captions=captions,
        downloadable=downloadable,
        download_error_class=download_error_class,
        download_error_message=_redact_caption_id(download.get("download_error_message"), str(selected_id)),
        selected_caption_language=selected_language,
        decision=decision,
        media_download_mode=media_mode,
        acceptance_mode=settings.acceptance_mode,
    )


def format_captions_probe_human(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    for video in payload.get("videos", []):
        lines.append(f"Video {video.get('video_id')}")
        lines.append(f"metadata: {video.get('metadata', {}).get('status', 'unknown')}")
        lines.append(f"captions: {video.get('captions_count', 0)}")
        lines.append(f"downloadable: {'yes' if video.get('downloadable') else 'no'}")
        lines.append(f"decision: {video.get('transcript_source_decision')}")
        blocker = video.get("pipeline_blocker")
        if blocker:
            lines.append(f"blocker: {blocker}")
        lines.append("")
    summary = payload.get("summary") or {}
    lines.append(f"can_run_data_api_only: {str(summary.get('can_run_data_api_only')).lower()}")
    lines.append(f"can_run_full_processing: {str(summary.get('can_run_full_processing')).lower()}")
    return "\n".join(lines).rstrip()


def _probe_channel(
    target: str,
    *,
    limit: int | None,
    settings: Settings,
    youtube_service: Any,
) -> dict[str, Any]:
    try:
        listing = fetch_channel_listing(target, limit=limit, settings=settings, youtube_service=youtube_service)
    except ChannelDataApiError as exc:
        raise CaptionsProbeError(sanitize_google_api_message(exc)) from exc
    videos = [
        probe_video_captions(video.video_id, settings=settings, youtube_service=youtube_service)
        for video in listing.videos[:limit]
    ]
    return _payload(target, "channel", limit, settings, videos)


def _payload(target: str, target_kind: str, limit: int | None, settings: Settings, videos: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "target": target,
        "target_kind": target_kind,
        "limit": limit,
        "target_language": settings.target_language,
        "media_download_mode": _media_download_mode(settings),
        "acceptance_mode": settings.acceptance_mode,
        "summary": _summary(videos, settings),
        "videos": videos,
    }


def _video_payload(
    video_id: str,
    *,
    metadata: dict[str, Any],
    captions: list[dict[str, Any]],
    downloadable: bool,
    download_error_class: object,
    download_error_message: object,
    selected_caption_language: str | None,
    decision: str,
    media_download_mode: str,
    acceptance_mode: str,
) -> dict[str, Any]:
    whisper_required = decision == "needs_whisper_fallback"
    whisper_allowed = whisper_required and media_download_mode == "enabled" and acceptance_mode == "full_processing"
    return {
        "video_id": video_id,
        "canonical_url": f"https://www.youtube.com/watch?v={video_id}",
        "metadata": metadata,
        "captions_count": len(captions),
        "captions": captions,
        "downloadable": downloadable,
        "download_error_class": download_error_class,
        "download_error_message": download_error_message,
        "selected_caption_language": selected_caption_language,
        "transcript_source_decision": decision,
        "whisper_fallback_required": whisper_required,
        "whisper_fallback_allowed": whisper_allowed,
        "media_download_mode": media_download_mode,
        "acceptance_mode": acceptance_mode,
        "pipeline_blocker": BLOCKER_CAPTIONS_UNAVAILABLE_MEDIA_DISABLED
        if whisper_required and not whisper_allowed
        else None,
    }


def _client_error_video_payload(video_id: str, settings: Settings, exc: BaseException) -> dict[str, Any]:
    return _video_payload(
        video_id,
        metadata={
            "status": "error",
            "error_class": "auth_error",
            "error_message": sanitize_google_api_message(exc),
        },
        captions=[],
        downloadable=False,
        download_error_class="auth_error",
        download_error_message=sanitize_google_api_message(exc),
        selected_caption_language=None,
        decision="auth_error",
        media_download_mode=_media_download_mode(settings),
        acceptance_mode=settings.acceptance_mode,
    )


def _probe_metadata(youtube: Any, video_id: str) -> dict[str, Any]:
    try:
        response = youtube.videos().list(part="snippet,contentDetails,status", id=video_id).execute()
    except Exception as exc:
        error_class = _classify_exception(exc)
        return {
            "status": "error",
            "error_class": error_class,
            "error_message": sanitize_google_api_message(exc),
        }
    items = response.get("items") or []
    if not items:
        return {"status": "error", "error_class": "unavailable", "error_message": "video metadata unavailable"}
    item = items[0]
    snippet = item.get("snippet") or {}
    content_details = item.get("contentDetails") or {}
    return {
        "status": "ok",
        "title": snippet.get("title"),
        "channel": snippet.get("channelTitle"),
        "default_language": snippet.get("defaultLanguage") or snippet.get("defaultAudioLanguage"),
        "duration_seconds": parse_youtube_api_duration(content_details.get("duration")),
    }


def _caption_payload(item: dict[str, Any]) -> dict[str, Any]:
    snippet = item.get("snippet") or {}
    return {
        "caption_id_masked": _mask_caption_id(str(item.get("id") or "")),
        "language": snippet.get("language"),
        "name": snippet.get("name"),
        "trackKind": snippet.get("trackKind"),
        "isAutoSynced": snippet.get("isAutoSynced"),
        "status": snippet.get("status"),
    }


def _select_caption(items: list[dict[str, Any]], target_language: str, metadata: dict[str, Any]) -> dict[str, Any] | None:
    languages = [target_language, metadata.get("default_language")]
    by_language = {
        (item.get("snippet") or {}).get("language"): item
        for item in items
        if (item.get("snippet") or {}).get("language")
    }
    for language in dict.fromkeys(language for language in languages if language):
        if language in by_language:
            return by_language[language]
    return items[0] if items else None


def _summary(videos: list[dict[str, Any]], settings: Settings) -> dict[str, Any]:
    ready = sum(1 for video in videos if video.get("transcript_source_decision") == "data_api_captions_ready")
    fallback = sum(1 for video in videos if video.get("transcript_source_decision") == "needs_whisper_fallback")
    unavailable = sum(1 for video in videos if video.get("transcript_source_decision") == "unavailable")
    return {
        "total_videos": len(videos),
        "data_api_caption_ready_count": ready,
        "needs_whisper_fallback_count": fallback,
        "unavailable_count": unavailable,
        "can_run_data_api_only": bool(videos) and ready == len(videos),
        "can_run_full_processing": bool(videos)
        and unavailable == 0
        and (fallback == 0 or settings.media_download_enabled),
    }


def _resolve_target(target: str) -> TargetResolution:
    stripped = target.strip()
    parsed = urlparse(stripped)
    if parsed.scheme and "youtube.com" not in parsed.netloc.lower() and "youtu.be" not in parsed.netloc.lower():
        raise CaptionsProbeError(f"Unsupported captions probe target: {target}")
    try:
        parsed_video = parse_youtube_url(stripped)
        return TargetResolution(kind="video", video_id=parsed_video.video_id)
    except InvalidYouTubeUrl:
        pass
    if not parsed.scheme and not parsed.netloc and _looks_like_video_id(stripped):
        return TargetResolution(kind="video", video_id=stripped)
    return TargetResolution(kind="channel", channel_target=stripped)


def _looks_like_video_id(value: str) -> bool:
    return len(value) == VIDEO_ID_LENGTH and all(ch.isalnum() or ch in {"_", "-"} for ch in value)


def _classify_exception(exc: BaseException) -> str:
    if isinstance(exc, YouTubeDataApiClientError):
        return "auth_error"
    return google_api_error_class(exc)


def _decision_from_error_class(error_class: str) -> str:
    if error_class == "auth_error":
        return "auth_error"
    if error_class == "quota_or_rate_limited":
        return "quota_or_rate_limited"
    if error_class == "unavailable":
        return "unavailable"
    return "unknown_error"


def _decision_from_download_error(error_class: object) -> str:
    if error_class in {"auth_error", "quota_or_rate_limited", "unavailable", "unknown_error"}:
        return str(error_class)
    return "needs_whisper_fallback"


def _media_download_mode(settings: Settings) -> str:
    return "enabled" if settings.media_download_enabled else "disabled"


def _mask_caption_id(caption_id: str) -> str:
    if not caption_id:
        return ""
    if len(caption_id) <= 8:
        return f"{caption_id[:2]}..."
    return f"{caption_id[:3]}...{caption_id[-3:]}"


def _redact_caption_id(message: object, caption_id: str) -> object:
    if message is None or not caption_id:
        return message
    return str(message).replace(caption_id, _mask_caption_id(caption_id))
