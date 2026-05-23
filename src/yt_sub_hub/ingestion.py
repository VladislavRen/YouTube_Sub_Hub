from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from yt_sub_hub.config import Settings, get_settings
from yt_sub_hub.models import ParsedYouTubeUrl, VideoMetadata
from yt_sub_hub.youtube_data_api import YOUTUBE_API_SCOPES, build_youtube_client

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,}$")
YOUTUBE_API_DURATION_RE = re.compile(
    r"^P(?:\d+Y)?(?:\d+M)?(?:\d+W)?(?:\d+D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$"
)


class InvalidYouTubeUrl(ValueError):
    """Raised when a URL is not a supported YouTube video URL."""


class YouTubeAuthRequiredError(RuntimeError):
    """Raised when YouTube requires explicit authenticated cookies."""


BOT_CHECK_MARKERS = (
    "sign in to confirm",
    "not a bot",
    "use --cookies-from-browser",
    "use --cookies",
)


def explain_youtube_auth_required() -> str:
    return (
        "Видео требует авторизованные cookies. Включите cookies явно через "
        "YT_SUB_HUB_YTDLP_COOKIES_FILE=/secure/path/youtube-cookies.txt. "
        "Browser cookies are disabled by default."
    )


def raise_if_youtube_auth_required(error: Exception) -> None:
    message = str(error).lower()
    if any(marker in message for marker in BOT_CHECK_MARKERS):
        raise YouTubeAuthRequiredError(explain_youtube_auth_required()) from error


def parse_youtube_url(url: str) -> ParsedYouTubeUrl:
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower().removeprefix("www.")
    video_id: str | None = None

    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
    elif host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]
        elif parsed.path.startswith("/embed/") or parsed.path.startswith("/shorts/"):
            video_id = parsed.path.strip("/").split("/")[1]

    if not video_id or not VIDEO_ID_RE.match(video_id):
        raise InvalidYouTubeUrl(f"Unsupported YouTube URL: {url}")

    return ParsedYouTubeUrl(
        video_id=video_id,
        canonical_url=f"https://www.youtube.com/watch?v={video_id}",
    )


def ytdlp_options(settings: Settings | None = None, **overrides):
    settings = settings or get_settings()
    options = dict(overrides)
    options.setdefault("socket_timeout", settings.ytdlp_socket_timeout)
    options.setdefault("retries", settings.ytdlp_retries)
    options.setdefault("fragment_retries", settings.ytdlp_retries)
    if settings.ytdlp_proxy:
        options["proxy"] = settings.ytdlp_proxy
    if settings.ytdlp_cookies_file:
        options["cookiefile"] = settings.ytdlp_cookies_file
    elif settings.ytdlp_cookies_from_browser:
        options["cookiesfrombrowser"] = (settings.ytdlp_cookies_from_browser,)
    if settings.ytdlp_js_runtime:
        options["js_runtimes"] = {settings.ytdlp_js_runtime: {}}
    return options


def parse_youtube_api_duration(value: str | None) -> int | None:
    if not value:
        return None
    match = YOUTUBE_API_DURATION_RE.match(value)
    if not match:
        return None
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = int(match.group("seconds") or 0)
    return hours * 3600 + minutes * 60 + seconds


def _youtube_api_token_path(settings: Settings) -> Path | None:
    if not settings.youtube_api_token_file:
        return None
    return Path(settings.youtube_api_token_file)


def _fetch_youtube_api_item(parsed: ParsedYouTubeUrl, settings: Settings) -> dict | None:
    token_path = _youtube_api_token_path(settings)
    if token_path is None or not token_path.exists():
        return None
    youtube = build_youtube_client(settings, persist_refresh=True)
    response = (
        youtube.videos()
        .list(
            part="snippet,contentDetails,status",
            id=parsed.video_id,
        )
        .execute()
    )
    items = response.get("items") or []
    return items[0] if items else None


def _metadata_from_youtube_api_item(parsed: ParsedYouTubeUrl, source_url: str, item: dict) -> VideoMetadata:
    snippet = item.get("snippet") or {}
    content_details = item.get("contentDetails") or {}
    return VideoMetadata(
        video_id=parsed.video_id,
        canonical_url=parsed.canonical_url,
        source_url=source_url,
        title=snippet.get("title") or parsed.video_id,
        channel=snippet.get("channelTitle"),
        duration_seconds=parse_youtube_api_duration(content_details.get("duration")),
        description=snippet.get("description"),
    )


def fetch_video_metadata(url: str, settings: Settings | None = None) -> VideoMetadata:
    parsed = parse_youtube_url(url)
    settings = settings or get_settings()
    api_item = _fetch_youtube_api_item(parsed, settings)
    if api_item is not None:
        return _metadata_from_youtube_api_item(parsed, url, api_item)

    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("yt-dlp is required to fetch video metadata") from exc

    options = ytdlp_options(settings, quiet=True, skip_download=True, extract_flat=False)
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(parsed.canonical_url, download=False)
    except Exception as exc:
        raise_if_youtube_auth_required(exc)
        raise

    return VideoMetadata(
        video_id=parsed.video_id,
        canonical_url=parsed.canonical_url,
        source_url=url,
        title=info.get("title") or parsed.video_id,
        channel=info.get("channel") or info.get("uploader"),
        duration_seconds=info.get("duration"),
        description=info.get("description"),
    )
