from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Iterable, Literal, Protocol

from yt_sub_hub.config import Settings, get_settings
from yt_sub_hub.ingestion import raise_if_youtube_auth_required, ytdlp_options
from yt_sub_hub.models import Transcript, TranscriptBundle, TranscriptSegment
from yt_sub_hub.youtube_data_api import (
    YouTubeDataApiClientError,
    build_youtube_client,
    google_api_error_class,
    sanitize_google_api_message,
)
from yt_sub_hub.whisper_backends import WhisperBackend, build_whisper_backend


class TranscriptResolutionError(RuntimeError):
    """Raised when no transcript source can produce text."""


TranscriptSourceReason = Literal[
    "captions_unavailable",
    "captions_forbidden",
    "auth_error",
    "quota_or_rate_limited",
    "unavailable",
    "dependency_error",
    "unknown_error",
]


class TranscriptSourceError(TranscriptResolutionError):
    """Raised when a transcript source fails with a classified reason."""

    def __init__(self, message: str, *, reason: TranscriptSourceReason) -> None:
        self.reason = reason
        super().__init__(message)


SKIPPED_REQUIRES_MEDIA_REASON = "skipped_requires_media"
EXPECTED_MEDIA_FALLBACK_REASONS = {"captions_unavailable", "captions_forbidden", "unavailable"}


def explain_data_api_only_requires_media() -> str:
    return "requires media/Whisper fallback, but data_api_only mode forbids media download"


def explain_full_processing_media_download_disabled() -> str:
    return "full_processing requires YT_SUB_HUB_MEDIA_DOWNLOAD=enabled before Whisper fallback can run"


def explain_media_download_disabled() -> str:
    return (
        "Через YouTube Data API нет доступных captions или нет прав; "
        "media download отключен (YT_SUB_HUB_MEDIA_DOWNLOAD=disabled). "
        "yt-dlp fallback не будет запущен."
    )


class TranscriptProvider(Protocol):
    def fetch(self, video_id: str) -> Transcript: ...


class WhisperProviderProtocol(Protocol):
    def transcribe(self, video_id: str) -> Transcript | list[TranscriptSegment]: ...


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_transcript(items: Iterable[dict], source: str, video_id: str = "") -> Transcript:
    segments = []
    for item in items:
        raw_text = str(item.get("text", ""))
        segments.append(
            TranscriptSegment(
                start=float(item.get("start", 0.0)),
                duration=float(item.get("duration", 0.0)),
                text=_clean_text(raw_text),
                raw_text=raw_text,
            )
        )
    return Transcript(video_id=video_id, source=source, segments=[segment for segment in segments if segment.text])


SRT_TIMESTAMP_RE = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2}[,.]\d{1,3})"
)


def _parse_caption_timestamp(value: str) -> float:
    hours, minutes, seconds = value.replace(",", ".").split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _parse_srt_captions(content: str) -> list[dict]:
    snippets: list[dict] = []
    for block in re.split(r"\n\s*\n", content.replace("\r\n", "\n").strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        timestamp_index = next((index for index, line in enumerate(lines) if "-->" in line), None)
        if timestamp_index is None:
            continue
        match = SRT_TIMESTAMP_RE.search(lines[timestamp_index])
        if match is None:
            continue
        start = _parse_caption_timestamp(match.group("start"))
        end = _parse_caption_timestamp(match.group("end"))
        text = " ".join(lines[timestamp_index + 1 :])
        snippets.append({"start": start, "duration": max(0.0, end - start), "text": text})
    return snippets


class YouTubeDataApiCaptionProvider:
    def __init__(
        self,
        settings: Settings | None = None,
        languages: list[str] | None = None,
        youtube_service=None,
    ) -> None:
        self.settings = settings or get_settings()
        self.languages = languages or [self.settings.target_language, "ru", "en"]
        self.youtube_service = youtube_service

    def fetch(self, video_id: str) -> Transcript:
        return self.fetch_bundle(video_id)[1]

    def fetch_bundle(self, video_id: str) -> tuple[dict, Transcript]:
        try:
            youtube = self.youtube_service or self._build_service()
        except TranscriptSourceError:
            raise
        except Exception as exc:
            raise _transcript_source_error(exc, prefix="YouTube Data API client failed") from exc
        try:
            items = list_youtube_data_api_captions(video_id, settings=self.settings, youtube_service=youtube)
        except TranscriptSourceError:
            raise
        except Exception as exc:
            raise _transcript_source_error(exc, prefix="YouTube Data API captions.list failed") from exc

        item = self._select_caption(items)
        if item is None:
            raise TranscriptSourceError(
                "YouTube Data API captions are unavailable for this video",
                reason="captions_unavailable",
            )
        caption_id = item.get("id")
        language = (item.get("snippet") or {}).get("language")
        if not caption_id:
            raise TranscriptSourceError("YouTube Data API caption item is missing id", reason="unknown_error")

        try:
            download = probe_youtube_data_api_caption_download(
                video_id,
                caption_id,
                settings=self.settings,
                youtube_service=youtube,
            )
        except TranscriptSourceError:
            raise
        except Exception as exc:
            raise _transcript_source_error(exc, prefix="YouTube Data API captions.download failed") from exc

        download_error_class = download.get("download_error_class")
        if download_error_class is not None:
            raise _transcript_source_error_from_class(
                str(download_error_class),
                str(download.get("download_error_message") or "YouTube Data API captions.download failed"),
                prefix="YouTube Data API captions.download failed",
            )
        content = str(download["content"])
        snippets = _parse_srt_captions(content)
        if not snippets:
            raise TranscriptSourceError(
                "YouTube Data API caption download returned no transcript segments",
                reason="captions_unavailable",
            )
        normalized = normalize_transcript(snippets, source="youtube_data_api_captions", video_id=video_id)
        normalized.language = language
        return (
            {
                "source": "youtube_data_api_captions",
                "caption_id": caption_id,
                "language": language,
                "format": "srt",
                "items": snippets,
            },
            normalized,
        )

    def _build_service(self):
        try:
            return build_youtube_client(self.settings, persist_refresh=True)
        except TranscriptSourceError:
            raise
        except Exception as exc:
            raise _transcript_source_error(exc, prefix="YouTube Data API client failed") from exc

    def _select_caption(self, items: list[dict]) -> dict | None:
        by_language = {
            (item.get("snippet") or {}).get("language"): item
            for item in items
            if (item.get("snippet") or {}).get("language")
        }
        for language in dict.fromkeys(self.languages):
            if language in by_language:
                return by_language[language]
        return items[0] if items else None


def list_youtube_data_api_captions(
    video_id: str,
    *,
    settings: Settings | None = None,
    youtube_service=None,
) -> list[dict]:
    settings = settings or get_settings()
    youtube = youtube_service or build_youtube_client(settings, persist_refresh=True)
    response = youtube.captions().list(part="id,snippet", videoId=video_id).execute()
    return list(response.get("items") or [])


def probe_youtube_data_api_caption_download(
    video_id: str,
    caption_id: str,
    *,
    settings: Settings | None = None,
    youtube_service=None,
) -> dict:
    settings = settings or get_settings()
    youtube = youtube_service or build_youtube_client(settings, persist_refresh=True)
    try:
        payload = youtube.captions().download(id=caption_id, tfmt="srt").execute()
    except Exception as exc:
        return {
            "video_id": video_id,
            "caption_id": caption_id,
            "downloadable": False,
            "download_error_class": google_api_error_class(exc),
            "download_error_message": sanitize_google_api_message(exc),
            "content": "",
        }
    content = payload.decode("utf-8", errors="replace") if isinstance(payload, bytes) else str(payload)
    return {
        "video_id": video_id,
        "caption_id": caption_id,
        "downloadable": bool(content.strip()),
        "download_error_class": None,
        "download_error_message": None,
        "content": content,
    }


def _transcript_source_error(
    exc: BaseException,
    *,
    prefix: str,
) -> TranscriptSourceError:
    if isinstance(exc, TranscriptSourceError):
        return exc
    reason = _transcript_reason_from_exception(exc)
    message = sanitize_google_api_message(exc)
    return TranscriptSourceError(f"{prefix}: {message}", reason=reason)


def _transcript_source_error_from_class(
    error_class: str,
    message: str,
    *,
    prefix: str,
) -> TranscriptSourceError:
    reason = _transcript_reason_from_google_class(error_class)
    sanitized = sanitize_google_api_message(message)
    return TranscriptSourceError(f"{prefix}: {sanitized}", reason=reason)


def _transcript_reason_from_exception(exc: BaseException) -> TranscriptSourceReason:
    if isinstance(exc, YouTubeDataApiClientError):
        text = sanitize_google_api_message(exc).lower()
        if any(marker in text for marker in ("google-api-python-client", "google-auth", "not installed", "missing dependency")):
            return "dependency_error"
        if any(marker in text for marker in ("invalid grant", "invalid_grant", "token", "credential", "scope")):
            return "auth_error"
        if "required" in text:
            return "dependency_error"
    text = str(exc).lower()
    if isinstance(exc, ImportError) or any(marker in text for marker in ("no module named", "not installed")):
        return "dependency_error"
    if any(marker in text for marker in ("invalid grant", "invalid_grant")):
        return "auth_error"
    return _transcript_reason_from_google_class(google_api_error_class(exc))


def _transcript_reason_from_google_class(error_class: str) -> TranscriptSourceReason:
    if error_class == "forbidden":
        return "captions_forbidden"
    if error_class == "auth_error":
        return "auth_error"
    if error_class == "quota_or_rate_limited":
        return "quota_or_rate_limited"
    if error_class == "unavailable":
        return "unavailable"
    return "unknown_error"


class CompositeTranscriptProvider:
    def __init__(self, providers: list[TranscriptProvider]) -> None:
        self.providers = providers

    def fetch(self, video_id: str) -> Transcript:
        return self.fetch_bundle(video_id)[1]

    def fetch_bundle(self, video_id: str) -> tuple[object, Transcript]:
        errors: list[str] = []
        first_expected_error: TranscriptSourceError | None = None
        for provider in self.providers:
            try:
                if hasattr(provider, "fetch_bundle"):
                    return provider.fetch_bundle(video_id)  # type: ignore[attr-defined]
                transcript = provider.fetch(video_id)
                return transcript.to_dict(), transcript
            except TranscriptSourceError as exc:
                errors.append(str(exc))
                if exc.reason not in EXPECTED_MEDIA_FALLBACK_REASONS:
                    raise
                if first_expected_error is None:
                    first_expected_error = exc
            except Exception as exc:
                errors.append(str(exc))
        if first_expected_error is not None:
            raise first_expected_error
        raise TranscriptResolutionError("; ".join(error for error in errors if error) or "no transcript providers succeeded")


def build_transcript_provider(
    settings: Settings | None = None,
    *,
    allow_unofficial_provider: bool = True,
) -> TranscriptProvider:
    settings = settings or get_settings()
    providers: list[TranscriptProvider] = []
    if settings.youtube_api_token_file or not allow_unofficial_provider:
        providers.append(YouTubeDataApiCaptionProvider(settings=settings))
    if allow_unofficial_provider:
        providers.append(YouTubeTranscriptProvider())
    return CompositeTranscriptProvider(providers)


class YouTubeTranscriptProvider:
    def __init__(self, languages: list[str] | None = None) -> None:
        self.languages = languages or ["en", "ru"]

    def fetch(self, video_id: str) -> Transcript:
        raw, normalized = self.fetch_bundle(video_id)
        return normalized

    def fetch_bundle(self, video_id: str) -> tuple[list[dict], Transcript]:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError as exc:
            raise TranscriptResolutionError("youtube-transcript-api is not installed") from exc

        try:
            api = YouTubeTranscriptApi()
            if hasattr(api, "fetch"):
                fetched = api.fetch(video_id, languages=self.languages)
                snippets = [
                    {"start": snippet.start, "duration": snippet.duration, "text": snippet.text}
                    for snippet in fetched
                ]
            else:
                snippets = YouTubeTranscriptApi.get_transcript(video_id, languages=self.languages)
        except Exception as exc:  # external library exposes several exception classes
            raise TranscriptResolutionError(str(exc)) from exc

        return snippets, normalize_transcript(snippets, source="youtube", video_id=video_id)


class WhisperAudioProvider:
    def __init__(
        self,
        canonical_url: str,
        audio_dir: Path,
        model_name: str | None = None,
        keep_audio: bool = True,
        settings: Settings | None = None,
        backend: WhisperBackend | None = None,
    ) -> None:
        self.canonical_url = canonical_url
        self.audio_dir = audio_dir
        self.model_name = model_name
        self.keep_audio = keep_audio
        self.settings = settings
        self.backend = backend

    def transcribe(self, video_id: str) -> Transcript:
        return self.transcribe_bundle(video_id).normalized

    def download_audio(self, video_id: str) -> Path:
        return self._download_audio(video_id)

    def transcribe_bundle(self, video_id: str) -> TranscriptBundle:
        audio_path = self._download_audio(video_id)
        try:
            backend = self.backend or build_whisper_backend(self.settings or get_settings(), self.model_name)
            return backend.transcribe(audio_path, video_id=video_id)
        except Exception as exc:
            raise TranscriptResolutionError(f"Whisper backend failed: {exc}") from exc
        finally:
            if not self.keep_audio:
                audio_path.unlink(missing_ok=True)

    def _download_audio(self, video_id: str) -> Path:
        settings = self.settings or get_settings()
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = self.audio_dir / f"{video_id}.mp3"
        if audio_path.exists() and audio_path.stat().st_size > 0:
            return audio_path
        if not settings.media_download_enabled:
            raise TranscriptResolutionError(explain_media_download_disabled())
        if shutil.which("ffmpeg") is None:
            raise TranscriptResolutionError("ffmpeg missing: install ffmpeg before using Whisper fallback")
        try:
            import yt_dlp
        except ImportError as exc:
            raise TranscriptResolutionError("yt-dlp is required for Whisper fallback") from exc

        output_template = str(self.audio_dir / f"{video_id}.%(ext)s")
        options = ytdlp_options(
            settings,
            format="bestaudio/best",
            outtmpl=output_template,
            quiet=True,
            postprocessors=[{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
        )
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.download([self.canonical_url])
        except Exception as exc:
            raise_if_youtube_auth_required(exc)
            raise
        if not audio_path.exists():
            raise TranscriptResolutionError(f"audio download failed: expected {audio_path}")
        return audio_path


def resolve_transcript(
    video_id: str,
    transcript_provider: TranscriptProvider | None = None,
    whisper_provider: WhisperProviderProtocol | None = None,
) -> Transcript:
    return resolve_transcript_bundle(video_id, transcript_provider, whisper_provider).normalized


def resolve_transcript_bundle(
    video_id: str,
    transcript_provider: TranscriptProvider | None = None,
    whisper_provider: WhisperProviderProtocol | None = None,
) -> TranscriptBundle:
    transcript_provider = transcript_provider or YouTubeTranscriptProvider()
    try:
        if hasattr(transcript_provider, "fetch_bundle"):
            raw_payload, normalized = transcript_provider.fetch_bundle(video_id)  # type: ignore[attr-defined]
        else:
            normalized = transcript_provider.fetch(video_id)
            raw_payload = normalized.to_dict()
        normalized.video_id = video_id
        return TranscriptBundle(video_id=video_id, source=normalized.source, raw_payload=raw_payload, normalized=normalized)
    except TranscriptResolutionError as transcript_error:
        if whisper_provider is None:
            raise transcript_error

    try:
        if hasattr(whisper_provider, "transcribe_bundle"):
            bundle = whisper_provider.transcribe_bundle(video_id)  # type: ignore[attr-defined]
            bundle.normalized.video_id = video_id
            return bundle
        fallback = whisper_provider.transcribe(video_id)
    except Exception as exc:
        raise TranscriptResolutionError(f"Whisper fallback failed: {exc}") from exc

    if isinstance(fallback, Transcript):
        fallback.video_id = video_id
        fallback.source = fallback.source or "whisper"
        fallback.segments = [
            TranscriptSegment(segment.start, segment.duration, _clean_text(segment.text), segment.raw_text or segment.text)
            for segment in fallback.segments
            if _clean_text(segment.text)
        ]
        return TranscriptBundle(video_id=video_id, source="whisper", raw_payload=fallback.to_dict(), normalized=fallback)

    normalized = Transcript(
        video_id=video_id,
        source="whisper",
        segments=[
            TranscriptSegment(segment.start, segment.duration, _clean_text(segment.text), segment.raw_text or segment.text)
            for segment in fallback
            if _clean_text(segment.text)
        ],
    )
    return TranscriptBundle(video_id=video_id, source="whisper", raw_payload=normalized.to_dict(), normalized=normalized)
