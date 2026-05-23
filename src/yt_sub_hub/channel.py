from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from yt_sub_hub.config import Settings, get_settings
from yt_sub_hub.library import LibraryStore
from yt_sub_hub.models import slugify
from yt_sub_hub.runtime import ChannelJobRecord, RuntimeStore, VideoJobRecord
from yt_sub_hub.youtube_data_api import YouTubeDataApiClientError, build_youtube_client


@dataclass(frozen=True)
class ChannelVideo:
    video_id: str
    canonical_url: str
    title: str | None = None
    published_at: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


@dataclass(frozen=True)
class ChannelFetchResult:
    channel_id: str
    channel_url: str
    uploads_playlist_id: str
    title: str | None
    videos: list[ChannelVideo]


class ChannelDataApiError(RuntimeError):
    pass


def channel_id_from_url(channel_url: str) -> str:
    parsed = urlparse(channel_url.strip())
    if not parsed.scheme and not parsed.netloc:
        return channel_url.strip()
    path_parts = [part for part in parsed.path.split("/") if part]
    if not path_parts:
        return slugify(parsed.netloc)
    first = path_parts[0]
    if first.startswith("@"):
        return slugify(first[1:])
    if first.lower() in {"channel", "c", "user"} and len(path_parts) > 1:
        return slugify(path_parts[1])
    return slugify(first)


def fetch_channel_videos(
    channel_url: str,
    *,
    limit: int | None = None,
    settings: Settings | None = None,
    youtube_service=None,
) -> list[ChannelVideo]:
    return fetch_channel_listing(channel_url, limit=limit, settings=settings, youtube_service=youtube_service).videos


def fetch_channel_listing(
    channel_url_or_id: str,
    *,
    limit: int | None = None,
    settings: Settings | None = None,
    youtube_service=None,
) -> ChannelFetchResult:
    settings = settings or get_settings()
    youtube = youtube_service or _build_youtube_client(settings)
    channel = _resolve_channel(youtube, channel_url_or_id)
    videos = _fetch_upload_playlist_videos(
        youtube,
        channel["uploads_playlist_id"],
        limit=limit,
    )
    return ChannelFetchResult(
        channel_id=channel["channel_id"],
        channel_url=f"https://www.youtube.com/channel/{channel['channel_id']}",
        uploads_playlist_id=channel["uploads_playlist_id"],
        title=channel.get("title"),
        videos=videos,
    )


def enqueue_channel_videos(
    runtime: RuntimeStore,
    library: LibraryStore,
    *,
    channel_url: str,
    videos: list[ChannelVideo],
    channel_id: str | None = None,
    uploads_playlist_id: str | None = None,
    limit: int | None = None,
    quality_mode: str = "strict",
    acceptance_mode: str = "data_api_only",
    force: bool = False,
) -> tuple[ChannelJobRecord, list[VideoJobRecord]]:
    channel_id = channel_id or channel_id_from_url(channel_url)
    selected = videos[:limit] if limit is not None else videos
    channel_job = runtime.create_channel_job(
        channel_url=channel_url,
        channel_id=channel_id,
        requested_limit=limit,
        quality_mode=quality_mode,
        acceptance_mode=acceptance_mode,
    )
    video_jobs: list[VideoJobRecord] = []
    for video in selected:
        video_job = runtime.create_video_job(
            video_id=video.video_id,
            canonical_url=video.canonical_url,
            channel_job_id=channel_job.id,
            quality_mode=quality_mode,
            acceptance_mode=acceptance_mode,
        )
        if force and video_job.status in {"failed", "skipped"}:
            video_job = runtime.retry_video_job_for_channel(video_job.id)
            video_job = runtime.update_video_job_quality_mode(video_job.id, quality_mode)
            video_job = runtime.update_video_job_acceptance_mode(video_job.id, acceptance_mode)
        elif video_job.status not in {"completed", "completed_with_warning", "failed", "skipped"}:
            video_job = runtime.update_video_job_quality_mode(video_job.id, quality_mode)
            video_job = runtime.update_video_job_acceptance_mode(video_job.id, acceptance_mode)
        runtime.ensure_stage_plan(video_job.id)
        video_jobs.append(video_job)
    write_channel_artifacts(
        library.root,
        channel_id=channel_id,
        channel_url=channel_url,
        uploads_playlist_id=uploads_playlist_id,
        videos=selected,
    )
    return runtime.recompute_channel_job_status(channel_job.id), video_jobs


def write_channel_artifacts(
    library_root: Path | str,
    *,
    channel_id: str,
    channel_url: str,
    uploads_playlist_id: str | None = None,
    videos: list[ChannelVideo],
) -> Path:
    channel_dir = Path(library_root) / "channels" / channel_id
    channel_dir.mkdir(parents=True, exist_ok=True)
    videos_path = channel_dir / "videos.json"
    videos_path.write_text(
        json.dumps(
            {
                "channel_id": channel_id,
                "channel_url": channel_url,
                "uploads_playlist_id": uploads_playlist_id,
                "videos": [video.to_dict() for video in videos],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (channel_dir / "channel_index.md").write_text(
        "\n".join(
            [
                f"# {channel_id}",
                "",
                *[
                    f"- [{video.title or video.video_id}]({video.canonical_url})"
                    for video in videos
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )
    return videos_path


def _build_youtube_client(settings: Settings):
    try:
        return build_youtube_client(settings, persist_refresh=True)
    except YouTubeDataApiClientError as exc:
        raise ChannelDataApiError(str(exc)) from exc


def _resolve_channel(youtube, channel_url_or_id: str) -> dict[str, str | None]:
    parsed = _parse_channel_reference(channel_url_or_id)
    if parsed["uploads_playlist_id"]:
        playlist_id = str(parsed["uploads_playlist_id"])
        channel_id = _channel_id_from_uploads_playlist(playlist_id)
        return {
            "channel_id": channel_id,
            "uploads_playlist_id": playlist_id,
            "title": None,
        }

    response = None
    if parsed["channel_id"]:
        response = _execute_data_api(
            youtube.channels().list(part="snippet,contentDetails", id=parsed["channel_id"]),
            "channels.list",
        )
    else:
        handle = parsed["handle"] or parsed["username"]
        if handle:
            response = _execute_data_api(
                youtube.channels().list(part="snippet,contentDetails", forHandle=handle),
                "channels.list",
            )
            if not (response.get("items") or []) and parsed["username"]:
                response = _execute_data_api(
                    youtube.channels().list(part="snippet,contentDetails", forUsername=parsed["username"]),
                    "channels.list",
                )
    items = (response or {}).get("items") or []
    if not items:
        raise ChannelDataApiError(f"YouTube Data API could not resolve channel: {channel_url_or_id}")
    item = items[0]
    uploads = ((item.get("contentDetails") or {}).get("relatedPlaylists") or {}).get("uploads")
    if not uploads:
        raise ChannelDataApiError(f"YouTube Data API channel is missing uploads playlist: {channel_url_or_id}")
    return {
        "channel_id": str(item["id"]),
        "uploads_playlist_id": str(uploads),
        "title": (item.get("snippet") or {}).get("title"),
    }


def _parse_channel_reference(value: str) -> dict[str, str | None]:
    stripped = value.strip()
    parsed = urlparse(stripped)
    query = parse_qs(parsed.query)
    playlist_id = (query.get("list") or [None])[0]
    if playlist_id and (playlist_id.startswith("UU") or playlist_id.startswith("PL")):
        return {"channel_id": None, "uploads_playlist_id": playlist_id, "handle": None, "username": None}
    if not parsed.scheme and not parsed.netloc:
        if stripped.startswith("UC"):
            return {"channel_id": stripped, "uploads_playlist_id": None, "handle": None, "username": None}
        if stripped.startswith("UU"):
            return {"channel_id": None, "uploads_playlist_id": stripped, "handle": None, "username": None}
        if stripped.startswith("@"):
            return {"channel_id": None, "uploads_playlist_id": None, "handle": stripped[1:], "username": None}
        return {"channel_id": None, "uploads_playlist_id": None, "handle": stripped, "username": stripped}
    parts = [part for part in parsed.path.split("/") if part]
    if not parts:
        raise ChannelDataApiError(f"Unsupported YouTube channel URL: {value}")
    first = parts[0]
    if first == "playlist" and playlist_id:
        return {"channel_id": None, "uploads_playlist_id": playlist_id, "handle": None, "username": None}
    if first.startswith("@"):
        return {"channel_id": None, "uploads_playlist_id": None, "handle": first[1:], "username": None}
    if first.lower() == "channel" and len(parts) > 1:
        return {"channel_id": parts[1], "uploads_playlist_id": None, "handle": None, "username": None}
    if first.lower() in {"c", "user"} and len(parts) > 1:
        return {"channel_id": None, "uploads_playlist_id": None, "handle": parts[1], "username": parts[1]}
    raise ChannelDataApiError(f"Unsupported YouTube channel URL: {value}")


def _channel_id_from_uploads_playlist(playlist_id: str) -> str:
    if playlist_id.startswith("UU") and len(playlist_id) > 2:
        return "UC" + playlist_id[2:]
    return playlist_id


def _fetch_upload_playlist_videos(youtube, playlist_id: str, *, limit: int | None) -> list[ChannelVideo]:
    videos: list[ChannelVideo] = []
    page_token = None
    while limit is None or len(videos) < limit:
        max_results = min(50, (limit - len(videos)) if limit is not None else 50)
        response = _execute_data_api(
            youtube.playlistItems().list(
                part="snippet,contentDetails",
                playlistId=playlist_id,
                maxResults=max_results,
                pageToken=page_token,
            ),
            "playlistItems.list",
        )
        for item in response.get("items") or []:
            snippet = item.get("snippet") or {}
            resource = snippet.get("resourceId") or {}
            video_id = resource.get("videoId") or (item.get("contentDetails") or {}).get("videoId")
            if not video_id:
                continue
            videos.append(
                ChannelVideo(
                    video_id=str(video_id),
                    canonical_url=f"https://www.youtube.com/watch?v={video_id}",
                    title=snippet.get("title"),
                    published_at=snippet.get("publishedAt"),
                )
            )
            if limit is not None and len(videos) >= limit:
                break
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return videos


def _execute_data_api(request, action: str) -> dict:
    try:
        return request.execute()
    except Exception as exc:
        raise ChannelDataApiError(f"YouTube Data API {action} failed: {exc}") from exc
