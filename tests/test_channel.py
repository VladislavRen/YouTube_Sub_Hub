import json
from pathlib import Path

import pytest

from yt_sub_hub.channel import ChannelFetchResult, ChannelVideo, fetch_channel_listing, enqueue_channel_videos
from yt_sub_hub.config import Settings
from yt_sub_hub.library import LibraryStore
from yt_sub_hub.runtime import RuntimeStore


def _settings(tmp_path: Path, token: Path | None = None) -> Settings:
    return Settings(
        library_dir=tmp_path,
        lm_studio_base_url="http://localhost:1234/v1",
        lm_studio_api_key="lm-studio",
        qwen_model="qwen",
        gemma_model="gemma",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        keep_audio=True,
        enable_graph=True,
        graph_best_effort=True,
        lm_studio_timeout_seconds=120,
        max_prompt_chars=3500,
        ytdlp_cookies_from_browser="chrome",
        ytdlp_cookies_file=None,
        ytdlp_proxy=None,
        ytdlp_js_runtime=None,
        ytdlp_socket_timeout=60,
        ytdlp_retries=10,
        youtube_api_token_file=str(token) if token else None,
    )


class _FakeRequest:
    def __init__(self, payload):
        self.payload = payload

    def execute(self):
        return self.payload


class _FakeYouTube:
    def __init__(self):
        self.calls = []

    def channels(self):
        return self

    def playlistItems(self):
        return self

    def list(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("forHandle") == "ExampleLab" or kwargs.get("id") == "UCexample123":
            return _FakeRequest(
                {
                    "items": [
                        {
                            "id": "UCexample123",
                            "snippet": {"title": "Example Lab"},
                            "contentDetails": {"relatedPlaylists": {"uploads": "UUexample123"}},
                        }
                    ]
                }
            )
        if kwargs.get("playlistId") == "UUexample123":
            return _FakeRequest(
                {
                    "items": [
                        {
                            "snippet": {
                                "title": "First",
                                "resourceId": {"videoId": "abc123def45"},
                                "publishedAt": "2026-01-01T00:00:00Z",
                            }
                        },
                        {
                            "snippet": {
                                "title": "Second",
                                "resourceId": {"videoId": "xyz123def45"},
                                "publishedAt": "2026-01-02T00:00:00Z",
                            }
                        },
                    ]
                }
            )
        raise AssertionError(f"unexpected API call: {kwargs}")


def test_fetch_channel_listing_uses_youtube_data_api_for_handle(tmp_path, monkeypatch):
    token = tmp_path / "token.json"
    token.write_text("{}", encoding="utf-8")
    fake = _FakeYouTube()
    monkeypatch.setattr("yt_sub_hub.channel._build_youtube_client", lambda settings: fake)

    listing = fetch_channel_listing("https://www.youtube.com/@ExampleLab", limit=2, settings=_settings(tmp_path, token))

    assert listing.channel_id == "UCexample123"
    assert listing.uploads_playlist_id == "UUexample123"
    assert [video.video_id for video in listing.videos] == ["abc123def45", "xyz123def45"]
    assert not any("extract_flat" in str(call) for call in fake.calls)


def test_fetch_channel_listing_fails_without_data_api_token(tmp_path):
    with pytest.raises(RuntimeError, match="YouTube Data API token file is required"):
        fetch_channel_listing("https://www.youtube.com/@ExampleLab", limit=2, settings=_settings(tmp_path))


def test_enqueue_channel_is_idempotent_and_creates_stage_plans(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    library = LibraryStore(tmp_path)
    listing = ChannelFetchResult(
        channel_id="UCexample123",
        channel_url="https://www.youtube.com/channel/UCexample123",
        uploads_playlist_id="UUexample123",
        title="Example Lab",
        videos=[
            ChannelVideo("abc123def45", "https://www.youtube.com/watch?v=abc123def45", "First"),
            ChannelVideo("xyz123def45", "https://www.youtube.com/watch?v=xyz123def45", "Second"),
        ],
    )

    first_job, first_videos = enqueue_channel_videos(
        runtime,
        library,
        channel_url="https://www.youtube.com/@ExampleLab",
        videos=listing.videos,
        channel_id=listing.channel_id,
        uploads_playlist_id=listing.uploads_playlist_id,
        limit=1,
        quality_mode="draft",
    )
    second_job, second_videos = enqueue_channel_videos(
        runtime,
        library,
        channel_url="https://www.youtube.com/channel/UCexample123",
        videos=listing.videos,
        channel_id=listing.channel_id,
        uploads_playlist_id=listing.uploads_playlist_id,
        limit=2,
        quality_mode="draft",
    )

    assert second_job.id == first_job.id
    assert len(runtime.list_video_jobs(first_job.id)) == 2
    assert len(first_videos) == 1
    assert len(second_videos) == 2
    assert {job.quality_mode for job in runtime.list_video_jobs(first_job.id)} == {"draft"}
    assert {job.acceptance_mode for job in runtime.list_video_jobs(first_job.id)} == {"data_api_only"}
    assert all(runtime.list_stage_plan(job.id) for job in runtime.list_video_jobs(first_job.id))
    payload = json.loads((tmp_path / "channels" / "UCexample123" / "videos.json").read_text(encoding="utf-8"))
    assert payload["channel_id"] == "UCexample123"
    assert len(payload["videos"]) == 2


def test_reenqueue_channel_updates_non_terminal_child_quality_mode(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    library = LibraryStore(tmp_path)
    videos = [ChannelVideo("abc123def45", "https://www.youtube.com/watch?v=abc123def45", "First")]

    channel_job, [first_video] = enqueue_channel_videos(
        runtime,
        library,
        channel_url="https://www.youtube.com/@ExampleLab",
        videos=videos,
        channel_id="UCexample123",
        quality_mode="draft",
    )
    _, [second_video] = enqueue_channel_videos(
        runtime,
        library,
        channel_url="https://www.youtube.com/@ExampleLab",
        videos=videos,
        channel_id="UCexample123",
        quality_mode="strict",
    )

    assert first_video.id == second_video.id
    assert runtime.load_channel_job(channel_job.id).quality_mode == "strict"
    assert runtime.load_video_job(first_video.id).quality_mode == "strict"
    assert second_video.quality_mode == "strict"


def test_reenqueue_channel_updates_non_terminal_child_acceptance_mode(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    library = LibraryStore(tmp_path)
    videos = [ChannelVideo("abc123def45", "https://www.youtube.com/watch?v=abc123def45", "First")]

    channel_job, [first_video] = enqueue_channel_videos(
        runtime,
        library,
        channel_url="https://www.youtube.com/@ExampleLab",
        videos=videos,
        channel_id="UCexample123",
        acceptance_mode="data_api_only",
    )
    _, [second_video] = enqueue_channel_videos(
        runtime,
        library,
        channel_url="https://www.youtube.com/@ExampleLab",
        videos=videos,
        channel_id="UCexample123",
        acceptance_mode="full_processing",
    )

    assert first_video.id == second_video.id
    assert runtime.load_channel_job(channel_job.id).acceptance_mode == "full_processing"
    assert runtime.load_video_job(first_video.id).acceptance_mode == "full_processing"
    assert second_video.acceptance_mode == "full_processing"


def test_reenqueue_channel_preserves_completed_child_quality_mode_without_force(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    library = LibraryStore(tmp_path)
    videos = [
        ChannelVideo("abc123def45", "https://www.youtube.com/watch?v=abc123def45", "First"),
        ChannelVideo("xyz123def45", "https://www.youtube.com/watch?v=xyz123def45", "Second"),
    ]
    channel_job, [completed_video] = enqueue_channel_videos(
        runtime,
        library,
        channel_url="https://www.youtube.com/@ExampleLab",
        videos=videos,
        channel_id="UCexample123",
        limit=1,
        quality_mode="draft",
    )
    runtime.complete_video_job(completed_video.id)

    _, second_videos = enqueue_channel_videos(
        runtime,
        library,
        channel_url="https://www.youtube.com/@ExampleLab",
        videos=videos,
        channel_id="UCexample123",
        limit=2,
        quality_mode="strict",
    )

    stored_jobs = {job.video_id: job for job in runtime.list_video_jobs(channel_job.id)}
    assert runtime.load_channel_job(channel_job.id).quality_mode == "strict"
    assert stored_jobs["abc123def45"].status == "completed"
    assert stored_jobs["abc123def45"].quality_mode == "draft"
    assert stored_jobs["xyz123def45"].status == "queued"
    assert stored_jobs["xyz123def45"].quality_mode == "strict"
    assert {job.quality_mode for job in second_videos} == {"draft", "strict"}


def test_reenqueue_channel_preserves_completed_child_acceptance_mode_without_force(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    library = LibraryStore(tmp_path)
    videos = [
        ChannelVideo("abc123def45", "https://www.youtube.com/watch?v=abc123def45", "First"),
        ChannelVideo("xyz123def45", "https://www.youtube.com/watch?v=xyz123def45", "Second"),
    ]
    channel_job, [completed_video] = enqueue_channel_videos(
        runtime,
        library,
        channel_url="https://www.youtube.com/@ExampleLab",
        videos=videos,
        channel_id="UCexample123",
        limit=1,
        acceptance_mode="data_api_only",
    )
    runtime.complete_video_job(completed_video.id)

    _, second_videos = enqueue_channel_videos(
        runtime,
        library,
        channel_url="https://www.youtube.com/@ExampleLab",
        videos=videos,
        channel_id="UCexample123",
        limit=2,
        acceptance_mode="full_processing",
    )

    stored_jobs = {job.video_id: job for job in runtime.list_video_jobs(channel_job.id)}
    assert runtime.load_channel_job(channel_job.id).acceptance_mode == "full_processing"
    assert stored_jobs["abc123def45"].status == "completed"
    assert stored_jobs["abc123def45"].acceptance_mode == "data_api_only"
    assert stored_jobs["xyz123def45"].status == "queued"
    assert stored_jobs["xyz123def45"].acceptance_mode == "full_processing"
    assert {job.acceptance_mode for job in second_videos} == {"data_api_only", "full_processing"}


def test_reenqueue_channel_with_force_requeues_failed_child_with_new_quality_mode(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    library = LibraryStore(tmp_path)
    videos = [ChannelVideo("abc123def45", "https://www.youtube.com/watch?v=abc123def45", "First")]
    channel_job, [failed_video] = enqueue_channel_videos(
        runtime,
        library,
        channel_url="https://www.youtube.com/@ExampleLab",
        videos=videos,
        channel_id="UCexample123",
        quality_mode="draft",
    )
    runtime.fail_video_job(failed_video.id, error="boom")

    _, [retried_video] = enqueue_channel_videos(
        runtime,
        library,
        channel_url="https://www.youtube.com/@ExampleLab",
        videos=videos,
        channel_id="UCexample123",
        quality_mode="strict",
        force=True,
    )

    stored = runtime.load_video_job(failed_video.id)
    assert runtime.load_channel_job(channel_job.id).quality_mode == "strict"
    assert stored.status == "queued"
    assert stored.quality_mode == "strict"
    assert retried_video.status == "queued"
    assert retried_video.quality_mode == "strict"


def test_reenqueue_channel_with_force_requeues_skipped_child_with_new_acceptance_mode(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    library = LibraryStore(tmp_path)
    videos = [ChannelVideo("abc123def45", "https://www.youtube.com/watch?v=abc123def45", "First")]
    channel_job, [skipped_video] = enqueue_channel_videos(
        runtime,
        library,
        channel_url="https://www.youtube.com/@ExampleLab",
        videos=videos,
        channel_id="UCexample123",
        acceptance_mode="data_api_only",
    )
    runtime.mark_video_job_skipped_requires_media(skipped_video.id)

    _, [retried_video] = enqueue_channel_videos(
        runtime,
        library,
        channel_url="https://www.youtube.com/@ExampleLab",
        videos=videos,
        channel_id="UCexample123",
        acceptance_mode="full_processing",
        force=True,
    )

    stored = runtime.load_video_job(skipped_video.id)
    assert runtime.load_channel_job(channel_job.id).acceptance_mode == "full_processing"
    assert stored.status == "queued"
    assert stored.acceptance_mode == "full_processing"
    assert retried_video.status == "queued"
    assert retried_video.acceptance_mode == "full_processing"
