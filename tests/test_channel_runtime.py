import json

from yt_sub_hub.channel import ChannelVideo, channel_id_from_url, enqueue_channel_videos
from yt_sub_hub.library import LibraryStore
from yt_sub_hub.runtime import RuntimeStore


def test_channel_id_from_url_supports_handle_channel_and_playlist_like_urls():
    assert channel_id_from_url("https://www.youtube.com/@Example-Lab") == "example-lab"
    assert channel_id_from_url("https://www.youtube.com/channel/UCabcDEF123") == "ucabcdef123"
    assert channel_id_from_url("https://www.youtube.com/c/ScienceHub") == "sciencehub"


def test_enqueue_channel_videos_creates_channel_artifacts_and_video_jobs(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    library = LibraryStore(tmp_path)
    videos = [
        ChannelVideo(
            video_id="abc123def45",
            canonical_url="https://www.youtube.com/watch?v=abc123def45",
            title="First",
        ),
        ChannelVideo(
            video_id="xyz123def45",
            canonical_url="https://www.youtube.com/watch?v=xyz123def45",
            title="Second",
        ),
    ]

    channel_job, video_jobs = enqueue_channel_videos(
        runtime,
        library,
        channel_url="https://www.youtube.com/@Example-Lab",
        videos=videos,
        limit=1,
    )

    videos_json = tmp_path / "channels" / "example-lab" / "videos.json"
    assert channel_job.channel_id == "example-lab"
    assert len(video_jobs) == 1
    assert video_jobs[0].video_id == "abc123def45"
    assert videos_json.exists()
    assert json.loads(videos_json.read_text(encoding="utf-8"))["videos"][0]["title"] == "First"
