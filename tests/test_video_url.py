import pytest

from yt_sub_hub.ingestion import InvalidYouTubeUrl, parse_youtube_url


def test_parse_standard_watch_url_extracts_video_id_and_ignores_playlist():
    parsed = parse_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=abc")

    assert parsed.video_id == "dQw4w9WgXcQ"
    assert parsed.canonical_url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def test_parse_short_url_extracts_video_id():
    parsed = parse_youtube_url("https://youtu.be/dQw4w9WgXcQ?t=43")

    assert parsed.video_id == "dQw4w9WgXcQ"
    assert parsed.canonical_url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def test_parse_embed_url_extracts_video_id():
    parsed = parse_youtube_url("https://www.youtube.com/embed/dQw4w9WgXcQ")

    assert parsed.video_id == "dQw4w9WgXcQ"


def test_parse_invalid_url_raises_clear_error():
    with pytest.raises(InvalidYouTubeUrl, match="Unsupported YouTube URL"):
        parse_youtube_url("https://example.com/not-youtube")
