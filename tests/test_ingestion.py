from pathlib import Path

from yt_sub_hub.config import Settings
import pytest

from yt_sub_hub.ingestion import YouTubeAuthRequiredError, fetch_video_metadata, ytdlp_options


def test_ytdlp_options_include_browser_cookies_and_js_runtime():
    settings = Settings(
        library_dir=Path("library"),
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
        max_prompt_chars=24000,
        ytdlp_cookies_from_browser="chrome",
        ytdlp_cookies_file=None,
        ytdlp_proxy=None,
        ytdlp_js_runtime="node",
        ytdlp_socket_timeout=90,
        ytdlp_retries=12,
    )

    options = ytdlp_options(settings, quiet=True)

    assert options["cookiesfrombrowser"] == ("chrome",)
    assert options["js_runtimes"] == {"node": {}}
    assert options["socket_timeout"] == 90
    assert options["retries"] == 12
    assert options["fragment_retries"] == 12
    assert options["quiet"] is True


def test_ytdlp_options_use_cookie_file_without_browser_keychain():
    settings = Settings(
        library_dir=Path("library"),
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
        max_prompt_chars=24000,
        ytdlp_cookies_from_browser="chrome",
        ytdlp_cookies_file="/secure/youtube-cookies.txt",
        ytdlp_proxy="http://127.0.0.1:12334",
        ytdlp_js_runtime=None,
        ytdlp_socket_timeout=90,
        ytdlp_retries=12,
    )

    options = ytdlp_options(settings)

    assert options["cookiefile"] == "/secure/youtube-cookies.txt"
    assert options["proxy"] == "http://127.0.0.1:12334"
    assert "cookiesfrombrowser" not in options


def test_ytdlp_options_include_proxy_when_configured():
    settings = Settings(
        library_dir=Path("library"),
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
        max_prompt_chars=24000,
        ytdlp_cookies_from_browser=None,
        ytdlp_cookies_file=None,
        ytdlp_proxy="http://127.0.0.1:12334",
        ytdlp_js_runtime=None,
        ytdlp_socket_timeout=60,
        ytdlp_retries=10,
    )

    options = ytdlp_options(settings)

    assert options["proxy"] == "http://127.0.0.1:12334"


def test_ytdlp_options_do_not_use_browser_cookies_by_default():
    settings = Settings(
        library_dir=Path("library"),
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
        max_prompt_chars=24000,
        ytdlp_cookies_from_browser=None,
        ytdlp_cookies_file=None,
        ytdlp_proxy=None,
        ytdlp_js_runtime=None,
        ytdlp_socket_timeout=60,
        ytdlp_retries=10,
    )

    options = ytdlp_options(settings)

    assert "cookiesfrombrowser" not in options
    assert "cookiefile" not in options


def test_fetch_video_metadata_reports_bot_check_without_browser_cookies(monkeypatch):
    class FakeYoutubeDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, url, download=False):
            raise RuntimeError("Sign in to confirm you’re not a bot. Use --cookies-from-browser or --cookies")

    import yt_dlp

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYoutubeDL)

    expected = (
        "Видео требует авторизованные cookies. Включите cookies явно через "
        "YT_SUB_HUB_YTDLP_COOKIES_FILE=/secure/path/youtube-cookies.txt. "
        "Browser cookies are disabled by default."
    )
    settings = Settings(
        library_dir=Path("library"),
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
        max_prompt_chars=24000,
        ytdlp_cookies_from_browser=None,
        ytdlp_cookies_file=None,
        ytdlp_proxy=None,
        ytdlp_js_runtime=None,
        ytdlp_socket_timeout=60,
        ytdlp_retries=10,
        youtube_api_token_file=None,
    )
    with pytest.raises(YouTubeAuthRequiredError, match=expected):
        fetch_video_metadata("https://www.youtube.com/watch?v=abc123def45", settings=settings)


def test_fetch_video_metadata_uses_youtube_api_token_before_ytdlp(tmp_path, monkeypatch):
    token = tmp_path / "token.json"
    token.write_text("{}", encoding="utf-8")
    settings = Settings(
        library_dir=Path("library"),
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
        max_prompt_chars=24000,
        ytdlp_cookies_from_browser=None,
        ytdlp_cookies_file=None,
        ytdlp_proxy=None,
        ytdlp_js_runtime=None,
        ytdlp_socket_timeout=60,
        ytdlp_retries=10,
        youtube_api_token_file=str(token),
    )

    def fake_youtube_api(parsed, settings):
        return {
            "id": parsed.video_id,
            "snippet": {
                "title": "API title",
                "channelTitle": "API channel",
                "description": "API description",
            },
            "contentDetails": {"duration": "PT18M27S"},
        }

    class FailingYoutubeDL:
        def __init__(self, options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, url, download=False):
            raise AssertionError("yt-dlp should not be used when token.json is available")

    import yt_dlp

    monkeypatch.setattr("yt_sub_hub.ingestion._fetch_youtube_api_item", fake_youtube_api)
    monkeypatch.setattr(yt_dlp, "YoutubeDL", FailingYoutubeDL)

    metadata = fetch_video_metadata("https://www.youtube.com/watch?v=UUyelVhs2rI", settings=settings)

    assert metadata.video_id == "UUyelVhs2rI"
    assert metadata.title == "API title"
    assert metadata.channel == "API channel"
    assert metadata.description == "API description"
    assert metadata.duration_seconds == 1107


def test_fetch_video_metadata_falls_back_to_ytdlp_when_youtube_api_token_missing(monkeypatch):
    settings = Settings(
        library_dir=Path("library"),
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
        max_prompt_chars=24000,
        ytdlp_cookies_from_browser=None,
        ytdlp_cookies_file=None,
        ytdlp_proxy=None,
        ytdlp_js_runtime=None,
        ytdlp_socket_timeout=60,
        ytdlp_retries=10,
        youtube_api_token_file="/missing/token.json",
    )

    class FakeYoutubeDL:
        def __init__(self, options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, url, download=False):
            return {
                "title": "yt-dlp title",
                "channel": "yt-dlp channel",
                "duration": 42,
                "description": "yt-dlp description",
            }

    import yt_dlp

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYoutubeDL)

    metadata = fetch_video_metadata("https://www.youtube.com/watch?v=abc123def45", settings=settings)

    assert metadata.title == "yt-dlp title"
    assert metadata.channel == "yt-dlp channel"
    assert metadata.duration_seconds == 42


@pytest.mark.parametrize(
    ("duration", "seconds"),
    [
        ("PT18M27S", 1107),
        ("PT1H2M3S", 3723),
        ("PT45S", 45),
        ("PT2M", 120),
    ],
)
def test_parse_youtube_api_duration(duration, seconds):
    from yt_sub_hub.ingestion import parse_youtube_api_duration

    assert parse_youtube_api_duration(duration) == seconds
