from pathlib import Path

from dataclasses import replace

from yt_sub_hub.config import Settings, get_settings
from yt_sub_hub.models import Transcript, TranscriptBundle, TranscriptSegment
from yt_sub_hub.transcripts import (
    TranscriptResolutionError,
    WhisperAudioProvider,
    YouTubeDataApiCaptionProvider,
    YouTubeTranscriptProvider,
    build_transcript_provider,
    normalize_transcript,
    resolve_transcript,
    resolve_transcript_bundle,
)


class EmptyTranscriptProvider:
    def fetch(self, video_id: str):
        raise TranscriptResolutionError("no captions")


class FakeGoogleError(Exception):
    def __init__(self, status, reason, message):
        self.resp = type("Resp", (), {"status": status, "reason": reason})()
        self.content = (
            '{"error":{"message":"'
            + message
            + '","errors":[{"reason":"'
            + reason
            + '"}]}}'
        ).encode()
        super().__init__(message)


class FakeRequest:
    def __init__(self, payload=None, exc=None):
        self.payload = payload
        self.exc = exc

    def execute(self):
        if self.exc is not None:
            raise self.exc
        return self.payload


class FakeCaptionsResource:
    def __init__(self, *, items=None, download_payload=b"", list_error=None, download_error=None):
        self.items = items or []
        self.download_payload = download_payload
        self.list_error = list_error
        self.download_error = download_error

    def list(self, **kwargs):
        if self.list_error is not None:
            return FakeRequest(exc=self.list_error)
        return FakeRequest({"items": self.items})

    def download(self, **kwargs):
        if self.download_error is not None:
            return FakeRequest(exc=self.download_error)
        return FakeRequest(self.download_payload)


class FakeYoutube:
    def __init__(self, captions):
        self._captions = captions

    def captions(self):
        return self._captions


class FallbackWhisperProvider:
    def transcribe(self, video_id: str):
        return [
            TranscriptSegment(start=0.0, duration=2.5, text="  First line. "),
            TranscriptSegment(start=2.5, duration=3.0, text="Second   line."),
        ]


def test_resolve_transcript_falls_back_to_whisper_when_captions_fail():
    transcript = resolve_transcript(
        "video123",
        transcript_provider=EmptyTranscriptProvider(),
        whisper_provider=FallbackWhisperProvider(),
    )

    assert transcript.source == "whisper"
    assert [segment.text for segment in transcript.segments] == ["First line.", "Second line."]


def test_normalize_transcript_collapses_whitespace_and_keeps_timestamps():
    normalized = normalize_transcript(
        [
            {"start": 1, "duration": 2, "text": " Hello\n world "},
            {"start": 3.5, "duration": 1, "text": "Next    part"},
        ],
        source="youtube",
    )

    assert normalized.source == "youtube"
    assert normalized.segments[0].start == 1.0
    assert normalized.segments[0].text == "Hello world"
    assert normalized.segments[0].raw_text == " Hello\n world "
    assert normalized.segments[1].text == "Next part"


def test_youtube_data_api_caption_provider_downloads_and_normalizes_srt():
    class FakeCaptions:
        def list(self, **kwargs):
            assert kwargs == {"part": "id,snippet", "videoId": "video123"}
            return self

        def download(self, **kwargs):
            assert kwargs == {"id": "caption-ru", "tfmt": "srt"}
            return FakeDownload()

        def execute(self):
            return {
                "items": [
                    {"id": "caption-en", "snippet": {"language": "en", "trackKind": "ASR"}},
                    {"id": "caption-ru", "snippet": {"language": "ru", "trackKind": "standard"}},
                ]
            }

    class FakeDownload:
        def execute(self):
            return (
                "1\n"
                "00:00:01,500 --> 00:00:04,000\n"
                " Первый  фрагмент \n\n"
                "2\n"
                "00:00:05,000 --> 00:00:07,250\n"
                "Второй\nфрагмент\n"
            ).encode("utf-8")

    class FakeYoutube:
        def captions(self):
            return FakeCaptions()

    raw, normalized = YouTubeDataApiCaptionProvider(
        youtube_service=FakeYoutube(),
        languages=["ru", "en"],
    ).fetch_bundle("video123")

    assert raw["source"] == "youtube_data_api_captions"
    assert raw["caption_id"] == "caption-ru"
    assert raw["language"] == "ru"
    assert normalized.source == "youtube_data_api_captions"
    assert normalized.language == "ru"
    assert [(segment.start, segment.duration, segment.text) for segment in normalized.segments] == [
        (1.5, 2.5, "Первый фрагмент"),
        (5.0, 2.25, "Второй фрагмент"),
    ]


def test_build_transcript_provider_data_api_only_can_exclude_unofficial_provider():
    settings = replace(
        get_settings(),
        youtube_api_token_file="token.json",
        acceptance_mode="data_api_only",
    )

    provider = build_transcript_provider(settings, allow_unofficial_provider=False)

    assert all(not isinstance(candidate, YouTubeTranscriptProvider) for candidate in provider.providers)


def test_build_transcript_provider_full_processing_keeps_unofficial_fallback():
    settings = replace(
        get_settings(),
        youtube_api_token_file="token.json",
        acceptance_mode="full_processing",
    )

    provider = build_transcript_provider(settings)

    assert [type(candidate).__name__ for candidate in provider.providers] == [
        "YouTubeDataApiCaptionProvider",
        "YouTubeTranscriptProvider",
    ]


def test_youtube_data_api_caption_provider_classifies_empty_caption_list():
    import yt_sub_hub.transcripts as transcripts

    provider = YouTubeDataApiCaptionProvider(youtube_service=FakeYoutube(FakeCaptionsResource(items=[])))

    try:
        provider.fetch_bundle("video123")
    except transcripts.TranscriptSourceError as exc:
        assert exc.reason == "captions_unavailable"
    else:
        raise AssertionError("empty captions should be classified")


def test_youtube_data_api_caption_provider_classifies_forbidden_download():
    import yt_sub_hub.transcripts as transcripts

    provider = YouTubeDataApiCaptionProvider(
        youtube_service=FakeYoutube(
            FakeCaptionsResource(
                items=[{"id": "caption-1", "snippet": {"language": "en"}}],
                download_error=FakeGoogleError(403, "forbidden", "caption download forbidden refresh_token=hidden"),
            )
        )
    )

    try:
        provider.fetch_bundle("video123")
    except transcripts.TranscriptSourceError as exc:
        assert exc.reason == "captions_forbidden"
        assert "refresh_token=hidden" not in str(exc)
    else:
        raise AssertionError("forbidden downloads should be classified")


def test_youtube_data_api_caption_provider_classifies_auth_and_quota_errors():
    import yt_sub_hub.transcripts as transcripts

    cases = [
        (FakeGoogleError(401, "authError", "invalid credentials access_token=hidden"), "auth_error"),
        (FakeGoogleError(403, "quotaExceeded", "quota exceeded"), "quota_or_rate_limited"),
    ]
    for error, reason in cases:
        provider = YouTubeDataApiCaptionProvider(
            youtube_service=FakeYoutube(FakeCaptionsResource(list_error=error))
        )
        try:
            provider.fetch_bundle("video123")
        except transcripts.TranscriptSourceError as exc:
            assert exc.reason == reason
            assert "access_token=hidden" not in str(exc)
        else:
            raise AssertionError(f"{reason} should be classified")


class RawTranscriptProvider:
    def fetch_bundle(self, video_id: str):
        raw = [{"start": 0, "duration": 1, "text": " Raw\n text "}]
        return raw, normalize_transcript(raw, source="youtube", video_id=video_id)


def test_resolve_transcript_bundle_preserves_raw_provider_payload():
    bundle = resolve_transcript_bundle(
        "video123",
        transcript_provider=RawTranscriptProvider(),
        whisper_provider=FallbackWhisperProvider(),
    )

    assert bundle.raw_payload[0]["text"] == " Raw\n text "
    assert bundle.normalized.segments[0].text == "Raw text"
    assert bundle.raw_payload != bundle.normalized.to_dict()


def test_whisper_audio_provider_reuses_existing_audio_file(tmp_path, monkeypatch):
    video_id = "abc123def45"
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    audio_path = audio_dir / f"{video_id}.mp3"
    audio_path.write_bytes(b"cached audio")

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")

    provider = WhisperAudioProvider("https://www.youtube.com/watch?v=abc123def45", audio_dir)

    assert provider._download_audio(video_id) == audio_path


def test_whisper_audio_provider_refuses_media_download_when_disabled(tmp_path):
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
        max_prompt_chars=3500,
        ytdlp_cookies_from_browser=None,
        ytdlp_cookies_file=None,
        ytdlp_proxy=None,
        ytdlp_js_runtime=None,
        ytdlp_socket_timeout=60,
        ytdlp_retries=10,
        media_download_enabled=False,
    )

    provider = WhisperAudioProvider(
        "https://www.youtube.com/watch?v=abc123def45",
        tmp_path / "audio",
        settings=settings,
    )

    try:
        provider._download_audio("abc123def45")
    except TranscriptResolutionError as exc:
        assert "YT_SUB_HUB_MEDIA_DOWNLOAD=disabled" in str(exc)
        assert "yt-dlp fallback не будет запущен" in str(exc)
    else:
        raise AssertionError("media download should be disabled")


def test_whisper_audio_provider_passes_proxy_to_ytdlp(tmp_path, monkeypatch):
    video_id = "abc123def45"
    captured = {}

    class FakeYoutubeDL:
        def __init__(self, options):
            captured["options"] = options

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def download(self, urls):
            (tmp_path / "audio" / f"{video_id}.mp3").write_bytes(b"audio")

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
        max_prompt_chars=3500,
        ytdlp_cookies_from_browser=None,
        ytdlp_cookies_file=None,
        ytdlp_proxy="http://127.0.0.1:12334",
        ytdlp_js_runtime=None,
        ytdlp_socket_timeout=60,
        ytdlp_retries=10,
        media_download_enabled=True,
    )

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    import yt_dlp

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYoutubeDL)

    provider = WhisperAudioProvider(
        "https://www.youtube.com/watch?v=abc123def45",
        tmp_path / "audio",
        settings=settings,
    )

    assert provider._download_audio(video_id) == tmp_path / "audio" / f"{video_id}.mp3"
    assert captured["options"]["proxy"] == "http://127.0.0.1:12334"


def test_whisper_audio_provider_uses_backend_bundle_without_changing_transcript_schema(tmp_path, monkeypatch):
    video_id = "abc123def45"
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    audio_path = audio_dir / f"{video_id}.mp3"
    audio_path.write_bytes(b"cached audio")
    captured = {}

    class FakeBackend:
        name = "openai-whisper-cpu"
        model = "base"
        device = "cpu"

        def transcribe(self, path, *, video_id=""):
            captured["path"] = path
            captured["video_id"] = video_id
            transcript = Transcript(
                video_id=video_id,
                source="whisper",
                language="en",
                segments=[TranscriptSegment(0.0, 1.0, "Hello world", " Hello world ")],
            )
            return TranscriptBundle(
                video_id=video_id,
                source="whisper",
                raw_payload={"segments": [{"start": 0, "end": 1, "text": " Hello world "}], "language": "en"},
                normalized=transcript,
            )

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")

    provider = WhisperAudioProvider(
        "https://www.youtube.com/watch?v=abc123def45",
        audio_dir,
        backend=FakeBackend(),
    )
    bundle = provider.transcribe_bundle(video_id)

    assert captured == {"path": audio_path, "video_id": video_id}
    assert bundle.raw_payload["segments"][0]["text"] == " Hello world "
    assert bundle.normalized.to_dict() == {
        "video_id": video_id,
        "source": "whisper",
        "language": "en",
        "segments": [
            {
                "start": 0.0,
                "duration": 1.0,
                "text": "Hello world",
                "raw_text": " Hello world ",
            }
        ],
    }
