import json
from dataclasses import replace

import pytest

from yt_sub_hub.channel import ChannelFetchResult, ChannelVideo
from yt_sub_hub.config import get_settings


class FakeRequest:
    def __init__(self, payload=None, exc=None):
        self.payload = payload
        self.exc = exc

    def execute(self):
        if self.exc is not None:
            raise self.exc
        return self.payload


class FakeGoogleError(Exception):
    def __init__(self, status, reason, message):
        self.resp = type("Resp", (), {"status": status, "reason": reason})()
        self.content = json.dumps({"error": {"message": message, "errors": [{"reason": reason}]}}).encode()
        super().__init__(f"<HttpError {status} {message} access_token=secret-token client_secret=secret>")


class FakeYouTube:
    def __init__(self, *, metadata=None, captions=None, downloads=None, list_error=None, download_errors=None):
        self.metadata = metadata or {}
        self.captions_payload = captions or {}
        self.downloads = downloads or {}
        self.list_error = list_error
        self.download_errors = download_errors or {}
        self.calls = []
        self._resource = None

    def videos(self):
        self._resource = "videos"
        return self

    def captions(self):
        self._resource = "captions"
        return self

    def list(self, **kwargs):
        self.calls.append((self._resource, "list", kwargs))
        if self._resource == "videos":
            video_id = kwargs["id"]
            item = self.metadata.get(video_id)
            return FakeRequest({"items": [item] if item else []})
        if self.list_error is not None:
            return FakeRequest(exc=self.list_error)
        return FakeRequest({"items": self.captions_payload.get(kwargs["videoId"], [])})

    def download(self, **kwargs):
        self.calls.append((self._resource, "download", kwargs))
        caption_id = kwargs["id"]
        if caption_id in self.download_errors:
            return FakeRequest(exc=self.download_errors[caption_id])
        return FakeRequest(self.downloads[caption_id])


def _settings(tmp_path, *, media_download_enabled=False):
    token = tmp_path / "token.json"
    token.write_text("{}", encoding="utf-8")
    return replace(
        get_settings(),
        library_dir=tmp_path,
        youtube_api_token_file=str(token),
        target_language="ru",
        media_download_enabled=media_download_enabled,
        acceptance_mode="full_processing" if media_download_enabled else "data_api_only",
    )


def _metadata(video_id="abc123def45", *, language="en"):
    return {
        "id": video_id,
        "snippet": {"title": "API title", "channelTitle": "API channel", "defaultLanguage": language},
        "contentDetails": {"duration": "PT1M"},
    }


def _caption(caption_id="caption-ru-secret", language="ru", *, track_kind="standard"):
    return {
        "id": caption_id,
        "snippet": {
            "videoId": "abc123def45",
            "language": language,
            "name": f"{language} captions",
            "trackKind": track_kind,
            "isAutoSynced": False,
            "status": "serving",
        },
    }


def test_video_probe_captions_download_ready(tmp_path):
    from yt_sub_hub.captions_probe import probe_video_captions

    youtube = FakeYouTube(
        metadata={"abc123def45": _metadata()},
        captions={"abc123def45": [_caption()]},
        downloads={"caption-ru-secret": b"1\n00:00:00,000 --> 00:00:01,000\nHello\n"},
    )

    payload = probe_video_captions("abc123def45", settings=_settings(tmp_path), youtube_service=youtube)

    assert payload["metadata"]["status"] == "ok"
    assert payload["captions_count"] == 1
    assert payload["captions"][0]["caption_id_masked"] != "caption-ru-secret"
    assert payload["selected_caption_language"] == "ru"
    assert payload["downloadable"] is True
    assert payload["transcript_source_decision"] == "data_api_captions_ready"
    assert payload["whisper_fallback_required"] is False


def test_video_probe_empty_captions_needs_whisper_with_disabled_blocker(tmp_path):
    from yt_sub_hub.captions_probe import probe_video_captions

    youtube = FakeYouTube(metadata={"abc123def45": _metadata()}, captions={"abc123def45": []})

    payload = probe_video_captions("abc123def45", settings=_settings(tmp_path), youtube_service=youtube)

    assert payload["captions_count"] == 0
    assert payload["downloadable"] is False
    assert payload["transcript_source_decision"] == "needs_whisper_fallback"
    assert payload["whisper_fallback_required"] is True
    assert payload["whisper_fallback_allowed"] is False
    assert payload["media_download_mode"] == "disabled"
    assert payload["acceptance_mode"] == "data_api_only"
    assert payload["pipeline_blocker"] == "captions_unavailable_media_download_disabled"


def test_video_probe_data_api_only_blocks_whisper_even_when_media_enabled(tmp_path):
    from dataclasses import replace

    from yt_sub_hub.captions_probe import probe_video_captions

    youtube = FakeYouTube(metadata={"abc123def45": _metadata()}, captions={"abc123def45": []})
    settings = replace(_settings(tmp_path, media_download_enabled=True), acceptance_mode="data_api_only")

    payload = probe_video_captions("abc123def45", settings=settings, youtube_service=youtube)

    assert payload["transcript_source_decision"] == "needs_whisper_fallback"
    assert payload["acceptance_mode"] == "data_api_only"
    assert payload["media_download_mode"] == "enabled"
    assert payload["whisper_fallback_required"] is True
    assert payload["whisper_fallback_allowed"] is False
    assert payload["pipeline_blocker"] == "captions_unavailable_media_download_disabled"


def test_video_probe_media_download_enabled_allows_whisper_without_blocker(tmp_path):
    from yt_sub_hub.captions_probe import probe_video_captions

    youtube = FakeYouTube(metadata={"abc123def45": _metadata()}, captions={"abc123def45": []})

    payload = probe_video_captions("abc123def45", settings=_settings(tmp_path, media_download_enabled=True), youtube_service=youtube)

    assert payload["transcript_source_decision"] == "needs_whisper_fallback"
    assert payload["acceptance_mode"] == "full_processing"
    assert payload["whisper_fallback_allowed"] is True
    assert payload["media_download_mode"] == "enabled"
    assert payload["pipeline_blocker"] is None


def test_video_probe_download_forbidden_needs_whisper_and_sanitizes_error(tmp_path):
    from yt_sub_hub.captions_probe import probe_video_captions

    youtube = FakeYouTube(
        metadata={"abc123def45": _metadata()},
        captions={"abc123def45": [_caption()]},
        download_errors={"caption-ru-secret": FakeGoogleError(403, "forbidden", "caption download forbidden refresh_token=hidden")},
    )

    payload = probe_video_captions("abc123def45", settings=_settings(tmp_path), youtube_service=youtube)

    assert payload["downloadable"] is False
    assert payload["download_error_class"] == "forbidden"
    assert payload["transcript_source_decision"] == "needs_whisper_fallback"
    dumped = json.dumps(payload)
    assert "secret-token" not in dumped
    assert "client_secret" not in dumped
    assert "refresh_token=hidden" not in dumped


def test_video_probe_auth_error_from_captions_list(tmp_path):
    from yt_sub_hub.captions_probe import probe_video_captions

    youtube = FakeYouTube(
        metadata={"abc123def45": _metadata()},
        list_error=FakeGoogleError(401, "authError", "invalid credentials access_token=hidden"),
    )

    payload = probe_video_captions("abc123def45", settings=_settings(tmp_path), youtube_service=youtube)

    assert payload["transcript_source_decision"] == "auth_error"
    assert payload["download_error_class"] == "auth_error"
    assert "access_token" not in json.dumps(payload)


def test_provider_classification_matches_captions_probe_for_forbidden_auth_and_quota(tmp_path):
    import yt_sub_hub.transcripts as transcripts
    from yt_sub_hub.captions_probe import probe_video_captions
    from yt_sub_hub.transcripts import YouTubeDataApiCaptionProvider

    cases = [
        (
            FakeGoogleError(403, "forbidden", "caption download forbidden"),
            "captions_forbidden",
            "forbidden",
            "needs_whisper_fallback",
        ),
        (
            FakeGoogleError(401, "authError", "invalid credentials"),
            "auth_error",
            "auth_error",
            "auth_error",
        ),
        (
            FakeGoogleError(403, "quotaExceeded", "quota exceeded"),
            "quota_or_rate_limited",
            "quota_or_rate_limited",
            "quota_or_rate_limited",
        ),
    ]
    for error, provider_reason, probe_error_class, probe_decision in cases:
        youtube = FakeYouTube(
            metadata={"abc123def45": _metadata()},
            captions={"abc123def45": [_caption()]},
            download_errors={"caption-ru-secret": error},
        )
        provider = YouTubeDataApiCaptionProvider(settings=_settings(tmp_path), youtube_service=youtube)

        try:
            provider.fetch_bundle("abc123def45")
        except transcripts.TranscriptSourceError as exc:
            assert exc.reason == provider_reason
        else:
            raise AssertionError(f"{provider_reason} should be classified")

        payload = probe_video_captions("abc123def45", settings=_settings(tmp_path), youtube_service=youtube)
        assert payload["download_error_class"] == probe_error_class
        assert payload["transcript_source_decision"] == probe_decision


def test_video_probe_quota_error_from_metadata(tmp_path):
    from yt_sub_hub.captions_probe import probe_video_captions

    class QuotaYouTube(FakeYouTube):
        def list(self, **kwargs):
            if self._resource == "videos":
                return FakeRequest(exc=FakeGoogleError(403, "quotaExceeded", "quota exceeded"))
            return super().list(**kwargs)

    payload = probe_video_captions("abc123def45", settings=_settings(tmp_path), youtube_service=QuotaYouTube())

    assert payload["metadata"]["status"] == "error"
    assert payload["transcript_source_decision"] == "quota_or_rate_limited"


def test_video_probe_missing_token_returns_auth_error_payload(tmp_path):
    from yt_sub_hub.captions_probe import probe_captions

    settings = replace(get_settings(), library_dir=tmp_path, youtube_api_token_file=str(tmp_path / "missing-token.json"))

    payload = probe_captions("abc123def45", settings=settings)

    assert payload["target_kind"] == "video"
    assert payload["videos"][0]["metadata"]["status"] == "error"
    assert payload["videos"][0]["transcript_source_decision"] == "auth_error"
    assert payload["videos"][0]["download_error_class"] == "auth_error"


def test_bare_unavailable_video_id_does_not_fall_back_to_channel_listing(tmp_path, monkeypatch):
    from yt_sub_hub.captions_probe import probe_captions

    def fail_listing(*args, **kwargs):
        raise AssertionError("bare video id must not fall back to channel listing")

    youtube = FakeYouTube(metadata={})
    monkeypatch.setattr("yt_sub_hub.captions_probe.fetch_channel_listing", fail_listing)

    payload = probe_captions("zzzzzzzzzzz", settings=_settings(tmp_path), youtube_service=youtube)

    assert payload["schema_version"] == "captions_probe.v1"
    assert payload["target_kind"] == "video"
    assert payload["summary"]["total_videos"] == 1
    assert payload["summary"]["unavailable_count"] == 1
    assert payload["summary"]["can_run_data_api_only"] is False
    assert payload["summary"]["can_run_full_processing"] is False
    video = payload["videos"][0]
    assert video["video_id"] == "zzzzzzzzzzz"
    assert video["metadata"]["status"] == "error"
    assert video["metadata"]["error_class"] == "unavailable"
    assert video["transcript_source_decision"] == "unavailable"
    assert video["whisper_fallback_required"] is False
    assert video["pipeline_blocker"] is None


def test_unavailable_video_url_returns_video_payload_without_channel_error(tmp_path, monkeypatch):
    from yt_sub_hub.captions_probe import probe_captions

    def fail_listing(*args, **kwargs):
        raise AssertionError("video URL must not call channel listing")

    youtube = FakeYouTube(metadata={})
    monkeypatch.setattr("yt_sub_hub.captions_probe.fetch_channel_listing", fail_listing)

    payload = probe_captions("https://www.youtube.com/watch?v=zzzzzzzzzzz", settings=_settings(tmp_path), youtube_service=youtube)

    assert payload["target_kind"] == "video"
    assert payload["videos"][0]["video_id"] == "zzzzzzzzzzz"
    assert payload["videos"][0]["metadata"]["error_class"] == "unavailable"
    assert payload["videos"][0]["transcript_source_decision"] == "unavailable"


def test_channel_probe_aggregates_mixed_results_and_honors_limit(tmp_path, monkeypatch):
    from yt_sub_hub.captions_probe import probe_captions

    settings = _settings(tmp_path, media_download_enabled=True)
    videos = [
        ChannelVideo("ready123456", "https://www.youtube.com/watch?v=ready123456", "Ready"),
        ChannelVideo("needs123456", "https://www.youtube.com/watch?v=needs123456", "Needs"),
        ChannelVideo("gone1234567", "https://www.youtube.com/watch?v=gone1234567", "Gone"),
        ChannelVideo("skip1234567", "https://www.youtube.com/watch?v=skip1234567", "Skip"),
    ]
    seen_limits = []

    def fake_listing(target, *, limit=None, settings=None, youtube_service=None):
        seen_limits.append(limit)
        return ChannelFetchResult(
            channel_id="UCexample123",
            channel_url="https://www.youtube.com/channel/UCexample123",
            uploads_playlist_id="UUexample123",
            title="Example",
            videos=videos[:limit],
        )

    youtube = FakeYouTube(
        metadata={
            "ready123456": _metadata("ready123456"),
            "needs123456": _metadata("needs123456"),
        },
        captions={
            "ready123456": [_caption("caption-ready", "ru")],
            "needs123456": [],
        },
        downloads={"caption-ready": b"1\n00:00:00,000 --> 00:00:01,000\nReady\n"},
    )
    monkeypatch.setattr("yt_sub_hub.captions_probe.fetch_channel_listing", fake_listing)

    payload = probe_captions("@Example", limit=3, settings=settings, youtube_service=youtube)

    assert seen_limits == [3]
    assert payload["target_kind"] == "channel"
    assert [video["video_id"] for video in payload["videos"]] == ["ready123456", "needs123456", "gone1234567"]
    assert payload["summary"] == {
        "total_videos": 3,
        "data_api_caption_ready_count": 1,
        "needs_whisper_fallback_count": 1,
        "unavailable_count": 1,
        "can_run_data_api_only": False,
        "can_run_full_processing": False,
    }


def test_channel_probe_passes_read_only_youtube_client_to_listing(tmp_path, monkeypatch):
    from yt_sub_hub.captions_probe import probe_captions

    settings = _settings(tmp_path)
    youtube = FakeYouTube(
        metadata={"abc123def45": _metadata()},
        captions={"abc123def45": []},
    )
    observed = {}

    def fake_listing(target, *, limit=None, settings=None, youtube_service=None):
        observed["youtube_service"] = youtube_service
        return ChannelFetchResult(
            channel_id="UCexample123",
            channel_url="https://www.youtube.com/channel/UCexample123",
            uploads_playlist_id="UUexample123",
            title="Example",
            videos=[ChannelVideo("abc123def45", "https://www.youtube.com/watch?v=abc123def45", "First")],
        )

    monkeypatch.setattr("yt_sub_hub.captions_probe.fetch_channel_listing", fake_listing)

    probe_captions("@Example", limit=1, settings=settings, youtube_service=youtube)

    assert observed["youtube_service"] is youtube


def test_valid_channel_target_continues_to_call_channel_listing(tmp_path, monkeypatch):
    from yt_sub_hub.captions_probe import probe_captions

    youtube = FakeYouTube(
        metadata={"abc123def45": _metadata()},
        captions={"abc123def45": []},
    )
    observed = {"called": False}

    def fake_listing(target, *, limit=None, settings=None, youtube_service=None):
        observed["called"] = True
        assert target == "@Example"
        return ChannelFetchResult(
            channel_id="UCexample123",
            channel_url="https://www.youtube.com/channel/UCexample123",
            uploads_playlist_id="UUexample123",
            title="Example",
            videos=[ChannelVideo("abc123def45", "https://www.youtube.com/watch?v=abc123def45", "First")],
        )

    monkeypatch.setattr("yt_sub_hub.captions_probe.fetch_channel_listing", fake_listing)

    payload = probe_captions("@Example", limit=1, settings=_settings(tmp_path), youtube_service=youtube)

    assert observed["called"] is True
    assert payload["target_kind"] == "channel"


@pytest.mark.parametrize(
    ("status", "reason", "expected"),
    [
        (401, "authError", "auth_error"),
        (403, "quotaExceeded", "quota_or_rate_limited"),
    ],
)
def test_video_probe_download_auth_and_quota_errors_are_not_whisper_fallback(tmp_path, status, reason, expected):
    from yt_sub_hub.captions_probe import probe_video_captions

    youtube = FakeYouTube(
        metadata={"abc123def45": _metadata()},
        captions={"abc123def45": [_caption()]},
        download_errors={"caption-ru-secret": FakeGoogleError(status, reason, f"{reason} caption-ru-secret")},
    )

    payload = probe_video_captions("abc123def45", settings=_settings(tmp_path), youtube_service=youtube)

    assert payload["downloadable"] is False
    assert payload["download_error_class"] == expected
    assert payload["transcript_source_decision"] == expected
    assert payload["whisper_fallback_required"] is False
    assert "caption-ru-secret" not in json.dumps(payload)


def test_probe_rejects_invalid_target_with_controlled_error(tmp_path):
    from yt_sub_hub.captions_probe import CaptionsProbeError, probe_captions

    with pytest.raises(CaptionsProbeError, match="Unsupported captions probe target"):
        probe_captions("https://example.com/not-youtube", settings=_settings(tmp_path))


def test_channel_probe_is_read_only_and_does_not_enqueue_or_write_transcripts(tmp_path, monkeypatch):
    from yt_sub_hub.captions_probe import probe_captions

    settings = _settings(tmp_path)

    def fake_listing(target, *, limit=None, settings=None, youtube_service=None):
        return ChannelFetchResult(
            channel_id="UCexample123",
            channel_url="https://www.youtube.com/channel/UCexample123",
            uploads_playlist_id="UUexample123",
            title="Example",
            videos=[ChannelVideo("abc123def45", "https://www.youtube.com/watch?v=abc123def45", "First")],
        )

    def fail_enqueue(*args, **kwargs):
        raise AssertionError("captions probe must not create channel/video jobs")

    class FailingYoutubeDL:
        def __init__(self, *args, **kwargs):
            raise AssertionError("captions probe must not invoke yt-dlp")

    youtube = FakeYouTube(
        metadata={"abc123def45": _metadata()},
        captions={"abc123def45": []},
    )
    monkeypatch.setattr("yt_sub_hub.captions_probe.fetch_channel_listing", fake_listing)
    monkeypatch.setattr("yt_sub_hub.channel.enqueue_channel_videos", fail_enqueue)
    import yt_dlp

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FailingYoutubeDL)

    payload = probe_captions("@Example", limit=1, settings=settings, youtube_service=youtube)

    assert payload["summary"]["needs_whisper_fallback_count"] == 1
    assert not (tmp_path / "index.sqlite3").exists()
    assert not (tmp_path / "videos" / "abc123def45" / "transcript.raw.json").exists()
    assert not (tmp_path / "videos" / "abc123def45" / "transcript.normalized.json").exists()
