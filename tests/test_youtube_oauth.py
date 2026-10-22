from __future__ import annotations

import json
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest
from google.oauth2.credentials import Credentials

from yt_sub_hub.config import get_settings
from yt_sub_hub.youtube_data_api import YOUTUBE_API_SCOPES, YouTubeDataApiClientError, build_youtube_client


def _token(path, scopes):
    path.write_text(json.dumps({
        "token": "synthetic-access", "refresh_token": "synthetic-refresh",
        "client_id": "synthetic-client", "client_secret": "synthetic-secret",
        "scopes": scopes, "expiry": "2099-01-01T00:00:00Z",
    }))


@pytest.mark.parametrize("scopes", [[], ["https://www.googleapis.com/auth/youtube.readonly"]])
def test_client_rejects_missing_or_readonly_grants(tmp_path, monkeypatch, scopes):
    path = tmp_path / "token.json"
    _token(path, scopes)
    monkeypatch.setattr("googleapiclient.discovery.build", lambda *a, **kw: object())
    settings = replace(get_settings(), youtube_api_token_file=str(path))
    with pytest.raises(YouTubeDataApiClientError, match="scope"):
        build_youtube_client(settings)


def test_token_write_is_private_and_preserves_actual_grants(tmp_path):
    from yt_sub_hub.youtube_oauth import save_credentials

    path = tmp_path / "private" / "token.json"
    creds = Credentials("synthetic-access", refresh_token="synthetic-refresh", token_uri="https://oauth2.googleapis.com/token",
                        client_id="synthetic-client", client_secret="synthetic-secret", scopes=YOUTUBE_API_SCOPES)
    save_credentials(path, creds)
    saved = json.loads(path.read_text())
    assert saved["scopes"] == YOUTUBE_API_SCOPES
    assert saved["refresh_token"] == "synthetic-refresh"
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600


def test_failed_authorization_preserves_existing_token(tmp_path, monkeypatch):
    from yt_sub_hub.youtube_oauth import authorize

    path = tmp_path / "token.json"
    path.write_text("old token")
    client = tmp_path / "client.json"
    client.write_text(json.dumps({"installed": {}}))
    def rejected(**kwargs):
        raise RuntimeError("consent denied")
    monkeypatch.setattr("google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file",
                        lambda *a, **kw: SimpleNamespace(run_local_server=rejected))
    with pytest.raises(RuntimeError, match="consent denied"):
        authorize(client, path, open_browser=False)
    assert path.read_text() == "old token"


def test_authorization_does_not_save_partial_grant(tmp_path, monkeypatch):
    from yt_sub_hub.youtube_oauth import authorize

    client = tmp_path / "client.json"
    client.write_text(json.dumps({"installed": {}}))
    creds = Credentials("synthetic-access", refresh_token="synthetic-refresh", scopes=YOUTUBE_API_SCOPES,
                        granted_scopes=["https://www.googleapis.com/auth/youtube.readonly"])
    monkeypatch.setattr("google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file",
                        lambda *a, **kw: SimpleNamespace(run_local_server=lambda **kwargs: creds))
    with pytest.raises(ValueError, match="scope"):
        authorize(client, tmp_path / "token.json", open_browser=False)
    assert not (tmp_path / "token.json").exists()


def test_api_client_honors_system_https_proxy_and_bypass(tmp_path, monkeypatch):
    # OAuth uses Requests (including macOS system proxies); the API must take
    # the same configured route instead of silently attempting a blocked direct TLS connection.
    import urllib.request

    monkeypatch.delenv("https_proxy", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.setattr(urllib.request, "getproxies", lambda: {"https": "http://127.0.0.1:19080"})
    monkeypatch.setattr(urllib.request, "proxy_bypass", lambda host: host == "bypass.example")
    path = tmp_path / "token.json"
    _token(path, YOUTUBE_API_SCOPES)
    service = build_youtube_client(replace(get_settings(), youtube_api_token_file=str(path)))
    try:
        http = service._http.http
        proxy = http.proxy_info("https")
        assert proxy is not None
        assert proxy.isgood(), "Configured proxies require the PySocks transport dependency"
        assert (proxy.proxy_host, proxy.proxy_port) == ("127.0.0.1", 19080)
        assert proxy.applies_to("youtube.googleapis.com")
        assert not proxy.applies_to("bypass.example")
        assert 0 < http.timeout <= 60
    finally:
        service.close()


@pytest.mark.parametrize("url,kind,port,remote_dns", [
    ("http://proxy.example", 3, 80, True),
    ("socks5h://proxy.example:1080", 2, 1080, True),
    ("socks5://proxy.example", 2, 1080, False),
])
def test_google_proxy_scheme_defaults_match_route(url, kind, port, remote_dns, monkeypatch):
    from yt_sub_hub.youtube_data_api import _authorized_http

    monkeypatch.setattr("urllib.request.getproxies", lambda: {"all": url})
    transport = _authorized_http(Credentials("synthetic"))
    try:
        proxy = transport.http.proxy_info("https")
        assert (proxy.proxy_type, proxy.proxy_port, proxy.proxy_rdns) == (kind, port, remote_dns)
    finally:
        transport.close()


def test_google_proxy_rejects_tls_proxy_without_sending_credentials_in_cleartext(monkeypatch):
    from yt_sub_hub.youtube_data_api import _authorized_http

    monkeypatch.setattr("urllib.request.getproxies", lambda: {"https": "https://user:synthetic@proxy.example"})
    transport = _authorized_http(Credentials("synthetic"))
    try:
        with pytest.raises(ValueError, match="TLS-to-proxy") as error:
            transport.http.proxy_info("https")
        assert "synthetic" not in str(error.value)
    finally:
        transport.close()
