from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from yt_sub_hub.config import Settings

YOUTUBE_API_SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]


class YouTubeDataApiClientError(RuntimeError):
    pass


SECRET_PATTERNS = (
    re.compile(r"access_token=[^\s,;]+", re.IGNORECASE),
    re.compile(r"refresh_token=[^\s,;]+", re.IGNORECASE),
    re.compile(r"client_secret=[^\s,;]+", re.IGNORECASE),
    re.compile(r'("(?:access_token|refresh_token|client_secret)"\s*:\s*")[^"]+(")', re.IGNORECASE),
)


def sanitize_google_api_message(exc: BaseException | str) -> str:
    message = _extract_google_error_message(exc) if isinstance(exc, BaseException) else str(exc)
    for pattern in SECRET_PATTERNS:
        if pattern.pattern.startswith('("'):
            message = pattern.sub('"credential":"<redacted>"', message)
        else:
            message = pattern.sub("credential=<redacted>", message)
    return message.replace("\n", " ").strip()


def google_api_error_class(exc: BaseException) -> str:
    status = getattr(getattr(exc, "resp", None), "status", None)
    reason = _extract_google_error_reason(exc).lower()
    text = str(exc).lower()
    if status == 429 or any(marker in reason or marker in text for marker in ("quota", "ratelimit", "rate_limit", "userratelimit")):
        return "quota_or_rate_limited"
    if status == 403 or "forbidden" in reason or "forbidden" in text:
        return "forbidden"
    if status in {401} or any(
        marker in reason or marker in text
        for marker in ("autherror", "invalidauth", "invalidcredentials", "unauthorized", "invalidgrant", "invalid_grant", "invalid grant")
    ):
        return "auth_error"
    if status == 404 or "notfound" in reason:
        return "unavailable"
    return "unknown_error"


def build_youtube_client(settings: Settings, *, persist_refresh: bool = True):
    token_path = Path(settings.youtube_api_token_file or "")
    if not settings.youtube_api_token_file or not token_path.exists():
        raise YouTubeDataApiClientError(
            "YouTube Data API token file is required. Set YT_SUB_HUB_YOUTUBE_API_TOKEN_FILE."
        )
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise YouTubeDataApiClientError("google-api-python-client and google-auth are required") from exc

    try:
        creds = Credentials.from_authorized_user_file(str(token_path), YOUTUBE_API_SCOPES)
        if hasattr(creds, "has_scopes") and not creds.has_scopes(YOUTUBE_API_SCOPES):
            raise YouTubeDataApiClientError(
                "YouTube Data API token does not include required youtube.force-ssl scope."
            )
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            if persist_refresh:
                token_path.write_text(creds.to_json(), encoding="utf-8")
        if getattr(creds, "expired", False) and not getattr(creds, "refresh_token", None):
            raise YouTubeDataApiClientError("YouTube Data API token is expired and has no refresh token.")
        return build("youtube", "v3", credentials=creds)
    except YouTubeDataApiClientError:
        raise
    except Exception as exc:
        raise YouTubeDataApiClientError(f"YouTube Data API credentials are not usable: {sanitize_google_api_message(exc)}") from exc


def _extract_google_error_message(exc: BaseException) -> str:
    parsed = _google_error_content(exc)
    error = parsed.get("error") if isinstance(parsed, dict) else None
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])
    return str(exc)


def _extract_google_error_reason(exc: BaseException) -> str:
    parsed = _google_error_content(exc)
    error = parsed.get("error") if isinstance(parsed, dict) else None
    if isinstance(error, dict):
        errors = error.get("errors")
        if isinstance(errors, list) and errors:
            reason = errors[0].get("reason") if isinstance(errors[0], dict) else None
            if reason:
                return str(reason)
        if error.get("status"):
            return str(error["status"])
    return str(getattr(getattr(exc, "resp", None), "reason", "") or "")


def _google_error_content(exc: BaseException) -> dict[str, Any]:
    content = getattr(exc, "content", None)
    if not content:
        return {}
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(str(content))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
