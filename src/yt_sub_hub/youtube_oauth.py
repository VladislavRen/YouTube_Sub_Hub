"""Local desktop OAuth; credentials are never printed or written before consent."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

YOUTUBE_API_SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]


def validate_scopes(credentials) -> list[str]:
    scopes = credentials.granted_scopes
    if scopes is None:
        scopes = credentials.scopes
    if isinstance(scopes, str):
        scopes = scopes.split()
    if not set(YOUTUBE_API_SCOPES).issubset(scopes or []):
        raise ValueError("Required youtube.force-ssl scope was not granted. Run yt-sub-hub-auth again.")
    return list(scopes)


def save_credentials(path: Path | str, credentials) -> None:
    """Replace atomically, including the actual scopes and owner-only permissions."""
    payload = json.loads(credentials.to_json())
    payload["scopes"] = validate_scopes(credentials)
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".oauth-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def find_client_secrets_file() -> Path:
    explicit = Path("client_secrets.json")
    if explicit.is_file():
        return explicit
    matches = sorted(Path.cwd().glob("client_secret*.json"))
    if len(matches) != 1:
        raise ValueError("Provide exactly one Desktop OAuth client JSON, or use --client-secrets PATH.")
    return matches[0]


def authorize(client_file: Path, token_file: Path, *, open_browser: bool = True, timeout_seconds: int = 600):
    from google_auth_oauthlib.flow import InstalledAppFlow

    client_file = Path(client_file).expanduser()
    config = json.loads(client_file.read_text(encoding="utf-8"))
    if "installed" not in config:
        raise ValueError("A Google OAuth client of type Desktop app is required.")
    flow = InstalledAppFlow.from_client_secrets_file(
        str(client_file), YOUTUBE_API_SCOPES, autogenerate_code_verifier=True,
    )
    credentials = flow.run_local_server(
        host="localhost", bind_addr="127.0.0.1", port=0,
        open_browser=open_browser, timeout_seconds=timeout_seconds,
        prompt="consent", access_type="offline",
        success_message="YouTube Sub Hub: authorization complete. You can close this tab.",
    )
    validate_scopes(credentials)
    if not credentials.refresh_token:
        raise ValueError("Google did not return a refresh token. Reauthorize with offline access.")
    save_credentials(token_file, credentials)
    return credentials


def main() -> None:
    from yt_sub_hub.config import get_settings
    from yt_sub_hub.youtube_data_api import sanitize_google_api_message

    parser = argparse.ArgumentParser(description="Authorize your own YouTube account with a Desktop OAuth client.")
    parser.add_argument("--client-secrets", type=Path)
    parser.add_argument("--token-file", type=Path, help="Defaults to YT_SUB_HUB_YOUTUBE_API_TOKEN_FILE from .env.")
    parser.add_argument("--no-browser", action="store_true", help="Print the URL for opening in a local browser.")
    parser.add_argument("--timeout", type=int, default=600, help="Seconds to wait for browser consent.")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    path = args.token_file or Path(get_settings().youtube_api_token_file or "token.json")
    try:
        authorize(args.client_secrets or find_client_secrets_file(), path,
                  open_browser=not args.no_browser, timeout_seconds=args.timeout)
    except Exception as exc:
        parser.exit(1, f"OAuth failed: {sanitize_google_api_message(exc)}\n")
    print(f"OAuth credentials saved securely to {path}. Do not share this file.")
