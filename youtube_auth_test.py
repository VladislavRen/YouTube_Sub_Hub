import os
from pathlib import Path

import google_auth_oauthlib.flow


# Needed for YouTube Data API captions.list/captions.download in addition to videos.list.
SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]


def find_client_secrets_file() -> str:
    explicit = Path("client_secrets.json")
    if explicit.exists():
        return str(explicit)

    matches = sorted(Path(".").glob("client_secret*.json"))
    if not matches:
        raise FileNotFoundError(
            "Не найден файл client_secrets.json или client_secret*.json. "
            "Положи OAuth client secret JSON из Google Cloud в корень проекта."
        )
    if len(matches) > 1:
        names = ", ".join(str(path) for path in matches)
        raise RuntimeError(f"Найдено несколько client_secret*.json файлов: {names}")

    return str(matches[0])


def perform_initial_auth() -> None:
    client_secrets_file = find_client_secrets_file()

    flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
        client_secrets_file,
        SCOPES,
    )

    creds = flow.run_local_server(port=0)

    with open("token.json", "w", encoding="utf-8") as token:
        token.write(creds.to_json())

    print("Успех! Файл token.json создан и сохранен в текущей папке.")


if __name__ == "__main__":
    # Only for local OAuth callback during development.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    perform_initial_auth()
