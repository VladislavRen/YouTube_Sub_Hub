# YouTube Sub Hub

Local-first service that turns one YouTube video into a scientific Markdown note, stores all source artifacts in a folder library, and writes extracted concepts, claims, and topics to Neo4j.

For a new machine, follow the [self-hosting guide](docs/self-hosting.md).
See the [2026-10-22 fixes and validation](docs/deployment-fixes-2026-10-22.md)
for dependency versions and Google access requirements, and the
[handoff validation](docs/handoff-validation-2026-10-22.md) for the live runtime checks.

## Runtime Shape

- Python target: `3.11`
- Interfaces: CLI and FastAPI
- Metadata and official captions: YouTube Data API when `token.json` is configured
- Public transcript fallback: `youtube-transcript-api`
- Explicit media fallback: `yt-dlp` plus pluggable Whisper backends, only when `YT_SUB_HUB_MEDIA_DOWNLOAD=enabled`
- LLM: LM Studio OpenAI-compatible endpoint
- Graph: Neo4j

## Quick Start

```bash
uv sync --locked --extra dev --python 3.11
. .venv/bin/activate
cp .env.example .env
# Edit .env: set your own Neo4j password and installed LM Studio model IDs.
# Install ffmpeg/ffprobe separately and make sure they are on PATH.
docker compose up -d neo4j
```

`uv sync --locked` reproduces the checked-in dependency versions. Avoid
`uv pip install -e .` for a reproducible handoff because it does not use `uv.lock`.

LM Studio must be running on `http://localhost:1234/v1`. The example uses
`google/gemma-4-e4b` for both analysis roles and translation. Set the IDs to models
installed on your machine. Translation is enabled because the strict queue/API
quality gate requires Russian artifacts. Graph write errors fail processing by default.

Check the runtime before processing:

```bash
yt-sub-hub doctor --strict
```

Run one basic pipeline:

```bash
yt-sub-hub process "https://www.youtube.com/watch?v=VIDEO_ID"
```

To fetch live metadata and transcripts for an existing video, use `process URL
--refresh`. It regenerates dependent artifacts and preserves existing notes in
timestamped backups. `--force` reanalyzes using fresh cached inputs when available.
Raw inputs expire after 30 days. `yt-sub-hub cache-cleanup` previews expired raw
files; add `--apply` to delete them. It retains notes, derived results, discovery
data, media and Neo4j; see the self-hosting guide for retention limits.

If another model times out or cannot finish structured JSON output, switch to a
model that passes `doctor --strict` and a real processing run. For example:

```bash
YT_SUB_HUB_QWEN_MODEL=google/gemma-4-e4b yt-sub-hub doctor --strict
YT_SUB_HUB_QWEN_MODEL=google/gemma-4-e4b yt-sub-hub process "https://www.youtube.com/watch?v=VIDEO_ID"
```

For longer videos on Gemma's smaller context window, keep `YT_SUB_HUB_MAX_PROMPT_CHARS=3500` or lower.

By default the service runs in official/Data API-first mode and does not download media for Whisper. YouTube Data API can fetch metadata and, when your OAuth account has rights, caption files through `captions.download` with the `https://www.googleapis.com/auth/youtube.force-ssl` scope. It cannot download arbitrary video/audio streams.

## OAuth Credentials

Keep Google OAuth files local. `token.json`, `client_secret*.json`, and `client_secrets.json` are ignored by git and must not be committed.

To use official YouTube Data API metadata/caption access:

1. Create your own Google Cloud project and enable **YouTube Data API v3**.
2. Configure Google Auth Platform: audience, test users if in Testing, and the
   `https://www.googleapis.com/auth/youtube.force-ssl` scope under Data Access.
3. Create an OAuth client of type **Desktop app** and save its JSON as
   `client_secrets.json` in the project root. The helper only searches the current directory.
4. Run the local OAuth helper from the project root, on the machine with your browser:

```bash
python youtube_auth_test.py --timeout 1800
```

The helper reads `YT_SUB_HUB_YOUTUBE_API_TOKEN_FILE` from `.env` and defaults to
`token.json`. It checks granted scopes and atomically saves credentials with
owner-only permissions on POSIX systems. Failed authorization leaves the existing
file intact. You can also use the installed command and explicit paths:

```bash
yt-sub-hub-auth --client-secrets /secure/path/client_secrets.json --token-file /secure/path/token.json
```

`--no-browser` prints a local browser authorization URL; `--timeout` controls how
long to wait for consent. Keep the helper running until credentials are saved.
If the callback shows `localhost — ERR_CONNECTION_REFUSED`, start a new helper
session and open its new URL on the same machine. If overriding the token path on the command line, set
the same `YT_SUB_HUB_YOUTUBE_API_TOKEN_FILE` in `.env` for subsequent service calls.
Each operator must authorize their own account; do not share tokens or browser cookies.
The API client honors configured HTTP CONNECT/SOCKS proxies, including macOS
system settings and bypass rules; TLS-to-proxy (`https://` proxy URLs) is unsupported.

The scope is still supported, but official caption downloads require permission to
edit the video. Public visibility alone is insufficient. External apps in Testing
normally receive refresh tokens that expire after seven days. See the
[self-hosting guide](docs/self-hosting.md) for sources, verification rules, quota
limits, and troubleshooting.

For public subtitles without audio/video download, explicitly select
`--acceptance-mode full_processing` while leaving `YT_SUB_HUB_MEDIA_DOWNLOAD=disabled`.
This allows the non-official public transcript provider. If no usable subtitles
are available, processing fails; enabling media fallback remains a separate choice.
The strict doctor checks the configured default acceptance profile and currently
requires enabled media when the global profile is `full_processing`.

To allow non-official media fallback for videos without usable captions, enable it explicitly:

```bash
YT_SUB_HUB_MEDIA_DOWNLOAD=enabled yt-sub-hub process "https://www.youtube.com/watch?v=VIDEO_ID"
```

YouTube cookies are disabled by default. If the explicit media fallback reaches a YouTube bot-check, prefer an exported cookies file:

```bash
YT_SUB_HUB_YTDLP_COOKIES_FILE=/secure/path/youtube-cookies.txt yt-sub-hub process "https://www.youtube.com/watch?v=VIDEO_ID"
```

`YT_SUB_HUB_YTDLP_COOKIES_FROM_BROWSER=chrome` is still supported as an explicit fallback, but it reads browser cookies and can trigger macOS Keychain prompts.

The locked `yt-dlp[default]` includes EJS challenge scripts. Media fallback also
needs a supported JavaScript runtime on PATH: Deno is the yt-dlp default; to use
Node.js >=22, set `YT_SUB_HUB_YTDLP_JS_RUNTIME=node`. `doctor` checks for the runtime
and EJS when media fallback is enabled.

## Whisper Backend

The default Whisper fallback is:

```bash
YT_SUB_HUB_WHISPER_BACKEND=openai-whisper-cpu
YT_SUB_HUB_WHISPER_MODEL=base
YT_SUB_HUB_WHISPER_FP16=false
```

CPU remains the default because it is the safest cross-machine behavior, and `openai-whisper` on Apple MPS does not guarantee faster transcription. On some short local tests MPS can be slower than CPU.

Enable OpenAI Whisper on Apple MPS explicitly:

```bash
YT_SUB_HUB_WHISPER_BACKEND=openai-whisper-mps
YT_SUB_HUB_WHISPER_FP16=false
```

Enable a preinstalled whisper.cpp Metal backend:

```bash
YT_SUB_HUB_WHISPER_BACKEND=whisper-cpp-metal
YT_SUB_HUB_WHISPER_CPP_THREADS=6
```

Prepare the managed model file and print the exact environment paths:

```bash
yt-sub-hub setup-whisper-cpp --model base --download-model
```

The service auto-discovers `whisper-cli` from `PATH` and looks for models under
`library/assets/whisper.cpp/models/ggml-<model>.bin`. You can still override both paths explicitly:

```bash
YT_SUB_HUB_WHISPER_CPP_BIN=/path/to/whisper-cli
YT_SUB_HUB_WHISPER_CPP_MODEL=/path/to/ggml-model.bin
```

The service does not compile whisper.cpp or install native binaries automatically. Install whisper.cpp separately so `whisper-cli` is available, then run `setup-whisper-cpp` to manage the model file. `doctor --strict` validates the selected backend and reports missing binary/model paths with actionable messages. For Apple Silicon, whisper.cpp Metal/Core ML is often a better acceleration candidate than `openai-whisper` MPS.

Benchmark a configured backend without writing transcript artifacts:

```bash
yt-sub-hub benchmark-whisper library/assets/audio/VIDEO_ID.mp3
yt-sub-hub benchmark-whisper VIDEO_ID
```

The video-id form reads `library/assets/audio/<video_id>.mp3` and reports backend, model, device, elapsed seconds, detected language, segment count, and a short text sample.

## LM Studio Translation Benchmark

Translation chunks are processed conservatively by default:

```bash
YT_SUB_HUB_LM_STUDIO_TRANSLATION_CONCURRENCY=1
```

Before increasing that value for local LM Studio, benchmark the same translation provider and chunking logic without writing `transcript.ru.json` or queue artifacts:

```bash
NO_PROXY=localhost,127.0.0.1 \
YT_SUB_HUB_QWEN_MODEL=google/gemma-4-e4b \
yt-sub-hub benchmark-lm-studio-translation PL6ZiIly1mE \
  --concurrency 1,2,4 \
  --sample-chunks 12 \
  --timeout-seconds 120
```

The report is written to `library/benchmarks/lm_studio_translation_<video_id>_<timestamp>.json` by default. The CLI prints per-concurrency success, timeout/error/parse-error counts, latency percentiles, chunks/minute, and a deterministic safe-default recommendation.

Run one video:

```bash
yt-sub-hub process "https://www.youtube.com/watch?v=VIDEO_ID"
```

Example:

```bash
yt-sub-hub process "https://www.youtube.com/watch?v=w2uUux5abuY"
```

Start API:

```bash
uvicorn yt_sub_hub.api:create_app --factory --reload
```

## Library Layout

```text
library/
  videos/
    <video_id>/
      note.md
      metadata.json
      transcript.raw.json
      transcript.normalized.json
      analysis.json
  topics/
  graph/
    exports/
  assets/
    audio/
```

## Development Notes

The checked-in `pyproject.toml` targets Python 3.11. Use that version to reproduce
the tested environment; support for other Python versions has not been established.

Before publishing or opening a pull request, run:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --locked --extra dev python -m pytest -q
uv run --locked --extra dev python -m compileall -q src tests
```

Before committing, verify that only source, tests, docs, `pyproject.toml`, `uv.lock`, `.env.example`, and other safe project files are staged:

```bash
git diff --cached --name-only
```

Do not stage local OAuth files, `.env`, cookies, `library/`, sqlite databases, audio/video artifacts, virtualenvs, or IDE/cache directories.
