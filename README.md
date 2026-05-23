# YouTube Sub Hub

Local-first service that turns one YouTube video into a scientific Markdown note, stores all source artifacts in a folder library, and writes extracted concepts, claims, and topics to Neo4j.

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
uv venv --python 3.11 .venv311
. .venv311/bin/activate
uv pip install --python .venv311/bin/python -e ".[dev]"
cp .env.example .env
python - <<'PY'
from pathlib import Path
path = Path(".env")
text = path.read_text()
text = text.replace("YT_SUB_HUB_NEO4J_PASSWORD=change-me-in-.env", "YT_SUB_HUB_NEO4J_PASSWORD=local-dev-password")
path.write_text(text)
PY
docker compose up -d neo4j
```

LM Studio must be running on `http://localhost:1234/v1`. The default model IDs in `.env.example` match the local models observed during setup:

- `qwen/qwen3.6-35b-a3b`
- `google/gemma-4-e4b`

Check the runtime before processing:

```bash
yt-sub-hub doctor --strict
```

Run one basic pipeline:

```bash
yt-sub-hub process "https://www.youtube.com/watch?v=VIDEO_ID"
```

If LM Studio refuses to load the Qwen model because of local memory guardrails, use the loaded Gemma model explicitly for a working local smoke run:

```bash
YT_SUB_HUB_QWEN_MODEL=google/gemma-4-e4b yt-sub-hub doctor --strict
YT_SUB_HUB_QWEN_MODEL=google/gemma-4-e4b yt-sub-hub process "https://www.youtube.com/watch?v=w2uUux5abuY"
```

For longer videos on Gemma's smaller context window, keep `YT_SUB_HUB_MAX_PROMPT_CHARS=3500` or lower.

By default the service runs in official/Data API-first mode and does not download media for Whisper. YouTube Data API can fetch metadata and, when your OAuth account has rights, caption files through `captions.download` with the `https://www.googleapis.com/auth/youtube.force-ssl` scope. It cannot download arbitrary video/audio streams.

## OAuth Credentials

Keep Google OAuth files local. `token.json`, `client_secret*.json`, and `client_secrets.json` are ignored by git and must not be committed.

To use official YouTube Data API metadata/caption access:

1. Create an OAuth installed-app client in Google Cloud.
2. Download the client secret JSON into the project root or another private path.
3. Run the local OAuth helper:

```bash
python youtube_auth_test.py
```

The helper writes `token.json` by default. That file contains refresh credentials, so keep it private. To store it elsewhere, set `YT_SUB_HUB_YOUTUBE_API_TOKEN_FILE=/secure/path/token.json` in `.env`.

To allow non-official media fallback for videos without usable captions, enable it explicitly:

```bash
YT_SUB_HUB_MEDIA_DOWNLOAD=enabled yt-sub-hub process "https://www.youtube.com/watch?v=VIDEO_ID"
```

YouTube cookies are disabled by default. If the explicit media fallback reaches a YouTube bot-check, prefer an exported cookies file:

```bash
YT_SUB_HUB_YTDLP_COOKIES_FILE=/secure/path/youtube-cookies.txt yt-sub-hub process "https://www.youtube.com/watch?v=VIDEO_ID"
```

`YT_SUB_HUB_YTDLP_COOKIES_FROM_BROWSER=chrome` is still supported as an explicit fallback, but it reads browser cookies and can trigger macOS Keychain prompts.

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

The checked-in `pyproject.toml` intentionally targets Python 3.11 because Whisper packages are not a good fit for the existing Python 3.14 virtualenv in this folder.

Before publishing or opening a pull request, run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv311/bin/python -m pytest -q
.venv311/bin/python -m compileall -q src tests
```

Before committing, verify that only source, tests, docs, `pyproject.toml`, `uv.lock`, `.env.example`, and other safe project files are staged:

```bash
git diff --cached --name-only
```

Do not stage local OAuth files, `.env`, cookies, `library/`, sqlite databases, audio/video artifacts, virtualenvs, or IDE/cache directories.
