from pathlib import Path

from yt_sub_hub.config import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        library_dir=tmp_path,
        lm_studio_base_url="http://localhost:1234/v1",
        lm_studio_api_key="lm-studio",
        qwen_model="qwen",
        gemma_model="gemma",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        keep_audio=True,
        enable_graph=False,
        graph_best_effort=True,
        lm_studio_timeout_seconds=120,
        max_prompt_chars=3500,
        ytdlp_cookies_from_browser=None,
        ytdlp_cookies_file=None,
        ytdlp_proxy=None,
        ytdlp_js_runtime=None,
        ytdlp_socket_timeout=60,
        ytdlp_retries=10,
        whisper_backend="whisper-cpp-metal",
    )


def test_discovers_whisper_cpp_binary_from_path(tmp_path, monkeypatch):
    from yt_sub_hub.whisper_cpp_setup import resolve_whisper_cpp_binary

    binary = tmp_path / "whisper-cli"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setattr("shutil.which", lambda name: str(binary) if name == "whisper-cli" else None)

    assert resolve_whisper_cpp_binary(_settings(tmp_path)) == str(binary)


def test_resolves_managed_whisper_cpp_model_path(tmp_path):
    from yt_sub_hub.whisper_cpp_setup import managed_whisper_cpp_model_path, resolve_whisper_cpp_model

    settings = _settings(tmp_path)
    model_path = managed_whisper_cpp_model_path(settings, "base")

    assert model_path == tmp_path / "assets" / "whisper.cpp" / "models" / "ggml-base.bin"
    assert resolve_whisper_cpp_model(settings) is None

    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"model")

    assert resolve_whisper_cpp_model(settings) == str(model_path)


def test_prepare_whisper_cpp_downloads_managed_model(tmp_path, monkeypatch):
    from yt_sub_hub.whisper_cpp_setup import prepare_whisper_cpp

    binary = tmp_path / "whisper-cli"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setattr("shutil.which", lambda name: str(binary) if name == "whisper-cli" else None)

    downloads = []

    def fake_urlretrieve(url, filename):
        downloads.append((url, filename))
        Path(filename).write_bytes(b"model")
        return filename, None

    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)

    status = prepare_whisper_cpp(_settings(tmp_path), model_name="base", download_model=True)

    assert status.binary_path == str(binary)
    assert status.model_path.endswith("ggml-base.bin")
    assert status.model_ready is True
    assert status.model_downloaded is True
    assert downloads[0][0].endswith("/ggml-base.bin")
