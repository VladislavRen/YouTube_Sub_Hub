import json
import sqlite3
import time
from pathlib import Path

import pytest

from yt_sub_hub.artifact_browser import (
    ArtifactKeyError,
    ArtifactMissingError,
    build_artifact_browser_for_job,
    build_artifact_browser_for_video,
    read_artifact_content,
    resolve_artifact,
)
from yt_sub_hub.library import LibraryStore
from yt_sub_hub.runtime import RuntimeStore


def _write_video_artifacts(root: Path, video_id: str = "abc123def45") -> None:
    video_dir = root / "videos" / video_id
    (video_dir / "review").mkdir(parents=True)
    (video_dir / "metadata.json").write_text(json.dumps({"video_id": video_id, "title": "Artifact Video"}), encoding="utf-8")
    (video_dir / "transcript.normalized.json").write_text(json.dumps({"segments": [{"text": "hello"}]}), encoding="utf-8")
    (video_dir / "transcript.ru.json").write_text(json.dumps({"segments": [{"text": "привет"}]}), encoding="utf-8")
    (video_dir / "analysis.ru.json").write_text(json.dumps({"summary": "анализ"}), encoding="utf-8")
    (video_dir / "note.md").write_text("# Russian Note\n", encoding="utf-8")
    (video_dir / "graph_extraction.json").write_text(json.dumps({"nodes": [], "relationships": []}), encoding="utf-8")
    (video_dir / "review" / "ru_artifact_review.json").write_text(json.dumps({"status": "passed"}), encoding="utf-8")


def test_browser_payload_includes_required_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("YT_SUB_HUB_NEO4J_BROWSER_URL", "http://neo4j.local/browser/")
    _write_video_artifacts(tmp_path)

    payload = build_artifact_browser_for_video(LibraryStore(tmp_path), "abc123def45")

    keys = {artifact["key"] for artifact in payload["artifacts"]}
    assert {
        "transcript_source",
        "transcript_ru",
        "analysis_ru",
        "note_md",
        "graph_extraction",
    }.issubset(keys)
    assert payload["title"] == "Artifact Video"
    assert "artifact_quality_score" in payload["product_quality"]
    assert payload["readiness"] == {"ready_for_reading": True, "ready_for_graph": True}
    note = next(artifact for artifact in payload["artifacts"] if artifact["key"] == "note_md")
    assert note["path"] == "videos/abc123def45/note.md"
    assert note["exists"] is True
    assert note["size_bytes"] > 0
    assert note["api_url"] == "/videos/abc123def45/artifacts/note_md"
    assert note["download_url"] == "/videos/abc123def45/artifacts/note_md/download"


def test_browser_payload_includes_youtube_discovery_artifacts(tmp_path):
    _write_video_artifacts(tmp_path)
    discovery_dir = tmp_path / "videos" / "abc123def45" / "youtube_discovery"
    discovery_dir.mkdir()
    (discovery_dir / "seeds.json").write_text(json.dumps({"items": []}), encoding="utf-8")
    (discovery_dir / "queries.json").write_text(json.dumps({"items": []}), encoding="utf-8")
    (discovery_dir / "video_candidates.json").write_text(json.dumps({"items": []}), encoding="utf-8")
    (discovery_dir / "channel_candidates.json").write_text(json.dumps({"items": []}), encoding="utf-8")
    (discovery_dir / "review.json").write_text(json.dumps({"status": "passed"}), encoding="utf-8")
    (discovery_dir / "graph.json").write_text(json.dumps({"nodes": [], "relationships": []}), encoding="utf-8")
    (discovery_dir / "discovery_report.md").write_text("# Discovery\n", encoding="utf-8")

    payload = build_artifact_browser_for_video(LibraryStore(tmp_path), "abc123def45")

    artifacts = {artifact["key"]: artifact for artifact in payload["artifacts"]}
    assert artifacts["youtube_discovery_report"]["exists"] is True
    assert artifacts["youtube_discovery_video_candidates"]["path"] == "videos/abc123def45/youtube_discovery/video_candidates.json"


def test_missing_artifacts_are_listed_as_not_existing(tmp_path):
    video_dir = tmp_path / "videos" / "abc123def45"
    video_dir.mkdir(parents=True)
    (video_dir / "metadata.json").write_text(json.dumps({"video_id": "abc123def45"}), encoding="utf-8")

    payload = build_artifact_browser_for_video(LibraryStore(tmp_path), "abc123def45")

    artifacts = {artifact["key"]: artifact for artifact in payload["artifacts"]}
    assert artifacts["transcript_ru"]["exists"] is False
    assert artifacts["transcript_ru"]["size_bytes"] is None
    assert artifacts["transcript_ru"]["updated_at"] is None
    assert payload["readiness"] == {"ready_for_reading": False, "ready_for_graph": False}


def test_invalid_artifact_key_is_rejected(tmp_path):
    store = LibraryStore(tmp_path)

    with pytest.raises(ArtifactKeyError):
        resolve_artifact(store, "abc123def45", "../token.json")

    with pytest.raises(ArtifactKeyError):
        resolve_artifact(store, "abc123def45", "token.json")


def test_browser_payload_does_not_probe_outside_library_for_unsafe_video_id(tmp_path):
    (tmp_path.parent / "note.md").write_text("outside", encoding="utf-8")

    payload = build_artifact_browser_for_video(LibraryStore(tmp_path), "..")

    note = next(artifact for artifact in payload["artifacts"] if artifact["key"] == "note_md")
    assert note["exists"] is False
    assert note["size_bytes"] is None


def test_artifact_content_reader_returns_json_and_markdown(tmp_path):
    _write_video_artifacts(tmp_path)
    store = LibraryStore(tmp_path)

    assert read_artifact_content(store, "abc123def45", "graph_extraction") == {"nodes": [], "relationships": []}
    assert read_artifact_content(store, "abc123def45", "note_md") == {
        "content": "# Russian Note\n",
        "content_type": "text/markdown",
    }


def test_missing_allowlisted_artifact_content_fails(tmp_path):
    store = LibraryStore(tmp_path)

    with pytest.raises(ArtifactMissingError):
        read_artifact_content(store, "abc123def45", "note_md")


def test_neo4j_payload_uses_config_and_parameterized_query(tmp_path, monkeypatch):
    monkeypatch.setenv("YT_SUB_HUB_NEO4J_BROWSER_URL", "http://neo4j.local/browser/")
    unsafe_video_id = "abc' RETURN secrets //"
    _write_video_artifacts(tmp_path, unsafe_video_id)

    payload = build_artifact_browser_for_video(LibraryStore(tmp_path), unsafe_video_id)

    assert payload["neo4j"]["browser_url"] == "http://neo4j.local/browser/"
    assert "$video_id" in payload["neo4j"]["query"]
    assert unsafe_video_id not in payload["neo4j"]["query"]
    assert payload["neo4j"]["parameters"] == {"video_id": unsafe_video_id}
    assert payload["neo4j"]["status"] == "unknown"


def test_job_artifact_browser_includes_job_metadata_and_neo4j_status(tmp_path):
    _write_video_artifacts(tmp_path)
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        job_id="job-1",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
        status="completed",
    )
    runtime.ensure_stage_plan(job.id)
    now = time.time()
    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute(
            "UPDATE stage_plans SET status = 'completed', completed_at = ?, updated_at = ? WHERE video_job_id = ? AND stage_name = 'neo4j_sync'",
            (now, now, job.id),
        )

    payload = build_artifact_browser_for_job(runtime, LibraryStore(tmp_path), job.id)

    assert payload["job_id"] == "job-1"
    assert payload["video_id"] == "abc123def45"
    assert payload["status"] == "completed"
    assert payload["neo4j"]["status"] == "available"
    assert payload["quality_mode"] == "strict"
    assert payload["product_quality"]["mode"] == "strict"
    assert "artifact_quality_score" in payload["product_quality"]


def test_job_artifact_browser_for_discovery_job_uses_discovery_quality_payload(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.queue_youtube_discovery("srcVideo123", limit=5, max_quota_units=300)

    payload = build_artifact_browser_for_job(runtime, LibraryStore(tmp_path), job.id)

    assert payload["job_id"] == job.id
    assert payload["product_quality"]["mode"] == "youtube_discovery"
    assert payload["product_quality"]["production_ready"] is False
    assert "artifact_quality_score" not in payload["product_quality"]


def test_job_artifact_browser_unknown_job_raises(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")

    with pytest.raises(KeyError):
        build_artifact_browser_for_job(runtime, LibraryStore(tmp_path), "missing")
