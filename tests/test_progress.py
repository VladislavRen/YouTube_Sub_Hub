import json
import sqlite3
import time

import pytest

from test_artifact_review import _write_review_video
from yt_sub_hub.artifact_review import review_ru_artifacts
from yt_sub_hub.library import LibraryStore
from yt_sub_hub.progress import (
    JobNotFoundError,
    build_channel_job_progress,
    build_job_artifacts_status,
    build_job_quality_status,
    build_job_stage_progress,
)
from yt_sub_hub.runtime import RuntimeStore


def _job_with_stage_plan(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        job_id="job-1",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    return runtime, job


def test_stage_progress_groups_chunks_and_retries(tmp_path):
    runtime, job = _job_with_stage_plan(tmp_path)
    runtime.expand_translation_chunk_plan(
        job.id,
        [
            {"chunk_index": 0, "total_chunks": 3, "target_language": "ru", "source_segment_start_index": 0, "source_segment_end_index": 0},
            {"chunk_index": 1, "total_chunks": 3, "target_language": "ru", "source_segment_start_index": 1, "source_segment_end_index": 1},
            {"chunk_index": 2, "total_chunks": 3, "target_language": "ru", "source_segment_start_index": 2, "source_segment_end_index": 2},
        ],
    )
    runtime.expand_analysis_chunk_plan(
        job.id,
        [
            {"chunk_index": 0, "source_segment_start_index": 0, "source_segment_end_index": 0, "source_transcript_hash": "hash-a"},
            {"chunk_index": 1, "source_segment_start_index": 1, "source_segment_end_index": 1, "source_transcript_hash": "hash-a"},
        ],
    )
    now = time.time()
    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute(
            "UPDATE stage_plans SET status = 'completed', attempt = 1, completed_at = ?, updated_at = ? WHERE video_job_id = ? AND stage_name = 'transcript_translation_chunk_000'",
            (now, now, job.id),
        )
        conn.execute(
            "UPDATE stage_plans SET status = 'failed', attempt = 2, error = 'chunk failed', completed_at = ?, updated_at = ? WHERE video_job_id = ? AND stage_name = 'transcript_translation_chunk_001'",
            (now, now, job.id),
        )
        conn.execute(
            "UPDATE stage_plans SET status = 'running', attempt = 1, updated_at = ? WHERE video_job_id = ? AND stage_name = 'transcript_translation_chunk_002'",
            (now, job.id),
        )
        conn.execute(
            "UPDATE stage_plans SET status = 'completed_with_warning', attempt = 2, warning = 'cache hit', completed_at = ?, updated_at = ? WHERE video_job_id = ? AND stage_name = 'analysis_chunk_000'",
            (now, now, job.id),
        )
        conn.execute(
            "UPDATE stage_plans SET status = 'pending', attempt = 0, updated_at = ? WHERE video_job_id = ? AND stage_name = 'analysis_chunk_001'",
            (now, job.id),
        )
        conn.execute(
            """
            INSERT INTO stage_runs
              (id, video_job_id, stage_name, status, attempt, started_at, finished_at, error_message)
            VALUES
              ('run-retry-1', ?, 'transcript_translation_chunk_001', 'failed', 1, ?, ?, 'timeout'),
              ('run-retry-2', ?, 'analysis_chunk_000', 'failed', 1, ?, ?, 'rate limit')
            """,
            (job.id, now - 10, now - 5, job.id, now - 9, now - 4),
        )
        conn.execute(
            "INSERT INTO retries (id, stage_run_id, attempt, scheduled_at, reason, backoff_seconds) VALUES ('retry-1', 'run-retry-1', 2, ?, 'timeout', 1)",
            (now,),
        )
        conn.execute(
            "UPDATE video_jobs SET status = 'running', current_stage = 'transcript_translation_chunk_002' WHERE id = ?",
            (job.id,),
        )

    progress = build_job_stage_progress(runtime, LibraryStore(tmp_path), job.id)

    metadata_stage = next(stage for stage in progress["stages"] if stage["stage_name"] == "metadata")
    assert metadata_stage["id"]
    assert metadata_stage["video_job_id"] == job.id
    assert "hard_failure" in metadata_stage
    assert "retry_after" in metadata_stage
    assert "stage_payload_json" in metadata_stage
    translation = progress["chunks"]["transcript_translation"]
    assert translation["total"] == 3
    assert translation["completed"] == 1
    assert translation["failed"] == 1
    assert translation["running"] == 1
    assert translation["pending"] == 0
    assert translation["retried"] == 1
    assert translation["progress_ratio"] == pytest.approx(1 / 3)
    analysis = progress["chunks"]["analysis"]
    assert analysis["completed"] == 1
    assert analysis["pending"] == 1
    assert analysis["retried"] == 1
    assert not any(stage["name"].startswith("analysis_chunk_") for stage in progress["stages"])


def test_discovery_stage_progress_quality_and_artifact_status_use_discovery_payloads(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.queue_youtube_discovery("srcVideo123", limit=7, max_quota_units=300)
    store = LibraryStore(tmp_path)

    progress = build_job_stage_progress(runtime, store, job.id)

    assert progress["job_kind"] == "youtube_discovery"
    assert progress["product_quality"]["status"] == "queued"
    quality = build_job_quality_status(runtime, store, job.id)
    artifacts = build_job_artifacts_status(runtime, store, job.id)
    assert quality["job_kind"] == "youtube_discovery"
    assert quality["youtube_discovery"]["status"] == "queued"
    assert artifacts["job_kind"] == "youtube_discovery"
    assert "discovery_report" in artifacts["artifacts"]


def test_stage_progress_include_chunks_and_collects_reasons(tmp_path):
    runtime, job = _job_with_stage_plan(tmp_path)
    runtime.expand_translation_chunk_plan(
        job.id,
        [{"chunk_index": 0, "total_chunks": 1, "target_language": "ru", "source_segment_start_index": 0, "source_segment_end_index": 0}],
    )
    now = time.time()
    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute(
            "UPDATE stage_plans SET status = 'completed_with_warning', warning = 'needs whisper', completed_at = ?, updated_at = ? WHERE video_job_id = ? AND stage_name = 'transcript_source'",
            (now, now, job.id),
        )
        conn.execute(
            "UPDATE stage_plans SET status = 'failed', error = 'analysis.ru.json has sparse concepts', completed_at = ?, updated_at = ? WHERE video_job_id = ? AND stage_name = 'ru_artifact_review'",
            (now, now, job.id),
        )

    progress = build_job_stage_progress(runtime, LibraryStore(tmp_path), job.id, include_chunks=True)

    assert {"stage": "transcript_source", "message": "needs whisper"} in progress["warnings"]
    assert {"stage": "ru_artifact_review", "message": "analysis.ru.json has sparse concepts"} in progress["hard_failures"]
    assert any(stage["stage_name"] == "transcript_translation_chunk_000" for stage in progress["stages"])


def test_stage_progress_eta_does_not_crash_without_history(tmp_path):
    runtime, job = _job_with_stage_plan(tmp_path)

    progress = build_job_stage_progress(runtime, LibraryStore(tmp_path), job.id)

    assert progress["eta"]["seconds_remaining"] is None
    assert progress["eta"]["confidence"] in {"none", "low"}


def test_stage_progress_eta_clamps_lm_studio_translation_concurrency(tmp_path, monkeypatch):
    monkeypatch.setenv("YT_SUB_HUB_LM_STUDIO_TRANSLATION_CONCURRENCY", "9")
    runtime, job = _job_with_stage_plan(tmp_path)
    runtime.expand_translation_chunk_plan(
        job.id,
        [
            {"chunk_index": 0, "total_chunks": 4, "target_language": "ru", "source_segment_start_index": 0, "source_segment_end_index": 0},
            {"chunk_index": 1, "total_chunks": 4, "target_language": "ru", "source_segment_start_index": 1, "source_segment_end_index": 1},
            {"chunk_index": 2, "total_chunks": 4, "target_language": "ru", "source_segment_start_index": 2, "source_segment_end_index": 2},
            {"chunk_index": 3, "total_chunks": 4, "target_language": "ru", "source_segment_start_index": 3, "source_segment_end_index": 3},
        ],
    )
    now = time.time()
    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute(
            "UPDATE stage_plans SET status = 'completed', attempt = 1, completed_at = ?, updated_at = ? WHERE video_job_id = ? AND position < (SELECT position FROM stage_plans WHERE video_job_id = ? AND stage_name = 'transcript_translation_chunk_000')",
            (now, now, job.id, job.id),
        )
        conn.execute(
            """
            INSERT INTO stage_runs
              (id, video_job_id, stage_name, status, attempt, started_at, finished_at)
            VALUES ('run-history', ?, 'transcript_translation_chunk_000', 'completed', 1, ?, ?)
            """,
            (job.id, now - 10, now),
        )

    progress = build_job_stage_progress(runtime, LibraryStore(tmp_path), job.id, include_chunks=True)

    assert progress["eta"]["seconds_remaining"] == 20


def test_stage_progress_allows_missing_stage_plan(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        job_id="job-1",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )

    progress = build_job_stage_progress(runtime, LibraryStore(tmp_path), job.id)

    assert progress["status"] == "queued"
    assert progress["overall"]["total_stages"] == 0
    assert progress["stages"] == []
    assert progress["recovery"]["stuck"] is False
    assert progress["recovery"]["retryable"] is False
    assert "resume_job" not in progress["recovery"]["recommended_actions"]


def test_artifacts_status_reports_files_and_readiness(tmp_path):
    runtime, job = _job_with_stage_plan(tmp_path)
    store = LibraryStore(tmp_path)
    _write_review_video(store)
    review_ru_artifacts(job.video_id, store)

    payload = build_job_artifacts_status(runtime, store, job.id)

    assert payload["artifacts"]["metadata"]["exists"] is True
    assert payload["artifacts"]["metadata"]["size_bytes"] > 0
    assert payload["artifacts"]["metadata"]["path"] == "videos/abc123def45/metadata.json"
    assert payload["artifacts"]["ru_artifact_review"]["exists"] is True
    assert payload["readiness"]["has_russian_note"] is True
    assert payload["readiness"]["has_russian_graph"] is True
    assert payload["readiness"]["ready_for_neo4j"] is True
    assert payload["readiness"]["ready_for_user_reading"] is True
    assert payload["quality_mode"] == "strict"
    assert payload["acceptance_mode"] == "data_api_only"
    assert payload["product_quality"]["mode"] == "strict"


def test_quality_status_includes_ru_artifact_review_and_neo4j_stage(tmp_path):
    runtime, job = _job_with_stage_plan(tmp_path)
    store = LibraryStore(tmp_path)
    _write_review_video(store)
    review_ru_artifacts(job.video_id, store)
    now = time.time()
    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute(
            "UPDATE stage_plans SET status = 'completed', completed_at = ?, updated_at = ? WHERE video_job_id = ? AND stage_name = 'neo4j_sync'",
            (now, now, job.id),
        )

    payload = build_job_quality_status(runtime, store, job.id)

    assert payload["status"] == "completed"
    assert payload["quality_gate"]["status"] == "completed"
    assert payload["ru_artifact_review"]["status"] == "passed"
    assert "covered_segment_ratio" in payload["ru_artifact_review"]["coverage"]
    assert payload["neo4j"]["sync_stage_status"] == "completed"
    assert payload["neo4j"]["consistency"] == {"checked": False, "missing": None, "extra": None}
    assert payload["quality_mode"] == "strict"
    assert payload["acceptance_mode"] == "data_api_only"
    assert payload["product_quality"]["artifact_quality_score"]["status"] == "ready"
    assert payload["product_quality"]["artifact_quality_score"]["overall_score"] >= 0.85
    assert {key: payload["product_quality"][key] for key in ("mode", "status", "production_ready", "graph_sync_allowed", "reasons")} == {
        "mode": "strict",
        "status": "ready",
        "production_ready": True,
        "graph_sync_allowed": True,
        "reasons": [],
    }


def test_stage_progress_includes_draft_product_quality(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        job_id="job-draft",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
        quality_mode="draft",
    )
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)

    progress = build_job_stage_progress(runtime, LibraryStore(tmp_path), job.id)

    assert progress["quality_mode"] == "draft"
    assert progress["acceptance_mode"] == "data_api_only"
    assert progress["product_quality"]["status"] == "draft"
    assert progress["product_quality"]["production_ready"] is False


def test_missing_job_raises_job_not_found(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")

    with pytest.raises(JobNotFoundError):
        build_job_stage_progress(runtime, LibraryStore(tmp_path), "missing")


def test_channel_progress_aggregates_child_status_and_video_summaries(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    store = LibraryStore(tmp_path)
    channel = runtime.create_channel_job(
        channel_url="https://www.youtube.com/channel/UCexample123",
        channel_id="UCexample123",
        requested_limit=2,
        quality_mode="draft",
        acceptance_mode="data_api_only",
    )
    first = runtime.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
        channel_job_id=channel.id,
        quality_mode="draft",
        acceptance_mode="data_api_only",
    )
    second = runtime.create_video_job(
        video_id="xyz123def45",
        canonical_url="https://www.youtube.com/watch?v=xyz123def45",
        channel_job_id=channel.id,
        quality_mode="draft",
        acceptance_mode="data_api_only",
    )
    for job in [first, second]:
        runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    runtime.complete_video_job(first.id, warning="draft mode: not production validated")

    payload = build_channel_job_progress(runtime, store, channel.id)

    assert payload["channel_job"]["id"] == channel.id
    assert payload["status"] == "running"
    assert payload["total_videos"] == 2
    assert payload["counts"]["completed_with_warning"] == 1
    assert payload["counts"]["queued"] == 1
    assert payload["acceptance_mode"] == "data_api_only"
    assert payload["skipped_requires_media_count"] == 0
    assert {video["video_id"] for video in payload["videos"]} == {"abc123def45", "xyz123def45"}
    assert payload["videos"][0]["product_quality"]["status"] in {"draft", "processing"}
    assert "artifact_readiness" in payload["videos"][0]
    assert all("recovery" in video for video in payload["videos"])
    assert {video["recovery"]["retryable"] for video in payload["videos"]} <= {False}
    assert all("resume_job" not in video["recovery"]["recommended_actions"] for video in payload["videos"])
    assert payload["active_stages"] == []


def test_channel_progress_counts_skipped_requires_media(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    store = LibraryStore(tmp_path)
    channel = runtime.create_channel_job(
        channel_url="https://www.youtube.com/channel/UCexample123",
        channel_id="UCexample123",
        requested_limit=3,
        acceptance_mode="data_api_only",
    )
    for video_id in ["abc123def45", "xyz123def45", "def123abc45"]:
        job = runtime.create_video_job(
            video_id=video_id,
            canonical_url=f"https://www.youtube.com/watch?v={video_id}",
            channel_job_id=channel.id,
            acceptance_mode="data_api_only",
        )
        runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
        runtime.mark_video_job_skipped_requires_media(job.id)

    payload = build_channel_job_progress(runtime, store, channel.id)

    assert payload["status"] == "completed_with_warning"
    assert payload["counts"]["skipped"] == 3
    assert payload["skipped_requires_media_count"] == 3
    assert {video["acceptance_mode"] for video in payload["videos"]} == {"data_api_only"}
    assert {video["status"] for video in payload["videos"]} == {"skipped"}


def test_job_progress_does_not_treat_auth_failure_as_skipped_requires_media(tmp_path):
    runtime, job = _job_with_stage_plan(tmp_path)
    runtime.update_video_job_acceptance_mode(job.id, "data_api_only")
    now = time.time()
    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute(
            """
            UPDATE stage_plans
            SET status = 'failed', attempt = max_attempts, error = ?, completed_at = ?, updated_at = ?
            WHERE video_job_id = ? AND stage_name = 'transcript_source'
            """,
            ("invalid token", now, now, job.id),
        )
        conn.execute(
            "UPDATE video_jobs SET status = 'failed', current_stage = 'transcript_source', error = ? WHERE id = ?",
            ("invalid token", job.id),
        )

    payload = build_job_stage_progress(runtime, LibraryStore(tmp_path), job.id)

    assert payload["status"] == "failed"
    assert payload["product_quality"]["status"] != "skipped_requires_media"
    assert "retry_with_full_processing" not in payload["recovery"]["recommended_actions"]
