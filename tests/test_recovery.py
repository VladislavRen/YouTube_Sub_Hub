import json
import sqlite3
import time

import pytest

from yt_sub_hub.library import LibraryStore
from yt_sub_hub.recovery import build_job_recovery_dashboard
from yt_sub_hub.runtime import RuntimeStore


def _stage_by_name(runtime: RuntimeStore, job_id: str, stage_name: str):
    return [stage for stage in runtime.list_stage_plan(job_id) if stage.stage_name == stage_name][0]


def _job_with_plan(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        job_id="job-1",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    return runtime, job


def test_recovery_dashboard_uses_discovery_stage_plan_for_discovery_jobs(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.queue_youtube_discovery("srcVideo123", limit=7, max_quota_units=300)

    dashboard = build_job_recovery_dashboard(runtime, LibraryStore(tmp_path), job.id)

    assert dashboard["job_kind"] == "youtube_discovery"
    assert "missing_static_stage_plan" not in dashboard["mismatches"]
    assert "unexpected_static_stage_plan" not in dashboard["mismatches"]


def test_recovery_dashboard_groups_failed_discovery_query_chunks(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.queue_youtube_discovery("srcVideo123", limit=7, max_quota_units=300)
    runtime.expand_youtube_discovery_query_plan(
        job.id,
        {"items": [{"query_id": "ydq-001", "query": "graph review", "seed_ids": ["seed-001"]}]},
    )
    now = time.time()
    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute(
            """
            UPDATE stage_plans
            SET status = 'failed', attempt = max_attempts, error = ?, completed_at = ?, updated_at = ?
            WHERE video_job_id = ? AND stage_name = 'youtube_discovery_query_000'
            """,
            ("quota failed", now, now, job.id),
        )
        conn.execute(
            "UPDATE video_jobs SET status = 'failed', current_stage = 'youtube_discovery_query_000', error = ? WHERE id = ?",
            ("quota failed", job.id),
        )

    dashboard = build_job_recovery_dashboard(runtime, LibraryStore(tmp_path), job.id)

    assert dashboard["failed_chunks"]["youtube_discovery_queries"][0]["stage_name"] == "youtube_discovery_query_000"
    assert "youtube_discovery_queries" in dashboard["actions"]["retry_failed_chunks"]["groups"]


def test_recovery_dashboard_shows_failed_hard_stage_with_retry_action(tmp_path):
    runtime, job = _job_with_plan(tmp_path)
    now = time.time()
    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute(
            """
            UPDATE stage_plans
            SET status = 'failed', attempt = max_attempts, error = ?, completed_at = ?, updated_at = ?
            WHERE video_job_id = ? AND stage_name = 'transcript_source'
            """,
            ("captions unavailable", now, now, job.id),
        )
        conn.execute(
            "UPDATE video_jobs SET status = 'failed', current_stage = 'transcript_source', error = ? WHERE id = ?",
            ("captions unavailable", job.id),
        )

    payload = build_job_recovery_dashboard(runtime, LibraryStore(tmp_path), job.id)

    assert payload["job_id"] == job.id
    assert payload["stuck"] is True
    assert "pending_blocked_by_failed_upstream" in payload["stuck_reasons"]
    assert payload["failed_stages"] == [
        {
            "stage_name": "transcript_source",
            "status": "failed",
            "hard_failure": True,
            "error": "captions unavailable",
            "warning": None,
            "retry_after": None,
        }
    ]
    assert "retry_stage" in payload["recommended_actions"]
    assert payload["actions"]["retry_stage"]["applicable"] is True
    assert set(payload["recommended_actions"]) <= {
        name for name, action in payload["actions"].items() if action["applicable"]
    }


def test_recovery_dashboard_recommends_full_processing_for_skipped_requires_media(tmp_path):
    runtime, job = _job_with_plan(tmp_path)
    runtime.update_video_job_acceptance_mode(job.id, "data_api_only")
    runtime.mark_video_job_skipped_requires_media(job.id)

    payload = build_job_recovery_dashboard(runtime, LibraryStore(tmp_path), job.id)

    assert payload["acceptance_mode"] == "data_api_only"
    assert payload["skip_reason"] == "skipped_requires_media"
    assert payload["recommended_actions"] == ["retry_with_full_processing"]
    assert payload["actions"]["retry_with_full_processing"]["applicable"] is True


def test_recovery_dashboard_recommends_normal_retry_for_auth_failure(tmp_path):
    runtime, job = _job_with_plan(tmp_path)
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

    payload = build_job_recovery_dashboard(runtime, LibraryStore(tmp_path), job.id)

    assert payload["acceptance_mode"] == "data_api_only"
    assert payload["skip_reason"] is None
    assert "retry_with_full_processing" not in payload["recommended_actions"]
    assert "retry_stage" in payload["recommended_actions"]


def test_recovery_dashboard_groups_failed_translation_and_analysis_chunks(tmp_path):
    runtime, job = _job_with_plan(tmp_path)
    runtime.expand_translation_chunk_plan(
        job.id,
        [
            {"chunk_index": 0, "total_chunks": 2, "target_language": "ru", "source_segment_start_index": 0, "source_segment_end_index": 0},
            {"chunk_index": 1, "total_chunks": 2, "target_language": "ru", "source_segment_start_index": 1, "source_segment_end_index": 1},
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
            "UPDATE stage_plans SET status = 'failed', attempt = max_attempts, error = ?, completed_at = ?, updated_at = ? WHERE video_job_id = ? AND stage_name = 'transcript_translation_chunk_001'",
            ("translation chunk failed", now, now, job.id),
        )
        conn.execute(
            "UPDATE stage_plans SET status = 'failed', attempt = max_attempts, error = ?, completed_at = ?, updated_at = ? WHERE video_job_id = ? AND stage_name = 'analysis_chunk_000'",
            ("analysis chunk failed", now, now, job.id),
        )
        conn.execute("UPDATE video_jobs SET status = 'failed', error = ? WHERE id = ?", ("chunk failed", job.id))

    payload = build_job_recovery_dashboard(runtime, LibraryStore(tmp_path), job.id)

    assert [chunk["stage_name"] for chunk in payload["failed_chunks"]["transcript_translation"]] == [
        "transcript_translation_chunk_001"
    ]
    assert [chunk["stage_name"] for chunk in payload["failed_chunks"]["analysis"]] == ["analysis_chunk_000"]
    assert "retry_failed_chunks" in payload["recommended_actions"]
    assert set(payload["recommended_actions"]) <= {
        name for name, action in payload["actions"].items() if action["applicable"]
    }


def test_recovery_dashboard_detects_expired_running_stage_retry_and_lock(tmp_path):
    runtime, job = _job_with_plan(tmp_path)
    now = time.time()
    run = runtime.start_stage_run(video_job_id=job.id, stage_name="scientific_analysis")
    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute(
            """
            UPDATE stage_plans
            SET status = 'running', lease_owner = ?, lease_until = ?, resource_slot = ?, retry_after = ?, updated_at = ?
            WHERE video_job_id = ? AND stage_name = 'scientific_analysis'
            """,
            ("old-worker", now - 30, "lm_studio_analysis#0", now + 90, now, job.id),
        )
        conn.execute(
            "INSERT INTO locks (lock_name, owner, lease_until) VALUES (?, ?, ?)",
            ("lm_studio_analysis#0", "old-worker", now - 30),
        )
        conn.execute(
            "INSERT INTO retries (id, stage_run_id, attempt, scheduled_at, reason, backoff_seconds) VALUES (?, ?, ?, ?, ?, ?)",
            ("retry-1", run.id, 2, now + 90, "rate limit", 90),
        )
        conn.execute(
            "UPDATE video_jobs SET status = 'running', current_stage = 'scientific_analysis', lease_owner = ?, lease_until = ? WHERE id = ?",
            ("old-worker", now - 30, job.id),
        )

    payload = build_job_recovery_dashboard(runtime, LibraryStore(tmp_path), job.id)

    assert payload["stuck"] is True
    assert "expired_stage_lease" in payload["stuck_reasons"]
    assert "expired_job_lease" in payload["stuck_reasons"]
    assert payload["running_stages"][0]["expired"] is True
    assert payload["pending_retries"][0]["stage_name"] == "scientific_analysis"
    assert payload["locks"][0]["stale"] is True
    assert payload["actions"]["resume_job"]["applicable"] is True
    assert "resume_job" in payload["recommended_actions"]
    assert set(payload["recommended_actions"]) <= {
        name for name, action in payload["actions"].items() if action["applicable"]
    }


def test_recovery_recommendations_match_applicable_actions_for_healthy_active_jobs(tmp_path):
    runtime, queued = _job_with_plan(tmp_path)

    queued_payload = build_job_recovery_dashboard(runtime, LibraryStore(tmp_path), queued.id)

    assert queued_payload["actions"]["resume_job"]["applicable"] is False
    assert "resume_job" not in queued_payload["recommended_actions"]
    assert set(queued_payload["recommended_actions"]) <= {
        name for name, action in queued_payload["actions"].items() if action["applicable"]
    }

    running = runtime.create_video_job(
        job_id="running-job",
        video_id="xyz123def45",
        canonical_url="https://www.youtube.com/watch?v=xyz123def45",
    )
    runtime.ensure_stage_plan(running.id, enable_deep_graph_llm=False)
    now = time.time()
    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute(
            """
            UPDATE video_jobs
            SET status = 'running', current_stage = 'metadata', lease_owner = ?, lease_until = ?
            WHERE id = ?
            """,
            ("worker-1", now + 300, running.id),
        )
        conn.execute(
            """
            UPDATE stage_plans
            SET status = 'running', lease_owner = ?, lease_until = ?, updated_at = ?
            WHERE video_job_id = ? AND stage_name = 'metadata'
            """,
            ("worker-1", now + 300, now, running.id),
        )

    running_payload = build_job_recovery_dashboard(runtime, LibraryStore(tmp_path), running.id)

    assert running_payload["stuck"] is False
    assert running_payload["actions"]["resume_job"]["applicable"] is False
    assert "resume_job" not in running_payload["recommended_actions"]
    assert set(running_payload["recommended_actions"]) <= {
        name for name, action in running_payload["actions"].items() if action["applicable"]
    }


def test_runtime_retry_failed_chunks_preserves_completed_later_sibling_chunk(tmp_path):
    runtime, job = _job_with_plan(tmp_path)
    runtime.expand_translation_chunk_plan(
        job.id,
        [
            {"chunk_index": 0, "total_chunks": 3, "target_language": "ru", "source_segment_start_index": 0, "source_segment_end_index": 0},
            {"chunk_index": 1, "total_chunks": 3, "target_language": "ru", "source_segment_start_index": 1, "source_segment_end_index": 1},
            {"chunk_index": 2, "total_chunks": 3, "target_language": "ru", "source_segment_start_index": 2, "source_segment_end_index": 2},
        ],
    )
    now = time.time()
    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute(
            "UPDATE stage_plans SET status = 'completed', attempt = 1, completed_at = ?, updated_at = ? WHERE video_job_id = ? AND position < 5",
            (now, now, job.id),
        )
        conn.execute(
            "UPDATE stage_plans SET status = 'failed', attempt = max_attempts, error = ?, completed_at = ?, updated_at = ? WHERE video_job_id = ? AND stage_name = 'transcript_translation_chunk_000'",
            ("chunk failed", now, now, job.id),
        )
        conn.execute(
            "UPDATE stage_plans SET status = 'completed', attempt = 1, completed_at = ?, updated_at = ? WHERE video_job_id = ? AND stage_name IN ('transcript_translation_chunk_001', 'transcript_translation_chunk_002', 'transcript_translation_merge', 'scientific_analysis')",
            (now, now, job.id),
        )
        conn.execute(
            "INSERT INTO locks (lock_name, owner, lease_until) VALUES (?, ?, ?)",
            ("local_validation#0", "old-worker", now - 10),
        )
        conn.execute(
            "UPDATE stage_plans SET resource_slot = ?, lease_owner = ? WHERE video_job_id = ? AND stage_name = 'transcript_translation_merge'",
            ("local_validation#0", "old-worker", job.id),
        )
        conn.execute("UPDATE video_jobs SET status = 'failed', error = ? WHERE id = ?", ("chunk failed", job.id))

    record = runtime.retry_failed_chunks(job.id, "transcript_translation")

    assert record.status == "queued"
    assert record.current_stage == "transcript_translation_chunk_000"
    assert _stage_by_name(runtime, job.id, "transcript_translation_chunk_000").status == "pending"
    assert _stage_by_name(runtime, job.id, "transcript_translation_chunk_001").status == "completed"
    assert _stage_by_name(runtime, job.id, "transcript_translation_chunk_002").status == "completed"
    assert _stage_by_name(runtime, job.id, "transcript_translation_merge").status == "pending"
    assert _stage_by_name(runtime, job.id, "scientific_analysis").status == "pending"
    with sqlite3.connect(runtime.db_path) as conn:
        assert conn.execute("SELECT 1 FROM locks WHERE lock_name = 'local_validation#0'").fetchone() is None


def test_runtime_retry_failed_analysis_chunks_resets_reduce_and_downstream(tmp_path):
    runtime, job = _job_with_plan(tmp_path)
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
            "UPDATE stage_plans SET status = 'completed', attempt = 1, completed_at = ?, updated_at = ? WHERE video_job_id = ? AND position < (SELECT position FROM stage_plans WHERE video_job_id = ? AND stage_name = 'analysis_chunk_000')",
            (now, now, job.id, job.id),
        )
        conn.execute(
            "UPDATE stage_plans SET status = 'failed', attempt = max_attempts, error = ?, completed_at = ?, updated_at = ? WHERE video_job_id = ? AND stage_name = 'analysis_chunk_000'",
            ("chunk failed", now, now, job.id),
        )
        conn.execute(
            "UPDATE stage_plans SET status = 'completed', attempt = 1, completed_at = ?, updated_at = ? WHERE video_job_id = ? AND stage_name IN ('analysis_chunk_001', 'analysis_reduce', 'analysis_translation')",
            (now, now, job.id),
        )
        conn.execute("UPDATE video_jobs SET status = 'failed', error = ? WHERE id = ?", ("chunk failed", job.id))

    runtime.retry_failed_chunks(job.id, "analysis")

    assert _stage_by_name(runtime, job.id, "analysis_chunk_000").status == "pending"
    assert _stage_by_name(runtime, job.id, "analysis_chunk_001").status == "completed"
    assert _stage_by_name(runtime, job.id, "analysis_reduce").status == "pending"
    assert _stage_by_name(runtime, job.id, "analysis_translation").status == "pending"


def test_runtime_reset_downstream_and_resume_preserve_upstream_completed_work(tmp_path):
    runtime, job = _job_with_plan(tmp_path)
    runtime.expand_translation_chunk_plan(
        job.id,
        [{"chunk_index": 0, "total_chunks": 1, "target_language": "ru", "source_segment_start_index": 0, "source_segment_end_index": 0}],
    )
    now = time.time()
    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute(
            "UPDATE stage_plans SET status = 'completed', attempt = 1, completed_at = ?, updated_at = ? WHERE video_job_id = ?",
            (now, now, job.id),
        )
        conn.execute(
            "UPDATE stage_plans SET status = 'failed', attempt = max_attempts, error = ?, completed_at = ?, updated_at = ? WHERE video_job_id = ? AND stage_name = 'analysis_ru_review'",
            ("review failed", now, now, job.id),
        )
        conn.execute("UPDATE video_jobs SET status = 'failed', current_stage = 'analysis_ru_review', error = ? WHERE id = ?", ("review failed", job.id))

    runtime.reset_downstream(job.id, "analysis_translation")
    assert _stage_by_name(runtime, job.id, "transcript_translation_chunk_000").status == "completed"
    assert _stage_by_name(runtime, job.id, "analysis_translation").status == "pending"

    resume_job = runtime.create_video_job(
        job_id="job-2",
        video_id="xyz123def45",
        canonical_url="https://www.youtube.com/watch?v=xyz123def45",
    )
    runtime.ensure_stage_plan(resume_job.id, enable_deep_graph_llm=False)
    runtime.expand_translation_chunk_plan(
        resume_job.id,
        [{"chunk_index": 0, "total_chunks": 1, "target_language": "ru", "source_segment_start_index": 0, "source_segment_end_index": 0}],
    )
    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute(
            "UPDATE stage_plans SET status = 'completed', attempt = 1, completed_at = ?, updated_at = ? WHERE video_job_id = ?",
            (now, now, resume_job.id),
        )
        conn.execute(
            "UPDATE stage_plans SET status = 'failed', attempt = max_attempts, error = ?, completed_at = ?, updated_at = ? WHERE video_job_id = ? AND stage_name = 'analysis_ru_review'",
            ("review failed", now, now, resume_job.id),
        )
        conn.execute("UPDATE video_jobs SET status = 'failed', current_stage = 'analysis_ru_review', error = ? WHERE id = ?", ("review failed", resume_job.id))

    resumed = runtime.resume_job(resume_job.id)
    assert resumed.current_stage == "analysis_ru_review"
    assert _stage_by_name(runtime, resume_job.id, "transcript_translation_chunk_000").status == "completed"
    assert _stage_by_name(runtime, resume_job.id, "analysis_translation").status == "completed"
    assert _stage_by_name(runtime, resume_job.id, "analysis_ru_review").status == "pending"


def test_runtime_cancel_job_skips_queued_pending_stages(tmp_path):
    runtime, job = _job_with_plan(tmp_path)

    cancelled = runtime.cancel_job(job.id)

    assert cancelled.status == "skipped"
    assert cancelled.cancel_requested is True
    assert all(stage.status == "skipped" for stage in runtime.list_stage_plan(job.id) if stage.hard_failure)


def test_runtime_retry_failed_chunks_rejects_invalid_or_empty_group(tmp_path):
    runtime, job = _job_with_plan(tmp_path)

    with pytest.raises(ValueError, match="invalid chunk group"):
        runtime.retry_failed_chunks(job.id, "unknown")
    with pytest.raises(ValueError, match="no failed chunks"):
        runtime.retry_failed_chunks(job.id, "analysis")


def test_recovery_dashboard_ignores_unrelated_stale_global_lock(tmp_path):
    runtime, job = _job_with_plan(tmp_path)
    now = time.time()
    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute(
            "INSERT INTO locks (lock_name, owner, lease_until) VALUES (?, ?, ?)",
            ("local_validation#0", "other-job", now - 60),
        )

    payload = build_job_recovery_dashboard(runtime, LibraryStore(tmp_path), job.id)

    assert payload["locks"] == []
    assert payload["stuck"] is False


def test_runtime_recovery_actions_reject_non_applicable_completed_job(tmp_path):
    runtime, job = _job_with_plan(tmp_path)
    runtime.complete_video_job(job.id)

    with pytest.raises(ValueError, match="cancel not applicable"):
        runtime.cancel_job(job.id)
    with pytest.raises(ValueError, match="resume not applicable"):
        runtime.resume_job(job.id)
    with pytest.raises(ValueError, match="retry stage not applicable"):
        runtime.retry_stage(job.id, "analysis_translation")
    with pytest.raises(ValueError, match="reset downstream not applicable"):
        runtime.reset_downstream(job.id, "analysis_translation")
