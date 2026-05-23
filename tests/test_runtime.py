import json
import sqlite3
import time
from dataclasses import replace
from pathlib import Path

import pytest

from yt_sub_hub.config import get_settings
from yt_sub_hub.library import LibraryStore
from yt_sub_hub.models import JobRecord
from yt_sub_hub.orchestration import recover_interrupted_legacy_jobs
from yt_sub_hub.runtime import (
    DEFAULT_CONCURRENCY_LIMITS,
    RuntimeStore,
)


def _stage_by_name(store: RuntimeStore, job_id: str, stage_name: str):
    return [stage for stage in store.list_stage_plan(job_id) if stage.stage_name == stage_name][0]


def test_runtime_store_creates_queue_stage_artifact_lock_and_retry_tables(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    store.ensure_db()

    with sqlite3.connect(store.db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
            if row[0]
        }
        stage_plan_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(stage_plans)").fetchall()
        }
        video_job_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(video_jobs)").fetchall()
        }
        channel_job_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(channel_jobs)").fetchall()
        }

    assert {
        "channel_jobs",
        "video_jobs",
        "stage_runs",
        "stage_artifacts",
        "locks",
        "retries",
    } <= tables
    assert "idx_video_jobs_status_priority_updated" in indexes
    assert "idx_video_jobs_identity" in indexes
    assert "idx_stage_artifacts_video_type_language_status" in indexes
    assert "stage_payload_json" in stage_plan_columns
    assert "job_kind" in video_job_columns
    assert "quality_mode" in video_job_columns
    assert "acceptance_mode" in video_job_columns
    assert "acceptance_mode" in channel_job_columns


def test_video_job_stores_and_exposes_quality_mode(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")

    job = store.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
        quality_mode="best_effort",
    )

    loaded = store.load_video_job(job.id)
    assert loaded.quality_mode == "best_effort"
    assert loaded.to_dict()["quality_mode"] == "best_effort"


def test_video_job_defaults_quality_mode_to_strict(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")

    job = store.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )

    assert job.quality_mode == "strict"


def test_video_and_channel_jobs_store_acceptance_mode(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")

    channel = store.create_channel_job(
        channel_url="https://www.youtube.com/channel/UCexample123",
        channel_id="UCexample123",
        acceptance_mode="full_processing",
    )
    job = store.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
        channel_job_id=channel.id,
        acceptance_mode="full_processing",
    )

    assert store.load_channel_job(channel.id).acceptance_mode == "full_processing"
    assert store.load_video_job(job.id).acceptance_mode == "full_processing"
    assert store.load_video_job(job.id).to_dict()["acceptance_mode"] == "full_processing"


def test_mark_video_job_skipped_requires_media_marks_downstream_skipped(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    job = store.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
        acceptance_mode="data_api_only",
    )
    store.ensure_stage_plan(job.id, enable_deep_graph_llm=True)

    skipped = store.mark_video_job_skipped_requires_media(job.id)
    stages = {stage.stage_name: stage for stage in store.list_stage_plan(job.id)}

    assert skipped.status == "skipped"
    assert skipped.current_stage == "transcript_source"
    assert skipped.warning == "skipped_requires_media"
    assert stages["transcript_source"].status == "skipped"
    assert stages["transcript_source"].warning == "skipped_requires_media"
    assert stages["audio_download"].status == "skipped"
    assert stages["whisper_transcription"].status == "skipped"


def test_force_retry_skipped_requires_media_resets_downstream_skips_with_same_reason(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    job = store.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
        acceptance_mode="data_api_only",
    )
    store.ensure_stage_plan(job.id, enable_deep_graph_llm=True)
    store.mark_video_job_skipped_requires_media(job.id)

    retried = store.retry_video_job_for_channel(job.id)
    stages = {stage.stage_name: stage for stage in store.list_stage_plan(job.id)}

    assert retried.status == "queued"
    assert stages["transcript_source"].status == "pending"
    assert stages["audio_download"].status == "pending"
    assert stages["deep_graph_llm"].status == "pending"
    assert stages["neo4j_sync"].status == "pending"


def test_runtime_migrates_existing_video_jobs_to_strict_quality_mode(tmp_path):
    db_path = tmp_path / "index.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE video_jobs (
                id TEXT PRIMARY KEY,
                channel_job_id TEXT,
                video_id TEXT NOT NULL,
                canonical_url TEXT NOT NULL,
                status TEXT NOT NULL,
                current_stage TEXT,
                priority INTEGER NOT NULL DEFAULT 100,
                lease_owner TEXT,
                lease_until REAL,
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                warning TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                completed_at REAL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO video_jobs
              (id, video_id, canonical_url, status, priority, cancel_requested, created_at, updated_at)
            VALUES ('legacy-job', 'abc123def45', 'https://www.youtube.com/watch?v=abc123def45',
                    'queued', 100, 0, 1.0, 1.0)
            """
        )

    store = RuntimeStore(db_path)
    loaded = store.load_video_job("legacy-job")

    assert loaded.quality_mode == "strict"


def test_runtime_stage_plan_uses_translation_expand_and_merge_barrier(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    job = store.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )

    plan = store.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    names = [stage.stage_name for stage in plan]

    assert "transcript_translation" not in names
    assert names[names.index("whisper_transcription") + 1] == "transcript_translation_expand"
    assert names[names.index("transcript_translation_expand") + 1] == "transcript_translation_merge"
    assert names[names.index("transcript_translation_merge") + 1] == "scientific_analysis"
    assert names[names.index("scientific_analysis") + 1] == "analysis_reduce"
    assert names[names.index("analysis_reduce") + 1] == "analysis_translation"
    assert names[names.index("analysis_translation") + 1] == "analysis_ru_review"
    assert names[names.index("analysis_ru_review") + 1] == "markdown_render"
    assert names[names.index("visual_frame_analysis") + 1] == "ru_artifact_review"
    assert names[names.index("ru_artifact_review") + 1] == "neo4j_sync"
    assert _stage_by_name(store, job.id, "transcript_translation_merge").resource_lock == "local_validation"
    assert _stage_by_name(store, job.id, "analysis_reduce").resource_lock == "local_validation"
    assert _stage_by_name(store, job.id, "analysis_ru_review").resource_lock == "local_validation"
    assert _stage_by_name(store, job.id, "ru_artifact_review").resource_lock == "critic_review"


def test_queue_youtube_discovery_creates_discovery_stage_plan(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")

    job = store.queue_youtube_discovery("srcVideo123", limit=20, max_quota_units=900)
    stages = store.list_stage_plan(job.id)

    assert job.video_id == "srcVideo123"
    assert job.job_kind == "youtube_discovery"
    assert job.canonical_url == "https://www.youtube.com/watch?v=srcVideo123"
    assert [stage.stage_name for stage in stages] == [
        "youtube_discovery_seed_extract",
        "youtube_discovery_query_expand",
        "youtube_discovery_candidate_fetch",
        "youtube_discovery_candidate_score",
        "youtube_discovery_review",
        "youtube_discovery_graph",
        "youtube_discovery_report",
    ]
    assert _stage_by_name(store, job.id, "youtube_discovery_seed_extract").resource_lock == "local_validation"
    assert _stage_by_name(store, job.id, "youtube_discovery_candidate_score").resource_lock == "local_candidate_scoring"
    assert _stage_by_name(store, job.id, "youtube_discovery_review").resource_lock == "local_validation"
    payload = json.loads(_stage_by_name(store, job.id, "youtube_discovery_seed_extract").stage_payload_json)
    assert payload["limit"] == 20
    assert payload["max_quota_units"] == 900


def test_queue_youtube_discovery_does_not_mutate_completed_core_video_job(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    core = store.create_video_job(
        video_id="srcVideo123",
        canonical_url="https://www.youtube.com/watch?v=srcVideo123",
    )
    store.ensure_stage_plan(core.id, enable_deep_graph_llm=False)
    now = time.time()
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE video_jobs SET status = 'completed', current_stage = 'quality_gate', completed_at = ?, updated_at = ? WHERE id = ?",
            (now, now, core.id),
        )

    discovery = store.queue_youtube_discovery("srcVideo123", limit=5, max_quota_units=300)
    loaded_core = store.load_video_job(core.id)

    assert loaded_core.status == "completed"
    assert loaded_core.current_stage == "quality_gate"
    assert loaded_core.job_kind == "video_processing"
    assert discovery.id != core.id
    assert discovery.job_kind == "youtube_discovery"
    assert discovery.status == "queued"
    assert discovery.current_stage == "youtube_discovery_seed_extract"


def test_create_video_job_separates_job_kind(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")

    core = store.create_video_job(
        video_id="srcVideo123",
        canonical_url="https://www.youtube.com/watch?v=srcVideo123",
    )
    discovery = store.create_video_job(
        video_id="srcVideo123",
        canonical_url="https://www.youtube.com/watch?v=srcVideo123",
        job_kind="youtube_discovery",
    )

    assert core.id != discovery.id
    assert store._find_video_job("srcVideo123", None, "video_processing").id == core.id
    assert store._find_video_job("srcVideo123", None, "youtube_discovery").id == discovery.id


def test_ensure_core_stage_plan_rejects_discovery_job(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    discovery = store.queue_youtube_discovery("srcVideo123", limit=5, max_quota_units=300)

    with pytest.raises(ValueError, match="core stage plan requires job_kind=video_processing"):
        store.ensure_stage_plan(discovery.id)

    names = [stage.stage_name for stage in store.list_stage_plan(discovery.id)]
    assert "metadata" not in names


def test_ensure_youtube_discovery_stage_plan_rejects_core_job(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    core = store.create_video_job(
        video_id="srcVideo123",
        canonical_url="https://www.youtube.com/watch?v=srcVideo123",
    )

    with pytest.raises(ValueError, match="youtube discovery stage plan requires job_kind=youtube_discovery"):
        store.ensure_youtube_discovery_stage_plan(core.id)

    assert store.list_stage_plan(core.id) == []


def test_runtime_migrates_old_video_jobs_without_job_kind(tmp_path):
    db_path = tmp_path / "index.sqlite3"
    now = time.time()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE video_jobs (
                id TEXT PRIMARY KEY,
                channel_job_id TEXT,
                video_id TEXT NOT NULL,
                canonical_url TEXT NOT NULL,
                status TEXT NOT NULL,
                current_stage TEXT,
                priority INTEGER NOT NULL DEFAULT 100,
                lease_owner TEXT,
                lease_until REAL,
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                warning TEXT,
                quality_mode TEXT NOT NULL DEFAULT 'strict',
                acceptance_mode TEXT NOT NULL DEFAULT 'data_api_only',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                completed_at REAL,
                UNIQUE(video_id, channel_job_id)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO video_jobs
              (id, channel_job_id, video_id, canonical_url, status, current_stage, priority,
               lease_owner, lease_until, cancel_requested, error, warning, quality_mode, acceptance_mode,
               created_at, updated_at, completed_at)
            VALUES (?, NULL, ?, ?, 'completed', 'quality_gate', 100, NULL, NULL, 0, NULL, NULL, 'strict', 'data_api_only', ?, ?, ?)
            """,
            ("legacy-job", "abc123def45", "https://www.youtube.com/watch?v=abc123def45", now, now, now),
        )

    store = RuntimeStore(db_path)
    loaded = store.load_video_job("legacy-job")

    assert loaded.job_kind == "video_processing"
    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(video_jobs)").fetchall()}
    assert "job_kind" in columns


def test_queue_youtube_discovery_without_force_keeps_completed_discovery_job(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    job = store.queue_youtube_discovery("srcVideo123", limit=5, max_quota_units=300)
    now = time.time()
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE video_jobs SET status = 'completed', current_stage = 'youtube_discovery_report', completed_at = ?, updated_at = ? WHERE id = ?",
            (now, now, job.id),
        )
        conn.execute(
            "UPDATE stage_plans SET status = 'completed', completed_at = ?, updated_at = ? WHERE video_job_id = ?",
            (now, now, job.id),
        )

    returned = store.queue_youtube_discovery("srcVideo123", limit=9, max_quota_units=100, force=False)

    assert returned.id == job.id
    assert returned.status == "completed"
    assert returned.current_stage == "youtube_discovery_report"


def test_queue_youtube_discovery_force_requeues_existing_discovery_job(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    job = store.queue_youtube_discovery("srcVideo123", limit=5, max_quota_units=300)
    store.expand_youtube_discovery_query_plan(
        job.id,
        {"items": [{"query_id": "ydq-001", "query": "graph", "seed_ids": ["seed-001"]}]},
    )
    now = time.time()
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE video_jobs SET status = 'completed', current_stage = 'youtube_discovery_report', warning = 'old', completed_at = ?, updated_at = ? WHERE id = ?",
            (now, now, job.id),
        )
        conn.execute(
            "UPDATE stage_plans SET status = 'completed', completed_at = ?, updated_at = ? WHERE video_job_id = ?",
            (now, now, job.id),
        )

    returned = store.queue_youtube_discovery("srcVideo123", limit=9, max_quota_units=100, force=True)
    stages = {stage.stage_name: stage for stage in store.list_stage_plan(job.id)}

    assert returned.id == job.id
    assert returned.status == "queued"
    assert returned.current_stage == "youtube_discovery_seed_extract"
    assert returned.warning is None
    assert "youtube_discovery_query_000" not in stages
    assert stages["youtube_discovery_seed_extract"].status == "pending"
    payload = json.loads(stages["youtube_discovery_seed_extract"].stage_payload_json)
    assert payload["limit"] == 9
    assert payload["max_quota_units"] == 100


def test_force_reset_rejects_stale_running_stage_completion(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    job = store.queue_youtube_discovery("srcVideo123", limit=5, max_quota_units=300)
    claim = store.claim_next_stage(owner="worker-1", lease_seconds=60)
    assert claim.stage.stage_name == "youtube_discovery_seed_extract"

    store.queue_youtube_discovery("srcVideo123", limit=5, max_quota_units=300, force=True)

    with pytest.raises(ValueError, match="stage run is not running"):
        store.complete_stage(claim.stage_run.id)
    assert _stage_by_name(store, job.id, "youtube_discovery_seed_extract").status == "pending"


def test_claim_next_stage_ignores_core_stage_rows_on_discovery_job(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    job = store.queue_youtube_discovery("srcVideo123", limit=5, max_quota_units=300)
    now = time.time()
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            """
            INSERT INTO stage_plans
              (id, video_job_id, stage_name, status, position, attempt, max_attempts, hard_failure,
               resource_lock, resource_slot, lease_owner, lease_until, retry_after, error, warning,
               stage_payload_json, created_at, updated_at, completed_at)
            VALUES ('bad-core-stage', ?, 'metadata', 'pending', -1, 0, 1, 1,
                    'youtube_metadata', NULL, NULL, NULL, NULL, NULL, NULL, NULL, ?, ?, NULL)
            """,
            (job.id, now, now),
        )

    claim = store.claim_next_stage(owner="worker-1", lease_seconds=60)

    assert claim is not None
    assert claim.stage.stage_name == "youtube_discovery_seed_extract"
    assert _stage_by_name(store, job.id, "metadata").status == "pending"


def test_expand_youtube_discovery_query_plan_reconciles_dynamic_query_stages(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    job = store.queue_youtube_discovery("srcVideo123", limit=20, max_quota_units=900)
    queries = {
        "artifact_hash": "queries-v1",
        "input_hashes": {"input": "hash-1"},
        "items": [
            {
                "query_id": "ydq-001",
                "query": "graph provenance",
                "seed_ids": ["seed-001"],
                "intent": "same_topic",
                "language": "ru",
                "query_source": "seed_text",
                "rejected_reason": None,
            },
            {
                "query_id": "ydq-002",
                "query": "graph review",
                "seed_ids": ["seed-002"],
                "intent": "deep_dive",
                "language": "en",
                "query_source": "latin_entity",
                "rejected_reason": None,
            },
            {
                "query_id": "ydq-rejected",
                "query": "unsafe duplicate",
                "seed_ids": ["seed-003"],
                "intent": "same_topic",
                "language": "en",
                "query_source": "seed_text",
                "rejected_reason": "duplicate_query",
            },
        ],
    }

    store.expand_youtube_discovery_query_plan(job.id, queries)
    first = {stage.stage_name: stage for stage in store.list_stage_plan(job.id)}
    assert first["youtube_discovery_query_000"].position == first["youtube_discovery_query_001"].position
    assert first["youtube_discovery_query_000"].resource_lock == "youtube_data_api"
    assert "youtube_discovery_query_002" not in first
    payload = json.loads(first["youtube_discovery_query_000"].stage_payload_json)
    assert payload["query_id"] == "ydq-001"
    assert payload["language"] == "ru"
    assert payload["intent"] == "same_topic"
    assert payload["query_source"] == "seed_text"
    assert payload["queries_artifact_hash"] == "queries-v1"
    assert payload["query"]["language"] == "ru"
    assert "ydq-rejected" not in {
        json.loads(stage.stage_payload_json)["query_id"]
        for stage in first.values()
        if stage.stage_name.startswith("youtube_discovery_query_") and stage.stage_name != "youtube_discovery_query_expand"
    }

    claim = store.claim_next_stage(owner="worker-1", lease_seconds=60)
    assert claim.stage.stage_name == "youtube_discovery_seed_extract"
    store.complete_stage(claim.stage_run.id)
    claim = store.claim_next_stage(owner="worker-1", lease_seconds=60)
    assert claim.stage.stage_name == "youtube_discovery_query_expand"
    store.complete_stage(claim.stage_run.id)
    query_claim = store.claim_next_stage(owner="worker-1", lease_seconds=60)
    assert query_claim.stage.stage_name == "youtube_discovery_query_000"
    store.complete_stage(query_claim.stage_run.id)
    completed_before = _stage_by_name(store, job.id, "youtube_discovery_query_000").completed_at

    store.expand_youtube_discovery_query_plan(job.id, queries)
    unchanged = _stage_by_name(store, job.id, "youtube_discovery_query_000")
    assert unchanged.status == "completed"
    assert unchanged.completed_at == completed_before

    updated_queries = {
        **queries,
        "artifact_hash": "queries-v2",
        "items": [
            {
                "query_id": "ydq-001",
                "query": "changed graph provenance",
                "seed_ids": ["seed-001"],
                "intent": "same_topic",
                "language": "en",
                "query_source": "seed_text",
                "rejected_reason": None,
            },
        ],
    }
    store.expand_youtube_discovery_query_plan(job.id, updated_queries)
    reset = _stage_by_name(store, job.id, "youtube_discovery_query_000")
    assert reset.status == "pending"
    assert reset.attempt == 0
    assert reset.completed_at is None
    assert "youtube_discovery_query_001" not in {stage.stage_name for stage in store.list_stage_plan(job.id)}


def test_expand_youtube_discovery_query_plan_uses_only_executable_queries(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    job = store.queue_youtube_discovery("srcVideo123", limit=20, max_quota_units=900)
    queries = {
        "artifact_hash": "queries-v1",
        "input_hashes": {"input": "hash-1"},
        "items": [
            {
                "query_id": "ydq-active",
                "query": "Mortal Shell 2",
                "seed_ids": ["seed-001"],
                "intent": "same_topic",
                "language": "en",
                "query_source": "latin_entity",
                "rejected_reason": None,
            },
            {
                "query_id": "ydq-rejected",
                "query": "русская строка",
                "seed_ids": ["seed-002"],
                "intent": "same_topic",
                "language": "en",
                "query_source": "seed_text",
                "rejected_reason": "query_language_mismatch",
            },
        ],
    }

    store.expand_youtube_discovery_query_plan(job.id, queries)

    stages = {stage.stage_name: stage for stage in store.list_stage_plan(job.id)}
    assert "youtube_discovery_query_000" in stages
    assert "youtube_discovery_query_001" not in stages
    payload = json.loads(stages["youtube_discovery_query_000"].stage_payload_json)
    assert payload["total_queries"] == 1
    assert payload["query_id"] == "ydq-active"
    assert payload["language"] == "en"
    assert payload["intent"] == "same_topic"
    assert payload["query_source"] == "latin_entity"
    assert payload["queries_artifact_hash"] == "queries-v1"


def test_ensure_stage_plan_migrates_existing_plan_with_analysis_ru_review_barrier(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    job = store.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    store.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    now = time.time()
    with sqlite3.connect(store.db_path) as conn:
        review_row = conn.execute(
            "SELECT position FROM stage_plans WHERE video_job_id = ? AND stage_name = 'analysis_ru_review'",
            (job.id,),
        ).fetchone()
        if review_row is not None:
            old_position = int(review_row[0])
            conn.execute("DELETE FROM stage_plans WHERE video_job_id = ? AND stage_name = 'analysis_ru_review'", (job.id,))
            conn.execute(
                "UPDATE stage_plans SET position = position - 1 WHERE video_job_id = ? AND position > ?",
                (job.id, old_position),
            )
        conn.execute(
            """
            UPDATE stage_plans
            SET status = 'completed', attempt = 1, completed_at = ?, updated_at = ?
            WHERE video_job_id = ? AND stage_name IN ('analysis_translation', 'markdown_render', 'deterministic_graph', 'quality_gate')
            """,
            (now, now, job.id),
        )
        conn.execute("UPDATE video_jobs SET status = 'queued', current_stage = 'markdown_render' WHERE id = ?", (job.id,))

    migrated = store.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    names = [stage.stage_name for stage in migrated]

    assert names[names.index("analysis_translation") + 1] == "analysis_ru_review"
    assert names[names.index("analysis_ru_review") + 1] == "markdown_render"
    assert _stage_by_name(store, job.id, "analysis_translation").status == "completed"
    assert _stage_by_name(store, job.id, "analysis_ru_review").status == "pending"
    assert _stage_by_name(store, job.id, "markdown_render").status == "pending"
    assert _stage_by_name(store, job.id, "deterministic_graph").status == "pending"
    assert _stage_by_name(store, job.id, "quality_gate").status == "pending"


def test_ensure_stage_plan_adds_pending_analysis_ru_review_to_terminal_job_without_requeue(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    job = store.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    store.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    now = time.time()
    with sqlite3.connect(store.db_path) as conn:
        review_row = conn.execute(
            "SELECT position FROM stage_plans WHERE video_job_id = ? AND stage_name = 'analysis_ru_review'",
            (job.id,),
        ).fetchone()
        if review_row is not None:
            old_position = int(review_row[0])
            conn.execute("DELETE FROM stage_plans WHERE video_job_id = ? AND stage_name = 'analysis_ru_review'", (job.id,))
            conn.execute(
                "UPDATE stage_plans SET position = position - 1 WHERE video_job_id = ? AND position > ?",
                (job.id, old_position),
            )
        conn.execute(
            "UPDATE stage_plans SET status = 'completed', attempt = 1, completed_at = ?, updated_at = ? WHERE video_job_id = ?",
            (now, now, job.id),
        )
        conn.execute(
            "UPDATE video_jobs SET status = 'completed', current_stage = 'quality_gate', completed_at = ?, updated_at = ? WHERE id = ?",
            (now, now, job.id),
        )

    store.ensure_stage_plan(job.id, enable_deep_graph_llm=False)

    assert _stage_by_name(store, job.id, "analysis_ru_review").status == "pending"
    assert _stage_by_name(store, job.id, "markdown_render").status == "completed"
    assert store.load_video_job(job.id).status == "completed"
    assert store.list_runnable_stages() == []


def test_ensure_stage_plan_migrates_existing_plan_with_ru_artifact_review_barrier(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    job = store.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    store.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    now = time.time()
    with sqlite3.connect(store.db_path) as conn:
        review_row = conn.execute(
            "SELECT position FROM stage_plans WHERE video_job_id = ? AND stage_name = 'ru_artifact_review'",
            (job.id,),
        ).fetchone()
        old_position = int(review_row[0])
        conn.execute("DELETE FROM stage_plans WHERE video_job_id = ? AND stage_name = 'ru_artifact_review'", (job.id,))
        conn.execute("UPDATE stage_plans SET position = position - 1 WHERE video_job_id = ? AND position > ?", (job.id, old_position))
        conn.execute(
            """
            UPDATE stage_plans
            SET status = 'completed', attempt = 1, completed_at = ?, updated_at = ?
            WHERE video_job_id = ? AND stage_name IN ('neo4j_sync', 'quality_gate')
            """,
            (now, now, job.id),
        )
        conn.execute("UPDATE video_jobs SET status = 'queued', current_stage = 'neo4j_sync' WHERE id = ?", (job.id,))

    migrated = store.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    names = [stage.stage_name for stage in migrated]

    assert names[names.index("visual_frame_analysis") + 1] == "ru_artifact_review"
    assert names[names.index("ru_artifact_review") + 1] == "neo4j_sync"
    assert _stage_by_name(store, job.id, "ru_artifact_review").status == "pending"
    assert _stage_by_name(store, job.id, "neo4j_sync").status == "pending"
    assert _stage_by_name(store, job.id, "quality_gate").status == "pending"
    assert store.load_video_job(job.id).current_stage == "ru_artifact_review"


def test_runtime_expands_translation_chunks_with_payloads_and_grouped_status(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    job = store.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    store.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    chunks = [
        {
            "chunk_index": 0,
            "total_chunks": 2,
            "target_language": "ru",
            "source_segment_start_index": 0,
            "source_segment_end_index": 1,
        },
        {
            "chunk_index": 1,
            "total_chunks": 2,
            "target_language": "ru",
            "source_segment_start_index": 2,
            "source_segment_end_index": 3,
        },
    ]

    expanded = store.expand_translation_chunk_plan(job.id, chunks)

    chunk_stages = [stage for stage in expanded if stage.stage_name.startswith("transcript_translation_chunk_")]
    assert [stage.stage_name for stage in chunk_stages] == [
        "transcript_translation_chunk_000",
        "transcript_translation_chunk_001",
    ]
    assert {stage.position for stage in chunk_stages} == {5}
    assert all(stage.resource_lock == "lm_studio_translation" for stage in chunk_stages)
    assert json.loads(chunk_stages[0].stage_payload_json)["source_segment_end_index"] == 1
    assert _stage_by_name(store, job.id, "transcript_translation_merge").position == 6
    status = store.stage_queue_status()
    assert status["stage_counts"]["transcript_translation_chunk_000:pending"] == 1
    assert status["stage_group_counts"]["transcript_translation_chunks:pending"] == 2


def test_runtime_groups_future_analysis_chunk_stage_status(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    job = store.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    store.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    now = time.time()
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            """
            INSERT INTO stage_plans
              (id, video_job_id, stage_name, status, position, attempt, max_attempts, hard_failure,
               resource_lock, created_at, updated_at)
            VALUES ('analysis-chunk-test', ?, 'analysis_chunk_000', 'pending', 7, 0, 2, 1,
                    'lm_studio_analysis', ?, ?)
            """,
            (job.id, now, now),
        )

    status = store.stage_queue_status()

    assert status["stage_counts"]["analysis_chunk_000:pending"] == 1
    assert status["stage_group_counts"]["analysis_chunks:pending"] == 1


def test_runtime_expands_analysis_chunks_before_reduce_with_payloads_and_grouped_status(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    job = store.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    store.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    chunks = [
        {
            "chunk_index": 0,
            "total_chunks": 2,
            "source_transcript": "transcript.ru.json",
            "source_segment_start_index": 0,
            "source_segment_end_index": 1,
            "source_transcript_hash": "hash-a",
            "language": "ru",
        },
        {
            "chunk_index": 1,
            "total_chunks": 2,
            "source_transcript": "transcript.ru.json",
            "source_segment_start_index": 2,
            "source_segment_end_index": 3,
            "source_transcript_hash": "hash-a",
            "language": "ru",
        },
    ]

    expanded = store.expand_analysis_chunk_plan(job.id, chunks)

    chunk_stages = [stage for stage in expanded if stage.stage_name.startswith("analysis_chunk_")]
    assert [stage.stage_name for stage in chunk_stages] == ["analysis_chunk_000", "analysis_chunk_001"]
    assert {stage.position for stage in chunk_stages} == {
        _stage_by_name(store, job.id, "scientific_analysis").position + 1
    }
    assert all(stage.resource_lock == "lm_studio_analysis" for stage in chunk_stages)
    assert json.loads(chunk_stages[0].stage_payload_json)["source_transcript"] == "transcript.ru.json"
    assert _stage_by_name(store, job.id, "analysis_reduce").position == chunk_stages[0].position + 1
    status = store.stage_queue_status()
    assert status["stage_counts"]["analysis_chunk_000:pending"] == 1
    assert status["stage_group_counts"]["analysis_chunks:pending"] == 2


def test_runtime_reexpanding_analysis_chunks_resets_stale_completed_chunks(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    job = store.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    store.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    store.expand_analysis_chunk_plan(
        job.id,
        [
            {"chunk_index": 0, "source_segment_start_index": 0, "source_segment_end_index": 0, "source_transcript_hash": "hash-a"},
            {"chunk_index": 1, "source_segment_start_index": 1, "source_segment_end_index": 1, "source_transcript_hash": "hash-a"},
        ],
    )
    now = time.time()
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE stage_plans SET status = 'completed', attempt = 1, completed_at = ?, updated_at = ? WHERE video_job_id = ? AND stage_name like 'analysis_chunk_%'",
            (now, now, job.id),
        )

    store.expand_analysis_chunk_plan(
        job.id,
        [
            {"chunk_index": 0, "source_segment_start_index": 0, "source_segment_end_index": 0, "source_transcript_hash": "hash-b"},
            {"chunk_index": 1, "source_segment_start_index": 1, "source_segment_end_index": 1, "source_transcript_hash": "hash-a"},
        ],
    )

    changed = _stage_by_name(store, job.id, "analysis_chunk_000")
    unchanged = _stage_by_name(store, job.id, "analysis_chunk_001")
    assert changed.status == "pending"
    assert changed.attempt == 0
    assert changed.completed_at is None
    assert json.loads(changed.stage_payload_json)["source_transcript_hash"] == "hash-b"
    assert unchanged.status == "completed"
    assert unchanged.attempt == 1


def test_analysis_reduce_not_runnable_until_all_analysis_chunks_complete(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    job = store.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    store.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    store.expand_analysis_chunk_plan(
        job.id,
        [
            {"chunk_index": 0, "source_segment_start_index": 0, "source_segment_end_index": 0, "source_transcript_hash": "hash-a"},
            {"chunk_index": 1, "source_segment_start_index": 1, "source_segment_end_index": 1, "source_transcript_hash": "hash-a"},
        ],
    )
    now = time.time()
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE stage_plans SET status = 'completed', attempt = 1, completed_at = ?, updated_at = ? WHERE video_job_id = ? AND position < (SELECT position FROM stage_plans WHERE video_job_id = ? AND stage_name = 'analysis_chunk_000')",
            (now, now, job.id, job.id),
        )
        conn.execute(
            "UPDATE stage_plans SET status = 'completed', attempt = 1, completed_at = ?, updated_at = ? WHERE video_job_id = ? AND stage_name = 'analysis_chunk_000'",
            (now, now, job.id),
        )

    assert [stage.stage_name for stage in store.list_runnable_stages()] == ["analysis_chunk_001"]

    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE stage_plans SET status = 'completed', attempt = 1, completed_at = ?, updated_at = ? WHERE video_job_id = ? AND stage_name = 'analysis_chunk_001'",
            (now, now, job.id),
        )

    assert [stage.stage_name for stage in store.list_runnable_stages()] == ["analysis_reduce"]


def test_runtime_store_idempotently_creates_and_claims_video_jobs(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    channel = store.create_channel_job(
        channel_url="https://www.youtube.com/@example",
        channel_id="example",
        requested_limit=2,
    )
    first = store.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
        channel_job_id=channel.id,
    )
    second = store.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
        channel_job_id=channel.id,
    )

    claimed = store.claim_next_video_job(owner="worker-1", lease_seconds=60)
    blocked = store.claim_next_video_job(owner="worker-2", lease_seconds=60)

    assert second.id == first.id
    assert claimed is not None
    assert claimed.id == first.id
    assert claimed.status == "running"
    assert claimed.lease_owner == "worker-1"
    assert blocked is None
    assert store.queue_status()["running"] == 1


def test_channel_job_status_recompute_counts_children(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    channel = store.create_channel_job(
        channel_url="https://www.youtube.com/channel/UCexample123",
        channel_id="UCexample123",
        requested_limit=3,
        quality_mode="draft",
    )
    completed = store.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
        channel_job_id=channel.id,
        quality_mode="draft",
    )
    failed = store.create_video_job(
        video_id="xyz123def45",
        canonical_url="https://www.youtube.com/watch?v=xyz123def45",
        channel_job_id=channel.id,
        quality_mode="draft",
    )
    queued = store.create_video_job(
        video_id="qwe123def45",
        canonical_url="https://www.youtube.com/watch?v=qwe123def45",
        channel_job_id=channel.id,
        quality_mode="draft",
    )
    store.complete_video_job(completed.id)
    store.fail_video_job(failed.id, error="boom")

    running = store.recompute_channel_job_status(channel.id)
    store.skip_video_job(queued.id, warning="already processed")
    warning = store.recompute_channel_job_status(channel.id)

    assert running.status == "running"
    assert running.queued_count == 1
    assert warning.status == "completed_with_warning"
    assert warning.video_count == 3
    assert warning.completed_count == 1
    assert warning.failed_count == 1
    assert warning.skipped_count == 1
    assert warning.quality_mode == "draft"


def test_cancel_channel_marks_non_terminal_child_jobs(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    channel = store.create_channel_job(channel_url="https://www.youtube.com/channel/UCexample123", channel_id="UCexample123")
    completed = store.create_video_job(video_id="abc123def45", canonical_url="https://www.youtube.com/watch?v=abc123def45", channel_job_id=channel.id)
    queued = store.create_video_job(video_id="xyz123def45", canonical_url="https://www.youtube.com/watch?v=xyz123def45", channel_job_id=channel.id)
    store.complete_video_job(completed.id)

    cancelled = store.cancel_channel_job(channel.id)

    assert store.load_video_job(completed.id).cancel_requested is False
    assert store.load_video_job(queued.id).cancel_requested is True
    assert store.load_video_job(queued.id).status == "skipped"
    assert cancelled.status == "completed_with_warning"


def test_retry_channel_failed_requeues_failed_children_only(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    channel = store.create_channel_job(channel_url="https://www.youtube.com/channel/UCexample123", channel_id="UCexample123")
    completed = store.create_video_job(video_id="abc123def45", canonical_url="https://www.youtube.com/watch?v=abc123def45", channel_job_id=channel.id)
    failed = store.create_video_job(video_id="xyz123def45", canonical_url="https://www.youtube.com/watch?v=xyz123def45", channel_job_id=channel.id)
    for job in [completed, failed]:
        store.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    store.complete_video_job(completed.id)
    now = time.time()
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE stage_plans SET status = 'failed', attempt = max_attempts, error = 'boom', completed_at = ?, updated_at = ? WHERE video_job_id = ? AND stage_name = 'metadata'",
            (now, now, failed.id),
        )
        conn.execute("UPDATE video_jobs SET status = 'failed', error = 'boom', completed_at = ?, updated_at = ? WHERE id = ?", (now, now, failed.id))

    retried = store.retry_channel_failed(channel.id)

    assert retried.status == "running"
    assert store.load_video_job(completed.id).status == "completed"
    assert store.load_video_job(failed.id).status == "queued"
    assert _stage_by_name(store, failed.id, "metadata").status == "pending"


def test_retry_channel_after_cancel_resets_skipped_stage_plans(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    channel = store.create_channel_job(channel_url="https://www.youtube.com/channel/UCexample123", channel_id="UCexample123")
    job = store.create_video_job(video_id="abc123def45", canonical_url="https://www.youtube.com/watch?v=abc123def45", channel_job_id=channel.id)
    store.ensure_stage_plan(job.id, enable_deep_graph_llm=False)

    store.cancel_channel_job(channel.id)
    retried = store.retry_channel_failed(channel.id)

    assert retried.status == "running"
    assert store.load_video_job(job.id).status == "queued"
    assert _stage_by_name(store, job.id, "metadata").status == "pending"
    assert [stage.stage_name for stage in store.list_runnable_stages()] == ["metadata"]


def test_runtime_store_claims_specific_video_job_without_claiming_earlier_queue_item(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    first = store.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
        priority=1,
    )
    second = store.create_video_job(
        video_id="xyz123def45",
        canonical_url="https://www.youtube.com/watch?v=xyz123def45",
        priority=100,
    )

    claimed = store.claim_video_job(second.id, owner="api-background", lease_seconds=60)

    assert claimed is not None
    assert claimed.id == second.id
    assert store.load_video_job(first.id).status == "queued"
    assert store.load_video_job(second.id).status == "running"


def test_runtime_store_recovers_expired_leases_and_honors_cancellation(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    job = store.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    claimed = store.claim_next_video_job(owner="worker-1", lease_seconds=1)
    assert claimed is not None
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE video_jobs SET lease_until = ? WHERE id = ?",
            (time.time() - 10, job.id),
        )

    recovered = store.claim_next_video_job(owner="worker-2", lease_seconds=60)
    store.request_cancel(job.id)

    assert recovered is not None
    assert recovered.lease_owner == "worker-2"
    assert store.load_video_job(job.id).cancel_requested is True


def test_runtime_store_cancelled_queued_job_becomes_terminal_and_not_claimed(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    job = store.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )

    cancelled = store.request_cancel(job.id)
    claimed = store.claim_next_video_job(owner="worker-1", lease_seconds=60)

    assert cancelled.status == "skipped"
    assert cancelled.cancel_requested is True
    assert claimed is None
    assert store.queue_status()["skipped"] == 1


def test_runtime_store_complete_video_job_with_warning_uses_warning_status(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    job = store.create_video_job(
        video_id="warn123def45",
        canonical_url="https://www.youtube.com/watch?v=warn123def45",
    )

    completed = store.complete_video_job(job.id, warning="quality gate warning")

    assert completed.status == "completed_with_warning"
    assert completed.warning == "quality gate warning"
    assert store.queue_status()["completed_with_warning"] == 1


def test_runtime_store_queue_status_counts_all_runtime_terminal_states(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    queued = store.create_video_job(video_id="queued1", canonical_url="https://www.youtube.com/watch?v=queued1")
    running = store.create_video_job(video_id="running1", canonical_url="https://www.youtube.com/watch?v=running1")
    completed = store.create_video_job(video_id="done123", canonical_url="https://www.youtube.com/watch?v=done123")
    warned = store.create_video_job(video_id="warn123", canonical_url="https://www.youtube.com/watch?v=warn123")
    failed = store.create_video_job(video_id="fail123", canonical_url="https://www.youtube.com/watch?v=fail123")
    skipped = store.create_video_job(video_id="skip123", canonical_url="https://www.youtube.com/watch?v=skip123")

    assert store.claim_video_job(running.id, owner="worker-1", lease_seconds=60) is not None
    store.complete_video_job(completed.id)
    store.complete_video_job(warned.id, warning="quality warning")
    store.fail_video_job(failed.id, error="quality failure")
    store.skip_video_job(skipped.id, warning="cancel requested")

    assert store.queue_status() == {
        "queued": 1,
        "running": 1,
        "completed": 1,
        "completed_with_warning": 1,
        "failed": 1,
        "skipped": 1,
    }
    assert store.load_video_job(queued.id).status == "queued"


def test_runtime_queue_status_does_not_count_missing_or_expired_lease_as_active_running(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    missing_lease = store.create_video_job(
        video_id="missing123",
        canonical_url="https://www.youtube.com/watch?v=missing123",
    )
    expired = store.create_video_job(
        video_id="expired123",
        canonical_url="https://www.youtube.com/watch?v=expired123",
    )
    active = store.create_video_job(
        video_id="active123",
        canonical_url="https://www.youtube.com/watch?v=active123",
    )
    assert store.claim_video_job(missing_lease.id, owner="worker-1", lease_seconds=60) is not None
    assert store.claim_video_job(expired.id, owner="worker-1", lease_seconds=60) is not None
    assert store.claim_video_job(active.id, owner="worker-1", lease_seconds=60) is not None
    with sqlite3.connect(store.db_path) as conn:
        conn.execute("UPDATE video_jobs SET lease_owner = NULL, lease_until = NULL WHERE id = ?", (missing_lease.id,))
        conn.execute("UPDATE video_jobs SET lease_until = ? WHERE id = ?", (time.time() - 10, expired.id))

    status = store.queue_status()

    assert status["running"] == 1
    assert status["queued"] == 2


def test_recovery_marks_legacy_running_failed_when_runtime_is_terminal(tmp_path):
    legacy = LibraryStore(tmp_path)
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    legacy.save_job(
        JobRecord(
            job_id="legacy-job",
            url="https://www.youtube.com/watch?v=abc123def45",
            video_id="abc123def45",
            status="running",
            stage="transcript_resolved",
        )
    )
    runtime.create_video_job(
        job_id="legacy-job",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
        status="queued",
    )
    runtime.fail_video_job("legacy-job", error="worker stopped")

    recovered = recover_interrupted_legacy_jobs(legacy, runtime, stale_after_seconds=1800)

    loaded = legacy.load_jobs()["legacy-job"]
    assert [record.job_id for record in recovered] == ["legacy-job"]
    assert loaded.status == "failed"
    assert loaded.stage == "failed"
    assert loaded.error == "Interrupted worker run recovered; previous stage: transcript_resolved"


def test_recovery_preserves_legacy_running_when_runtime_lease_is_active(tmp_path):
    legacy = LibraryStore(tmp_path)
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    legacy.save_job(
        JobRecord(
            job_id="legacy-job",
            url="https://www.youtube.com/watch?v=abc123def45",
            video_id="abc123def45",
            status="running",
            stage="metadata_fetched",
        )
    )
    runtime.create_video_job(
        job_id="legacy-job",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
        status="queued",
    )
    assert runtime.claim_video_job("legacy-job", owner="worker-1", lease_seconds=60) is not None

    recovered = recover_interrupted_legacy_jobs(legacy, runtime, stale_after_seconds=1)

    assert recovered == []
    loaded = legacy.load_jobs()["legacy-job"]
    assert loaded.status == "running"
    assert loaded.stage == "metadata_fetched"


def test_recovery_is_idempotent_for_interrupted_legacy_job(tmp_path):
    legacy = LibraryStore(tmp_path)
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    legacy.save_job(
        JobRecord(
            job_id="legacy-job",
            url="https://www.youtube.com/watch?v=abc123def45",
            video_id="abc123def45",
            status="running",
            stage="transcript_resolved",
        )
    )
    runtime.create_video_job(
        job_id="legacy-job",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
        status="queued",
    )
    runtime.fail_video_job("legacy-job", error="worker stopped")

    first = recover_interrupted_legacy_jobs(legacy, runtime, stale_after_seconds=1800)
    history_after_first = legacy.load_job_history("legacy-job")
    second = recover_interrupted_legacy_jobs(legacy, runtime, stale_after_seconds=1800)

    assert len(first) == 1
    assert second == []
    assert legacy.load_job_history("legacy-job") == history_after_first


def test_runtime_store_records_stage_runs_artifacts_and_retries(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    job = store.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    stage = store.start_stage_run(
        video_job_id=job.id,
        stage_name="metadata",
        input_hash="input-v1",
    )
    artifact_path = tmp_path / "videos" / "abc123def45" / "metadata.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text('{"video_id":"abc123def45"}', encoding="utf-8")

    artifact = store.record_stage_artifact(
        stage_run_id=stage.id,
        video_id="abc123def45",
        artifact_type="metadata",
        path=artifact_path,
        language=None,
        provenance={"stage": "metadata"},
    )
    store.complete_stage_run(stage.id, output_hash="output-v1")
    retry = store.schedule_retry(
        stage_run_id=stage.id,
        attempt=2,
        reason="rate_limit",
        backoff_seconds=30,
    )

    loaded_artifacts = store.list_stage_artifacts(video_id="abc123def45")
    loaded_retries = store.list_retries(stage_run_id=stage.id)

    assert artifact.status == "committed"
    assert artifact.content_hash
    assert json.loads(artifact.provenance_json) == {"stage": "metadata"}
    assert store.load_stage_run(stage.id).status == "completed"
    assert loaded_artifacts[0].id == artifact.id
    assert loaded_retries[0].id == retry.id


def test_runtime_reset_failed_stage_plan_requeues_first_failed_stage_and_downstream(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    job = store.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    store.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    now = time.time()
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            """
            UPDATE stage_plans
            SET status = 'completed', attempt = 1, completed_at = ?, updated_at = ?
            WHERE video_job_id = ? AND stage_name = 'metadata'
            """,
            (now, now, job.id),
        )
        conn.execute(
            """
            UPDATE stage_plans
            SET status = 'failed', attempt = max_attempts, retry_after = ?, error = ?,
                warning = ?, lease_owner = ?, lease_until = ?, resource_slot = ?,
                completed_at = ?, updated_at = ?
            WHERE video_job_id = ? AND stage_name = 'transcript_source'
            """,
            (now + 60, "max attempts exceeded", "old warning", "old-owner", now + 60, "youtube_metadata#0", now, now, job.id),
        )
        conn.execute(
            "UPDATE video_jobs SET status = 'failed', current_stage = 'transcript_source', error = ? WHERE id = ?",
            ("max attempts exceeded", job.id),
        )

    record = store.reset_failed_stage_plan_for_retry(job.id)

    metadata = _stage_by_name(store, job.id, "metadata")
    transcript = _stage_by_name(store, job.id, "transcript_source")
    audio = _stage_by_name(store, job.id, "audio_download")
    deep_graph = _stage_by_name(store, job.id, "deep_graph_llm")
    assert record.status == "queued"
    assert record.current_stage == "transcript_source"
    assert metadata.status == "completed"
    assert transcript.status == "pending"
    assert transcript.attempt == 0
    assert transcript.retry_after is None
    assert transcript.error is None
    assert transcript.warning is None
    assert transcript.lease_owner is None
    assert transcript.lease_until is None
    assert transcript.resource_slot is None
    assert transcript.completed_at is None
    assert audio.status == "pending"
    assert deep_graph.status == "skipped"


def test_expiring_stale_stage_claim_closes_running_stage_run(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    job = store.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    store.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    claim = store.claim_next_stage(owner="worker-1", lease_seconds=60)
    assert claim is not None

    with sqlite3.connect(store.db_path) as conn:
        conn.execute("UPDATE stage_plans SET lease_until = ? WHERE id = ?", (time.time() - 10, claim.stage.id))

    expired = store.expire_stale_stage_claims()

    loaded_run = store.load_stage_run(claim.stage_run.id)
    loaded_stage = store.load_stage_plan(claim.stage.id)
    assert expired == 1
    assert loaded_stage.status == "pending"
    assert loaded_run.status == "failed"
    assert loaded_run.error_class == "RuntimeLeaseExpired"
    assert loaded_run.error_message == "stage lease expired"


def test_expiring_stale_stage_claim_closes_orphan_running_stage_run(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    job = store.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    store.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    claim = store.claim_next_stage(owner="worker-1", lease_seconds=60)
    assert claim is not None

    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            """
            UPDATE stage_plans
            SET status = 'pending', lease_owner = NULL, lease_until = NULL, resource_slot = NULL
            WHERE id = ?
            """,
            (claim.stage.id,),
        )

    expired = store.expire_stale_stage_claims()

    loaded_run = store.load_stage_run(claim.stage_run.id)
    assert expired == 1
    assert loaded_run.status == "failed"
    assert loaded_run.error_class == "RuntimeLeaseExpired"
    assert loaded_run.error_message == "stage lease expired"


def test_expiring_stage_claim_closes_dead_local_worker_owner_before_lease_deadline(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    job = store.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    store.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    claim = store.claim_next_stage(owner="local-stage-worker-99999999-0-dead", lease_seconds=3600)
    assert claim is not None

    expired = store.expire_stale_stage_claims()

    loaded_run = store.load_stage_run(claim.stage_run.id)
    loaded_stage = store.load_stage_plan(claim.stage.id)
    assert expired == 1
    assert loaded_stage.status == "pending"
    assert loaded_run.status == "failed"
    assert loaded_run.error_class == "RuntimeLeaseExpired"


def test_runtime_renews_active_stage_claim_lease(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    job = store.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    store.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    claim = store.claim_next_stage(owner="worker-1", lease_seconds=60)
    assert claim is not None
    original_lease = claim.stage.lease_until

    renewed = store.renew_stage_claim(claim.stage.id, owner="worker-1", lease_seconds=300)

    stage = store.load_stage_plan(claim.stage.id)
    job = store.load_video_job(job.id)
    assert renewed is True
    assert original_lease is not None
    assert stage.lease_until is not None
    assert stage.lease_until > original_lease
    assert job.lease_until == stage.lease_until


def test_runtime_reset_failed_stage_plan_from_later_stage_preserves_transcript_work(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    job = store.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    store.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    now = time.time()
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            """
            UPDATE stage_plans
            SET status = 'completed', attempt = 1, completed_at = ?, updated_at = ?
            WHERE video_job_id = ? AND position < (
                SELECT position FROM stage_plans WHERE video_job_id = ? AND stage_name = 'scientific_analysis'
            )
            """,
            (now, now, job.id, job.id),
        )
        conn.execute(
            """
            UPDATE stage_plans
            SET status = 'failed', attempt = max_attempts, error = ?, completed_at = ?, updated_at = ?
            WHERE video_job_id = ? AND stage_name = 'scientific_analysis'
            """,
            ("analysis failed", now, now, job.id),
        )
        conn.execute(
            "UPDATE video_jobs SET status = 'failed', current_stage = 'scientific_analysis', error = ? WHERE id = ?",
            ("analysis failed", job.id),
        )

    record = store.reset_failed_stage_plan_for_retry(job.id)

    assert record.status == "queued"
    assert record.current_stage == "scientific_analysis"
    assert _stage_by_name(store, job.id, "transcript_source").status == "completed"
    assert _stage_by_name(store, job.id, "audio_download").status == "completed"
    assert _stage_by_name(store, job.id, "whisper_transcription").status == "completed"
    assert _stage_by_name(store, job.id, "scientific_analysis").status == "pending"
    assert _stage_by_name(store, job.id, "markdown_render").status == "pending"
    assert _stage_by_name(store, job.id, "deep_graph_llm").status == "skipped"


def test_markdown_render_not_runnable_until_analysis_ru_review_succeeds(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    job = store.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    store.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    now = time.time()
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            """
            UPDATE stage_plans
            SET status = 'completed', attempt = 1, completed_at = ?, updated_at = ?
            WHERE video_job_id = ? AND position < (
                SELECT position FROM stage_plans WHERE video_job_id = ? AND stage_name = 'analysis_ru_review'
            )
            """,
            (now, now, job.id, job.id),
        )

    runnable = store.list_runnable_stages()
    assert [stage.stage_name for stage in runnable] == ["analysis_ru_review"]

    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            """
            UPDATE stage_plans
            SET status = 'failed', attempt = max_attempts, error = ?, completed_at = ?, updated_at = ?
            WHERE video_job_id = ? AND stage_name = 'analysis_ru_review'
            """,
            ("analysis.ru.json must exist", now, now, job.id),
        )
        conn.execute("UPDATE video_jobs SET status = 'failed', error = ? WHERE id = ?", ("analysis.ru.json must exist", job.id))

    assert all(stage.stage_name != "markdown_render" for stage in store.list_runnable_stages())


def test_runtime_retry_failed_analysis_ru_review_preserves_transcript_translation_work(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    job = store.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    store.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    store.expand_translation_chunk_plan(
        job.id,
        [
            {"chunk_index": 0, "total_chunks": 2, "target_language": "ru", "source_segment_start_index": 0, "source_segment_end_index": 0},
            {"chunk_index": 1, "total_chunks": 2, "target_language": "ru", "source_segment_start_index": 1, "source_segment_end_index": 1},
        ],
    )
    now = time.time()
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            """
            UPDATE stage_plans
            SET status = 'completed', attempt = 1, completed_at = ?, updated_at = ?
            WHERE video_job_id = ? AND position < (
                SELECT position FROM stage_plans WHERE video_job_id = ? AND stage_name = 'analysis_ru_review'
            )
            """,
            (now, now, job.id, job.id),
        )
        conn.execute(
            """
            UPDATE stage_plans
            SET status = 'failed', attempt = max_attempts, error = ?, completed_at = ?, updated_at = ?
            WHERE video_job_id = ? AND stage_name = 'analysis_ru_review'
            """,
            ("analysis.ru.json abstract ru_word_ratio 0.20 below 0.70", now, now, job.id),
        )
        conn.execute(
            """
            UPDATE stage_plans
            SET status = 'completed', attempt = 1, completed_at = ?, updated_at = ?
            WHERE video_job_id = ? AND position > (
                SELECT position FROM stage_plans WHERE video_job_id = ? AND stage_name = 'analysis_ru_review'
            ) AND status != 'skipped'
            """,
            (now, now, job.id, job.id),
        )
        conn.execute("UPDATE video_jobs SET status = 'failed', error = ? WHERE id = ?", ("analysis review failed", job.id))

    record = store.reset_failed_stage_plan_for_retry(job.id)

    assert record.status == "queued"
    assert record.current_stage == "analysis_ru_review"
    assert _stage_by_name(store, job.id, "transcript_translation_chunk_000").status == "completed"
    assert _stage_by_name(store, job.id, "transcript_translation_chunk_001").status == "completed"
    assert _stage_by_name(store, job.id, "transcript_translation_merge").status == "completed"
    assert _stage_by_name(store, job.id, "analysis_translation").status == "completed"
    assert _stage_by_name(store, job.id, "analysis_ru_review").status == "pending"
    assert _stage_by_name(store, job.id, "markdown_render").status == "pending"
    assert _stage_by_name(store, job.id, "deterministic_graph").status == "pending"


def test_runtime_explicit_reset_from_analysis_translation_preserves_transcript_work(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    job = store.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    store.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    store.expand_translation_chunk_plan(
        job.id,
        [{"chunk_index": 0, "total_chunks": 1, "target_language": "ru", "source_segment_start_index": 0, "source_segment_end_index": 0}],
    )
    now = time.time()
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            """
            UPDATE stage_plans
            SET status = 'completed', attempt = 1, completed_at = ?, updated_at = ?
            WHERE video_job_id = ?
            """,
            (now, now, job.id),
        )
        conn.execute(
            """
            UPDATE stage_plans
            SET status = 'skipped', warning = ?, completed_at = ?, updated_at = ?
            WHERE video_job_id = ? AND stage_name = 'analysis_translation'
            """,
            ("translation disabled", now, now, job.id),
        )
        conn.execute("UPDATE video_jobs SET status = 'completed', current_stage = 'quality_gate', completed_at = ? WHERE id = ?", (now, job.id))

    record = store.reset_failed_stage_plan_for_retry(job.id, from_stage="analysis_translation")

    assert record.status == "queued"
    assert record.current_stage == "analysis_translation"
    assert _stage_by_name(store, job.id, "transcript_translation_chunk_000").status == "completed"
    assert _stage_by_name(store, job.id, "transcript_translation_merge").status == "completed"
    assert _stage_by_name(store, job.id, "analysis_translation").status == "pending"
    assert _stage_by_name(store, job.id, "analysis_ru_review").status == "pending"
    assert _stage_by_name(store, job.id, "markdown_render").status == "pending"


def test_runtime_reset_failed_stage_plan_from_explicit_stage_rewinds_completed_downstream(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    job = store.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    store.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    now = time.time()
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            """
            UPDATE stage_plans
            SET status = 'completed', attempt = 1, completed_at = ?, updated_at = ?
            WHERE video_job_id = ? AND position < (
                SELECT position FROM stage_plans WHERE video_job_id = ? AND stage_name = 'scientific_analysis'
            )
            """,
            (now, now, job.id, job.id),
        )
        conn.execute(
            """
            UPDATE stage_plans
            SET status = 'failed', attempt = max_attempts, error = ?, completed_at = ?, updated_at = ?
            WHERE video_job_id = ? AND stage_name = 'scientific_analysis'
            """,
            ("analysis failed", now, now, job.id),
        )
        conn.execute("UPDATE video_jobs SET status = 'failed', error = ? WHERE id = ?", ("analysis failed", job.id))

    record = store.reset_failed_stage_plan_for_retry(job.id, from_stage="transcript_source")

    assert record.status == "queued"
    assert record.current_stage == "transcript_source"
    assert _stage_by_name(store, job.id, "metadata").status == "completed"
    assert _stage_by_name(store, job.id, "transcript_source").status == "pending"
    assert _stage_by_name(store, job.id, "audio_download").status == "pending"
    assert _stage_by_name(store, job.id, "whisper_transcription").status == "pending"
    assert _stage_by_name(store, job.id, "scientific_analysis").status == "pending"
    assert _stage_by_name(store, job.id, "deep_graph_llm").status == "skipped"


def test_runtime_reset_failed_stage_plan_closes_stale_running_stage_runs(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    job = store.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    store.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    running = store.start_stage_run(video_job_id=job.id, stage_name="transcript_source")
    now = time.time()
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            """
            UPDATE stage_plans
            SET status = 'failed', attempt = max_attempts, resource_slot = ?, lease_owner = ?,
                lease_until = ?, error = ?, completed_at = ?, updated_at = ?
            WHERE video_job_id = ? AND stage_name = 'transcript_source'
            """,
            ("youtube_metadata#0", "old-owner", now + 60, "max attempts exceeded", now, now, job.id),
        )
        conn.execute(
            "INSERT INTO locks (lock_name, owner, lease_until) VALUES (?, ?, ?)",
            ("youtube_metadata#0", "old-owner", now + 60),
        )
        conn.execute("UPDATE video_jobs SET status = 'failed', error = ? WHERE id = ?", ("max attempts exceeded", job.id))

    store.reset_failed_stage_plan_for_retry(job.id)

    with sqlite3.connect(store.db_path) as conn:
        stage_run = conn.execute(
            "SELECT status, finished_at, error_class, error_message FROM stage_runs WHERE id = ?",
            (running.id,),
        ).fetchone()
        lock = conn.execute("SELECT lock_name FROM locks WHERE lock_name = ?", ("youtube_metadata#0",)).fetchone()
    assert stage_run[0] == "failed"
    assert stage_run[1] is not None
    assert stage_run[2] == "RuntimeReset"
    assert "stage reset for retry" in stage_run[3]
    assert lock is None


def test_runtime_reset_failed_stage_plan_allows_claim_after_real_max_attempt_failure(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    job = store.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    store.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    metadata = store.claim_next_stage(owner="worker-1", lease_seconds=60)
    assert metadata is not None
    store.complete_stage(metadata.stage_run.id)

    first = store.claim_next_stage(owner="worker-1", lease_seconds=60)
    assert first is not None
    assert first.stage.stage_name == "transcript_source"
    store.fail_stage(first.stage_run.id, error="captions unavailable", backoff_seconds=0)
    second = store.claim_next_stage(owner="worker-1", lease_seconds=60)
    assert second is not None
    assert second.stage.stage_name == "transcript_source"
    store.fail_stage(second.stage_run.id, error="captions unavailable", backoff_seconds=0)
    assert store.load_video_job(job.id).status == "failed"
    assert _stage_by_name(store, job.id, "transcript_source").status == "failed"

    store.reset_failed_stage_plan_for_retry(job.id)
    claimed = store.claim_next_stage(owner="worker-2", lease_seconds=60)

    assert claimed is not None
    assert claimed.stage.stage_name == "transcript_source"
    assert claimed.stage.attempt == 1
    assert claimed.stage_run.attempt == 3


def test_runtime_retry_failed_chunk_preserves_completed_sibling_chunks(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    job = store.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    store.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    store.expand_translation_chunk_plan(
        job.id,
        [
            {"chunk_index": 0, "total_chunks": 2, "target_language": "ru", "source_segment_start_index": 0, "source_segment_end_index": 0},
            {"chunk_index": 1, "total_chunks": 2, "target_language": "ru", "source_segment_start_index": 1, "source_segment_end_index": 1},
        ],
    )
    now = time.time()
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE stage_plans SET status = 'completed', attempt = 1, completed_at = ?, updated_at = ? WHERE video_job_id = ? AND position < 5",
            (now, now, job.id),
        )
        conn.execute(
            "UPDATE stage_plans SET status = 'completed', attempt = 1, completed_at = ?, updated_at = ? WHERE video_job_id = ? AND stage_name = 'transcript_translation_chunk_000'",
            (now, now, job.id),
        )
        conn.execute(
            "UPDATE stage_plans SET status = 'failed', attempt = max_attempts, error = ?, completed_at = ?, updated_at = ? WHERE video_job_id = ? AND stage_name = 'transcript_translation_chunk_001'",
            ("chunk failed", now, now, job.id),
        )
        conn.execute("UPDATE video_jobs SET status = 'failed', error = ? WHERE id = ?", ("chunk failed", job.id))

    record = store.reset_failed_stage_plan_for_retry(job.id)

    assert record.status == "queued"
    assert record.current_stage == "transcript_translation_chunk_001"
    assert _stage_by_name(store, job.id, "transcript_translation_chunk_000").status == "completed"
    assert _stage_by_name(store, job.id, "transcript_translation_chunk_001").status == "pending"
    assert _stage_by_name(store, job.id, "transcript_translation_merge").status == "pending"


def test_runtime_retry_failed_analysis_chunk_preserves_completed_sibling_chunks(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    job = store.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    store.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    store.expand_analysis_chunk_plan(
        job.id,
        [
            {"chunk_index": 0, "source_segment_start_index": 0, "source_segment_end_index": 0, "source_transcript_hash": "hash-a"},
            {"chunk_index": 1, "source_segment_start_index": 1, "source_segment_end_index": 1, "source_transcript_hash": "hash-a"},
        ],
    )
    now = time.time()
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE stage_plans SET status = 'completed', attempt = 1, completed_at = ?, updated_at = ? WHERE video_job_id = ? AND position < (SELECT position FROM stage_plans WHERE video_job_id = ? AND stage_name = 'analysis_chunk_000')",
            (now, now, job.id, job.id),
        )
        conn.execute(
            "UPDATE stage_plans SET status = 'completed', attempt = 1, completed_at = ?, updated_at = ? WHERE video_job_id = ? AND stage_name = 'analysis_chunk_000'",
            (now, now, job.id),
        )
        conn.execute(
            "UPDATE stage_plans SET status = 'failed', attempt = max_attempts, error = ?, completed_at = ?, updated_at = ? WHERE video_job_id = ? AND stage_name = 'analysis_chunk_001'",
            ("chunk failed", now, now, job.id),
        )
        conn.execute("UPDATE video_jobs SET status = 'failed', error = ? WHERE id = ?", ("chunk failed", job.id))

    record = store.reset_failed_stage_plan_for_retry(job.id)

    assert record.status == "queued"
    assert record.current_stage == "analysis_chunk_001"
    assert _stage_by_name(store, job.id, "analysis_chunk_000").status == "completed"
    assert _stage_by_name(store, job.id, "analysis_chunk_001").status == "pending"
    assert _stage_by_name(store, job.id, "analysis_reduce").status == "pending"


def test_runtime_explicit_retry_analysis_chunk_preserves_completed_sibling_chunks(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    job = store.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    store.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    store.expand_analysis_chunk_plan(
        job.id,
        [
            {"chunk_index": 0, "source_segment_start_index": 0, "source_segment_end_index": 0, "source_transcript_hash": "hash-a"},
            {"chunk_index": 1, "source_segment_start_index": 1, "source_segment_end_index": 1, "source_transcript_hash": "hash-a"},
        ],
    )
    now = time.time()
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE stage_plans SET status = 'completed', attempt = 1, completed_at = ?, updated_at = ? WHERE video_job_id = ? AND position < (SELECT position FROM stage_plans WHERE video_job_id = ? AND stage_name = 'analysis_chunk_000')",
            (now, now, job.id, job.id),
        )
        conn.execute(
            "UPDATE stage_plans SET status = 'completed', attempt = 1, completed_at = ?, updated_at = ? WHERE video_job_id = ? AND stage_name = 'analysis_chunk_000'",
            (now, now, job.id),
        )
        conn.execute(
            "UPDATE stage_plans SET status = 'failed', attempt = max_attempts, error = ?, completed_at = ?, updated_at = ? WHERE video_job_id = ? AND stage_name = 'analysis_chunk_001'",
            ("chunk failed", now, now, job.id),
        )
        conn.execute("UPDATE video_jobs SET status = 'failed', error = ? WHERE id = ?", ("chunk failed", job.id))

    record = store.reset_failed_stage_plan_for_retry(job.id, from_stage="analysis_chunk_001")

    assert record.status == "queued"
    assert record.current_stage == "analysis_chunk_001"
    assert _stage_by_name(store, job.id, "analysis_chunk_000").status == "completed"
    assert _stage_by_name(store, job.id, "analysis_chunk_001").status == "pending"
    assert _stage_by_name(store, job.id, "analysis_reduce").status == "pending"


def test_runtime_lock_lease_prevents_parallel_resource_use(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")

    assert store.acquire_lock("lm_studio_analysis", owner="worker-1", lease_seconds=60) is True
    assert store.acquire_lock("lm_studio_analysis", owner="worker-2", lease_seconds=60) is False
    store.release_lock("lm_studio_analysis", owner="worker-1")
    assert store.acquire_lock("lm_studio_analysis", owner="worker-2", lease_seconds=60) is True


def test_runtime_claim_respects_retry_schedule(tmp_path):
    store = RuntimeStore(tmp_path / "index.sqlite3")
    job = store.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    stage = store.start_stage_run(video_job_id=job.id, stage_name="metadata")
    store.schedule_retry(stage_run_id=stage.id, attempt=2, reason="rate_limit", backoff_seconds=60)
    store.resume_video_job(job.id)

    assert store.claim_next_video_job(owner="worker-1", lease_seconds=60) is None


def test_default_concurrency_limits_protect_local_bottlenecks():
    assert DEFAULT_CONCURRENCY_LIMITS["youtube_metadata"] == 3
    assert DEFAULT_CONCURRENCY_LIMITS["audio_download"] == 2
    assert DEFAULT_CONCURRENCY_LIMITS["whisper_cpu"] == 1
    assert DEFAULT_CONCURRENCY_LIMITS["whisper_accelerated"] == 1
    assert DEFAULT_CONCURRENCY_LIMITS["lm_studio_translation"] == 1
    assert DEFAULT_CONCURRENCY_LIMITS["lm_studio_analysis"] == 1
    assert DEFAULT_CONCURRENCY_LIMITS["deep_graph_llm"] == 1
    assert DEFAULT_CONCURRENCY_LIMITS["neo4j_sync"] == 1


def test_lm_studio_translation_concurrency_is_clamped_to_safe_max(tmp_path, monkeypatch):
    monkeypatch.setenv("YT_SUB_HUB_LM_STUDIO_TRANSLATION_CONCURRENCY", "9")
    store = RuntimeStore(tmp_path / "index.sqlite3")
    store.ensure_db()
    with sqlite3.connect(store.db_path) as conn:
        now = time.time()
        assert store._acquire_resource_slot(conn, "lm_studio_translation", "worker-1", now + 60, now) == "lm_studio_translation#0"
        assert store._acquire_resource_slot(conn, "lm_studio_translation", "worker-2", now + 60, now) == "lm_studio_translation#1"
        assert store._acquire_resource_slot(conn, "lm_studio_translation", "worker-3", now + 60, now) is None


def test_lm_studio_translation_runtime_concurrency_uses_settings_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("YT_SUB_HUB_LM_STUDIO_TRANSLATION_CONCURRENCY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("YT_SUB_HUB_LM_STUDIO_TRANSLATION_CONCURRENCY=2\n", encoding="utf-8")
    settings = get_settings(env_file=env_file)
    assert settings.lm_studio_translation_concurrency == 2

    store = RuntimeStore(tmp_path / "index.sqlite3", settings=settings)
    store.ensure_db()

    with sqlite3.connect(store.db_path) as conn:
        now = time.time()
        assert store._acquire_resource_slot(conn, "lm_studio_translation", "worker-1", now + 60, now) == "lm_studio_translation#0"
        assert store._acquire_resource_slot(conn, "lm_studio_translation", "worker-2", now + 60, now) == "lm_studio_translation#1"
        assert store._acquire_resource_slot(conn, "lm_studio_translation", "worker-3", now + 60, now) is None


def test_lm_studio_translation_runtime_concurrency_clamps_settings_value(tmp_path, monkeypatch):
    monkeypatch.delenv("YT_SUB_HUB_LM_STUDIO_TRANSLATION_CONCURRENCY", raising=False)
    base = get_settings(env_file=Path("/tmp/non-existent-yt-sub-hub.env"))

    high_store = RuntimeStore(
        tmp_path / "high.sqlite3",
        settings=replace(base, lm_studio_translation_concurrency=9),
    )
    high_store.ensure_db()
    with sqlite3.connect(high_store.db_path) as conn:
        now = time.time()
        assert high_store._acquire_resource_slot(conn, "lm_studio_translation", "worker-1", now + 60, now) == "lm_studio_translation#0"
        assert high_store._acquire_resource_slot(conn, "lm_studio_translation", "worker-2", now + 60, now) == "lm_studio_translation#1"
        assert high_store._acquire_resource_slot(conn, "lm_studio_translation", "worker-3", now + 60, now) is None

    low_store = RuntimeStore(
        tmp_path / "low.sqlite3",
        settings=replace(base, lm_studio_translation_concurrency=0),
    )
    low_store.ensure_db()
    with sqlite3.connect(low_store.db_path) as conn:
        now = time.time()
        assert low_store._acquire_resource_slot(conn, "lm_studio_translation", "worker-1", now + 60, now) == "lm_studio_translation#0"
        assert low_store._acquire_resource_slot(conn, "lm_studio_translation", "worker-2", now + 60, now) is None

    invalid_store = RuntimeStore(
        tmp_path / "invalid.sqlite3",
        settings=replace(base, lm_studio_translation_concurrency="invalid"),
    )
    invalid_store.ensure_db()
    with sqlite3.connect(invalid_store.db_path) as conn:
        now = time.time()
        assert invalid_store._acquire_resource_slot(conn, "lm_studio_translation", "worker-1", now + 60, now) == "lm_studio_translation#0"
        assert invalid_store._acquire_resource_slot(conn, "lm_studio_translation", "worker-2", now + 60, now) is None
