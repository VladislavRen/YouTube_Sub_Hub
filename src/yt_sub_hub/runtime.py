from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from yt_sub_hub.config import Settings


DEFAULT_CONCURRENCY_LIMITS = {
    "youtube_metadata": 3,
    "audio_download": 2,
    "whisper_cpu": 1,
    "whisper_accelerated": 1,
    "lm_studio_translation": 1,
    "lm_studio_analysis": 1,
    "deep_graph_llm": 1,
    "markdown_render": 4,
    "local_validation": 4,
    "critic_review": 2,
    "youtube_data_api": 2,
    "local_candidate_scoring": 2,
    "llm_discovery_review": 1,
    "neo4j_sync": 1,
    "queue_mutation": 1,
}

DISCOVERY_STATIC_STAGES = [
    "youtube_discovery_seed_extract",
    "youtube_discovery_query_expand",
    "youtube_discovery_candidate_fetch",
    "youtube_discovery_candidate_score",
    "youtube_discovery_review",
    "youtube_discovery_graph",
    "youtube_discovery_report",
]

VIDEO_PROCESSING_JOB_KIND = "video_processing"
YOUTUBE_DISCOVERY_JOB_KIND = "youtube_discovery"
ACTIVE_JOB_STATUSES = {"queued", "running"}
TERMINAL_JOB_STATUSES = {"completed", "completed_with_warning", "failed", "skipped"}


@dataclass
class ChannelJobRecord:
    id: str
    channel_url: str
    status: str
    channel_id: str | None = None
    requested_limit: int | None = None
    quality_mode: str = "strict"
    acceptance_mode: str = "data_api_only"
    limit: int | None = None
    video_count: int = 0
    queued_count: int = 0
    completed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    error: str | None = None
    warning: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0
    completed_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VideoJobRecord:
    id: str
    video_id: str
    canonical_url: str
    status: str
    job_kind: str = VIDEO_PROCESSING_JOB_KIND
    channel_job_id: str | None = None
    current_stage: str | None = None
    priority: int = 100
    lease_owner: str | None = None
    lease_until: float | None = None
    cancel_requested: bool = False
    error: str | None = None
    warning: str | None = None
    quality_mode: str = "strict"
    acceptance_mode: str = "data_api_only"
    created_at: float = 0.0
    updated_at: float = 0.0
    completed_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StageRunRecord:
    id: str
    video_job_id: str
    stage_name: str
    status: str
    attempt: int
    started_at: float | None = None
    finished_at: float | None = None
    input_hash: str | None = None
    output_hash: str | None = None
    error_class: str | None = None
    error_message: str | None = None
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StagePlanRecord:
    id: str
    video_job_id: str
    stage_name: str
    status: str
    position: int
    attempt: int
    max_attempts: int
    hard_failure: bool
    resource_lock: str | None = None
    resource_slot: str | None = None
    lease_owner: str | None = None
    lease_until: float | None = None
    retry_after: float | None = None
    error: str | None = None
    warning: str | None = None
    stage_payload_json: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0
    completed_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClaimedStageRecord:
    job: VideoJobRecord
    stage: StagePlanRecord
    stage_run: StageRunRecord

    def to_dict(self) -> dict[str, Any]:
        return {
            "job": self.job.to_dict(),
            "stage": self.stage.to_dict(),
            "stage_run": self.stage_run.to_dict(),
        }


@dataclass
class StageArtifactRecord:
    id: str
    stage_run_id: str
    video_id: str
    artifact_type: str
    path: str
    language: str | None
    content_hash: str
    status: str
    provenance_json: str
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetryRecord:
    id: str
    stage_run_id: str
    attempt: int
    scheduled_at: float
    reason: str
    backoff_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RuntimeStore:
    def __init__(self, db_path: Path | str, settings: Settings | None = None) -> None:
        self.db_path = Path(db_path)
        self.settings = settings

    def _connect(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path)

    def ensure_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS channel_jobs (
                    id TEXT PRIMARY KEY,
                    channel_id TEXT,
                    channel_url TEXT NOT NULL,
                    status TEXT NOT NULL,
                    requested_limit INTEGER,
                    quality_mode TEXT NOT NULL DEFAULT 'strict',
                    acceptance_mode TEXT NOT NULL DEFAULT 'data_api_only',
                    "limit" INTEGER,
                    video_count INTEGER NOT NULL DEFAULT 0,
                    queued_count INTEGER NOT NULL DEFAULT 0,
                    completed_count INTEGER NOT NULL DEFAULT 0,
                    failed_count INTEGER NOT NULL DEFAULT 0,
                    skipped_count INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    warning TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    completed_at REAL
                )
                """
            )
            channel_job_columns = {row[1] for row in conn.execute("PRAGMA table_info(channel_jobs)").fetchall()}
            channel_additions = {
                "quality_mode": "TEXT NOT NULL DEFAULT 'strict'",
                "acceptance_mode": "TEXT NOT NULL DEFAULT 'data_api_only'",
                "limit": "INTEGER",
                "video_count": "INTEGER NOT NULL DEFAULT 0",
                "queued_count": "INTEGER NOT NULL DEFAULT 0",
                "completed_count": "INTEGER NOT NULL DEFAULT 0",
                "failed_count": "INTEGER NOT NULL DEFAULT 0",
                "skipped_count": "INTEGER NOT NULL DEFAULT 0",
            }
            for name, definition in channel_additions.items():
                if name not in channel_job_columns:
                    column = '"limit"' if name == "limit" else name
                    conn.execute(f"ALTER TABLE channel_jobs ADD COLUMN {column} {definition}")
            conn.execute("UPDATE channel_jobs SET quality_mode = 'strict' WHERE quality_mode IS NULL OR quality_mode = ''")
            conn.execute("UPDATE channel_jobs SET acceptance_mode = 'data_api_only' WHERE acceptance_mode IS NULL OR acceptance_mode = ''")
            conn.execute('UPDATE channel_jobs SET "limit" = requested_limit WHERE "limit" IS NULL AND requested_limit IS NOT NULL')
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS video_jobs (
                    id TEXT PRIMARY KEY,
                    channel_job_id TEXT,
                    video_id TEXT NOT NULL,
                    canonical_url TEXT NOT NULL,
                    status TEXT NOT NULL,
                    job_kind TEXT NOT NULL DEFAULT 'video_processing',
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
                    UNIQUE(video_id, channel_job_id, job_kind)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS stage_runs (
                    id TEXT PRIMARY KEY,
                    video_job_id TEXT NOT NULL,
                    stage_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    started_at REAL,
                    finished_at REAL,
                    input_hash TEXT,
                    output_hash TEXT,
                    error_class TEXT,
                    error_message TEXT,
                    warning TEXT,
                    UNIQUE(video_job_id, stage_name, attempt)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS stage_plans (
                    id TEXT PRIMARY KEY,
                    video_job_id TEXT NOT NULL,
                    stage_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL,
                    hard_failure INTEGER NOT NULL DEFAULT 1,
                    resource_lock TEXT,
                    resource_slot TEXT,
                    lease_owner TEXT,
                    lease_until REAL,
                    retry_after REAL,
                    error TEXT,
                    warning TEXT,
                    stage_payload_json TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    completed_at REAL,
                    UNIQUE(video_job_id, stage_name)
                )
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(stage_plans)").fetchall()}
            if "stage_payload_json" not in columns:
                conn.execute("ALTER TABLE stage_plans ADD COLUMN stage_payload_json TEXT")
            video_job_columns = {row[1] for row in conn.execute("PRAGMA table_info(video_jobs)").fetchall()}
            if "quality_mode" not in video_job_columns:
                conn.execute("ALTER TABLE video_jobs ADD COLUMN quality_mode TEXT NOT NULL DEFAULT 'strict'")
            if "acceptance_mode" not in video_job_columns:
                conn.execute("ALTER TABLE video_jobs ADD COLUMN acceptance_mode TEXT NOT NULL DEFAULT 'data_api_only'")
            if "job_kind" not in video_job_columns:
                conn.execute("ALTER TABLE video_jobs ADD COLUMN job_kind TEXT NOT NULL DEFAULT 'video_processing'")
            conn.execute("UPDATE video_jobs SET quality_mode = 'strict' WHERE quality_mode IS NULL OR quality_mode = ''")
            conn.execute("UPDATE video_jobs SET acceptance_mode = 'data_api_only' WHERE acceptance_mode IS NULL OR acceptance_mode = ''")
            conn.execute("UPDATE video_jobs SET job_kind = 'video_processing' WHERE job_kind IS NULL OR job_kind = ''")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS stage_artifacts (
                    id TEXT PRIMARY KEY,
                    stage_run_id TEXT NOT NULL,
                    video_id TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    path TEXT NOT NULL,
                    language TEXT,
                    content_hash TEXT,
                    status TEXT NOT NULL,
                    provenance_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS locks (
                    lock_name TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    lease_until REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS retries (
                    id TEXT PRIMARY KEY,
                    stage_run_id TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    scheduled_at REAL NOT NULL,
                    reason TEXT NOT NULL,
                    backoff_seconds REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_video_jobs_status_priority_updated "
                "ON video_jobs(status, priority, updated_at)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_video_jobs_video_id ON video_jobs(video_id)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_video_jobs_identity "
                "ON video_jobs(video_id, channel_job_id, job_kind)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_stage_runs_video_stage_status "
                "ON stage_runs(video_job_id, stage_name, status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_stage_plans_status_position "
                "ON stage_plans(status, position, updated_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_stage_plans_video_position "
                "ON stage_plans(video_job_id, position)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_stage_artifacts_video_type_language_status "
                "ON stage_artifacts(video_id, artifact_type, language, status)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_locks_lease_until ON locks(lease_until)")

    def create_channel_job(
        self,
        *,
        channel_url: str,
        channel_id: str | None = None,
        requested_limit: int | None = None,
        limit: int | None = None,
        quality_mode: str = "strict",
        acceptance_mode: str = "data_api_only",
        status: str = "queued",
    ) -> ChannelJobRecord:
        self.ensure_db()
        from yt_sub_hub.config import normalize_acceptance_mode, normalize_quality_mode

        quality_mode = normalize_quality_mode(quality_mode)
        acceptance_mode = normalize_acceptance_mode(acceptance_mode)
        now = time.time()
        effective_limit = limit if limit is not None else requested_limit
        job_id = _stable_id("channel", channel_id or channel_url.strip(), length=16)
        record = ChannelJobRecord(
            id=job_id,
            channel_id=channel_id,
            channel_url=channel_url,
            status=status,
            requested_limit=effective_limit,
            quality_mode=quality_mode,
            acceptance_mode=acceptance_mode,
            limit=effective_limit,
            created_at=now,
            updated_at=now,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO channel_jobs
                  (id, channel_id, channel_url, status, requested_limit, quality_mode, acceptance_mode, "limit",
                   video_count, queued_count, completed_count, failed_count, skipped_count,
                   error, warning, created_at, updated_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    channel_id=excluded.channel_id,
                    channel_url=excluded.channel_url,
                    status=excluded.status,
                    requested_limit=excluded.requested_limit,
                    quality_mode=excluded.quality_mode,
                    acceptance_mode=excluded.acceptance_mode,
                    "limit"=excluded."limit",
                    updated_at=excluded.updated_at
                """,
                (
                    record.id,
                    record.channel_id,
                    record.channel_url,
                    record.status,
                    record.requested_limit,
                    record.quality_mode,
                    record.acceptance_mode,
                    record.limit,
                    record.video_count,
                    record.queued_count,
                    record.completed_count,
                    record.failed_count,
                    record.skipped_count,
                    record.error,
                    record.warning,
                    record.created_at,
                    record.updated_at,
                    record.completed_at,
                ),
            )
        return self.load_channel_job(job_id)

    def load_channel_job(self, job_id: str) -> ChannelJobRecord:
        self.ensure_db()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, channel_id, channel_url, status, requested_limit, quality_mode, acceptance_mode, "limit",
                       video_count, queued_count, completed_count, failed_count, skipped_count,
                       error, warning, created_at, updated_at, completed_at
                FROM channel_jobs WHERE id = ?
                """,
                (job_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown channel job: {job_id}")
        return _channel_job_from_row(row)

    def recompute_channel_job_status(self, channel_job_id: str) -> ChannelJobRecord:
        self.ensure_db()
        now = time.time()
        with self._connect() as conn:
            channel = conn.execute("SELECT id FROM channel_jobs WHERE id = ?", (channel_job_id,)).fetchone()
            if channel is None:
                raise KeyError(f"Unknown channel job: {channel_job_id}")
            rows = conn.execute(
                """
                SELECT status, warning, error
                FROM video_jobs
                WHERE channel_job_id = ? AND job_kind = ?
                """,
                (channel_job_id, VIDEO_PROCESSING_JOB_KIND),
            ).fetchall()
            total = len(rows)
            counts: dict[str, int] = {}
            warnings = []
            errors = []
            for status, warning, error in rows:
                counts[str(status)] = counts.get(str(status), 0) + 1
                if warning:
                    warnings.append(str(warning))
                if error:
                    errors.append(str(error))
            active = counts.get("queued", 0) + counts.get("running", 0)
            completed = counts.get("completed", 0)
            completed_warning = counts.get("completed_with_warning", 0)
            failed = counts.get("failed", 0)
            skipped = counts.get("skipped", 0)
            if total == 0:
                status = "queued"
                completed_at = None
            elif active:
                status = "running"
                completed_at = None
            elif completed == total:
                status = "completed"
                completed_at = now
            elif failed and (failed == total or failed / total >= 0.5):
                status = "failed"
                completed_at = now
            else:
                status = "completed_with_warning"
                completed_at = now
            warning = "; ".join(dict.fromkeys(warnings)) if warnings and status != "failed" else None
            error = "; ".join(dict.fromkeys(errors)) if errors and status == "failed" else None
            conn.execute(
                """
                UPDATE channel_jobs
                SET status = ?, video_count = ?, queued_count = ?, completed_count = ?,
                    failed_count = ?, skipped_count = ?, warning = ?, error = ?,
                    updated_at = ?, completed_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    total,
                    counts.get("queued", 0),
                    completed + completed_warning,
                    failed,
                    skipped,
                    warning,
                    error,
                    now,
                    completed_at,
                    channel_job_id,
                ),
            )
        return self.load_channel_job(channel_job_id)

    def cancel_channel_job(self, channel_job_id: str) -> ChannelJobRecord:
        self.ensure_db()
        self.load_channel_job(channel_job_id)
        for job in self.list_video_jobs(channel_job_id, job_kind=VIDEO_PROCESSING_JOB_KIND):
            if job.status not in {"completed", "completed_with_warning", "failed", "skipped"}:
                self.request_cancel(job.id)
        return self.recompute_channel_job_status(channel_job_id)

    def retry_channel_failed(self, channel_job_id: str) -> ChannelJobRecord:
        self.ensure_db()
        self.load_channel_job(channel_job_id)
        for job in self.list_video_jobs(channel_job_id, job_kind=VIDEO_PROCESSING_JOB_KIND):
            if job.status not in {"failed", "skipped"}:
                continue
            try:
                if job.status == "skipped":
                    from_stage = self._first_resettable_skipped_stage(job.id)
                    if from_stage is None:
                        self.resume_video_job(job.id)
                    else:
                        self.reset_failed_stage_plan_for_retry(job.id, from_stage=from_stage)
                else:
                    self.reset_failed_stage_plan_for_retry(job.id)
            except ValueError:
                self.resume_video_job(job.id)
        return self.recompute_channel_job_status(channel_job_id)

    def retry_video_job_for_channel(self, job_id: str) -> VideoJobRecord:
        job = self.load_video_job(job_id)
        if job.status == "skipped":
            from_stage = self._first_resettable_skipped_stage(job.id)
            if from_stage is not None:
                return self.reset_failed_stage_plan_for_retry(job.id, from_stage=from_stage)
            return self.resume_video_job(job.id)
        try:
            return self.reset_failed_stage_plan_for_retry(job.id)
        except ValueError:
            return self.resume_video_job(job.id)

    def retry_stage(self, job_id: str, stage_name: str) -> VideoJobRecord:
        self._validate_stage_for_job(job_id, stage_name)
        if not self._recovery_reset_applicable(job_id):
            raise ValueError(f"retry stage not applicable for job: {job_id}")
        return self.reset_failed_stage_plan_for_retry(job_id, from_stage=stage_name)

    def reset_downstream(self, job_id: str, from_stage: str) -> VideoJobRecord:
        self._validate_stage_for_job(job_id, from_stage)
        if not self._recovery_reset_applicable(job_id):
            raise ValueError(f"reset downstream not applicable for job: {job_id}")
        return self.reset_failed_stage_plan_for_retry(job_id, from_stage=from_stage)

    def cancel_job(self, job_id: str) -> VideoJobRecord:
        job = self.load_video_job(job_id)
        if job.status in {"completed", "completed_with_warning", "failed", "skipped"}:
            raise ValueError(f"cancel not applicable for job: {job_id}")
        return self.request_cancel(job_id)

    def resume_job(self, job_id: str) -> VideoJobRecord:
        job = self.load_video_job(job_id)
        failed = [stage for stage in self.list_stage_plan(job_id) if stage.status == "failed"]
        if failed:
            return self.reset_failed_stage_plan_for_retry(job.id)
        if job.status == "skipped":
            from_stage = self._first_resettable_skipped_stage(job.id)
            if from_stage is not None:
                return self.reset_failed_stage_plan_for_retry(job.id, from_stage=from_stage)
        if job.status == "completed" or job.status == "completed_with_warning":
            raise ValueError(f"resume not applicable for job: {job_id}")
        if job.status == "queued" and not self._job_has_stale_lease(job):
            raise ValueError(f"resume not applicable for job: {job_id}")
        if job.status == "running" and not self._job_has_stale_lease(job):
            raise ValueError(f"resume not applicable for job: {job_id}")
        return self.resume_video_job(job.id)

    def retry_failed_chunks(self, job_id: str, group: str) -> VideoJobRecord:
        if group not in {"transcript_translation", "analysis", "youtube_discovery_queries"}:
            raise ValueError(f"invalid chunk group: {group}")
        if group == "transcript_translation":
            prefix = "transcript_translation_chunk_%"
            anchor_stage = "transcript_translation_merge"
        elif group == "analysis":
            prefix = "analysis_chunk_%"
            anchor_stage = "analysis_reduce"
        else:
            prefix = "youtube_discovery_query_%"
            anchor_stage = "youtube_discovery_candidate_fetch"
        self.ensure_db()
        now = time.time()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute("SELECT 1 FROM video_jobs WHERE id = ?", (job_id,)).fetchone() is None:
                raise KeyError(f"Unknown video job: {job_id}")
            failed_rows = conn.execute(
                """
                SELECT stage_name, position, resource_slot, lease_owner
                FROM stage_plans
                WHERE video_job_id = ?
                  AND stage_name LIKE ?
                  AND status = 'failed'
                ORDER BY stage_name
                """,
                (job_id, prefix),
            ).fetchall()
            if not failed_rows:
                raise ValueError(f"no failed chunks for group: {group}")
            anchor_row = conn.execute(
                """
                SELECT position
                FROM stage_plans
                WHERE video_job_id = ? AND stage_name = ?
                """,
                (job_id, anchor_stage),
            ).fetchone()
            anchor_position = int(anchor_row[0]) if anchor_row is not None else max(int(row[1]) for row in failed_rows)
            downstream_rows = conn.execute(
                """
                SELECT stage_name, resource_slot, lease_owner
                FROM stage_plans
                WHERE video_job_id = ?
                  AND (
                    stage_name = ?
                    OR position > ?
                  )
                  AND (
                    status IN ('failed', 'pending', 'running', 'completed', 'completed_with_warning')
                    OR (status = 'skipped' AND hard_failure = 1)
                  )
                ORDER BY position, stage_name
                """,
                (job_id, anchor_stage, anchor_position),
            ).fetchall()
            reset_by_name: dict[str, tuple[str | None, str | None]] = {
                str(row[0]): (row[2], row[3]) for row in failed_rows
            }
            for stage_name, resource_slot, owner in downstream_rows:
                reset_by_name.setdefault(str(stage_name), (resource_slot, owner))
            reset_stage_names = list(reset_by_name)
            first_reset_stage_name = str(failed_rows[0][0])
            for resource_slot, owner in reset_by_name.values():
                if resource_slot and owner:
                    conn.execute("DELETE FROM locks WHERE lock_name = ? AND owner = ?", (resource_slot, owner))
            placeholders = ",".join("?" for _ in reset_stage_names)
            conn.execute(
                f"""
                DELETE FROM retries
                WHERE stage_run_id IN (
                    SELECT id FROM stage_runs
                    WHERE video_job_id = ?
                      AND stage_name IN ({placeholders})
                )
                """,
                (job_id, *reset_stage_names),
            )
            conn.execute(
                f"""
                UPDATE stage_runs
                SET status = 'failed',
                    finished_at = ?,
                    error_class = 'RuntimeReset',
                    error_message = 'stage reset for retry'
                WHERE video_job_id = ?
                  AND status = 'running'
                  AND stage_name IN ({placeholders})
                """,
                (now, job_id, *reset_stage_names),
            )
            conn.execute(
                f"""
                UPDATE stage_plans
                SET status = 'pending',
                    attempt = 0,
                    retry_after = NULL,
                    error = NULL,
                    warning = NULL,
                    lease_owner = NULL,
                    lease_until = NULL,
                    resource_slot = NULL,
                    completed_at = NULL,
                    created_at = ?,
                    updated_at = ?
                WHERE video_job_id = ?
                  AND stage_name IN ({placeholders})
                """,
                (now, now, job_id, *reset_stage_names),
            )
            conn.execute(
                """
                UPDATE video_jobs
                SET status = 'queued',
                    current_stage = ?,
                    lease_owner = NULL,
                    lease_until = NULL,
                    cancel_requested = 0,
                    error = NULL,
                    warning = NULL,
                    completed_at = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (first_reset_stage_name, now, job_id),
            )
        return self.load_video_job(job_id)

    def _validate_stage_for_job(self, job_id: str, stage_name: str) -> None:
        self.load_video_job(job_id)
        if not any(stage.stage_name == stage_name for stage in self.list_stage_plan(job_id)):
            raise KeyError(f"Unknown stage for job {job_id}: {stage_name}")

    def _recovery_reset_applicable(self, job_id: str) -> bool:
        job = self.load_video_job(job_id)
        if job.status not in {"failed", "skipped", "running"}:
            return False
        if job.status == "running" and not self._job_has_stale_lease(job):
            return False
        return True

    def _job_has_stale_lease(self, job: VideoJobRecord) -> bool:
        return job.lease_until is None or float(job.lease_until) < time.time()

    def _first_resettable_skipped_stage(self, job_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT stage_name
                FROM stage_plans
                WHERE video_job_id = ?
                  AND status = 'skipped'
                  AND hard_failure = 1
                ORDER BY position
                LIMIT 1
                """,
                (job_id,),
            ).fetchone()
        return str(row[0]) if row is not None else None

    def create_video_job(
        self,
        *,
        video_id: str,
        canonical_url: str,
        channel_job_id: str | None = None,
        job_id: str | None = None,
        job_kind: str = VIDEO_PROCESSING_JOB_KIND,
        priority: int = 100,
        status: str = "queued",
        quality_mode: str = "strict",
        acceptance_mode: str = "data_api_only",
    ) -> VideoJobRecord:
        self.ensure_db()
        from yt_sub_hub.config import normalize_acceptance_mode, normalize_quality_mode

        quality_mode = normalize_quality_mode(quality_mode)
        acceptance_mode = normalize_acceptance_mode(acceptance_mode)
        job_kind = _normalize_job_kind(job_kind)
        if job_id is not None:
            try:
                return self.load_video_job(job_id)
            except KeyError:
                pass
        existing = self._find_video_job(video_id, channel_job_id, job_kind)
        if existing is not None:
            return existing
        now = time.time()
        if job_id is None:
            if job_kind == VIDEO_PROCESSING_JOB_KIND:
                job_id = _stable_id("video", channel_job_id or "standalone", video_id, length=20)
            else:
                job_id = _stable_id("video", job_kind, channel_job_id or "standalone", video_id, length=20)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO video_jobs
                  (id, channel_job_id, video_id, canonical_url, status, job_kind, current_stage, priority,
                   lease_owner, lease_until, cancel_requested, error, warning, quality_mode, acceptance_mode,
                   created_at, updated_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    channel_job_id,
                    video_id,
                    canonical_url,
                    status,
                    job_kind,
                    None,
                    priority,
                    None,
                    None,
                    0,
                    None,
                    None,
                    quality_mode,
                    acceptance_mode,
                    now,
                    now,
                    None,
                ),
            )
        return self.load_video_job(job_id)

    def _find_video_job(self, video_id: str, channel_job_id: str | None, job_kind: str = VIDEO_PROCESSING_JOB_KIND) -> VideoJobRecord | None:
        job_kind = _normalize_job_kind(job_kind)
        with self._connect() as conn:
            if channel_job_id is None:
                row = conn.execute(
                    """
                    SELECT id, channel_job_id, video_id, canonical_url, status, current_stage, priority,
                           lease_owner, lease_until, cancel_requested, error, warning, quality_mode, acceptance_mode, job_kind,
                           created_at, updated_at, completed_at
                    FROM video_jobs WHERE video_id = ? AND channel_job_id IS NULL AND job_kind = ?
                    """,
                    (video_id, job_kind),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT id, channel_job_id, video_id, canonical_url, status, current_stage, priority,
                           lease_owner, lease_until, cancel_requested, error, warning, quality_mode, acceptance_mode, job_kind,
                           created_at, updated_at, completed_at
                    FROM video_jobs WHERE video_id = ? AND channel_job_id = ? AND job_kind = ?
                    """,
                    (video_id, channel_job_id, job_kind),
                ).fetchone()
        return _video_job_from_row(row) if row is not None else None

    def load_video_job(self, job_id: str) -> VideoJobRecord:
        self.ensure_db()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, channel_job_id, video_id, canonical_url, status, current_stage, priority,
                       lease_owner, lease_until, cancel_requested, error, warning, quality_mode, acceptance_mode, job_kind,
                       created_at, updated_at, completed_at
                FROM video_jobs WHERE id = ?
                """,
                (job_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown video job: {job_id}")
        return _video_job_from_row(row)

    def update_video_job_modes(
        self, job_id: str, *, quality_mode: str | None = None, acceptance_mode: str | None = None,
    ) -> VideoJobRecord:
        """Apply explicit retry options without changing an active worker's contract."""
        from yt_sub_hub.config import normalize_acceptance_mode, normalize_quality_mode

        self.ensure_db()
        quality = normalize_quality_mode(quality_mode) if quality_mode is not None else None
        acceptance = normalize_acceptance_mode(acceptance_mode) if acceptance_mode is not None else None
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT status, lease_owner, quality_mode, acceptance_mode FROM video_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown video job: {job_id}")
            next_modes = (quality or row[2], acceptance or row[3])
            if next_modes != (row[2], row[3]):
                if row[0] == "running" or row[1] is not None:
                    raise RuntimeError("Cannot change processing modes while the job is running; wait or cancel it first.")
                conn.execute(
                    "UPDATE video_jobs SET quality_mode = ?, acceptance_mode = ?, updated_at = ? WHERE id = ?",
                    (*next_modes, time.time(), job_id),
                )
        return self.load_video_job(job_id)

    def update_video_job_quality_mode(self, job_id: str, quality_mode: str) -> VideoJobRecord:
        self.ensure_db()
        from yt_sub_hub.config import normalize_quality_mode

        quality_mode = normalize_quality_mode(quality_mode)
        now = time.time()
        with self._connect() as conn:
            row = conn.execute("SELECT id FROM video_jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown video job: {job_id}")
            conn.execute(
                """
                UPDATE video_jobs
                SET quality_mode = ?, updated_at = ?
                WHERE id = ?
                """,
                (quality_mode, now, job_id),
            )
        return self.load_video_job(job_id)

    def update_video_job_acceptance_mode(self, job_id: str, acceptance_mode: str) -> VideoJobRecord:
        self.ensure_db()
        from yt_sub_hub.config import normalize_acceptance_mode

        acceptance_mode = normalize_acceptance_mode(acceptance_mode)
        now = time.time()
        with self._connect() as conn:
            row = conn.execute("SELECT id FROM video_jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown video job: {job_id}")
            conn.execute(
                """
                UPDATE video_jobs
                SET acceptance_mode = ?, updated_at = ?
                WHERE id = ?
                """,
                (acceptance_mode, now, job_id),
            )
        return self.load_video_job(job_id)

    def list_video_jobs(self, channel_job_id: str | None = None, *, job_kind: str | None = None) -> list[VideoJobRecord]:
        self.ensure_db()
        job_kind = _normalize_job_kind(job_kind) if job_kind is not None else None
        with self._connect() as conn:
            if channel_job_id is None and job_kind is None:
                rows = conn.execute(
                    """
                    SELECT id, channel_job_id, video_id, canonical_url, status, current_stage, priority,
                           lease_owner, lease_until, cancel_requested, error, warning, quality_mode, acceptance_mode, job_kind,
                           created_at, updated_at, completed_at
                    FROM video_jobs ORDER BY priority, created_at, id
                    """
                ).fetchall()
            elif channel_job_id is None:
                rows = conn.execute(
                    """
                    SELECT id, channel_job_id, video_id, canonical_url, status, current_stage, priority,
                           lease_owner, lease_until, cancel_requested, error, warning, quality_mode, acceptance_mode, job_kind,
                           created_at, updated_at, completed_at
                    FROM video_jobs WHERE job_kind = ? ORDER BY priority, created_at, id
                    """,
                    (job_kind,),
                ).fetchall()
            elif job_kind is None:
                rows = conn.execute(
                    """
                    SELECT id, channel_job_id, video_id, canonical_url, status, current_stage, priority,
                           lease_owner, lease_until, cancel_requested, error, warning, quality_mode, acceptance_mode, job_kind,
                           created_at, updated_at, completed_at
                    FROM video_jobs WHERE channel_job_id = ? ORDER BY priority, created_at, id
                    """,
                    (channel_job_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, channel_job_id, video_id, canonical_url, status, current_stage, priority,
                           lease_owner, lease_until, cancel_requested, error, warning, quality_mode, acceptance_mode, job_kind,
                           created_at, updated_at, completed_at
                    FROM video_jobs WHERE channel_job_id = ? AND job_kind = ? ORDER BY priority, created_at, id
                    """,
                    (channel_job_id, job_kind),
                ).fetchall()
        return [_video_job_from_row(row) for row in rows]

    def claim_next_video_job(self, *, owner: str, lease_seconds: float = 300) -> VideoJobRecord | None:
        self.ensure_db()
        now = time.time()
        lease_until = now + lease_seconds
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT id
                FROM video_jobs
                WHERE cancel_requested = 0
                  AND job_kind = ?
                  AND (status = 'queued' OR (status = 'running' AND (lease_until IS NULL OR lease_until < ?)))
                  AND NOT EXISTS (
                      SELECT 1
                      FROM stage_runs sr
                      JOIN retries r ON r.stage_run_id = sr.id
                      WHERE sr.video_job_id = video_jobs.id
                        AND r.scheduled_at > ?
                  )
                ORDER BY priority, updated_at, id
                LIMIT 1
                """,
                (VIDEO_PROCESSING_JOB_KIND, now, now),
            ).fetchone()
            if row is None:
                return None
            updated = conn.execute(
                """
                UPDATE video_jobs
                SET status = 'running', lease_owner = ?, lease_until = ?, updated_at = ?
                WHERE id = ?
                  AND cancel_requested = 0
                  AND (status = 'queued' OR (status = 'running' AND (lease_until IS NULL OR lease_until < ?)))
                """,
                (owner, lease_until, now, row[0], now),
            )
            if updated.rowcount != 1:
                return None
        return self.load_video_job(row[0])

    def claim_video_job(self, job_id: str, *, owner: str, lease_seconds: float = 300) -> VideoJobRecord | None:
        self.ensure_db()
        now = time.time()
        lease_until = now + lease_seconds
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT id FROM video_jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown video job: {job_id}")
            updated = conn.execute(
                """
                UPDATE video_jobs
                SET status = 'running', lease_owner = ?, lease_until = ?, updated_at = ?
                WHERE id = ?
                  AND cancel_requested = 0
                  AND (status = 'queued' OR (status = 'running' AND (lease_until IS NULL OR lease_until < ?)))
                """,
                (owner, lease_until, now, job_id, now),
            )
            if updated.rowcount != 1:
                return None
        return self.load_video_job(job_id)

    def complete_video_job(self, job_id: str, *, warning: str | None = None) -> VideoJobRecord:
        status = "completed_with_warning" if warning else "completed"
        return self._update_video_job_status(job_id, status, warning=warning, completed=True)

    def complete_video_job_with_warning(self, job_id: str, warning: str) -> VideoJobRecord:
        return self.complete_video_job(job_id, warning=warning)

    def fail_video_job(self, job_id: str, *, error: str) -> VideoJobRecord:
        return self._update_video_job_status(job_id, "failed", error=error, completed=True)

    def skip_video_job(self, job_id: str, *, warning: str | None = None) -> VideoJobRecord:
        return self._update_video_job_status(job_id, "skipped", warning=warning, completed=True)

    def mark_video_job_skipped_requires_media(self, job_id: str) -> VideoJobRecord:
        self.ensure_db()
        now = time.time()
        reason = "skipped_requires_media"
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT id FROM video_jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown video job: {job_id}")
            transcript_row = conn.execute(
                "SELECT position FROM stage_plans WHERE video_job_id = ? AND stage_name = 'transcript_source'",
                (job_id,),
            ).fetchone()
            transcript_position = int(transcript_row[0]) if transcript_row is not None else -1
            conn.execute(
                """
                UPDATE stage_plans
                SET status = 'skipped',
                    warning = ?,
                    error = NULL,
                    lease_owner = NULL,
                    lease_until = NULL,
                    resource_slot = NULL,
                    retry_after = NULL,
                    updated_at = ?,
                    completed_at = ?
                WHERE video_job_id = ?
                  AND stage_name = 'transcript_source'
                  AND status IN ('pending', 'running', 'failed')
                """,
                (reason, now, now, job_id),
            )
            if transcript_position >= 0:
                conn.execute(
                    """
                    UPDATE stage_plans
                    SET status = 'skipped',
                        warning = COALESCE(warning, ?),
                        error = NULL,
                        lease_owner = NULL,
                        lease_until = NULL,
                        resource_slot = NULL,
                        retry_after = NULL,
                        updated_at = ?,
                        completed_at = ?
                    WHERE video_job_id = ?
                      AND position > ?
                      AND status IN ('pending', 'running')
                    """,
                    (reason, now, now, job_id, transcript_position),
                )
            conn.execute(
                """
                UPDATE video_jobs
                SET status = 'skipped',
                    current_stage = 'transcript_source',
                    warning = ?,
                    error = NULL,
                    lease_owner = NULL,
                    lease_until = NULL,
                    updated_at = ?,
                    completed_at = ?
                WHERE id = ?
                """,
                (reason, now, now, job_id),
            )
        return self.load_video_job(job_id)

    def resume_video_job(self, job_id: str) -> VideoJobRecord:
        return self._update_video_job_status(job_id, "queued", clear_cancel=True)

    def release_video_job_claim(self, job_id: str, *, owner: str) -> VideoJobRecord:
        self.ensure_db()
        now = time.time()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT cancel_requested FROM video_jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown video job: {job_id}")
            next_status = "skipped" if bool(row[0]) else "queued"
            completed_at = now if next_status == "skipped" else None
            conn.execute(
                """
                UPDATE video_jobs
                SET status = ?, lease_owner = NULL, lease_until = NULL,
                    updated_at = ?, completed_at = COALESCE(?, completed_at)
                WHERE id = ? AND lease_owner = ?
                """,
                (next_status, now, completed_at, job_id, owner),
            )
        return self.load_video_job(job_id)

    def request_cancel(self, job_id: str) -> VideoJobRecord:
        self.ensure_db()
        now = time.time()
        with self._connect() as conn:
            row = conn.execute("SELECT status FROM video_jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown video job: {job_id}")
            next_status = "skipped" if row[0] == "queued" else row[0]
            completed_at = now if row[0] == "queued" else None
            conn.execute(
                """
                UPDATE video_jobs
                SET cancel_requested = 1, status = ?, completed_at = COALESCE(?, completed_at),
                    lease_owner = CASE WHEN ? = 'skipped' THEN NULL ELSE lease_owner END,
                    lease_until = CASE WHEN ? = 'skipped' THEN NULL ELSE lease_until END,
                    updated_at = ?
                WHERE id = ?
                """,
                (next_status, completed_at, next_status, next_status, now, job_id),
            )
            if next_status == "skipped":
                conn.execute(
                    """
                    UPDATE stage_plans
                    SET status = 'skipped', warning = COALESCE(warning, 'cancel requested'),
                        lease_owner = NULL, lease_until = NULL, resource_slot = NULL,
                        updated_at = ?, completed_at = COALESCE(completed_at, ?)
                    WHERE video_job_id = ? AND status = 'pending'
                    """,
                    (now, now, job_id),
                )
        return self.load_video_job(job_id)

    def _update_video_job_status(
        self,
        job_id: str,
        status: str,
        *,
        error: str | None = None,
        warning: str | None = None,
        completed: bool = False,
        clear_cancel: bool = False,
    ) -> VideoJobRecord:
        self.ensure_db()
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE video_jobs
                SET status = ?, error = ?, warning = ?, lease_owner = NULL, lease_until = NULL,
                    cancel_requested = CASE WHEN ? THEN 0 ELSE cancel_requested END,
                    updated_at = ?, completed_at = CASE WHEN ? THEN ? ELSE completed_at END
                WHERE id = ?
                """,
                (status, error, warning, 1 if clear_cancel else 0, now, 1 if completed else 0, now, job_id),
            )
        return self.load_video_job(job_id)

    def queue_status(self) -> dict[str, int]:
        self.ensure_db()
        now = time.time()
        result = {"queued": 0, "running": 0, "completed": 0, "completed_with_warning": 0, "failed": 0, "skipped": 0}
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, lease_until FROM video_jobs WHERE job_kind = ?",
                (VIDEO_PROCESSING_JOB_KIND,),
            ).fetchall()
        for status, lease_until in rows:
            if status == "running" and (lease_until is None or float(lease_until) < now):
                result["queued"] += 1
            else:
                result[str(status)] = result.get(str(status), 0) + 1
        return result

    def latest_youtube_discovery_job_for_video(self, video_id: str) -> VideoJobRecord | None:
        self.ensure_db()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, channel_job_id, video_id, canonical_url, status, current_stage, priority,
                       lease_owner, lease_until, cancel_requested, error, warning, quality_mode, acceptance_mode, job_kind,
                       created_at, updated_at, completed_at
                FROM video_jobs
                WHERE video_id = ? AND job_kind = ?
                ORDER BY
                    CASE WHEN status IN ('queued', 'running') THEN 0 ELSE 1 END,
                    updated_at DESC,
                    created_at DESC,
                    id DESC
                LIMIT 1
                """,
                (video_id, YOUTUBE_DISCOVERY_JOB_KIND),
            ).fetchone()
        return _video_job_from_row(row) if row is not None else None

    def ensure_stage_plan(
        self,
        video_job_id: str,
        *,
        whisper_backend: str = "openai-whisper-cpu",
        enable_deep_graph_llm: bool = True,
        include_visual_frame_analysis: bool = False,
    ) -> list[StagePlanRecord]:
        from yt_sub_hub.stages import PIPELINE_STAGES
        from yt_sub_hub.whisper_backends import whisper_resource_lock

        self.ensure_db()
        job = self.load_video_job(video_job_id)
        if job.job_kind != VIDEO_PROCESSING_JOB_KIND:
            raise ValueError(f"core stage plan requires job_kind={VIDEO_PROCESSING_JOB_KIND}: {video_job_id}")
        now = time.time()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT COUNT(*) FROM stage_plans WHERE video_job_id = ?",
                (video_job_id,),
            ).fetchone()[0]
            if not existing:
                for position, stage in enumerate(PIPELINE_STAGES):
                    status = "pending"
                    resource_lock = stage.resource_lock
                    if stage.name == "whisper_transcription":
                        resource_lock = whisper_resource_lock(whisper_backend)
                    if stage.name == "deep_graph_llm" and not enable_deep_graph_llm:
                        status = "skipped"
                    if stage.name == "visual_frame_analysis" and not include_visual_frame_analysis:
                        status = "skipped"
                    conn.execute(
                        """
                        INSERT INTO stage_plans
                          (id, video_job_id, stage_name, status, position, attempt, max_attempts, hard_failure,
                           resource_lock, resource_slot, lease_owner, lease_until, retry_after, error, warning,
                           stage_payload_json, created_at, updated_at, completed_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            _stable_id("stage-plan", video_job_id, stage.name, length=24),
                            video_job_id,
                            stage.name,
                            status,
                            position,
                            0,
                            stage.max_attempts,
                            1 if stage.hard_failure else 0,
                            resource_lock,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            now,
                            now,
                            now if status == "skipped" else None,
                        ),
                    )
                if job.status not in {"completed", "completed_with_warning", "failed", "skipped"}:
                    conn.execute(
                        "UPDATE video_jobs SET status = 'queued', updated_at = ? WHERE id = ?",
                        (now, video_job_id),
                    )
            else:
                self._migrate_legacy_transcript_translation_plan(conn, video_job_id, now)
                self._migrate_missing_analysis_reduce_stage(conn, video_job_id, now)
                self._migrate_missing_analysis_ru_review_stage(conn, video_job_id, now)
                self._migrate_missing_ru_artifact_review_stage(conn, video_job_id, now)
        return self.list_stage_plan(video_job_id)

    def queue_youtube_discovery(
        self,
        video_id: str,
        *,
        limit: int = 20,
        max_quota_units: int = 900,
        priority: int = 100,
        force: bool = False,
    ) -> VideoJobRecord:
        canonical_url = f"https://www.youtube.com/watch?v={video_id}"
        job = self.create_video_job(
            video_id=video_id,
            canonical_url=canonical_url,
            priority=priority,
            job_kind=YOUTUBE_DISCOVERY_JOB_KIND,
        )
        if job.status in {"completed", "completed_with_warning"} and not force:
            return job
        if force:
            self._reset_youtube_discovery_job(job.id)
        self.ensure_youtube_discovery_stage_plan(job.id, limit=limit, max_quota_units=max_quota_units)
        return self.load_video_job(job.id)

    def _reset_youtube_discovery_job(self, video_job_id: str) -> None:
        self.ensure_db()
        now = time.time()
        static_placeholders = ",".join("?" for _ in DISCOVERY_STATIC_STAGES)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT stage_name, resource_slot, lease_owner
                FROM stage_plans
                WHERE video_job_id = ?
                  AND (
                    stage_name IN ({})
                    OR (stage_name LIKE 'youtube_discovery_query_%' AND stage_name != 'youtube_discovery_query_expand')
                  )
                """.format(static_placeholders),
                (video_job_id, *DISCOVERY_STATIC_STAGES),
            ).fetchall()
            stage_names = [str(row[0]) for row in rows]
            for _, resource_slot, owner in rows:
                if resource_slot and owner:
                    conn.execute("DELETE FROM locks WHERE lock_name = ? AND owner = ?", (resource_slot, owner))
            if stage_names:
                stage_placeholders = ",".join("?" for _ in stage_names)
                conn.execute(
                    f"""
                    DELETE FROM retries
                    WHERE stage_run_id IN (
                        SELECT id FROM stage_runs
                        WHERE video_job_id = ? AND stage_name IN ({stage_placeholders})
                    )
                    """,
                    (video_job_id, *stage_names),
                )
                conn.execute(
                    f"""
                    UPDATE stage_runs
                    SET status = 'failed',
                        finished_at = ?,
                        error_class = 'RuntimeReset',
                        error_message = 'youtube discovery reset for retry'
                    WHERE video_job_id = ?
                      AND status = 'running'
                      AND stage_name IN ({stage_placeholders})
                    """,
                    (now, video_job_id, *stage_names),
                )
            conn.execute(
                """
                DELETE FROM stage_plans
                WHERE video_job_id = ?
                  AND stage_name LIKE 'youtube_discovery_query_%'
                  AND stage_name != 'youtube_discovery_query_expand'
                """,
                (video_job_id,),
            )
            conn.execute(
                """
                UPDATE stage_plans
                SET status = 'pending',
                    attempt = 0,
                    retry_after = NULL,
                    error = NULL,
                    warning = NULL,
                    lease_owner = NULL,
                    lease_until = NULL,
                    resource_slot = NULL,
                    completed_at = NULL,
                    updated_at = ?
                WHERE video_job_id = ?
                  AND stage_name IN ({})
                """.format(static_placeholders),
                (now, video_job_id, *DISCOVERY_STATIC_STAGES),
            )
            conn.execute(
                """
                UPDATE video_jobs
                SET status = 'queued',
                    current_stage = 'youtube_discovery_seed_extract',
                    lease_owner = NULL,
                    lease_until = NULL,
                    cancel_requested = 0,
                    error = NULL,
                    warning = NULL,
                    completed_at = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, video_job_id),
            )

    def ensure_youtube_discovery_stage_plan(
        self,
        video_job_id: str,
        *,
        limit: int = 20,
        max_quota_units: int = 900,
    ) -> list[StagePlanRecord]:
        from yt_sub_hub.stages import stage_by_name

        self.ensure_db()
        job = self.load_video_job(video_job_id)
        if job.job_kind != YOUTUBE_DISCOVERY_JOB_KIND:
            raise ValueError(f"youtube discovery stage plan requires job_kind={YOUTUBE_DISCOVERY_JOB_KIND}: {video_job_id}")
        now = time.time()
        payload = {
            "kind": "youtube_discovery",
            "limit": int(limit),
            "max_quota_units": int(max_quota_units),
        }
        payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            max_position_row = conn.execute(
                "SELECT MAX(position) FROM stage_plans WHERE video_job_id = ?",
                (video_job_id,),
            ).fetchone()
            existing_discovery = conn.execute(
                """
                SELECT stage_name, position
                FROM stage_plans
                WHERE video_job_id = ? AND stage_name IN ({})
                ORDER BY position
                """.format(",".join("?" for _ in DISCOVERY_STATIC_STAGES)),
                (video_job_id, *DISCOVERY_STATIC_STAGES),
            ).fetchall()
            start_position = (
                min(int(row[1]) for row in existing_discovery)
                if existing_discovery
                else int(max_position_row[0] if max_position_row and max_position_row[0] is not None else -1) + 1
            )
            for offset, stage_name in enumerate(DISCOVERY_STATIC_STAGES):
                definition = stage_by_name(stage_name)
                conn.execute(
                    """
                    INSERT INTO stage_plans
                      (id, video_job_id, stage_name, status, position, attempt, max_attempts, hard_failure,
                       resource_lock, resource_slot, lease_owner, lease_until, retry_after, error, warning,
                       stage_payload_json, created_at, updated_at, completed_at)
                    VALUES (?, ?, ?, 'pending', ?, 0, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, ?, ?, ?, NULL)
                    ON CONFLICT(video_job_id, stage_name) DO UPDATE SET
                        position = excluded.position,
                        max_attempts = excluded.max_attempts,
                        hard_failure = excluded.hard_failure,
                        resource_lock = excluded.resource_lock,
                        stage_payload_json = excluded.stage_payload_json,
                        status = CASE
                            WHEN stage_plans.stage_payload_json IS NOT excluded.stage_payload_json THEN 'pending'
                            ELSE stage_plans.status
                        END,
                        attempt = CASE
                            WHEN stage_plans.stage_payload_json IS NOT excluded.stage_payload_json THEN 0
                            ELSE stage_plans.attempt
                        END,
                        resource_slot = CASE
                            WHEN stage_plans.stage_payload_json IS NOT excluded.stage_payload_json THEN NULL
                            ELSE stage_plans.resource_slot
                        END,
                        lease_owner = CASE
                            WHEN stage_plans.stage_payload_json IS NOT excluded.stage_payload_json THEN NULL
                            ELSE stage_plans.lease_owner
                        END,
                        lease_until = CASE
                            WHEN stage_plans.stage_payload_json IS NOT excluded.stage_payload_json THEN NULL
                            ELSE stage_plans.lease_until
                        END,
                        retry_after = CASE
                            WHEN stage_plans.stage_payload_json IS NOT excluded.stage_payload_json THEN NULL
                            ELSE stage_plans.retry_after
                        END,
                        error = CASE
                            WHEN stage_plans.stage_payload_json IS NOT excluded.stage_payload_json THEN NULL
                            ELSE stage_plans.error
                        END,
                        warning = CASE
                            WHEN stage_plans.stage_payload_json IS NOT excluded.stage_payload_json THEN NULL
                            ELSE stage_plans.warning
                        END,
                        completed_at = CASE
                            WHEN stage_plans.stage_payload_json IS NOT excluded.stage_payload_json THEN NULL
                            ELSE stage_plans.completed_at
                        END,
                        updated_at = excluded.updated_at
                    """,
                    (
                        _stable_id("stage-plan", video_job_id, stage_name, length=24),
                        video_job_id,
                        stage_name,
                        start_position + offset,
                        definition.max_attempts,
                        1 if definition.hard_failure else 0,
                        definition.resource_lock,
                        payload_json,
                        now,
                        now,
                    ),
                )
            conn.execute(
                """
                UPDATE video_jobs
                SET status = 'queued',
                    current_stage = 'youtube_discovery_seed_extract',
                    lease_owner = NULL,
                    lease_until = NULL,
                    cancel_requested = 0,
                    error = NULL,
                    warning = NULL,
                    completed_at = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, video_job_id),
            )
        return self.list_stage_plan(video_job_id)

    def expand_youtube_discovery_query_plan(self, video_job_id: str, queries: dict[str, Any]) -> list[StagePlanRecord]:
        self.ensure_db()
        job = self.load_video_job(video_job_id)
        if job.job_kind != YOUTUBE_DISCOVERY_JOB_KIND:
            raise ValueError(f"youtube discovery query plan requires job_kind={YOUTUBE_DISCOVERY_JOB_KIND}: {video_job_id}")
        now = time.time()
        query_items = [item for item in queries.get("items") or [] if isinstance(item, dict) and not item.get("rejected_reason")]
        total_queries = len(query_items)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            expand_row = conn.execute(
                "SELECT position FROM stage_plans WHERE video_job_id = ? AND stage_name = 'youtube_discovery_query_expand'",
                (video_job_id,),
            ).fetchone()
            if expand_row is None:
                raise KeyError(f"youtube_discovery_query_expand is not planned for job: {video_job_id}")
            query_position = int(expand_row[0]) + 1
            downstream = [
                "youtube_discovery_candidate_fetch",
                "youtube_discovery_candidate_score",
                "youtube_discovery_review",
                "youtube_discovery_graph",
                "youtube_discovery_report",
            ]
            for offset, stage_name in enumerate(downstream, start=1):
                conn.execute(
                    """
                    UPDATE stage_plans
                    SET position = ?, updated_at = ?
                    WHERE video_job_id = ? AND stage_name = ?
                    """,
                    (query_position + offset, now, video_job_id, stage_name),
                )
            existing = conn.execute(
                """
                SELECT stage_name, resource_slot, lease_owner, stage_payload_json
                FROM stage_plans
                WHERE video_job_id = ? AND stage_name LIKE 'youtube_discovery_query_%'
                  AND stage_name != 'youtube_discovery_query_expand'
                """,
                (video_job_id,),
            ).fetchall()
            existing_payloads = {str(row[0]): row[3] for row in existing}
            existing_locks = {str(row[0]): (row[1], row[2]) for row in existing}
            desired_stage_names = {f"youtube_discovery_query_{index:03d}" for index, _ in enumerate(query_items)}
            for stage_name, resource_slot, owner, _ in existing:
                stage_name = str(stage_name)
                if stage_name in desired_stage_names:
                    continue
                if resource_slot and owner:
                    conn.execute("DELETE FROM locks WHERE lock_name = ? AND owner = ?", (resource_slot, owner))
                conn.execute(
                    "DELETE FROM stage_plans WHERE video_job_id = ? AND stage_name = ?",
                    (video_job_id, stage_name),
                )
            for index, query in enumerate(query_items):
                stage_name = f"youtube_discovery_query_{index:03d}"
                payload = {
                    "query_index": index,
                    "total_queries": total_queries,
                    "query_id": str(query.get("query_id") or f"ydq-{index + 1:03d}"),
                    "language": query.get("language"),
                    "intent": query.get("intent"),
                    "query_source": query.get("query_source"),
                    "query": query,
                    "queries_artifact_hash": queries.get("artifact_hash"),
                    "input_hashes": queries.get("input_hashes") or {},
                }
                payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
                conn.execute(
                    """
                    INSERT INTO stage_plans
                      (id, video_job_id, stage_name, status, position, attempt, max_attempts, hard_failure,
                       resource_lock, resource_slot, lease_owner, lease_until, retry_after, error, warning,
                       stage_payload_json, created_at, updated_at, completed_at)
                    VALUES (?, ?, ?, 'pending', ?, 0, 2, 1, 'youtube_data_api', NULL, NULL, NULL, NULL, NULL, NULL, ?, ?, ?, NULL)
                    ON CONFLICT(video_job_id, stage_name) DO UPDATE SET
                        position = excluded.position,
                        resource_lock = excluded.resource_lock,
                        stage_payload_json = excluded.stage_payload_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        _stable_id("stage-plan", video_job_id, stage_name, length=24),
                        video_job_id,
                        stage_name,
                        query_position,
                        payload_json,
                        now,
                        now,
                    ),
                )
                if existing_payloads.get(stage_name) not in {None, payload_json}:
                    resource_slot, owner = existing_locks.get(stage_name) or (None, None)
                    if resource_slot and owner:
                        conn.execute("DELETE FROM locks WHERE lock_name = ? AND owner = ?", (resource_slot, owner))
                    conn.execute(
                        """
                        UPDATE stage_plans
                        SET status = 'pending',
                            attempt = 0,
                            resource_slot = NULL,
                            lease_owner = NULL,
                            lease_until = NULL,
                            retry_after = NULL,
                            error = NULL,
                            warning = NULL,
                            completed_at = NULL,
                            updated_at = ?
                        WHERE video_job_id = ? AND stage_name = ?
                        """,
                        (now, video_job_id, stage_name),
                    )
            conn.execute(
                """
                UPDATE video_jobs
                SET status = 'queued', lease_owner = NULL, lease_until = NULL, updated_at = ?
                WHERE id = ? AND status NOT IN ('completed', 'completed_with_warning', 'failed', 'skipped')
                """,
                (now, video_job_id),
            )
        return self.list_stage_plan(video_job_id)

    def list_stage_plan(self, video_job_id: str | None = None) -> list[StagePlanRecord]:
        self.ensure_db()
        with self._connect() as conn:
            if video_job_id is None:
                rows = conn.execute(
                    """
                    SELECT id, video_job_id, stage_name, status, position, attempt, max_attempts, hard_failure,
                           resource_lock, resource_slot, lease_owner, lease_until, retry_after, error, warning, stage_payload_json,
                           created_at, updated_at, completed_at
                    FROM stage_plans ORDER BY video_job_id, position, stage_name
                    """
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, video_job_id, stage_name, status, position, attempt, max_attempts, hard_failure,
                           resource_lock, resource_slot, lease_owner, lease_until, retry_after, error, warning, stage_payload_json,
                           created_at, updated_at, completed_at
                    FROM stage_plans WHERE video_job_id = ? ORDER BY position, stage_name
                    """,
                    (video_job_id,),
                ).fetchall()
        return [_stage_plan_from_row(row) for row in rows]

    def expand_translation_chunk_plan(self, video_job_id: str, chunks: list[dict[str, Any]]) -> list[StagePlanRecord]:
        self.ensure_db()
        job = self.load_video_job(video_job_id)
        if job.job_kind != VIDEO_PROCESSING_JOB_KIND:
            raise ValueError(f"translation chunk plan requires job_kind={VIDEO_PROCESSING_JOB_KIND}: {video_job_id}")
        now = time.time()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            expand_row = conn.execute(
                "SELECT position FROM stage_plans WHERE video_job_id = ? AND stage_name = 'transcript_translation_expand'",
                (video_job_id,),
            ).fetchone()
            if expand_row is None:
                raise KeyError(f"transcript_translation_expand is not planned for job: {video_job_id}")
            chunk_position = int(expand_row[0]) + 1
            existing_chunks = conn.execute(
                """
                SELECT stage_name FROM stage_plans
                WHERE video_job_id = ? AND stage_name LIKE 'transcript_translation_chunk_%'
                """,
                (video_job_id,),
            ).fetchall()
            if not existing_chunks:
                conn.execute(
                    """
                    UPDATE stage_plans
                    SET position = position + 1, updated_at = ?
                    WHERE video_job_id = ? AND position >= ?
                    """,
                    (now, video_job_id, chunk_position),
                )
            total_chunks = len(chunks)
            for chunk in chunks:
                chunk_index = int(chunk["chunk_index"])
                payload = {**chunk, "chunk_index": chunk_index, "total_chunks": total_chunks}
                stage_name = f"transcript_translation_chunk_{chunk_index:03d}"
                conn.execute(
                    """
                    INSERT INTO stage_plans
                      (id, video_job_id, stage_name, status, position, attempt, max_attempts, hard_failure,
                       resource_lock, resource_slot, lease_owner, lease_until, retry_after, error, warning,
                       stage_payload_json, created_at, updated_at, completed_at)
                    VALUES (?, ?, ?, 'pending', ?, 0, 2, 1, 'lm_studio_translation', NULL, NULL, NULL, NULL, NULL, NULL, ?, ?, ?, NULL)
                    ON CONFLICT(video_job_id, stage_name) DO UPDATE SET
                        position = excluded.position,
                        status = CASE
                            WHEN stage_plans.stage_payload_json IS NOT excluded.stage_payload_json THEN 'pending'
                            ELSE stage_plans.status
                        END,
                        attempt = CASE
                            WHEN stage_plans.stage_payload_json IS NOT excluded.stage_payload_json THEN 0
                            ELSE stage_plans.attempt
                        END,
                        resource_lock = excluded.resource_lock,
                        resource_slot = CASE
                            WHEN stage_plans.stage_payload_json IS NOT excluded.stage_payload_json THEN NULL
                            ELSE stage_plans.resource_slot
                        END,
                        lease_owner = CASE
                            WHEN stage_plans.stage_payload_json IS NOT excluded.stage_payload_json THEN NULL
                            ELSE stage_plans.lease_owner
                        END,
                        lease_until = CASE
                            WHEN stage_plans.stage_payload_json IS NOT excluded.stage_payload_json THEN NULL
                            ELSE stage_plans.lease_until
                        END,
                        retry_after = CASE
                            WHEN stage_plans.stage_payload_json IS NOT excluded.stage_payload_json THEN NULL
                            ELSE stage_plans.retry_after
                        END,
                        error = CASE
                            WHEN stage_plans.stage_payload_json IS NOT excluded.stage_payload_json THEN NULL
                            ELSE stage_plans.error
                        END,
                        warning = CASE
                            WHEN stage_plans.stage_payload_json IS NOT excluded.stage_payload_json THEN NULL
                            ELSE stage_plans.warning
                        END,
                        completed_at = CASE
                            WHEN stage_plans.stage_payload_json IS NOT excluded.stage_payload_json THEN NULL
                            ELSE stage_plans.completed_at
                        END,
                        stage_payload_json = excluded.stage_payload_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        _stable_id("stage-plan", video_job_id, stage_name, length=24),
                        video_job_id,
                        stage_name,
                        chunk_position,
                        json.dumps(payload, ensure_ascii=False, sort_keys=True),
                        now,
                        now,
                    ),
                )
            conn.execute(
                """
                UPDATE stage_plans
                SET status = CASE WHEN status = 'skipped' THEN 'pending' ELSE status END,
                    updated_at = ?
                WHERE video_job_id = ? AND stage_name = 'transcript_translation_merge'
                """,
                (now, video_job_id),
            )
            conn.execute(
                """
                UPDATE video_jobs
                SET status = 'queued', lease_owner = NULL, lease_until = NULL, updated_at = ?
                WHERE id = ? AND status NOT IN ('completed', 'completed_with_warning', 'failed', 'skipped')
                """,
                (now, video_job_id),
            )
        return self.list_stage_plan(video_job_id)

    def expand_analysis_chunk_plan(self, video_job_id: str, chunks: list[dict[str, Any]]) -> list[StagePlanRecord]:
        self.ensure_db()
        job = self.load_video_job(video_job_id)
        if job.job_kind != VIDEO_PROCESSING_JOB_KIND:
            raise ValueError(f"analysis chunk plan requires job_kind={VIDEO_PROCESSING_JOB_KIND}: {video_job_id}")
        now = time.time()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            expand_row = conn.execute(
                "SELECT position FROM stage_plans WHERE video_job_id = ? AND stage_name = 'scientific_analysis'",
                (video_job_id,),
            ).fetchone()
            if expand_row is None:
                raise KeyError(f"scientific_analysis is not planned for job: {video_job_id}")
            chunk_position = int(expand_row[0]) + 1
            if conn.execute(
                "SELECT 1 FROM stage_plans WHERE video_job_id = ? AND stage_name = 'analysis_reduce'",
                (video_job_id,),
            ).fetchone() is None:
                conn.execute(
                    """
                    UPDATE stage_plans
                    SET position = position + 1, updated_at = ?
                    WHERE video_job_id = ? AND position >= ?
                    """,
                    (now, video_job_id, chunk_position),
                )
                conn.execute(
                    """
                    INSERT INTO stage_plans
                      (id, video_job_id, stage_name, status, position, attempt, max_attempts, hard_failure,
                       resource_lock, resource_slot, lease_owner, lease_until, retry_after, error, warning,
                       stage_payload_json, created_at, updated_at, completed_at)
                    VALUES (?, ?, 'analysis_reduce', 'pending', ?, 0, 1, 1,
                            'local_validation', NULL, NULL, NULL, NULL, NULL, NULL, NULL, ?, ?, NULL)
                    """,
                    (
                        _stable_id("stage-plan", video_job_id, "analysis_reduce", length=24),
                        video_job_id,
                        chunk_position,
                        now,
                        now,
                    ),
                )
            existing_chunks = conn.execute(
                """
                SELECT stage_name, resource_slot, lease_owner, stage_payload_json FROM stage_plans
                WHERE video_job_id = ? AND stage_name LIKE 'analysis_chunk_%'
                """,
                (video_job_id,),
            ).fetchall()
            existing_payloads = {row[0]: row[3] for row in existing_chunks}
            existing_locks = {row[0]: (row[1], row[2]) for row in existing_chunks}
            if not existing_chunks:
                conn.execute(
                    """
                    UPDATE stage_plans
                    SET position = position + 1, updated_at = ?
                    WHERE video_job_id = ? AND position >= ?
                    """,
                    (now, video_job_id, chunk_position),
                )
            total_chunks = len(chunks)
            desired_stage_names = {f"analysis_chunk_{int(chunk['chunk_index']):03d}" for chunk in chunks}
            for stage_name, resource_slot, owner, _ in existing_chunks:
                if stage_name in desired_stage_names:
                    continue
                if resource_slot and owner:
                    conn.execute("DELETE FROM locks WHERE lock_name = ? AND owner = ?", (resource_slot, owner))
                conn.execute(
                    "DELETE FROM stage_plans WHERE video_job_id = ? AND stage_name = ?",
                    (video_job_id, stage_name),
                )
            for chunk in chunks:
                chunk_index = int(chunk["chunk_index"])
                payload = {**chunk, "chunk_index": chunk_index, "total_chunks": total_chunks}
                stage_name = f"analysis_chunk_{chunk_index:03d}"
                payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
                conn.execute(
                    """
                    INSERT INTO stage_plans
                      (id, video_job_id, stage_name, status, position, attempt, max_attempts, hard_failure,
                       resource_lock, resource_slot, lease_owner, lease_until, retry_after, error, warning,
                       stage_payload_json, created_at, updated_at, completed_at)
                    VALUES (?, ?, ?, 'pending', ?, 0, 2, 1, 'lm_studio_analysis', NULL, NULL, NULL, NULL, NULL, NULL, ?, ?, ?, NULL)
                    ON CONFLICT(video_job_id, stage_name) DO UPDATE SET
                        position = excluded.position,
                        resource_lock = excluded.resource_lock,
                        stage_payload_json = excluded.stage_payload_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        _stable_id("stage-plan", video_job_id, stage_name, length=24),
                        video_job_id,
                        stage_name,
                        chunk_position,
                        payload_json,
                        now,
                        now,
                    ),
                )
                if existing_payloads.get(stage_name) not in {None, payload_json}:
                    resource_slot, owner = existing_locks.get(stage_name) or (None, None)
                    if resource_slot and owner:
                        conn.execute("DELETE FROM locks WHERE lock_name = ? AND owner = ?", (resource_slot, owner))
                    conn.execute(
                        """
                        UPDATE stage_plans
                        SET status = 'pending',
                            attempt = 0,
                            resource_slot = NULL,
                            lease_owner = NULL,
                            lease_until = NULL,
                            retry_after = NULL,
                            error = NULL,
                            warning = NULL,
                            completed_at = NULL,
                            updated_at = ?
                        WHERE video_job_id = ? AND stage_name = ?
                        """,
                        (now, video_job_id, stage_name),
                    )
            conn.execute(
                """
                UPDATE stage_plans
                SET status = CASE WHEN status = 'skipped' THEN 'pending' ELSE status END,
                    updated_at = ?
                WHERE video_job_id = ? AND stage_name = 'analysis_reduce'
                """,
                (now, video_job_id),
            )
            conn.execute(
                """
                UPDATE video_jobs
                SET status = 'queued', lease_owner = NULL, lease_until = NULL, updated_at = ?
                WHERE id = ? AND status NOT IN ('completed', 'completed_with_warning', 'failed', 'skipped')
                """,
                (now, video_job_id),
            )
        return self.list_stage_plan(video_job_id)

    def _migrate_legacy_transcript_translation_plan(self, conn, video_job_id: str, now: float) -> None:
        legacy = conn.execute(
            """
            SELECT position, status, attempt, max_attempts, hard_failure, retry_after, error, warning, completed_at
            FROM stage_plans
            WHERE video_job_id = ? AND stage_name = 'transcript_translation'
            """,
            (video_job_id,),
        ).fetchone()
        if legacy is None:
            return
        if conn.execute(
            "SELECT 1 FROM stage_plans WHERE video_job_id = ? AND stage_name = 'transcript_translation_expand'",
            (video_job_id,),
        ).fetchone():
            return
        position = int(legacy[0])
        conn.execute(
            """
            UPDATE stage_plans
            SET stage_name = 'transcript_translation_expand',
                resource_lock = 'local_validation',
                status = 'pending',
                attempt = 0,
                max_attempts = 1,
                hard_failure = 1,
                retry_after = NULL,
                error = NULL,
                warning = NULL,
                completed_at = NULL,
                updated_at = ?
            WHERE video_job_id = ? AND stage_name = 'transcript_translation'
            """,
            (now, video_job_id),
        )
        conn.execute(
            """
            UPDATE stage_plans
            SET position = position + 1, updated_at = ?
            WHERE video_job_id = ? AND position > ?
            """,
            (now, video_job_id, position),
        )
        conn.execute(
            """
            INSERT INTO stage_plans
              (id, video_job_id, stage_name, status, position, attempt, max_attempts, hard_failure,
               resource_lock, resource_slot, lease_owner, lease_until, retry_after, error, warning,
               stage_payload_json, created_at, updated_at, completed_at)
            VALUES (?, ?, 'transcript_translation_merge', 'pending', ?, 0, 1, 1,
                    'local_validation', NULL, NULL, NULL, NULL, NULL, NULL, NULL, ?, ?, NULL)
            ON CONFLICT(video_job_id, stage_name) DO NOTHING
            """,
            (
                _stable_id("stage-plan", video_job_id, "transcript_translation_merge", length=24),
                video_job_id,
                position + 1,
                now,
                now,
            ),
        )
        conn.execute(
            """
            UPDATE video_jobs
            SET current_stage = CASE WHEN current_stage = 'transcript_translation' THEN 'transcript_translation_expand' ELSE current_stage END,
                updated_at = ?
            WHERE id = ?
            """,
            (now, video_job_id),
        )

    def _migrate_missing_analysis_reduce_stage(self, conn, video_job_id: str, now: float) -> None:
        if conn.execute(
            "SELECT 1 FROM stage_plans WHERE video_job_id = ? AND stage_name = 'analysis_reduce'",
            (video_job_id,),
        ).fetchone():
            conn.execute(
                """
                UPDATE stage_plans
                SET resource_lock = 'local_validation',
                    max_attempts = 1,
                    hard_failure = 1,
                    updated_at = ?
                WHERE video_job_id = ? AND stage_name = 'analysis_reduce'
                """,
                (now, video_job_id),
            )
            return
        analysis_row = conn.execute(
            """
            SELECT position
            FROM stage_plans
            WHERE video_job_id = ? AND stage_name = 'scientific_analysis'
            """,
            (video_job_id,),
        ).fetchone()
        if analysis_row is None:
            return
        job_row = conn.execute("SELECT status FROM video_jobs WHERE id = ?", (video_job_id,)).fetchone()
        terminal = job_row is not None and job_row[0] in {"completed", "completed_with_warning", "failed", "skipped"}
        insert_position = int(analysis_row[0]) + 1
        conn.execute(
            """
            UPDATE stage_plans
            SET position = position + 1, updated_at = ?
            WHERE video_job_id = ? AND position >= ?
            """,
            (now, video_job_id, insert_position),
        )
        conn.execute(
            """
            INSERT INTO stage_plans
              (id, video_job_id, stage_name, status, position, attempt, max_attempts, hard_failure,
               resource_lock, resource_slot, lease_owner, lease_until, retry_after, error, warning,
               stage_payload_json, created_at, updated_at, completed_at)
            VALUES (?, ?, 'analysis_reduce', 'pending', ?, 0, 1, 1,
                    'local_validation', NULL, NULL, NULL, NULL, NULL, NULL, NULL, ?, ?, NULL)
            """,
            (
                _stable_id("stage-plan", video_job_id, "analysis_reduce", length=24),
                video_job_id,
                insert_position,
                now,
                now,
            ),
        )
        if terminal:
            return
        reset_rows = conn.execute(
            """
            SELECT stage_name, resource_slot, lease_owner
            FROM stage_plans
            WHERE video_job_id = ?
              AND position > ?
              AND status != 'skipped'
            """,
            (video_job_id, insert_position),
        ).fetchall()
        for _, resource_slot, owner in reset_rows:
            if resource_slot and owner:
                conn.execute("DELETE FROM locks WHERE lock_name = ? AND owner = ?", (resource_slot, owner))
        conn.execute(
            """
            UPDATE stage_runs
            SET status = 'failed',
                finished_at = ?,
                error_class = 'RuntimeMigration',
                error_message = 'stage reset for analysis_reduce migration'
            WHERE video_job_id = ?
              AND status = 'running'
              AND stage_name IN (
                  SELECT stage_name FROM stage_plans
                  WHERE video_job_id = ? AND position > ? AND status != 'skipped'
              )
            """,
            (now, video_job_id, video_job_id, insert_position),
        )
        conn.execute(
            """
            UPDATE stage_plans
            SET status = 'pending',
                attempt = 0,
                retry_after = NULL,
                error = NULL,
                warning = NULL,
                lease_owner = NULL,
                lease_until = NULL,
                resource_slot = NULL,
                completed_at = NULL,
                created_at = ?,
                updated_at = ?
            WHERE video_job_id = ?
              AND position > ?
              AND status != 'skipped'
            """,
            (now, now, video_job_id, insert_position),
        )
        conn.execute(
            """
            UPDATE video_jobs
            SET status = 'queued',
                current_stage = 'analysis_reduce',
                lease_owner = NULL,
                lease_until = NULL,
                error = NULL,
                warning = NULL,
                completed_at = NULL,
                updated_at = ?
            WHERE id = ?
              AND status NOT IN ('completed', 'completed_with_warning', 'failed', 'skipped')
            """,
            (now, video_job_id),
        )

    def _migrate_missing_analysis_ru_review_stage(self, conn, video_job_id: str, now: float) -> None:
        if conn.execute(
            "SELECT 1 FROM stage_plans WHERE video_job_id = ? AND stage_name = 'analysis_ru_review'",
            (video_job_id,),
        ).fetchone():
            conn.execute(
                """
                UPDATE stage_plans
                SET resource_lock = 'local_validation',
                    max_attempts = 1,
                    hard_failure = 1,
                    updated_at = ?
                WHERE video_job_id = ? AND stage_name = 'analysis_ru_review'
                """,
                (now, video_job_id),
            )
            return
        analysis_row = conn.execute(
            """
            SELECT position
            FROM stage_plans
            WHERE video_job_id = ? AND stage_name = 'analysis_translation'
            """,
            (video_job_id,),
        ).fetchone()
        if analysis_row is None:
            return
        job_row = conn.execute("SELECT status FROM video_jobs WHERE id = ?", (video_job_id,)).fetchone()
        terminal = job_row is not None and job_row[0] in {"completed", "completed_with_warning", "failed", "skipped"}
        insert_position = int(analysis_row[0]) + 1
        conn.execute(
            """
            UPDATE stage_plans
            SET position = position + 1, updated_at = ?
            WHERE video_job_id = ? AND position >= ?
            """,
            (now, video_job_id, insert_position),
        )
        conn.execute(
            """
            INSERT INTO stage_plans
              (id, video_job_id, stage_name, status, position, attempt, max_attempts, hard_failure,
               resource_lock, resource_slot, lease_owner, lease_until, retry_after, error, warning,
               stage_payload_json, created_at, updated_at, completed_at)
            VALUES (?, ?, 'analysis_ru_review', 'pending', ?, 0, 1, 1,
                    'local_validation', NULL, NULL, NULL, NULL, NULL, NULL, NULL, ?, ?, NULL)
            """,
            (
                _stable_id("stage-plan", video_job_id, "analysis_ru_review", length=24),
                video_job_id,
                insert_position,
                now,
                now,
            ),
        )
        if terminal:
            return
        reset_rows = conn.execute(
            """
            SELECT stage_name, resource_slot, lease_owner
            FROM stage_plans
            WHERE video_job_id = ?
              AND position > ?
              AND status != 'skipped'
            """,
            (video_job_id, insert_position),
        ).fetchall()
        for _, resource_slot, owner in reset_rows:
            if resource_slot and owner:
                conn.execute("DELETE FROM locks WHERE lock_name = ? AND owner = ?", (resource_slot, owner))
        conn.execute(
            """
            UPDATE stage_runs
            SET status = 'failed',
                finished_at = ?,
                error_class = 'RuntimeMigration',
                error_message = 'stage reset for analysis_ru_review migration'
            WHERE video_job_id = ?
              AND status = 'running'
              AND stage_name IN (
                  SELECT stage_name FROM stage_plans
                  WHERE video_job_id = ? AND position > ? AND status != 'skipped'
              )
            """,
            (now, video_job_id, video_job_id, insert_position),
        )
        conn.execute(
            """
            UPDATE stage_plans
            SET status = 'pending',
                attempt = 0,
                retry_after = NULL,
                error = NULL,
                warning = NULL,
                lease_owner = NULL,
                lease_until = NULL,
                resource_slot = NULL,
                completed_at = NULL,
                created_at = ?,
                updated_at = ?
            WHERE video_job_id = ?
              AND position > ?
              AND status != 'skipped'
            """,
            (now, now, video_job_id, insert_position),
        )
        conn.execute(
            """
            UPDATE video_jobs
            SET status = 'queued',
                current_stage = 'analysis_ru_review',
                lease_owner = NULL,
                lease_until = NULL,
                error = NULL,
                warning = NULL,
                completed_at = NULL,
                updated_at = ?
            WHERE id = ?
              AND status NOT IN ('completed', 'completed_with_warning', 'failed', 'skipped')
            """,
            (now, video_job_id),
        )

    def _migrate_missing_ru_artifact_review_stage(self, conn, video_job_id: str, now: float) -> None:
        if conn.execute(
            "SELECT 1 FROM stage_plans WHERE video_job_id = ? AND stage_name = 'ru_artifact_review'",
            (video_job_id,),
        ).fetchone():
            conn.execute(
                """
                UPDATE stage_plans
                SET resource_lock = 'critic_review',
                    max_attempts = 1,
                    hard_failure = 1,
                    updated_at = ?
                WHERE video_job_id = ? AND stage_name = 'ru_artifact_review'
                """,
                (now, video_job_id),
            )
            return
        anchor_row = conn.execute(
            """
            SELECT position
            FROM stage_plans
            WHERE video_job_id = ? AND stage_name = 'neo4j_sync'
            """,
            (video_job_id,),
        ).fetchone()
        if anchor_row is None:
            return
        job_row = conn.execute("SELECT status FROM video_jobs WHERE id = ?", (video_job_id,)).fetchone()
        terminal = job_row is not None and job_row[0] in {"completed", "completed_with_warning", "failed", "skipped"}
        insert_position = int(anchor_row[0])
        conn.execute(
            """
            UPDATE stage_plans
            SET position = position + 1, updated_at = ?
            WHERE video_job_id = ? AND position >= ?
            """,
            (now, video_job_id, insert_position),
        )
        conn.execute(
            """
            INSERT INTO stage_plans
              (id, video_job_id, stage_name, status, position, attempt, max_attempts, hard_failure,
               resource_lock, resource_slot, lease_owner, lease_until, retry_after, error, warning,
               stage_payload_json, created_at, updated_at, completed_at)
            VALUES (?, ?, 'ru_artifact_review', 'pending', ?, 0, 1, 1,
                    'critic_review', NULL, NULL, NULL, NULL, NULL, NULL, NULL, ?, ?, NULL)
            """,
            (
                _stable_id("stage-plan", video_job_id, "ru_artifact_review", length=24),
                video_job_id,
                insert_position,
                now,
                now,
            ),
        )
        if terminal:
            return
        reset_rows = conn.execute(
            """
            SELECT stage_name, resource_slot, lease_owner
            FROM stage_plans
            WHERE video_job_id = ?
              AND position > ?
              AND status != 'skipped'
            """,
            (video_job_id, insert_position),
        ).fetchall()
        for _, resource_slot, owner in reset_rows:
            if resource_slot and owner:
                conn.execute("DELETE FROM locks WHERE lock_name = ? AND owner = ?", (resource_slot, owner))
        conn.execute(
            """
            UPDATE stage_runs
            SET status = 'failed',
                finished_at = ?,
                error_class = 'RuntimeMigration',
                error_message = 'stage reset for ru_artifact_review migration'
            WHERE video_job_id = ?
              AND status = 'running'
              AND stage_name IN (
                  SELECT stage_name FROM stage_plans
                  WHERE video_job_id = ? AND position > ? AND status != 'skipped'
              )
            """,
            (now, video_job_id, video_job_id, insert_position),
        )
        conn.execute(
            """
            UPDATE stage_plans
            SET status = 'pending',
                attempt = 0,
                retry_after = NULL,
                error = NULL,
                warning = NULL,
                lease_owner = NULL,
                lease_until = NULL,
                resource_slot = NULL,
                completed_at = NULL,
                created_at = ?,
                updated_at = ?
            WHERE video_job_id = ?
              AND position > ?
              AND status != 'skipped'
            """,
            (now, now, video_job_id, insert_position),
        )
        conn.execute(
            """
            UPDATE video_jobs
            SET status = 'queued',
                current_stage = 'ru_artifact_review',
                lease_owner = NULL,
                lease_until = NULL,
                error = NULL,
                warning = NULL,
                completed_at = NULL,
                updated_at = ?
            WHERE id = ?
              AND status NOT IN ('completed', 'completed_with_warning', 'failed', 'skipped')
            """,
            (now, video_job_id),
        )

    def reset_failed_stage_plan_for_retry(self, job_id: str, from_stage: str | None = None) -> VideoJobRecord:
        self.ensure_db()
        now = time.time()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            job_row = conn.execute("SELECT id FROM video_jobs WHERE id = ?", (job_id,)).fetchone()
            if job_row is None:
                raise KeyError(f"Unknown video job: {job_id}")
            if from_stage is None:
                start = conn.execute(
                    """
                    SELECT stage_name, position
                    FROM stage_plans
                    WHERE video_job_id = ? AND status = 'failed'
                    ORDER BY position
                    LIMIT 1
                    """,
                    (job_id,),
                ).fetchone()
                if start is None:
                    raise ValueError(f"no failed stage plan for job: {job_id}")
            else:
                start = conn.execute(
                    """
                    SELECT stage_name, position
                    FROM stage_plans
                    WHERE video_job_id = ? AND stage_name = ?
                    """,
                    (job_id, from_stage),
                ).fetchone()
                if start is None:
                    raise KeyError(f"Unknown stage for job {job_id}: {from_stage}")
            start_stage_name, start_position = start
            if from_stage is not None:
                reset_statuses = ("failed", "pending", "running", "completed", "completed_with_warning")
                status_placeholders = ",".join("?" for _ in reset_statuses)
                rows = conn.execute(
                    f"""
                    SELECT stage_name, resource_slot, lease_owner
                    FROM stage_plans
                    WHERE video_job_id = ?
                      AND (stage_name = ? OR position > ?)
                      AND (
                        status IN ({status_placeholders})
                        OR (status = 'skipped' AND (hard_failure = 1 OR warning = 'skipped_requires_media'))
                        OR stage_name = ?
                      )
                    ORDER BY position
                    """,
                    (job_id, start_stage_name, start_position, *reset_statuses, from_stage),
                ).fetchall()
            else:
                reset_statuses = ("failed", "pending", "running")
                downstream_statuses = ("completed", "completed_with_warning")
                status_placeholders = ",".join("?" for _ in reset_statuses)
                downstream_placeholders = ",".join("?" for _ in downstream_statuses)
                rows = conn.execute(
                    f"""
                    SELECT stage_name, resource_slot, lease_owner
                    FROM stage_plans
                    WHERE video_job_id = ?
                      AND (
                          (position >= ? AND status IN ({status_placeholders}))
                          OR (position > ? AND status IN ({downstream_placeholders}))
                      )
                    ORDER BY position
                    """,
                    (job_id, start_position, *reset_statuses, start_position, *downstream_statuses),
                ).fetchall()
            if not rows:
                raise ValueError(f"no resettable stage plan for job: {job_id}")
            reset_stage_names = [row[0] for row in rows]
            first_reset_stage_name = reset_stage_names[0]
            for _, resource_slot, owner in rows:
                if resource_slot and owner:
                    conn.execute("DELETE FROM locks WHERE lock_name = ? AND owner = ?", (resource_slot, owner))
            placeholders = ",".join("?" for _ in reset_stage_names)
            conn.execute(
                f"""
                DELETE FROM retries
                WHERE stage_run_id IN (
                    SELECT id FROM stage_runs
                    WHERE video_job_id = ?
                      AND stage_name IN ({placeholders})
                )
                """,
                (job_id, *reset_stage_names),
            )
            conn.execute(
                f"""
                UPDATE stage_runs
                SET status = 'failed',
                    finished_at = ?,
                    error_class = 'RuntimeReset',
                    error_message = 'stage reset for retry'
                WHERE video_job_id = ?
                  AND status = 'running'
                  AND stage_name IN ({placeholders})
                """,
                (now, job_id, *reset_stage_names),
            )
            conn.execute(
                f"""
                UPDATE stage_plans
                SET status = 'pending',
                    attempt = 0,
                    retry_after = NULL,
                    error = NULL,
                    warning = NULL,
                    lease_owner = NULL,
                    lease_until = NULL,
                    resource_slot = NULL,
                    completed_at = NULL,
                    created_at = ?,
                    updated_at = ?
                WHERE video_job_id = ?
                  AND stage_name IN ({placeholders})
                """,
                (now, now, job_id, *reset_stage_names),
            )
            conn.execute(
                """
                UPDATE video_jobs
                SET status = 'queued',
                    current_stage = ?,
                    lease_owner = NULL,
                    lease_until = NULL,
                    cancel_requested = 0,
                    error = NULL,
                    warning = NULL,
                    completed_at = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (first_reset_stage_name, now, job_id),
            )
        return self.load_video_job(job_id)

    def list_runnable_stages(self) -> list[StagePlanRecord]:
        self.ensure_db()
        now = time.time()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT sp.id, sp.video_job_id, sp.stage_name, sp.status, sp.position, sp.attempt, sp.max_attempts,
                       sp.hard_failure, sp.resource_lock, sp.resource_slot, sp.lease_owner, sp.lease_until,
                       sp.retry_after, sp.error, sp.warning, sp.stage_payload_json, sp.created_at, sp.updated_at, sp.completed_at
                FROM stage_plans sp
                JOIN video_jobs vj ON vj.id = sp.video_job_id
                WHERE sp.status = 'pending'
                  AND vj.cancel_requested = 0
                  AND (
                      (vj.job_kind = 'youtube_discovery' AND sp.stage_name LIKE 'youtube_discovery_%')
                      OR (vj.job_kind = 'video_processing' AND sp.stage_name NOT LIKE 'youtube_discovery_%')
                  )
                  AND vj.status NOT IN ('completed', 'completed_with_warning', 'failed', 'skipped')
                  AND (sp.retry_after IS NULL OR sp.retry_after <= ?)
                  AND NOT EXISTS (
                      SELECT 1 FROM stage_plans dep
                      WHERE dep.video_job_id = sp.video_job_id
                        AND (
                            (vj.job_kind = 'youtube_discovery' AND dep.stage_name LIKE 'youtube_discovery_%')
                            OR (vj.job_kind = 'video_processing' AND dep.stage_name NOT LIKE 'youtube_discovery_%')
                        )
                        AND dep.position < sp.position
                        AND dep.status NOT IN ('completed', 'completed_with_warning', 'skipped')
                  )
                ORDER BY vj.priority, sp.position, sp.updated_at, sp.id
                """,
                (now,),
            ).fetchall()
        return [_stage_plan_from_row(row) for row in rows]

    def claim_next_stage(self, *, owner: str, lease_seconds: float = 300) -> ClaimedStageRecord | None:
        self.ensure_db()
        now = time.time()
        lease_until = now + lease_seconds
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            candidates = conn.execute(
                """
                SELECT sp.id, sp.video_job_id, sp.stage_name, sp.status, sp.position, sp.attempt, sp.max_attempts,
                       sp.hard_failure, sp.resource_lock, sp.resource_slot, sp.lease_owner, sp.lease_until,
                       sp.retry_after, sp.error, sp.warning, sp.stage_payload_json, sp.created_at, sp.updated_at, sp.completed_at
                FROM stage_plans sp
                JOIN video_jobs vj ON vj.id = sp.video_job_id
                WHERE sp.status = 'pending'
                  AND vj.cancel_requested = 0
                  AND (
                      (vj.job_kind = 'youtube_discovery' AND sp.stage_name LIKE 'youtube_discovery_%')
                      OR (vj.job_kind = 'video_processing' AND sp.stage_name NOT LIKE 'youtube_discovery_%')
                  )
                  AND vj.status NOT IN ('completed', 'completed_with_warning', 'failed', 'skipped')
                  AND (sp.retry_after IS NULL OR sp.retry_after <= ?)
                  AND NOT EXISTS (
                      SELECT 1 FROM stage_plans dep
                      WHERE dep.video_job_id = sp.video_job_id
                        AND (
                            (vj.job_kind = 'youtube_discovery' AND dep.stage_name LIKE 'youtube_discovery_%')
                            OR (vj.job_kind = 'video_processing' AND dep.stage_name NOT LIKE 'youtube_discovery_%')
                        )
                        AND dep.position < sp.position
                        AND dep.status NOT IN ('completed', 'completed_with_warning', 'skipped')
                  )
                ORDER BY vj.priority, sp.position, sp.updated_at, sp.id
                """,
                (now,),
            ).fetchall()
            for row in candidates:
                stage = _stage_plan_from_row(row)
                slot = None
                if stage.resource_lock:
                    slot = self._acquire_resource_slot(conn, stage.resource_lock, owner, lease_until, now)
                    if slot is None:
                        continue
                latest_attempt_row = conn.execute(
                    "SELECT max(attempt) FROM stage_runs WHERE video_job_id = ? AND stage_name = ? AND started_at >= ?",
                    (stage.video_job_id, stage.stage_name, stage.created_at),
                ).fetchone()
                attempt = max(stage.attempt, int(latest_attempt_row[0] or 0)) + 1
                latest_run_attempt_row = conn.execute(
                    "SELECT max(attempt) FROM stage_runs WHERE video_job_id = ? AND stage_name = ?",
                    (stage.video_job_id, stage.stage_name),
                ).fetchone()
                run_attempt = max(attempt, int(latest_run_attempt_row[0] or 0) + 1)
                if attempt > stage.max_attempts:
                    if slot is not None:
                        conn.execute("DELETE FROM locks WHERE lock_name = ? AND owner = ?", (slot, owner))
                    conn.execute(
                        """
                        UPDATE stage_plans
                        SET status = 'failed', error = ?, updated_at = ?, completed_at = ?
                        WHERE id = ?
                        """,
                        ("max attempts exceeded", now, now, stage.id),
                    )
                    continue
                updated = conn.execute(
                    """
                    UPDATE stage_plans
                    SET status = 'running', attempt = ?, lease_owner = ?, lease_until = ?,
                        resource_slot = ?, retry_after = NULL, error = NULL, warning = NULL, updated_at = ?
                    WHERE id = ? AND status = 'pending'
                    """,
                    (attempt, owner, lease_until, slot, now, stage.id),
                )
                if updated.rowcount != 1:
                    if slot is not None:
                        conn.execute("DELETE FROM locks WHERE lock_name = ? AND owner = ?", (slot, owner))
                    continue
                stage_run_id = _stable_id("stage", stage.video_job_id, stage.stage_name, run_attempt, uuid.uuid4().hex, length=24)
                conn.execute(
                    """
                    INSERT INTO stage_runs
                      (id, video_job_id, stage_name, status, attempt, started_at, finished_at,
                       input_hash, output_hash, error_class, error_message, warning)
                    VALUES (?, ?, ?, 'running', ?, ?, NULL, ?, NULL, NULL, NULL, NULL)
                    """,
                    (stage_run_id, stage.video_job_id, stage.stage_name, run_attempt, now, stage.stage_name),
                )
                conn.execute(
                    """
                    UPDATE video_jobs
                    SET status = 'running', current_stage = ?, lease_owner = ?, lease_until = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (stage.stage_name, owner, lease_until, now, stage.video_job_id),
                )
                break
            else:
                return None
        return ClaimedStageRecord(
            job=self.load_video_job(stage.video_job_id),
            stage=self.load_stage_plan(stage.id),
            stage_run=self.load_stage_run(stage_run_id),
        )

    def renew_stage_claim(self, stage_plan_id: str, *, owner: str, lease_seconds: float = 300) -> bool:
        self.ensure_db()
        now = time.time()
        lease_until = now + lease_seconds
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT video_job_id, resource_slot
                FROM stage_plans
                WHERE id = ?
                  AND status = 'running'
                  AND lease_owner = ?
                """,
                (stage_plan_id, owner),
            ).fetchone()
            if row is None:
                return False
            video_job_id, resource_slot = row
            conn.execute(
                """
                UPDATE stage_plans
                SET lease_until = ?, updated_at = ?
                WHERE id = ? AND status = 'running' AND lease_owner = ?
                """,
                (lease_until, now, stage_plan_id, owner),
            )
            conn.execute(
                """
                UPDATE video_jobs
                SET lease_until = ?, updated_at = ?
                WHERE id = ? AND status = 'running' AND lease_owner = ?
                """,
                (lease_until, now, video_job_id, owner),
            )
            if resource_slot:
                conn.execute(
                    """
                    UPDATE locks
                    SET lease_until = ?
                    WHERE lock_name = ? AND owner = ?
                    """,
                    (lease_until, resource_slot, owner),
                )
        return True

    def load_stage_plan(self, stage_plan_id: str) -> StagePlanRecord:
        self.ensure_db()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, video_job_id, stage_name, status, position, attempt, max_attempts, hard_failure,
                       resource_lock, resource_slot, lease_owner, lease_until, retry_after, error, warning, stage_payload_json,
                       created_at, updated_at, completed_at
                FROM stage_plans WHERE id = ?
                """,
                (stage_plan_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown stage plan: {stage_plan_id}")
        return _stage_plan_from_row(row)

    def complete_stage(
        self,
        stage_run_id: str,
        *,
        output_hash: str | None = None,
        warning: str | None = None,
        skip_stage_names: list[str] | None = None,
    ) -> StagePlanRecord:
        self.ensure_db()
        now = time.time()
        stage_run = self.load_stage_run(stage_run_id)
        if stage_run.status != "running":
            raise ValueError(f"stage run is not running: {stage_run_id}")
        status = "completed_with_warning" if warning else "completed"
        skip_stage_names = skip_stage_names or []
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT id, resource_slot, lease_owner FROM stage_plans WHERE video_job_id = ? AND stage_name = ?",
                (stage_run.video_job_id, stage_run.stage_name),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown stage plan for run: {stage_run_id}")
            stage_plan_id, resource_slot, owner = row
            conn.execute(
                """
                UPDATE stage_runs
                SET status = ?, finished_at = ?, output_hash = ?, warning = ?
                WHERE id = ?
                """,
                (status, now, output_hash, warning, stage_run_id),
            )
            conn.execute(
                """
                UPDATE stage_plans
                SET status = ?, lease_owner = NULL, lease_until = NULL, resource_slot = NULL,
                    warning = ?, updated_at = ?, completed_at = ?
                WHERE id = ?
                """,
                (status, warning, now, now, stage_plan_id),
            )
            if resource_slot and owner:
                conn.execute("DELETE FROM locks WHERE lock_name = ? AND owner = ?", (resource_slot, owner))
            for stage_name in skip_stage_names:
                conn.execute(
                    """
                    UPDATE stage_plans
                    SET status = 'skipped', warning = COALESCE(warning, ?), updated_at = ?, completed_at = ?
                    WHERE video_job_id = ? AND stage_name = ? AND status = 'pending'
                    """,
                    ("skipped by dependency branch", now, now, stage_run.video_job_id, stage_name),
                )
            self._finish_job_if_stage_plan_terminal(conn, stage_run.video_job_id, now)
        return self.load_stage_plan(stage_plan_id)

    def fail_stage(self, stage_run_id: str, *, error: Exception | str, backoff_seconds: float | None = None) -> StagePlanRecord:
        self.ensure_db()
        now = time.time()
        stage_run = self.load_stage_run(stage_run_id)
        if stage_run.status != "running":
            raise ValueError(f"stage run is not running: {stage_run_id}")
        error_class = error.__class__.__name__ if isinstance(error, Exception) else "RuntimeError"
        error_message = str(error)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT id, attempt, max_attempts, hard_failure, resource_slot, lease_owner
                FROM stage_plans WHERE video_job_id = ? AND stage_name = ?
                """,
                (stage_run.video_job_id, stage_run.stage_name),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown stage plan for run: {stage_run_id}")
            stage_plan_id, attempt, max_attempts, hard_failure, resource_slot, owner = row
            conn.execute(
                """
                UPDATE stage_runs
                SET status = 'failed', finished_at = ?, error_class = ?, error_message = ?
                WHERE id = ?
                """,
                (now, error_class, error_message, stage_run_id),
            )
            if resource_slot and owner:
                conn.execute("DELETE FROM locks WHERE lock_name = ? AND owner = ?", (resource_slot, owner))
            if not hard_failure:
                next_status = "completed_with_warning"
                retry_after = None
                completed_at = now
                warning = error_message
            elif attempt < max_attempts:
                next_status = "pending"
                retry_after = now + (backoff_seconds if backoff_seconds is not None else 30.0 * (2 ** max(0, attempt - 1)))
                completed_at = None
                warning = None
                conn.execute(
                    """
                    INSERT INTO retries (id, stage_run_id, attempt, scheduled_at, reason, backoff_seconds)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _stable_id("retry", stage_run_id, attempt + 1, error_message, now, length=24),
                        stage_run_id,
                        attempt + 1,
                        retry_after,
                        error_message,
                        max(0.0, retry_after - now),
                    ),
                )
            else:
                next_status = "failed"
                retry_after = None
                completed_at = now
                warning = None
            conn.execute(
                """
                UPDATE stage_plans
                SET status = ?, lease_owner = NULL, lease_until = NULL, resource_slot = NULL,
                    retry_after = ?, error = ?, warning = ?, updated_at = ?, completed_at = ?
                WHERE id = ?
                """,
                (next_status, retry_after, error_message, warning, now, completed_at, stage_plan_id),
            )
            if next_status == "failed":
                conn.execute(
                    """
                    UPDATE video_jobs
                    SET status = 'failed', error = ?, lease_owner = NULL, lease_until = NULL,
                        updated_at = ?, completed_at = ?
                    WHERE id = ?
                    """,
                    (error_message, now, now, stage_run.video_job_id),
                )
            else:
                self._finish_job_if_stage_plan_terminal(conn, stage_run.video_job_id, now)
        return self.load_stage_plan(stage_plan_id)

    def skip_stage(self, stage_plan_id: str, *, warning: str | None = None) -> StagePlanRecord:
        self.ensure_db()
        now = time.time()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT resource_slot, lease_owner FROM stage_plans WHERE id = ?", (stage_plan_id,)).fetchone()
            conn.execute(
                """
                UPDATE stage_plans
                SET status = 'skipped', warning = ?, lease_owner = NULL, lease_until = NULL,
                    resource_slot = NULL, updated_at = ?, completed_at = ?
                WHERE id = ?
                """,
                (warning, now, now, stage_plan_id),
            )
            if row is not None and row[0] and row[1]:
                conn.execute("DELETE FROM locks WHERE lock_name = ? AND owner = ?", (row[0], row[1]))
            row = conn.execute("SELECT video_job_id FROM stage_plans WHERE id = ?", (stage_plan_id,)).fetchone()
            if row is not None:
                self._finish_job_if_stage_plan_terminal(conn, row[0], now)
        return self.load_stage_plan(stage_plan_id)

    def expire_stale_stage_claims(self) -> int:
        self.ensure_db()
        now = time.time()
        expired = 0
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT id, resource_slot, lease_owner
                FROM stage_plans
                WHERE status = 'running'
                  AND (
                    (lease_until IS NOT NULL AND lease_until < ?)
                    OR lease_owner LIKE 'local-stage-worker-%'
                  )
                """,
                (now,),
            ).fetchall()
            for stage_id, resource_slot, owner in rows:
                stage_lease = conn.execute(
                    "SELECT lease_until FROM stage_plans WHERE id = ?",
                    (stage_id,),
                ).fetchone()
                lease_until = float(stage_lease[0]) if stage_lease and stage_lease[0] is not None else None
                if lease_until is not None and lease_until >= now and not _local_stage_owner_is_dead(owner):
                    continue
                stage_row = conn.execute(
                    "SELECT video_job_id, stage_name FROM stage_plans WHERE id = ?",
                    (stage_id,),
                ).fetchone()
                if stage_row is not None:
                    conn.execute(
                        """
                        UPDATE stage_runs
                        SET status = 'failed',
                            finished_at = ?,
                            error_class = 'RuntimeLeaseExpired',
                            error_message = 'stage lease expired'
                        WHERE video_job_id = ?
                          AND stage_name = ?
                          AND status = 'running'
                        """,
                        (now, stage_row[0], stage_row[1]),
                    )
                conn.execute(
                    """
                    UPDATE stage_plans
                    SET status = 'pending', lease_owner = NULL, lease_until = NULL, resource_slot = NULL, updated_at = ?
                    WHERE id = ?
                    """,
                    (now, stage_id),
                )
                if resource_slot and owner:
                    conn.execute("DELETE FROM locks WHERE lock_name = ? AND owner = ?", (resource_slot, owner))
                expired += 1
            orphan_rows = conn.execute(
                """
                SELECT sr.id
                FROM stage_runs sr
                JOIN stage_plans sp
                  ON sp.video_job_id = sr.video_job_id
                 AND sp.stage_name = sr.stage_name
                WHERE sr.status = 'running'
                  AND sp.status != 'running'
                """
            ).fetchall()
            for (stage_run_id,) in orphan_rows:
                conn.execute(
                    """
                    UPDATE stage_runs
                    SET status = 'failed',
                        finished_at = ?,
                        error_class = 'RuntimeLeaseExpired',
                        error_message = 'stage lease expired'
                    WHERE id = ?
                    """,
                    (now, stage_run_id),
                )
                expired += 1
        return expired

    def stage_queue_status(self) -> dict[str, Any]:
        self.ensure_db()
        with self._connect() as conn:
            status_rows = conn.execute("SELECT status, COUNT(*) FROM stage_plans GROUP BY status").fetchall()
            stage_rows = conn.execute(
                "SELECT stage_name, status, COUNT(*) FROM stage_plans GROUP BY stage_name, status ORDER BY stage_name"
            ).fetchall()
        return {
            "statuses": {row[0]: row[1] for row in status_rows},
            "stage_counts": {
                f"{stage_name}:{status}": count for stage_name, status, count in stage_rows
            },
            "stage_group_counts": _stage_group_counts(stage_rows),
        }

    def _acquire_resource_slot(self, conn, resource_lock: str, owner: str, lease_until: float, now: float) -> str | None:
        limit = _resource_concurrency_limit(resource_lock, self.settings)
        for index in range(limit):
            lock_name = f"{resource_lock}#{index}"
            row = conn.execute("SELECT owner, lease_until FROM locks WHERE lock_name = ?", (lock_name,)).fetchone()
            if row is not None and row[0] != owner and float(row[1]) >= now:
                continue
            conn.execute(
                """
                INSERT INTO locks (lock_name, owner, lease_until)
                VALUES (?, ?, ?)
                ON CONFLICT(lock_name) DO UPDATE SET owner=excluded.owner, lease_until=excluded.lease_until
                """,
                (lock_name, owner, lease_until),
            )
            return lock_name
        return None

    def _finish_job_if_stage_plan_terminal(self, conn, video_job_id: str, now: float) -> None:
        rows = conn.execute(
            "SELECT status, warning FROM stage_plans WHERE video_job_id = ? ORDER BY position",
            (video_job_id,),
        ).fetchall()
        if not rows:
            return
        statuses = [row[0] for row in rows]
        if any(status in {"pending", "running"} for status in statuses):
            conn.execute(
                "UPDATE video_jobs SET status = 'queued', lease_owner = NULL, lease_until = NULL, updated_at = ? WHERE id = ?",
                (now, video_job_id),
            )
            return
        if any(status == "failed" for status in statuses):
            return
        warnings = [row[1] for row in rows if row[1]]
        warning = "; ".join(warnings) if warnings else None
        status = "completed_with_warning" if warning else "completed"
        conn.execute(
            """
            UPDATE video_jobs
            SET status = ?, warning = ?, lease_owner = NULL, lease_until = NULL, updated_at = ?, completed_at = ?
            WHERE id = ?
            """,
            (status, warning, now, now, video_job_id),
        )

    def start_stage_run(self, *, video_job_id: str, stage_name: str, input_hash: str | None = None) -> StageRunRecord:
        self.ensure_db()
        now = time.time()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT max(attempt) FROM stage_runs WHERE video_job_id = ? AND stage_name = ?",
                (video_job_id, stage_name),
            ).fetchone()
            attempt = int(row[0] or 0) + 1
            stage_id = _stable_id("stage", video_job_id, stage_name, attempt, uuid.uuid4().hex, length=24)
            conn.execute(
                """
                INSERT INTO stage_runs
                  (id, video_job_id, stage_name, status, attempt, started_at, finished_at,
                   input_hash, output_hash, error_class, error_message, warning)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (stage_id, video_job_id, stage_name, "running", attempt, now, None, input_hash, None, None, None, None),
            )
            conn.execute(
                "UPDATE video_jobs SET current_stage = ?, updated_at = ? WHERE id = ?",
                (stage_name, now, video_job_id),
            )
        return self.load_stage_run(stage_id)

    def load_stage_run(self, stage_run_id: str) -> StageRunRecord:
        self.ensure_db()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, video_job_id, stage_name, status, attempt, started_at, finished_at,
                       input_hash, output_hash, error_class, error_message, warning
                FROM stage_runs WHERE id = ?
                """,
                (stage_run_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown stage run: {stage_run_id}")
        return _stage_run_from_row(row)

    def complete_stage_run(self, stage_run_id: str, *, output_hash: str | None = None, warning: str | None = None) -> StageRunRecord:
        self.ensure_db()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE stage_runs
                SET status = ?, finished_at = ?, output_hash = ?, warning = ?
                WHERE id = ?
                """,
                ("completed_with_warning" if warning else "completed", time.time(), output_hash, warning, stage_run_id),
            )
        return self.load_stage_run(stage_run_id)

    def fail_stage_run(self, stage_run_id: str, *, error: Exception | str) -> StageRunRecord:
        self.ensure_db()
        error_class = error.__class__.__name__ if isinstance(error, Exception) else "RuntimeError"
        error_message = str(error)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE stage_runs
                SET status = 'failed', finished_at = ?, error_class = ?, error_message = ?
                WHERE id = ?
                """,
                (time.time(), error_class, error_message, stage_run_id),
            )
        return self.load_stage_run(stage_run_id)

    def record_stage_artifact(
        self,
        *,
        stage_run_id: str,
        video_id: str,
        artifact_type: str,
        path: Path | str,
        provenance: dict[str, Any],
        language: str | None = None,
        status: str = "committed",
    ) -> StageArtifactRecord:
        self.ensure_db()
        artifact_path = Path(path)
        content_hash = _file_hash(artifact_path)
        artifact_id = _stable_id("artifact", stage_run_id, artifact_type, str(artifact_path), content_hash, length=24)
        created_at = time.time()
        provenance_json = json.dumps(provenance, ensure_ascii=False, sort_keys=True)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO stage_artifacts
                  (id, stage_run_id, video_id, artifact_type, path, language, content_hash, status, provenance_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET status=excluded.status, content_hash=excluded.content_hash
                """,
                (
                    artifact_id,
                    stage_run_id,
                    video_id,
                    artifact_type,
                    str(artifact_path),
                    language,
                    content_hash,
                    status,
                    provenance_json,
                    created_at,
                ),
            )
        return StageArtifactRecord(
            id=artifact_id,
            stage_run_id=stage_run_id,
            video_id=video_id,
            artifact_type=artifact_type,
            path=str(artifact_path),
            language=language,
            content_hash=content_hash,
            status=status,
            provenance_json=provenance_json,
            created_at=created_at,
        )

    def commit_json_artifact(
        self,
        *,
        stage_run_id: str,
        video_id: str,
        artifact_type: str,
        path: Path | str,
        data: Any,
        provenance: dict[str, Any],
        language: str | None = None,
    ) -> StageArtifactRecord:
        artifact_path = Path(path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = artifact_path.with_name(f"{artifact_path.name}.{uuid.uuid4().hex}.tmp")
        tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp_path, artifact_path)
        return self.record_stage_artifact(
            stage_run_id=stage_run_id,
            video_id=video_id,
            artifact_type=artifact_type,
            path=artifact_path,
            language=language,
            provenance=provenance,
        )

    def list_stage_artifacts(self, *, video_id: str | None = None) -> list[StageArtifactRecord]:
        self.ensure_db()
        with self._connect() as conn:
            if video_id is None:
                rows = conn.execute(
                    """
                    SELECT id, stage_run_id, video_id, artifact_type, path, language, content_hash, status, provenance_json, created_at
                    FROM stage_artifacts ORDER BY created_at, id
                    """
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, stage_run_id, video_id, artifact_type, path, language, content_hash, status, provenance_json, created_at
                    FROM stage_artifacts WHERE video_id = ? ORDER BY created_at, id
                    """,
                    (video_id,),
                ).fetchall()
        return [_stage_artifact_from_row(row) for row in rows]

    def schedule_retry(
        self,
        *,
        stage_run_id: str,
        attempt: int,
        reason: str,
        backoff_seconds: float,
    ) -> RetryRecord:
        self.ensure_db()
        retry_id = _stable_id("retry", stage_run_id, attempt, reason, time.time(), length=24)
        scheduled_at = time.time() + backoff_seconds
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO retries (id, stage_run_id, attempt, scheduled_at, reason, backoff_seconds)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (retry_id, stage_run_id, attempt, scheduled_at, reason, backoff_seconds),
            )
        return RetryRecord(retry_id, stage_run_id, attempt, scheduled_at, reason, backoff_seconds)

    def list_retries(self, *, stage_run_id: str | None = None) -> list[RetryRecord]:
        self.ensure_db()
        with self._connect() as conn:
            if stage_run_id is None:
                rows = conn.execute(
                    "SELECT id, stage_run_id, attempt, scheduled_at, reason, backoff_seconds FROM retries ORDER BY scheduled_at"
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, stage_run_id, attempt, scheduled_at, reason, backoff_seconds
                    FROM retries WHERE stage_run_id = ? ORDER BY scheduled_at
                    """,
                    (stage_run_id,),
                ).fetchall()
        return [RetryRecord(row[0], row[1], row[2], row[3], row[4], row[5]) for row in rows]

    def acquire_lock(self, lock_name: str, *, owner: str, lease_seconds: float) -> bool:
        self.ensure_db()
        now = time.time()
        lease_until = now + lease_seconds
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT owner, lease_until FROM locks WHERE lock_name = ?",
                (lock_name,),
            ).fetchone()
            if row is not None and row[0] != owner and float(row[1]) >= now:
                return False
            conn.execute(
                """
                INSERT INTO locks (lock_name, owner, lease_until)
                VALUES (?, ?, ?)
                ON CONFLICT(lock_name) DO UPDATE SET owner=excluded.owner, lease_until=excluded.lease_until
                """,
                (lock_name, owner, lease_until),
            )
        return True

    def release_lock(self, lock_name: str, *, owner: str) -> None:
        self.ensure_db()
        with self._connect() as conn:
            conn.execute("DELETE FROM locks WHERE lock_name = ? AND owner = ?", (lock_name, owner))


def _stable_id(*parts: Any, length: int = 20) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:length]


def _local_stage_owner_is_dead(owner: str | None) -> bool:
    if not owner:
        return False
    match = re.search(r"local-stage-worker-(\d+)(?:-|$)", owner)
    if match is None:
        return False
    pid = int(match.group(1))
    if pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _channel_job_from_row(row) -> ChannelJobRecord:
    has_acceptance_mode = len(row) > 17
    return ChannelJobRecord(
        id=row[0],
        channel_id=row[1],
        channel_url=row[2],
        status=row[3],
        requested_limit=row[4],
        quality_mode=row[5] if len(row) > 10 else "strict",
        acceptance_mode=row[6] if has_acceptance_mode else "data_api_only",
        limit=row[7] if has_acceptance_mode else (row[6] if len(row) > 10 else row[4]),
        video_count=int(row[8] or 0) if has_acceptance_mode else (int(row[7] or 0) if len(row) > 10 else 0),
        queued_count=int(row[9] or 0) if has_acceptance_mode else (int(row[8] or 0) if len(row) > 10 else 0),
        completed_count=int(row[10] or 0) if has_acceptance_mode else (int(row[9] or 0) if len(row) > 10 else 0),
        failed_count=int(row[11] or 0) if has_acceptance_mode else (int(row[10] or 0) if len(row) > 10 else 0),
        skipped_count=int(row[12] or 0) if has_acceptance_mode else (int(row[11] or 0) if len(row) > 10 else 0),
        error=row[13] if has_acceptance_mode else (row[12] if len(row) > 10 else row[5]),
        warning=row[14] if has_acceptance_mode else (row[13] if len(row) > 10 else row[6]),
        created_at=row[15] if has_acceptance_mode else (row[14] if len(row) > 10 else row[7]),
        updated_at=row[16] if has_acceptance_mode else (row[15] if len(row) > 10 else row[8]),
        completed_at=row[17] if has_acceptance_mode else (row[16] if len(row) > 10 else row[9]),
    )


def _video_job_from_row(row) -> VideoJobRecord:
    has_job_kind = len(row) >= 18
    return VideoJobRecord(
        id=row[0],
        channel_job_id=row[1],
        video_id=row[2],
        canonical_url=row[3],
        status=row[4],
        current_stage=row[5],
        priority=row[6],
        lease_owner=row[7],
        lease_until=row[8],
        cancel_requested=bool(row[9]),
        error=row[10],
        warning=row[11],
        quality_mode=row[12] or "strict",
        acceptance_mode=row[13] if len(row) > 16 else "data_api_only",
        job_kind=(row[14] or VIDEO_PROCESSING_JOB_KIND) if has_job_kind else VIDEO_PROCESSING_JOB_KIND,
        created_at=row[15] if has_job_kind else (row[14] if len(row) > 16 else row[13]),
        updated_at=row[16] if has_job_kind else (row[15] if len(row) > 16 else row[14]),
        completed_at=row[17] if has_job_kind else (row[16] if len(row) > 16 else row[15]),
    )


def _normalize_job_kind(job_kind: str) -> str:
    normalized = str(job_kind or VIDEO_PROCESSING_JOB_KIND).strip()
    if not normalized:
        return VIDEO_PROCESSING_JOB_KIND
    return normalized


def _stage_run_from_row(row) -> StageRunRecord:
    return StageRunRecord(
        id=row[0],
        video_job_id=row[1],
        stage_name=row[2],
        status=row[3],
        attempt=row[4],
        started_at=row[5],
        finished_at=row[6],
        input_hash=row[7],
        output_hash=row[8],
        error_class=row[9],
        error_message=row[10],
        warning=row[11],
    )


def _stage_plan_from_row(row) -> StagePlanRecord:
    return StagePlanRecord(
        id=row[0],
        video_job_id=row[1],
        stage_name=row[2],
        status=row[3],
        position=row[4],
        attempt=row[5],
        max_attempts=row[6],
        hard_failure=bool(row[7]),
        resource_lock=row[8],
        resource_slot=row[9],
        lease_owner=row[10],
        lease_until=row[11],
        retry_after=row[12],
        error=row[13],
        warning=row[14],
        stage_payload_json=row[15] if len(row) > 18 else None,
        created_at=row[16] if len(row) > 18 else row[15],
        updated_at=row[17] if len(row) > 18 else row[16],
        completed_at=row[18] if len(row) > 18 else row[17],
    )


def _stage_group_counts(stage_rows: list[tuple[str, str, int]]) -> dict[str, int]:
    grouped: dict[str, int] = {}
    for stage_name, status, count in stage_rows:
        group_name = _stage_group_name(str(stage_name))
        if group_name == stage_name:
            continue
        key = f"{group_name}:{status}"
        grouped[key] = grouped.get(key, 0) + int(count)
    return grouped


def _stage_group_name(stage_name: str) -> str:
    if stage_name.startswith("transcript_translation_chunk_"):
        return "transcript_translation_chunks"
    if stage_name.startswith("analysis_chunk_"):
        return "analysis_chunks"
    if stage_name.startswith("youtube_discovery_query_") and stage_name != "youtube_discovery_query_expand":
        return "youtube_discovery_queries"
    return stage_name


def _clamp_lm_studio_translation_concurrency(value: Any) -> int:
    try:
        return min(2, max(1, int(value)))
    except (TypeError, ValueError):
        return 1


def _clamp_positive_concurrency(value: Any, *, default: int, maximum: int = 16) -> int:
    try:
        return min(maximum, max(1, int(value)))
    except (TypeError, ValueError):
        return default


def _resource_concurrency_limit(resource_lock: str, settings: Settings | None = None) -> int:
    if resource_lock == "lm_studio_translation":
        if settings is not None:
            return _clamp_lm_studio_translation_concurrency(
                getattr(settings, "lm_studio_translation_concurrency", 1)
            )
        raw = os.getenv("YT_SUB_HUB_LM_STUDIO_TRANSLATION_CONCURRENCY")
        if raw:
            return _clamp_lm_studio_translation_concurrency(raw)
    setting_names = {
        "youtube_data_api": "youtube_data_api_concurrency",
        "local_candidate_scoring": "local_candidate_scoring_concurrency",
        "llm_discovery_review": "llm_discovery_review_concurrency",
        "neo4j_sync": "neo4j_sync_concurrency",
        "queue_mutation": "queue_mutation_concurrency",
    }
    env_names = {
        "youtube_data_api": "YT_SUB_HUB_YOUTUBE_DATA_API_CONCURRENCY",
        "local_candidate_scoring": "YT_SUB_HUB_LOCAL_CANDIDATE_SCORING_CONCURRENCY",
        "llm_discovery_review": "YT_SUB_HUB_LLM_DISCOVERY_REVIEW_CONCURRENCY",
        "neo4j_sync": "YT_SUB_HUB_NEO4J_SYNC_CONCURRENCY",
        "queue_mutation": "YT_SUB_HUB_QUEUE_MUTATION_CONCURRENCY",
    }
    if resource_lock in setting_names:
        default = int(DEFAULT_CONCURRENCY_LIMITS.get(resource_lock, 1))
        if settings is not None:
            return _clamp_positive_concurrency(getattr(settings, setting_names[resource_lock], default), default=default)
        raw = os.getenv(env_names[resource_lock])
        if raw:
            return _clamp_positive_concurrency(raw, default=default)
    return max(1, int(DEFAULT_CONCURRENCY_LIMITS.get(resource_lock, 1)))


def _stage_artifact_from_row(row) -> StageArtifactRecord:
    return StageArtifactRecord(
        id=row[0],
        stage_run_id=row[1],
        video_id=row[2],
        artifact_type=row[3],
        path=row[4],
        language=row[5],
        content_hash=row[6],
        status=row[7],
        provenance_json=row[8],
        created_at=row[9],
    )
