from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from yt_sub_hub.library import LibraryStore
from yt_sub_hub.models import JobRecord
from yt_sub_hub.runtime import RuntimeStore, VideoJobRecord


TERMINAL_RUNTIME_STATUSES = {"failed", "skipped", "completed", "completed_with_warning"}


def ensure_legacy_job_for_runtime(store: LibraryStore, runtime_record: VideoJobRecord) -> None:
    if runtime_record.id in store.load_jobs():
        return
    store.save_job(
        JobRecord(
            job_id=runtime_record.id,
            url=runtime_record.canonical_url,
            video_id=runtime_record.video_id,
            status="queued",
            stage="queued",
        )
    )


def recover_interrupted_legacy_jobs(
    store: LibraryStore,
    runtime: RuntimeStore,
    *,
    stale_after_seconds: int = 1800,
) -> list[JobRecord]:
    now = time.time()
    runtime_by_id = {job.id: job for job in runtime.list_video_jobs()}
    recovered: list[JobRecord] = []
    for legacy_record, updated_at in _load_legacy_running_jobs_with_updated_at(store):
        runtime_record = runtime_by_id.get(legacy_record.job_id)
        if not _should_recover_legacy_running_job(
            legacy_record,
            updated_at=updated_at,
            runtime_record=runtime_record,
            now=now,
            stale_after_seconds=stale_after_seconds,
        ):
            continue
        previous_stage = legacy_record.stage or "running"
        error = f"Interrupted worker run recovered; previous stage: {previous_stage}"
        recovered.append(
            store.update_job_stage(
                legacy_record.job_id,
                status="failed",
                stage="failed",
                error=error,
            )
        )
    return recovered


def sync_legacy_job_from_runtime(store: LibraryStore, runtime_record: VideoJobRecord) -> JobRecord | None:
    jobs = store.load_jobs()
    legacy = jobs.get(runtime_record.id)
    if legacy is None:
        return None
    if runtime_record.status == "failed":
        return store.update_job_stage(
            runtime_record.id,
            status="failed",
            stage="failed",
            error=runtime_record.error,
        )
    if runtime_record.status == "skipped":
        return store.update_job_stage(
            runtime_record.id,
            status="skipped",
            stage="cancelled",
            warning=runtime_record.warning or ("cancel requested" if runtime_record.cancel_requested else None),
        )
    if runtime_record.status in {"completed", "completed_with_warning"}:
        return store.update_job_stage(
            runtime_record.id,
            status=runtime_record.status,
            stage=legacy.stage,
            note_path=legacy.note_path or _existing_note_path(store, runtime_record.video_id),
            warning=runtime_record.warning,
        )
    if runtime_record.status == "queued":
        return _sync_legacy_queued(store, runtime_record)
    return None


def requeue_failed_job(runtime: RuntimeStore, store: LibraryStore, job_id: str, *, from_stage: str | None = None) -> VideoJobRecord:
    current = runtime.load_video_job(job_id)
    if from_stage is None and current.status != "failed" and not _has_failed_stage_plan(runtime, job_id):
        raise ValueError(f"job is not failed: {job_id}")
    try:
        runtime_record = runtime.reset_failed_stage_plan_for_retry(job_id, from_stage=from_stage)
    except ValueError:
        if from_stage is not None:
            raise
        runtime_record = runtime.resume_video_job(job_id)
    _sync_legacy_queued(store, runtime_record)
    return runtime_record


def _has_failed_stage_plan(runtime: RuntimeStore, job_id: str) -> bool:
    return any(stage.status == "failed" for stage in runtime.list_stage_plan(job_id))


def sync_legacy_requeued_job(store: LibraryStore, runtime_record: VideoJobRecord) -> JobRecord | None:
    return _sync_legacy_queued(store, runtime_record)


def _sync_legacy_queued(store: LibraryStore, runtime_record: VideoJobRecord) -> JobRecord | None:
    if runtime_record.id not in store.load_jobs():
        return None
    return store.update_job_stage(
        runtime_record.id,
        status="queued",
        stage="queued",
        error=None,
        warning=None,
    )


def _should_recover_legacy_running_job(
    legacy_record: JobRecord,
    *,
    updated_at: float,
    runtime_record: VideoJobRecord | None,
    now: float,
    stale_after_seconds: int,
) -> bool:
    if legacy_record.status != "running":
        return False
    if runtime_record is None:
        return updated_at < now - stale_after_seconds
    if runtime_record.status in TERMINAL_RUNTIME_STATUSES:
        return True
    if runtime_record.status == "running":
        if runtime_record.lease_until is None:
            return True
        return runtime_record.lease_until < now
    return False


def _load_legacy_running_jobs_with_updated_at(store: LibraryStore) -> list[tuple[JobRecord, float]]:
    store.ensure_layout()
    with sqlite3.connect(store.db_path) as conn:
        rows = conn.execute(
            """
            SELECT job_id, url, video_id, status, stage, note_path, error, warning, updated_at
            FROM jobs
            WHERE status = ?
            ORDER BY updated_at, job_id
            """,
            ("running",),
        ).fetchall()
    return [
        (
            JobRecord(
                job_id=row[0],
                url=row[1],
                video_id=row[2],
                status=row[3],
                stage=row[4],
                note_path=row[5],
                error=row[6],
                warning=row[7],
            ),
            float(row[8]),
        )
        for row in rows
    ]


def _existing_note_path(store: LibraryStore, video_id: str) -> str | None:
    path = Path(store.note_path(video_id))
    if path.exists():
        return str(path)
    return None
