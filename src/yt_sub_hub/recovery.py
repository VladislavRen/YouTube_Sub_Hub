from __future__ import annotations

import re
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from yt_sub_hub.library import LibraryStore
from yt_sub_hub.runtime import DISCOVERY_STATIC_STAGES, RuntimeStore, StagePlanRecord, VideoJobRecord
from yt_sub_hub.stages import PIPELINE_STAGES


_CHUNK_GROUPS = {
    "transcript_translation": re.compile(r"^transcript_translation_chunk_\d+$"),
    "analysis": re.compile(r"^analysis_chunk_\d+$"),
    "youtube_discovery_queries": re.compile(r"^youtube_discovery_query_\d+$"),
}


def build_job_recovery_dashboard(runtime: RuntimeStore, store: LibraryStore, job_id: str) -> dict[str, Any]:
    del store
    job = runtime.load_video_job(job_id)
    stages = runtime.list_stage_plan(job_id)
    now = time.time()
    failed_chunks = _failed_chunks(stages)
    failed_stages = _failed_stages(stages)
    running_stages = _running_stages(stages, now)
    pending_retries = _pending_retries(runtime, job_id, stages, now)
    locks = _locks(runtime, stages, now)
    stuck_reasons = _stuck_reasons(job, stages, running_stages, locks, now)
    mismatches = _stage_plan_mismatches(job, stages)
    actions = _actions(job, stages, failed_stages, failed_chunks, stuck_reasons)
    recommended_actions = _recommended_actions(actions)
    return {
        "job_id": job.id,
        "job_kind": job.job_kind,
        "video_id": job.video_id,
        "status": job.status,
        "current_stage": job.current_stage,
        "acceptance_mode": job.acceptance_mode,
        "skip_reason": "skipped_requires_media" if _is_skipped_requires_media(job) else None,
        "stuck": bool(stuck_reasons),
        "stuck_reasons": stuck_reasons,
        "failed_stages": failed_stages,
        "failed_chunks": failed_chunks,
        "running_stages": running_stages,
        "pending_retries": pending_retries,
        "locks": locks,
        "mismatches": mismatches,
        "recommended_actions": recommended_actions,
        "actions": actions,
    }


def build_job_recovery_summary(
    runtime: RuntimeStore,
    store: LibraryStore,
    job_id: str,
    *,
    job: VideoJobRecord | None = None,
    stages: list[StagePlanRecord] | None = None,
) -> dict[str, Any]:
    if job is None or stages is None:
        dashboard = build_job_recovery_dashboard(runtime, store, job_id)
        failed_chunk_count = sum(len(items) for items in dashboard["failed_chunks"].values())
        return {
            "status": dashboard["status"],
            "job_kind": dashboard["job_kind"],
            "acceptance_mode": dashboard["acceptance_mode"],
            "skip_reason": dashboard["skip_reason"],
            "failed_stage_count": len(dashboard["failed_stages"]),
            "failed_chunk_count": failed_chunk_count,
            "stuck": dashboard["stuck"],
            "retryable": _retryable(dashboard["actions"]),
            "recommended_actions": dashboard["recommended_actions"],
        }
    now = time.time()
    failed_chunks = _failed_chunks(stages)
    failed_stages = _failed_stages(stages)
    running_stages = _running_stages(stages, now)
    locks = _locks(runtime, stages, now)
    stuck_reasons = _stuck_reasons(job, stages, running_stages, locks, now)
    actions = _actions(job, stages, failed_stages, failed_chunks, stuck_reasons)
    return {
        "status": job.status,
        "job_kind": job.job_kind,
        "acceptance_mode": job.acceptance_mode,
        "skip_reason": "skipped_requires_media" if _is_skipped_requires_media(job) else None,
        "failed_stage_count": len(failed_stages),
        "failed_chunk_count": sum(len(items) for items in failed_chunks.values()),
        "stuck": bool(stuck_reasons),
        "retryable": _retryable(actions),
        "recommended_actions": _recommended_actions(actions),
    }


def _failed_stages(stages: list[StagePlanRecord]) -> list[dict[str, Any]]:
    failures = []
    for stage in stages:
        if _chunk_group(stage.stage_name) is not None:
            continue
        if stage.status == "failed" and stage.hard_failure:
            failures.append(_stage_issue(stage))
        elif stage.status == "completed_with_warning" and not stage.hard_failure:
            failures.append(_stage_issue(stage))
    return failures


def _failed_chunks(stages: list[StagePlanRecord]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {
        "transcript_translation": [],
        "analysis": [],
        "youtube_discovery_queries": [],
    }
    for stage in stages:
        group = _chunk_group(stage.stage_name)
        if group is not None and stage.status == "failed":
            result[group].append(_stage_issue(stage))
    return result


def _stage_issue(stage: StagePlanRecord) -> dict[str, Any]:
    return {
        "stage_name": stage.stage_name,
        "status": stage.status,
        "hard_failure": stage.hard_failure,
        "error": stage.error,
        "warning": stage.warning,
        "retry_after": _format_ts(stage.retry_after),
    }


def _running_stages(stages: list[StagePlanRecord], now: float) -> list[dict[str, Any]]:
    running = []
    for stage in stages:
        if stage.status != "running":
            continue
        expired = stage.lease_until is None or float(stage.lease_until) < now
        running.append(
            {
                "stage_name": stage.stage_name,
                "lease_owner": stage.lease_owner,
                "lease_until": _format_ts(stage.lease_until),
                "resource_slot": stage.resource_slot,
                "expired": expired,
            }
        )
    return running


def _pending_retries(
    runtime: RuntimeStore,
    job_id: str,
    stages: list[StagePlanRecord],
    now: float,
) -> list[dict[str, Any]]:
    stage_by_name = {stage.stage_name: stage for stage in stages}
    retry_rows: list[dict[str, Any]] = []
    with _readonly_connection(runtime.db_path) as conn:
        rows = conn.execute(
            """
            SELECT sr.stage_name, r.attempt, r.scheduled_at, r.reason, r.backoff_seconds
            FROM retries r
            JOIN stage_runs sr ON sr.id = r.stage_run_id
            WHERE sr.video_job_id = ?
              AND r.scheduled_at > ?
            ORDER BY r.scheduled_at, sr.stage_name
            """,
            (job_id, now),
        ).fetchall()
    seen = set()
    for stage_name, attempt, scheduled_at, reason, backoff_seconds in rows:
        seen.add(str(stage_name))
        retry_rows.append(
            {
                "stage_name": str(stage_name),
                "attempt": int(attempt),
                "retry_after": _format_ts(scheduled_at),
                "reason": reason,
                "backoff_seconds": float(backoff_seconds),
            }
        )
    for stage_name, stage in stage_by_name.items():
        if stage.retry_after is None or float(stage.retry_after) <= now or stage_name in seen:
            continue
        retry_rows.append(
            {
                "stage_name": stage_name,
                "attempt": stage.attempt + 1,
                "retry_after": _format_ts(stage.retry_after),
                "reason": stage.error or "retry scheduled",
                "backoff_seconds": round(float(stage.retry_after) - now, 3),
            }
        )
    retry_rows.sort(key=lambda item: (item["retry_after"] or "", item["stage_name"]))
    return retry_rows


def _locks(runtime: RuntimeStore, stages: list[StagePlanRecord], now: float) -> list[dict[str, Any]]:
    resource_slots = {stage.resource_slot for stage in stages if stage.resource_slot}
    owners = {stage.lease_owner for stage in stages if stage.lease_owner}
    if not resource_slots and not owners:
        return []
    with _readonly_connection(runtime.db_path) as conn:
        rows = conn.execute("SELECT lock_name, owner, lease_until FROM locks ORDER BY lock_name").fetchall()
    result = []
    for lock_name, owner, lease_until in rows:
        lock_name = str(lock_name)
        if lock_name not in resource_slots and owner not in owners:
            continue
        result.append(
            {
                "lock_name": lock_name,
                "owner": owner,
                "lease_until": _format_ts(lease_until),
                "stale": lease_until is None or float(lease_until) < now,
            }
        )
    return result


def _stuck_reasons(
    job: VideoJobRecord,
    stages: list[StagePlanRecord],
    running_stages: list[dict[str, Any]],
    locks: list[dict[str, Any]],
    now: float,
) -> list[str]:
    reasons = []
    if job.status == "running" and (job.lease_until is None or float(job.lease_until) < now):
        reasons.append("expired_job_lease")
    if any(stage["expired"] for stage in running_stages):
        reasons.append("expired_stage_lease")
    if any(lock["stale"] for lock in locks):
        reasons.append("stale_lock")
    if _blocked_pending_stages(stages):
        reasons.append("pending_blocked_by_failed_upstream")
    return _dedupe(reasons)


def _blocked_pending_stages(stages: list[StagePlanRecord]) -> list[str]:
    failed_positions = [stage.position for stage in stages if stage.status == "failed"]
    if not failed_positions:
        return []
    first_failed = min(failed_positions)
    return [stage.stage_name for stage in stages if stage.status == "pending" and stage.position > first_failed]


def _stage_plan_mismatches(job: VideoJobRecord, stages: list[StagePlanRecord]) -> list[str]:
    if not stages:
        return ["missing_stage_plan"]
    if job.job_kind == "youtube_discovery":
        expected = set(DISCOVERY_STATIC_STAGES)
    else:
        expected = {stage.name for stage in PIPELINE_STAGES}
    names = [stage.stage_name for stage in stages]
    mismatches = []
    missing = sorted(expected - {name for name in names if _chunk_group(name) is None})
    if missing:
        mismatches.append("missing_static_stage_plan")
    unexpected = sorted(name for name in names if _chunk_group(name) is None and name not in expected)
    if unexpected:
        mismatches.append("unexpected_static_stage_plan")
    if len(names) != len(set(names)):
        mismatches.append("duplicate_stage_plan")
    return mismatches


def _recommended_actions(actions: dict[str, dict[str, Any]]) -> list[str]:
    order = ["retry_with_full_processing", "retry_stage", "retry_failed_chunks", "reset_downstream", "cancel_job", "resume_job"]
    return [name for name in order if actions.get(name, {}).get("applicable") is True]


def _actions(
    job: VideoJobRecord,
    stages: list[StagePlanRecord],
    failed_stages: list[dict[str, Any]],
    failed_chunks: dict[str, list[dict[str, Any]]],
    stuck_reasons: list[str],
) -> dict[str, dict[str, Any]]:
    stage_names = [stage.stage_name for stage in stages]
    failed_chunk_groups = [group for group, items in failed_chunks.items() if items]
    skipped_requires_media = _is_skipped_requires_media(job)
    return {
        "retry_with_full_processing": {
            "applicable": skipped_requires_media,
            "acceptance_mode": "full_processing",
            "reason": "requires media/Whisper fallback, but data_api_only mode forbids media download",
        },
        "retry_stage": {
            "applicable": bool(failed_stages or any(failed_chunks.values())) and not skipped_requires_media,
            "stage_names": [item["stage_name"] for item in failed_stages]
            + [item["stage_name"] for items in failed_chunks.values() for item in items],
        },
        "retry_failed_chunks": {
            "applicable": bool(failed_chunk_groups),
            "groups": failed_chunk_groups,
        },
        "reset_downstream": {
            "applicable": bool(stage_names and (failed_stages or any(failed_chunks.values()) or stuck_reasons)),
            "stage_names": stage_names,
        },
        "cancel_job": {
            "applicable": job.status not in {"completed", "completed_with_warning", "failed", "skipped"} and not job.cancel_requested,
        },
        "resume_job": {
            "applicable": (job.status in {"failed", "skipped"} or bool(stuck_reasons)) and not skipped_requires_media,
        },
    }


def _retryable(actions: dict[str, dict[str, Any]]) -> bool:
    return any(
        actions[name]["applicable"]
        for name in ("retry_with_full_processing", "retry_stage", "retry_failed_chunks", "reset_downstream", "resume_job")
    )


def _is_skipped_requires_media(job: VideoJobRecord) -> bool:
    return (
        job.status == "skipped"
        and job.current_stage == "transcript_source"
        and "skipped_requires_media" in str(job.warning or job.error or "")
    )


def _chunk_group(stage_name: str) -> str | None:
    for group, pattern in _CHUNK_GROUPS.items():
        if pattern.match(stage_name):
            return group
    return None


def _dedupe(values: list[str]) -> list[str]:
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _readonly_connection(db_path: Path | str):
    path = Path(db_path)
    uri = f"file:{path}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _format_ts(value: float | int | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(float(value), UTC).isoformat().replace("+00:00", "Z")
