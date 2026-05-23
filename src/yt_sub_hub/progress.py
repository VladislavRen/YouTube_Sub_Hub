from __future__ import annotations

import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from yt_sub_hub.artifact_review import load_ru_artifact_review
from yt_sub_hub.library import LibraryStore
from yt_sub_hub.product_quality import build_product_quality_payload
from yt_sub_hub.quality import run_quality_gate
from yt_sub_hub.runtime import RuntimeStore, StagePlanRecord, _resource_concurrency_limit


class JobNotFoundError(KeyError):
    pass


class ChannelJobNotFoundError(KeyError):
    pass


_CHUNK_GROUPS = {
    "transcript_translation": re.compile(r"^transcript_translation_chunk_\d+$"),
    "analysis": re.compile(r"^analysis_chunk_\d+$"),
    "youtube_discovery_queries": re.compile(r"^youtube_discovery_query_\d+$"),
}


def build_job_stage_progress(
    runtime: RuntimeStore,
    store: LibraryStore,
    job_id: str,
    *,
    include_chunks: bool = False,
) -> dict[str, Any]:
    job = _load_job(runtime, job_id)
    stages = runtime.list_stage_plan(job_id)
    retry_counts = _retry_counts_by_stage(runtime, job_id)
    latest_runs = _latest_runs_by_stage(runtime, job_id)
    historical_durations = _historical_durations_by_stage(runtime)
    non_chunk_stages = [stage for stage in stages if _chunk_group(stage.stage_name) is None]
    visible_stages = stages if include_chunks else non_chunk_stages

    product_quality = _product_quality_payload(store, job, stages)
    from yt_sub_hub.recovery import build_job_recovery_summary

    return {
        "job_id": job.id,
        "job_kind": job.job_kind,
        "video_id": job.video_id,
        "status": job.status,
        "current_stage": job.current_stage,
        "quality_mode": job.quality_mode,
        "acceptance_mode": job.acceptance_mode,
        "product_quality": product_quality,
        "recovery": build_job_recovery_summary(runtime, store, job_id, job=job, stages=stages),
        "overall": _overall_progress(non_chunk_stages),
        "chunks": _group_chunk_stages(stages, retry_counts),
        "eta": _estimate_eta(runtime, stages, historical_durations),
        "warnings": _collect_warning_reasons(stages, job.warning),
        "hard_failures": _collect_hard_failures(stages, job.error),
        "stages": [_stage_payload(stage, latest_runs.get(stage.stage_name)) for stage in visible_stages],
    }


def build_job_artifacts_status(runtime: RuntimeStore, store: LibraryStore, job_id: str) -> dict[str, Any]:
    job = _load_job(runtime, job_id)
    if job.job_kind == "youtube_discovery":
        return _build_discovery_artifacts_status(runtime, store, job)
    paths = store.artifact_paths(job.video_id)
    artifacts = {
        "metadata": _artifact_info(paths.metadata, store.root),
        "transcript_raw": _artifact_info(paths.raw_transcript, store.root),
        "transcript_normalized": _artifact_info(paths.normalized_transcript, store.root),
        "transcript_ru": _artifact_info(paths.transcript_ru, store.root),
        "analysis_source": _artifact_info(paths.analysis_source, store.root),
        "analysis_ru": _artifact_info(paths.analysis_ru, store.root),
        "note": _artifact_info(paths.note, store.root),
        "graph_extraction": _artifact_info(paths.graph_extraction, store.root),
        "ru_artifact_review": _artifact_info(paths.ru_artifact_review, store.root),
    }
    visual_dir = store.video_dir(job.video_id) / "visual"
    visual_summary = visual_dir / "visual_summary.json"
    visual_note = visual_dir / "visual_note.md"
    if visual_summary.exists():
        artifacts["visual_summary"] = _artifact_info(visual_summary, store.root)
    if visual_note.exists():
        artifacts["visual_note"] = _artifact_info(visual_note, store.root)

    readiness = {
        "has_russian_note": paths.note.exists() and paths.analysis_ru.exists() and paths.transcript_ru.exists(),
        "has_russian_graph": paths.graph_extraction.exists(),
        "ready_for_neo4j": paths.graph_extraction.exists() and paths.ru_artifact_review.exists(),
        "ready_for_user_reading": paths.note.exists() and paths.analysis_ru.exists(),
    }
    product_quality = _product_quality_payload(store, job, runtime.list_stage_plan(job_id))
    return {
        "job_id": job.id,
        "job_kind": job.job_kind,
        "video_id": job.video_id,
        "quality_mode": job.quality_mode,
        "acceptance_mode": job.acceptance_mode,
        "product_quality": product_quality,
        "artifacts": artifacts,
        "readiness": readiness,
    }


def build_job_quality_status(runtime: RuntimeStore, store: LibraryStore, job_id: str) -> dict[str, Any]:
    job = _load_job(runtime, job_id)
    if job.job_kind == "youtube_discovery":
        stages = runtime.list_stage_plan(job_id)
        return {
            "job_id": job.id,
            "job_kind": job.job_kind,
            "video_id": job.video_id,
            "status": job.status,
            "quality_mode": job.quality_mode,
            "acceptance_mode": job.acceptance_mode,
            "product_quality": _product_quality_payload(store, job, stages),
            "youtube_discovery": {
                "status": job.status,
                "terminal": job.status in {"completed", "completed_with_warning", "failed", "skipped"},
                "warning": job.warning,
                "error": job.error,
            },
        }
    stages = runtime.list_stage_plan(job_id)
    if job.quality_mode == "draft":
        quality_payload = {
            "status": "completed_with_warning",
            "hard_failures": [],
            "warnings": ["draft mode: not production validated"],
            "details": {},
        }
        quality_status = quality_payload["status"]
    else:
        quality = run_quality_gate(store, job.video_id)
        quality_payload = quality.as_dict()
        quality_status = quality.status
    review = load_ru_artifact_review(store, job.video_id)
    review_payload = review if review is not None else {"status": "missing", "issues": [], "coverage": {}}
    neo4j_stage = next((stage for stage in stages if stage.stage_name == "neo4j_sync"), None)
    product_quality = build_product_quality_payload(store, job.video_id, job.quality_mode, quality_payload, review_payload, stages)
    return {
        "job_id": job.id,
        "job_kind": job.job_kind,
        "video_id": job.video_id,
        "status": quality_status,
        "quality_mode": job.quality_mode,
        "acceptance_mode": job.acceptance_mode,
        "product_quality": product_quality,
        "quality_gate": {
            "status": quality_payload["status"],
            "hard_failures": quality_payload["hard_failures"],
            "warnings": quality_payload["warnings"],
            "details": quality_payload["details"],
        },
        "ru_artifact_review": {
            "status": review_payload.get("status", "missing"),
            "issues": review_payload.get("issues") or [],
            "coverage": review_payload.get("coverage") or {},
        },
        "neo4j": {
            "sync_stage_status": neo4j_stage.status if neo4j_stage is not None else "missing",
            "consistency": {"checked": False, "missing": None, "extra": None},
        },
    }


def build_channel_job_progress(runtime: RuntimeStore, store: LibraryStore, channel_job_id: str) -> dict[str, Any]:
    try:
        channel_job = runtime.recompute_channel_job_status(channel_job_id)
    except KeyError as exc:
        raise ChannelJobNotFoundError(str(exc)) from exc
    video_jobs = runtime.list_video_jobs(channel_job_id, job_kind="video_processing")
    counts = {
        "queued": 0,
        "running": 0,
        "completed": 0,
        "completed_with_warning": 0,
        "failed": 0,
        "skipped": 0,
    }
    for job in video_jobs:
        counts[job.status] = counts.get(job.status, 0) + 1
    skipped_requires_media_count = sum(1 for job in video_jobs if _is_skipped_requires_media(job))
    stage_plans = {job.id: runtime.list_stage_plan(job.id) for job in video_jobs}
    active_stages = [
        {
            "job_id": job.id,
            "video_id": job.video_id,
            "stage_name": stage.stage_name,
            "resource_lock": stage.resource_lock,
            "lease_owner": stage.lease_owner,
        }
        for job in video_jobs
        for stage in stage_plans[job.id]
        if stage.status == "running"
    ]
    videos = [_channel_video_summary(runtime, store, job, stage_plans[job.id]) for job in video_jobs]
    eta_values = [
        video.get("eta", {}).get("seconds_remaining")
        for video in videos
        if video.get("eta", {}).get("seconds_remaining") is not None
    ]
    return {
        "channel_job": channel_job.to_dict(),
        "channel_job_id": channel_job.id,
        "channel_id": channel_job.channel_id,
        "status": channel_job.status,
        "quality_mode": channel_job.quality_mode,
        "acceptance_mode": channel_job.acceptance_mode,
        "total_videos": len(video_jobs),
        "counts": counts,
        "skipped_requires_media_count": skipped_requires_media_count,
        "active_stages": active_stages,
        "video_jobs": [job.to_dict() for job in video_jobs],
        "videos": videos,
        "eta": {
            "seconds_remaining": round(sum(float(value) for value in eta_values), 3) if eta_values else None,
            "confidence": "low" if eta_values else "none",
            "basis": "child_eta_sum" if eta_values else "not_enough_history",
        },
    }


def _channel_video_summary(
    runtime: RuntimeStore,
    store: LibraryStore,
    job,
    stages: list[StagePlanRecord],
) -> dict[str, Any]:
    title = None
    metadata = store.load_video_metadata(job.video_id)
    if metadata is not None:
        title = metadata.title
    product_quality = _product_quality_payload(store, job, stages)
    artifacts = build_job_artifacts_status(runtime, store, job.id)
    from yt_sub_hub.recovery import build_job_recovery_summary

    return {
        "video_id": job.video_id,
        "title": title,
        "job_id": job.id,
        "status": job.status,
        "current_stage": job.current_stage,
        "quality_mode": job.quality_mode,
        "acceptance_mode": job.acceptance_mode,
        "product_quality": product_quality,
        "recovery": build_job_recovery_summary(runtime, store, job.id, job=job, stages=stages),
        "artifact_readiness": artifacts["readiness"],
        "eta": _estimate_eta(runtime, stages),
    }


def _product_quality_payload(store: LibraryStore, job, stages: list[StagePlanRecord]) -> dict[str, Any]:
    if getattr(job, "job_kind", "video_processing") == "youtube_discovery":
        return {
            "mode": "youtube_discovery",
            "status": job.status,
            "production_ready": False,
            "graph_sync_allowed": False,
            "reasons": ["youtube discovery jobs do not produce core video product artifacts"],
        }
    if _is_skipped_requires_media(job):
        return {
            "mode": job.quality_mode,
            "status": "skipped_requires_media",
            "production_ready": False,
            "graph_sync_allowed": False,
            "reasons": ["requires media/Whisper fallback, but data_api_only mode forbids media download"],
        }
    if job.quality_mode == "draft":
        return build_product_quality_payload(store, job.video_id, job.quality_mode, None, None, stages)
    if job.status not in {"completed", "completed_with_warning", "failed", "skipped"}:
        return {
            "mode": job.quality_mode,
            "status": "processing",
            "production_ready": False,
            "graph_sync_allowed": False,
            "reasons": ["job is still processing"],
        }
    quality = _safe_quality_gate(store, job.video_id)
    review = load_ru_artifact_review(store, job.video_id)
    return build_product_quality_payload(store, job.video_id, job.quality_mode, quality, review, stages)


def _build_discovery_artifacts_status(runtime: RuntimeStore, store: LibraryStore, job) -> dict[str, Any]:
    discovery_dir = store.video_dir(job.video_id) / "youtube_discovery"
    artifacts = {
        "seeds": _artifact_info(discovery_dir / "seeds.json", store.root),
        "queries": _artifact_info(discovery_dir / "queries.json", store.root),
        "video_candidates": _artifact_info(discovery_dir / "video_candidates.json", store.root),
        "channel_candidates": _artifact_info(discovery_dir / "channel_candidates.json", store.root),
        "review": _artifact_info(discovery_dir / "review.json", store.root),
        "graph": _artifact_info(discovery_dir / "graph.json", store.root),
        "discovery_report": _artifact_info(discovery_dir / "discovery_report.md", store.root),
    }
    return {
        "job_id": job.id,
        "job_kind": job.job_kind,
        "video_id": job.video_id,
        "quality_mode": job.quality_mode,
        "acceptance_mode": job.acceptance_mode,
        "product_quality": _product_quality_payload(store, job, runtime.list_stage_plan(job.id)),
        "artifacts": artifacts,
        "readiness": {
            "has_discovery_report": artifacts["discovery_report"]["exists"],
            "has_discovery_review": artifacts["review"]["exists"],
            "ready_for_user_reading": artifacts["discovery_report"]["exists"],
            "ready_for_neo4j": artifacts["graph"]["exists"],
        },
    }


def _is_skipped_requires_media(job) -> bool:
    return (
        getattr(job, "status", None) == "skipped"
        and getattr(job, "current_stage", None) == "transcript_source"
        and "skipped_requires_media" in str(getattr(job, "warning", "") or getattr(job, "error", ""))
    )


def _safe_quality_gate(store: LibraryStore, video_id: str):
    try:
        return run_quality_gate(store, video_id)
    except Exception as exc:
        return {"status": "failed", "hard_failures": [str(exc)], "warnings": [], "details": {}}


def _load_job(runtime: RuntimeStore, job_id: str):
    try:
        return runtime.load_video_job(job_id)
    except KeyError as exc:
        raise JobNotFoundError(str(exc)) from exc


def _group_chunk_stages(stages: list[StagePlanRecord], retry_counts: dict[str, int] | None = None) -> dict[str, dict[str, Any]]:
    retry_counts = retry_counts or {}
    grouped: dict[str, dict[str, Any]] = {
        "transcript_translation": _empty_chunk_group(),
        "analysis": _empty_chunk_group(),
        "youtube_discovery_queries": _empty_chunk_group(),
    }
    for stage in stages:
        group = _chunk_group(stage.stage_name)
        if group is None:
            continue
        bucket = grouped[group]
        bucket["total"] += 1
        if stage.status == "completed_with_warning":
            bucket["completed"] += 1
        elif stage.status in {"completed", "failed", "running", "pending"}:
            bucket[stage.status] += 1
        elif stage.status == "skipped":
            bucket["completed"] += 1
        bucket["retried"] += retry_counts.get(stage.stage_name, max(0, stage.attempt - 1))
    for bucket in grouped.values():
        bucket["progress_ratio"] = _ratio(bucket["completed"], bucket["total"])
    return grouped


def _stage_duration_from_runs(runtime: RuntimeStore, stage_plan: StagePlanRecord) -> float | None:
    runs = _historical_durations_by_stage(runtime, video_job_id=stage_plan.video_job_id)
    durations = runs.get(stage_plan.stage_name) or []
    return durations[-1] if durations else None


def _artifact_info(path: Path, library_root: Path) -> dict[str, Any]:
    path = Path(path)
    try:
        display_path = str(path.relative_to(library_root))
    except ValueError:
        display_path = str(path)
    if not path.exists():
        return {"path": display_path, "exists": False, "size_bytes": None, "updated_at": None}
    stat = path.stat()
    return {
        "path": display_path,
        "exists": True,
        "size_bytes": stat.st_size,
        "updated_at": _format_ts(stat.st_mtime),
    }


def _estimate_eta(
    runtime: RuntimeStore,
    stages: list[StagePlanRecord],
    historical_durations: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    active = [stage for stage in stages if stage.status in {"pending", "running"}]
    if not active:
        return {"seconds_remaining": None, "confidence": "none", "basis": "not_enough_history"}
    historical_durations = historical_durations or _historical_durations_by_stage(runtime)
    active_group = _chunk_group(active[0].stage_name)
    if active_group is not None:
        group_stages = [stage for stage in stages if _chunk_group(stage.stage_name) == active_group]
        remaining = sum(1 for stage in group_stages if stage.status in {"pending", "running"})
        durations = [
            duration
            for stage in group_stages
            for duration in historical_durations.get(stage.stage_name, [])
            if duration > 0
        ]
        if durations:
            resource_lock = next((stage.resource_lock for stage in group_stages if stage.resource_lock), None)
            seconds = mean(durations) * remaining / _resource_concurrency(resource_lock, runtime)
            return {"seconds_remaining": round(seconds, 3), "confidence": "medium", "basis": "chunk_average"}
        return {"seconds_remaining": None, "confidence": "low", "basis": "not_enough_history"}
    stage_name = active[0].stage_name
    durations = [duration for duration in historical_durations.get(stage_name, []) if duration > 0]
    if durations:
        return {"seconds_remaining": round(mean(durations) * len(active), 3), "confidence": "low", "basis": "stage_run_history"}
    return {"seconds_remaining": None, "confidence": "low", "basis": "not_enough_history"}


def _collect_warning_reasons(stages: list[StagePlanRecord], job_warning: str | None = None) -> list[dict[str, str]]:
    warnings = [{"stage": stage.stage_name, "message": stage.warning} for stage in stages if stage.warning]
    if job_warning and not warnings:
        warnings.append({"stage": "job", "message": job_warning})
    return warnings


def _collect_hard_failures(stages: list[StagePlanRecord], job_error: str | None = None) -> list[dict[str, str]]:
    failures = [
        {"stage": stage.stage_name, "message": stage.error or "failed"}
        for stage in stages
        if stage.status == "failed" and stage.hard_failure
    ]
    if job_error and not failures:
        failures.append({"stage": "job", "message": job_error})
    return failures


def _overall_progress(stages: list[StagePlanRecord]) -> dict[str, Any]:
    result = {
        "total_stages": len(stages),
        "completed": 0,
        "completed_with_warning": 0,
        "failed": 0,
        "pending": 0,
        "running": 0,
        "skipped": 0,
    }
    for stage in stages:
        if stage.status in result:
            result[stage.status] += 1
    terminal = result["completed"] + result["completed_with_warning"] + result["skipped"]
    result["progress_ratio"] = _ratio(terminal, result["total_stages"])
    return result


def _stage_payload(stage: StagePlanRecord, run: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = stage.to_dict()
    payload.update(
        {
            "name": stage.stage_name,
            "stage_name": stage.stage_name,
            "status": stage.status,
            "position": stage.position,
            "attempt": stage.attempt,
            "max_attempts": stage.max_attempts,
            "resource_lock": stage.resource_lock,
            "started_at": _format_ts(run.get("started_at")) if run else None,
            "completed_at": _format_ts(stage.completed_at),
            "duration_seconds": _duration(run),
            "warning": stage.warning,
            "error": stage.error,
        }
    )
    return payload


def _empty_chunk_group() -> dict[str, Any]:
    return {
        "total": 0,
        "completed": 0,
        "failed": 0,
        "running": 0,
        "pending": 0,
        "retried": 0,
        "progress_ratio": 0.0,
    }


def _retry_counts_by_stage(runtime: RuntimeStore, job_id: str) -> dict[str, int]:
    with _readonly_connection(runtime.db_path) as conn:
        rows = conn.execute(
            """
            SELECT sr.stage_name, COUNT(r.id)
            FROM stage_runs sr
            JOIN retries r ON r.stage_run_id = sr.id
            WHERE sr.video_job_id = ?
            GROUP BY sr.stage_name
            """,
            (job_id,),
        ).fetchall()
    return {str(stage_name): int(count) for stage_name, count in rows}


def _latest_runs_by_stage(runtime: RuntimeStore, job_id: str) -> dict[str, dict[str, Any]]:
    with _readonly_connection(runtime.db_path) as conn:
        rows = conn.execute(
            """
            SELECT stage_name, started_at, finished_at, status, warning, error_message
            FROM stage_runs
            WHERE video_job_id = ?
            ORDER BY started_at, attempt
            """,
            (job_id,),
        ).fetchall()
    latest: dict[str, dict[str, Any]] = {}
    for stage_name, started_at, finished_at, status, warning, error_message in rows:
        latest[str(stage_name)] = {
            "started_at": started_at,
            "finished_at": finished_at,
            "status": status,
            "warning": warning,
            "error_message": error_message,
        }
    return latest


def _historical_durations_by_stage(
    runtime: RuntimeStore,
    *,
    video_job_id: str | None = None,
) -> dict[str, list[float]]:
    query = """
        SELECT stage_name, started_at, finished_at
        FROM stage_runs
        WHERE started_at IS NOT NULL
          AND finished_at IS NOT NULL
          AND status IN ('completed', 'completed_with_warning')
    """
    params: tuple[Any, ...] = ()
    if video_job_id is not None:
        query += " AND video_job_id = ?"
        params = (video_job_id,)
    with _readonly_connection(runtime.db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    durations: dict[str, list[float]] = {}
    for stage_name, started_at, finished_at in rows:
        duration = float(finished_at) - float(started_at)
        if duration >= 0:
            durations.setdefault(str(stage_name), []).append(duration)
    return durations


def _readonly_connection(db_path: Path | str):
    path = Path(db_path)
    uri = f"file:{path}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _chunk_group(stage_name: str) -> str | None:
    for group, pattern in _CHUNK_GROUPS.items():
        if pattern.match(stage_name):
            return group
    return None


def _resource_concurrency(resource_lock: str | None, runtime: RuntimeStore | None = None) -> int:
    if not resource_lock:
        return 1
    return _resource_concurrency_limit(resource_lock, getattr(runtime, "settings", None))


def _duration(run: dict[str, Any] | None) -> float | None:
    if not run or run.get("started_at") is None or run.get("finished_at") is None:
        return None
    return round(max(0.0, float(run["finished_at"]) - float(run["started_at"])), 3)


def _format_ts(value: float | int | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(float(value), UTC).isoformat().replace("+00:00", "Z")


def _ratio(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) / float(denominator), 6)
