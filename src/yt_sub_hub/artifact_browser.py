from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from yt_sub_hub.config import get_settings
from yt_sub_hub.library import LibraryStore
from yt_sub_hub.product_quality import build_product_quality_payload
from yt_sub_hub.quality import run_quality_gate
from yt_sub_hub.runtime import RuntimeStore, StagePlanRecord


class ArtifactKeyError(KeyError):
    pass


class ArtifactMissingError(FileNotFoundError):
    pass


@dataclass(frozen=True)
class ArtifactSpec:
    key: str
    label: str
    kind: str
    relative_path: Path
    content_type: str


@dataclass(frozen=True)
class ResolvedArtifact:
    spec: ArtifactSpec
    path: Path
    relative_path: str


ARTIFACT_SPECS: dict[str, ArtifactSpec] = {
    "transcript_source": ArtifactSpec(
        key="transcript_source",
        label="Transcript Source",
        kind="json",
        relative_path=Path("transcript.normalized.json"),
        content_type="application/json",
    ),
    "transcript_ru": ArtifactSpec(
        key="transcript_ru",
        label="Transcript RU",
        kind="json",
        relative_path=Path("transcript.ru.json"),
        content_type="application/json",
    ),
    "analysis_ru": ArtifactSpec(
        key="analysis_ru",
        label="Analysis RU",
        kind="json",
        relative_path=Path("analysis.ru.json"),
        content_type="application/json",
    ),
    "note_md": ArtifactSpec(
        key="note_md",
        label="Russian Note",
        kind="markdown",
        relative_path=Path("note.md"),
        content_type="text/markdown",
    ),
    "graph_extraction": ArtifactSpec(
        key="graph_extraction",
        label="Graph Extraction",
        kind="json",
        relative_path=Path("graph_extraction.json"),
        content_type="application/json",
    ),
    "ru_artifact_review": ArtifactSpec(
        key="ru_artifact_review",
        label="RU Artifact Review",
        kind="json",
        relative_path=Path("review") / "ru_artifact_review.json",
        content_type="application/json",
    ),
    "metadata": ArtifactSpec(
        key="metadata",
        label="Metadata",
        kind="json",
        relative_path=Path("metadata.json"),
        content_type="application/json",
    ),
    "research_report": ArtifactSpec(
        key="research_report",
        label="Research Report",
        kind="markdown",
        relative_path=Path("research") / "research_report.md",
        content_type="text/markdown",
    ),
    "research_findings": ArtifactSpec(
        key="research_findings",
        label="Research Findings",
        kind="json",
        relative_path=Path("research") / "findings.ru.json",
        content_type="application/json",
    ),
    "research_review": ArtifactSpec(
        key="research_review",
        label="Research Artifact Review",
        kind="json",
        relative_path=Path("research") / "review" / "research_artifact_review.json",
        content_type="application/json",
    ),
    "research_graph": ArtifactSpec(
        key="research_graph",
        label="Research Graph",
        kind="json",
        relative_path=Path("research") / "research_graph.json",
        content_type="application/json",
    ),
    "youtube_discovery_seeds": ArtifactSpec(
        key="youtube_discovery_seeds",
        label="YouTube Discovery Seeds",
        kind="json",
        relative_path=Path("youtube_discovery") / "seeds.json",
        content_type="application/json",
    ),
    "youtube_discovery_queries": ArtifactSpec(
        key="youtube_discovery_queries",
        label="YouTube Discovery Queries",
        kind="json",
        relative_path=Path("youtube_discovery") / "queries.json",
        content_type="application/json",
    ),
    "youtube_discovery_video_candidates": ArtifactSpec(
        key="youtube_discovery_video_candidates",
        label="YouTube Discovery Video Candidates",
        kind="json",
        relative_path=Path("youtube_discovery") / "video_candidates.json",
        content_type="application/json",
    ),
    "youtube_discovery_channel_candidates": ArtifactSpec(
        key="youtube_discovery_channel_candidates",
        label="YouTube Discovery Channel Candidates",
        kind="json",
        relative_path=Path("youtube_discovery") / "channel_candidates.json",
        content_type="application/json",
    ),
    "youtube_discovery_review": ArtifactSpec(
        key="youtube_discovery_review",
        label="YouTube Discovery Review",
        kind="json",
        relative_path=Path("youtube_discovery") / "review.json",
        content_type="application/json",
    ),
    "youtube_discovery_graph": ArtifactSpec(
        key="youtube_discovery_graph",
        label="YouTube Discovery Graph",
        kind="json",
        relative_path=Path("youtube_discovery") / "graph.json",
        content_type="application/json",
    ),
    "youtube_discovery_report": ArtifactSpec(
        key="youtube_discovery_report",
        label="YouTube Discovery Report",
        kind="markdown",
        relative_path=Path("youtube_discovery") / "discovery_report.md",
        content_type="text/markdown",
    ),
}


NEO4J_BROWSER_QUERY = "MATCH p=(v:Video {video_id: $video_id})-[*1..2]-(n) RETURN p LIMIT 100"


def build_artifact_browser_for_job(
    runtime: RuntimeStore,
    store: LibraryStore,
    job_id: str,
    *,
    base_url: str | None = None,
) -> dict[str, Any]:
    job = runtime.load_video_job(job_id)
    stages = runtime.list_stage_plan(job_id)
    return build_artifact_browser_for_video(
        store,
        job.video_id,
        base_url=base_url,
        job_id=job.id,
        status=job.status,
        quality_mode=job.quality_mode,
        product_quality=_product_quality_for_job(store, job, stages),
        neo4j_status=_neo4j_status(store, job.video_id, stages),
    )


def build_artifact_browser_for_video(
    store: LibraryStore,
    video_id: str,
    *,
    base_url: str | None = None,
    job_id: str | None = None,
    status: str | None = None,
    quality_mode: str | None = None,
    product_quality: dict[str, Any] | None = None,
    neo4j_status: str | None = None,
) -> dict[str, Any]:
    artifacts = [_artifact_entry(store, video_id, spec, base_url=base_url) for spec in ARTIFACT_SPECS.values()]
    artifact_by_key = {artifact["key"]: artifact for artifact in artifacts}
    metadata = _read_metadata(store, video_id)
    graph_exists = bool(artifact_by_key["graph_extraction"]["exists"])

    mode = quality_mode or get_settings().quality_mode
    return {
        "job_id": job_id,
        "video_id": video_id,
        "title": metadata.get("title"),
        "status": status,
        "quality_mode": mode,
        "product_quality": product_quality or _product_quality_for_video(store, video_id, mode, []),
        "artifacts": artifacts,
        "neo4j": {
            "browser_url": get_settings().neo4j_browser_url,
            "query": NEO4J_BROWSER_QUERY,
            "parameters": {"video_id": video_id},
            "status": neo4j_status or ("unknown" if graph_exists else "unavailable"),
        },
        "readiness": {
            "ready_for_reading": bool(
                artifact_by_key["note_md"]["exists"]
                and artifact_by_key["analysis_ru"]["exists"]
                and artifact_by_key["transcript_ru"]["exists"]
            ),
            "ready_for_graph": graph_exists,
        },
    }


def resolve_artifact(store: LibraryStore, video_id: str, artifact_key: str) -> ResolvedArtifact:
    spec = ARTIFACT_SPECS.get(artifact_key)
    if spec is None:
        raise ArtifactKeyError(artifact_key)

    path = store.video_dir(video_id) / spec.relative_path
    root = store.root.resolve()
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ArtifactKeyError(artifact_key)
    relative_path = str(resolved.relative_to(root))
    return ResolvedArtifact(spec=spec, path=resolved, relative_path=relative_path)


def read_artifact_content(store: LibraryStore, video_id: str, artifact_key: str) -> Any:
    artifact = resolve_artifact(store, video_id, artifact_key)
    if not artifact.path.exists() or not artifact.path.is_file():
        raise ArtifactMissingError(artifact.relative_path)
    if artifact.spec.kind == "json":
        return json.loads(artifact.path.read_text(encoding="utf-8"))
    return {
        "content": artifact.path.read_text(encoding="utf-8"),
        "content_type": artifact.spec.content_type,
    }


def _artifact_entry(
    store: LibraryStore,
    video_id: str,
    spec: ArtifactSpec,
    *,
    base_url: str | None,
) -> dict[str, Any]:
    try:
        artifact = resolve_artifact(store, video_id, spec.key)
        path = artifact.path
        relative_path = artifact.relative_path
    except ArtifactKeyError:
        path = None
        relative_path = str(Path("videos") / video_id / spec.relative_path)

    exists = path is not None and path.exists() and path.is_file()
    stat = path.stat() if exists else None
    quoted_video_id = quote(video_id, safe="")
    api_url = _join_url(base_url, f"/videos/{quoted_video_id}/artifacts/{spec.key}")
    download_url = _join_url(base_url, f"/videos/{quoted_video_id}/artifacts/{spec.key}/download")
    return {
        "key": spec.key,
        "label": spec.label,
        "kind": spec.kind,
        "path": relative_path,
        "exists": exists,
        "size_bytes": stat.st_size if stat is not None else None,
        "updated_at": _format_ts(stat.st_mtime) if stat is not None else None,
        "api_url": api_url,
        "download_url": download_url,
    }


def _read_metadata(store: LibraryStore, video_id: str) -> dict[str, Any]:
    try:
        artifact = resolve_artifact(store, video_id, "metadata")
    except ArtifactKeyError:
        return {}
    if not artifact.path.exists():
        return {}
    try:
        data = json.loads(artifact.path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _neo4j_status(store: LibraryStore, video_id: str, stages: list[StagePlanRecord]) -> str:
    try:
        graph_artifact = resolve_artifact(store, video_id, "graph_extraction")
    except ArtifactKeyError:
        return "unavailable"
    if not graph_artifact.path.exists():
        return "unavailable"
    neo4j_stage = next((stage for stage in stages if stage.stage_name == "neo4j_sync"), None)
    if neo4j_stage is None:
        return "unknown"
    if neo4j_stage.status in {"completed", "completed_with_warning"}:
        return "available"
    if neo4j_stage.status in {"failed", "skipped"}:
        return "unavailable"
    return "unknown"


def _product_quality_for_video(
    store: LibraryStore,
    video_id: str,
    quality_mode: str,
    stages: list[StagePlanRecord],
) -> dict[str, Any]:
    if quality_mode == "draft":
        return build_product_quality_payload(store, video_id, quality_mode, None, None, stages)
    from yt_sub_hub.artifact_review import load_ru_artifact_review

    return build_product_quality_payload(
        store,
        video_id,
        quality_mode,
        _safe_quality_gate(store, video_id),
        load_ru_artifact_review(store, video_id),
        stages,
    )


def _product_quality_for_job(
    store: LibraryStore,
    job,
    stages: list[StagePlanRecord],
) -> dict[str, Any]:
    if getattr(job, "job_kind", "video_processing") == "youtube_discovery":
        return {
            "mode": "youtube_discovery",
            "status": job.status,
            "production_ready": False,
            "graph_sync_allowed": False,
            "reasons": ["youtube discovery jobs do not produce core video product artifacts"],
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
    return _product_quality_for_video(store, job.video_id, job.quality_mode, stages)


def _safe_quality_gate(store: LibraryStore, video_id: str):
    try:
        return run_quality_gate(store, video_id)
    except Exception as exc:
        return {"status": "failed", "hard_failures": [str(exc)], "warnings": [], "details": {}}


def _join_url(base_url: str | None, path: str) -> str:
    if not base_url:
        return path
    return f"{base_url.rstrip('/')}{path}"


def _format_ts(value: float) -> str:
    return datetime.fromtimestamp(float(value), UTC).isoformat().replace("+00:00", "Z")
