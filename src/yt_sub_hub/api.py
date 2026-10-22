from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from yt_sub_hub.config import get_settings
from yt_sub_hub.graph import Neo4jGraphStore
from yt_sub_hub.library import LibraryStore
from yt_sub_hub.orchestration import (
    ensure_legacy_job_for_runtime,
    recover_interrupted_legacy_jobs,
    requeue_failed_job,
    sync_legacy_job_from_runtime,
    sync_legacy_requeued_job,
)
from yt_sub_hub.pipeline import VideoProcessingPipeline, job_id_for_url
from yt_sub_hub.visual import VisualAnalysisError, VisualAnalysisService

try:
    from pydantic import BaseModel
except ImportError:  # pragma: no cover - handled by create_app
    BaseModel = None

try:
    from fastapi import BackgroundTasks as FastAPIBackgroundTasks
except ImportError:  # pragma: no cover - handled by create_app
    FastAPIBackgroundTasks = None

if BaseModel is not None:

    class JobRequest(BaseModel):
        url: str
        force: bool = False
        quality_mode: str | None = None
        acceptance_mode: str | None = None

    class ChannelJobRequest(BaseModel):
        channel_url: str
        limit: int | None = None
        quality_mode: str | None = None
        acceptance_mode: str | None = None
        force: bool = False

    class FrameAnalysisRequest(BaseModel):
        timestamp_seconds: float
        question: str | None = None
        force: bool = False

    class RecoveryRetryStageRequest(BaseModel):
        stage_name: str

    class RecoveryRetryFailedChunksRequest(BaseModel):
        group: str

    class RecoveryResetDownstreamRequest(BaseModel):
        from_stage: str

    class YouTubeDiscoveryRunRequest(BaseModel):
        limit: int = 20
        max_quota_units: int = 900

    class YouTubeDiscoveryQueueRequest(BaseModel):
        limit: int = 20
        max_quota_units: int = 900
        force: bool = False

    class YouTubeDiscoveryEnqueueRequest(BaseModel):
        min_score: float = 0.75
        limit: int = 5
        quality_mode: str | None = None
        acceptance_mode: str | None = None

    class YouTubeDiscoveryChannelEnqueueRequest(BaseModel):
        channel_id: str
        limit: int | None = None
        quality_mode: str | None = None
        acceptance_mode: str | None = None


def create_job_record(url: str, library_dir):
    store = LibraryStore(library_dir)
    pipeline = VideoProcessingPipeline(store=store, enable_default_graph=False)
    return pipeline.submit(url)


def process_existing_job(job_id: str, library_dir=None) -> None:
    settings = get_settings()
    root = Path(library_dir) if library_dir is not None else settings.library_dir
    store = LibraryStore(root)

    from yt_sub_hub.runtime import RuntimeStore
    from yt_sub_hub.worker import StageWorker

    runtime = RuntimeStore(root / "index.sqlite3", settings=replace(settings, library_dir=root))
    recover_interrupted_legacy_jobs(
        store,
        runtime,
        stale_after_seconds=settings.runtime_stale_after_seconds,
    )

    def cancel_requested(cancel_job_id: str) -> bool:
        try:
            return runtime.load_video_job(cancel_job_id).cancel_requested
        except KeyError:
            return False

    def processor(job):
        _ensure_legacy_job_for_runtime(store, job)
        pipeline_settings = replace(
            settings,
            library_dir=root,
            quality_mode=job.quality_mode,
            acceptance_mode=job.acceptance_mode,
        )
        return VideoProcessingPipeline(store=store, cancel_checker=cancel_requested, settings=pipeline_settings).process_job(job.id)

    try:
        runtime_record = runtime.load_video_job(job_id)
    except KeyError:
        runtime_record = None
    if runtime_record is None:
        VideoProcessingPipeline(store=store, cancel_checker=cancel_requested).process_job(job_id)
        return
    if runtime_record.cancel_requested and runtime_record.status == "running" and runtime_record.lease_owner is None:
        result = runtime.skip_video_job(job_id, warning="cancel requested")
        _sync_legacy_job_from_runtime(store, result)
        return

    result = StageWorker(
        runtime,
        owner="api-background",
        lease_seconds=300,
        library_dir=root,
        target_language=get_settings().target_language,
        quality_gate_enabled=True,
    ).process_job(job_id, processor)
    if result is not None:
        _sync_legacy_job_from_runtime(store, result)
    if result is not None and result.status == "failed" and not (result.error or "").startswith("Quality gate"):
        raise RuntimeError(result.error or "runtime worker failed")


def _ensure_legacy_job_for_runtime(store: LibraryStore, job) -> None:
    ensure_legacy_job_for_runtime(store, job)


def _sync_legacy_job_from_runtime(store: LibraryStore, runtime_record) -> None:
    sync_legacy_job_from_runtime(store, runtime_record)


def create_app():
    try:
        from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
        from fastapi.responses import FileResponse
    except ImportError as exc:
        raise RuntimeError("fastapi and pydantic are required for the API") from exc
    if BaseModel is None:
        raise RuntimeError("fastapi and pydantic are required for the API")

    app = FastAPI(title="YouTube Sub Hub")
    settings = get_settings()
    store = LibraryStore(settings.library_dir)

    def runtime_store():
        from yt_sub_hub.runtime import RuntimeStore

        return RuntimeStore(settings.library_dir / "index.sqlite3", settings=settings)

    @app.post("/jobs")
    def create_job(request: JobRequest, background_tasks: FastAPIBackgroundTasks):
        pipeline = VideoProcessingPipeline(store=store, enable_default_graph=False)
        existing = pipeline.already_processed(request.url)
        if existing is not None and not request.force:
            video_id, note_path = existing
            return {
                "status": "already_processed",
                "video_id": video_id,
                "note_path": note_path,
                "message": "Video already analyzed. Send force=true to run a new analysis.",
            }
        from yt_sub_hub.config import normalize_acceptance_mode, normalize_quality_mode

        quality_mode = normalize_quality_mode(request.quality_mode or settings.quality_mode)
        acceptance_mode = normalize_acceptance_mode(request.acceptance_mode or settings.acceptance_mode)
        runtime = runtime_store()
        # Reject conflicting retry options before submit can recover an aged legacy record.
        try:
            runtime.update_video_job_modes(
                job_id_for_url(request.url), quality_mode=request.quality_mode, acceptance_mode=request.acceptance_mode,
            )
        except KeyError:
            pass  # First submission has no runtime record yet.
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        record = create_job_record(request.url, settings.library_dir)
        runtime_record = runtime.create_video_job(
            job_id=record.job_id,
            video_id=record.video_id,
            canonical_url=f"https://www.youtube.com/watch?v={record.video_id}",
            status=record.status,
            quality_mode=quality_mode,
            acceptance_mode=acceptance_mode,
        )
        try:
            runtime_record = runtime.update_video_job_modes(
                runtime_record.id, quality_mode=request.quality_mode, acceptance_mode=request.acceptance_mode,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if runtime_record.status not in {"queued", "running"}:
            runtime_record = runtime.resume_video_job(runtime_record.id)
            sync_legacy_requeued_job(store, runtime_record)
        background_tasks.add_task(process_existing_job, record.job_id, settings.library_dir)
        payload = record.to_dict()
        payload["quality_mode"] = runtime_record.quality_mode
        payload["acceptance_mode"] = runtime_record.acceptance_mode
        return payload

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str):
        record = store.load_jobs().get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="job not found")
        return record.to_dict()

    @app.get("/jobs/{job_id}/stages")
    def get_job_stages(job_id: str, include_chunks: bool = Query(False)):
        from yt_sub_hub.progress import JobNotFoundError, build_job_stage_progress

        runtime = runtime_store()
        try:
            return build_job_stage_progress(runtime, store, job_id, include_chunks=include_chunks)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @app.get("/jobs/{job_id}/artifacts")
    def get_job_artifacts(job_id: str):
        from yt_sub_hub.progress import JobNotFoundError, build_job_artifacts_status

        try:
            return build_job_artifacts_status(runtime_store(), store, job_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @app.get("/jobs/{job_id}/artifact-browser")
    def get_job_artifact_browser(job_id: str):
        from yt_sub_hub.artifact_browser import build_artifact_browser_for_job

        try:
            return build_artifact_browser_for_job(runtime_store(), store, job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @app.get("/jobs/{job_id}/quality")
    def get_job_quality(job_id: str):
        from yt_sub_hub.progress import JobNotFoundError, build_job_quality_status

        try:
            return build_job_quality_status(runtime_store(), store, job_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    def _recovery_dashboard(runtime, job_id: str):
        from yt_sub_hub.recovery import build_job_recovery_dashboard

        return build_job_recovery_dashboard(runtime, store, job_id)

    def _recovery_error(exc: Exception):
        message = str(exc)
        if isinstance(exc, KeyError):
            if "Unknown stage" in message:
                return HTTPException(status_code=400, detail="invalid stage")
            return HTTPException(status_code=404, detail="job not found")
        if "invalid chunk group" in message or "Unknown stage" in message:
            return HTTPException(status_code=400, detail=message)
        return HTTPException(status_code=409, detail=message)

    @app.get("/jobs/{job_id}/recovery")
    def get_job_recovery(job_id: str):
        runtime = runtime_store()
        try:
            return _recovery_dashboard(runtime, job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @app.post("/jobs/{job_id}/recovery/retry-stage")
    def recovery_retry_stage(job_id: str, request: RecoveryRetryStageRequest):
        runtime = runtime_store()
        try:
            runtime_record = runtime.retry_stage(job_id, request.stage_name)
            sync_legacy_requeued_job(store, runtime_record)
            return _recovery_dashboard(runtime, job_id)
        except (KeyError, ValueError) as exc:
            raise _recovery_error(exc) from exc

    @app.post("/jobs/{job_id}/recovery/retry-failed-chunks")
    def recovery_retry_failed_chunks(job_id: str, request: RecoveryRetryFailedChunksRequest):
        runtime = runtime_store()
        try:
            runtime_record = runtime.retry_failed_chunks(job_id, request.group)
            sync_legacy_requeued_job(store, runtime_record)
            return _recovery_dashboard(runtime, job_id)
        except (KeyError, ValueError) as exc:
            raise _recovery_error(exc) from exc

    @app.post("/jobs/{job_id}/recovery/reset-downstream")
    def recovery_reset_downstream(job_id: str, request: RecoveryResetDownstreamRequest):
        runtime = runtime_store()
        try:
            runtime_record = runtime.reset_downstream(job_id, request.from_stage)
            sync_legacy_requeued_job(store, runtime_record)
            return _recovery_dashboard(runtime, job_id)
        except (KeyError, ValueError) as exc:
            raise _recovery_error(exc) from exc

    @app.post("/jobs/{job_id}/recovery/cancel")
    def recovery_cancel_job(job_id: str):
        runtime = runtime_store()
        try:
            runtime_record = runtime.cancel_job(job_id)
            sync_legacy_job_from_runtime(store, runtime_record)
            return _recovery_dashboard(runtime, job_id)
        except (KeyError, ValueError) as exc:
            raise _recovery_error(exc) from exc

    @app.post("/jobs/{job_id}/recovery/resume")
    def recovery_resume_job(job_id: str):
        runtime = runtime_store()
        try:
            runtime_record = runtime.resume_job(job_id)
            sync_legacy_requeued_job(store, runtime_record)
            return _recovery_dashboard(runtime, job_id)
        except (KeyError, ValueError) as exc:
            raise _recovery_error(exc) from exc

    @app.post("/jobs/{job_id}/cancel")
    def cancel_runtime_job(job_id: str):
        try:
            runtime_record = runtime_store().request_cancel(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        sync_legacy_job_from_runtime(store, runtime_record)
        payload = runtime_record.to_dict()
        payload["message"] = "cancel requested; running job will stop at next cancellation checkpoint"
        return payload

    @app.post("/jobs/{job_id}/resume")
    def resume_runtime_job(job_id: str):
        try:
            runtime_record = runtime_store().resume_video_job(job_id)
            sync_legacy_requeued_job(store, runtime_record)
            return runtime_record.to_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @app.post("/jobs/{job_id}/retry-failed")
    def retry_failed_runtime_job(job_id: str, from_stage: str | None = None):
        try:
            runtime = runtime_store()
            try:
                return requeue_failed_job(runtime, store, job_id, from_stage=from_stage).to_dict()
            except ValueError:
                raise HTTPException(status_code=409, detail="job is not failed")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @app.get("/queue")
    def get_queue(stages: bool = Query(False)):
        runtime = runtime_store()
        recover_interrupted_legacy_jobs(store, runtime, stale_after_seconds=settings.runtime_stale_after_seconds)
        payload = runtime.queue_status()
        if stages:
            payload = {**payload, **runtime.stage_queue_status()}
        return payload

    @app.get("/captions-probe")
    def get_captions_probe(target: str, limit: int = Query(3)):
        from yt_sub_hub.captions_probe import CaptionsProbeError, probe_captions

        try:
            return probe_captions(target, limit=limit, settings=settings)
        except CaptionsProbeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/channels/jobs")
    def create_channel_job(request: ChannelJobRequest):
        from yt_sub_hub.channel import ChannelDataApiError, enqueue_channel_videos, fetch_channel_listing
        from yt_sub_hub.config import normalize_acceptance_mode, normalize_quality_mode
        from yt_sub_hub.progress import build_channel_job_progress

        try:
            listing = fetch_channel_listing(request.channel_url, limit=request.limit, settings=settings)
        except ChannelDataApiError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        quality_mode = normalize_quality_mode(request.quality_mode or settings.quality_mode)
        acceptance_mode = normalize_acceptance_mode(request.acceptance_mode or settings.acceptance_mode)
        channel_job, video_jobs = enqueue_channel_videos(
            runtime_store(),
            store,
            channel_url=listing.channel_url,
            channel_id=listing.channel_id,
            uploads_playlist_id=listing.uploads_playlist_id,
            videos=listing.videos,
            limit=request.limit,
            quality_mode=quality_mode,
            acceptance_mode=acceptance_mode,
            force=request.force,
        )
        return build_channel_job_progress(runtime_store(), store, channel_job.id)

    @app.get("/channels/jobs/{job_id}")
    def get_channel_job(job_id: str):
        runtime = runtime_store()
        from yt_sub_hub.progress import ChannelJobNotFoundError, build_channel_job_progress

        try:
            return build_channel_job_progress(runtime, store, job_id)
        except ChannelJobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="channel job not found") from exc

    @app.get("/channels/jobs/{job_id}/videos")
    def get_channel_job_videos(job_id: str):
        runtime = runtime_store()
        from yt_sub_hub.progress import ChannelJobNotFoundError, build_channel_job_progress

        try:
            payload = build_channel_job_progress(runtime, store, job_id)
        except ChannelJobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="channel job not found") from exc
        return {
            "channel_job_id": job_id,
            "videos": payload["videos"],
        }

    @app.post("/channels/jobs/{job_id}/cancel")
    def cancel_channel_job(job_id: str):
        from yt_sub_hub.progress import ChannelJobNotFoundError, build_channel_job_progress

        try:
            runtime_store().cancel_channel_job(job_id)
            payload = build_channel_job_progress(runtime_store(), store, job_id)
        except (KeyError, ChannelJobNotFoundError) as exc:
            raise HTTPException(status_code=404, detail="channel job not found") from exc
        payload["message"] = "cancel requested for non-terminal child jobs"
        return payload

    @app.post("/channels/jobs/{job_id}/retry-failed")
    def retry_failed_channel_job(job_id: str):
        from yt_sub_hub.progress import ChannelJobNotFoundError, build_channel_job_progress

        try:
            runtime_store().retry_channel_failed(job_id)
            return build_channel_job_progress(runtime_store(), store, job_id)
        except (KeyError, ChannelJobNotFoundError) as exc:
            raise HTTPException(status_code=404, detail="channel job not found") from exc

    @app.get("/channels/{channel_id}")
    def get_channel(channel_id: str):
        path = settings.library_dir / "channels" / channel_id / "videos.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="channel not found")
        return LibraryStore._read_json(path)

    @app.get("/channels/{channel_id}/graph")
    def get_channel_graph(channel_id: str):
        path = settings.library_dir / "channels" / channel_id / "graph_export.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="channel graph not found")
        return LibraryStore._read_json(path)

    @app.get("/videos/{video_id}")
    def get_video(video_id: str):
        metadata_path = store.video_dir(video_id) / "metadata.json"
        if not metadata_path.exists():
            raise HTTPException(status_code=404, detail="video not found")
        return LibraryStore._read_json(metadata_path)

    @app.get("/videos/{video_id}/note")
    def get_note(video_id: str):
        note_path = store.note_path(video_id)
        if not note_path.exists():
            raise HTTPException(status_code=404, detail="note not found")
        return {"video_id": video_id, "markdown": note_path.read_text(encoding="utf-8")}

    @app.get("/videos/{video_id}/graph")
    def get_video_graph(video_id: str):
        extraction = store.load_graph_extraction(video_id)
        if extraction is None:
            raise HTTPException(status_code=404, detail="graph extraction not found")
        return extraction

    @app.get("/videos/{video_id}/artifact-browser")
    def get_video_artifact_browser(video_id: str):
        from yt_sub_hub.artifact_browser import build_artifact_browser_for_video

        return build_artifact_browser_for_video(store, video_id)

    @app.get("/videos/{video_id}/artifacts/{artifact_key}")
    def get_video_artifact(video_id: str, artifact_key: str):
        from yt_sub_hub.artifact_browser import ArtifactKeyError, ArtifactMissingError, read_artifact_content

        try:
            return read_artifact_content(store, video_id, artifact_key)
        except (ArtifactKeyError, ArtifactMissingError) as exc:
            raise HTTPException(status_code=404, detail="artifact not found") from exc

    @app.get("/videos/{video_id}/artifacts/{artifact_key}/download")
    def download_video_artifact(video_id: str, artifact_key: str):
        from yt_sub_hub.artifact_browser import ArtifactKeyError, ArtifactMissingError, resolve_artifact

        try:
            artifact = resolve_artifact(store, video_id, artifact_key)
        except ArtifactKeyError as exc:
            raise HTTPException(status_code=404, detail="artifact not found") from exc
        if not artifact.path.exists() or not artifact.path.is_file():
            raise HTTPException(status_code=404, detail="artifact not found")
        return FileResponse(
            artifact.path,
            media_type=artifact.spec.content_type,
            filename=artifact.path.name,
        )

    @app.get("/videos/{video_id}/review/ru-artifacts")
    def get_video_ru_artifact_review(video_id: str):
        path = store.artifact_paths(video_id).ru_artifact_review
        if not path.exists():
            raise HTTPException(status_code=404, detail="ru artifact review not found")
        return LibraryStore._read_json(path)

    @app.post("/videos/{video_id}/research/run")
    def run_video_research(video_id: str):
        from yt_sub_hub.research_enrichment import ResearchValidationError, run_video_research

        try:
            return run_video_research(store, video_id)
        except ResearchValidationError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.get("/videos/{video_id}/research")
    def get_video_research(video_id: str):
        from yt_sub_hub.research_enrichment import build_research_payload

        return build_research_payload(store, video_id)

    @app.get("/videos/{video_id}/research/report")
    def get_video_research_report(video_id: str):
        from yt_sub_hub.research_enrichment import load_research_report_markdown

        try:
            return {"video_id": video_id, "markdown": load_research_report_markdown(store, video_id)}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="research report not found") from exc

    @app.get("/videos/{video_id}/research/review")
    def get_video_research_review(video_id: str):
        from yt_sub_hub.research_enrichment import load_research_review

        try:
            return load_research_review(store, video_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="research review not found") from exc

    @app.post("/videos/{video_id}/research/retry")
    def retry_video_research(video_id: str):
        from yt_sub_hub.research_enrichment import ResearchValidationError, run_video_research

        try:
            return run_video_research(store, video_id)
        except ResearchValidationError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/videos/{video_id}/research/sync-neo4j")
    def sync_video_research_neo4j(video_id: str):
        from yt_sub_hub.research_enrichment import ResearchValidationError, sync_research_graph_to_neo4j

        try:
            payload = sync_research_graph_to_neo4j(store, video_id, settings=settings)
        except ResearchValidationError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            path = Path(str(exc).strip("'"))
            if path.name == "research_graph.json":
                raise HTTPException(status_code=404, detail="research graph not found") from exc
            raise HTTPException(status_code=409, detail="research review not passed") from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        if payload.get("missing") or payload.get("extra"):
            raise HTTPException(status_code=502, detail=payload)
        return payload

    @app.post("/videos/{video_id}/discovery/youtube/run")
    def run_video_youtube_discovery(video_id: str, request: YouTubeDiscoveryRunRequest):
        from yt_sub_hub.youtube_discovery import DiscoveryProviderError, DiscoveryValidationError, run_youtube_discovery

        try:
            return run_youtube_discovery(
                store,
                video_id,
                settings=settings,
                limit=request.limit,
                max_quota_units=request.max_quota_units,
            )
        except DiscoveryValidationError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except DiscoveryProviderError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/videos/{video_id}/discovery/youtube/queue")
    def queue_video_youtube_discovery(video_id: str, request: YouTubeDiscoveryQueueRequest):
        runtime = runtime_store()
        job = runtime.queue_youtube_discovery(
            video_id,
            limit=request.limit,
            max_quota_units=request.max_quota_units,
            force=request.force,
        )
        stages = runtime.list_stage_plan(job.id)
        return {
            "job_id": job.id,
            "job_kind": job.job_kind,
            "video_id": job.video_id,
            "status": job.status,
            "stage_count": len(stages),
            "stages": [stage.stage_name for stage in stages],
        }

    @app.get("/videos/{video_id}/discovery/youtube/stages")
    def get_video_youtube_discovery_stages(video_id: str):
        from yt_sub_hub.progress import build_job_stage_progress

        runtime = runtime_store()
        job = runtime.latest_youtube_discovery_job_for_video(video_id)
        if job is None:
            raise HTTPException(status_code=404, detail="discovery stage plan not found")
        return build_job_stage_progress(runtime, store, job.id, include_chunks=True)

    @app.get("/videos/{video_id}/discovery/youtube")
    def get_video_youtube_discovery(video_id: str):
        from yt_sub_hub.youtube_discovery import build_discovery_payload

        return build_discovery_payload(store, video_id)

    @app.get("/videos/{video_id}/discovery/youtube/review")
    def get_video_youtube_discovery_review(video_id: str):
        from yt_sub_hub.youtube_discovery import load_discovery_review

        try:
            return load_discovery_review(store, video_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="discovery review not found") from exc

    @app.post("/videos/{video_id}/discovery/youtube/sync-neo4j")
    def sync_video_youtube_discovery_neo4j(video_id: str):
        from yt_sub_hub.youtube_discovery import DiscoveryValidationError, sync_discovery_graph_to_neo4j

        try:
            payload = sync_discovery_graph_to_neo4j(store, video_id, settings=settings)
        except DiscoveryValidationError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="discovery graph or review not found") from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        if payload.get("missing") or payload.get("extra"):
            raise HTTPException(status_code=502, detail=payload)
        return payload

    @app.post("/videos/{video_id}/discovery/youtube/enqueue")
    def enqueue_video_youtube_discovery(video_id: str, request: YouTubeDiscoveryEnqueueRequest):
        from yt_sub_hub.config import normalize_acceptance_mode, normalize_quality_mode
        from yt_sub_hub.youtube_discovery import DiscoveryValidationError, enqueue_discovered_videos

        try:
            return enqueue_discovered_videos(
                runtime_store(),
                store,
                video_id,
                min_score=request.min_score,
                limit=request.limit,
                quality_mode=normalize_quality_mode(request.quality_mode or settings.quality_mode),
                acceptance_mode=normalize_acceptance_mode(request.acceptance_mode or settings.acceptance_mode),
            )
        except DiscoveryValidationError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="discovery review or candidates not found") from exc

    @app.post("/videos/{video_id}/discovery/youtube/enqueue-channel")
    def enqueue_channel_youtube_discovery(video_id: str, request: YouTubeDiscoveryChannelEnqueueRequest):
        from yt_sub_hub.config import normalize_acceptance_mode, normalize_quality_mode
        from yt_sub_hub.youtube_discovery import DiscoveryValidationError, enqueue_discovered_channel

        try:
            return enqueue_discovered_channel(
                runtime_store(),
                store,
                video_id,
                channel_id=request.channel_id,
                limit=request.limit,
                quality_mode=normalize_quality_mode(request.quality_mode or settings.quality_mode),
                acceptance_mode=normalize_acceptance_mode(request.acceptance_mode or settings.acceptance_mode),
                settings=settings,
            )
        except DiscoveryValidationError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="discovery review or candidates not found") from exc

    @app.post("/videos/{video_id}/frames/analyze")
    def analyze_video_frame(video_id: str, request: FrameAnalysisRequest):
        service = VisualAnalysisService(settings=settings, store=store)
        try:
            return service.analyze_frame(
                video_id,
                request.timestamp_seconds,
                question=request.question,
                force=request.force,
            )
        except VisualAnalysisError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/graph/concepts")
    def get_concepts(q: str | None = None, limit: int = 100):
        graph_store = Neo4jGraphStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
        try:
            return graph_store.list_concepts(q=q, limit=limit)
        finally:
            graph_store.close()

    @app.get("/graph/neighbors/{concept_id}")
    def get_neighbors(concept_id: str):
        graph_store = Neo4jGraphStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
        try:
            try:
                return graph_store.get_neighbors(concept_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            graph_store.close()

    @app.get("/graph/concepts/{concept_id}/neighbors")
    def get_concept_neighbors(concept_id: str, depth: int = 1, relation: str | None = None, limit: int = 100):
        graph_store = Neo4jGraphStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
        try:
            try:
                return graph_store.get_neighbors(concept_id, depth=depth, relation=relation, limit=limit)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            graph_store.close()

    @app.get("/graph/path")
    def get_graph_path(from_: str = Query(alias="from"), to: str = "", max_depth: int = 4):
        graph_store = Neo4jGraphStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
        try:
            return graph_store.find_path(from_, to, max_depth=max_depth)
        finally:
            graph_store.close()

    @app.get("/graph/claims")
    def get_graph_claims(concept: str | None = None, min_confidence: float = 0.0, limit: int = 100):
        graph_store = Neo4jGraphStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
        try:
            return graph_store.get_claims(concept=concept, min_confidence=min_confidence, limit=limit)
        finally:
            graph_store.close()

    return app
