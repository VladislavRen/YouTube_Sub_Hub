from __future__ import annotations

import json
import os
from pathlib import Path

from yt_sub_hub.lm_benchmark import BenchmarkOptions, BenchmarkReportError, benchmark_lm_studio_translation
from yt_sub_hub.stage_pipeline import StagePipeline
from yt_sub_hub.whisper_benchmark import run_whisper_benchmark


def _pipeline():
    from yt_sub_hub.pipeline import VideoProcessingPipeline

    return VideoProcessingPipeline()


def _visual_service():
    from yt_sub_hub.visual import VisualAnalysisService

    return VisualAnalysisService()


def _runtime_store():
    from yt_sub_hub.config import get_settings
    from yt_sub_hub.runtime import RuntimeStore

    settings = get_settings()
    return RuntimeStore(settings.library_dir / "index.sqlite3", settings=settings)


def _worker_pipeline(store, cancel_checker):
    from yt_sub_hub.pipeline import VideoProcessingPipeline

    return VideoProcessingPipeline(store=store, cancel_checker=cancel_checker)


def _ensure_legacy_job_for_runtime(store, job):
    from yt_sub_hub.orchestration import ensure_legacy_job_for_runtime

    ensure_legacy_job_for_runtime(store, job)


def _recover_interrupted_jobs(store, runtime, stale_after_seconds):
    from yt_sub_hub.orchestration import recover_interrupted_legacy_jobs

    return recover_interrupted_legacy_jobs(store, runtime, stale_after_seconds=stale_after_seconds)


try:
    import typer
except ImportError:  # pragma: no cover - only used when runtime deps are missing
    typer = None


if typer is not None:
    app = typer.Typer(help="YouTube Sub Hub CLI")

    def _echo_payload(payload, *, json_output: bool = False) -> None:
        if json_output:
            typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            return
        typer.echo(payload)

    def _artifact_browser_payload(video_id_or_job_id: str):
        from yt_sub_hub.artifact_browser import build_artifact_browser_for_job, build_artifact_browser_for_video
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.library import LibraryStore

        settings = get_settings()
        runtime = _runtime_store()
        store = LibraryStore(settings.library_dir)
        try:
            return build_artifact_browser_for_job(runtime, store, video_id_or_job_id)
        except KeyError:
            pass

        matching_jobs = [
            job
            for job in runtime.list_video_jobs()
            if job.video_id == video_id_or_job_id and job.job_kind == "video_processing"
        ]
        if matching_jobs:
            latest_job = max(matching_jobs, key=lambda job: job.updated_at)
            return build_artifact_browser_for_job(runtime, store, latest_job.id)
        return build_artifact_browser_for_video(store, video_id_or_job_id)

    def _recovery_dashboard_payload(runtime, job_id: str):
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.library import LibraryStore
        from yt_sub_hub.recovery import build_job_recovery_dashboard

        return build_job_recovery_dashboard(runtime, LibraryStore(get_settings().library_dir), job_id)

    def _echo_recovery_action_error(exc: Exception, job_id: str) -> None:
        message = str(exc)
        if isinstance(exc, KeyError) and "Unknown stage" not in message:
            typer.echo(f"job not found: {job_id}", err=True)
            raise typer.Exit(code=1) from exc
        typer.echo(message, err=True)
        raise typer.Exit(code=1) from exc

    def _latest_runtime_job_for_video(runtime, video_id: str):
        matches = [job for job in runtime.list_video_jobs() if job.video_id == video_id and job.job_kind == "video_processing"]
        return max(matches, key=lambda job: job.updated_at) if matches else None

    @app.command()
    def process(
        url: str,
        force: bool = typer.Option(False, "--force", "-f", help="Reanalyze without prompting; fresh source caches may be reused."),
        refresh: bool = typer.Option(False, "--refresh", help="Fetch live metadata and transcript, then regenerate dependent artifacts; never reuse stale inputs on failure."),
        quality_mode: str | None = typer.Option(None, "--quality-mode", help="Quality mode: strict, best_effort, or draft."),
        acceptance_mode: str | None = typer.Option(None, "--acceptance-mode", help="Acceptance mode: data_api_only or full_processing."),
    ) -> None:
        from dataclasses import replace
        from yt_sub_hub.config import normalize_acceptance_mode, normalize_quality_mode

        pipeline = _pipeline()
        updates = {}
        if quality_mode is not None:
            updates["quality_mode"] = normalize_quality_mode(quality_mode)
        if acceptance_mode is not None:
            updates["acceptance_mode"] = normalize_acceptance_mode(acceptance_mode)
        if updates:
            pipeline.settings = replace(pipeline.settings, **updates)
        existing = pipeline.already_processed(url)
        if existing is not None and not force and not refresh:
            video_id, existing_note = existing
            should_reprocess = typer.confirm(
                (
                    f"Видео {video_id} уже анализировалось: {existing_note}. "
                    "Сделать новый анализ и перезаписать note.md/analysis.json?"
                ),
                default=False,
            )
            if not should_reprocess:
                typer.echo(f"skipped_existing: {video_id} -> {existing_note}")
                return
        try:
            record = pipeline.process(url, refresh=True) if refresh else pipeline.process(url)
        except Exception as exc:
            from yt_sub_hub.ingestion import YouTubeAuthRequiredError

            if isinstance(exc, YouTubeAuthRequiredError):
                typer.echo(str(exc), err=True)
                raise typer.Exit(code=1) from exc
            raise
        suffix = f" warning={record.warning}" if record.warning else ""
        typer.echo(f"{record.status}: {record.video_id} -> {record.note_path}{suffix}")

    @app.command("cache-cleanup")
    def cache_cleanup(
        apply: bool = typer.Option(False, "--apply", help="Delete expired raw metadata/transcript files. Without this option, only preview candidates."),
    ) -> None:
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.library import LibraryStore

        candidates = LibraryStore(get_settings().library_dir).cleanup_expired_raw_cache(apply=apply)
        _echo_payload({
            "mode": "apply" if apply else "dry-run",
            "max_age_days": 30,
            "files": [str(path) for path in candidates],
            "scope": "raw metadata and source transcripts only; notes, derived files, discovery data, and Neo4j are retained",
        }, json_output=True)

    @app.command()
    def status(job_id: str) -> None:
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.library import LibraryStore

        jobs = LibraryStore(get_settings().library_dir).load_jobs()
        record = jobs.get(job_id)
        if record is None:
            raise typer.Exit(code=1)
        typer.echo(record.to_dict())

    @app.command("queue-status")
    def queue_status(stages: bool = typer.Option(False, "--stages", help="Include stage queue counts.")) -> None:
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.library import LibraryStore

        settings = get_settings()
        runtime = _runtime_store()
        _recover_interrupted_jobs(LibraryStore(settings.library_dir), runtime, settings.runtime_stale_after_seconds)
        payload = runtime.queue_status()
        if stages:
            payload = {**payload, **runtime.stage_queue_status()}
        typer.echo(payload)

    @app.command("enqueue-video")
    def enqueue_video(
        url: str,
        quality_mode: str | None = typer.Option(None, "--quality-mode", help="Quality mode: strict, best_effort, or draft."),
        acceptance_mode: str | None = typer.Option(None, "--acceptance-mode", help="Acceptance mode: data_api_only or full_processing."),
    ) -> None:
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.config import normalize_acceptance_mode
        from yt_sub_hub.config import normalize_quality_mode
        from yt_sub_hub.ingestion import parse_youtube_url
        from yt_sub_hub.library import LibraryStore
        from yt_sub_hub.pipeline import VideoProcessingPipeline

        settings = get_settings()
        resolved_quality_mode = normalize_quality_mode(quality_mode or settings.quality_mode)
        resolved_acceptance_mode = normalize_acceptance_mode(acceptance_mode or settings.acceptance_mode)
        parsed = parse_youtube_url(url)
        store = LibraryStore(settings.library_dir)
        from yt_sub_hub.pipeline import job_id_for_url

        try:
            _runtime_store().update_video_job_modes(
                job_id_for_url(url), quality_mode=quality_mode, acceptance_mode=acceptance_mode,
            )
        except KeyError:
            pass
        except RuntimeError as exc:
            raise typer.BadParameter(str(exc)) from exc
        record = VideoProcessingPipeline(store=store, enable_default_graph=False).submit(url)
        runtime_record = _runtime_store().create_video_job(
            job_id=record.job_id,
            video_id=parsed.video_id,
            canonical_url=parsed.canonical_url,
            status=record.status,
            quality_mode=resolved_quality_mode,
            acceptance_mode=resolved_acceptance_mode,
        )
        try:
            runtime_record = _runtime_store().update_video_job_modes(
                runtime_record.id, quality_mode=quality_mode, acceptance_mode=acceptance_mode,
            )
        except RuntimeError as exc:
            raise typer.BadParameter(str(exc)) from exc
        if runtime_record.status not in {"queued", "running"}:
            runtime_record = _runtime_store().resume_video_job(runtime_record.id)
        typer.echo(
            {
                "job_id": runtime_record.id,
                "video_id": runtime_record.video_id,
                "status": runtime_record.status,
                "canonical_url": runtime_record.canonical_url,
                "quality_mode": runtime_record.quality_mode,
                "acceptance_mode": runtime_record.acceptance_mode,
            }
        )

    @app.command("worker-once")
    def worker_once(
        owner: str | None = typer.Option(None, "--owner", help="Runtime lease owner."),
        lease_seconds: float = typer.Option(900, "--lease-seconds", help="Runtime lease duration in seconds."),
    ) -> None:
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.library import LibraryStore
        from yt_sub_hub.worker import StageWorker

        settings = get_settings()
        runtime = _runtime_store()
        store = LibraryStore(settings.library_dir)
        resolved_owner = owner or f"local-worker-{os.getpid()}"
        _recover_interrupted_jobs(store, runtime, settings.runtime_stale_after_seconds)

        def cancel_requested(job_id: str) -> bool:
            try:
                return runtime.load_video_job(job_id).cancel_requested
            except KeyError:
                return False

        def processor(job):
            from dataclasses import replace

            _ensure_legacy_job_for_runtime(store, job)
            pipeline = _worker_pipeline(store, cancel_requested)
            pipeline.settings = replace(
                settings, quality_mode=job.quality_mode, acceptance_mode=job.acceptance_mode,
            )
            return pipeline.process_job(job.id)

        result = StageWorker(
            runtime,
            owner=resolved_owner,
            lease_seconds=lease_seconds,
            library_dir=settings.library_dir,
            target_language=settings.target_language,
            quality_gate_enabled=True,
            whisper_backend=settings.whisper_backend,
        ).process_next(processor)
        if result is None:
            typer.echo("no queued job available")
            return
        typer.echo(result.to_dict())

    @app.command("queue-plan")
    def queue_plan(
        job_id: str,
        whisper_backend: str | None = typer.Option(None, "--whisper-backend", help="Whisper backend for resource lock selection."),
    ) -> None:
        from yt_sub_hub.config import get_settings

        settings = get_settings()
        runtime = _runtime_store()
        try:
            plan = runtime.ensure_stage_plan(
                job_id,
                whisper_backend=whisper_backend or settings.whisper_backend,
                enable_deep_graph_llm=settings.enable_deep_graph_llm,
                include_visual_frame_analysis=settings.enable_visual_analysis,
            )
        except KeyError:
            typer.echo(f"job not found: {job_id}", err=True)
            raise typer.Exit(code=1)
        typer.echo([stage.to_dict() for stage in plan])

    @app.command("job-stages")
    def job_stages(
        job_id: str,
        include_chunks: bool = typer.Option(False, "--include-chunks", help="Include individual chunk stage rows."),
        json_output: bool = typer.Option(False, "--json", help="Emit strict JSON."),
    ) -> None:
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.library import LibraryStore
        from yt_sub_hub.progress import JobNotFoundError, build_job_stage_progress

        settings = get_settings()
        try:
            payload = build_job_stage_progress(
                _runtime_store(),
                LibraryStore(settings.library_dir),
                job_id,
                include_chunks=include_chunks,
            )
        except JobNotFoundError:
            typer.echo(f"job not found: {job_id}", err=True)
            raise typer.Exit(code=1)
        _echo_payload(payload, json_output=json_output)

    @app.command("job-artifacts")
    def job_artifacts(
        job_id: str,
        json_output: bool = typer.Option(False, "--json", help="Emit strict JSON."),
    ) -> None:
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.library import LibraryStore
        from yt_sub_hub.progress import JobNotFoundError, build_job_artifacts_status

        settings = get_settings()
        try:
            payload = build_job_artifacts_status(_runtime_store(), LibraryStore(settings.library_dir), job_id)
        except JobNotFoundError:
            typer.echo(f"job not found: {job_id}", err=True)
            raise typer.Exit(code=1)
        _echo_payload(payload, json_output=json_output)

    @app.command("artifacts")
    def artifacts_command(
        video_id_or_job_id: str,
        json_output: bool = typer.Option(False, "--json", help="Emit strict JSON."),
    ) -> None:
        payload = _artifact_browser_payload(video_id_or_job_id)
        if json_output:
            _echo_payload(payload, json_output=True)
            return
        typer.echo("key\texists\tpath\tapi_url")
        for artifact in payload["artifacts"]:
            typer.echo(
                f"{artifact['key']}\t{artifact['exists']}\t{artifact['path']}\t{artifact['api_url']}"
            )

    @app.command("open-artifact")
    def open_artifact(video_id: str, artifact_key: str) -> None:
        from yt_sub_hub.artifact_browser import ArtifactKeyError, ArtifactMissingError, resolve_artifact
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.library import LibraryStore

        store = LibraryStore(get_settings().library_dir)
        try:
            artifact = resolve_artifact(store, video_id, artifact_key)
        except ArtifactKeyError:
            typer.echo(f"artifact not found: {artifact_key}", err=True)
            raise typer.Exit(code=1)
        if not artifact.path.exists() or not artifact.path.is_file():
            typer.echo(f"artifact missing: {artifact_key}", err=True)
            raise typer.Exit(code=1)
        typer.echo(f"path: {artifact.path}")
        typer.echo(f"api_url: /videos/{video_id}/artifacts/{artifact_key}")
        typer.echo(f"download_url: /videos/{video_id}/artifacts/{artifact_key}/download")

    @app.command("job-quality")
    def job_quality(
        job_id: str,
        json_output: bool = typer.Option(False, "--json", help="Emit strict JSON."),
    ) -> None:
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.library import LibraryStore
        from yt_sub_hub.progress import JobNotFoundError, build_job_quality_status

        settings = get_settings()
        try:
            payload = build_job_quality_status(_runtime_store(), LibraryStore(settings.library_dir), job_id)
        except JobNotFoundError:
            typer.echo(f"job not found: {job_id}", err=True)
            raise typer.Exit(code=1)
        _echo_payload(payload, json_output=json_output)

    @app.command("worker-stage-once")
    def worker_stage_once(
        owner: str | None = typer.Option(None, "--owner", help="Runtime lease owner."),
        lease_seconds: float = typer.Option(900, "--lease-seconds", help="Runtime lease duration in seconds."),
        orchestrator: str = typer.Option("local", "--orchestrator", help="Orchestrator backend: local or beeai."),
    ) -> None:
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.orchestrator import build_orchestrator

        settings = get_settings()
        runtime = _runtime_store()
        runtime.expire_stale_stage_claims()
        for job in runtime.list_video_jobs():
            if job.job_kind == "video_processing" and job.status not in {"completed", "completed_with_warning", "failed", "skipped"}:
                runtime.ensure_stage_plan(
                    job.id,
                    whisper_backend=settings.whisper_backend,
                    enable_deep_graph_llm=settings.enable_deep_graph_llm,
                    include_visual_frame_analysis=settings.enable_visual_analysis,
                )
        resolved_owner = owner or f"local-stage-worker-{os.getpid()}"
        pipeline = StagePipeline(library_dir=settings.library_dir, settings=settings)
        try:
            result = build_orchestrator(
                orchestrator,
                runtime,
                pipeline=pipeline,
                owner_prefix=resolved_owner,
                lease_seconds=lease_seconds,
            ).run_once()
        except RuntimeError as exc:
            typer.echo(str(exc))
            raise typer.Exit(code=1) from exc
        if result is None:
            if orchestrator == "beeai":
                typer.echo(f"orchestrator={orchestrator}")
            typer.echo("no runnable stage available")
            return
        if orchestrator == "beeai":
            typer.echo(f"orchestrator={orchestrator}")
        typer.echo(result.to_dict())

    @app.command("worker-stage-loop")
    def worker_stage_loop(
        concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of local worker threads."),
        owner: str | None = typer.Option(None, "--owner", help="Runtime lease owner prefix."),
        lease_seconds: float = typer.Option(900, "--lease-seconds", help="Runtime lease duration in seconds."),
        orchestrator: str = typer.Option("local", "--orchestrator", help="Orchestrator backend: local or beeai."),
    ) -> None:
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.orchestrator import build_orchestrator

        settings = get_settings()
        runtime = _runtime_store()
        for job in runtime.list_video_jobs():
            if job.job_kind == "video_processing" and job.status not in {"completed", "completed_with_warning", "failed", "skipped"}:
                runtime.ensure_stage_plan(
                    job.id,
                    whisper_backend=settings.whisper_backend,
                    enable_deep_graph_llm=settings.enable_deep_graph_llm,
                    include_visual_frame_analysis=settings.enable_visual_analysis,
                )
        pipeline = StagePipeline(library_dir=settings.library_dir, settings=settings)
        try:
            results = build_orchestrator(
                orchestrator,
                runtime,
                pipeline=pipeline,
                owner_prefix=owner or f"local-stage-worker-{os.getpid()}",
                lease_seconds=lease_seconds,
            ).run_loop(concurrency=concurrency)
        except RuntimeError as exc:
            typer.echo(str(exc))
            raise typer.Exit(code=1) from exc
        if orchestrator == "beeai":
            typer.echo(f"orchestrator={orchestrator}")
        typer.echo({"processed_stages": len(results), "queue": runtime.queue_status(), **runtime.stage_queue_status()})

    @app.command("cancel")
    def cancel_job(job_id: str) -> None:
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.library import LibraryStore
        from yt_sub_hub.orchestration import sync_legacy_job_from_runtime

        settings = get_settings()
        try:
            record = _runtime_store().request_cancel(job_id)
        except KeyError:
            typer.echo(f"job not found: {job_id}", err=True)
            raise typer.Exit(code=1)
        sync_legacy_job_from_runtime(LibraryStore(settings.library_dir), record)
        typer.echo({"status": record.status, "job_id": record.id, "cancel_requested": record.cancel_requested})

    @app.command("resume")
    def resume(job_id: str) -> None:
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.library import LibraryStore
        from yt_sub_hub.orchestration import sync_legacy_requeued_job

        settings = get_settings()
        try:
            record = _runtime_store().resume_video_job(job_id)
        except KeyError:
            typer.echo(f"job not found: {job_id}", err=True)
            raise typer.Exit(code=1)
        sync_legacy_requeued_job(LibraryStore(settings.library_dir), record)
        typer.echo(record.to_dict())

    @app.command("retry-failed")
    def retry_failed(job_id: str, from_stage: str | None = typer.Option(None, "--from-stage", help="Reset from this stage instead of the first failed stage.")) -> None:
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.library import LibraryStore
        from yt_sub_hub.orchestration import requeue_failed_job

        settings = get_settings()
        try:
            runtime = _runtime_store()
            record = requeue_failed_job(runtime, LibraryStore(settings.library_dir), job_id, from_stage=from_stage)
        except ValueError:
            typer.echo(f"job is not failed: {job_id}", err=True)
            raise typer.Exit(code=1)
        except KeyError:
            typer.echo(f"job not found: {job_id}", err=True)
            raise typer.Exit(code=1)
        typer.echo(record.to_dict())

    @app.command("recovery")
    def recovery_command(job_id: str, json_output: bool = typer.Option(False, "--json", help="Emit strict JSON.")) -> None:
        runtime = _runtime_store()
        try:
            payload = _recovery_dashboard_payload(runtime, job_id)
        except KeyError:
            typer.echo(f"job not found: {job_id}", err=True)
            raise typer.Exit(code=1)
        _echo_payload(payload, json_output=json_output)

    @app.command("retry-stage")
    def retry_stage_command(
        job_id: str,
        stage_name: str,
        json_output: bool = typer.Option(False, "--json", help="Emit strict JSON."),
    ) -> None:
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.library import LibraryStore
        from yt_sub_hub.orchestration import sync_legacy_requeued_job

        runtime = _runtime_store()
        try:
            record = runtime.retry_stage(job_id, stage_name)
            sync_legacy_requeued_job(LibraryStore(get_settings().library_dir), record)
            payload = _recovery_dashboard_payload(runtime, job_id)
        except (KeyError, ValueError) as exc:
            _echo_recovery_action_error(exc, job_id)
        _echo_payload(payload, json_output=json_output)

    @app.command("retry-failed-chunks")
    def retry_failed_chunks_command(
        job_id: str,
        group: str = typer.Option(..., "--group", help="Chunk group: transcript_translation or analysis."),
        json_output: bool = typer.Option(False, "--json", help="Emit strict JSON."),
    ) -> None:
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.library import LibraryStore
        from yt_sub_hub.orchestration import sync_legacy_requeued_job

        runtime = _runtime_store()
        try:
            record = runtime.retry_failed_chunks(job_id, group)
            sync_legacy_requeued_job(LibraryStore(get_settings().library_dir), record)
            payload = _recovery_dashboard_payload(runtime, job_id)
        except (KeyError, ValueError) as exc:
            _echo_recovery_action_error(exc, job_id)
        _echo_payload(payload, json_output=json_output)

    @app.command("reset-downstream")
    def reset_downstream_command(
        job_id: str,
        from_stage: str,
        json_output: bool = typer.Option(False, "--json", help="Emit strict JSON."),
    ) -> None:
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.library import LibraryStore
        from yt_sub_hub.orchestration import sync_legacy_requeued_job

        runtime = _runtime_store()
        try:
            record = runtime.reset_downstream(job_id, from_stage)
            sync_legacy_requeued_job(LibraryStore(get_settings().library_dir), record)
            payload = _recovery_dashboard_payload(runtime, job_id)
        except (KeyError, ValueError) as exc:
            _echo_recovery_action_error(exc, job_id)
        _echo_payload(payload, json_output=json_output)

    @app.command("cancel-job")
    def cancel_job_command(job_id: str, json_output: bool = typer.Option(False, "--json", help="Emit strict JSON.")) -> None:
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.library import LibraryStore
        from yt_sub_hub.orchestration import sync_legacy_job_from_runtime

        runtime = _runtime_store()
        try:
            record = runtime.cancel_job(job_id)
            sync_legacy_job_from_runtime(LibraryStore(get_settings().library_dir), record)
            payload = _recovery_dashboard_payload(runtime, job_id)
        except (KeyError, ValueError) as exc:
            _echo_recovery_action_error(exc, job_id)
        _echo_payload(payload, json_output=json_output)

    @app.command("resume-job")
    def resume_job_command(job_id: str, json_output: bool = typer.Option(False, "--json", help="Emit strict JSON.")) -> None:
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.library import LibraryStore
        from yt_sub_hub.orchestration import sync_legacy_requeued_job

        runtime = _runtime_store()
        try:
            record = runtime.resume_job(job_id)
            sync_legacy_requeued_job(LibraryStore(get_settings().library_dir), record)
            payload = _recovery_dashboard_payload(runtime, job_id)
        except (KeyError, ValueError) as exc:
            _echo_recovery_action_error(exc, job_id)
        _echo_payload(payload, json_output=json_output)

    @app.command("review-ru-artifacts")
    def review_ru_artifacts_command(video_id: str) -> None:
        from yt_sub_hub.artifact_review import review_ru_artifacts
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.library import LibraryStore

        settings = get_settings()
        report = review_ru_artifacts(video_id, LibraryStore(settings.library_dir), settings)
        typer.echo({"video_id": video_id, "status": report.status})
        for issue in report.issues:
            typer.echo(f"{issue.severity}: {issue.code}: {issue.artifact}: {issue.message}")
        if report.status == "failed":
            raise typer.Exit(code=1)

    @app.command("research-video")
    def research_video(video_id: str, max_sources: int = typer.Option(10, "--max-sources", help="Maximum normalized sources to fetch.")) -> None:
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.library import LibraryStore
        from yt_sub_hub.research_enrichment import run_video_research

        settings = get_settings()
        try:
            payload = run_video_research(LibraryStore(settings.library_dir), video_id, max_sources=max_sources)
        except Exception as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        typer.echo({"video_id": video_id, "status": payload["status"]["status"]})

    @app.command("research-status")
    def research_status(video_id: str, json_output: bool = typer.Option(False, "--json", help="Emit strict JSON.")) -> None:
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.library import LibraryStore
        from yt_sub_hub.research_enrichment import build_research_payload

        payload = build_research_payload(LibraryStore(get_settings().library_dir), video_id)
        _echo_payload(payload["status"] if json_output else payload, json_output=json_output)

    @app.command("research-channel")
    def research_channel(channel_id: str, json_output: bool = typer.Option(False, "--json", help="Emit strict JSON.")) -> None:
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.library import LibraryStore
        from yt_sub_hub.research_enrichment import build_channel_research_summary

        try:
            payload = build_channel_research_summary(LibraryStore(get_settings().library_dir), channel_id)
        except FileNotFoundError as exc:
            typer.echo(f"channel not found: {channel_id}", err=True)
            raise typer.Exit(code=1) from exc
        _echo_payload(payload, json_output=json_output)

    @app.command("review-research")
    def review_research(video_id: str, json_output: bool = typer.Option(False, "--json", help="Emit strict JSON.")) -> None:
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.library import LibraryStore
        from yt_sub_hub.research_enrichment import load_research_review

        try:
            review = load_research_review(LibraryStore(get_settings().library_dir), video_id)
        except FileNotFoundError as exc:
            typer.echo(f"research review not found: {video_id}", err=True)
            raise typer.Exit(code=1) from exc
        _echo_payload(review, json_output=json_output)
        if review.get("status") == "failed":
            raise typer.Exit(code=1)

    @app.command("research-sync-neo4j")
    def research_sync_neo4j(video_id: str, json_output: bool = typer.Option(False, "--json", help="Emit strict JSON.")) -> None:
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.library import LibraryStore
        from yt_sub_hub.research_enrichment import ResearchValidationError, sync_research_graph_to_neo4j

        settings = get_settings()
        try:
            payload = sync_research_graph_to_neo4j(LibraryStore(settings.library_dir), video_id, settings=settings)
        except FileNotFoundError as exc:
            typer.echo(f"research graph or review not found: {video_id}", err=True)
            raise typer.Exit(code=1) from exc
        except ResearchValidationError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        except Exception as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        _echo_payload(payload, json_output=json_output)
        if payload.get("missing") or payload.get("extra"):
            raise typer.Exit(code=1)

    @app.command("discover-youtube")
    def discover_youtube(
        video_id: str,
        limit: int = typer.Option(20, "--limit", help="Maximum normalized video candidates."),
        max_quota_units: int = typer.Option(900, "--max-quota-units", help="Maximum general Data API units for this discovery run; search has a separate 100-call budget."),
        json_output: bool = typer.Option(False, "--json", help="Emit strict JSON."),
    ) -> None:
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.library import LibraryStore
        from yt_sub_hub.youtube_discovery import DiscoveryValidationError, run_youtube_discovery

        settings = get_settings()
        try:
            payload = run_youtube_discovery(
                LibraryStore(settings.library_dir),
                video_id,
                settings=settings,
                limit=limit,
                max_quota_units=max_quota_units,
            )
        except DiscoveryValidationError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        except Exception as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        _echo_payload(payload if json_output else {"video_id": video_id, "status": payload["status"]["status"]}, json_output=json_output)

    @app.command("queue-discovery")
    def queue_discovery(
        video_id: str,
        limit: int = typer.Option(20, "--limit", help="Maximum normalized video candidates."),
        max_quota_units: int = typer.Option(900, "--max-quota-units", help="Maximum general Data API units for this discovery run; search has a separate 100-call budget."),
        force: bool = typer.Option(False, "--force", help="Requeue the existing discovery job for this source video."),
        json_output: bool = typer.Option(False, "--json", help="Emit strict JSON."),
    ) -> None:
        runtime = _runtime_store()
        job = runtime.queue_youtube_discovery(video_id, limit=limit, max_quota_units=max_quota_units, force=force)
        stages = runtime.list_stage_plan(job.id)
        payload = {
            "job_id": job.id,
            "job_kind": job.job_kind,
            "video_id": job.video_id,
            "status": job.status,
            "stage_count": len(stages),
            "stages": [stage.stage_name for stage in stages],
        }
        _echo_payload(payload, json_output=json_output)

    @app.command("discovery-status")
    def discovery_status(
        video_id: str,
        stages: bool = typer.Option(False, "--stages", help="Include staged discovery progress if queued."),
        json_output: bool = typer.Option(False, "--json", help="Emit strict JSON."),
    ) -> None:
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.library import LibraryStore
        from yt_sub_hub.youtube_discovery import build_discovery_payload

        settings = get_settings()
        runtime = _runtime_store()
        store = LibraryStore(settings.library_dir)
        payload = build_discovery_payload(store, video_id)
        if stages:
            from yt_sub_hub.progress import build_job_stage_progress

            job = runtime.latest_youtube_discovery_job_for_video(video_id)
            if job is not None:
                payload["stage_progress"] = build_job_stage_progress(runtime, store, job.id, include_chunks=True)
        _echo_payload(payload, json_output=json_output)

    @app.command("review-discovery")
    def review_discovery(video_id: str, json_output: bool = typer.Option(False, "--json", help="Emit strict JSON.")) -> None:
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.library import LibraryStore
        from yt_sub_hub.youtube_discovery import load_discovery_review

        try:
            review = load_discovery_review(LibraryStore(get_settings().library_dir), video_id)
        except FileNotFoundError as exc:
            typer.echo(f"discovery review not found: {video_id}", err=True)
            raise typer.Exit(code=1) from exc
        _echo_payload(review, json_output=json_output)
        if review.get("status") == "failed":
            raise typer.Exit(code=1)

    @app.command("enqueue-discovered-videos")
    def enqueue_discovered_videos_command(
        video_id: str,
        min_score: float = typer.Option(0.75, "--min-score", help="Minimum discovery score to enqueue."),
        limit: int = typer.Option(5, "--limit", help="Maximum accepted candidates to enqueue."),
        quality_mode: str | None = typer.Option(None, "--quality-mode", help="Quality mode: strict, best_effort, or draft."),
        acceptance_mode: str | None = typer.Option(None, "--acceptance-mode", help="Acceptance mode: data_api_only or full_processing."),
        json_output: bool = typer.Option(False, "--json", help="Emit strict JSON."),
    ) -> None:
        from yt_sub_hub.config import get_settings, normalize_acceptance_mode, normalize_quality_mode
        from yt_sub_hub.library import LibraryStore
        from yt_sub_hub.youtube_discovery import DiscoveryValidationError, enqueue_discovered_videos

        settings = get_settings()
        try:
            payload = enqueue_discovered_videos(
                _runtime_store(),
                LibraryStore(settings.library_dir),
                video_id,
                min_score=min_score,
                limit=limit,
                quality_mode=normalize_quality_mode(quality_mode or settings.quality_mode),
                acceptance_mode=normalize_acceptance_mode(acceptance_mode or settings.acceptance_mode),
            )
        except (DiscoveryValidationError, FileNotFoundError) as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        _echo_payload(payload, json_output=json_output)

    @app.command("enqueue-discovered-channel")
    def enqueue_discovered_channel_command(
        video_id: str,
        channel_id: str = typer.Option(..., "--channel-id", help="Accepted discovery channel candidate id."),
        limit: int | None = typer.Option(None, "--limit", help="Maximum channel videos to enqueue."),
        json_output: bool = typer.Option(False, "--json", help="Emit strict JSON."),
    ) -> None:
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.library import LibraryStore
        from yt_sub_hub.youtube_discovery import DiscoveryValidationError, enqueue_discovered_channel

        settings = get_settings()
        try:
            payload = enqueue_discovered_channel(
                _runtime_store(),
                LibraryStore(settings.library_dir),
                video_id,
                channel_id=channel_id,
                limit=limit,
                settings=settings,
            )
        except (DiscoveryValidationError, FileNotFoundError) as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        _echo_payload(payload, json_output=json_output)

    @app.command("discovery-sync-neo4j")
    def discovery_sync_neo4j(video_id: str, json_output: bool = typer.Option(False, "--json", help="Emit strict JSON.")) -> None:
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.library import LibraryStore
        from yt_sub_hub.youtube_discovery import DiscoveryValidationError, sync_discovery_graph_to_neo4j

        settings = get_settings()
        try:
            payload = sync_discovery_graph_to_neo4j(LibraryStore(settings.library_dir), video_id, settings=settings)
        except (DiscoveryValidationError, FileNotFoundError) as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        except Exception as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        _echo_payload(payload, json_output=json_output)
        if payload.get("missing") or payload.get("extra"):
            raise typer.Exit(code=1)

    @app.command("process-channel")
    def process_channel(
        channel_url: str,
        limit: int | None = typer.Option(None, "--limit", help="Maximum number of videos to enqueue."),
    ) -> None:
        _enqueue_channel_command(channel_url, limit=limit, quality_mode=None, acceptance_mode=None, force=False)

    @app.command("enqueue-channel")
    def enqueue_channel(
        channel_url_or_id: str,
        limit: int | None = typer.Option(None, "--limit", help="Maximum number of videos to enqueue."),
        quality_mode: str | None = typer.Option(None, "--quality-mode", help="Quality mode: strict, best_effort, or draft."),
        acceptance_mode: str | None = typer.Option(None, "--acceptance-mode", help="Acceptance mode: data_api_only or full_processing."),
        force: bool = typer.Option(False, "--force", help="Requeue failed/skipped child jobs without overwriting ready artifacts."),
    ) -> None:
        _enqueue_channel_command(channel_url_or_id, limit=limit, quality_mode=quality_mode, acceptance_mode=acceptance_mode, force=force)

    def _enqueue_channel_command(
        channel_url_or_id: str,
        *,
        limit: int | None,
        quality_mode: str | None,
        acceptance_mode: str | None,
        force: bool,
    ) -> None:
        from yt_sub_hub.channel import ChannelDataApiError, enqueue_channel_videos, fetch_channel_listing
        from yt_sub_hub.config import get_settings, normalize_acceptance_mode, normalize_quality_mode
        from yt_sub_hub.library import LibraryStore

        settings = get_settings()
        resolved_quality_mode = normalize_quality_mode(quality_mode or settings.quality_mode)
        resolved_acceptance_mode = normalize_acceptance_mode(acceptance_mode or settings.acceptance_mode)
        try:
            listing = fetch_channel_listing(channel_url_or_id, limit=limit, settings=settings)
        except ChannelDataApiError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        channel_job, video_jobs = enqueue_channel_videos(
            _runtime_store(),
            LibraryStore(settings.library_dir),
            channel_url=listing.channel_url,
            channel_id=listing.channel_id,
            uploads_playlist_id=listing.uploads_playlist_id,
            videos=listing.videos,
            limit=limit,
            quality_mode=resolved_quality_mode,
            acceptance_mode=resolved_acceptance_mode,
            force=force,
        )
        typer.echo(
            {
                "channel_job_id": channel_job.id,
                "channel_id": channel_job.channel_id,
                "status": channel_job.status,
                "video_jobs": len(video_jobs),
                "quality_mode": channel_job.quality_mode,
                "acceptance_mode": channel_job.acceptance_mode,
            }
        )

    @app.command("channel-status")
    def channel_status(
        channel_job_id: str,
        json_output: bool = typer.Option(False, "--json", help="Emit strict JSON."),
    ) -> None:
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.library import LibraryStore
        from yt_sub_hub.progress import ChannelJobNotFoundError, build_channel_job_progress

        runtime = _runtime_store()
        try:
            payload = build_channel_job_progress(runtime, LibraryStore(get_settings().library_dir), channel_job_id)
        except ChannelJobNotFoundError:
            typer.echo(f"channel job not found: {channel_job_id}", err=True)
            raise typer.Exit(code=1)
        _echo_payload(payload, json_output=json_output)

    @app.command("cancel-channel")
    def cancel_channel(channel_job_id: str, json_output: bool = typer.Option(False, "--json", help="Emit strict JSON.")) -> None:
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.library import LibraryStore
        from yt_sub_hub.progress import ChannelJobNotFoundError, build_channel_job_progress

        runtime = _runtime_store()
        try:
            runtime.cancel_channel_job(channel_job_id)
            payload = build_channel_job_progress(runtime, LibraryStore(get_settings().library_dir), channel_job_id)
        except (KeyError, ChannelJobNotFoundError):
            typer.echo(f"channel job not found: {channel_job_id}", err=True)
            raise typer.Exit(code=1)
        _echo_payload(payload, json_output=json_output)

    @app.command("retry-channel-failed")
    def retry_channel_failed(channel_job_id: str, json_output: bool = typer.Option(False, "--json", help="Emit strict JSON.")) -> None:
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.library import LibraryStore
        from yt_sub_hub.progress import ChannelJobNotFoundError, build_channel_job_progress

        runtime = _runtime_store()
        try:
            runtime.retry_channel_failed(channel_job_id)
            payload = build_channel_job_progress(runtime, LibraryStore(get_settings().library_dir), channel_job_id)
        except (KeyError, ChannelJobNotFoundError):
            typer.echo(f"channel job not found: {channel_job_id}", err=True)
            raise typer.Exit(code=1)
        _echo_payload(payload, json_output=json_output)

    @app.command()
    def doctor(strict: bool = False) -> None:
        from yt_sub_hub.doctor import format_report, run_doctor

        report = run_doctor(strict=strict)
        typer.echo(format_report(report))
        if not report.ok:
            raise typer.Exit(code=1)

    @app.command("captions-probe")
    def captions_probe_command(
        target: str,
        limit: int = typer.Option(3, "--limit", help="Maximum channel videos to probe."),
        json_output: bool = typer.Option(False, "--json", help="Emit strict JSON."),
    ) -> None:
        from yt_sub_hub.captions_probe import CaptionsProbeError, format_captions_probe_human, probe_captions
        from yt_sub_hub.config import get_settings

        try:
            payload = probe_captions(target, limit=limit, settings=get_settings())
        except CaptionsProbeError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        if json_output:
            _echo_payload(payload, json_output=True)
            return
        typer.echo(format_captions_probe_human(payload))

    @app.command("benchmark-whisper")
    def benchmark_whisper(
        video_id_or_audio_path: str,
        seconds: int = typer.Option(30, "--seconds", help="Audio duration to benchmark from the beginning."),
    ) -> None:
        from yt_sub_hub.config import get_settings

        try:
            result = run_whisper_benchmark(video_id_or_audio_path, settings=get_settings(), seconds=seconds)
        except Exception as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        for key in ["backend", "model", "device", "elapsed_seconds", "language", "segments", "sample"]:
            typer.echo(f"{key}: {result.get(key, '')}")

    @app.command("benchmark-lm-studio-translation")
    def benchmark_lm_studio_translation_command(
        video_id: str,
        concurrency: str = typer.Option("1,2,4", "--concurrency", help="Comma-separated concurrency levels, e.g. 1,2,4."),
        sample_chunks: int | None = typer.Option(None, "--sample-chunks", help="Number of translation chunks to sample."),
        timeout_seconds: float | None = typer.Option(None, "--timeout-seconds", help="Per-chunk LM Studio timeout."),
        run_timeout_seconds: float | None = typer.Option(None, "--run-timeout-seconds", help="Maximum wall time for each benchmark run."),
        continue_after_timeout: bool = typer.Option(
            False,
            "--continue-after-timeout/--stop-after-timeout",
            help="Continue to the next concurrency after a benchmark run timeout.",
        ),
        output: Path | None = typer.Option(None, "--output", help="Benchmark JSON report path."),
        chunk_selection: str = typer.Option("evenly-spaced", "--chunk-selection", help="Chunk selection: first or evenly-spaced."),
        json_output: bool = typer.Option(False, "--json", help="Print the full JSON report."),
    ) -> None:
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.library import LibraryStore

        settings = get_settings()
        try:
            options = BenchmarkOptions(
                video_id=video_id,
                concurrencies=_parse_concurrency_values(concurrency),
                sample_chunks=sample_chunks if sample_chunks is not None else settings.lm_studio_benchmark_sample_chunks,
                timeout_seconds=timeout_seconds if timeout_seconds is not None else settings.lm_studio_benchmark_timeout_seconds,
                run_timeout_seconds=(
                    run_timeout_seconds if run_timeout_seconds is not None else settings.lm_studio_benchmark_run_timeout_seconds
                ),
                continue_after_timeout=continue_after_timeout,
                output=output,
                chunk_selection=chunk_selection,
            )
            report = benchmark_lm_studio_translation(LibraryStore(settings.library_dir), settings, options)
        except BenchmarkReportError as exc:
            report = exc.report
            report_path = report.get("report_path")
            if not report_path or not Path(str(report_path)).exists():
                typer.echo(str(exc), err=True)
                raise typer.Exit(code=1) from exc
        except Exception as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        if json_output:
            typer.echo(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
            return
        for run in report.get("runs", []):
            latency = run.get("latency_seconds") or {}
            throughput = run.get("throughput") or {}
            typer.echo(
                " ".join(
                    [
                        f"concurrency={run.get('concurrency')}",
                        f"success={run.get('chunks_succeeded')}/{run.get('chunks_total')}",
                        f"timeout={_count_from_rate(run.get('timeout_rate'), run.get('chunks_total'))}",
                        f"error={_count_from_rate(run.get('error_rate'), run.get('chunks_total'))}",
                        f"parse_error={_count_from_rate(run.get('parse_error_rate'), run.get('chunks_total'))}",
                        f"p50={latency.get('p50')}",
                        f"p95={latency.get('p95')}",
                        f"chunks/min={throughput.get('chunks_per_minute')}",
                    ]
                )
            )
        recommendation = report.get("recommendation") or {}
        typer.echo(f"recommendation safe_default={recommendation.get('safe_default')} reason={recommendation.get('reason')}")
        typer.echo(f"report: {report.get('report_path')}")

    @app.command("setup-whisper-cpp")
    def setup_whisper_cpp(
        model: str | None = typer.Option(None, "--model", help="whisper.cpp model name, e.g. base, small, base.en."),
        download_model: bool = typer.Option(False, "--download-model", help="Download the managed ggml model file."),
    ) -> None:
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.whisper_cpp_setup import managed_whisper_cpp_model_path, prepare_whisper_cpp

        settings = get_settings()
        status = prepare_whisper_cpp(settings, model_name=model, download_model=download_model)
        target_model = managed_whisper_cpp_model_path(settings, model or settings.whisper_model)
        typer.echo(f"binary: {status.binary_path or 'missing'}")
        typer.echo(f"model: {status.model_path or target_model}")
        typer.echo(f"model_ready: {status.model_ready}")
        typer.echo(f"model_downloaded: {status.model_downloaded}")
        if status.message:
            typer.echo(status.message)
        typer.echo("env:")
        if status.binary_path:
            typer.echo(f"  YT_SUB_HUB_WHISPER_CPP_BIN={status.binary_path}")
        typer.echo(f"  YT_SUB_HUB_WHISPER_CPP_MODEL={status.model_path or target_model}")
        if not status.binary_path or not status.model_ready:
            raise typer.Exit(code=1)

    @app.command("rebuild-graph")
    def rebuild_graph(
        video_id: str | None = typer.Option(None, "--video-id", help="Rebuild only one video."),
        dry_run: bool = typer.Option(False, "--dry-run", help="Build local graph extraction without writing to Neo4j."),
        clear: bool = typer.Option(False, "--clear", help="Clear Neo4j before rebuilding."),
        min_confidence: float = typer.Option(0.0, "--min-confidence", help="Minimum relationship confidence to persist."),
    ) -> None:
        from yt_sub_hub.config import get_settings

        settings = get_settings()
        graph_store = None
        if not dry_run:
            from yt_sub_hub.graph import Neo4jGraphStore

            graph_store = Neo4jGraphStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
        try:
            count = _pipeline().rebuild_graph(
                graph_store,
                video_id=video_id,
                dry_run=dry_run,
                clear=clear,
                min_confidence=min_confidence,
            )
        finally:
            if graph_store is not None:
                graph_store.close()
        suffix = " (dry run)" if dry_run else ""
        typer.echo(f"rebuilt graph from {count} videos{suffix}")

    @app.command("analyze-frame")
    def analyze_frame(
        video_id: str,
        at: float = typer.Option(..., "--at", help="Timestamp in seconds."),
        question: str | None = typer.Option(None, "--question", help="Optional question for the vision model."),
        force: bool = typer.Option(False, "--force", help="Regenerate even if cached artifacts exist."),
    ) -> None:
        from yt_sub_hub.visual import VisualAnalysisError

        service = _visual_service()
        try:
            result = service.analyze_frame(video_id, at, question=question, force=force)
        except VisualAnalysisError as exc:
            typer.echo(str(exc))
            raise typer.Exit(code=1)
        seconds = int(at) if float(at).is_integer() else at
        typer.echo(f"visual_frame_analyzed: {video_id} @ {seconds}s")
        typer.echo(f"screen_type: {result.get('screen_type', 'unknown')}")
        typer.echo(f"summary: {result.get('summary', '')}")
        typer.echo(f"saved: {service.normalized_path(video_id, at)}")
        typer.echo(f"note: {service.visual_note_path(video_id)}")

    @app.command("inspect-graph")
    def inspect_graph(video_id: str, limit: int = typer.Option(10, "--limit", help="Number of node and relationship samples to show.")) -> None:
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.library import LibraryStore
        from collections import Counter

        store = LibraryStore(get_settings().library_dir)
        extraction = store.load_graph_extraction(video_id)
        if extraction is None:
            typer.echo(f"graph extraction not found for {video_id}")
            raise typer.Exit(code=1)
        quality = extraction.get("quality_report", {})
        chunks = extraction.get("chunks", [])
        nodes = extraction.get("nodes", [])
        relationships = extraction.get("relationships", [])
        node_counts = quality.get("node_count_by_label") or dict(Counter(node.get("label") for node in nodes))
        relationship_counts = quality.get("relationship_count_by_type") or dict(Counter(rel.get("type") for rel in relationships))
        llm_relationships = [
            relationship
            for relationship in relationships
            if relationship.get("audit_status", "accepted") == "accepted"
            and (
                relationship.get("extraction_method") == "llm_deep_graph"
                or relationship.get("reliability") == "llm_extracted"
            )
        ]
        typer.echo(f"{extraction.get('schema_version')}: {video_id}")
        typer.echo(f"chunks: {len(chunks)}")
        typer.echo("nodes:")
        for label, count in sorted(node_counts.items()):
            typer.echo(f"  {label}: {count}")
        typer.echo("relationships:")
        for rel_type, count in sorted(relationship_counts.items()):
            typer.echo(f"  {rel_type}: {count}")
        typer.echo(f"llm_relationships: {len(llm_relationships)}")
        rejected = quality.get("rejected_relationships", [])
        low_confidence = quality.get("low_confidence_extractions", [])
        duplicates = quality.get("duplicate_concept_candidates", [])
        typer.echo(f"rejected_relationships: {len(rejected)}")
        typer.echo(f"low_confidence_extractions: {len(low_confidence)}")
        typer.echo(f"duplicate_concept_candidates: {len(duplicates)}")
        rejection_reasons = Counter(item.get("reason", "unknown") for item in rejected + low_confidence)
        typer.echo("rejection reasons:")
        for reason, count in sorted(rejection_reasons.items()):
            typer.echo(f"  {reason}: {count}")
        typer.echo("sample chunks:")
        for chunk in chunks[:limit]:
            chunk_id = chunk.get("chunk_id") or chunk.get("id") or ""
            start = chunk.get("start")
            end = chunk.get("end")
            time_range = f" {start}-{end}" if start is not None or end is not None else ""
            typer.echo(f"  {chunk_id}{time_range}")
        typer.echo("sample nodes:")
        for node in nodes[:limit]:
            name = node.get("name") or node.get("title") or node.get("text") or ""
            typer.echo(f"  {node.get('label')} {node.get('id')} {name}")
        typer.echo("sample relationships:")
        for relationship in relationships[:limit]:
            typer.echo(
                f"  {relationship.get('type')} {relationship.get('from')} -> {relationship.get('to')} "
                f"confidence={relationship.get('confidence')}"
            )
        typer.echo("sample llm relationships:")
        for relationship in llm_relationships[:limit]:
            typer.echo(
                f"  {relationship.get('type')} {relationship.get('from')} -> {relationship.get('to')} "
                f"confidence={relationship.get('confidence')}"
            )

    @app.command("graph-export")
    def graph_export(
        format: str = typer.Option("json", "--format", help="Export format: json or graphml."),
        output: Path | None = typer.Option(None, "--output", help="Output file path."),
        video_id: str | None = typer.Option(None, "--video-id", help="Export only one video's graph extraction."),
    ) -> None:
        from yt_sub_hub.config import get_settings
        from yt_sub_hub.graph_export import export_graph_json, export_graphml
        from yt_sub_hub.library import LibraryStore

        store = LibraryStore(get_settings().library_dir)
        try:
            if format == "json":
                path = export_graph_json(store, output, video_id=video_id)
            elif format == "graphml":
                path = export_graphml(store, output, video_id=video_id)
            else:
                typer.echo("unsupported graph export format")
                raise typer.Exit(code=1)
        except FileNotFoundError as exc:
            typer.echo(str(exc))
            raise typer.Exit(code=1)
        typer.echo(f"exported graph: {path}")
else:
    app = None


def note_path(video_id: str, root: str = "library") -> Path:
    return Path(root) / "videos" / video_id / "note.md"


def _parse_concurrency_values(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--concurrency must be a comma-separated list of integers") from exc
    if not values:
        raise ValueError("--concurrency must contain at least one integer")
    return [max(1, item) for item in values]


def _count_from_rate(rate: object, total: object) -> int:
    try:
        return round(float(rate or 0) * int(total or 0))
    except (TypeError, ValueError):
        return 0
