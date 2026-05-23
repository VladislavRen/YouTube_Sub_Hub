from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import threading
import time
from typing import Any

from yt_sub_hub.config import normalize_quality_mode
from yt_sub_hub.library import LibraryStore
from yt_sub_hub.orchestration import sync_legacy_job_from_runtime
from yt_sub_hub.quality import run_quality_gate
from yt_sub_hub.runtime import RuntimeStore, VideoJobRecord
from yt_sub_hub.stage_pipeline import StagePipeline, StageResult
from yt_sub_hub.stages import retry_delay_seconds, stage_by_name
from yt_sub_hub.transcripts import SKIPPED_REQUIRES_MEDIA_REASON
from yt_sub_hub.whisper_backends import whisper_resource_lock


class StageWorker:
    def __init__(
        self,
        runtime: RuntimeStore,
        *,
        owner: str,
        lease_seconds: float = 300,
        library_dir: Path | str | None = None,
        target_language: str = "ru",
        quality_gate_enabled: bool = True,
        whisper_backend: str = "openai-whisper-cpu",
    ) -> None:
        self.runtime = runtime
        self.owner = owner
        self.lease_seconds = lease_seconds
        self.library_dir = Path(library_dir) if library_dir is not None else runtime.db_path.parent
        self.target_language = target_language
        self.quality_gate_enabled = quality_gate_enabled
        self.whisper_backend = whisper_backend

    def process_next_stage(self, pipeline: StagePipeline) -> VideoJobRecord | None:
        self.runtime.expire_stale_stage_claims()
        claim = self.runtime.claim_next_stage(owner=self.owner, lease_seconds=self.lease_seconds)
        if claim is None:
            for job in self.runtime.list_video_jobs():
                if job.cancel_requested and job.status not in {"completed", "completed_with_warning", "failed", "skipped"}:
                    return self._sync_runtime_result(self.runtime.skip_video_job(job.id, warning="cancel requested"))
            return None
        heartbeat_stop, heartbeat_thread = self._start_stage_heartbeat(claim.stage.id)
        try:
            current_job = self.runtime.load_video_job(claim.job.id)
            if current_job.cancel_requested:
                self.runtime.skip_stage(claim.stage.id, warning="cancel requested")
                return self._sync_runtime_result(self.runtime.skip_video_job(claim.job.id, warning="cancel requested"))
            if hasattr(pipeline, "context"):
                context = pipeline.context(claim.job, claim.stage, runtime=self.runtime)
            else:
                from yt_sub_hub.stage_pipeline import StageContext
                from yt_sub_hub.config import get_settings

                settings = get_settings()
                context = StageContext(
                    job=claim.job,
                    stage=claim.stage,
                    settings=settings,
                    store=LibraryStore(self.library_dir),
                    pipeline=None,
                    runtime=self.runtime,
                )
            result = pipeline.run(context)
            if (
                result.status == "skipped"
                and claim.stage.stage_name == "transcript_source"
                and result.warning == SKIPPED_REQUIRES_MEDIA_REASON
            ):
                self.runtime.complete_stage_run(claim.stage_run.id, warning=result.warning)
                return self._sync_runtime_result(self.runtime.mark_video_job_skipped_requires_media(claim.job.id))
            if result.status == "skipped":
                self.runtime.complete_stage_run(claim.stage_run.id, warning=result.warning or "skipped")
                self.runtime.skip_stage(claim.stage.id, warning=result.warning)
            elif result.status == "failed":
                self.runtime.fail_stage(claim.stage_run.id, error=result.error or "stage failed")
            else:
                self.runtime.complete_stage(
                    claim.stage_run.id,
                    warning=result.warning,
                    skip_stage_names=result.skip_stages,
                )
        except Exception as exc:
            self.runtime.fail_stage(
                claim.stage_run.id,
                error=exc,
                backoff_seconds=retry_delay_seconds(max(1, claim.stage.attempt)),
            )
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=1)
        return self._sync_runtime_result(self.runtime.load_video_job(claim.job.id))

    def _start_stage_heartbeat(self, stage_plan_id: str) -> tuple[threading.Event, threading.Thread]:
        stop = threading.Event()
        interval = max(1.0, min(30.0, self.lease_seconds / 3.0))

        def heartbeat() -> None:
            while not stop.wait(interval):
                if not self.runtime.renew_stage_claim(
                    stage_plan_id,
                    owner=self.owner,
                    lease_seconds=self.lease_seconds,
                ):
                    return

        thread = threading.Thread(target=heartbeat, name=f"stage-lease-heartbeat-{self.owner}", daemon=True)
        thread.start()
        return stop, thread

    def process_next(
        self,
        processor: Callable[[VideoJobRecord], Any],
        *,
        stage_name: str = "video_processing",
        resource_lock: str | None = None,
        retry_failed: bool = False,
    ) -> VideoJobRecord | None:
        resolved_lock = self._resolve_resource_lock(stage_name, resource_lock)
        job = self.runtime.claim_next_video_job(owner=self.owner, lease_seconds=self.lease_seconds)
        if job is None:
            return None
        return self._process_claimed_with_lock(
            job,
            processor,
            stage_name=stage_name,
            resource_lock=resolved_lock,
            retry_failed=retry_failed,
        )

    def process_job(
        self,
        job_id: str,
        processor: Callable[[VideoJobRecord], Any],
        *,
        stage_name: str = "video_processing",
        resource_lock: str | None = None,
        retry_failed: bool = False,
    ) -> VideoJobRecord | None:
        resolved_lock = self._resolve_resource_lock(stage_name, resource_lock)
        claimed = self.runtime.claim_video_job(job_id, owner=self.owner, lease_seconds=self.lease_seconds)
        if claimed is None:
            return None
        return self._process_claimed_with_lock(
            claimed,
            processor,
            stage_name=stage_name,
            resource_lock=resolved_lock,
            retry_failed=retry_failed,
        )

    def _process_claimed_with_lock(
        self,
        job: VideoJobRecord,
        processor: Callable[[VideoJobRecord], Any],
        *,
        stage_name: str,
        resource_lock: str | None,
        retry_failed: bool,
    ) -> VideoJobRecord:
        lock_acquired = False
        if resource_lock is not None:
            lock_acquired = self.runtime.acquire_lock(
                resource_lock,
                owner=self.owner,
                lease_seconds=self.lease_seconds,
            )
            if not lock_acquired:
                self.runtime.release_video_job_claim(job.id, owner=self.owner)
                return None
        processing_warning = None
        should_complete = False
        stage = None
        try:
            stage = self.runtime.start_stage_run(
                video_job_id=job.id,
                stage_name=stage_name,
                input_hash=job.canonical_url,
            )
            current_job = self.runtime.load_video_job(job.id)
            if current_job.cancel_requested:
                self.runtime.complete_stage_run(stage.id, warning="cancel requested")
                return self._sync_runtime_result(self.runtime.skip_video_job(job.id, warning="cancel requested"))
            processor_result = processor(job)
            current_job = self.runtime.load_video_job(job.id)
            if current_job.cancel_requested:
                self.runtime.complete_stage_run(stage.id, warning="cancel requested")
                return self._sync_runtime_result(self.runtime.skip_video_job(job.id, warning="cancel requested"))
            if _processor_status(processor_result) == "skipped":
                warning = _processor_warning(processor_result) or "skipped"
                self.runtime.complete_stage_run(stage.id, warning=warning)
                return self._sync_runtime_result(self.runtime.skip_video_job(job.id, warning=warning))
            processing_warning = _processor_warning(processor_result)
            self.runtime.complete_stage_run(stage.id, warning=processing_warning)
            should_complete = True
        except Exception as exc:
            if stage is not None:
                self.runtime.fail_stage_run(stage.id, error=exc)
            if retry_failed:
                if stage is not None:
                    self.runtime.schedule_retry(
                        stage_run_id=stage.id,
                        attempt=stage.attempt + 1,
                        reason=f"{exc.__class__.__name__}: {exc}",
                        backoff_seconds=retry_delay_seconds(stage.attempt),
                    )
                return self._sync_runtime_result(self.runtime.resume_video_job(job.id))
            return self._sync_runtime_result(self.runtime.fail_video_job(job.id, error=str(exc)))
        finally:
            if lock_acquired and resource_lock is not None:
                self.runtime.release_lock(resource_lock, owner=self.owner)
        if not should_complete:
            return self.runtime.load_video_job(job.id)
        if not self.quality_gate_enabled:
            return self._sync_runtime_result(self.runtime.complete_video_job(job.id, warning=processing_warning))
        return self._complete_with_quality_gate(job, processing_warning=processing_warning)

    def _complete_with_quality_gate(self, job: VideoJobRecord, *, processing_warning: str | None = None) -> VideoJobRecord:
        if normalize_quality_mode(job.quality_mode) == "draft":
            return self._sync_runtime_result(
                self.runtime.complete_video_job(job.id, warning="draft mode: not production validated")
            )

        gate_stage = self.runtime.start_stage_run(
            video_job_id=job.id,
            stage_name="quality_gate",
            input_hash=f"{job.video_id}:{self.target_language}",
        )
        try:
            store = LibraryStore(self.library_dir)
            if self.target_language.strip().lower() == "ru":
                from yt_sub_hub.artifact_review import review_ru_artifacts

                review_ru_artifacts(job.video_id, store, None)
            result = run_quality_gate(
                store,
                job.video_id,
                target_language=self.target_language,
            )
        except Exception as exc:
            self.runtime.fail_stage_run(gate_stage.id, error=exc)
            return self._sync_runtime_result(self.runtime.fail_video_job(job.id, error=f"Quality gate failed: {exc}"))

        if result.hard_failures:
            message = "Quality gate hard failures: " + "; ".join(result.hard_failures)
            self.runtime.fail_stage_run(gate_stage.id, error=RuntimeError(message))
            return self._sync_runtime_result(self.runtime.fail_video_job(job.id, error=message))

        gate_warning = None
        if result.warnings:
            gate_warning = "Quality gate warnings: " + "; ".join(result.warnings)
        self.runtime.complete_stage_run(gate_stage.id, warning=gate_warning)
        warnings = [warning for warning in [processing_warning, gate_warning] if warning]
        return self._sync_runtime_result(
            self.runtime.complete_video_job(job.id, warning="; ".join(warnings) if warnings else None)
        )

    def _sync_runtime_result(self, record: VideoJobRecord) -> VideoJobRecord:
        sync_legacy_job_from_runtime(LibraryStore(self.library_dir), record)
        return record

    def _resolve_resource_lock(self, stage_name: str, explicit_lock: str | None) -> str | None:
        if stage_name == "video_processing":
            return explicit_lock
        stage = stage_by_name(stage_name)
        if explicit_lock is None and stage.name == "whisper_transcription":
            return whisper_resource_lock(self.whisper_backend)
        return explicit_lock if explicit_lock is not None else stage.resource_lock


def _processor_status(processor_result: Any) -> str | None:
    return getattr(processor_result, "status", None)


def _processor_warning(processor_result: Any) -> str | None:
    warning = getattr(processor_result, "warning", None)
    return str(warning) if warning else None
