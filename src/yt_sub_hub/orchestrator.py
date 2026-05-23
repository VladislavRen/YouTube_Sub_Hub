from __future__ import annotations

import asyncio
import inspect
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, Field

from yt_sub_hub.config import get_settings
from yt_sub_hub.runtime import RuntimeStore, VideoJobRecord
from yt_sub_hub.stage_pipeline import StagePipeline


class OrchestratorBackend(Protocol):
    def run_once(self) -> VideoJobRecord | None: ...

    def run_loop(self, *, concurrency: int = 1) -> list[VideoJobRecord]: ...


@dataclass
class LocalSQLiteOrchestrator:
    runtime: RuntimeStore
    pipeline: StagePipeline
    owner_prefix: str = "local-stage-worker"
    lease_seconds: float = 300

    def run_once(self) -> VideoJobRecord | None:
        from yt_sub_hub.worker import StageWorker

        owner = f"{self.owner_prefix}-{uuid.uuid4().hex[:8]}"
        settings = getattr(self.pipeline, "settings", get_settings())
        return StageWorker(
            self.runtime,
            owner=owner,
            lease_seconds=self.lease_seconds,
            library_dir=getattr(settings, "library_dir", self.runtime.db_path.parent),
            target_language=getattr(settings, "target_language", "ru"),
            whisper_backend=getattr(settings, "whisper_backend", "openai-whisper-cpu"),
        ).process_next_stage(self.pipeline)

    def run_loop(self, *, concurrency: int = 1) -> list[VideoJobRecord]:
        results: list[VideoJobRecord] = []
        errors: list[BaseException] = []
        results_lock = threading.Lock()
        workers = max(1, int(concurrency))

        def worker_loop(index: int) -> None:
            from yt_sub_hub.worker import StageWorker

            try:
                settings = getattr(self.pipeline, "settings", get_settings())
                worker = StageWorker(
                    self.runtime,
                    owner=f"{self.owner_prefix}-{index}-{uuid.uuid4().hex[:8]}",
                    lease_seconds=self.lease_seconds,
                    library_dir=getattr(settings, "library_dir", self.runtime.db_path.parent),
                    target_language=getattr(settings, "target_language", "ru"),
                    whisper_backend=getattr(settings, "whisper_backend", "openai-whisper-cpu"),
                )
                while True:
                    result = worker.process_next_stage(self.pipeline)
                    if result is None:
                        status = self.runtime.stage_queue_status().get("statuses", {})
                        if status.get("running", 0) or self.runtime.list_runnable_stages():
                            time.sleep(0.01)
                            continue
                        break
                    with results_lock:
                        results.append(result)
            except BaseException as exc:
                with results_lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker_loop, args=(index,)) for index in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise RuntimeError(f"stage worker failed: {errors[0]}") from errors[0]
        return results


class BeeAIOrchestrator:
    def __init__(
        self,
        runtime: RuntimeStore,
        *,
        pipeline: StagePipeline,
        owner_prefix: str = "beeai-stage-worker",
        lease_seconds: float = 300,
    ) -> None:
        self.runtime = runtime
        self.pipeline = pipeline
        self.owner_prefix = owner_prefix
        self.lease_seconds = lease_seconds
        self._beeai = _load_beeai_api()

    def run_once(self) -> VideoJobRecord | None:
        owner = f"{self.owner_prefix}-{uuid.uuid4().hex[:8]}"
        workflow = self._build_stage_workflow()
        payload = self._run_beeai_workflow(workflow, owner=owner)
        return _job_from_payload(self.runtime, payload)

    def run_loop(self, *, concurrency: int = 1) -> list[VideoJobRecord]:
        results: list[VideoJobRecord] = []
        errors: list[BaseException] = []
        results_lock = threading.Lock()
        workers = max(1, int(concurrency))

        def worker_loop(index: int) -> None:
            try:
                owner = f"{self.owner_prefix}-{index}-{uuid.uuid4().hex[:8]}"
                workflow = self._build_stage_workflow()
                while True:
                    payload = self._run_beeai_workflow(workflow, owner=owner)
                    record = _job_from_payload(self.runtime, payload)
                    if record is None:
                        status = self.runtime.stage_queue_status().get("statuses", {})
                        if status.get("running", 0) or self.runtime.list_runnable_stages():
                            time.sleep(0.01)
                            continue
                        break
                    with results_lock:
                        results.append(record)
            except BaseException as exc:
                with results_lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker_loop, args=(index,)) for index in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise RuntimeError(f"BeeAI stage worker failed: {errors[0]}") from errors[0]
        return results

    def _build_stage_workflow(self) -> Any:
        workflow_class = self._beeai.workflow
        workflow = workflow_class(_BeeAIStageState, name="YouTubeSubHubStageWorkflow")

        def process_next_stage(state: _BeeAIStageState) -> Any:
            owner = state.owner
            payload = self._process_next_stage_payload(owner)
            state.claimed = bool(payload.get("claimed"))
            state.job_id = payload.get("job_id")
            state.video_id = payload.get("video_id")
            state.status = payload.get("status")
            state.job_status = payload.get("job_status")
            state.stage_name = payload.get("stage_name")
            state.stage_status = payload.get("stage_status")
            state.stage_warning = payload.get("stage_warning")
            state.stage_error = payload.get("stage_error")
            return None

        def ru_artifact_critic(state: _BeeAIStageState) -> Any:
            state.critic_role = "ru_artifact_critic"
            if not state.claimed or state.stage_name != "ru_artifact_review" or not state.video_id:
                state.critic_status = "not_applicable"
                state.critic_issue_count = 0
                state.critic_hard_failures = 0
                state.critic_issues = []
                return workflow_class.END
            report = _load_ru_artifact_review(self.pipeline, state.video_id)
            if report is None:
                state.critic_status = "missing_report"
                state.critic_issue_count = 0
                state.critic_hard_failures = 0
                state.critic_issues = []
                return workflow_class.END
            issues = list(report.get("issues") or [])
            state.critic_status = str(report.get("status") or "unknown")
            state.critic_issue_count = len(issues)
            state.critic_hard_failures = sum(1 for issue in issues if issue.get("severity") == "hard")
            state.critic_issues = issues
            return workflow_class.END

        workflow.add_step("process_next_stage", process_next_stage)
        workflow.add_step("ru_artifact_critic", ru_artifact_critic)
        return workflow

    def _process_next_stage_payload(self, owner: str) -> dict[str, Any]:
        from yt_sub_hub.worker import StageWorker

        settings = getattr(self.pipeline, "settings", get_settings())
        result = StageWorker(
            self.runtime,
            owner=owner,
            lease_seconds=self.lease_seconds,
            library_dir=getattr(settings, "library_dir", self.runtime.db_path.parent),
            target_language=getattr(settings, "target_language", "ru"),
            whisper_backend=getattr(settings, "whisper_backend", "openai-whisper-cpu"),
        ).process_next_stage(self.pipeline)
        if result is None:
            return {
                "claimed": False,
                "job_id": None,
                "video_id": None,
                "status": None,
                "job_status": None,
                "stage_name": None,
                "stage_status": None,
                "stage_warning": None,
                "stage_error": None,
            }
        stage_name = result.current_stage
        stage = None
        if stage_name:
            stage = next((item for item in self.runtime.list_stage_plan(result.id) if item.stage_name == stage_name), None)
        return {
            "claimed": True,
            "job_id": result.id,
            "video_id": result.video_id,
            "status": result.status,
            "job_status": result.status,
            "stage_name": stage_name,
            "stage_status": stage.status if stage is not None else None,
            "stage_warning": stage.warning if stage is not None else None,
            "stage_error": stage.error if stage is not None else None,
        }

    def _run_beeai_workflow(self, workflow: Any, *, owner: str) -> dict[str, Any]:
        return asyncio.run(self._run_beeai_workflow_async(workflow, owner=owner))

    async def _run_beeai_workflow_async(self, workflow: Any, *, owner: str) -> dict[str, Any]:
        run = workflow.run({"owner": owner})
        result = await run if inspect.isawaitable(run) else run
        state = getattr(result, "result", None) or getattr(result, "state", result)
        return _state_to_payload(state)


@dataclass(frozen=True)
class _BeeAIAPI:
    workflow: Any


class _BeeAIStageState(BaseModel):
    owner: str
    critic_role: str | None = None
    claimed: bool = False
    job_id: str | None = None
    video_id: str | None = None
    status: str | None = None
    job_status: str | None = None
    stage_name: str | None = None
    stage_status: str | None = None
    stage_warning: str | None = None
    stage_error: str | None = None
    critic_status: str | None = None
    critic_issue_count: int = 0
    critic_hard_failures: int = 0
    critic_issues: list[dict[str, Any]] = Field(default_factory=list)


def _load_beeai_api() -> _BeeAIAPI:
    try:
        import beeai_framework  # noqa: F401
        from beeai_framework.workflows import Workflow
    except Exception as exc:
        raise RuntimeError(
            'beeai-framework is not installed. Install optional dependency with '
            '`.venv311/bin/python -m pip install -e ".[beeai]"`.'
        ) from exc
    return _BeeAIAPI(workflow=Workflow)


def _state_to_payload(state: Any) -> dict[str, Any]:
    if isinstance(state, dict):
        return state
    if hasattr(state, "model_dump"):
        return state.model_dump()
    if hasattr(state, "dict"):
        return state.dict()
    return {
        "claimed": getattr(state, "claimed", False),
        "job_id": getattr(state, "job_id", None),
        "video_id": getattr(state, "video_id", None),
        "status": getattr(state, "status", None),
        "job_status": getattr(state, "job_status", None),
        "stage_name": getattr(state, "stage_name", None),
        "stage_status": getattr(state, "stage_status", None),
        "stage_warning": getattr(state, "stage_warning", None),
        "stage_error": getattr(state, "stage_error", None),
        "critic_role": getattr(state, "critic_role", None),
        "critic_status": getattr(state, "critic_status", None),
        "critic_issue_count": getattr(state, "critic_issue_count", 0),
        "critic_hard_failures": getattr(state, "critic_hard_failures", 0),
        "critic_issues": getattr(state, "critic_issues", []),
    }


def _load_ru_artifact_review(pipeline: StagePipeline, video_id: str) -> dict[str, Any] | None:
    from yt_sub_hub.artifact_review import load_ru_artifact_review
    from yt_sub_hub.library import LibraryStore

    store = getattr(pipeline, "store", None)
    if store is None:
        settings = getattr(pipeline, "settings", get_settings())
        store = LibraryStore(getattr(settings, "library_dir", get_settings().library_dir))
    return load_ru_artifact_review(store, video_id)


def _job_from_payload(runtime: RuntimeStore, payload: dict[str, Any]) -> VideoJobRecord | None:
    job_id = payload.get("job_id")
    if not job_id:
        return None
    return runtime.load_video_job(str(job_id))


def build_orchestrator(
    backend: str,
    runtime: RuntimeStore,
    *,
    pipeline: StagePipeline,
    owner_prefix: str = "stage-worker",
    lease_seconds: float = 300,
) -> OrchestratorBackend:
    if backend == "local":
        return LocalSQLiteOrchestrator(runtime, pipeline=pipeline, owner_prefix=owner_prefix, lease_seconds=lease_seconds)
    if backend == "beeai":
        return BeeAIOrchestrator(runtime, pipeline=pipeline, owner_prefix=owner_prefix, lease_seconds=lease_seconds)
    raise ValueError(f"Unknown orchestrator backend: {backend}")
