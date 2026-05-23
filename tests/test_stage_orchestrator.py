import sys
import sqlite3
import threading
import time
import types

import pytest

from yt_sub_hub.artifact_review import ReviewReport
from yt_sub_hub.library import LibraryStore
from yt_sub_hub.orchestrator import BeeAIOrchestrator, LocalSQLiteOrchestrator
from yt_sub_hub.runtime import RuntimeStore
from yt_sub_hub.stage_pipeline import StagePipeline, StageResult
from test_youtube_discovery import _write_core_and_research


def _install_fake_beeai(monkeypatch):
    calls = {"decorated": [], "tool_runs": [], "workflow_instances": [], "steps": [], "runs": [], "states": []}
    root = types.ModuleType("beeai_framework")
    tools = types.ModuleType("beeai_framework.tools")
    workflows = types.ModuleType("beeai_framework.workflows")

    class StringToolOutput:
        def __init__(self, content):
            self.content = content

        def get_text_content(self):
            return self.content

    class FakeTool:
        def __init__(self, func):
            self.func = func
            self.name = func.__name__
            calls["decorated"].append(self.name)

        async def run(self, input=None, **kwargs):
            payload = input or kwargs
            calls["tool_runs"].append(payload)
            raise AssertionError("BeeAI direct tool.run bypassed Workflow orchestration")
            return self.func(**payload)

    def tool(func):
        return FakeTool(func)

    class FakeWorkflow:
        END = "__end__"

        def __init__(self, schema, name="Workflow", **kwargs):
            self.schema = schema
            self.name = name
            self.steps = []
            calls["workflow_instances"].append({"schema": schema, "name": name, "kwargs": kwargs})

        def add_step(self, step_name, runnable):
            self.steps.append((step_name, runnable))
            calls["steps"].append(step_name)
            return self

        async def run(self, state, options=None):
            calls["runs"].append(state)
            state_obj = self.schema(**state) if isinstance(state, dict) else state
            for _, runnable in self.steps:
                result = runnable(state_obj)
                if hasattr(result, "__await__"):
                    result = await result
                if result == self.END:
                    break
            calls["states"].append(state_obj)
            return types.SimpleNamespace(state=state_obj, result=state_obj, steps=[])

    tools.StringToolOutput = StringToolOutput
    tools.tool = tool
    workflows.Workflow = FakeWorkflow
    monkeypatch.setitem(sys.modules, "beeai_framework", root)
    monkeypatch.setitem(sys.modules, "beeai_framework.tools", tools)
    monkeypatch.setitem(sys.modules, "beeai_framework.workflows", workflows)
    return calls


def test_beeai_orchestrator_is_optional_when_package_missing(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "beeai_framework", None)

    with pytest.raises(RuntimeError, match=r'\.venv311/bin/python -m pip install -e "\.\[beeai\]"'):
        BeeAIOrchestrator(RuntimeStore(tmp_path / "index.sqlite3"), pipeline=StagePipeline(library_dir=tmp_path))


def test_beeai_orchestrator_uses_workflow_or_agent_primitive(tmp_path, monkeypatch):
    calls = _install_fake_beeai(monkeypatch)

    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(video_id="abc123def45", canonical_url="https://www.youtube.com/watch?v=abc123def45")
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    stages = []
    pipeline = StagePipeline(
        library_dir=tmp_path,
        handlers={
            "metadata": lambda ctx: stages.append(ctx.stage.stage_name) or StageResult.completed(),
        },
    )

    orchestrator = BeeAIOrchestrator(runtime, pipeline=pipeline, owner_prefix="bee")
    result = orchestrator.run_once()

    assert result is not None
    assert stages == ["metadata"]
    assert calls["runs"][0]["owner"].startswith("bee-")
    assert calls["workflow_instances"][0]["name"] == "YouTubeSubHubStageWorkflow"
    assert calls["steps"] == ["process_next_stage", "ru_artifact_critic"]
    assert calls["tool_runs"] == []
    assert not hasattr(orchestrator, "_delegate")
    assert calls["states"][-1].stage_name == "metadata"
    assert calls["states"][-1].stage_status == "completed"
    assert calls["states"][-1].critic_status == "not_applicable"


def test_beeai_workflow_invokes_process_next_stage_tool(tmp_path, monkeypatch):
    calls = _install_fake_beeai(monkeypatch)
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(video_id="abc123def45", canonical_url="https://www.youtube.com/watch?v=abc123def45")
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    stages = []
    pipeline = StagePipeline(
        library_dir=tmp_path,
        handlers={
            "metadata": lambda ctx: stages.append((ctx.stage.stage_name, ctx.job.video_id)) or StageResult.completed(),
        },
    )

    BeeAIOrchestrator(runtime, pipeline=pipeline, owner_prefix="bee").run_once()

    assert calls["runs"]
    assert calls["steps"] == ["process_next_stage", "ru_artifact_critic"]
    assert stages == [("metadata", "abc123def45")]
    assert runtime.list_stage_plan(job.id)[0].status == "completed"
    assert calls["states"][-1].stage_name == "metadata"
    assert calls["states"][-1].stage_status == "completed"
    assert calls["states"][-1].critic_status == "not_applicable"


def test_beeai_ru_artifact_critic_reads_processed_review_report(tmp_path, monkeypatch):
    calls = _install_fake_beeai(monkeypatch)
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(video_id="abc123def45", canonical_url="https://www.youtube.com/watch?v=abc123def45")
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute(
            "UPDATE stage_plans SET status = 'completed', attempt = 1 WHERE video_job_id = ? AND position < (SELECT position FROM stage_plans WHERE video_job_id = ? AND stage_name = 'ru_artifact_review')",
            (job.id, job.id),
        )
        conn.execute(
            "UPDATE video_jobs SET status = 'queued', current_stage = 'ru_artifact_review' WHERE id = ?",
            (job.id,),
        )

    def review_handler(ctx):
        report = ReviewReport(
            video_id=ctx.job.video_id,
            target_language="ru",
            status="passed",
            reviewer={"kind": "deterministic", "version": "test"},
            checked_artifacts={},
            hashes={},
            coverage={},
            issues=[],
        )
        path = LibraryStore(tmp_path).artifact_paths(ctx.job.video_id).ru_artifact_review
        path.parent.mkdir(parents=True, exist_ok=True)
        LibraryStore._write_json(path, report.to_dict())
        return StageResult.completed()

    pipeline = StagePipeline(library_dir=tmp_path, handlers={"ru_artifact_review": review_handler})

    result = BeeAIOrchestrator(runtime, pipeline=pipeline, owner_prefix="bee").run_once()

    assert result is not None
    state = calls["states"][-1]
    assert state.stage_name == "ru_artifact_review"
    assert state.stage_status == "completed"
    assert state.critic_role == "ru_artifact_critic"
    assert state.critic_status == "passed"
    assert state.critic_issue_count == 0
    assert state.critic_hard_failures == 0


def test_beeai_swarm_concurrency_respects_runtime_locks(tmp_path, monkeypatch):
    _install_fake_beeai(monkeypatch)
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    events = []
    events_lock = threading.Lock()

    def make_handler(name, delay):
        def handler(ctx):
            started = time.monotonic()
            with events_lock:
                events.append((name, ctx.job.video_id, "start", started))
            time.sleep(delay)
            finished = time.monotonic()
            with events_lock:
                events.append((name, ctx.job.video_id, "finish", finished))
            if name == "transcript_source":
                return StageResult.completed(skip_stages=["audio_download", "whisper_transcription"])
            if name in {"deep_graph_llm", "visual_frame_analysis", "neo4j_sync"}:
                return StageResult.skipped()
            return StageResult.completed()

        return handler

    handlers = {
        stage: make_handler(stage, 0.04)
        for stage in [
            "metadata",
            "transcript_source",
            "transcript_translation_expand",
            "transcript_translation_merge",
            "scientific_analysis",
            "analysis_reduce",
            "analysis_translation",
            "analysis_ru_review",
            "markdown_render",
            "deterministic_graph",
            "quality_gate",
            "deep_graph_llm",
            "visual_frame_analysis",
            "ru_artifact_review",
            "neo4j_sync",
        ]
    }
    for index in range(3):
        job = runtime.create_video_job(
            video_id=f"vid{index:08d}",
            canonical_url=f"https://www.youtube.com/watch?v=vid{index:08d}",
        )
        runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)

    pipeline = StagePipeline(library_dir=tmp_path, handlers=handlers)
    BeeAIOrchestrator(runtime, pipeline=pipeline, owner_prefix="bee").run_loop(concurrency=3)

    metadata_starts = [event for event in events if event[0] == "metadata" and event[2] == "start"]
    analysis_windows = [
        (event[1], event[3], next(f[3] for f in events if f[0] == "scientific_analysis" and f[1] == event[1] and f[2] == "finish"))
        for event in events
        if event[0] == "scientific_analysis" and event[2] == "start"
    ]
    assert len(metadata_starts) == 3
    assert max(event[3] for event in metadata_starts) - min(event[3] for event in metadata_starts) < 0.05
    for _, start_a, finish_a in analysis_windows:
        overlaps = [1 for _, start_b, finish_b in analysis_windows if start_a < finish_b and start_b < finish_a]
        assert len(overlaps) == 1
    assert {job.status for job in runtime.list_video_jobs()} <= {"completed", "completed_with_warning"}


def test_beeai_orchestrator_processes_staged_youtube_discovery_without_monolith(tmp_path, monkeypatch):
    calls = _install_fake_beeai(monkeypatch)
    _write_core_and_research(tmp_path)
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.queue_youtube_discovery("srcVideo123", limit=5, max_quota_units=900)
    monkeypatch.setattr(
        "yt_sub_hub.youtube_discovery.run_youtube_discovery",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("BeeAI staged path must not call monolithic discovery")),
    )

    result = BeeAIOrchestrator(runtime, pipeline=StagePipeline(library_dir=tmp_path), owner_prefix="bee").run_once()

    assert result is not None
    stages = {stage.stage_name: stage for stage in runtime.list_stage_plan(job.id)}
    assert stages["youtube_discovery_seed_extract"].status == "completed"
    assert (tmp_path / "videos" / "srcVideo123" / "youtube_discovery" / "seeds.json").exists()
    assert calls["steps"] == ["process_next_stage", "ru_artifact_critic"]
    assert calls["tool_runs"] == []
    assert calls["states"][-1].stage_name == "youtube_discovery_seed_extract"
    assert calls["states"][-1].stage_status == "completed"


def test_local_orchestrator_parallelizes_discovery_query_stages_under_youtube_data_api_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("YT_SUB_HUB_YOUTUBE_DATA_API_CONCURRENCY", "2")
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.queue_youtube_discovery("srcVideo123", limit=5, max_quota_units=900)
    runtime.expand_youtube_discovery_query_plan(
        job.id,
        {
            "artifact_hash": "queries-v1",
            "input_hashes": {"seed": "hash-1"},
            "items": [
                {"query_id": f"ydq-{index:03d}", "query": f"query {index}", "seed_ids": ["seed-001"]}
                for index in range(5)
            ],
        },
    )
    now = time.time()
    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute(
            """
            UPDATE stage_plans
            SET status = 'completed', attempt = 1, completed_at = ?, updated_at = ?
            WHERE video_job_id = ? AND stage_name IN ('youtube_discovery_seed_extract', 'youtube_discovery_query_expand')
            """,
            (now, now, job.id),
        )
        conn.execute("UPDATE video_jobs SET status = 'queued', current_stage = 'youtube_discovery_query_000' WHERE id = ?", (job.id,))

    events = []
    events_lock = threading.Lock()
    active = {"count": 0, "max": 0}

    def query_handler(ctx):
        started = time.monotonic()
        with events_lock:
            active["count"] += 1
            active["max"] = max(active["max"], active["count"])
            events.append((ctx.stage.stage_name, "start", started))
        time.sleep(0.06)
        finished = time.monotonic()
        with events_lock:
            events.append((ctx.stage.stage_name, "finish", finished))
            active["count"] -= 1
        return StageResult.completed()

    pipeline = StagePipeline(
        library_dir=tmp_path,
        handlers={f"youtube_discovery_query_{index:03d}": query_handler for index in range(5)},
    )

    LocalSQLiteOrchestrator(runtime, pipeline=pipeline, owner_prefix="discovery").run_loop(concurrency=5)

    windows = [
        (event[0], event[2], next(finish[2] for finish in events if finish[0] == event[0] and finish[1] == "finish"))
        for event in events
        if event[1] == "start"
    ]
    assert len(windows) == 5
    assert max(start for _, start, _ in windows[:2]) - min(start for _, start, _ in windows[:2]) < 0.05
    assert active["max"] <= 2
    assert all(_stage.status == "completed" for _stage in runtime.list_stage_plan(job.id) if _stage.stage_name.startswith("youtube_discovery_query_"))


def test_local_orchestrator_exits_when_pending_stages_are_not_runnable(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(video_id="abc123def45", canonical_url="https://www.youtube.com/watch?v=abc123def45")
    runtime.ensure_stage_plan(job.id)
    runtime.request_cancel(job.id)

    results = LocalSQLiteOrchestrator(runtime, pipeline=StagePipeline(library_dir=tmp_path)).run_loop(concurrency=2)

    assert results == []
