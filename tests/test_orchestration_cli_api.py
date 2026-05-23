import json
import sqlite3
import sys
import time
import ast

import pytest
from typer.testing import CliRunner

from yt_sub_hub import cli
from yt_sub_hub.library import LibraryStore
from yt_sub_hub.models import JobRecord
from yt_sub_hub.pipeline import job_id_for_url
from yt_sub_hub.runtime import RuntimeStore
from test_stage_worker import _write_russian_video
from test_artifact_review import _write_review_video


def _mark_transcript_source_failed(runtime: RuntimeStore, job_id: str) -> None:
    runtime.ensure_stage_plan(job_id, enable_deep_graph_llm=False)
    now = time.time()
    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute(
            "UPDATE stage_plans SET status = 'completed', attempt = 1, completed_at = ? WHERE video_job_id = ? AND stage_name = 'metadata'",
            (now, job_id),
        )
        conn.execute(
            """
            UPDATE stage_plans
            SET status = 'failed', attempt = max_attempts, retry_after = ?, error = ?,
                warning = ?, lease_owner = ?, lease_until = ?, resource_slot = ?,
                completed_at = ?
            WHERE video_job_id = ? AND stage_name = 'transcript_source'
            """,
            (now + 60, "max attempts exceeded", "old warning", "old-owner", now + 60, "youtube_metadata#0", now, job_id),
        )
        conn.execute(
            """
            UPDATE video_jobs
            SET status = 'failed', current_stage = 'transcript_source', error = ?, warning = ?, completed_at = ?
            WHERE id = ?
            """,
            ("max attempts exceeded", "old warning", now, job_id),
        )


def test_queue_status_cli_reports_runtime_queue_counts(tmp_path, monkeypatch):
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    runtime.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )

    result = CliRunner().invoke(cli.app, ["queue-status"])

    assert result.exit_code == 0
    assert "'queued': 1" in result.output


def test_queue_status_cli_can_include_stage_counts(tmp_path, monkeypatch):
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    runtime.ensure_stage_plan(job.id)

    result = CliRunner().invoke(cli.app, ["queue-status", "--stages"])

    assert result.exit_code == 0
    assert "stage_counts" in result.output
    assert "metadata" in result.output
    assert "analysis_ru_review" in result.output
    assert "ru_artifact_review" in result.output


def test_queue_plan_cli_creates_and_prints_stage_plan(tmp_path, monkeypatch):
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        job_id="legacy-job",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )

    result = CliRunner().invoke(cli.app, ["queue-plan", job.id])

    assert result.exit_code == 0
    assert "quality_gate" in result.output
    assert "analysis_ru_review" in result.output
    assert "ru_artifact_review" in result.output
    assert RuntimeStore(tmp_path / "index.sqlite3").list_stage_plan(job.id)[0].stage_name == "metadata"


def test_job_stages_api_includes_analysis_ru_review(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from yt_sub_hub.api import create_app

    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        job_id="legacy-job",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    runtime.ensure_stage_plan(job.id)

    response = TestClient(create_app()).get(f"/jobs/{job.id}/stages")

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] == job.id
    assert payload["video_id"] == "abc123def45"
    assert "overall" in payload
    assert "chunks" in payload
    assert "eta" in payload
    stage_names = [stage["stage_name"] for stage in payload["stages"]]
    assert "analysis_ru_review" in stage_names
    assert "ru_artifact_review" in stage_names


def test_job_stages_api_include_chunks_true_returns_chunk_rows(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from yt_sub_hub.api import create_app

    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        job_id="legacy-job",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    runtime.ensure_stage_plan(job.id)
    runtime.expand_translation_chunk_plan(
        job.id,
        [{"chunk_index": 0, "total_chunks": 1, "target_language": "ru", "source_segment_start_index": 0, "source_segment_end_index": 0}],
    )

    response = TestClient(create_app()).get(f"/jobs/{job.id}/stages?include_chunks=true")

    assert response.status_code == 200
    assert response.json()["chunks"]["transcript_translation"]["total"] == 1
    assert any(stage["stage_name"] == "transcript_translation_chunk_000" for stage in response.json()["stages"])


def test_job_artifacts_and_quality_api_return_progress_payloads(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from yt_sub_hub.api import create_app
    from yt_sub_hub.artifact_review import review_ru_artifacts
    from yt_sub_hub.config import get_settings

    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        job_id="legacy-job",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    runtime.ensure_stage_plan(job.id)
    store = LibraryStore(tmp_path)
    _write_review_video(store)
    review_ru_artifacts("abc123def45", store, get_settings())
    client = TestClient(create_app())

    artifacts = client.get(f"/jobs/{job.id}/artifacts")
    quality = client.get(f"/jobs/{job.id}/quality")

    assert artifacts.status_code == 200
    assert artifacts.json()["artifacts"]["note"]["exists"] is True
    assert artifacts.json()["readiness"]["ready_for_user_reading"] is True
    assert quality.status_code == 200
    assert quality.json()["ru_artifact_review"]["status"] == "passed"


def test_job_progress_api_unknown_job_returns_404(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from yt_sub_hub.api import create_app

    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    client = TestClient(create_app())

    assert client.get("/jobs/missing/stages").status_code == 404
    assert client.get("/jobs/missing/artifacts").status_code == 404
    assert client.get("/jobs/missing/quality").status_code == 404


def test_job_progress_cli_commands_emit_json(tmp_path, monkeypatch):
    from yt_sub_hub.artifact_review import review_ru_artifacts
    from yt_sub_hub.config import get_settings

    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        job_id="legacy-job",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    runtime.ensure_stage_plan(job.id)
    store = LibraryStore(tmp_path)
    _write_review_video(store)
    review_ru_artifacts("abc123def45", store, get_settings())
    now = time.time()
    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute(
            "UPDATE video_jobs SET status = 'completed', completed_at = ?, updated_at = ? WHERE id = ?",
            (now, now, job.id),
        )
        conn.execute(
            "UPDATE stage_plans SET status = 'completed', completed_at = ?, updated_at = ? WHERE video_job_id = ? AND stage_name = 'neo4j_sync'",
            (now, now, job.id),
        )

    stages = CliRunner().invoke(cli.app, ["job-stages", job.id, "--json"])
    artifacts = CliRunner().invoke(cli.app, ["job-artifacts", job.id, "--json"])
    artifact_browser = CliRunner().invoke(cli.app, ["artifacts", job.video_id, "--json"])
    quality = CliRunner().invoke(cli.app, ["job-quality", job.id, "--json"])

    assert stages.exit_code == 0
    assert json.loads(stages.output)["overall"]["total_stages"] > 0
    assert artifacts.exit_code == 0
    artifacts_payload = json.loads(artifacts.output)
    assert artifacts_payload["artifacts"]["metadata"]["exists"] is True
    assert "artifact_quality_score" in artifacts_payload["product_quality"]
    assert artifact_browser.exit_code == 0
    assert "artifact_quality_score" in json.loads(artifact_browser.output)["product_quality"]
    assert quality.exit_code == 0
    quality_payload = json.loads(quality.output)
    assert quality_payload["ru_artifact_review"]["status"] == "passed"
    assert "artifact_quality_score" in quality_payload["product_quality"]


def test_review_ru_artifacts_cli_reports_failed_for_sparse_fixture(tmp_path, monkeypatch):
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    store = LibraryStore(tmp_path)
    _write_review_video(store, source_count=40, ru_count=3, generic=True)

    result = CliRunner().invoke(cli.app, ["review-ru-artifacts", "abc123def45"])

    assert result.exit_code == 1
    assert "failed" in result.output
    assert "analysis_ru_sparse_concepts" in result.output


def test_video_ru_artifact_review_api_returns_report(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from yt_sub_hub.api import create_app
    from yt_sub_hub.artifact_review import review_ru_artifacts
    from yt_sub_hub.config import get_settings

    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    store = LibraryStore(tmp_path)
    _write_review_video(store)
    review_ru_artifacts("abc123def45", store, get_settings())

    response = TestClient(create_app()).get("/videos/abc123def45/review/ru-artifacts")

    assert response.status_code == 200
    assert response.json()["status"] == "passed"


def test_worker_stage_once_cli_processes_one_runnable_stage(tmp_path, monkeypatch):
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        job_id="legacy-job",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    runtime.ensure_stage_plan(job.id)

    class FakePipeline:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, ctx):
            return __import__("yt_sub_hub.stage_pipeline", fromlist=["StageResult"]).StageResult.completed()

    monkeypatch.setattr("yt_sub_hub.cli.StagePipeline", FakePipeline)

    result = CliRunner().invoke(cli.app, ["worker-stage-once", "--owner", "stage-worker"])

    assert result.exit_code == 0
    assert "metadata" in result.output
    assert RuntimeStore(tmp_path / "index.sqlite3").list_stage_plan(job.id)[0].status == "completed"


def test_worker_stage_once_cli_routes_to_beeai_orchestrator(tmp_path, monkeypatch):
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        job_id="legacy-job",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    runtime.ensure_stage_plan(job.id)
    observed = {}

    class FakeOrchestrator:
        def run_once(self):
            return runtime.load_video_job(job.id)

    def fake_build_orchestrator(backend, runtime_arg, *, pipeline, owner_prefix, lease_seconds):
        observed.update(
            backend=backend,
            runtime=runtime_arg,
            pipeline=pipeline,
            owner_prefix=owner_prefix,
            lease_seconds=lease_seconds,
        )
        return FakeOrchestrator()

    monkeypatch.setattr("yt_sub_hub.orchestrator.build_orchestrator", fake_build_orchestrator)

    result = CliRunner().invoke(cli.app, ["worker-stage-once", "--owner", "stage-worker", "--orchestrator", "beeai"])

    assert result.exit_code == 0
    assert observed["backend"] == "beeai"
    assert observed["owner_prefix"] == "stage-worker"
    assert "orchestrator=beeai" in result.output
    assert job.id in result.output


def test_worker_stage_loop_cli_routes_to_beeai_orchestrator(tmp_path, monkeypatch):
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        job_id="legacy-job",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    runtime.ensure_stage_plan(job.id)
    observed = {}

    class FakeOrchestrator:
        def run_loop(self, *, concurrency=1):
            observed["concurrency"] = concurrency
            return [runtime.load_video_job(job.id)]

    def fake_build_orchestrator(backend, runtime_arg, *, pipeline, owner_prefix, lease_seconds):
        observed.update(backend=backend, owner_prefix=owner_prefix, lease_seconds=lease_seconds)
        return FakeOrchestrator()

    monkeypatch.setattr("yt_sub_hub.orchestrator.build_orchestrator", fake_build_orchestrator)

    result = CliRunner().invoke(cli.app, ["worker-stage-loop", "--orchestrator", "beeai", "--concurrency", "2"])

    assert result.exit_code == 0
    assert observed["backend"] == "beeai"
    assert observed["concurrency"] == 2
    assert "orchestrator=beeai" in result.output
    assert "processed_stages" in result.output


def test_queue_discovery_cli_creates_staged_discovery_plan(tmp_path, monkeypatch):
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))

    result = CliRunner().invoke(cli.app, ["queue-discovery", "srcVideo123", "--limit", "7", "--max-quota-units", "300", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    stages = runtime.list_stage_plan(payload["job_id"])
    assert payload["video_id"] == "srcVideo123"
    assert payload["job_kind"] == "youtube_discovery"
    assert payload["stage_count"] == 7
    assert [stage.stage_name for stage in stages][:2] == ["youtube_discovery_seed_extract", "youtube_discovery_query_expand"]


def test_discovery_status_cli_can_include_stage_progress(tmp_path, monkeypatch):
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.queue_youtube_discovery("srcVideo123", limit=7, max_quota_units=300)
    core = runtime.create_video_job(
        video_id="srcVideo123",
        canonical_url="https://www.youtube.com/watch?v=srcVideo123",
    )
    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute("UPDATE video_jobs SET updated_at = ? WHERE id = ?", (time.time() + 10, core.id))

    result = CliRunner().invoke(cli.app, ["discovery-status", "srcVideo123", "--stages", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["stage_progress"]["job_id"] == job.id
    assert payload["stage_progress"]["job_kind"] == "youtube_discovery"
    assert payload["stage_progress"]["chunks"]["youtube_discovery_queries"]["total"] == 0


def test_worker_stage_once_cli_reports_missing_beeai_dependency_cleanly(tmp_path, monkeypatch):
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "beeai_framework", None)

    result = CliRunner().invoke(cli.app, ["worker-stage-once", "--orchestrator", "beeai"])

    assert result.exit_code == 1
    assert "beeai-framework is not installed" in result.output
    assert '.venv311/bin/python -m pip install -e ".[beeai]"' in result.output


def test_cancel_cli_sets_cancel_requested(tmp_path, monkeypatch):
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )

    result = CliRunner().invoke(cli.app, ["cancel", job.id])

    assert result.exit_code == 0
    assert "cancel_requested" in result.output
    assert RuntimeStore(tmp_path / "index.sqlite3").load_video_job(job.id).cancel_requested is True


def test_enqueue_video_cli_creates_legacy_and_runtime_job_with_stable_id(tmp_path, monkeypatch):
    url = "https://www.youtube.com/watch?v=abc123def45"
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))

    result = CliRunner().invoke(cli.app, ["enqueue-video", url])

    assert result.exit_code == 0
    expected_job_id = job_id_for_url(url)
    legacy = LibraryStore(tmp_path).load_jobs()[expected_job_id]
    runtime = RuntimeStore(tmp_path / "index.sqlite3").load_video_job(expected_job_id)
    assert legacy.status == "queued"
    assert runtime.status == "queued"
    assert runtime.video_id == "abc123def45"
    assert runtime.canonical_url == url
    assert expected_job_id in result.output


def test_enqueue_video_cli_accepts_quality_mode_override(tmp_path, monkeypatch):
    url = "https://www.youtube.com/watch?v=abc123def45"
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))

    result = CliRunner().invoke(cli.app, ["enqueue-video", url, "--quality-mode", "draft"])

    assert result.exit_code == 0
    runtime = RuntimeStore(tmp_path / "index.sqlite3").load_video_job(job_id_for_url(url))
    assert runtime.quality_mode == "draft"
    assert "quality_mode" in result.output


def test_enqueue_video_cli_accepts_acceptance_mode_override(tmp_path, monkeypatch):
    url = "https://www.youtube.com/watch?v=abc123def45"
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))

    result = CliRunner().invoke(cli.app, ["enqueue-video", url, "--acceptance-mode", "full_processing"])

    assert result.exit_code == 0
    runtime = RuntimeStore(tmp_path / "index.sqlite3").load_video_job(job_id_for_url(url))
    assert runtime.acceptance_mode == "full_processing"
    assert "acceptance_mode" in result.output


def test_enqueue_channel_cli_creates_channel_job_with_json_status(tmp_path, monkeypatch):
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))

    def fake_fetch(channel_url, *, limit=None, settings=None):
        from yt_sub_hub.channel import ChannelFetchResult, ChannelVideo

        return ChannelFetchResult(
            channel_id="UCexample123",
            channel_url="https://www.youtube.com/channel/UCexample123",
            uploads_playlist_id="UUexample123",
            title="Example Lab",
            videos=[
                ChannelVideo("abc123def45", "https://www.youtube.com/watch?v=abc123def45", "First"),
                ChannelVideo("xyz123def45", "https://www.youtube.com/watch?v=xyz123def45", "Second"),
            ],
        )

    monkeypatch.setattr("yt_sub_hub.channel.fetch_channel_listing", fake_fetch)

    enqueue = CliRunner().invoke(
        cli.app,
        [
            "enqueue-channel",
            "https://www.youtube.com/@ExampleLab",
            "--limit",
            "2",
            "--quality-mode",
            "draft",
            "--acceptance-mode",
            "full_processing",
        ],
    )
    channel_job_id = ast.literal_eval(enqueue.output)["channel_job_id"]
    status = CliRunner().invoke(cli.app, ["channel-status", channel_job_id, "--json"])

    assert enqueue.exit_code == 0
    assert status.exit_code == 0
    payload = json.loads(status.output)
    assert payload["channel_job"]["channel_id"] == "UCexample123"
    assert payload["total_videos"] == 2
    assert payload["counts"]["queued"] == 2
    assert {video["quality_mode"] for video in payload["videos"]} == {"draft"}
    assert {video["acceptance_mode"] for video in payload["videos"]} == {"full_processing"}
    assert payload["acceptance_mode"] == "full_processing"


def test_enqueue_video_cli_requeues_existing_terminal_runtime_job(tmp_path, monkeypatch):
    url = "https://www.youtube.com/watch?v=abc123def45"
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    job_id = job_id_for_url(url)
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    runtime.create_video_job(job_id=job_id, video_id="abc123def45", canonical_url=url)
    runtime.complete_video_job(job_id)

    result = CliRunner().invoke(cli.app, ["enqueue-video", url])

    assert result.exit_code == 0
    assert RuntimeStore(tmp_path / "index.sqlite3").load_video_job(job_id).status == "queued"


def test_worker_once_cli_processes_one_queued_job_through_stage_worker(tmp_path, monkeypatch):
    url = "https://www.youtube.com/watch?v=abc123def45"
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    LibraryStore(tmp_path).save_job(
        JobRecord(
            job_id=job_id_for_url(url),
            url=url,
            video_id="abc123def45",
            status="queued",
            stage="queued",
        )
    )
    RuntimeStore(tmp_path / "index.sqlite3").create_video_job(
        job_id=job_id_for_url(url),
        video_id="abc123def45",
        canonical_url=url,
    )
    observed = {}

    class FakePipeline:
        def __init__(self, store=None, cancel_checker=None):
            observed["cancel_checker"] = cancel_checker
            self.store = store

        def process_job(self, job_id):
            observed["job_id"] = job_id
            observed["cancel_requested"] = observed["cancel_checker"](job_id)
            _write_russian_video(self.store, "abc123def45")
            return JobRecord(job_id=job_id, url=url, video_id="abc123def45", status="completed", stage="graph_synced")

    monkeypatch.setattr(cli, "_worker_pipeline", lambda store, cancel_checker: FakePipeline(store, cancel_checker))

    result = CliRunner().invoke(cli.app, ["worker-once", "--owner", "worker-1", "--lease-seconds", "60"])

    assert result.exit_code == 0
    assert observed == {
        "cancel_checker": observed["cancel_checker"],
        "job_id": job_id_for_url(url),
        "cancel_requested": False,
    }
    assert RuntimeStore(tmp_path / "index.sqlite3").load_video_job(job_id_for_url(url)).status == "completed"
    assert "completed" in result.output


def test_worker_once_cli_reports_cleanly_when_no_job_available(tmp_path, monkeypatch):
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))

    result = CliRunner().invoke(cli.app, ["worker-once", "--owner", "worker-1"])

    assert result.exit_code == 0
    assert "no queued job available" in result.output


def test_worker_once_cli_recovers_stale_legacy_running_before_claiming(tmp_path, monkeypatch):
    url = "https://www.youtube.com/watch?v=abc123def45"
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    store = LibraryStore(tmp_path)
    store.save_job(
        JobRecord(
            job_id=job_id_for_url(url),
            url=url,
            video_id="abc123def45",
            status="running",
            stage="transcript_resolved",
        )
    )
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    runtime.create_video_job(
        job_id=job_id_for_url(url),
        video_id="abc123def45",
        canonical_url=url,
    )
    runtime.fail_video_job(job_id_for_url(url), error="interrupted")

    result = CliRunner().invoke(cli.app, ["worker-once", "--owner", "worker-1"])

    legacy = LibraryStore(tmp_path).load_jobs()[job_id_for_url(url)]
    assert result.exit_code == 0
    assert legacy.status == "failed"
    assert legacy.stage == "failed"
    assert legacy.error == "Interrupted worker run recovered; previous stage: transcript_resolved"


def test_worker_once_cli_does_not_pass_whisper_lock_to_video_processing(tmp_path, monkeypatch):
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    observed = {}

    class RecordingStageWorker:
        def __init__(self, runtime, **kwargs):
            observed["init_kwargs"] = kwargs

        def process_next(self, processor, **kwargs):
            observed["process_kwargs"] = kwargs
            return None

    monkeypatch.setattr("yt_sub_hub.worker.StageWorker", RecordingStageWorker)

    result = CliRunner().invoke(cli.app, ["worker-once", "--owner", "worker-1"])

    assert result.exit_code == 0
    assert observed["init_kwargs"]["whisper_backend"] == "openai-whisper-cpu"
    assert observed["process_kwargs"].get("resource_lock") is None


def test_worker_once_cli_runs_video_processing_when_whisper_lock_is_held_for_mps_backend(tmp_path, monkeypatch):
    url = "https://www.youtube.com/watch?v=abc123def45"
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    monkeypatch.setenv("YT_SUB_HUB_WHISPER_BACKEND", "openai-whisper-mps")
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    runtime.create_video_job(
        job_id=job_id_for_url(url),
        video_id="abc123def45",
        canonical_url=url,
    )
    runtime.acquire_lock("whisper_accelerated", owner="worker-1", lease_seconds=60)
    processed = []

    class FakePipeline:
        def __init__(self, store=None, cancel_checker=None):
            self.store = store

        def process_job(self, job_id):
            processed.append(job_id)
            _write_russian_video(self.store, "abc123def45")
            return JobRecord(job_id=job_id, url=url, video_id="abc123def45", status="completed")

    monkeypatch.setattr(cli, "_worker_pipeline", lambda store, cancel_checker: FakePipeline(store, cancel_checker))

    result = CliRunner().invoke(cli.app, ["worker-once", "--owner", "worker-2"])

    assert result.exit_code == 0
    assert processed == [job_id_for_url(url)]
    assert runtime.load_video_job(job_id_for_url(url)).status == "completed"
    assert "completed" in result.output


def test_worker_once_default_owner_is_process_unique(tmp_path, monkeypatch):
    url = "https://www.youtube.com/watch?v=abc123def45"
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    monkeypatch.setattr("os.getpid", lambda: 4242)
    observed = {}
    RuntimeStore(tmp_path / "index.sqlite3").create_video_job(
        job_id=job_id_for_url(url),
        video_id="abc123def45",
        canonical_url=url,
    )

    class RecordingStageWorker:
        def __init__(self, runtime, **kwargs):
            observed.update(kwargs)

        def process_next(self, processor, **kwargs):
            return None

    monkeypatch.setattr("yt_sub_hub.worker.StageWorker", RecordingStageWorker)

    result = CliRunner().invoke(cli.app, ["worker-once"])

    assert result.exit_code == 0
    assert observed["owner"] == "local-worker-4242"


def test_api_exposes_queue_and_cancel_endpoints(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from yt_sub_hub.api import create_app

    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )

    client = TestClient(create_app())
    assert client.get("/queue").json()["queued"] == 1

    response = client.post(f"/jobs/{job.id}/cancel")

    assert response.status_code == 200
    assert response.json()["cancel_requested"] is True
    assert "next cancellation checkpoint" in response.json()["message"]


def test_api_cancel_updates_queued_legacy_job_to_cancelled_terminal_state(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from yt_sub_hub.api import create_app

    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    LibraryStore(tmp_path).save_job(
        JobRecord(
            job_id="legacy-job",
            url="https://youtu.be/abc123def45",
            video_id="abc123def45",
            status="queued",
            stage="queued",
        )
    )
    RuntimeStore(tmp_path / "index.sqlite3").create_video_job(
        job_id="legacy-job",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )

    response = TestClient(create_app()).post("/jobs/legacy-job/cancel")

    assert response.status_code == 200
    legacy = LibraryStore(tmp_path).load_jobs()["legacy-job"]
    runtime = RuntimeStore(tmp_path / "index.sqlite3").load_video_job("legacy-job")
    assert legacy.status == "skipped"
    assert legacy.stage == "cancelled"
    assert legacy.warning == "cancel requested"
    assert runtime.status == "skipped"


def test_api_create_job_also_registers_runtime_video_job_for_cancel(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import yt_sub_hub.api as api_module
    from yt_sub_hub.api import create_app

    class FakePipeline:
        def already_processed(self, url):
            return None

    def fake_create_job_record(url, library_dir):
        from yt_sub_hub.models import JobRecord

        return JobRecord(
            job_id="legacy-job",
            url=url,
            video_id="abc123def45",
            status="queued",
            stage="queued",
        )

    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    monkeypatch.setattr(api_module, "VideoProcessingPipeline", lambda store=None, enable_default_graph=False: FakePipeline())
    monkeypatch.setattr(api_module, "create_job_record", fake_create_job_record)
    monkeypatch.setattr(api_module, "process_existing_job", lambda job_id, library_dir: None)
    client = TestClient(create_app())

    created = client.post("/jobs", json={"url": "https://youtu.be/abc123def45"}).json()
    cancelled = client.post(f"/jobs/{created['job_id']}/cancel")

    assert cancelled.status_code == 200
    assert cancelled.json()["cancel_requested"] is True


def test_api_create_job_requeues_existing_terminal_runtime_job(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import yt_sub_hub.api as api_module
    from yt_sub_hub.api import create_app

    class FakePipeline:
        def already_processed(self, url):
            return None

    def fake_create_job_record(url, library_dir):
        return JobRecord(
            job_id="legacy-job",
            url=url,
            video_id="abc123def45",
            status="queued",
            stage="queued",
        )

    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    RuntimeStore(tmp_path / "index.sqlite3").create_video_job(
        job_id="legacy-job",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    RuntimeStore(tmp_path / "index.sqlite3").fail_video_job("legacy-job", error="old failure")
    monkeypatch.setattr(api_module, "VideoProcessingPipeline", lambda store=None, enable_default_graph=False: FakePipeline())
    monkeypatch.setattr(api_module, "create_job_record", fake_create_job_record)
    monkeypatch.setattr(api_module, "process_existing_job", lambda job_id, library_dir: None)

    response = TestClient(create_app()).post("/jobs", json={"url": "https://youtu.be/abc123def45"})

    assert response.status_code == 200
    assert RuntimeStore(tmp_path / "index.sqlite3").load_video_job("legacy-job").status == "queued"


def test_api_background_worker_does_not_process_job_already_running_by_same_owner(tmp_path, monkeypatch):
    import yt_sub_hub.api as api_module
    from yt_sub_hub.api import process_existing_job

    LibraryStore(tmp_path).save_job(
        JobRecord(
            job_id="legacy-job",
            url="https://youtu.be/abc123def45",
            video_id="abc123def45",
            status="running",
            stage="running",
        )
    )
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    runtime.create_video_job(
        job_id="legacy-job",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
        status="queued",
    )
    assert runtime.claim_video_job("legacy-job", owner="api-background", lease_seconds=60) is not None
    observed = {}

    class FakePipeline:
        def __init__(self, store=None, cancel_checker=None, enable_default_graph=False):
            pass

        def process_job(self, job_id):
            observed["processed"] = True

    monkeypatch.setattr(api_module, "VideoProcessingPipeline", FakePipeline)

    process_existing_job("legacy-job", tmp_path)

    assert observed == {}


def test_process_existing_job_skips_pre_cancelled_runtime_job_without_processor(tmp_path, monkeypatch):
    import yt_sub_hub.api as api_module
    from yt_sub_hub.api import process_existing_job

    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    runtime.create_video_job(
        job_id="legacy-job",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
        status="running",
    )
    runtime.request_cancel("legacy-job")

    observed = {}

    class FakePipeline:
        def __init__(self, store=None, cancel_checker=None, enable_default_graph=False):
            observed["pipeline_constructed"] = True

        def process_job(self, job_id):
            observed["job_id"] = job_id

    monkeypatch.setattr(api_module, "VideoProcessingPipeline", FakePipeline)

    process_existing_job("legacy-job", tmp_path)

    runtime = RuntimeStore(tmp_path / "index.sqlite3").load_video_job("legacy-job")
    assert observed == {}
    assert runtime.status == "skipped"
    assert runtime.warning == "cancel requested"


def test_process_existing_job_claims_runtime_job_before_processing(tmp_path, monkeypatch):
    import yt_sub_hub.api as api_module
    from yt_sub_hub.api import process_existing_job

    LibraryStore(tmp_path).save_job(
        JobRecord(
            job_id="legacy-job",
            url="https://youtu.be/abc123def45",
            video_id="abc123def45",
            status="queued",
            stage="queued",
        )
    )
    RuntimeStore(tmp_path / "index.sqlite3").create_video_job(
        job_id="legacy-job",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
        status="queued",
    )
    observed = {}

    class FakePipeline:
        def __init__(self, store=None, cancel_checker=None, enable_default_graph=False, settings=None):
            self.store = store

        def process_job(self, job_id):
            observed["runtime_during_processing"] = RuntimeStore(tmp_path / "index.sqlite3").load_video_job(job_id).status
            _write_russian_video(self.store, "abc123def45")
            return JobRecord(
                job_id=job_id,
                url="https://youtu.be/abc123def45",
                video_id="abc123def45",
                status="completed",
                stage="graph_synced",
            )

    monkeypatch.setattr(api_module, "VideoProcessingPipeline", FakePipeline)

    process_existing_job("legacy-job", tmp_path)

    assert observed["runtime_during_processing"] == "running"
    assert RuntimeStore(tmp_path / "index.sqlite3").load_video_job("legacy-job").status == "completed"


def test_process_existing_job_uses_stage_worker_quality_gate_before_runtime_completion(tmp_path, monkeypatch):
    import yt_sub_hub.api as api_module
    from yt_sub_hub.api import process_existing_job

    LibraryStore(tmp_path).save_job(
        JobRecord(
            job_id="legacy-job",
            url="https://youtu.be/abc123def45",
            video_id="abc123def45",
            status="queued",
            stage="queued",
        )
    )
    RuntimeStore(tmp_path / "index.sqlite3").create_video_job(
        job_id="legacy-job",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
        status="queued",
    )

    class FakePipeline:
        def __init__(self, store=None, cancel_checker=None, enable_default_graph=False, settings=None):
            self.store = store

        def process_job(self, job_id):
            _write_russian_video(self.store, "abc123def45")
            self.store.artifact_paths("abc123def45").analysis_ru.unlink()
            return JobRecord(
                job_id=job_id,
                url="https://youtu.be/abc123def45",
                video_id="abc123def45",
                status="completed",
                stage="graph_synced",
            )

    monkeypatch.setattr(api_module, "VideoProcessingPipeline", FakePipeline)

    process_existing_job("legacy-job", tmp_path)

    runtime = RuntimeStore(tmp_path / "index.sqlite3").load_video_job("legacy-job")
    legacy = LibraryStore(tmp_path).load_jobs()["legacy-job"]
    assert runtime.status == "failed"
    assert legacy.status == "failed"
    assert legacy.stage == "failed"
    assert "analysis.ru.json must exist" in (legacy.error or "")
    assert "analysis.ru.json must exist" in (runtime.error or "")


def test_process_existing_job_marks_runtime_failed_when_pipeline_fails(tmp_path, monkeypatch):
    import yt_sub_hub.api as api_module
    from yt_sub_hub.api import process_existing_job

    LibraryStore(tmp_path).save_job(
        JobRecord(
            job_id="legacy-job",
            url="https://youtu.be/abc123def45",
            video_id="abc123def45",
            status="queued",
            stage="queued",
        )
    )
    RuntimeStore(tmp_path / "index.sqlite3").create_video_job(
        job_id="legacy-job",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
        status="queued",
    )

    class FakePipeline:
        def __init__(self, store=None, cancel_checker=None, enable_default_graph=False, settings=None):
            pass

        def process_job(self, job_id):
            raise RuntimeError("pipeline failed")

    monkeypatch.setattr(api_module, "VideoProcessingPipeline", FakePipeline)

    with pytest.raises(RuntimeError, match="pipeline failed"):
        process_existing_job("legacy-job", tmp_path)

    runtime = RuntimeStore(tmp_path / "index.sqlite3").load_video_job("legacy-job")
    assert runtime.status == "failed"
    assert runtime.error == "pipeline failed"


def test_api_background_process_recovers_stale_legacy_running_before_claiming(tmp_path):
    from yt_sub_hub.api import process_existing_job

    LibraryStore(tmp_path).save_job(
        JobRecord(
            job_id="legacy-job",
            url="https://youtu.be/abc123def45",
            video_id="abc123def45",
            status="running",
            stage="transcript_resolved",
        )
    )
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    runtime.create_video_job(
        job_id="legacy-job",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
        status="queued",
    )
    runtime.fail_video_job("legacy-job", error="interrupted")

    process_existing_job("legacy-job", tmp_path)

    legacy = LibraryStore(tmp_path).load_jobs()["legacy-job"]
    assert legacy.status == "failed"
    assert legacy.stage == "failed"
    assert legacy.error == "Interrupted worker run recovered; previous stage: transcript_resolved"


def test_running_api_cancel_is_checkpoint_based_until_pipeline_observes_it(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from yt_sub_hub.api import create_app

    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    LibraryStore(tmp_path).save_job(
        JobRecord(
            job_id="legacy-job",
            url="https://youtu.be/abc123def45",
            video_id="abc123def45",
            status="running",
            stage="metadata_fetched",
        )
    )
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    runtime.create_video_job(
        job_id="legacy-job",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
        status="queued",
    )
    runtime.claim_next_video_job(owner="api-worker", lease_seconds=60)

    response = TestClient(create_app()).post("/jobs/legacy-job/cancel")

    assert response.status_code == 200
    runtime_record = RuntimeStore(tmp_path / "index.sqlite3").load_video_job("legacy-job")
    legacy = LibraryStore(tmp_path).load_jobs()["legacy-job"]
    assert runtime_record.status == "running"
    assert runtime_record.cancel_requested is True
    assert legacy.status == "running"
    assert legacy.stage == "metadata_fetched"


def test_api_retry_failed_rejects_non_failed_runtime_job(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from yt_sub_hub.api import create_app

    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    runtime.create_video_job(
        job_id="legacy-job",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    runtime.complete_video_job("legacy-job")

    response = TestClient(create_app()).post("/jobs/legacy-job/retry-failed")

    assert response.status_code == 409
    assert RuntimeStore(tmp_path / "index.sqlite3").load_video_job("legacy-job").status == "completed"


def test_retry_failed_cli_requeues_runtime_and_legacy_job(tmp_path, monkeypatch):
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    LibraryStore(tmp_path).save_job(
        JobRecord(
            job_id="legacy-job",
            url="https://youtu.be/abc123def45",
            video_id="abc123def45",
            status="failed",
            stage="failed",
            error="previous failure",
            warning="previous warning",
        )
    )
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    runtime.create_video_job(
        job_id="legacy-job",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    _mark_transcript_source_failed(runtime, "legacy-job")

    result = CliRunner().invoke(cli.app, ["retry-failed", "legacy-job"])

    legacy = LibraryStore(tmp_path).load_jobs()["legacy-job"]
    runtime_record = RuntimeStore(tmp_path / "index.sqlite3").load_video_job("legacy-job")
    transcript = [
        stage
        for stage in RuntimeStore(tmp_path / "index.sqlite3").list_stage_plan("legacy-job")
        if stage.stage_name == "transcript_source"
    ][0]
    assert result.exit_code == 0
    assert runtime_record.status == "queued"
    assert runtime_record.current_stage == "transcript_source"
    assert runtime_record.error is None
    assert runtime_record.warning is None
    assert transcript.status == "pending"
    assert transcript.attempt == 0
    assert transcript.error is None
    assert transcript.warning is None
    assert transcript.retry_after is None
    assert legacy.status == "queued"
    assert legacy.stage == "queued"
    assert legacy.error is None


def test_retry_failed_cli_from_stage_requeues_terminal_job_from_analysis_ru_review(tmp_path, monkeypatch):
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    LibraryStore(tmp_path).save_job(
        JobRecord(
            job_id="legacy-job",
            url="https://youtu.be/abc123def45",
            video_id="abc123def45",
            status="completed",
            stage="quality_gate",
        )
    )
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    runtime.create_video_job(
        job_id="legacy-job",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    runtime.ensure_stage_plan("legacy-job", enable_deep_graph_llm=False)
    now = time.time()
    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute(
            "UPDATE stage_plans SET status = 'completed', attempt = 1, completed_at = ?, updated_at = ? WHERE video_job_id = ?",
            (now, now, "legacy-job"),
        )
        conn.execute(
            "UPDATE video_jobs SET status = 'completed', current_stage = 'quality_gate', completed_at = ?, updated_at = ? WHERE id = ?",
            (now, now, "legacy-job"),
        )

    result = CliRunner().invoke(cli.app, ["retry-failed", "legacy-job", "--from-stage", "analysis_ru_review"])

    runtime_record = RuntimeStore(tmp_path / "index.sqlite3").load_video_job("legacy-job")
    assert result.exit_code == 0
    assert runtime_record.status == "queued"
    assert runtime_record.current_stage == "analysis_ru_review"
    assert [stage for stage in RuntimeStore(tmp_path / "index.sqlite3").list_stage_plan("legacy-job") if stage.stage_name == "analysis_translation"][0].status == "completed"
    assert [stage for stage in RuntimeStore(tmp_path / "index.sqlite3").list_stage_plan("legacy-job") if stage.stage_name == "analysis_ru_review"][0].status == "pending"


def test_retry_failed_cli_requeues_stale_running_job_with_failed_stage_plan(tmp_path, monkeypatch):
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    LibraryStore(tmp_path).save_job(
        JobRecord(
            job_id="legacy-job",
            url="https://youtu.be/abc123def45",
            video_id="abc123def45",
            status="running",
            stage="transcript_translation_expand",
        )
    )
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    runtime.create_video_job(
        job_id="legacy-job",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    runtime.ensure_stage_plan("legacy-job", enable_deep_graph_llm=False)
    now = time.time()
    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute(
            "UPDATE stage_plans SET status = 'completed', completed_at = ? WHERE video_job_id = ? AND position < 4",
            (now, "legacy-job"),
        )
        conn.execute(
            """
            UPDATE stage_plans
            SET status = 'failed', attempt = max_attempts, error = ?, completed_at = ?
            WHERE video_job_id = ? AND stage_name = 'transcript_translation_expand'
            """,
            ("max attempts exceeded", now, "legacy-job"),
        )
        conn.execute(
            """
            UPDATE video_jobs
            SET status = 'running', current_stage = 'transcript_translation_expand',
                lease_owner = 'old-worker', lease_until = ?, completed_at = NULL
            WHERE id = ?
            """,
            (now - 60, "legacy-job"),
        )

    result = CliRunner().invoke(cli.app, ["retry-failed", "legacy-job"])

    legacy = LibraryStore(tmp_path).load_jobs()["legacy-job"]
    runtime_record = RuntimeStore(tmp_path / "index.sqlite3").load_video_job("legacy-job")
    stage = [
        item
        for item in RuntimeStore(tmp_path / "index.sqlite3").list_stage_plan("legacy-job")
        if item.stage_name == "transcript_translation_expand"
    ][0]
    assert result.exit_code == 0
    assert runtime_record.status == "queued"
    assert runtime_record.current_stage == "transcript_translation_expand"
    assert stage.status == "pending"
    assert stage.attempt == 0
    assert legacy.status == "queued"
    assert legacy.warning is None


def test_api_retry_failed_requeues_runtime_and_legacy_job(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from yt_sub_hub.api import create_app

    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    LibraryStore(tmp_path).save_job(
        JobRecord(
            job_id="legacy-job",
            url="https://youtu.be/abc123def45",
            video_id="abc123def45",
            status="failed",
            stage="failed",
            error="previous failure",
            warning="previous warning",
        )
    )
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    runtime.create_video_job(
        job_id="legacy-job",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    _mark_transcript_source_failed(runtime, "legacy-job")

    response = TestClient(create_app()).post("/jobs/legacy-job/retry-failed")

    legacy = LibraryStore(tmp_path).load_jobs()["legacy-job"]
    runtime_record = RuntimeStore(tmp_path / "index.sqlite3").load_video_job("legacy-job")
    transcript = [
        stage
        for stage in RuntimeStore(tmp_path / "index.sqlite3").list_stage_plan("legacy-job")
        if stage.stage_name == "transcript_source"
    ][0]
    assert response.status_code == 200
    assert response.json()["status"] == "queued"
    assert response.json()["current_stage"] == "transcript_source"
    assert runtime_record.status == "queued"
    assert runtime_record.current_stage == "transcript_source"
    assert transcript.status == "pending"
    assert transcript.attempt == 0
    assert transcript.error is None
    assert transcript.warning is None


def test_channel_api_returns_progress_videos_cancel_and_retry(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from yt_sub_hub.api import create_app

    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))

    def fake_fetch(channel_url, *, limit=None, settings=None):
        from yt_sub_hub.channel import ChannelFetchResult, ChannelVideo

        return ChannelFetchResult(
            channel_id="UCexample123",
            channel_url="https://www.youtube.com/channel/UCexample123",
            uploads_playlist_id="UUexample123",
            title="Example Lab",
            videos=[
                ChannelVideo("abc123def45", "https://www.youtube.com/watch?v=abc123def45", "First"),
                ChannelVideo("xyz123def45", "https://www.youtube.com/watch?v=xyz123def45", "Second"),
            ],
        )

    monkeypatch.setattr("yt_sub_hub.channel.fetch_channel_listing", fake_fetch)
    client = TestClient(create_app())

    created = client.post("/channels/jobs", json={"channel_url": "https://www.youtube.com/@ExampleLab", "limit": 2, "quality_mode": "draft"})
    channel_job_id = created.json()["channel_job"]["id"]
    videos = client.get(f"/channels/jobs/{channel_job_id}/videos")
    cancelled = client.post(f"/channels/jobs/{channel_job_id}/cancel")
    retried = client.post(f"/channels/jobs/{channel_job_id}/retry-failed")

    assert created.status_code == 200
    assert created.json()["total_videos"] == 2
    assert videos.status_code == 200
    assert len(videos.json()["videos"]) == 2
    assert cancelled.status_code == 200
    assert cancelled.json()["counts"]["skipped"] == 2
    assert retried.status_code == 200
    assert retried.json()["counts"]["queued"] == 2
    assert RuntimeStore(tmp_path / "index.sqlite3").list_runnable_stages()


def test_api_retry_failed_from_stage_requeues_terminal_job_from_analysis_ru_review(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from yt_sub_hub.api import create_app

    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    LibraryStore(tmp_path).save_job(
        JobRecord(
            job_id="legacy-job",
            url="https://youtu.be/abc123def45",
            video_id="abc123def45",
            status="completed",
            stage="quality_gate",
        )
    )
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    runtime.create_video_job(
        job_id="legacy-job",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    runtime.ensure_stage_plan("legacy-job", enable_deep_graph_llm=False)
    now = time.time()
    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute(
            "UPDATE stage_plans SET status = 'completed', attempt = 1, completed_at = ?, updated_at = ? WHERE video_job_id = ?",
            (now, now, "legacy-job"),
        )
        conn.execute(
            "UPDATE video_jobs SET status = 'completed', current_stage = 'quality_gate', completed_at = ?, updated_at = ? WHERE id = ?",
            (now, now, "legacy-job"),
        )

    response = TestClient(create_app()).post("/jobs/legacy-job/retry-failed?from_stage=analysis_ru_review")

    legacy = LibraryStore(tmp_path).load_jobs()["legacy-job"]
    runtime_record = RuntimeStore(tmp_path / "index.sqlite3").load_video_job("legacy-job")
    review = [
        stage
        for stage in RuntimeStore(tmp_path / "index.sqlite3").list_stage_plan("legacy-job")
        if stage.stage_name == "analysis_ru_review"
    ][0]
    assert response.status_code == 200
    assert response.json()["status"] == "queued"
    assert response.json()["current_stage"] == "analysis_ru_review"
    assert runtime_record.status == "queued"
    assert review.status == "pending"
    assert review.retry_after is None
    assert legacy.status == "queued"
    assert legacy.stage == "queued"
    assert legacy.error is None


def test_recovery_api_endpoints_return_updated_dashboard_and_errors(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from yt_sub_hub.api import create_app

    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        job_id="legacy-job",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    now = time.time()
    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute(
            "UPDATE stage_plans SET status = 'failed', attempt = max_attempts, error = ?, completed_at = ?, updated_at = ? WHERE video_job_id = ? AND stage_name = 'transcript_source'",
            ("captions unavailable", now, now, job.id),
        )
        conn.execute("UPDATE video_jobs SET status = 'failed', current_stage = 'transcript_source', error = ? WHERE id = ?", ("captions unavailable", job.id))

    client = TestClient(create_app())
    dashboard = client.get(f"/jobs/{job.id}/recovery")
    retried = client.post(f"/jobs/{job.id}/recovery/retry-stage", json={"stage_name": "transcript_source"})
    invalid_stage = client.post(f"/jobs/{job.id}/recovery/retry-stage", json={"stage_name": "not-a-stage"})
    empty_chunks = client.post(f"/jobs/{job.id}/recovery/retry-failed-chunks", json={"group": "analysis"})
    missing = client.get("/jobs/missing/recovery")

    assert dashboard.status_code == 200
    assert dashboard.json()["failed_stages"][0]["stage_name"] == "transcript_source"
    assert retried.status_code == 200
    assert retried.json()["status"] == "queued"
    assert retried.json()["failed_stages"] == []
    assert invalid_stage.status_code == 400
    assert empty_chunks.status_code == 409
    assert missing.status_code == 404


def test_recovery_api_rejects_non_applicable_terminal_actions(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from yt_sub_hub.api import create_app

    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        job_id="legacy-job",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    runtime.complete_video_job(job.id)

    client = TestClient(create_app())

    assert client.post(f"/jobs/{job.id}/recovery/cancel").status_code == 409
    assert client.post(f"/jobs/{job.id}/recovery/resume").status_code == 409
    assert client.post(f"/jobs/{job.id}/recovery/retry-stage", json={"stage_name": "analysis_translation"}).status_code == 409
    assert client.post(f"/jobs/{job.id}/recovery/reset-downstream", json={"from_stage": "analysis_translation"}).status_code == 409


def test_recovery_api_and_cli_payload_recommend_only_applicable_actions(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from yt_sub_hub.api import create_app

    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        job_id="legacy-job",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)

    api_payload = TestClient(create_app()).get(f"/jobs/{job.id}/recovery").json()
    cli_result = CliRunner().invoke(cli.app, ["recovery", job.id, "--json"])
    cli_payload = json.loads(cli_result.output)

    for payload in [api_payload, cli_payload]:
        assert payload["actions"]["resume_job"]["applicable"] is False
        assert "resume_job" not in payload["recommended_actions"]
        assert set(payload["recommended_actions"]) <= {
            name for name, action in payload["actions"].items() if action["applicable"]
        }


def test_recovery_cli_commands_emit_json_dashboard(tmp_path, monkeypatch):
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        job_id="legacy-job",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    now = time.time()
    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute(
            "UPDATE stage_plans SET status = 'failed', attempt = max_attempts, error = ?, completed_at = ?, updated_at = ? WHERE video_job_id = ? AND stage_name = 'analysis_translation'",
            ("translation failed", now, now, job.id),
        )
        conn.execute("UPDATE video_jobs SET status = 'failed', current_stage = 'analysis_translation', error = ? WHERE id = ?", ("translation failed", job.id))

    dashboard = CliRunner().invoke(cli.app, ["recovery", job.id, "--json"])
    retried = CliRunner().invoke(cli.app, ["retry-stage", job.id, "analysis_translation", "--json"])

    assert dashboard.exit_code == 0
    assert json.loads(dashboard.output)["failed_stages"][0]["stage_name"] == "analysis_translation"
    assert retried.exit_code == 0
    assert json.loads(retried.output)["status"] == "queued"
