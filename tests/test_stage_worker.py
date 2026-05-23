import json
import sqlite3

import pytest

from yt_sub_hub.artifact_review import analysis_data_hash, transcript_hash
from yt_sub_hub.library import LibraryStore
from yt_sub_hub.models import Analysis, Claim, Concept, JobRecord, NoteSection, Transcript, TranscriptSegment, VideoMetadata
from yt_sub_hub.notes import render_markdown_note
from yt_sub_hub.runtime import RuntimeStore
from yt_sub_hub.stages import PIPELINE_STAGES, retry_delay_seconds, stage_by_name
from yt_sub_hub.worker import StageWorker


def _russian_analysis() -> Analysis:
    return Analysis(
        language="ru",
        cleaned_title="Русский доклад",
        abstract="Русский научный конспект",
        tags=["граф"],
        outline=["Введение"],
        sections=[NoteSection(heading="Введение", summary="Русское резюме", timestamp_seconds=0, bullets=[])],
        concepts=[Concept(name="Граф знаний", definition="Связная структура понятий")],
        claims=[Claim(text="Граф помогает анализу.", evidence="В докладе это объясняется.")],
        limitations=[],
        open_questions=[],
        related_topics=["знания"],
    )


def _write_russian_video(store: LibraryStore, video_id: str = "abc123def45") -> None:
    metadata = VideoMetadata(
        video_id=video_id,
        canonical_url=f"https://www.youtube.com/watch?v={video_id}",
        title="Talk",
        channel="Channel",
    )
    source = Transcript(
        video_id=video_id,
        source="youtube",
        language="en",
        segments=[TranscriptSegment(0, 1, "English text")],
    )
    translated = Transcript(
        video_id=video_id,
        source="youtube",
        language="ru",
        segments=[TranscriptSegment(0, 1, "Русский текст", raw_text="English text")],
    )
    analysis = _russian_analysis()
    store.save_video_artifacts(metadata, source, source, analysis, "# Русский конспект\n\nТекст.")
    store.save_translated_transcript(video_id, translated)
    store.save_analysis_artifact(video_id, analysis, variant="ru", compatibility=True)
    analysis_data = store.load_analysis(video_id, variant="ru")
    note = render_markdown_note(
        metadata,
        analysis,
        provenance={
            "analysis_hash": analysis_data_hash(analysis_data),
            "transcript_hash": transcript_hash(translated),
            "generated_from_analysis": "analysis.ru.json",
        },
    )
    store.save_note_artifact(video_id, note)
    store.save_graph_extraction(
        video_id,
        {
            "schema_version": "graph-extraction-v1",
            "video_id": video_id,
            "generated_from": {
                "transcript": "transcript.ru.json",
                "analysis": "analysis.ru.json",
                "language": "ru",
                "analysis_hash": analysis_data_hash(analysis_data),
                "transcript_hash": transcript_hash(translated),
            },
            "nodes": [
                {"id": f"video:{video_id}", "label": "Video"},
                {"id": f"segment:{video_id}:0", "label": "SourceSegment"},
                {"id": "concept:graf-znanii", "label": "Concept", "name": "Граф знаний"},
            ],
            "relationships": [
                {
                    "id": "rel1",
                    "from": f"video:{video_id}",
                    "to": "concept:graf-znanii",
                    "type": "MENTIONS",
                    "source_video_id": video_id,
                }
            ],
            "quality_report": {},
        },
    )


def _stage_rows(runtime: RuntimeStore, job_id: str) -> list[tuple[str, str, str | None, str | None]]:
    with sqlite3.connect(runtime.db_path) as conn:
        return conn.execute(
            """
            SELECT stage_name, status, error_message, warning
            FROM stage_runs
            WHERE video_job_id = ?
            ORDER BY started_at, id
            """,
            (job_id,),
        ).fetchall()


def test_pipeline_stage_catalog_has_required_stage_order_and_resource_limits():
    names = [stage.name for stage in PIPELINE_STAGES]

    assert names == [
        "metadata",
        "transcript_source",
        "audio_download",
        "whisper_transcription",
        "transcript_translation_expand",
        "transcript_translation_merge",
        "scientific_analysis",
        "analysis_reduce",
        "analysis_translation",
        "analysis_ru_review",
        "markdown_render",
        "deterministic_graph",
        "deep_graph_llm",
        "visual_frame_analysis",
        "ru_artifact_review",
        "neo4j_sync",
        "quality_gate",
    ]
    assert stage_by_name("whisper_transcription").resource_lock == "whisper_cpu"
    assert stage_by_name("transcript_translation_chunk_000").resource_lock == "lm_studio_translation"
    assert stage_by_name("analysis_chunk_000").resource_lock == "lm_studio_analysis"
    assert stage_by_name("analysis_reduce").resource_lock == "local_validation"
    assert stage_by_name("analysis_ru_review").resource_lock == "local_validation"
    assert stage_by_name("analysis_ru_review").max_attempts == 1
    assert stage_by_name("analysis_ru_review").hard_failure is True
    assert stage_by_name("ru_artifact_review").resource_lock == "critic_review"
    assert stage_by_name("ru_artifact_review").max_attempts == 1
    assert stage_by_name("ru_artifact_review").hard_failure is True
    assert stage_by_name("lm_studio_analysis").name == "scientific_analysis"
    assert stage_by_name("deep_graph_llm").resource_lock == "deep_graph_llm"
    assert stage_by_name("neo4j_sync").resource_lock == "neo4j_sync"


def test_stage_worker_uses_accelerated_lock_for_mps_whisper_backend(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    runtime.acquire_lock("whisper_accelerated", owner="worker-1", lease_seconds=60)
    processed = []

    result = StageWorker(
        runtime,
        owner="worker-2",
        quality_gate_enabled=False,
        whisper_backend="openai-whisper-mps",
    ).process_next(
        lambda job: processed.append(job.video_id),
        stage_name="whisper_transcription",
    )

    assert result is None
    assert processed == []
    assert runtime.load_video_job(job.id).status == "queued"


def test_stage_worker_uses_cpu_lock_for_default_whisper_backend(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    runtime.acquire_lock("whisper_cpu", owner="worker-1", lease_seconds=60)
    processed = []

    result = StageWorker(
        runtime,
        owner="worker-2",
        quality_gate_enabled=False,
        whisper_backend="openai-whisper-cpu",
    ).process_next(
        lambda job: processed.append(job.video_id),
        stage_name="whisper_transcription",
    )

    assert result is None
    assert processed == []
    assert runtime.load_video_job(job.id).status == "queued"


def test_stage_worker_video_processing_has_no_implicit_whisper_lock(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    runtime.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    runtime.acquire_lock("whisper_cpu", owner="worker-1", lease_seconds=60)
    runtime.acquire_lock("whisper_accelerated", owner="worker-1", lease_seconds=60)
    processed = []

    result = StageWorker(
        runtime,
        owner="worker-2",
        quality_gate_enabled=False,
        whisper_backend="openai-whisper-mps",
    ).process_next(lambda job: processed.append(job.video_id))

    assert result is not None
    assert processed == ["abc123def45"]
    assert result.status == "completed"


def test_retry_delay_uses_bounded_exponential_backoff():
    assert retry_delay_seconds(1, base_seconds=10, max_seconds=120) == 10
    assert retry_delay_seconds(2, base_seconds=10, max_seconds=120) == 20
    assert retry_delay_seconds(5, base_seconds=10, max_seconds=120) == 120


def test_runtime_store_atomically_commits_json_artifact_and_records_manifest(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    stage = runtime.start_stage_run(video_job_id=job.id, stage_name="metadata")

    artifact = runtime.commit_json_artifact(
        stage_run_id=stage.id,
        video_id="abc123def45",
        artifact_type="metadata",
        path=tmp_path / "videos" / "abc123def45" / "metadata.json",
        data={"video_id": "abc123def45"},
        provenance={"stage": "metadata"},
    )

    assert json.loads(open(artifact.path, encoding="utf-8").read()) == {"video_id": "abc123def45"}
    assert not list((tmp_path / "videos" / "abc123def45").glob("*.tmp"))
    assert runtime.list_stage_artifacts(video_id="abc123def45")[0].id == artifact.id


def test_stage_worker_claims_processes_and_completes_one_job(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    runtime.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    processed = []

    def processor(job):
        processed.append(job.video_id)
        _write_russian_video(LibraryStore(tmp_path), job.video_id)

    result = StageWorker(runtime, owner="worker-1", library_dir=tmp_path).process_next(processor)

    assert result is not None
    assert processed == ["abc123def45"]
    assert runtime.load_video_job(result.id).status == "completed"
    assert _stage_rows(runtime, result.id) == [
        ("video_processing", "completed", None, None),
        ("quality_gate", "completed", None, None),
    ]


def test_stage_worker_success_syncs_legacy_completed_state(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        job_id="legacy-job",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    LibraryStore(tmp_path).save_job(
        JobRecord(
            job_id=job.id,
            url=job.canonical_url,
            video_id=job.video_id,
            status="queued",
            stage="queued",
        )
    )

    result = StageWorker(runtime, owner="worker-1", library_dir=tmp_path).process_next(
        lambda job: _write_russian_video(LibraryStore(tmp_path), job.video_id)
    )

    legacy = LibraryStore(tmp_path).load_jobs()[job.id]
    assert result is not None
    assert result.status == "completed"
    assert legacy.status == "completed"
    assert legacy.note_path == str(LibraryStore(tmp_path).note_path(job.video_id))


def test_stage_worker_quality_gate_hard_failure_fails_video_job(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    runtime.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )

    def processor(job):
        _write_russian_video(LibraryStore(tmp_path), job.video_id)
        LibraryStore(tmp_path).artifact_paths(job.video_id).analysis_ru.unlink()

    result = StageWorker(runtime, owner="worker-1", library_dir=tmp_path).process_next(processor)

    assert result is not None
    assert result.status == "failed"
    assert "analysis.ru.json must exist" in (result.error or "")
    assert _stage_rows(runtime, result.id)[-1][0:2] == ("quality_gate", "failed")


def test_stage_worker_completes_draft_job_without_quality_gate(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    runtime.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
        quality_mode="draft",
    )

    result = StageWorker(runtime, owner="worker-1", library_dir=tmp_path).process_next(lambda job: None)

    assert result is not None
    assert result.status == "completed_with_warning"
    assert result.warning == "draft mode: not production validated"
    assert _stage_rows(runtime, result.id) == [
        ("video_processing", "completed", None, None),
    ]


def test_stage_worker_processor_exception_syncs_legacy_failed_state(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        job_id="legacy-job",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    LibraryStore(tmp_path).save_job(
        JobRecord(
            job_id=job.id,
            url=job.canonical_url,
            video_id=job.video_id,
            status="queued",
            stage="queued",
        )
    )

    result = StageWorker(runtime, owner="worker-1", library_dir=tmp_path).process_next(
        lambda job: (_ for _ in ()).throw(RuntimeError("processor failed"))
    )

    legacy = LibraryStore(tmp_path).load_jobs()[job.id]
    assert result is not None
    assert result.status == "failed"
    assert legacy.status == "failed"
    assert legacy.stage == "failed"
    assert legacy.error == "processor failed"


def test_stage_worker_quality_gate_hard_failure_syncs_legacy_failed_state(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        job_id="legacy-job",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    LibraryStore(tmp_path).save_job(
        JobRecord(
            job_id=job.id,
            url=job.canonical_url,
            video_id=job.video_id,
            status="queued",
            stage="queued",
        )
    )

    def processor(job):
        _write_russian_video(LibraryStore(tmp_path), job.video_id)
        LibraryStore(tmp_path).artifact_paths(job.video_id).analysis_ru.unlink()

    result = StageWorker(runtime, owner="worker-1", library_dir=tmp_path).process_next(processor)

    legacy = LibraryStore(tmp_path).load_jobs()[job.id]
    assert result is not None
    assert result.status == "failed"
    assert legacy.status == "failed"
    assert legacy.stage == "failed"
    assert "analysis.ru.json must exist" in (legacy.error or "")


def test_stage_worker_ru_artifact_review_blocks_small_stale_graph(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    runtime.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )

    def processor(job):
        store = LibraryStore(tmp_path)
        _write_russian_video(store, job.video_id)
        extraction = store.load_graph_extraction(job.video_id)
        extraction["nodes"] = [{"id": f"video:{job.video_id}", "label": "Video"}]
        extraction["relationships"] = []
        store.save_graph_extraction(job.video_id, extraction)

    result = StageWorker(runtime, owner="worker-1", library_dir=tmp_path).process_next(processor)

    assert result is not None
    assert result.status == "failed"
    assert "graph Concept nodes do not cover enough concepts" in (result.error or "")
    assert _stage_rows(runtime, result.id)[-1][0:2] == ("quality_gate", "failed")


def test_stage_worker_ru_artifact_review_failure_syncs_legacy_failed_state(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        job_id="legacy-job",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    LibraryStore(tmp_path).save_job(
        JobRecord(
            job_id=job.id,
            url=job.canonical_url,
            video_id=job.video_id,
            status="queued",
            stage="queued",
        )
    )

    def processor(job):
        store = LibraryStore(tmp_path)
        _write_russian_video(store, job.video_id)
        extraction = store.load_graph_extraction(job.video_id)
        extraction["nodes"] = [{"id": f"video:{job.video_id}", "label": "Video"}]
        extraction["relationships"] = []
        store.save_graph_extraction(job.video_id, extraction)

    result = StageWorker(runtime, owner="worker-1", library_dir=tmp_path).process_next(processor)

    legacy = LibraryStore(tmp_path).load_jobs()[job.id]
    assert result is not None
    assert result.status == "failed"
    assert legacy.status == "failed"
    assert "graph Concept nodes do not cover enough concepts" in (legacy.error or "")


def test_stage_worker_does_not_run_quality_gate_when_processor_returns_skipped(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    runtime.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )

    def processor(job):
        return JobRecord(
            job_id=job.id,
            url=job.canonical_url,
            video_id=job.video_id,
            status="skipped",
            stage="cancelled",
            warning="cancel requested",
        )

    result = StageWorker(runtime, owner="worker-1", library_dir=tmp_path).process_next(processor)

    assert result is not None
    assert result.status == "skipped"
    assert _stage_rows(runtime, result.id) == [("video_processing", "completed_with_warning", None, "cancel requested")]


def test_stage_worker_skip_syncs_legacy_cancelled_state(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        job_id="legacy-job",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    LibraryStore(tmp_path).save_job(
        JobRecord(
            job_id=job.id,
            url=job.canonical_url,
            video_id=job.video_id,
            status="queued",
            stage="queued",
        )
    )

    def processor(job):
        return JobRecord(
            job_id=job.id,
            url=job.canonical_url,
            video_id=job.video_id,
            status="skipped",
            stage="cancelled",
            warning="cancel requested",
        )

    result = StageWorker(runtime, owner="worker-1", library_dir=tmp_path).process_next(processor)

    legacy = LibraryStore(tmp_path).load_jobs()[job.id]
    assert result is not None
    assert result.status == "skipped"
    assert legacy.status == "skipped"
    assert legacy.stage == "cancelled"
    assert legacy.warning == "cancel requested"


def test_stage_worker_uses_stage_resource_lock_and_requeues_when_unavailable(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    runtime.acquire_lock("lm_studio_analysis", owner="worker-1", lease_seconds=60)
    processed = []

    result = StageWorker(runtime, owner="worker-2", quality_gate_enabled=False).process_next(
        lambda job: processed.append(job.video_id),
        stage_name="scientific_analysis",
    )

    assert result is None
    assert processed == []
    requeued = runtime.load_video_job(job.id)
    assert requeued.status == "queued"
    assert requeued.lease_owner is None
    assert requeued.lease_until is None


def test_stage_worker_releases_resource_lock_after_success(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    runtime.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )

    result = StageWorker(runtime, owner="worker-1", quality_gate_enabled=False).process_next(
        lambda job: None,
        resource_lock="lm_studio_analysis",
    )

    assert result is not None
    assert runtime.acquire_lock("lm_studio_analysis", owner="worker-2", lease_seconds=60) is True


def test_stage_worker_releases_resource_lock_after_processor_exception(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    runtime.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )

    def processor(job):
        raise RuntimeError("temporary failure")

    result = StageWorker(runtime, owner="worker-1", quality_gate_enabled=False).process_next(
        processor,
        resource_lock="lm_studio_analysis",
    )

    assert result is not None
    assert result.status == "failed"
    assert runtime.acquire_lock("lm_studio_analysis", owner="worker-2", lease_seconds=60) is True


def test_stage_worker_records_actual_stage_name(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )

    StageWorker(runtime, owner="worker-1", quality_gate_enabled=False).process_next(lambda job: None, stage_name="metadata")

    with sqlite3.connect(runtime.db_path) as conn:
        stages = conn.execute("SELECT stage_name FROM stage_runs WHERE video_job_id = ?", (job.id,)).fetchall()
    assert stages == [("metadata",)]


def test_stage_worker_unknown_stage_name_fails_before_claiming(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )

    with pytest.raises(KeyError, match="Unknown pipeline stage: not_a_stage"):
        StageWorker(runtime, owner="worker-1", quality_gate_enabled=False).process_next(lambda job: None, stage_name="not_a_stage")

    assert runtime.load_video_job(job.id).status == "queued"


def test_stage_worker_reloads_cancellation_before_processor(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    processed = []
    original_start_stage_run = runtime.start_stage_run

    def start_stage_and_cancel(**kwargs):
        stage = original_start_stage_run(**kwargs)
        runtime.request_cancel(job.id)
        return stage

    runtime.start_stage_run = start_stage_and_cancel

    result = StageWorker(runtime, owner="worker-1", library_dir=tmp_path).process_next(lambda job: processed.append(job.id))

    assert processed == []
    assert result is not None
    assert result.status == "skipped"
    assert runtime.load_video_job(job.id).status == "skipped"
    assert [row[0] for row in _stage_rows(runtime, job.id)] == ["video_processing"]


def test_stage_worker_marks_failed_job_and_schedules_retry_without_hot_loop(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    runtime.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )

    def processor(job):
        raise RuntimeError("temporary failure")

    result = StageWorker(runtime, owner="worker-1").process_next(processor, retry_failed=True)

    assert result is not None
    assert runtime.load_video_job(result.id).status == "queued"
    retries = runtime.list_retries()
    assert retries
    assert retries[0].reason == "RuntimeError: temporary failure"
    assert runtime.claim_next_video_job(owner="worker-2", lease_seconds=60) is None
    assert [row[0] for row in _stage_rows(runtime, result.id)] == ["video_processing"]


def test_stage_worker_retry_path_releases_resource_lock(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    runtime.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )

    def processor(job):
        raise RuntimeError("temporary failure")

    result = StageWorker(runtime, owner="worker-1", quality_gate_enabled=False).process_next(
        processor,
        retry_failed=True,
        resource_lock="lm_studio_analysis",
    )

    assert result is not None
    assert result.status == "queued"
    assert runtime.acquire_lock("lm_studio_analysis", owner="worker-2", lease_seconds=60) is True


def test_stage_worker_process_job_does_not_reprocess_already_running_same_owner_job(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )
    assert runtime.claim_video_job(job.id, owner="worker-1", lease_seconds=60) is not None
    processed = []

    result = StageWorker(runtime, owner="worker-1", quality_gate_enabled=False).process_job(
        job.id,
        lambda job: processed.append(job.id),
    )

    assert result is None
    assert processed == []


def test_stage_worker_releases_resource_lock_after_quality_gate_failure(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    runtime.create_video_job(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
    )

    def processor(job):
        _write_russian_video(LibraryStore(tmp_path), job.video_id)
        LibraryStore(tmp_path).artifact_paths(job.video_id).analysis_ru.unlink()

    result = StageWorker(runtime, owner="worker-1", library_dir=tmp_path).process_next(
        processor,
        resource_lock="lm_studio_analysis",
    )

    assert result is not None
    assert result.status == "failed"
    assert runtime.acquire_lock("lm_studio_analysis", owner="worker-2", lease_seconds=60) is True
