import json
import sqlite3
import time
from dataclasses import replace

import pytest

from yt_sub_hub.artifact_review import analysis_data_hash, review_ru_artifacts, transcript_hash
from yt_sub_hub.library import LibraryStore
from yt_sub_hub.llm import parse_analysis_response
from yt_sub_hub.language_quality import validate_ru_analysis, validate_ru_markdown_note
from yt_sub_hub.models import Analysis, Claim, Concept, JobRecord, NoteSection, Transcript, TranscriptBundle, TranscriptSegment, VideoMetadata
from yt_sub_hub.orchestrator import LocalSQLiteOrchestrator
from yt_sub_hub.pipeline import job_id_for_url
from yt_sub_hub.quality import run_quality_gate
from yt_sub_hub.runtime import RuntimeStore
from yt_sub_hub.stage_pipeline import StageContext, StagePipeline, StageResult
from yt_sub_hub.transcripts import TranscriptResolutionError
from yt_sub_hub.worker import StageWorker
from yt_sub_hub.notes import render_markdown_note
from test_stage_worker import _write_russian_video
from test_youtube_discovery import FakeDiscoveryProvider, _write_core_and_research


def _job(runtime: RuntimeStore, video_id: str = "abc123def45", acceptance_mode: str = "data_api_only"):
    return runtime.create_video_job(
        job_id=job_id_for_url(f"https://www.youtube.com/watch?v={video_id}"),
        video_id=video_id,
        canonical_url=f"https://www.youtube.com/watch?v={video_id}",
        acceptance_mode=acceptance_mode,
    )


def _plan_statuses(runtime: RuntimeStore, job_id: str) -> list[tuple[str, str]]:
    return [(stage.stage_name, stage.status) for stage in runtime.list_stage_plan(job_id)]


def _stage(runtime: RuntimeStore, job_id: str, stage_name: str):
    return [stage for stage in runtime.list_stage_plan(job_id) if stage.stage_name == stage_name][0]


def _mark_stage_completed(runtime: RuntimeStore, job_id: str, stage_name: str) -> None:
    now = time.time()
    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute(
            """
            UPDATE stage_plans
            SET status = 'completed', attempt = 1, completed_at = ?, updated_at = ?
            WHERE video_job_id = ? AND stage_name = ?
            """,
            (now, now, job_id, stage_name),
        )


def _mark_stage_completed_with_warning(runtime: RuntimeStore, job_id: str, stage_name: str, warning: str) -> None:
    now = time.time()
    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute(
            """
            UPDATE stage_plans
            SET status = 'completed_with_warning', warning = ?, attempt = 1, completed_at = ?, updated_at = ?
            WHERE video_job_id = ? AND stage_name = ?
            """,
            (warning, now, now, job_id, stage_name),
        )


def _write_audio(tmp_path, video_id: str, content: bytes = b"audio") -> None:
    audio_dir = tmp_path / "assets" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    (audio_dir / f"{video_id}.mp3").write_bytes(content)


def _whisper_bundle(video_id: str, raw_payload=None) -> TranscriptBundle:
    raw_payload = raw_payload or {
        "language": "en",
        "segments": [{"start": 0.0, "end": 1.0, "text": " Whisper source text "}],
    }
    normalized = Transcript(
        video_id=video_id,
        source="whisper",
        language="en",
        segments=[TranscriptSegment(0.0, 1.0, "Whisper source text", " Whisper source text ")],
    )
    return TranscriptBundle(video_id=video_id, source="whisper", raw_payload=raw_payload, normalized=normalized)


def _patch_whisper_backend(monkeypatch, bundle: TranscriptBundle, captured: dict | None = None) -> None:
    class FakeBackend:
        def transcribe(self, audio_path, *, video_id=""):
            if captured is not None:
                captured["audio_path"] = audio_path
                captured["video_id"] = video_id
            return bundle

    monkeypatch.setattr("yt_sub_hub.stage_pipeline.build_whisper_backend", lambda settings: FakeBackend(), raising=False)
    monkeypatch.setattr(
        "yt_sub_hub.stage_pipeline.WhisperAudioProvider.transcribe_bundle",
        lambda self, video_id: (_ for _ in ()).throw(AssertionError("whisper stage must use the selected backend directly")),
    )


def _source_transcript(video_id: str = "abc123def45") -> Transcript:
    return Transcript(
        video_id=video_id,
        source="whisper",
        language="en",
        segments=[TranscriptSegment(0, 1, "English source text", raw_text="English source text")],
    )


def _ru_transcript(video_id: str = "abc123def45") -> Transcript:
    return Transcript(
        video_id=video_id,
        source="whisper",
        language="ru",
        segments=[TranscriptSegment(0, 1, "Русский перевод текста", raw_text="English source text")],
    )


def _rich_ru_transcript(video_id: str = "abc123def45") -> Transcript:
    return Transcript(
        video_id=video_id,
        source="whisper",
        language="ru",
        segments=[
            TranscriptSegment(0, 8, "В ролике обсуждаются ожидаемые игры и причины интереса игроков к новым проектам."),
            TranscriptSegment(40, 9, "Автор сравнивает боевые системы, исследование мира и развитие персонажей."),
            TranscriptSegment(90, 10, "Отдельно разбираются сюжетные завязки, атмосфера приключений и выбор фракций."),
            TranscriptSegment(150, 8, "Зрителям показывают трейлеры, даты выхода и платформы для будущих релизов."),
            TranscriptSegment(220, 9, "В середине выпуска перечисляются проекты с открытым миром и тактическими боями."),
            TranscriptSegment(310, 8, "Финальная часть подводит итог ожиданиям и объясняет, какие игры выглядят наиболее многообещающими."),
        ],
    )


def _long_source_transcript(video_id: str = "abc123def45", count: int = 5) -> Transcript:
    return Transcript(
        video_id=video_id,
        source="whisper",
        language="en",
        segments=[
            TranscriptSegment(index * 10, 5, f"English source segment {index} about knowledge graphs and evidence.")
            for index in range(count)
        ],
    )


def _analysis(language: str = "en", *, russian: bool = False) -> Analysis:
    if russian:
        return Analysis(
            language=language,
            cleaned_title="Русский доклад",
            abstract="Русский научный конспект",
            tags=["граф"],
            outline=["Введение"],
            sections=[NoteSection("Введение", "Русское резюме", 0, [])],
            concepts=[Concept("Граф знаний", "Связная структура понятий")],
            claims=[Claim("Граф помогает анализу.", "В докладе это объясняется.")],
            limitations=[],
            open_questions=[],
            related_topics=["знания"],
        )
    return Analysis(
        language=language,
        cleaned_title="English talk",
        abstract="English scientific summary",
        tags=["graph"],
        outline=["Intro"],
        sections=[NoteSection("Intro", "English summary", 0, [])],
        concepts=[Concept("Knowledge graph", "Connected concept structure")],
        claims=[Claim("Graphs help analysis.", "The talk explains it.")],
        limitations=[],
        open_questions=[],
        related_topics=["knowledge"],
    )


def _write_metadata_and_source_transcript(store: LibraryStore, video_id: str = "abc123def45") -> None:
    paths = store.artifact_paths(video_id)
    paths.video_dir.mkdir(parents=True, exist_ok=True)
    store._write_json(
        paths.metadata,
        VideoMetadata(
            video_id=video_id,
            canonical_url=f"https://www.youtube.com/watch?v={video_id}",
            title="Talk",
            channel="Channel",
        ).to_dict(),
    )
    store._write_json(paths.normalized_transcript, _source_transcript(video_id).to_dict())


def test_stage_plan_creation_and_dependency_scheduling(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)

    plan = runtime.ensure_stage_plan(job.id, whisper_backend="openai-whisper-cpu", enable_deep_graph_llm=False)
    runnable = runtime.list_runnable_stages()

    assert plan[0].stage_name == "metadata"
    assert plan[-1].stage_name == "quality_gate"
    assert [stage.stage_name for stage in runnable] == ["metadata"]

    claim = runtime.claim_next_stage(owner="worker-1", lease_seconds=60)
    assert claim is not None
    runtime.complete_stage(claim.stage_run.id)

    assert [stage.stage_name for stage in runtime.list_runnable_stages()] == ["transcript_source"]


def test_local_stage_orchestrator_processes_youtube_discovery_stages_without_monolith(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "yt_sub_hub.youtube_discovery.run_youtube_discovery",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("staged path must not call monolithic discovery")),
    )
    _write_core_and_research(tmp_path)
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.queue_youtube_discovery("srcVideo123", limit=5, max_quota_units=900)
    pipeline = StagePipeline(library_dir=tmp_path, discovery_provider=FakeDiscoveryProvider())

    LocalSQLiteOrchestrator(runtime, pipeline=pipeline, owner_prefix="discovery-test").run_loop(concurrency=2)

    stages = {stage.stage_name: stage for stage in runtime.list_stage_plan(job.id)}
    assert stages["youtube_discovery_report"].status == "completed"
    assert runtime.load_video_job(job.id).job_kind == "youtube_discovery"
    assert runtime.load_video_job(job.id).status in {"completed", "completed_with_warning"}
    assert not any(
        stage_name in stages
        for stage_name in {
            "metadata",
            "transcript_source",
            "scientific_analysis",
            "analysis_translation",
            "quality_gate",
        }
    )
    discovery_dir = tmp_path / "videos" / "srcVideo123" / "youtube_discovery"
    assert (discovery_dir / "seeds.json").exists()
    assert (discovery_dir / "queries.json").exists()
    assert list((discovery_dir / "query_results").glob("*.json"))
    assert (discovery_dir / "video_candidates.json").exists()
    assert (discovery_dir / "channel_candidates.json").exists()
    assert (discovery_dir / "review.json").exists()
    assert (discovery_dir / "graph.json").exists()
    assert (discovery_dir / "discovery_report.md").exists()
    assert [(queued.video_id, queued.job_kind) for queued in runtime.list_video_jobs()] == [("srcVideo123", "youtube_discovery")]


def test_youtube_discovery_failed_review_blocks_graph_and_report(tmp_path, monkeypatch):
    _write_core_and_research(tmp_path)
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.queue_youtube_discovery("srcVideo123", limit=5, max_quota_units=900)
    pipeline = StagePipeline(library_dir=tmp_path, discovery_provider=FakeDiscoveryProvider())

    def failed_review(**kwargs):
        return {
            "schema_version": "youtube-discovery-v1",
            "video_id": kwargs["video_id"],
            "research_run_id": kwargs["research_run_id"],
            "discovery_run_id": kwargs["discovery_run_id"],
            "status": "failed",
            "issues": [{"severity": "hard", "code": "forced", "candidate_id": "", "message": "forced"}],
            "accepted_candidate_ids": [],
            "rejected_candidate_ids": [],
            "needs_review_candidate_ids": [],
        }

    monkeypatch.setattr("yt_sub_hub.youtube_discovery.review_youtube_discovery", failed_review)

    LocalSQLiteOrchestrator(runtime, pipeline=pipeline, owner_prefix="discovery-test").run_loop(concurrency=2)

    stages = {stage.stage_name: stage for stage in runtime.list_stage_plan(job.id)}
    assert stages["youtube_discovery_review"].status == "failed"
    assert stages["youtube_discovery_graph"].status == "pending"
    assert stages["youtube_discovery_report"].status == "pending"
    assert runtime.load_video_job(job.id).status == "failed"


def test_stage_worker_marks_discovery_job_completed_with_warning_on_review_warning(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.queue_youtube_discovery("srcVideo123", limit=5, max_quota_units=900)
    discovery_stages = [
        "youtube_discovery_seed_extract",
        "youtube_discovery_query_expand",
        "youtube_discovery_candidate_fetch",
        "youtube_discovery_candidate_score",
        "youtube_discovery_review",
        "youtube_discovery_graph",
        "youtube_discovery_report",
    ]
    pipeline = StagePipeline(
        library_dir=tmp_path,
        handlers={
            stage_name: (
                (lambda ctx: StageResult.completed(warning="review warning"))
                if stage_name == "youtube_discovery_review"
                else (lambda ctx: StageResult.completed())
            )
            for stage_name in discovery_stages
        },
    )

    LocalSQLiteOrchestrator(runtime, pipeline=pipeline, owner_prefix="discovery-warning-test").run_loop(concurrency=1)

    loaded = runtime.load_video_job(job.id)
    stages = {stage.stage_name: stage for stage in runtime.list_stage_plan(job.id)}
    assert loaded.job_kind == "youtube_discovery"
    assert loaded.status == "completed_with_warning"
    assert stages["youtube_discovery_review"].status == "completed_with_warning"
    assert "review warning" in (loaded.warning or "")


def test_stage_claim_skips_busy_resource_and_claims_other_runnable_stage(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    first = _job(runtime, "aaa111aaa11")
    second = _job(runtime, "bbb222bbb22")
    for job in [first, second]:
        runtime.ensure_stage_plan(job.id, whisper_backend="openai-whisper-cpu", enable_deep_graph_llm=False)
        metadata = runtime.claim_next_stage(owner=f"metadata-{job.id}", lease_seconds=60)
        assert metadata is not None
        runtime.complete_stage(metadata.stage_run.id)
        transcript = runtime.claim_next_stage(owner=f"transcript-{job.id}", lease_seconds=60)
        assert transcript is not None
        runtime.complete_stage(transcript.stage_run.id)
        audio = runtime.claim_next_stage(owner=f"audio-{job.id}", lease_seconds=60)
        assert audio is not None
        runtime.complete_stage(audio.stage_run.id)

    held = runtime.claim_next_stage(owner="whisper-holder", lease_seconds=60)
    assert held is not None
    assert held.stage.stage_name == "whisper_transcription"

    other = runtime.claim_next_stage(owner="worker-2", lease_seconds=60)

    assert other is None
    runtime.complete_stage(held.stage_run.id)


def test_stage_resume_after_expired_lease_and_duplicate_prevention(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id)

    first = runtime.claim_next_stage(owner="worker-1", lease_seconds=60)
    duplicate = runtime.claim_next_stage(owner="worker-2", lease_seconds=60)
    assert first is not None
    assert duplicate is None

    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute("UPDATE stage_plans SET lease_until = ? WHERE id = ?", (time.time() - 10, first.stage.id))

    runtime.expire_stale_stage_claims()
    resumed = runtime.claim_next_stage(owner="worker-2", lease_seconds=60)

    assert resumed is not None
    assert resumed.stage.stage_name == "metadata"
    assert resumed.stage.attempt == 2


def test_stage_resume_uses_existing_stage_runs_to_avoid_duplicate_attempt(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id)
    first = runtime.claim_next_stage(owner="worker-1", lease_seconds=60)
    assert first is not None
    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute("UPDATE stage_plans SET status = 'pending', lease_owner = NULL, lease_until = NULL, resource_slot = NULL WHERE id = ?", (first.stage.id,))
        conn.execute("DELETE FROM locks")

    resumed = runtime.claim_next_stage(owner="worker-2", lease_seconds=60)

    assert resumed is not None
    assert resumed.stage.attempt == 2


def test_stage_claim_ignores_legacy_stage_runs_created_before_stage_plan(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    legacy_gate = runtime.start_stage_run(video_job_id=job.id, stage_name="quality_gate")
    runtime.complete_stage_run(legacy_gate.id)
    runtime.ensure_stage_plan(job.id)
    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute("UPDATE stage_plans SET status = 'completed', completed_at = ? WHERE stage_name != 'quality_gate'", (time.time(),))

    claim = runtime.claim_next_stage(owner="worker-1", lease_seconds=60)

    assert claim is not None
    assert claim.stage.stage_name == "quality_gate"
    assert claim.stage.attempt == 1


def test_stage_worker_retry_is_per_stage_not_per_video(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id)

    pipeline = StagePipeline(
        library_dir=tmp_path,
        handlers={"metadata": lambda ctx: (_ for _ in ()).throw(RuntimeError("temporary failure"))},
    )
    result = StageWorker(runtime, owner="worker-1", library_dir=tmp_path).process_next_stage(pipeline)

    loaded = runtime.load_video_job(job.id)
    plan = runtime.list_stage_plan(job.id)
    assert result is not None
    assert loaded.status == "queued"
    assert plan[0].status == "pending"
    assert plan[0].retry_after is not None
    assert plan[1].status == "pending"


def test_scientific_analysis_stage_uses_existing_analysis_artifact(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    _write_russian_video(LibraryStore(tmp_path), job.video_id)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute("UPDATE stage_plans SET status = 'completed', completed_at = ? WHERE position < (SELECT position FROM stage_plans WHERE stage_name = 'scientific_analysis' AND video_job_id = ?)", (time.time(), job.id))

    class AnalyzerShouldNotRun:
        def analyze(self, metadata, transcript):
            raise AssertionError("analysis should use cache")

    pipeline = StagePipeline(library_dir=tmp_path)
    pipeline.pipeline.analyzer = AnalyzerShouldNotRun()
    result = StageWorker(runtime, owner="worker-1", library_dir=tmp_path).process_next_stage(pipeline)

    assert result is not None
    scientific = [stage for stage in runtime.list_stage_plan(job.id) if stage.stage_name == "scientific_analysis"][0]
    assert scientific.status == "completed_with_warning"
    assert scientific.warning == "analysis cache hit"


def test_data_api_only_transcript_source_skips_requires_media_without_audio_or_whisper(tmp_path, monkeypatch):
    import yt_sub_hub.transcripts as transcripts

    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime, acceptance_mode="data_api_only")
    stage = runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)[1]
    pipeline = StagePipeline(library_dir=tmp_path)
    pipeline.settings = replace(pipeline.settings, media_download_enabled=False, acceptance_mode="data_api_only")

    def fail_captions(self, video_id):
        raise transcripts.TranscriptSourceError("no captions", reason="captions_unavailable")

    def fail_audio(*args, **kwargs):
        raise AssertionError("data_api_only must not invoke media download")

    monkeypatch.setattr("yt_sub_hub.transcripts.YouTubeDataApiCaptionProvider.fetch_bundle", fail_captions)
    monkeypatch.setattr("yt_sub_hub.transcripts.YouTubeTranscriptProvider.fetch_bundle", fail_captions)
    monkeypatch.setattr("yt_sub_hub.transcripts.WhisperAudioProvider._download_audio", fail_audio)

    result = pipeline.run(StageContext(job=job, stage=stage, settings=pipeline.settings, store=pipeline.store, pipeline=pipeline.pipeline))

    assert result.status == "skipped"
    assert result.warning == "skipped_requires_media"


def test_data_api_only_transcript_source_does_not_call_unofficial_provider(tmp_path, monkeypatch):
    import yt_sub_hub.transcripts as transcripts

    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime, acceptance_mode="data_api_only")
    stage = runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)[1]
    pipeline = StagePipeline(library_dir=tmp_path)
    pipeline.settings = replace(
        pipeline.settings,
        youtube_api_token_file="token.json",
        media_download_enabled=False,
        acceptance_mode="data_api_only",
    )

    def fail_data_api(self, video_id):
        raise transcripts.TranscriptSourceError("caption download forbidden", reason="captions_forbidden")

    def unofficial_must_not_run(self, video_id):
        raise AssertionError("data_api_only must not invoke youtube-transcript-api")

    monkeypatch.setattr("yt_sub_hub.transcripts.YouTubeDataApiCaptionProvider.fetch_bundle", fail_data_api)
    monkeypatch.setattr("yt_sub_hub.transcripts.YouTubeTranscriptProvider.fetch_bundle", unofficial_must_not_run)

    result = pipeline.run(StageContext(job=job, stage=stage, settings=pipeline.settings, store=pipeline.store, pipeline=pipeline.pipeline))

    assert result.status == "skipped"
    assert result.warning == "skipped_requires_media"


def test_full_processing_transcript_source_uses_unofficial_before_whisper(tmp_path, monkeypatch):
    import yt_sub_hub.transcripts as transcripts

    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime, acceptance_mode="full_processing")
    stage = runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)[1]
    pipeline = StagePipeline(library_dir=tmp_path)
    pipeline.settings = replace(
        pipeline.settings,
        youtube_api_token_file="token.json",
        media_download_enabled=True,
        acceptance_mode="full_processing",
    )

    def fail_data_api(self, video_id):
        raise transcripts.TranscriptSourceError("no official captions", reason="captions_unavailable")

    def succeed_unofficial(self, video_id):
        transcript = Transcript(video_id=video_id, source="youtube", segments=[TranscriptSegment(0, 1, "Unofficial")])
        return [{"start": 0, "duration": 1, "text": "Unofficial"}], transcript

    def fail_audio(*args, **kwargs):
        raise AssertionError("unofficial transcript should avoid media download")

    monkeypatch.setattr("yt_sub_hub.transcripts.YouTubeDataApiCaptionProvider.fetch_bundle", fail_data_api)
    monkeypatch.setattr("yt_sub_hub.transcripts.YouTubeTranscriptProvider.fetch_bundle", succeed_unofficial)
    monkeypatch.setattr("yt_sub_hub.transcripts.WhisperAudioProvider._download_audio", fail_audio)
    pipeline.store.artifact_paths(job.video_id).video_dir.mkdir(parents=True, exist_ok=True)

    result = pipeline.run(StageContext(job=job, stage=stage, settings=pipeline.settings, store=pipeline.store, pipeline=pipeline.pipeline))

    assert result.status == "completed"
    assert pipeline.store.load_normalized_transcript(job.video_id).source == "youtube"


@pytest.mark.parametrize(
    ("reason", "message"),
    [
        ("auth_error", "invalid token"),
        ("quota_or_rate_limited", "quota exceeded"),
        ("dependency_error", "google client missing"),
        ("unknown_error", "provider bug"),
        ("unavailable", "video unavailable"),
    ],
)
def test_data_api_only_transcript_source_fails_non_media_errors(tmp_path, monkeypatch, reason, message):
    import yt_sub_hub.transcripts as transcripts

    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime, acceptance_mode="data_api_only")
    stage = runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)[1]
    pipeline = StagePipeline(library_dir=tmp_path)
    pipeline.settings = replace(
        pipeline.settings,
        youtube_api_token_file="token.json",
        media_download_enabled=False,
        acceptance_mode="data_api_only",
    )

    def fail_data_api(self, video_id):
        raise transcripts.TranscriptSourceError(message, reason=reason)

    monkeypatch.setattr("yt_sub_hub.transcripts.YouTubeDataApiCaptionProvider.fetch_bundle", fail_data_api)

    result = pipeline.run(StageContext(job=job, stage=stage, settings=pipeline.settings, store=pipeline.store, pipeline=pipeline.pipeline))

    assert result.status == "failed"
    assert message in (result.error or "")


def test_data_api_only_transcript_source_fails_raw_provider_exception(tmp_path, monkeypatch):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime, acceptance_mode="data_api_only")
    stage = runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)[1]
    pipeline = StagePipeline(library_dir=tmp_path)
    pipeline.settings = replace(
        pipeline.settings,
        youtube_api_token_file="token.json",
        media_download_enabled=False,
        acceptance_mode="data_api_only",
    )

    def fail_data_api(self, video_id):
        raise RuntimeError("provider bug")

    monkeypatch.setattr("yt_sub_hub.transcripts.YouTubeDataApiCaptionProvider.fetch_bundle", fail_data_api)

    result = pipeline.run(StageContext(job=job, stage=stage, settings=pipeline.settings, store=pipeline.store, pipeline=pipeline.pipeline))

    assert result.status == "failed"
    assert "provider bug" in (result.error or "")


def test_data_api_only_worker_marks_job_skipped_requires_media_and_downstream_skipped(tmp_path, monkeypatch):
    import yt_sub_hub.transcripts as transcripts

    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime, acceptance_mode="data_api_only")
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    _mark_stage_completed(runtime, job.id, "metadata")
    pipeline = StagePipeline(library_dir=tmp_path)
    pipeline.settings = replace(pipeline.settings, media_download_enabled=False, acceptance_mode="data_api_only")

    def fail_captions(self, video_id):
        raise transcripts.TranscriptSourceError("no captions", reason="captions_unavailable")

    monkeypatch.setattr("yt_sub_hub.transcripts.YouTubeDataApiCaptionProvider.fetch_bundle", fail_captions)
    monkeypatch.setattr("yt_sub_hub.transcripts.YouTubeTranscriptProvider.fetch_bundle", fail_captions)

    result = StageWorker(runtime, owner="worker-1", library_dir=tmp_path).process_next_stage(pipeline)

    assert result.status == "skipped"
    assert result.current_stage == "transcript_source"
    assert result.warning == "skipped_requires_media"
    assert _stage(runtime, job.id, "transcript_source").status == "skipped"
    assert _stage(runtime, job.id, "audio_download").status == "skipped"
    assert _stage(runtime, job.id, "whisper_transcription").status == "skipped"


def test_data_api_only_worker_marks_auth_error_failed_not_skipped_requires_media(tmp_path, monkeypatch):
    import yt_sub_hub.transcripts as transcripts

    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime, acceptance_mode="data_api_only")
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    _mark_stage_completed(runtime, job.id, "metadata")
    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute(
            "UPDATE stage_plans SET max_attempts = 1 WHERE video_job_id = ? AND stage_name = 'transcript_source'",
            (job.id,),
        )
    pipeline = StagePipeline(library_dir=tmp_path)
    pipeline.settings = replace(
        pipeline.settings,
        youtube_api_token_file="token.json",
        media_download_enabled=False,
        acceptance_mode="data_api_only",
    )

    def fail_data_api(self, video_id):
        raise transcripts.TranscriptSourceError("invalid token", reason="auth_error")

    monkeypatch.setattr("yt_sub_hub.transcripts.YouTubeDataApiCaptionProvider.fetch_bundle", fail_data_api)

    result = StageWorker(runtime, owner="worker-1", library_dir=tmp_path).process_next_stage(pipeline)

    assert result.status == "failed"
    assert result.current_stage == "transcript_source"
    assert result.warning != "skipped_requires_media"
    assert "invalid token" in (result.error or "")
    assert _stage(runtime, job.id, "transcript_source").status == "failed"


def test_full_processing_transcript_source_needs_whisper_when_captions_unavailable(tmp_path, monkeypatch):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime, acceptance_mode="full_processing")
    stage = runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)[1]
    pipeline = StagePipeline(library_dir=tmp_path)
    pipeline.settings = replace(pipeline.settings, media_download_enabled=True, acceptance_mode="full_processing")

    def fail_captions(self, video_id):
        raise TranscriptResolutionError("no captions")

    monkeypatch.setattr("yt_sub_hub.transcripts.YouTubeDataApiCaptionProvider.fetch_bundle", fail_captions)
    monkeypatch.setattr("yt_sub_hub.transcripts.YouTubeTranscriptProvider.fetch_bundle", fail_captions)

    result = pipeline.run(StageContext(job=job, stage=stage, settings=pipeline.settings, store=pipeline.store, pipeline=pipeline.pipeline))

    assert result.status == "completed"
    assert result.warning == "needs whisper fallback"
    assert result.skip_stages == []


def test_full_processing_transcript_source_fails_fast_when_media_download_disabled(tmp_path, monkeypatch):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime, acceptance_mode="full_processing")
    stage = runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)[1]
    pipeline = StagePipeline(library_dir=tmp_path)
    pipeline.settings = replace(pipeline.settings, media_download_enabled=False, acceptance_mode="full_processing")

    def fail_captions(self, video_id):
        raise TranscriptResolutionError("no captions")

    monkeypatch.setattr("yt_sub_hub.transcripts.YouTubeDataApiCaptionProvider.fetch_bundle", fail_captions)
    monkeypatch.setattr("yt_sub_hub.transcripts.YouTubeTranscriptProvider.fetch_bundle", fail_captions)

    result = pipeline.run(StageContext(job=job, stage=stage, settings=pipeline.settings, store=pipeline.store, pipeline=pipeline.pipeline))

    assert result.status == "failed"
    assert "full_processing requires YT_SUB_HUB_MEDIA_DOWNLOAD=enabled" in (result.error or "")


def test_retry_failed_transcript_source_makes_stage_runnable(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    now = time.time()
    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute(
            "UPDATE stage_plans SET status = 'completed', completed_at = ? WHERE video_job_id = ? AND stage_name = 'metadata'",
            (now, job.id),
        )
        conn.execute(
            """
            UPDATE stage_plans
            SET status = 'failed', attempt = max_attempts, error = ?, completed_at = ?
            WHERE video_job_id = ? AND stage_name = 'transcript_source'
            """,
            ("max attempts exceeded", now, job.id),
        )
        conn.execute("UPDATE video_jobs SET status = 'failed', error = ? WHERE id = ?", ("max attempts exceeded", job.id))

    runtime.reset_failed_stage_plan_for_retry(job.id)

    assert [stage.stage_name for stage in runtime.list_runnable_stages()] == ["transcript_source"]


def test_worker_processes_transcript_source_after_retry_failed_reset(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    now = time.time()
    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute(
            "UPDATE stage_plans SET status = 'completed', completed_at = ? WHERE video_job_id = ? AND stage_name = 'metadata'",
            (now, job.id),
        )
        conn.execute(
            """
            UPDATE stage_plans
            SET status = 'failed', attempt = max_attempts, error = ?, completed_at = ?
            WHERE video_job_id = ? AND stage_name = 'transcript_source'
            """,
            ("max attempts exceeded", now, job.id),
        )
        conn.execute("UPDATE video_jobs SET status = 'failed', error = ? WHERE id = ?", ("max attempts exceeded", job.id))

    runtime.reset_failed_stage_plan_for_retry(job.id)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        handlers={
            "transcript_source": lambda ctx: StageResult.completed(
                warning="transcript cache hit",
                skip_stages=["audio_download", "whisper_transcription"],
            )
        },
    )

    result = StageWorker(runtime, owner="worker-1", library_dir=tmp_path).process_next_stage(pipeline)

    assert result is not None
    assert _stage(runtime, job.id, "transcript_source").status == "completed_with_warning"
    assert _stage(runtime, job.id, "audio_download").status == "skipped"
    assert _stage(runtime, job.id, "whisper_transcription").status == "skipped"


def test_staged_whisper_transcription_writes_transcript_artifacts_immediately(tmp_path, monkeypatch):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    _write_audio(tmp_path, job.video_id)
    bundle = _whisper_bundle(job.video_id)
    captured = {}
    _patch_whisper_backend(monkeypatch, bundle, captured)
    pipeline = StagePipeline(library_dir=tmp_path)

    result = pipeline.run(pipeline.context(job, _stage(runtime, job.id, "whisper_transcription"), runtime=runtime))

    paths = pipeline.store.artifact_paths(job.video_id)
    assert result.status == "completed"
    assert paths.raw_transcript.exists()
    assert paths.normalized_transcript.exists()
    assert json.loads(paths.raw_transcript.read_text(encoding="utf-8")) == bundle.raw_payload
    normalized = json.loads(paths.normalized_transcript.read_text(encoding="utf-8"))
    assert captured == {"audio_path": tmp_path / "assets" / "audio" / f"{job.video_id}.mp3", "video_id": job.video_id}
    assert normalized["source"] == "whisper"
    assert normalized["segments"]


def test_later_scientific_analysis_failure_keeps_whisper_transcript_artifacts(tmp_path, monkeypatch):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    _write_audio(tmp_path, job.video_id)
    bundle = _whisper_bundle(job.video_id)
    _patch_whisper_backend(monkeypatch, bundle)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        handlers={
            "metadata": lambda ctx: StageResult.completed(),
            "transcript_source": lambda ctx: StageResult.completed(warning="needs whisper"),
            "audio_download": lambda ctx: StageResult.completed(),
            "transcript_translation_expand": lambda ctx: StageResult.skipped("translation disabled"),
            "transcript_translation_merge": lambda ctx: StageResult.skipped("translation disabled"),
            "scientific_analysis": lambda ctx: (_ for _ in ()).throw(RuntimeError("lm timeout")),
        },
    )

    for _ in range(7):
        StageWorker(runtime, owner="worker-1", library_dir=tmp_path).process_next_stage(pipeline)
        if _stage(runtime, job.id, "scientific_analysis").attempt:
            break

    paths = pipeline.store.artifact_paths(job.video_id)
    assert _stage(runtime, job.id, "whisper_transcription").status == "completed"
    assert _stage(runtime, job.id, "scientific_analysis").status == "pending"
    assert "lm timeout" in (_stage(runtime, job.id, "scientific_analysis").error or "")
    assert json.loads(paths.raw_transcript.read_text(encoding="utf-8")) == bundle.raw_payload
    assert json.loads(paths.normalized_transcript.read_text(encoding="utf-8"))["segments"]


def test_cached_transcript_skips_staged_audio_download_and_whisper_transcription(tmp_path, monkeypatch):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    plan = runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    bundle = _whisper_bundle(job.video_id)
    paths = LibraryStore(tmp_path).artifact_paths(job.video_id)
    paths.video_dir.mkdir(parents=True, exist_ok=True)
    LibraryStore._write_json(paths.raw_transcript, bundle.raw_payload)
    LibraryStore._write_json(paths.normalized_transcript, bundle.normalized.to_dict())

    def fail_download(self, video_id):
        raise AssertionError("audio download should be skipped")

    def fail_transcribe(self, video_id):
        raise AssertionError("whisper transcription should be skipped")

    monkeypatch.setattr("yt_sub_hub.transcripts.WhisperAudioProvider._download_audio", fail_download)
    monkeypatch.setattr("yt_sub_hub.transcripts.WhisperAudioProvider.transcribe_bundle", fail_transcribe)
    pipeline = StagePipeline(library_dir=tmp_path)

    audio_result = pipeline.run(pipeline.context(job, plan[2], runtime=runtime))
    whisper_result = pipeline.run(pipeline.context(job, plan[3], runtime=runtime))

    assert audio_result.status == "skipped"
    assert audio_result.warning == "transcript already available"
    assert whisper_result.status == "skipped"
    assert whisper_result.warning == "transcript already available"


def test_staged_whisper_transcription_preserves_selected_backend_raw_payload_schema(tmp_path, monkeypatch):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    _write_audio(tmp_path, job.video_id)
    raw_payload = {
        "language": "en",
        "transcription": [
            {
                "offsets": {"from": 1000, "to": 2500},
                "text": " Whisper source text ",
                "speaker": "S1",
            }
        ],
        "backend_meta": {"engine": "whisper-cpp-metal"},
    }
    bundle = _whisper_bundle(job.video_id, raw_payload=raw_payload)
    _patch_whisper_backend(monkeypatch, bundle)
    pipeline = StagePipeline(library_dir=tmp_path)

    result = pipeline.run(pipeline.context(job, _stage(runtime, job.id, "whisper_transcription"), runtime=runtime))

    paths = pipeline.store.artifact_paths(job.video_id)
    assert result.status == "completed"
    assert json.loads(paths.raw_transcript.read_text(encoding="utf-8")) == raw_payload
    assert json.loads(paths.normalized_transcript.read_text(encoding="utf-8")) == bundle.normalized.to_dict()


def test_staged_whisper_transcription_removes_audio_when_keep_audio_is_false(tmp_path, monkeypatch):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    _write_audio(tmp_path, job.video_id)
    audio_path = tmp_path / "assets" / "audio" / f"{job.video_id}.mp3"
    bundle = _whisper_bundle(job.video_id)
    _patch_whisper_backend(monkeypatch, bundle)
    pipeline = StagePipeline(library_dir=tmp_path)
    pipeline.settings = replace(pipeline.settings, keep_audio=False)

    result = pipeline.run(pipeline.context(job, _stage(runtime, job.id, "whisper_transcription"), runtime=runtime))

    paths = pipeline.store.artifact_paths(job.video_id)
    assert result.status == "completed"
    assert not audio_path.exists()
    assert paths.raw_transcript.exists()
    assert paths.normalized_transcript.exists()


def test_staged_whisper_transcription_requires_existing_audio_without_download(tmp_path, monkeypatch):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)

    def fail_download(self, video_id):
        raise AssertionError("whisper_transcription must not download audio")

    monkeypatch.setattr("yt_sub_hub.transcripts.WhisperAudioProvider._download_audio", fail_download)
    pipeline = StagePipeline(library_dir=tmp_path)

    result = pipeline.run(pipeline.context(job, _stage(runtime, job.id, "whisper_transcription"), runtime=runtime))

    assert result.status == "failed"
    assert "audio artifact is required before whisper_transcription" in (result.error or "")
    assert not pipeline.store.artifact_paths(job.video_id).raw_transcript.exists()
    assert not pipeline.store.artifact_paths(job.video_id).normalized_transcript.exists()


def test_staged_transcript_translation_writes_chunk_then_merge_writes_transcript_ru(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        settings=replace(
            StagePipeline(library_dir=tmp_path).settings,
            enable_translation=True,
            target_language="ru",
            translation_chunk_max_segments=1,
            translation_chunk_max_chars=100,
        ),
    )
    _write_metadata_and_source_transcript(pipeline.store, job.video_id)
    paths = pipeline.store.artifact_paths(job.video_id)
    source = _source_transcript(job.video_id)
    source.segments.append(TranscriptSegment(1, 1, "Second English source text", raw_text="Second English source text"))
    pipeline.store._write_json(paths.normalized_transcript, source.to_dict())
    calls = []

    class Translator:
        def translate_transcript(self, transcript, target_language):
            calls.append([segment.text for segment in transcript.segments])
            return Transcript(
                video_id=transcript.video_id,
                source=transcript.source,
                language="ru",
                segments=[
                    TranscriptSegment(segment.start, segment.duration, f"Русский перевод {index}", segment.raw_text)
                    for index, segment in enumerate(transcript.segments)
                ],
            )

    pipeline.pipeline.translation_provider = Translator()

    expand = pipeline.run(pipeline.context(job, _stage(runtime, job.id, "transcript_translation_expand"), runtime=runtime))
    chunks = [
        stage
        for stage in runtime.list_stage_plan(job.id)
        if stage.stage_name.startswith("transcript_translation_chunk_")
    ]
    first_chunk = pipeline.run(pipeline.context(job, chunks[0], runtime=runtime))
    second_chunk = pipeline.run(pipeline.context(job, chunks[1], runtime=runtime))

    assert expand.status == "completed"
    assert first_chunk.status == "completed"
    assert second_chunk.status == "completed"
    assert not paths.transcript_ru.exists()
    chunk_payload = json.loads((paths.video_dir / "translation" / "chunks" / "000.json").read_text(encoding="utf-8"))
    assert chunk_payload["video_id"] == job.video_id
    assert chunk_payload["target_language"] == "ru"
    assert chunk_payload["chunk_index"] == 0
    assert chunk_payload["source_segment_start_index"] == 0
    assert chunk_payload["source_segment_end_index"] == 0
    assert chunk_payload["segments"][0]["index"] == 0
    assert chunk_payload["segments"][0]["source_text"] == "English source text"

    merge = pipeline.run(pipeline.context(job, _stage(runtime, job.id, "transcript_translation_merge"), runtime=runtime))

    assert merge.status == "completed"
    assert paths.transcript_ru.exists()
    payload = json.loads(paths.transcript_ru.read_text(encoding="utf-8"))
    assert payload["language"] == "ru"
    assert [segment["text"] for segment in payload["segments"]] == ["Русский перевод 0", "Русский перевод 0"]
    assert calls == [["English source text"], ["Second English source text"]]


def test_staged_transcript_translation_skips_valid_completed_chunk(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        settings=replace(
            StagePipeline(library_dir=tmp_path).settings,
            enable_translation=True,
            target_language="ru",
            translation_chunk_max_segments=1,
        ),
    )
    _write_metadata_and_source_transcript(pipeline.store, job.video_id)
    pipeline.run(pipeline.context(job, _stage(runtime, job.id, "transcript_translation_expand"), runtime=runtime))
    chunk_stage = [
        stage
        for stage in runtime.list_stage_plan(job.id)
        if stage.stage_name.startswith("transcript_translation_chunk_")
    ][0]
    source_hash = json.loads(chunk_stage.stage_payload_json)["source_transcript_hash"]
    chunk_path = pipeline.store.video_dir(job.video_id) / "translation" / "chunks" / "000.json"
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_path.write_text(
        json.dumps(
            {
                "video_id": job.video_id,
                "target_language": "ru",
                "language": "ru",
                "chunk_index": 0,
                "total_chunks": 1,
                "source_transcript_hash": source_hash,
                "source_segment_start_index": 0,
                "source_segment_end_index": 0,
                "segments": [
                    {
                        "index": 0,
                        "start": 0,
                        "duration": 1,
                        "source_text": "English source text",
                        "text": "Русский перевод текста",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    class TranslatorShouldNotRun:
        def translate_transcript(self, transcript, target_language):
            raise AssertionError("completed chunk should be skipped")

    pipeline.pipeline.translation_provider = TranslatorShouldNotRun()

    result = pipeline.run(pipeline.context(job, chunk_stage, runtime=runtime))

    assert result.status == "completed"
    assert result.warning == "translation chunk cache hit"


def test_staged_transcript_translation_merge_rejects_missing_chunk_coverage(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        settings=replace(
            StagePipeline(library_dir=tmp_path).settings,
            enable_translation=True,
            target_language="ru",
            translation_chunk_max_segments=1,
        ),
    )
    _write_metadata_and_source_transcript(pipeline.store, job.video_id)
    paths = pipeline.store.artifact_paths(job.video_id)
    source = _source_transcript(job.video_id)
    source.segments.append(TranscriptSegment(1, 1, "Second English source text", raw_text="Second English source text"))
    pipeline.store._write_json(paths.normalized_transcript, source.to_dict())
    pipeline.run(pipeline.context(job, _stage(runtime, job.id, "transcript_translation_expand"), runtime=runtime))

    with pytest.raises(RuntimeError, match="missing translation chunk"):
        pipeline.run(pipeline.context(job, _stage(runtime, job.id, "transcript_translation_merge"), runtime=runtime))

    assert not paths.transcript_ru.exists()


def test_staged_transcript_translation_merge_rejects_changed_timestamps(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        settings=replace(StagePipeline(library_dir=tmp_path).settings, enable_translation=True, target_language="ru"),
    )
    _write_metadata_and_source_transcript(pipeline.store, job.video_id)
    pipeline.run(pipeline.context(job, _stage(runtime, job.id, "transcript_translation_expand"), runtime=runtime))
    chunk_stage = [
        stage
        for stage in runtime.list_stage_plan(job.id)
        if stage.stage_name.startswith("transcript_translation_chunk_")
    ][0]
    source_hash = json.loads(chunk_stage.stage_payload_json)["source_transcript_hash"]
    chunk_path = pipeline.store.video_dir(job.video_id) / "translation" / "chunks" / "000.json"
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_path.write_text(
        json.dumps(
            {
                "video_id": job.video_id,
                "target_language": "ru",
                "language": "ru",
                "chunk_index": 0,
                "total_chunks": 1,
                "source_transcript_hash": source_hash,
                "source_segment_start_index": 0,
                "source_segment_end_index": 0,
                "segments": [
                    {
                        "index": 0,
                        "start": 0.25,
                        "duration": 1,
                        "source_text": "English source text",
                        "text": "Русский перевод текста",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="timestamp"):
        pipeline.run(pipeline.context(job, _stage(runtime, job.id, "transcript_translation_merge"), runtime=runtime))

    assert not pipeline.store.artifact_paths(job.video_id).transcript_ru.exists()


def test_later_scientific_analysis_failure_keeps_transcript_ru_artifact(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        settings=replace(StagePipeline(library_dir=tmp_path).settings, enable_translation=True, target_language="ru"),
    )
    _write_metadata_and_source_transcript(pipeline.store, job.video_id)

    class Translator:
        def translate_transcript(self, transcript, target_language):
            return _ru_transcript(transcript.video_id)

    class FailingAnalyzer:
        def analyze(self, metadata, transcript):
            raise RuntimeError("lm timeout")

    pipeline.pipeline.translation_provider = Translator()
    pipeline.pipeline.analyzer = FailingAnalyzer()
    pipeline.run(pipeline.context(job, _stage(runtime, job.id, "transcript_translation_expand"), runtime=runtime))
    for stage in runtime.list_stage_plan(job.id):
        if stage.stage_name.startswith("transcript_translation_chunk_"):
            pipeline.run(pipeline.context(job, stage, runtime=runtime))
    pipeline.run(pipeline.context(job, _stage(runtime, job.id, "transcript_translation_merge"), runtime=runtime))

    with pytest.raises(RuntimeError, match="lm timeout"):
        pipeline.run(pipeline.context(job, _stage(runtime, job.id, "scientific_analysis"), runtime=runtime))

    paths = pipeline.store.artifact_paths(job.video_id)
    assert paths.transcript_ru.exists()
    assert not paths.analysis_source.exists()


def test_staged_scientific_analysis_writes_source_analysis_immediately(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        settings=replace(StagePipeline(library_dir=tmp_path).settings, enable_translation=True, target_language="ru"),
    )
    _write_metadata_and_source_transcript(pipeline.store, job.video_id)
    pipeline.store.save_translated_transcript(job.video_id, _ru_transcript(job.video_id))

    class Analyzer:
        def analyze(self, metadata, transcript):
            assert transcript.language == "ru"
            return _analysis("ru", russian=True)

    pipeline.pipeline.analyzer = Analyzer()

    result = pipeline.run(pipeline.context(job, _stage(runtime, job.id, "scientific_analysis"), runtime=runtime))

    paths = pipeline.store.artifact_paths(job.video_id)
    assert result.status == "completed"
    assert paths.analysis_source.exists()
    assert not paths.analysis_ru.exists()
    assert json.loads(paths.analysis_source.read_text(encoding="utf-8"))["language"] == "ru"


def test_staged_short_scientific_analysis_skips_analysis_reduce(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        settings=replace(
            StagePipeline(library_dir=tmp_path).settings,
            enable_translation=False,
            analysis_map_reduce_enabled=True,
            analysis_direct_max_segments=10,
        ),
    )
    _write_metadata_and_source_transcript(pipeline.store, job.video_id)

    class Analyzer:
        def analyze(self, metadata, transcript):
            return _analysis()

    pipeline.pipeline.analyzer = Analyzer()

    result = pipeline.run(pipeline.context(job, _stage(runtime, job.id, "scientific_analysis"), runtime=runtime))

    assert result.status == "completed"
    assert result.skip_stages == ["analysis_reduce"]
    assert pipeline.store.artifact_paths(job.video_id).analysis_source.exists()


def test_staged_long_scientific_analysis_expands_analysis_chunks(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        settings=replace(
            StagePipeline(library_dir=tmp_path).settings,
            enable_translation=False,
            analysis_map_reduce_enabled=True,
            analysis_direct_max_segments=2,
            analysis_chunk_max_segments=2,
            analysis_chunk_max_chars=10_000,
        ),
    )
    paths = pipeline.store.artifact_paths(job.video_id)
    paths.video_dir.mkdir(parents=True, exist_ok=True)
    pipeline.store._write_json(paths.metadata, VideoMetadata(job.video_id, job.canonical_url, "Talk").to_dict())
    pipeline.store._write_json(paths.normalized_transcript, _long_source_transcript(job.video_id, count=5).to_dict())

    class Analyzer:
        def analyze(self, metadata, transcript):
            raise AssertionError("long staged analysis must not call monolithic analyze")

    pipeline.pipeline.analyzer = Analyzer()

    result = pipeline.run(pipeline.context(job, _stage(runtime, job.id, "scientific_analysis"), runtime=runtime))

    chunk_stages = [stage for stage in runtime.list_stage_plan(job.id) if stage.stage_name.startswith("analysis_chunk_")]
    assert result.status == "completed"
    assert not paths.analysis_source.exists()
    assert [stage.stage_name for stage in chunk_stages] == ["analysis_chunk_000", "analysis_chunk_001", "analysis_chunk_002"]
    assert json.loads(chunk_stages[0].stage_payload_json)["source_segment_end_index"] == 1


def test_staged_analysis_chunk_writes_artifact_immediately_and_skips_valid_cache(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        settings=replace(
            StagePipeline(library_dir=tmp_path).settings,
            enable_translation=False,
            analysis_direct_max_segments=1,
            analysis_chunk_max_segments=2,
            analysis_chunk_max_chars=10_000,
        ),
    )
    paths = pipeline.store.artifact_paths(job.video_id)
    paths.video_dir.mkdir(parents=True, exist_ok=True)
    pipeline.store._write_json(paths.metadata, VideoMetadata(job.video_id, job.canonical_url, "Talk").to_dict())
    pipeline.store._write_json(paths.normalized_transcript, _long_source_transcript(job.video_id, count=3).to_dict())
    pipeline.run(pipeline.context(job, _stage(runtime, job.id, "scientific_analysis"), runtime=runtime))

    calls = []

    class Analyzer:
        model = "test-model"

        def analyze_chunk(self, metadata, transcript, *, chunk_index, total_chunks):
            calls.append(chunk_index)
            return {
                "summary": f"Chunk {chunk_index} summary",
                "outline": [f"Chunk {chunk_index}"],
                "concepts": [{"name": f"Concept {chunk_index}", "definition": "Definition"}],
                "claims": [{"text": f"Claim {chunk_index}", "evidence": "Evidence"}],
                "evidence_segment_indexes": [0],
                "limitations": [],
                "open_questions": [],
                "related_topics": [f"Topic {chunk_index}"],
            }

    pipeline.pipeline.analyzer = Analyzer()
    chunk_stage = _stage(runtime, job.id, "analysis_chunk_000")

    result = pipeline.run(pipeline.context(job, chunk_stage, runtime=runtime))
    cached = pipeline.run(pipeline.context(job, chunk_stage, runtime=runtime))

    chunk_path = tmp_path / "videos" / job.video_id / "analysis" / "chunks" / "000.json"
    artifact = json.loads(chunk_path.read_text(encoding="utf-8"))
    assert result.status == "completed"
    assert cached.warning == "analysis chunk cache hit"
    assert calls == [0]
    assert artifact["video_id"] == job.video_id
    assert artifact["source_segment_start_index"] == 0
    assert artifact["source_segment_end_index"] == 1
    assert artifact["summary"] == "Chunk 0 summary"


def test_staged_analysis_chunk_reruns_when_artifact_hash_is_stale(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        settings=replace(
            StagePipeline(library_dir=tmp_path).settings,
            enable_translation=False,
            analysis_direct_max_segments=1,
            analysis_chunk_max_segments=1,
        ),
    )
    paths = pipeline.store.artifact_paths(job.video_id)
    paths.video_dir.mkdir(parents=True, exist_ok=True)
    pipeline.store._write_json(paths.metadata, VideoMetadata(job.video_id, job.canonical_url, "Talk").to_dict())
    pipeline.store._write_json(paths.normalized_transcript, _long_source_transcript(job.video_id, count=2).to_dict())
    pipeline.run(pipeline.context(job, _stage(runtime, job.id, "scientific_analysis"), runtime=runtime))
    chunk_path = tmp_path / "videos" / job.video_id / "analysis" / "chunks" / "000.json"
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_path.write_text(
        json.dumps(
            {
                "video_id": job.video_id,
                "chunk_index": 0,
                "total_chunks": 2,
                "source_transcript": "transcript.normalized.json",
                "source_segment_start_index": 0,
                "source_segment_end_index": 0,
                "source_transcript_hash": "stale",
                "language": "source",
                "model": "old",
                "prompt_version": "analysis-chunk-v1",
                "summary": "old",
                "outline": ["old"],
                "concepts": [],
                "claims": [],
                "evidence_segment_indexes": [0],
            }
        ),
        encoding="utf-8",
    )
    calls = []

    class Analyzer:
        model = "new"

        def analyze_chunk(self, metadata, transcript, *, chunk_index, total_chunks):
            calls.append(chunk_index)
            return {
                "summary": "new",
                "outline": ["new"],
                "concepts": [],
                "claims": [],
                "evidence_segment_indexes": [0],
                "limitations": [],
                "open_questions": [],
                "related_topics": [],
            }

    pipeline.pipeline.analyzer = Analyzer()

    pipeline.run(pipeline.context(job, _stage(runtime, job.id, "analysis_chunk_000"), runtime=runtime))

    assert calls == [0]
    assert json.loads(chunk_path.read_text(encoding="utf-8"))["source_transcript_hash"] != "stale"


def test_staged_analysis_chunk_rejects_stale_stage_payload_hash(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        settings=replace(
            StagePipeline(library_dir=tmp_path).settings,
            enable_translation=False,
            analysis_direct_max_segments=1,
            analysis_chunk_max_segments=1,
        ),
    )
    paths = pipeline.store.artifact_paths(job.video_id)
    paths.video_dir.mkdir(parents=True, exist_ok=True)
    pipeline.store._write_json(paths.metadata, VideoMetadata(job.video_id, job.canonical_url, "Talk").to_dict())
    pipeline.store._write_json(paths.normalized_transcript, _long_source_transcript(job.video_id, count=2).to_dict())
    pipeline.run(pipeline.context(job, _stage(runtime, job.id, "scientific_analysis"), runtime=runtime))
    pipeline.store._write_json(paths.normalized_transcript, _long_source_transcript(job.video_id, count=3).to_dict())

    class Analyzer:
        def analyze_chunk(self, metadata, transcript, *, chunk_index, total_chunks):
            raise AssertionError("stale payload must fail before analyzer call")

    pipeline.pipeline.analyzer = Analyzer()

    with pytest.raises(RuntimeError, match="payload transcript hash"):
        pipeline.run(pipeline.context(job, _stage(runtime, job.id, "analysis_chunk_000"), runtime=runtime))


def test_staged_analysis_reduce_rejects_missing_chunk_and_writes_source_analysis(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        settings=replace(
            StagePipeline(library_dir=tmp_path).settings,
            enable_translation=False,
            analysis_direct_max_segments=1,
            analysis_chunk_max_segments=1,
        ),
    )
    paths = pipeline.store.artifact_paths(job.video_id)
    paths.video_dir.mkdir(parents=True, exist_ok=True)
    pipeline.store._write_json(paths.metadata, VideoMetadata(job.video_id, job.canonical_url, "Talk").to_dict())
    pipeline.store._write_json(paths.normalized_transcript, _long_source_transcript(job.video_id, count=2).to_dict())
    pipeline.run(pipeline.context(job, _stage(runtime, job.id, "scientific_analysis"), runtime=runtime))

    with pytest.raises(RuntimeError, match="missing analysis chunk 000"):
        pipeline.run(pipeline.context(job, _stage(runtime, job.id, "analysis_reduce"), runtime=runtime))

    class Analyzer:
        model = "test-model"

        def analyze_chunk(self, metadata, transcript, *, chunk_index, total_chunks):
            return {
                "summary": f"summary {chunk_index}",
                "outline": [f"part {chunk_index}"],
                "concepts": [{"name": f"Concept {chunk_index}", "definition": "Definition"}],
                "claims": [{"text": f"Claim {chunk_index}", "evidence": f"Evidence {chunk_index}"}],
                "evidence_segment_indexes": [0],
                "limitations": [],
                "open_questions": [],
                "related_topics": [f"Topic {chunk_index}"],
            }

    pipeline.pipeline.analyzer = Analyzer()
    for stage in runtime.list_stage_plan(job.id):
        if stage.stage_name.startswith("analysis_chunk_"):
            pipeline.run(pipeline.context(job, stage, runtime=runtime))
    result = pipeline.run(pipeline.context(job, _stage(runtime, job.id, "analysis_reduce"), runtime=runtime))

    analysis = json.loads(paths.analysis_source.read_text(encoding="utf-8"))
    assert result.status == "completed"
    assert analysis["outline"] == ["part 0", "part 1"]
    assert len(analysis["sections"]) == 2


def test_analysis_translation_not_runnable_before_analysis_reduce(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    runtime.expand_analysis_chunk_plan(
        job.id,
        [{"chunk_index": 0, "source_segment_start_index": 0, "source_segment_end_index": 0, "source_transcript_hash": "hash-a"}],
    )
    now = time.time()
    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute(
            "UPDATE stage_plans SET status = 'completed', attempt = 1, completed_at = ?, updated_at = ? WHERE video_job_id = ? AND position < (SELECT position FROM stage_plans WHERE video_job_id = ? AND stage_name = 'analysis_reduce')",
            (now, now, job.id, job.id),
        )
        conn.execute(
            "UPDATE stage_plans SET status = 'pending' WHERE video_job_id = ? AND stage_name = 'analysis_reduce'",
            (job.id,),
        )

    assert [stage.stage_name for stage in runtime.list_runnable_stages()] == ["analysis_reduce"]


def test_staged_scientific_analysis_does_not_use_cached_source_before_translation_merge(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        settings=replace(StagePipeline(library_dir=tmp_path).settings, enable_translation=True, target_language="ru"),
    )
    _write_metadata_and_source_transcript(pipeline.store, job.video_id)
    pipeline.store.save_analysis_artifact(job.video_id, _analysis(), variant="source")

    with pytest.raises(RuntimeError, match="transcript.ru.json"):
        pipeline.run(pipeline.context(job, _stage(runtime, job.id, "scientific_analysis"), runtime=runtime))


def test_staged_analysis_translation_writes_analysis_ru_only_after_ru_gate(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        settings=replace(StagePipeline(library_dir=tmp_path).settings, enable_translation=True, target_language="ru"),
    )
    _write_metadata_and_source_transcript(pipeline.store, job.video_id)
    pipeline.store.save_translated_transcript(job.video_id, _ru_transcript(job.video_id))
    pipeline.store.save_analysis_artifact(job.video_id, _analysis(), variant="source")

    class Translator:
        def translate_analysis(self, analysis, target_language):
            return _analysis("ru", russian=True)

    pipeline.pipeline.translation_provider = Translator()

    result = pipeline.run(pipeline.context(job, _stage(runtime, job.id, "analysis_translation"), runtime=runtime))

    paths = pipeline.store.artifact_paths(job.video_id)
    assert result.status == "completed"
    assert result.warning is None
    assert paths.analysis_ru.exists()
    analysis_ru = json.loads(paths.analysis_ru.read_text(encoding="utf-8"))
    source_data = pipeline.store.load_analysis(job.video_id, variant="source")
    assert analysis_ru["abstract"] == "Русский научный конспект"
    assert analysis_ru["source_analysis_hash"] == analysis_data_hash(source_data)
    assert analysis_ru["translation_model"] == pipeline.settings.translation_model
    assert analysis_ru["translation_prompt_version"] == "analysis-translation-v2"
    assert analysis_ru["generated_from"]["strategy"] == "llm_analysis_translation"


def test_staged_analysis_ru_review_completes_for_valid_analysis_ru(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        settings=replace(StagePipeline(library_dir=tmp_path).settings, enable_translation=True, target_language="ru"),
    )
    pipeline.store.save_analysis_artifact(job.video_id, _analysis("ru", russian=True), variant="ru", compatibility=True)

    result = pipeline.run(pipeline.context(job, _stage(runtime, job.id, "analysis_ru_review"), runtime=runtime))

    assert result.status == "completed"
    assert result.warning is None


def test_staged_analysis_ru_review_fails_when_analysis_ru_missing(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        settings=replace(StagePipeline(library_dir=tmp_path).settings, enable_translation=True, target_language="ru"),
    )

    result = pipeline.run(pipeline.context(job, _stage(runtime, job.id, "analysis_ru_review"), runtime=runtime))

    assert result.status == "failed"
    assert result.error == "analysis.ru.json must exist"


def test_staged_analysis_ru_review_fails_for_english_abstract_with_russian_title(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        settings=replace(StagePipeline(library_dir=tmp_path).settings, enable_translation=True, target_language="ru"),
    )
    stale = _analysis("ru", russian=True)
    stale.cleaned_title = "РУССКОЕ НАЗВАНИЕ"
    stale.abstract = "Scientific summary of 'РУССКОЕ НАЗВАНИЕ'. The note is assembled from source transcript."
    pipeline.store.save_analysis_artifact(job.video_id, stale, variant="ru", compatibility=True)

    result = pipeline.run(pipeline.context(job, _stage(runtime, job.id, "analysis_ru_review"), runtime=runtime))

    assert result.status == "failed"
    assert "analysis.ru.json abstract contains blocked English phrase: Scientific summary" in (result.error or "")
    assert "analysis.ru.json abstract contains blocked English phrase: The note is assembled" in (result.error or "")


def test_staged_analysis_ru_review_fails_for_mixed_language_claim_evidence(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        settings=replace(StagePipeline(library_dir=tmp_path).settings, enable_translation=True, target_language="ru"),
    )
    stale = _analysis("ru", russian=True)
    stale.claims[0].evidence = "The video explains this with one русский word."
    pipeline.store.save_analysis_artifact(job.video_id, stale, variant="ru", compatibility=True)

    result = pipeline.run(pipeline.context(job, _stage(runtime, job.id, "analysis_ru_review"), runtime=runtime))

    assert result.status == "failed"
    assert "analysis.ru.json claims[0].evidence ru_word_ratio" in (result.error or "")


def test_staged_analysis_translation_best_effort_uses_fallback_when_model_returns_non_ru(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        settings=replace(
            StagePipeline(library_dir=tmp_path).settings,
            enable_translation=True,
            target_language="ru",
            translation_best_effort=True,
        ),
    )
    _write_metadata_and_source_transcript(pipeline.store, job.video_id)
    pipeline.store.save_translated_transcript(job.video_id, _ru_transcript(job.video_id))
    pipeline.store.save_analysis_artifact(job.video_id, _analysis(), variant="source")

    class Translator:
        def translate_analysis(self, analysis, target_language):
            return _analysis("ru", russian=False)

    pipeline.pipeline.translation_provider = Translator()

    result = pipeline.run(pipeline.context(job, _stage(runtime, job.id, "analysis_translation"), runtime=runtime))

    assert result.status == "completed"
    assert "deterministic sparse-analysis fallback" in (result.warning or "")
    analysis_ru = json.loads(pipeline.store.artifact_paths(job.video_id).analysis_ru.read_text(encoding="utf-8"))
    assert analysis_ru["language"] == "ru"
    assert analysis_ru["generated_from"]["strategy"] == "deterministic_sparse_fallback"


def test_staged_analysis_translation_strict_fails_without_saving_sparse_fallback(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        settings=replace(StagePipeline(library_dir=tmp_path).settings, enable_translation=True, target_language="ru"),
    )
    _write_metadata_and_source_transcript(pipeline.store, job.video_id)
    pipeline.store.save_translated_transcript(job.video_id, _ru_transcript(job.video_id))
    sparse = Analysis(
        language="unknown",
        cleaned_title="Русский доклад",
        abstract="Scientific summary of 'Русский доклад'. The note is assembled from source transcript.",
        tags=[],
        outline=[],
        sections=[NoteSection("Chunk", "Русское содержание раздела", 0, [])],
        concepts=[],
        claims=[],
        limitations=[],
        open_questions=[],
        related_topics=[],
    )
    pipeline.store.save_analysis_artifact(job.video_id, sparse, variant="source")

    class FailingTranslator:
        def translate_analysis(self, analysis, target_language):
            raise RuntimeError("model returned sparse analysis")

    pipeline.pipeline.translation_provider = FailingTranslator()

    result = pipeline.run(pipeline.context(job, _stage(runtime, job.id, "analysis_translation"), runtime=runtime))

    assert result.status == "failed"
    assert "Russian analysis was not produced" in (result.error or "")
    assert not pipeline.store.artifact_paths(job.video_id).analysis_ru.exists()
    diagnostics_path = pipeline.store.video_dir(job.video_id) / "translation" / "analysis_translation_error.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["stage"] == "analysis_translation"
    assert diagnostics["target_language"] == "ru"
    assert diagnostics["source_analysis_hash"] == analysis_data_hash(pipeline.store.load_analysis(job.video_id, variant="source"))
    assert "model returned sparse analysis" in diagnostics["error"]


def test_staged_analysis_translation_best_effort_adds_marked_ru_fallback_for_sparse_source_analysis(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        settings=replace(
            StagePipeline(library_dir=tmp_path).settings,
            enable_translation=True,
            target_language="ru",
            translation_best_effort=True,
        ),
    )
    _write_metadata_and_source_transcript(pipeline.store, job.video_id)
    pipeline.store.save_translated_transcript(job.video_id, _ru_transcript(job.video_id))
    sparse = Analysis(
        language="unknown",
        cleaned_title="Русский доклад",
        abstract="Scientific summary of 'Русский доклад'. The note is assembled from source transcript.",
        tags=[],
        outline=[],
        sections=[NoteSection("Chunk", "Русское содержание раздела", 0, [])],
        concepts=[],
        claims=[],
        limitations=[],
        open_questions=[],
        related_topics=[],
    )
    pipeline.store.save_analysis_artifact(job.video_id, sparse, variant="source")

    class FailingTranslator:
        def translate_analysis(self, analysis, target_language):
            raise RuntimeError("model returned sparse analysis")

    pipeline.pipeline.translation_provider = FailingTranslator()

    result = pipeline.run(pipeline.context(job, _stage(runtime, job.id, "analysis_translation"), runtime=runtime))

    assert result.status == "completed"
    assert "deterministic sparse-analysis fallback" in (result.warning or "")
    analysis_ru = json.loads(pipeline.store.artifact_paths(job.video_id).analysis_ru.read_text(encoding="utf-8"))
    assert analysis_ru["language"] == "ru"
    assert analysis_ru["generated_from"]["strategy"] == "deterministic_sparse_fallback"
    assert "Scientific summary" not in analysis_ru["abstract"]
    assert "The note is assembled" not in analysis_ru["abstract"]
    assert analysis_ru["concepts"]
    assert analysis_ru["claims"]


def test_staged_analysis_translation_ru_fallback_uses_transcript_not_source_english_fields(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        settings=replace(
            StagePipeline(library_dir=tmp_path).settings,
            enable_translation=True,
            target_language="ru",
            translation_best_effort=True,
        ),
    )
    _write_metadata_and_source_transcript(pipeline.store, job.video_id)
    pipeline.store.save_translated_transcript(job.video_id, _rich_ru_transcript(job.video_id))
    sparse = Analysis(
        language="unknown",
        cleaned_title="РУССКОЕ НАЗВАНИЕ",
        abstract="Scientific summary of 'РУССКОЕ НАЗВАНИЕ'. The note is assembled from source transcript.",
        tags=["source transcript"],
        outline=["Source outline"],
        sections=[NoteSection("Intro", "English section summary from source transcript.", 0, ["English source bullet"])],
        concepts=[Concept("Game concept", "English source definition from transcript")],
        claims=[Claim("В ролике перечислены ожидаемые игры.", "Evidence mixes source transcript and русское название.")],
        limitations=["Generated from source transcript"],
        open_questions=["What games will release?"],
        related_topics=["source transcript"],
    )
    pipeline.store.save_analysis_artifact(job.video_id, sparse, variant="source")

    class FailingTranslator:
        def translate_analysis(self, analysis, target_language):
            raise RuntimeError("model returned sparse analysis")

    pipeline.pipeline.translation_provider = FailingTranslator()

    result = pipeline.run(pipeline.context(job, _stage(runtime, job.id, "analysis_translation"), runtime=runtime))

    assert result.status == "completed"
    analysis_ru = json.loads(pipeline.store.artifact_paths(job.video_id).analysis_ru.read_text(encoding="utf-8"))
    serialized = json.dumps(analysis_ru, ensure_ascii=False)
    assert "Scientific summary" not in serialized
    assert "The note is assembled" not in serialized
    assert "source transcript" not in serialized
    assert "summary of" not in serialized
    assert "В ролике обсуждаются ожидаемые игры" in analysis_ru["abstract"]
    assert "В ролике обсуждаются ожидаемые игры" in analysis_ru["sections"][0]["summary"]
    assert "English section summary" not in analysis_ru["sections"][0]["summary"]
    assert "В ролике обсуждаются ожидаемые игры" in analysis_ru["claims"][0]["evidence"]
    assert "Evidence mixes" not in analysis_ru["claims"][0]["evidence"]
    assert validate_ru_analysis(parse_analysis_response(analysis_ru)) == []


def test_staged_analysis_translation_ru_fallback_preserves_rich_source_concepts(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        settings=replace(
            StagePipeline(library_dir=tmp_path).settings,
            enable_translation=True,
            target_language="ru",
            translation_best_effort=True,
        ),
    )
    _write_metadata_and_source_transcript(pipeline.store, job.video_id)
    pipeline.store.save_translated_transcript(job.video_id, _rich_ru_transcript(job.video_id))
    source = Analysis(
        language="unknown",
        cleaned_title="Future Games",
        abstract="Scientific summary assembled from source transcript.",
        tags=["games"],
        outline=["Games"],
        sections=[NoteSection("Games", "Русский раздел описывает подборку игр и анонсов.", 0, [])],
        concepts=[
            Concept(name, f"Русское описание проекта {name} связывает игру с выпуском и важными анонсами.")
            for name in [
                "ZX Pencil",
                "Mortal Shell",
                "GTA 6",
                "Forza Horizon",
                "Subnatch",
                "Artificial Detective",
                "Project Windless",
                "Magi Hero",
                "Fantoms Blade",
                "Future Game",
            ]
        ],
        claims=[
            Claim(
                f"Игра {name} занимает отдельное место в подборке выпуска.",
                f"Русская расшифровка связывает проект {name} с обсуждением релизов и игровых особенностей.",
            )
            for name in [
                "ZX Pencil",
                "Mortal Shell",
                "GTA 6",
                "Forza Horizon",
                "Subnatch",
                "Artificial Detective",
                "Project Windless",
                "Magi Hero",
                "Fantoms Blade",
                "Future Game",
            ]
        ],
        limitations=[],
        open_questions=[],
        related_topics=[],
    )
    pipeline.store.save_analysis_artifact(job.video_id, source, variant="source")

    class FullTranslator:
        def translate_analysis(self, analysis, target_language):
            return Analysis(
                language="ru",
                cleaned_title="Полный русский конспект",
                abstract="Русский полный конспект сохраняет структуру исходного анализа и все ключевые выводы.",
                tags=["игры", "анонсы", "релизы"],
                outline=["Игры", "Выводы"],
                sections=[
                    NoteSection(
                        f"Раздел {index}",
                        "Русское резюме сохраняет содержание исходного анализа и связь с расшифровкой.",
                        index,
                        ["Русский пункт сохраняет детали исходного анализа."],
                    )
                    for index in range(len(analysis.sections))
                ],
                concepts=[
                    Concept(f"Русская игра {index}", "Русское описание сохраняет смысл исходного игрового проекта и анонса.")
                    for index, _ in enumerate(analysis.concepts)
                ],
                claims=[
                    Claim(
                        f"Русский вывод {index} сохраняет утверждение исходного анализа.",
                        "Русское доказательство сохраняет связь с расшифровкой и ключевыми деталями выпуска.",
                    )
                    for index, _ in enumerate(analysis.claims)
                ],
                limitations=[],
                open_questions=[],
                related_topics=["игры", "релизы"],
                model=analysis.model,
                prompt_version=analysis.prompt_version,
            )

    pipeline.pipeline.translation_provider = FullTranslator()

    result = pipeline.run(pipeline.context(job, _stage(runtime, job.id, "analysis_translation"), runtime=runtime))

    assert result.status == "completed"
    analysis_ru = json.loads(pipeline.store.artifact_paths(job.video_id).analysis_ru.read_text(encoding="utf-8"))
    assert len(analysis_ru["concepts"]) >= 10
    assert len(analysis_ru["claims"]) >= 10
    assert analysis_ru["generated_from"]["strategy"] == "llm_analysis_translation"
    assert validate_ru_analysis(parse_analysis_response(analysis_ru)) == []


def test_staged_analysis_translation_rejects_sparse_cache_for_rich_source_analysis(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        settings=replace(StagePipeline(library_dir=tmp_path).settings, enable_translation=True, target_language="ru"),
    )
    _write_metadata_and_source_transcript(pipeline.store, job.video_id)
    target_transcript = _rich_ru_transcript(job.video_id)
    pipeline.store.save_translated_transcript(job.video_id, target_transcript)
    source = Analysis(
        language="unknown",
        cleaned_title="Future Games",
        abstract="Scientific summary assembled from source transcript.",
        tags=[],
        outline=[],
        sections=[NoteSection("Games", "Русский раздел описывает подборку игр и анонсов.", 0, [])],
        concepts=[Concept(f"Game {index}", "Русское описание связывает игру с выпуском и анонсами.") for index in range(10)],
        claims=[
            Claim(
                f"Игра {index} занимает отдельное место в подборке выпуска.",
                "Русская расшифровка связывает этот проект с обсуждением релизов и игровых особенностей.",
            )
            for index in range(10)
        ],
        limitations=[],
        open_questions=[],
        related_topics=[],
    )
    pipeline.store.save_analysis_artifact(job.video_id, source, variant="source")
    source_data = pipeline.store.load_analysis(job.video_id, variant="source")
    pipeline.store.save_analysis_artifact(
        job.video_id,
        Analysis(
            language="ru",
            cleaned_title="Конспект видео",
            abstract="Русский конспект описывает подборку игр и анонсов выпуска.",
            tags=["видео"],
            outline=["Обзор"],
            sections=[NoteSection("Обзор", "Русское резюме выпуска.", 0, [])],
            concepts=[Concept("Основная тема", "Главная тема видео описана по русской расшифровке.")],
            claims=[Claim("Видео раскрывает тему выпуска.", "Русская расшифровка описывает тему выпуска.")],
            limitations=[],
            open_questions=[],
            related_topics=[],
        ),
        variant="ru",
        compatibility=True,
        extra_fields={
            "source_analysis_hash": analysis_data_hash(source_data),
            "source_transcript_hash": transcript_hash(target_transcript),
        },
    )

    class FullTranslator:
        def translate_analysis(self, analysis, target_language):
            return Analysis(
                language="ru",
                cleaned_title="Полный русский конспект",
                abstract="Русский полный конспект сохраняет структуру исходного анализа и все ключевые выводы.",
                tags=["игры"],
                outline=["Раздел"],
                sections=[
                    NoteSection(
                        f"Раздел {index}",
                        "Русское резюме сохраняет содержание исходного анализа и связь с расшифровкой.",
                        index,
                        [],
                    )
                    for index, _ in enumerate(analysis.sections)
                ],
                concepts=[
                    Concept("Русское понятие игры", "Русское описание сохраняет смысл исходного понятия и анонса.")
                    for _ in analysis.concepts
                ],
                claims=[
                    Claim(
                        "Русский вывод сохраняет утверждение исходного анализа.",
                        "Русское доказательство сохраняет связь с расшифровкой и ключевыми деталями выпуска.",
                    )
                    for _ in analysis.claims
                ],
                limitations=[],
                open_questions=[],
                related_topics=["игры"],
                model=analysis.model,
                prompt_version=analysis.prompt_version,
            )

    pipeline.pipeline.translation_provider = FullTranslator()

    result = pipeline.run(pipeline.context(job, _stage(runtime, job.id, "analysis_translation"), runtime=runtime))

    assert result.status == "completed"
    assert "cache hit" not in (result.warning or "")
    analysis_ru = json.loads(pipeline.store.artifact_paths(job.video_id).analysis_ru.read_text(encoding="utf-8"))
    assert len(analysis_ru["concepts"]) >= 10
    assert len(analysis_ru["claims"]) >= 10
    assert analysis_ru["generated_from"]["strategy"] == "llm_analysis_translation"


def test_staged_analysis_translation_ru_fallback_requires_transcript_in_strict_mode(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        settings=replace(StagePipeline(library_dir=tmp_path).settings, enable_translation=True, target_language="ru"),
    )
    _write_metadata_and_source_transcript(pipeline.store, job.video_id)
    sparse = Analysis(
        language="unknown",
        cleaned_title="РУССКОЕ НАЗВАНИЕ",
        abstract="Scientific summary of 'РУССКОЕ НАЗВАНИЕ'. The note is assembled from source transcript.",
        tags=[],
        outline=[],
        sections=[],
        concepts=[],
        claims=[],
        limitations=[],
        open_questions=[],
        related_topics=[],
    )
    pipeline.store.save_analysis_artifact(job.video_id, sparse, variant="source")

    with pytest.raises(RuntimeError, match="transcript.ru.json"):
        pipeline.run(pipeline.context(job, _stage(runtime, job.id, "analysis_translation"), runtime=runtime))

    assert not pipeline.store.artifact_paths(job.video_id).analysis_ru.exists()


def test_staged_ru_fallback_markdown_and_quality_gate_pass_after_regeneration(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        settings=replace(
            StagePipeline(library_dir=tmp_path).settings,
            enable_translation=True,
            target_language="ru",
            translation_best_effort=True,
        ),
    )
    _write_metadata_and_source_transcript(pipeline.store, job.video_id)
    pipeline.store.save_translated_transcript(job.video_id, _rich_ru_transcript(job.video_id))
    sparse = Analysis(
        language="unknown",
        cleaned_title="РУССКОЕ НАЗВАНИЕ",
        abstract="Scientific summary of 'РУССКОЕ НАЗВАНИЕ'. The note is assembled from source transcript.",
        tags=[],
        outline=[],
        sections=[NoteSection("Intro", "English section summary from source transcript.", 0, [])],
        concepts=[],
        claims=[],
        limitations=[],
        open_questions=[],
        related_topics=[],
    )
    pipeline.store.save_analysis_artifact(job.video_id, sparse, variant="source")

    class FailingTranslator:
        def translate_analysis(self, analysis, target_language):
            raise RuntimeError("model returned sparse analysis")

    pipeline.pipeline.translation_provider = FailingTranslator()

    pipeline.run(pipeline.context(job, _stage(runtime, job.id, "analysis_translation"), runtime=runtime))
    assert pipeline.run(pipeline.context(job, _stage(runtime, job.id, "analysis_ru_review"), runtime=runtime)).status == "completed"
    _mark_stage_completed(runtime, job.id, "analysis_ru_review")
    pipeline.run(pipeline.context(job, _stage(runtime, job.id, "markdown_render"), runtime=runtime))
    pipeline.run(pipeline.context(job, _stage(runtime, job.id, "deterministic_graph"), runtime=runtime))

    note_text = pipeline.store.note_path(job.video_id).read_text(encoding="utf-8")
    assert "## Abstract" not in note_text
    assert "## Аннотация" in note_text
    assert "**Claim:**" not in note_text
    assert "Evidence:" not in note_text
    assert "Scientific summary" not in note_text
    assert "The note is assembled" not in note_text
    assert validate_ru_markdown_note(note_text) == []
    review_result = pipeline.run(pipeline.context(job, _stage(runtime, job.id, "ru_artifact_review"), runtime=runtime))
    assert review_result.status == "completed"
    assert run_quality_gate(pipeline.store, job.video_id, target_language="ru").status == "completed"


def test_staged_markdown_render_fails_in_strict_ru_mode_without_valid_analysis_ru(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        settings=replace(StagePipeline(library_dir=tmp_path).settings, enable_translation=True, target_language="ru"),
    )
    _write_metadata_and_source_transcript(pipeline.store, job.video_id)
    pipeline.store.save_translated_transcript(job.video_id, _ru_transcript(job.video_id))
    pipeline.store.save_analysis_artifact(job.video_id, _analysis(), variant="source")

    with pytest.raises(RuntimeError, match="analysis_ru_review"):
        pipeline.run(pipeline.context(job, _stage(runtime, job.id, "markdown_render"), runtime=runtime))

    assert not pipeline.store.artifact_paths(job.video_id).note.exists()


def test_staged_markdown_render_fails_in_strict_ru_mode_without_transcript_ru(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        settings=replace(StagePipeline(library_dir=tmp_path).settings, enable_translation=True, target_language="ru"),
    )
    _write_metadata_and_source_transcript(pipeline.store, job.video_id)
    pipeline.store.save_analysis_artifact(job.video_id, _analysis("ru", russian=True), variant="ru")
    _mark_stage_completed(runtime, job.id, "analysis_ru_review")

    with pytest.raises(RuntimeError, match="transcript.ru.json"):
        pipeline.run(pipeline.context(job, _stage(runtime, job.id, "markdown_render"), runtime=runtime))

    assert not pipeline.store.artifact_paths(job.video_id).note.exists()


def test_staged_deterministic_graph_uses_ru_pair_and_marks_generated_language_ru(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        settings=replace(StagePipeline(library_dir=tmp_path).settings, enable_translation=True, target_language="ru"),
    )
    _write_russian_video(pipeline.store, job.video_id)
    pipeline.store.artifact_paths(job.video_id).graph_extraction.unlink()
    _mark_stage_completed(runtime, job.id, "analysis_ru_review")

    result = pipeline.run(pipeline.context(job, _stage(runtime, job.id, "deterministic_graph"), runtime=runtime))

    extraction = pipeline.store.load_graph_extraction(job.video_id)
    assert result.status == "completed"
    assert extraction["generated_from"]["transcript"] == "transcript.ru.json"
    assert extraction["generated_from"]["analysis"] == "analysis.ru.json"
    assert extraction["generated_from"]["language"] == "ru"
    assert any(node.get("text") == "Русский текст" for node in extraction["nodes"])


def test_staged_deterministic_graph_rebuilds_cached_source_graph_without_matching_analysis_hash(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        settings=replace(StagePipeline(library_dir=tmp_path).settings, enable_translation=False),
    )
    _write_metadata_and_source_transcript(pipeline.store, job.video_id)
    pipeline.store.save_analysis_artifact(job.video_id, _analysis("en"), variant="source", compatibility=True)

    first = pipeline.run(pipeline.context(job, _stage(runtime, job.id, "deterministic_graph"), runtime=runtime))
    cached = pipeline.store.load_graph_extraction(job.video_id)
    pipeline.store.save_analysis_artifact(
        job.video_id,
        Analysis(
            language="en",
            cleaned_title="Changed English talk",
            abstract="Changed scientific summary",
            tags=["changed"],
            outline=["Changed"],
            sections=[NoteSection("Changed", "Changed summary", 0, [])],
            concepts=[Concept("Changed concept", "Changed definition")],
            claims=[Claim("Changed claim.", "Changed evidence.")],
            limitations=[],
            open_questions=[],
            related_topics=[],
        ),
        variant="source",
        compatibility=True,
    )

    result = pipeline.run(pipeline.context(job, _stage(runtime, job.id, "deterministic_graph"), runtime=runtime))

    extraction = pipeline.store.load_graph_extraction(job.video_id)
    assert first.status == "completed"
    assert cached["generated_from"]["analysis_hash"] != extraction["generated_from"]["analysis_hash"]
    assert result.warning != "graph cache hit"
    assert extraction["generated_from"]["analysis"] == "analysis.json"


def test_staged_deterministic_graph_rebuilds_cached_ru_graph_without_matching_analysis_hash(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        settings=replace(StagePipeline(library_dir=tmp_path).settings, enable_translation=True, target_language="ru"),
    )
    _write_russian_video(pipeline.store, job.video_id)
    cached = pipeline.store.load_graph_extraction(job.video_id)
    cached["generated_from"].pop("analysis_hash", None)
    pipeline.store.save_graph_extraction(job.video_id, cached)
    cached = pipeline.store.load_graph_extraction(job.video_id)
    assert "analysis_hash" not in cached["generated_from"]
    now = time.time()
    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute(
            """
            UPDATE stage_plans
            SET status = 'completed', attempt = 1, completed_at = ?, updated_at = ?
            WHERE video_job_id = ? AND stage_name = 'analysis_ru_review'
            """,
            (now, now, job.id),
        )

    result = pipeline.run(pipeline.context(job, _stage(runtime, job.id, "deterministic_graph"), runtime=runtime))

    extraction = pipeline.store.load_graph_extraction(job.video_id)
    assert result.status == "completed"
    assert result.warning != "graph cache hit"
    assert extraction["generated_from"]["analysis"] == "analysis.ru.json"
    assert extraction["generated_from"]["analysis_hash"]


def test_staged_neo4j_sync_deletes_video_relationships_before_upsert(tmp_path, monkeypatch):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        settings=replace(StagePipeline(library_dir=tmp_path).settings, enable_translation=True, target_language="ru", enable_graph=True),
    )
    _write_russian_video(pipeline.store, job.video_id)
    _mark_stage_completed(runtime, job.id, "analysis_ru_review")
    pipeline.run(pipeline.context(job, _stage(runtime, job.id, "markdown_render"), runtime=runtime))
    pipeline.run(pipeline.context(job, _stage(runtime, job.id, "deterministic_graph"), runtime=runtime))
    review_result = pipeline.run(pipeline.context(job, _stage(runtime, job.id, "ru_artifact_review"), runtime=runtime))
    assert review_result.status == "completed"
    _mark_stage_completed(runtime, job.id, "ru_artifact_review")
    operations = []

    class GraphStore:
        def __init__(self, uri, user, password):
            operations.append(("init", uri, user, password))

        def delete_video_relationships(self, video_id):
            operations.append(("delete_video_relationships", video_id))

        def upsert_payload(self, payload):
            operations.append(("upsert_payload", len(payload.nodes), len(payload.relationships)))

        def close(self):
            operations.append(("close",))

    monkeypatch.setattr("yt_sub_hub.stage_pipeline.Neo4jGraphStore", GraphStore)

    result = pipeline.run(pipeline.context(job, _stage(runtime, job.id, "neo4j_sync"), runtime=runtime))

    assert result.status == "completed"
    assert [operation[0] for operation in operations] == ["init", "delete_video_relationships", "upsert_payload", "close"]
    assert operations[1] == ("delete_video_relationships", job.video_id)


def test_staged_strict_operational_stage_warnings_allow_neo4j_sync(tmp_path, monkeypatch):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        job_id=job_id_for_url("https://www.youtube.com/watch?v=abc123def45"),
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
        quality_mode="strict",
    )
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        settings=replace(StagePipeline(library_dir=tmp_path).settings, enable_translation=True, target_language="ru", enable_graph=True),
    )
    _write_russian_video(pipeline.store, job.video_id)
    review_result = pipeline.run(pipeline.context(job, _stage(runtime, job.id, "ru_artifact_review"), runtime=runtime))
    assert review_result.status == "completed"
    _mark_stage_completed(runtime, job.id, "analysis_ru_review")
    _mark_stage_completed(runtime, job.id, "ru_artifact_review")
    _mark_stage_completed_with_warning(runtime, job.id, "transcript_source", "needs whisper")
    _mark_stage_completed_with_warning(runtime, job.id, "analysis_translation", "analysis translation cache hit")
    _mark_stage_completed_with_warning(runtime, job.id, "markdown_render", "note cache hit")
    _mark_stage_completed_with_warning(runtime, job.id, "deterministic_graph", "graph cache hit")
    _mark_stage_completed_with_warning(runtime, job.id, "visual_frame_analysis", "visual frame analysis is on-demand")
    operations = []

    class GraphStore:
        def __init__(self, uri, user, password):
            operations.append(("init", uri, user, password))

        def delete_video_relationships(self, video_id):
            operations.append(("delete_video_relationships", video_id))

        def upsert_payload(self, payload):
            operations.append(("upsert_payload", len(payload.nodes), len(payload.relationships)))

        def close(self):
            operations.append(("close",))

    monkeypatch.setattr("yt_sub_hub.stage_pipeline.Neo4jGraphStore", GraphStore)

    result = pipeline.run(pipeline.context(job, _stage(runtime, job.id, "neo4j_sync"), runtime=runtime))

    assert result.status == "completed"
    assert [operation[0] for operation in operations] == ["init", "delete_video_relationships", "upsert_payload", "close"]


def test_staged_neo4j_sync_does_not_connect_when_product_quality_blocks_graph_sync(tmp_path, monkeypatch):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        job_id=job_id_for_url("https://www.youtube.com/watch?v=abc123def45"),
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
        quality_mode="strict",
    )
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        settings=replace(StagePipeline(library_dir=tmp_path).settings, enable_translation=True, target_language="ru", enable_graph=True),
    )
    _write_russian_video(pipeline.store, job.video_id)
    _mark_stage_completed(runtime, job.id, "analysis_ru_review")
    _mark_stage_completed(runtime, job.id, "ru_artifact_review")

    monkeypatch.setattr(
        "yt_sub_hub.stage_pipeline.evaluate_product_quality",
        lambda *args, **kwargs: {
            "mode": "strict",
            "status": "warning",
            "production_ready": False,
            "graph_sync_allowed": False,
            "reasons": ["product quality blocked graph sync"],
        },
    )

    class GraphStoreShouldNotStart:
        def __init__(self, uri, user, password):
            raise AssertionError("Neo4jGraphStore must not be constructed when product quality blocks sync")

    monkeypatch.setattr("yt_sub_hub.stage_pipeline.Neo4jGraphStore", GraphStoreShouldNotStart)

    with pytest.raises(RuntimeError, match="product quality blocked graph sync"):
        pipeline.run(pipeline.context(job, _stage(runtime, job.id, "neo4j_sync"), runtime=runtime))


def test_staged_strict_review_warning_blocks_neo4j_sync(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        job_id=job_id_for_url("https://www.youtube.com/watch?v=abc123def45"),
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
        quality_mode="strict",
    )
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        settings=replace(StagePipeline(library_dir=tmp_path).settings, enable_translation=True, target_language="ru", enable_graph=True),
    )
    _write_russian_video(pipeline.store, job.video_id)
    report = review_ru_artifacts(job.video_id, pipeline.store, pipeline.settings)
    report_path = pipeline.store.artifact_paths(job.video_id).ru_artifact_review
    data = report.to_dict()
    data["status"] = "warning"
    data["issues"] = [{"severity": "warning", "code": "manual_warning", "message": "manual warning"}]
    report_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    _mark_stage_completed(runtime, job.id, "analysis_ru_review")
    now = time.time()
    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute(
            "UPDATE stage_plans SET status = 'completed_with_warning', warning = 'manual warning', completed_at = ?, updated_at = ? WHERE video_job_id = ? AND stage_name = 'ru_artifact_review'",
            (now, now, job.id),
        )

    with pytest.raises(RuntimeError, match="graph sync is not allowed in strict mode"):
        pipeline.run(pipeline.context(job, _stage(runtime, job.id, "neo4j_sync"), runtime=runtime))


def test_staged_best_effort_review_warning_allows_neo4j_sync(tmp_path, monkeypatch):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        job_id=job_id_for_url("https://www.youtube.com/watch?v=abc123def45"),
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
        quality_mode="best_effort",
    )
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        settings=replace(StagePipeline(library_dir=tmp_path).settings, enable_translation=True, target_language="ru", enable_graph=True),
    )
    _write_russian_video(pipeline.store, job.video_id)
    report = review_ru_artifacts(job.video_id, pipeline.store, pipeline.settings)
    report_path = pipeline.store.artifact_paths(job.video_id).ru_artifact_review
    data = report.to_dict()
    data["status"] = "warning"
    data["issues"] = [{"severity": "warning", "code": "manual_warning", "message": "manual warning"}]
    report_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    _mark_stage_completed(runtime, job.id, "analysis_ru_review")
    now = time.time()
    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute(
            "UPDATE stage_plans SET status = 'completed_with_warning', warning = 'manual warning', completed_at = ?, updated_at = ? WHERE video_job_id = ? AND stage_name = 'ru_artifact_review'",
            (now, now, job.id),
        )
    operations = []

    class GraphStore:
        def __init__(self, uri, user, password):
            pass

        def delete_video_relationships(self, video_id):
            operations.append(("delete_video_relationships", video_id))

        def upsert_payload(self, payload):
            operations.append(("upsert_payload", len(payload.nodes)))

        def close(self):
            pass

    monkeypatch.setattr("yt_sub_hub.stage_pipeline.Neo4jGraphStore", GraphStore)

    result = pipeline.run(pipeline.context(job, _stage(runtime, job.id, "neo4j_sync"), runtime=runtime))

    assert result.status == "completed"
    assert operations[0] == ("delete_video_relationships", job.video_id)


def test_staged_draft_skips_graph_sync_and_marks_quality_gate_warning(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        job_id=job_id_for_url("https://www.youtube.com/watch?v=abc123def45"),
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
        quality_mode="draft",
    )
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=True)
    pipeline = StagePipeline(library_dir=tmp_path)

    sync = pipeline.run(pipeline.context(job, _stage(runtime, job.id, "neo4j_sync"), runtime=runtime))
    gate = pipeline.run(pipeline.context(job, _stage(runtime, job.id, "quality_gate"), runtime=runtime))

    assert sync.status == "skipped"
    assert sync.warning == "draft mode skips graph sync and production validation"
    assert gate.status == "completed"
    assert gate.warning == "draft mode: not production validated"


def test_staged_cache_hits_validate_target_artifacts(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)
    pipeline = StagePipeline(
        library_dir=tmp_path,
        settings=replace(StagePipeline(library_dir=tmp_path).settings, enable_translation=True, target_language="ru"),
    )
    _write_russian_video(pipeline.store, job.video_id)
    source_data = pipeline.store.load_analysis(job.video_id)
    pipeline.store.save_analysis_artifact(job.video_id, parse_analysis_response(source_data), variant="source")
    analysis_data = pipeline.store.load_analysis(job.video_id, variant="ru")
    transcript = pipeline.store.load_translated_transcript(job.video_id, "ru")
    analysis = parse_analysis_response(analysis_data)
    pipeline.store.save_analysis_artifact(
        job.video_id,
        analysis,
        variant="ru",
        compatibility=True,
            extra_fields={
                "source_analysis_hash": analysis_data_hash(source_data),
                "source_transcript_hash": transcript_hash(transcript),
                "translation_prompt_version": "analysis-translation-v2",
                "translation_model": pipeline.settings.translation_model,
                "generated_from": {
                    "analysis": "analysis.source.json",
                    "transcript": "transcript.ru.json",
                    "target_language": "ru",
                    "strategy": "llm_analysis_translation",
                },
            },
        )
    metadata = pipeline.store.load_video_metadata(job.video_id)
    pipeline.store.save_note_artifact(
        job.video_id,
        render_markdown_note(
            metadata,
            analysis,
            provenance={
                "analysis_hash": analysis_data_hash(pipeline.store.load_analysis(job.video_id, variant="ru")),
                "transcript_hash": transcript_hash(transcript),
                "generated_from_analysis": "analysis.ru.json",
            },
        ),
    )

    class TranslatorShouldNotRun:
        def translate_transcript(self, transcript, target_language):
            raise AssertionError("transcript cache should be used")

        def translate_analysis(self, analysis, target_language):
            raise AssertionError("analysis cache should be used")

    pipeline.pipeline.translation_provider = TranslatorShouldNotRun()

    assert pipeline.run(pipeline.context(job, _stage(runtime, job.id, "transcript_translation_expand"), runtime=runtime)).warning == "translation cache hit"
    assert pipeline.run(pipeline.context(job, _stage(runtime, job.id, "analysis_translation"), runtime=runtime)).warning == "analysis translation cache hit"
    assert pipeline.run(pipeline.context(job, _stage(runtime, job.id, "analysis_ru_review"), runtime=runtime)).status == "completed"
    _mark_stage_completed(runtime, job.id, "analysis_ru_review")
    assert pipeline.run(pipeline.context(job, _stage(runtime, job.id, "markdown_render"), runtime=runtime)).warning == "note cache hit"
    assert pipeline.run(pipeline.context(job, _stage(runtime, job.id, "deterministic_graph"), runtime=runtime)).status == "completed"


def test_stage_worker_cancel_prevents_next_stage_start(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id)
    runtime.request_cancel(job.id)

    result = StageWorker(runtime, owner="worker-1", library_dir=tmp_path).process_next_stage(
        StagePipeline(library_dir=tmp_path)
    )

    assert result is None
    assert runtime.load_video_job(job.id).status == "skipped"


def test_cancelled_queued_job_marks_pending_stage_plan_skipped(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id)

    runtime.request_cancel(job.id)

    assert {stage.status for stage in runtime.list_stage_plan(job.id)} == {"skipped"}


def test_stage_worker_quality_gate_final_blocks_success(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = _job(runtime)
    runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)

    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute("UPDATE stage_plans SET status = 'completed', completed_at = ? WHERE stage_name != 'quality_gate'", (time.time(),))
        conn.execute("UPDATE stage_plans SET status = 'pending', attempt = 0, warning = NULL, completed_at = NULL WHERE stage_name = 'quality_gate'")
        conn.execute("UPDATE video_jobs SET status = 'queued', completed_at = NULL WHERE id = ?", (job.id,))

    def hard_fail(ctx):
        return StageResult.failed("Quality gate hard failures: missing analysis")

    result = StageWorker(runtime, owner="quality-worker", library_dir=tmp_path).process_next_stage(
        StagePipeline(library_dir=tmp_path, handlers={"quality_gate": hard_fail})
    )

    assert result is not None
    assert runtime.load_video_job(job.id).status == "failed"
    assert _plan_statuses(runtime, job.id)[-1] == ("quality_gate", "failed")


def test_local_orchestrator_parallelism_respects_lm_studio_lock(tmp_path):
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    events = []

    def make_handler(name, delay):
        def handler(ctx):
            started = time.monotonic()
            events.append((name, ctx.job.video_id, "start", started))
            time.sleep(delay)
            finished = time.monotonic()
            events.append((name, ctx.job.video_id, "finish", finished))
            if name == "transcript_source":
                return StageResult.completed(skip_stages=["audio_download", "whisper_transcription"])
            if name in {"deep_graph_llm", "visual_frame_analysis", "neo4j_sync"}:
                return StageResult.skipped()
            return StageResult.completed()

        return handler

    handlers = {
            stage: make_handler(stage, 0.08)
        for stage in [
            "metadata",
            "transcript_source",
            "transcript_translation_expand",
            "transcript_translation_merge",
            "scientific_analysis",
            "analysis_reduce",
            "analysis_translation",
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
        job = _job(runtime, f"vid{index:08d}")
        runtime.ensure_stage_plan(job.id, enable_deep_graph_llm=False)

    pipeline = StagePipeline(library_dir=tmp_path, handlers=handlers)
    start = time.monotonic()
    LocalSQLiteOrchestrator(runtime, pipeline=pipeline, owner_prefix="test").run_loop(concurrency=4)
    elapsed = time.monotonic() - start

    metadata_starts = [event for event in events if event[0] == "metadata" and event[2] == "start"]
    analysis_windows = [
        (event[1], event[3], next(f[3] for f in events if f[0] == "scientific_analysis" and f[1] == event[1] and f[2] == "finish"))
        for event in events
        if event[0] == "scientific_analysis" and event[2] == "start"
    ]
    assert len(metadata_starts) == 3
    assert max(event[3] for event in metadata_starts) - min(event[3] for event in metadata_starts) < 0.1
    for _, start_a, finish_a in analysis_windows:
        overlaps = [1 for _, start_b, finish_b in analysis_windows if start_a < finish_b and start_b < finish_a]
        assert len(overlaps) == 1
    assert elapsed < 3 * 10 * 0.08
    assert {job.status for job in runtime.list_video_jobs()} <= {"completed", "completed_with_warning"}
