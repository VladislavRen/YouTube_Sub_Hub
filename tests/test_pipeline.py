import sqlite3
import time
from dataclasses import replace

import pytest

from yt_sub_hub.artifact_review import ReviewIssue, ReviewReport
from yt_sub_hub.library import LibraryStore
from yt_sub_hub.models import Analysis, Claim, Concept, JobRecord, NoteSection, Transcript, TranscriptSegment, VideoMetadata
from yt_sub_hub.transcripts import TranscriptResolutionError
from yt_sub_hub.pipeline import VideoProcessingPipeline, _analysis_matches_target_language, job_id_for_url
from yt_sub_hub.quality import run_quality_gate
from yt_sub_hub.translation import TranslationError


class RecordingGraphStore:
    def __init__(self):
        self.payloads = []
        self.clear_calls = 0
        self.operations = []
        self.deleted_video_ids = []

    def upsert_payload(self, payload):
        self.operations.append(("upsert_payload", payload))
        self.payloads.append(payload)

    def clear(self):
        self.operations.append(("clear", None))
        self.clear_calls += 1

    def delete_video_relationships(self, video_id):
        self.operations.append(("delete_video_relationships", video_id))
        self.deleted_video_ids.append(video_id)


def deep_graph_chunk_with_relationships(video_id="abc123"):
    return {
        "chunk_id": "chunk-1",
        "video_id": video_id,
        "timestamp_start": 0,
        "timestamp_end": 1,
        "concepts": [
            {
                "local_id": "c1",
                "name": "Support",
                "definition": "A support zone.",
                "aliases": [],
                "confidence": 0.9,
                "evidence_quote": "Support",
                "timestamp_start": 0,
                "timestamp_end": 1,
            },
            {
                "local_id": "c2",
                "name": "Breakout",
                "definition": "A move above resistance.",
                "aliases": [],
                "confidence": 0.9,
                "evidence_quote": "Breakout",
                "timestamp_start": 0,
                "timestamp_end": 1,
            },
        ],
        "claims": [
            {
                "local_id": "cl1",
                "text": "Support can cause a breakout.",
                "paraphrased_evidence": "Support breakout.",
                "confidence": 0.9,
                "timestamp_start": 0,
                "timestamp_end": 1,
            },
            {
                "local_id": "cl2",
                "text": "Breakout confirms strength.",
                "paraphrased_evidence": "Breakout strength.",
                "confidence": 0.9,
                "timestamp_start": 0,
                "timestamp_end": 1,
            },
        ],
        "evidence": [],
        "concept_relations": [
            {
                "from_concept_id": "c1",
                "to_concept_id": "c2",
                "type": "RELATED_TO",
                "confidence": 0.8,
                "evidence_quote": "Support breakout.",
                "timestamp_start": 0,
                "timestamp_end": 1,
            },
            {
                "from_concept_id": "c1",
                "to_concept_id": "c2",
                "type": "CAUSES",
                "confidence": 0.82,
                "evidence_quote": "Support causes breakout.",
                "timestamp_start": 0,
                "timestamp_end": 1,
            },
            {
                "from_concept_id": "c1",
                "to_concept_id": "c2",
                "type": "CONTRASTS_WITH",
                "confidence": 0.81,
                "evidence_quote": "Support differs from breakout.",
                "timestamp_start": 0,
                "timestamp_end": 1,
            },
        ],
        "claim_relations": [
            {
                "from_claim_id": "cl1",
                "to_claim_id": "cl2",
                "type": "SUPPORTS",
                "confidence": 0.85,
                "evidence_quote": "Strength supports breakout.",
                "timestamp_start": 0,
                "timestamp_end": 1,
            }
        ],
        "methods": [],
        "tools": [],
        "metrics": [],
        "risks": [],
        "questions": [],
    }


def rejected_deep_graph_chunk(video_id="abc123", chunk_id="chunk-2", error="Invalid deep graph JSON"):
    return {
        "chunk_id": chunk_id,
        "video_id": video_id,
        "timestamp_start": 0,
        "timestamp_end": 1,
        "concepts": [],
        "claims": [],
        "evidence": [],
        "concept_relations": [],
        "claim_relations": [],
        "methods": [],
        "tools": [],
        "metrics": [],
        "risks": [],
        "questions": [],
        "rejected": True,
        "error": error,
    }


def enable_translation_settings(pipeline, best_effort=False):
    object.__setattr__(pipeline.settings, "enable_translation", True)
    object.__setattr__(pipeline.settings, "translation_best_effort", best_effort)
    object.__setattr__(pipeline.settings, "translation_target_language", "ru")


def failed_ru_review_report(video_id="abc123def45"):
    return ReviewReport(
        video_id=video_id,
        target_language="ru",
        status="failed",
        reviewer={"kind": "deterministic", "version": "test"},
        checked_artifacts={},
        hashes={},
        coverage={},
        issues=[
            ReviewIssue(
                severity="hard",
                code="stale_analysis_ru",
                artifact="analysis.ru.json",
                message="analysis.ru.json source_analysis_hash must match current analysis.source.json",
            )
        ],
    )


def warning_ru_review_report(video_id="abc123def45"):
    return ReviewReport(
        video_id=video_id,
        target_language="ru",
        status="warning",
        reviewer={"kind": "deterministic", "version": "test"},
        checked_artifacts={},
        hashes={},
        coverage={},
        issues=[
            ReviewIssue(
                severity="warning",
                code="manual_warning",
                artifact="analysis.ru.json",
                message="manual warning",
            )
        ],
    )


def sample_analysis(language="en", abstract="Abstract", concept="Graph", claim_text="Graphs help."):
    is_ru = str(language).strip().lower() == "ru"
    section_heading = "Обзор" if is_ru else "Intro"
    concept_definition = f"{concept} помогает связать ключевые идеи доклада." if is_ru else f"{concept} definition."
    claim_evidence = "В докладе подробно объясняется это утверждение." if is_ru else abstract
    return Analysis(
        language=language,
        cleaned_title="Talk",
        abstract=abstract,
        tags=["тема"] if is_ru else [],
        outline=["Обзор"] if is_ru else [],
        sections=[NoteSection(heading=section_heading, summary=abstract, timestamp_seconds=0, bullets=[])],
        concepts=[Concept(name=concept, definition=concept_definition)],
        claims=[Claim(text=claim_text, evidence=claim_evidence)],
        limitations=[],
        open_questions=[],
        related_topics=[],
    )


def test_process_job_cancellation_before_metadata_prevents_metadata_fetch(tmp_path, monkeypatch):
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    url = "https://youtu.be/abc123def45"
    record = pipeline.submit(url)
    pipeline.cancel_checker = lambda job_id: True

    def fail_metadata(url, settings=None):
        raise AssertionError("metadata fetch should not run after cancellation")

    monkeypatch.setattr("yt_sub_hub.pipeline.fetch_video_metadata", fail_metadata)

    cancelled = pipeline.process_job(record.job_id)

    assert cancelled.status == "skipped"
    assert cancelled.stage == "cancelled"
    assert cancelled.warning == "cancel requested"
    assert pipeline.store.load_jobs()[record.job_id].status == "skipped"


def test_process_job_cancellation_after_metadata_prevents_transcript_resolution(tmp_path, monkeypatch):
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    url = "https://youtu.be/abc123def45"
    record = pipeline.submit(url)
    metadata = VideoMetadata(
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
        title="Talk",
    )
    checks = iter([False, True])
    pipeline.cancel_checker = lambda job_id: next(checks, True)

    def fail_transcript(video_id, whisper_provider=None):
        raise AssertionError("transcript resolution should not run after cancellation")

    monkeypatch.setattr("yt_sub_hub.pipeline.fetch_video_metadata", lambda url, settings=None: metadata)
    monkeypatch.setattr("yt_sub_hub.pipeline.resolve_transcript_bundle", fail_transcript)

    cancelled = pipeline.process_job(record.job_id)

    assert cancelled.status == "skipped"
    assert cancelled.stage == "cancelled"
    assert cancelled.warning == "cancel requested"
    assert not pipeline.store.artifact_paths("abc123def45").metadata.exists()


def test_library_store_saves_and_loads_source_and_russian_artifacts(tmp_path):
    store = LibraryStore(tmp_path)
    paths = store.artifact_paths("abc123")
    transcript_ru = Transcript(
        video_id="abc123",
        source="youtube",
        language="ru",
        segments=[TranscriptSegment(0, 1, "Русский текст", raw_text="Source text")],
    )
    analysis_ru = sample_analysis(language="ru", abstract="Русский анализ", concept="Граф", claim_text="Граф помогает.")
    source_graph = {
        "schema_version": "graph-extraction-v1",
        "video_id": "abc123",
        "generated_from": {"transcript": "transcript.normalized.json", "analysis": "analysis.source.json"},
        "nodes": [],
        "relationships": [],
        "chunks": [],
        "quality_report": {},
    }

    store.save_translated_transcript("abc123", transcript_ru)
    store.save_analysis_artifact("abc123", analysis_ru, variant="ru")
    store.save_note_artifact("abc123", "# Source note", variant="source")
    store.save_graph_extraction("abc123", source_graph, variant="source")

    assert paths.transcript_ru.name == "transcript.ru.json"
    assert paths.analysis_source.name == "analysis.source.json"
    assert paths.analysis_ru.name == "analysis.ru.json"
    assert paths.note_source.name == "note.source.md"
    assert paths.graph_extraction_source.name == "graph_extraction.source.json"
    assert store.load_translated_transcript("abc123").segments[0].text == "Русский текст"
    assert store.load_analysis("abc123", variant="ru")["abstract"] == "Русский анализ"
    assert paths.note_source.read_text(encoding="utf-8") == "# Source note"
    assert store.load_graph_extraction("abc123", variant="source")["generated_from"]["analysis"] == "analysis.source.json"


def test_persist_writes_graph_payload_when_graph_store_is_configured(tmp_path):
    graph_store = RecordingGraphStore()
    pipeline = VideoProcessingPipeline.for_test(tmp_path, graph_store=graph_store)
    metadata = VideoMetadata(video_id="abc123", canonical_url="https://youtu.be/abc123", title="Talk")
    transcript = Transcript(
        video_id="abc123",
        source="youtube",
        segments=[TranscriptSegment(start=0, duration=1, text="Hello", raw_text=" Hello ")],
    )
    analysis = Analysis(
        language="en",
        cleaned_title="Talk",
        abstract="Abstract",
        tags=[],
        outline=[],
        sections=[NoteSection(heading="Intro", summary="Summary", timestamp_seconds=0, bullets=[])],
        concepts=[Concept(name="Graph", definition="A structure.")],
        claims=[Claim(text="Graphs help.", evidence="The transcript says so.")],
        limitations=[],
        open_questions=[],
        related_topics=[],
    )

    pipeline.persist(metadata, transcript, analysis)

    assert graph_store.payloads
    assert any(node["label"] == "Concept" for node in graph_store.payloads[0].nodes)
    assert (tmp_path / "videos" / "abc123" / "graph_extraction.json").exists()
    assert any(node["label"] == "SourceSegment" for node in graph_store.payloads[0].nodes)
    assert all("confidence" in rel for rel in graph_store.payloads[0].relationships)


def test_persist_completes_with_graph_warning_when_graph_store_fails(tmp_path):
    class FailingGraphStore:
        def upsert_payload(self, payload):
            raise RuntimeError("neo4j unavailable")

    pipeline = VideoProcessingPipeline.for_test(tmp_path, graph_store=FailingGraphStore())
    metadata = VideoMetadata(video_id="abc123", canonical_url="https://youtu.be/abc123", title="Talk")
    transcript = Transcript(video_id="abc123", source="youtube", segments=[TranscriptSegment(0, 1, "Hello")])
    analysis = Analysis(
        language="en",
        cleaned_title="Talk",
        abstract="Abstract",
        tags=[],
        outline=[],
        sections=[NoteSection(heading="Intro", summary="Summary", timestamp_seconds=0, bullets=[])],
        concepts=[],
        claims=[],
        limitations=[],
        open_questions=[],
        related_topics=[],
    )

    paths, warning = pipeline.persist(metadata, transcript, analysis)

    assert paths.note.exists()
    assert "neo4j unavailable" in warning


def test_legacy_draft_mode_skips_graph_sync_and_marks_note_frontmatter(tmp_path):
    graph_store = RecordingGraphStore()
    pipeline = VideoProcessingPipeline.for_test(tmp_path, graph_store=graph_store)
    object.__setattr__(pipeline.settings, "quality_mode", "draft")
    metadata = VideoMetadata(video_id="abc123", canonical_url="https://youtu.be/abc123", title="Talk")
    transcript = Transcript(video_id="abc123", source="youtube", language="en", segments=[TranscriptSegment(0, 1, "Hello")])
    analysis = sample_analysis(language="en")

    paths, warning = pipeline.persist(metadata, transcript, transcript, analysis)

    note = paths.note.read_text(encoding="utf-8")
    assert graph_store.payloads == []
    assert warning == "draft mode skips graph sync and production validation"
    assert 'quality_mode: "draft"' in note
    assert 'quality_status: "draft"' in note


def test_legacy_strict_review_warning_skips_graph_sync(tmp_path, monkeypatch):
    graph_store = RecordingGraphStore()
    pipeline = VideoProcessingPipeline.for_test(tmp_path, graph_store=graph_store)
    enable_translation_settings(pipeline)
    object.__setattr__(pipeline.settings, "quality_mode", "strict")
    metadata = VideoMetadata(video_id="abc123def45", canonical_url="https://youtu.be/abc123def45", title="Talk")
    source_transcript = Transcript("abc123def45", "youtube", [TranscriptSegment(0, 1, "Source")], language="en")
    russian_transcript = Transcript("abc123def45", "youtube", [TranscriptSegment(0, 1, "Русский текст")], language="ru")
    analysis_ru = sample_analysis(language="ru", abstract="Русский анализ", concept="Граф", claim_text="Граф помогает.")
    monkeypatch.setattr("yt_sub_hub.pipeline.review_ru_artifacts", lambda video_id, store, settings: warning_ru_review_report(video_id))

    paths, warning = pipeline.persist(
        metadata,
        source_transcript.to_dict(),
        source_transcript,
        analysis_ru,
        working_transcript=russian_transcript,
        transcript_translated=True,
        analysis_translated=True,
    )

    assert paths.note.exists()
    assert graph_store.payloads == []
    assert "Graph sync skipped because strict mode has review warnings" in warning


def test_persist_ru_review_failure_blocks_graph_payload_construction(tmp_path, monkeypatch):
    graph_store = RecordingGraphStore()
    pipeline = VideoProcessingPipeline.for_test(tmp_path, graph_store=graph_store)
    enable_translation_settings(pipeline)
    metadata = VideoMetadata(video_id="abc123def45", canonical_url="https://youtu.be/abc123def45", title="Talk")
    source_transcript = Transcript(
        video_id="abc123def45",
        source="youtube",
        language="en",
        segments=[TranscriptSegment(0, 1, "Graph text")],
    )
    russian_transcript = Transcript(
        video_id="abc123def45",
        source="youtube",
        language="ru",
        segments=[TranscriptSegment(0, 1, "Русский граф", raw_text="Graph text")],
    )
    analysis_ru = sample_analysis(language="ru", abstract="Русский анализ", concept="Граф", claim_text="Граф помогает.")
    monkeypatch.setattr("yt_sub_hub.pipeline.review_ru_artifacts", lambda video_id, store, settings: failed_ru_review_report(video_id))
    monkeypatch.setattr(
        "yt_sub_hub.pipeline.graph_payload_from_extraction",
        lambda extraction: (_ for _ in ()).throw(AssertionError("graph payload should not be built after failed RU review")),
    )

    with pytest.raises(RuntimeError, match="RU artifact review hard failures: stale_analysis_ru"):
        pipeline.persist(
            metadata,
            source_transcript.to_dict(),
            source_transcript,
            analysis_ru,
            working_transcript=russian_transcript,
            transcript_translated=True,
            analysis_translated=True,
        )

    assert graph_store.payloads == []


def test_process_ru_review_failure_marks_legacy_job_failed_and_skips_graph_sync(tmp_path, monkeypatch):
    video_id = "abc123def45"
    graph_store = RecordingGraphStore()
    pipeline = VideoProcessingPipeline.for_test(tmp_path, graph_store=graph_store)
    enable_translation_settings(pipeline)
    object.__setattr__(pipeline.settings, "graph_best_effort", True)
    metadata = VideoMetadata(video_id=video_id, canonical_url=f"https://youtu.be/{video_id}", title="Talk")
    source_transcript = Transcript(
        video_id=video_id,
        source="youtube",
        language="en",
        segments=[TranscriptSegment(0, 5, "Graph text", raw_text=" Graph text ")],
    )
    russian_transcript = Transcript(
        video_id=video_id,
        source="youtube",
        language="ru",
        segments=[TranscriptSegment(0, 5, "Русский граф", raw_text="Graph text")],
    )
    analysis_ru = sample_analysis(language="ru", abstract="Русский анализ", concept="Граф", claim_text="Граф помогает.")

    class TranslationProvider:
        def translate_transcript(self, transcript, target_language):
            return russian_transcript

    class RussianAnalyzer:
        def analyze(self, metadata, transcript):
            return analysis_ru

    monkeypatch.setattr("yt_sub_hub.pipeline.fetch_video_metadata", lambda url, settings=None: metadata)
    monkeypatch.setattr(
        "yt_sub_hub.pipeline.resolve_transcript_bundle",
        lambda video_id, transcript_provider=None, whisper_provider=None: type(
            "Bundle",
            (),
            {
                "video_id": video_id,
                "source": "youtube",
                "raw_payload": {"source": "payload"},
                "normalized": source_transcript,
            },
        )(),
    )
    monkeypatch.setattr("yt_sub_hub.pipeline.review_ru_artifacts", lambda video_id, store, settings: failed_ru_review_report(video_id))
    pipeline.translation_provider = TranslationProvider()
    pipeline.analyzer = RussianAnalyzer()

    with pytest.raises(RuntimeError, match="RU artifact review hard failures: stale_analysis_ru"):
        pipeline.process(f"https://youtu.be/{video_id}")

    record = pipeline.store.load_jobs()[job_id_for_url(f"https://youtu.be/{video_id}")]
    assert record.status == "failed"
    assert record.stage == "failed"
    assert "stale_analysis_ru" in record.error
    assert graph_store.payloads == []
    assert [operation[0] for operation in graph_store.operations] == []
    assert pipeline.already_processed(f"https://youtu.be/{video_id}") is None


def test_process_job_reuses_cached_metadata_and_transcript_when_youtube_is_blocked(tmp_path, monkeypatch):
    video_id = "abc123def45"
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    metadata = VideoMetadata(
        video_id=video_id,
        canonical_url=f"https://www.youtube.com/watch?v={video_id}",
        title="Cached Talk",
    )
    transcript = Transcript(
        video_id=video_id,
        source="whisper",
        segments=[TranscriptSegment(0, 10, "Cached transcript text.")],
    )
    old_analysis = Analysis(
        language="en",
        cleaned_title="Old",
        abstract="Old",
        tags=[],
        outline=[],
        sections=[NoteSection(heading="Old", summary="Old", timestamp_seconds=0, bullets=[])],
        concepts=[],
        claims=[],
        limitations=[],
        open_questions=[],
        related_topics=[],
    )
    pipeline.store.save_video_artifacts(metadata, transcript, transcript, old_analysis, "old note")

    def fail_metadata(url, settings=None):
        raise RuntimeError("bot check")

    def fail_transcript(video_id, whisper_provider=None):
        raise TranscriptResolutionError("download blocked")

    class CachedAnalyzer:
        def analyze(self, metadata, transcript):
            assert metadata.title == "Cached Talk"
            assert transcript.segments[0].text == "Cached transcript text."
            return Analysis(
                language="en",
                cleaned_title="Cached Talk",
                abstract="Fresh analysis",
                tags=[],
                outline=["Cached section"],
                sections=[NoteSection(heading="Cached section", summary="Fresh", timestamp_seconds=0, bullets=[])],
                concepts=[],
                claims=[],
                limitations=[],
                open_questions=[],
                related_topics=[],
            )

    monkeypatch.setattr("yt_sub_hub.pipeline.fetch_video_metadata", fail_metadata)
    monkeypatch.setattr("yt_sub_hub.pipeline.resolve_transcript_bundle", fail_transcript)
    pipeline.analyzer = CachedAnalyzer()

    record = pipeline.process(f"https://youtu.be/{video_id}")

    assert record.note_path is not None
    assert record.status == "completed_with_graph_warning"
    assert "Used cached metadata" in record.warning
    assert "Used cached transcript" in record.warning
    assert "Fresh analysis" in (tmp_path / "videos" / video_id / "note.md").read_text()


def test_process_job_prefers_cached_transcript_after_metadata_fetch(tmp_path, monkeypatch):
    video_id = "abc123def45"
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    metadata = VideoMetadata(
        video_id=video_id,
        canonical_url=f"https://www.youtube.com/watch?v={video_id}",
        title="Cached Talk",
    )
    transcript = Transcript(
        video_id=video_id,
        source="whisper",
        segments=[TranscriptSegment(0, 10, "Cached transcript text.")],
    )
    old_analysis = Analysis(
        language="en",
        cleaned_title="Old",
        abstract="Old",
        tags=[],
        outline=[],
        sections=[NoteSection(heading="Old", summary="Old", timestamp_seconds=0, bullets=[])],
        concepts=[],
        claims=[],
        limitations=[],
        open_questions=[],
        related_topics=[],
    )
    pipeline.store.save_video_artifacts(metadata, transcript, transcript, old_analysis, "old note")

    class CachedAnalyzer:
        def analyze(self, metadata, transcript):
            assert transcript.segments[0].text == "Cached transcript text."
            return old_analysis

    def fail_if_called(video_id, whisper_provider=None):
        raise AssertionError("live transcript resolver should not run when cached transcript exists")

    monkeypatch.setattr("yt_sub_hub.pipeline.fetch_video_metadata", lambda url, settings=None: metadata)
    monkeypatch.setattr("yt_sub_hub.pipeline.resolve_transcript_bundle", fail_if_called)
    pipeline.analyzer = CachedAnalyzer()

    record = pipeline.process(f"https://youtu.be/{video_id}")

    assert record.status == "completed"
    assert record.warning is None


def test_pipeline_youtube_caption_failure_uses_whisper_fallback_and_writes_source_transcripts(tmp_path, monkeypatch):
    video_id = "abc123def45"
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    pipeline.settings = replace(pipeline.settings, acceptance_mode="full_processing", media_download_enabled=True)
    metadata = VideoMetadata(
        video_id=video_id,
        canonical_url=f"https://www.youtube.com/watch?v={video_id}",
        title="Fallback Talk",
    )
    raw_payload = {
        "language": "en",
        "segments": [{"start": 0.0, "end": 1.0, "text": " Whisper source text "}],
    }
    normalized = Transcript(
        video_id=video_id,
        source="whisper",
        language="en",
        segments=[TranscriptSegment(0.0, 1.0, "Whisper source text", " Whisper source text ")],
    )

    class Analyzer:
        def analyze(self, metadata, transcript):
            assert transcript.source == "whisper"
            assert transcript.segments[0].text == "Whisper source text"
            return sample_analysis()

    def fail_youtube(self, requested_video_id):
        raise TranscriptResolutionError("no captions")

    def fake_whisper_bundle(self, requested_video_id):
        assert requested_video_id == video_id
        return type(
            "Bundle",
            (),
            {
                "video_id": video_id,
                "source": "whisper",
                "raw_payload": raw_payload,
                "normalized": normalized,
            },
        )()

    monkeypatch.setattr("yt_sub_hub.pipeline.fetch_video_metadata", lambda url, settings=None: metadata)
    monkeypatch.setattr("yt_sub_hub.transcripts.YouTubeTranscriptProvider.fetch_bundle", fail_youtube)
    monkeypatch.setattr("yt_sub_hub.transcripts.WhisperAudioProvider.transcribe_bundle", fake_whisper_bundle)
    pipeline.analyzer = Analyzer()

    record = pipeline.process(f"https://youtu.be/{video_id}")

    raw_path = tmp_path / "videos" / video_id / "transcript.raw.json"
    normalized_path = tmp_path / "videos" / video_id / "transcript.normalized.json"
    assert record.status == "completed"
    assert raw_path.exists()
    assert normalized_path.exists()
    assert '" Whisper source text "' in raw_path.read_text(encoding="utf-8")
    assert '"source": "whisper"' in normalized_path.read_text(encoding="utf-8")
    assert not (tmp_path / "videos" / video_id / "transcript.ru.json").exists()


def test_legacy_data_api_only_does_not_call_unofficial_transcript_provider(tmp_path, monkeypatch):
    import yt_sub_hub.transcripts as transcripts

    video_id = "abc123def45"
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    pipeline.settings = replace(
        pipeline.settings,
        youtube_api_token_file="token.json",
        acceptance_mode="data_api_only",
        media_download_enabled=False,
    )
    metadata = VideoMetadata(
        video_id=video_id,
        canonical_url=f"https://www.youtube.com/watch?v={video_id}",
        title="Data API only",
    )

    def fail_data_api(self, requested_video_id):
        raise transcripts.TranscriptSourceError("no official captions", reason="captions_unavailable")

    def unofficial_must_not_run(self, requested_video_id):
        raise AssertionError("legacy data_api_only must not use youtube-transcript-api")

    monkeypatch.setattr("yt_sub_hub.pipeline.fetch_video_metadata", lambda url, settings=None: metadata)
    monkeypatch.setattr("yt_sub_hub.transcripts.YouTubeDataApiCaptionProvider.fetch_bundle", fail_data_api)
    monkeypatch.setattr("yt_sub_hub.transcripts.YouTubeTranscriptProvider.fetch_bundle", unofficial_must_not_run)

    with pytest.raises(TranscriptResolutionError, match="no official captions"):
        pipeline.process(f"https://youtu.be/{video_id}")


def test_process_job_translation_mode_persists_russian_artifacts_and_graph(tmp_path, monkeypatch):
    video_id = "abc123def45"
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    enable_translation_settings(pipeline)
    object.__setattr__(pipeline.settings, "enable_deep_graph_llm", True)
    metadata = VideoMetadata(video_id=video_id, canonical_url=f"https://youtu.be/{video_id}", title="Talk")
    source_transcript = Transcript(
        video_id=video_id,
        source="youtube",
        language="en",
        segments=[TranscriptSegment(0, 5, "Graph text", raw_text=" Graph text ")],
    )
    russian_transcript = Transcript(
        video_id=video_id,
        source="youtube",
        language="ru",
        segments=[TranscriptSegment(0, 5, "Русский граф", raw_text="Graph text")],
    )
    analysis_ru = sample_analysis(language="ru", abstract="Русский анализ", concept="Граф", claim_text="Граф помогает.")

    class TranslationProvider:
        def __init__(self):
            self.calls = []

        def translate_transcript(self, transcript, target_language):
            self.calls.append((transcript.language, target_language, transcript.segments[0].text))
            return russian_transcript

    class RussianAnalyzer:
        def __init__(self):
            self.analysis_transcripts = []
            self.deep_graph_transcripts = []

        def analyze(self, metadata, transcript):
            self.analysis_transcripts.append(transcript)
            assert transcript.language == "ru"
            assert transcript.segments[0].text == "Русский граф"
            return analysis_ru

        def extract_deep_graph_chunks(self, metadata, transcript, max_chunk_chars):
            self.deep_graph_transcripts.append(transcript)
            assert transcript.language == "ru"
            assert transcript.segments[0].text == "Русский граф"
            return [deep_graph_chunk_with_relationships(metadata.video_id)]

    monkeypatch.setattr("yt_sub_hub.pipeline.fetch_video_metadata", lambda url, settings=None: metadata)
    monkeypatch.setattr(
        "yt_sub_hub.pipeline.resolve_transcript_bundle",
        lambda video_id, transcript_provider=None, whisper_provider=None: type(
            "Bundle",
            (),
            {
                "video_id": video_id,
                "source": "youtube",
                "raw_payload": {"source": "payload"},
                "normalized": source_transcript,
            },
        )(),
    )
    pipeline.translation_provider = TranslationProvider()
    pipeline.analyzer = RussianAnalyzer()

    record = pipeline.process(f"https://youtu.be/{video_id}")
    paths = pipeline.store.artifact_paths(video_id)
    graph_extraction = pipeline.store.load_graph_extraction(video_id)

    assert record.status == "completed"
    assert pipeline.translation_provider.calls == [("en", "ru", "Graph text")]
    assert pipeline.store.load_normalized_transcript(video_id).segments[0].text == "Graph text"
    assert pipeline.store.load_translated_transcript(video_id).segments[0].text == "Русский граф"
    assert pipeline.store.load_analysis(video_id, variant="ru")["language"] == "ru"
    assert pipeline.store.load_analysis(video_id)["language"] == "ru"
    assert "Русский анализ" in paths.note.read_text(encoding="utf-8")
    assert graph_extraction["generated_from"]["transcript"] == "transcript.ru.json"
    assert graph_extraction["generated_from"]["analysis"] == "analysis.ru.json"
    assert any(node.get("text") == "Русский граф" for node in graph_extraction["nodes"])
    assert pipeline.analyzer.analysis_transcripts == [russian_transcript]
    assert pipeline.analyzer.deep_graph_transcripts == [russian_transcript]


def test_process_translation_refresh_uses_cached_metadata_without_live_fetch(tmp_path, monkeypatch):
    video_id = "abc123def45"
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    enable_translation_settings(pipeline)
    metadata = VideoMetadata(video_id=video_id, canonical_url=f"https://youtu.be/{video_id}", title="Cached Talk")
    source_transcript = Transcript(video_id, "youtube", [TranscriptSegment(0, 1, "hola")], language="es")
    russian_transcript = Transcript(video_id, "youtube", [TranscriptSegment(0, 1, "привет")], language="ru")
    analysis_ru = sample_analysis(language="ru", abstract="Русский анализ", concept="Приветствие", claim_text="Приветствие важно.")
    pipeline.store.save_video_artifacts(
        metadata,
        source_transcript,
        source_transcript,
        sample_analysis(language="es", abstract="source"),
        "source note",
    )

    fetch_calls = []

    def live_metadata_fetch(url, settings=None):
        fetch_calls.append(url)
        raise RuntimeError("live metadata fetch should be skipped for translation refresh")

    class Translator:
        def translate_transcript(self, transcript, target_language):
            return russian_transcript

    class Analyzer:
        def analyze(self, metadata, transcript):
            assert metadata.title == "Cached Talk"
            assert transcript.language == "ru"
            return analysis_ru

    def fail_transcript(video_id, whisper_provider=None):
        raise AssertionError("live transcript resolver should not run when cached transcript exists")

    monkeypatch.setattr("yt_sub_hub.pipeline.fetch_video_metadata", live_metadata_fetch)
    monkeypatch.setattr("yt_sub_hub.pipeline.resolve_transcript_bundle", fail_transcript)
    pipeline.translation_provider = Translator()
    pipeline.analyzer = Analyzer()

    record = pipeline.process(f"https://youtu.be/{video_id}")

    assert record.status == "completed"
    assert record.warning is None
    assert fetch_calls == []
    assert pipeline.store.load_translated_transcript(video_id).segments[0].text == "привет"


def test_process_translation_refresh_skips_live_fetch_when_ru_analysis_is_wrong_language(tmp_path, monkeypatch):
    video_id = "abc123def45"
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    enable_translation_settings(pipeline)
    metadata = VideoMetadata(video_id=video_id, canonical_url=f"https://youtu.be/{video_id}", title="Cached Talk")
    source_transcript = Transcript(video_id, "youtube", [TranscriptSegment(0, 1, "hola")], language="es")
    russian_transcript = Transcript(video_id, "youtube", [TranscriptSegment(0, 1, "привет")], language="ru")
    pipeline.store.save_video_artifacts(
        metadata,
        source_transcript,
        source_transcript,
        sample_analysis(language="es", abstract="source"),
        "source note",
    )
    pipeline.store.save_translated_transcript(video_id, russian_transcript)
    pipeline.store.save_analysis_artifact(
        video_id,
        sample_analysis(language="unknown", abstract="English analysis"),
        variant="ru",
        compatibility=True,
    )
    fetch_calls = []

    def live_metadata_fetch(url, settings=None):
        fetch_calls.append(url)
        raise RuntimeError("live metadata fetch should be skipped when ru analysis is invalid")

    class Analyzer:
        def analyze(self, metadata, transcript):
            assert transcript.language == "ru"
            return sample_analysis(language="ru", abstract="Русский анализ", concept="Приветствие", claim_text="Приветствие важно.")

    monkeypatch.setattr("yt_sub_hub.pipeline.fetch_video_metadata", live_metadata_fetch)
    pipeline.analyzer = Analyzer()

    record = pipeline.process(f"https://youtu.be/{video_id}")

    assert record.status == "completed"
    assert fetch_calls == []
    assert pipeline.store.load_analysis(video_id, variant="ru")["language"] == "ru"


def test_process_translation_mode_translates_non_ru_analysis_before_persist(tmp_path, monkeypatch):
    video_id = "abc123def45"
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    enable_translation_settings(pipeline)
    metadata = VideoMetadata(video_id=video_id, canonical_url=f"https://youtu.be/{video_id}", title="Talk")
    source_transcript = Transcript(video_id, "youtube", [TranscriptSegment(0, 1, "hola")], language="es")
    russian_transcript = Transcript(video_id, "youtube", [TranscriptSegment(0, 1, "привет")], language="ru")
    english_analysis = sample_analysis(language="unknown", abstract="English analysis", concept="Greeting", claim_text="Greeting matters.")
    russian_analysis = sample_analysis(language="ru", abstract="Русский анализ", concept="Приветствие", claim_text="Приветствие важно.")

    class TranslationProvider:
        def __init__(self):
            self.analysis_calls = []

        def translate_transcript(self, transcript, target_language):
            return russian_transcript

        def translate_analysis(self, analysis, target_language):
            self.analysis_calls.append((analysis.language, target_language, analysis.abstract))
            return russian_analysis

    class Analyzer:
        def analyze(self, metadata, transcript):
            assert transcript.language == "ru"
            return english_analysis

    monkeypatch.setattr("yt_sub_hub.pipeline.fetch_video_metadata", lambda url, settings=None: metadata)
    monkeypatch.setattr(
        "yt_sub_hub.pipeline.resolve_transcript_bundle",
        lambda video_id, transcript_provider=None, whisper_provider=None: type(
            "Bundle",
            (),
            {"video_id": video_id, "source": "youtube", "raw_payload": source_transcript.to_dict(), "normalized": source_transcript},
        )(),
    )
    pipeline.translation_provider = TranslationProvider()
    pipeline.analyzer = Analyzer()

    record = pipeline.process(f"https://youtu.be/{video_id}")
    graph_extraction = pipeline.store.load_graph_extraction(video_id)

    assert record.status == "completed"
    assert pipeline.translation_provider.analysis_calls == [("unknown", "ru", "English analysis")]
    assert pipeline.store.load_analysis(video_id, variant="ru")["language"] == "ru"
    assert pipeline.store.load_analysis(video_id, variant="ru")["abstract"] == "Русский анализ"
    assert "Русский анализ" in pipeline.store.artifact_paths(video_id).note.read_text(encoding="utf-8")
    assert any(node.get("name") == "Приветствие" for node in graph_extraction["nodes"])
    assert any(node.get("text") == "Приветствие важно." for node in graph_extraction["nodes"])


def test_translation_does_not_save_non_ru_analysis_as_analysis_ru(tmp_path, monkeypatch):
    video_id = "abc123def45"
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    enable_translation_settings(pipeline, best_effort=False)
    metadata = VideoMetadata(video_id=video_id, canonical_url=f"https://youtu.be/{video_id}", title="Talk")
    source_transcript = Transcript(video_id, "youtube", [TranscriptSegment(0, 1, "hello")], language="en")
    russian_transcript = Transcript(video_id, "youtube", [TranscriptSegment(0, 1, "привет")], language="ru")
    english_analysis = sample_analysis(
        language="unknown",
        abstract="English analysis",
        concept="Greeting",
        claim_text="Greeting matters.",
    )

    class Translator:
        def translate_transcript(self, transcript, target_language):
            return russian_transcript

        def translate_analysis(self, analysis, target_language):
            return english_analysis

    class Analyzer:
        def analyze(self, metadata, transcript):
            assert transcript.language == "ru"
            return english_analysis

    monkeypatch.setattr("yt_sub_hub.pipeline.fetch_video_metadata", lambda url, settings=None: metadata)
    monkeypatch.setattr(
        "yt_sub_hub.pipeline.resolve_transcript_bundle",
        lambda video_id, transcript_provider=None, whisper_provider=None: type(
            "Bundle",
            (),
            {
                "video_id": video_id,
                "source": "youtube",
                "raw_payload": source_transcript.to_dict(),
                "normalized": source_transcript,
            },
        )(),
    )
    pipeline.translation_provider = Translator()
    pipeline.analyzer = Analyzer()

    with pytest.raises(TranslationError, match="Russian analysis was not produced"):
        pipeline.process(f"https://youtu.be/{video_id}")

    paths = pipeline.store.artifact_paths(video_id)
    assert not paths.analysis_ru.exists()
    assert not paths.note.exists()
    failed_record = next(iter(pipeline.store.load_jobs().values()))
    assert failed_record.status == "failed"


def test_translation_best_effort_does_not_label_graph_as_ru_when_analysis_translation_fails(tmp_path, monkeypatch):
    video_id = "abc123def45"
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    enable_translation_settings(pipeline, best_effort=True)
    metadata = VideoMetadata(video_id=video_id, canonical_url=f"https://youtu.be/{video_id}", title="Talk")
    source_transcript = Transcript(video_id, "youtube", [TranscriptSegment(0, 1, "hello")], language="en")
    russian_transcript = Transcript(video_id, "youtube", [TranscriptSegment(0, 1, "привет")], language="ru")
    english_analysis = sample_analysis(
        language="ru",
        abstract="English analysis",
        concept="Greeting",
        claim_text="Greeting matters.",
    )

    class Translator:
        def translate_transcript(self, transcript, target_language):
            return russian_transcript

        def translate_analysis(self, analysis, target_language):
            return english_analysis

    class Analyzer:
        def analyze(self, metadata, transcript):
            assert transcript.language == "ru"
            return english_analysis

    monkeypatch.setattr("yt_sub_hub.pipeline.fetch_video_metadata", lambda url, settings=None: metadata)
    monkeypatch.setattr(
        "yt_sub_hub.pipeline.resolve_transcript_bundle",
        lambda video_id, transcript_provider=None, whisper_provider=None: type(
            "Bundle",
            (),
            {
                "video_id": video_id,
                "source": "youtube",
                "raw_payload": source_transcript.to_dict(),
                "normalized": source_transcript,
            },
        )(),
    )
    pipeline.translation_provider = Translator()
    pipeline.analyzer = Analyzer()

    record = pipeline.process(f"https://youtu.be/{video_id}")
    paths = pipeline.store.artifact_paths(video_id)
    graph_extraction = pipeline.store.load_graph_extraction(video_id)

    assert record.status == "completed_with_graph_warning"
    assert "Russian analysis was not produced" in record.warning
    assert paths.transcript_ru.exists()
    assert not paths.analysis_ru.exists()
    assert graph_extraction["generated_from"]["transcript"] == "transcript.ru.json"
    assert graph_extraction["generated_from"]["analysis"] != "analysis.ru.json"
    assert graph_extraction["generated_from"]["language"] != "ru"


def test_persist_saves_analysis_ru_only_when_analysis_passes_target_gate(tmp_path):
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    enable_translation_settings(pipeline, best_effort=True)
    metadata_ru = VideoMetadata(video_id="ru123", canonical_url="https://youtu.be/ru123", title="Talk")
    metadata_en = VideoMetadata(video_id="en123", canonical_url="https://youtu.be/en123", title="Talk")
    source_transcript_ru = Transcript("ru123", "youtube", [TranscriptSegment(0, 1, "hola")], language="es")
    source_transcript_en = Transcript("en123", "youtube", [TranscriptSegment(0, 1, "hello")], language="en")
    russian_transcript = Transcript("ru123", "youtube", [TranscriptSegment(0, 1, "Русский текст")], language="ru")
    cached_russian_transcript = Transcript("en123", "youtube", [TranscriptSegment(0, 1, "Русский текст")], language="ru")
    russian_analysis = Analysis(
        language="ru",
        cleaned_title="Разбор",
        abstract="Русский анализ рынка.",
        tags=["рынок"],
        outline=["Секция"],
        sections=[NoteSection("Секция", "Русское резюме", 0, [])],
        concepts=[Concept("Поддержка", "Зона спроса")],
        claims=[Claim("Поддержка важна.", "Русский текст")],
        limitations=[],
        open_questions=[],
        related_topics=[],
    )
    english_analysis = sample_analysis(
        language="unknown",
        abstract="English analysis",
        concept="Support",
        claim_text="Support matters.",
    )

    pipeline.persist(
        metadata_ru,
        source_transcript_ru,
        source_transcript_ru,
        russian_analysis,
        working_transcript=russian_transcript,
        transcript_translated=True,
        analysis_translated=True,
    )
    ru_paths = pipeline.store.artifact_paths("ru123")
    ru_graph = pipeline.store.load_graph_extraction("ru123")

    assert ru_paths.analysis_ru.exists()
    assert "Русский анализ рынка" in ru_paths.note.read_text(encoding="utf-8")
    assert ru_graph["generated_from"]["transcript"] == "transcript.ru.json"
    assert ru_graph["generated_from"]["analysis"] == "analysis.ru.json"
    assert ru_graph["generated_from"]["language"] == "ru"

    pipeline.persist(
        metadata_en,
        source_transcript_en,
        source_transcript_en,
        english_analysis,
        working_transcript=cached_russian_transcript,
        transcript_translated=True,
        analysis_translated=False,
    )
    en_paths = pipeline.store.artifact_paths("en123")
    en_graph = pipeline.store.load_graph_extraction("en123")

    assert en_paths.transcript_ru.exists()
    assert not en_paths.analysis_ru.exists()
    assert en_graph["generated_from"]["transcript"] == "transcript.ru.json"
    assert en_graph["generated_from"]["analysis"] == "analysis.source.json"
    assert en_graph["generated_from"]["language"] != "ru"


def test_cached_ru_transcript_retry_keeps_source_normalized_untouched(tmp_path, monkeypatch):
    video_id = "abc123def45"
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    enable_translation_settings(pipeline, best_effort=False)
    metadata = VideoMetadata(video_id=video_id, canonical_url=f"https://youtu.be/{video_id}", title="Cached Talk")
    source_transcript = Transcript(video_id, "youtube", [TranscriptSegment(0, 1, "hola")], language="es")
    cached_russian_transcript = Transcript(video_id, "youtube", [TranscriptSegment(0, 1, "привет")], language="ru")
    pipeline.store.save_video_artifacts(
        metadata,
        source_transcript,
        source_transcript,
        sample_analysis(language="es", abstract="source"),
        "source note",
    )
    pipeline.store.save_translated_transcript(video_id, cached_russian_transcript)
    pipeline.store.save_analysis_artifact(
        video_id,
        sample_analysis(language="unknown", abstract="English analysis"),
        variant="ru",
        compatibility=True,
    )

    class Translator:
        def translate_transcript(self, transcript, target_language):
            raise AssertionError("cached transcript.ru.json should be reused")

        def translate_analysis(self, analysis, target_language):
                return Analysis(
                    language="ru",
                    cleaned_title="Разбор",
                    abstract="Русский анализ рынка.",
                    tags=["приветствие"],
                    outline=["Секция"],
                sections=[NoteSection("Секция", "Русское резюме", 0, [])],
                concepts=[Concept("Приветствие", "Русское определение")],
                claims=[Claim("Приветствие важно.", "привет")],
                limitations=[],
                open_questions=[],
                related_topics=[],
            )

    class Analyzer:
        def analyze(self, metadata, transcript):
            assert transcript.language == "ru"
            assert transcript.segments[0].text == "привет"
            return sample_analysis(language="unknown", abstract="English analysis")

    def fail_transcript(video_id, whisper_provider=None):
        raise AssertionError("live transcript resolver should not run when cached source transcript exists")

    monkeypatch.setattr("yt_sub_hub.pipeline.fetch_video_metadata", lambda url, settings=None: metadata)
    monkeypatch.setattr("yt_sub_hub.pipeline.resolve_transcript_bundle", fail_transcript)
    pipeline.translation_provider = Translator()
    pipeline.analyzer = Analyzer()

    record = pipeline.process(f"https://youtu.be/{video_id}")

    assert record.status == "completed"
    assert pipeline.store.load_normalized_transcript(video_id).language == "es"
    assert pipeline.store.load_normalized_transcript(video_id).segments[0].text == "hola"
    assert pipeline.store.load_translated_transcript(video_id).language == "ru"
    assert pipeline.store.load_analysis(video_id, variant="ru")["abstract"] == "Русский анализ рынка."


def test_retry_with_only_cached_ru_transcript_creates_downstream_artifacts_without_corrupting_source(tmp_path, monkeypatch):
    video_id = "abc123def45"
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    enable_translation_settings(pipeline, best_effort=False)
    metadata = VideoMetadata(video_id=video_id, canonical_url=f"https://youtu.be/{video_id}", title="Cached Talk")
    source_transcript = Transcript(video_id, "youtube", [TranscriptSegment(0, 1, "hello")], language="en")
    cached_russian_transcript = Transcript(
        video_id,
        "youtube",
        [TranscriptSegment(0, 1, "привет", raw_text="hello")],
        language="ru",
    )
    pipeline.store.save_translated_transcript(video_id, cached_russian_transcript)

    class Translator:
        def translate_transcript(self, transcript, target_language):
            raise AssertionError("transcript.ru.json should be reused")

        def translate_analysis(self, analysis, target_language):
            return Analysis(
                language="ru",
                cleaned_title="Разбор",
                abstract="Русский анализ видео.",
                tags=["видео"],
                outline=["Смысл"],
                sections=[NoteSection("Смысл", "Русское резюме", 0, [])],
                concepts=[Concept("Приветствие", "Русское определение")],
                claims=[Claim("Приветствие важно.", "привет")],
                limitations=[],
                open_questions=[],
                related_topics=[],
            )

    class Analyzer:
        def analyze(self, metadata, transcript):
            assert transcript.language == "ru"
            assert transcript.segments[0].text == "привет"
            return sample_analysis(language="unknown", abstract="English analysis")

    monkeypatch.setattr("yt_sub_hub.pipeline.fetch_video_metadata", lambda url, settings=None: metadata)
    monkeypatch.setattr(
        "yt_sub_hub.pipeline.resolve_transcript_bundle",
        lambda video_id, transcript_provider=None, whisper_provider=None: type(
            "Bundle",
            (),
            {
                "video_id": video_id,
                "source": "youtube",
                "raw_payload": source_transcript.to_dict(),
                "normalized": source_transcript,
            },
        )(),
    )
    pipeline.translation_provider = Translator()
    pipeline.analyzer = Analyzer()

    record = pipeline.process(f"https://youtu.be/{video_id}")
    paths = pipeline.store.artifact_paths(video_id)
    quality = run_quality_gate(pipeline.store, video_id, target_language="ru")

    assert record.status == "completed"
    assert paths.transcript_ru.exists()
    assert paths.analysis_ru.exists()
    assert paths.note.exists()
    assert paths.graph_extraction.exists()
    assert pipeline.store.load_normalized_transcript(video_id).language == "en"
    assert pipeline.store.load_normalized_transcript(video_id).segments[0].text == "hello"
    assert pipeline.store.load_translated_transcript(video_id).segments[0].text == "привет"
    assert quality.hard_failures == []


def test_analysis_language_gate_requires_cyrillic_user_text():
    english_with_ru_label = Analysis(
        language="ru",
        cleaned_title="Talk",
        abstract="English analysis",
        tags=[],
        outline=[],
        sections=[NoteSection("Section", "English summary", 0, [])],
        concepts=[Concept("Support", "Demand zone")],
        claims=[Claim("Support matters.", "Evidence")],
        limitations=[],
        open_questions=[],
        related_topics=[],
    )
    russian = Analysis(
        language="ru",
        cleaned_title="Разбор",
        abstract="Русский анализ рынка.",
        tags=["рынок"],
        outline=["Секция"],
        sections=[NoteSection("Секция", "Русское резюме", 0, [])],
        concepts=[Concept("Поддержка", "Русская зона спроса")],
        claims=[Claim("Поддержка важна для сделки.", "В докладе это объясняется.")],
        limitations=[],
        open_questions=[],
        related_topics=[],
    )
    mixed = Analysis(
        language="ru",
        cleaned_title="Talk",
        abstract="English analysis with one русский token",
        tags=[],
        outline=[],
        sections=[NoteSection("Section", "English summary with слово", 0, [])],
        concepts=[Concept("Support", "Demand zone with термин")],
        claims=[Claim("Support matters with слово.", "Evidence")],
        limitations=[],
        open_questions=[],
        related_topics=[],
    )

    assert _analysis_matches_target_language(english_with_ru_label, "ru") is False
    assert _analysis_matches_target_language(mixed, "ru") is False
    assert _analysis_matches_target_language(russian, "ru") is True


def test_analysis_language_gate_accepts_short_russian_heading_and_rejects_english_heading():
    valid = Analysis(
        language="ru",
        cleaned_title="Разбор",
        abstract="Русская аннотация подробно описывает главный тезис доклада.",
        tags=["тезис"],
        outline=["Обзор"],
        sections=[NoteSection("Обзор", "Русское резюме подробно раскрывает содержание раздела.", 0, [])],
        concepts=[Concept("Тезис", "Ключевая мысль, которую докладчик объясняет в видео.")],
        claims=[Claim("Доклад раскрывает главную тему.", "В докладе подробно объясняется главная тема.")],
        limitations=[],
        open_questions=[],
        related_topics=[],
    )
    invalid = Analysis(
        language="ru",
        cleaned_title="Разбор",
        abstract="Русская аннотация подробно описывает главный тезис доклада.",
        tags=["тезис"],
        outline=["Обзор"],
        sections=[NoteSection("Overview", "Русское резюме подробно раскрывает содержание раздела.", 0, [])],
        concepts=[Concept("Тезис", "Ключевая мысль, которую докладчик объясняет в видео.")],
        claims=[Claim("Доклад раскрывает главную тему.", "В докладе подробно объясняется главная тема.")],
        limitations=[],
        open_questions=[],
        related_topics=[],
    )

    assert _analysis_matches_target_language(valid, "ru") is True
    assert _analysis_matches_target_language(invalid, "ru") is False


def test_cached_transcript_bundle_uses_source_transcript_even_when_ru_exists(tmp_path):
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    enable_translation_settings(pipeline)

    video_id = "abc123"
    source = Transcript(
        video_id=video_id,
        source="youtube",
        language="es",
        segments=[TranscriptSegment(0, 1, "hola")],
    )
    translated = Transcript(
        video_id=video_id,
        source="youtube",
        language="ru",
        segments=[TranscriptSegment(0, 1, "привет")],
    )
    metadata = VideoMetadata(video_id=video_id, canonical_url="https://youtu.be/abc123", title="T")
    analysis = sample_analysis(language="es", abstract="source")

    pipeline.store.save_video_artifacts(metadata, source, source, analysis, "source note")
    pipeline.store.save_translated_transcript(video_id, translated)

    bundle = pipeline._cached_transcript_bundle(video_id)

    assert bundle.normalized.language == "es"
    assert bundle.normalized.segments[0].text == "hola"


def test_translate_transcript_or_source_reuses_existing_ru_transcript(tmp_path):
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    enable_translation_settings(pipeline)

    video_id = "abc123"
    source = Transcript(video_id, "youtube", [TranscriptSegment(0, 1, "hola")], language="es")
    translated = Transcript(video_id, "youtube", [TranscriptSegment(0, 1, "привет")], language="ru")
    pipeline.store.save_translated_transcript(video_id, translated)

    class FailingTranslator:
        def translate_transcript(self, transcript, target_language):
            raise AssertionError("translator should not be called when transcript.ru.json exists")

    pipeline.translation_provider = FailingTranslator()
    warnings = []

    working, translated_flag = pipeline._translate_transcript_or_source(source, warnings)

    assert translated_flag is True
    assert working.language == "ru"
    assert working.segments[0].text == "привет"
    assert warnings == []


def test_process_with_cached_translation_does_not_overwrite_source_transcript(tmp_path, monkeypatch):
    video_id = "abc123def45"
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    enable_translation_settings(pipeline)
    metadata = VideoMetadata(video_id=video_id, canonical_url=f"https://youtu.be/{video_id}", title="Talk")
    source = Transcript(video_id, "youtube", [TranscriptSegment(0, 1, "hola")], language="es")
    translated = Transcript(video_id, "youtube", [TranscriptSegment(0, 1, "привет")], language="ru")
    analysis_ru = sample_analysis(language="ru", abstract="Русский анализ", concept="Приветствие", claim_text="Приветствие важно.")

    pipeline.store.save_video_artifacts(metadata, source, source, sample_analysis(language="es", abstract="source"), "source note")
    pipeline.store.save_translated_transcript(video_id, translated)

    class FailingTranslator:
        def translate_transcript(self, transcript, target_language):
            raise AssertionError("translator should not be called when transcript.ru.json exists")

    class RussianAnalyzer:
        def analyze(self, metadata, transcript):
            assert transcript.language == "ru"
            assert transcript.segments[0].text == "привет"
            return analysis_ru

    def fail_transcript(video_id, whisper_provider=None):
        raise AssertionError("live transcript resolver should not run when cached transcript exists")

    monkeypatch.setattr("yt_sub_hub.pipeline.fetch_video_metadata", lambda url, settings=None: metadata)
    monkeypatch.setattr("yt_sub_hub.pipeline.resolve_transcript_bundle", fail_transcript)
    pipeline.translation_provider = FailingTranslator()
    pipeline.analyzer = RussianAnalyzer()

    record = pipeline.process(f"https://youtu.be/{video_id}")

    assert record.status == "completed"
    assert pipeline.store.load_normalized_transcript(video_id).language == "es"
    assert pipeline.store.load_normalized_transcript(video_id).segments[0].text == "hola"
    assert pipeline.store.load_translated_transcript(video_id).language == "ru"
    assert pipeline.store.load_translated_transcript(video_id).segments[0].text == "привет"
    assert pipeline.store.load_graph_extraction(video_id)["generated_from"]["translated_from"] == "es"


def test_process_job_translation_failure_raises_when_not_best_effort(tmp_path, monkeypatch):
    video_id = "abc123def45"
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    enable_translation_settings(pipeline, best_effort=False)
    metadata = VideoMetadata(video_id=video_id, canonical_url=f"https://youtu.be/{video_id}", title="Talk")
    transcript = Transcript(
        video_id=video_id,
        source="youtube",
        language="en",
        segments=[TranscriptSegment(0, 5, "Graph text")],
    )

    class FailingTranslationProvider:
        def translate_transcript(self, transcript, target_language):
            raise RuntimeError("translator down")

    class Analyzer:
        def analyze(self, metadata, transcript):
            raise AssertionError("analysis should not run after hard translation failure")

    monkeypatch.setattr("yt_sub_hub.pipeline.fetch_video_metadata", lambda url, settings=None: metadata)
    monkeypatch.setattr(
        "yt_sub_hub.pipeline.resolve_transcript_bundle",
        lambda video_id, transcript_provider=None, whisper_provider=None: type(
            "Bundle",
            (),
            {"video_id": video_id, "source": "youtube", "raw_payload": transcript.to_dict(), "normalized": transcript},
        )(),
    )
    pipeline.translation_provider = FailingTranslationProvider()
    pipeline.analyzer = Analyzer()

    with pytest.raises(RuntimeError, match="translator down"):
        pipeline.process(f"https://youtu.be/{video_id}")

    failed_record = next(iter(pipeline.store.load_jobs().values()))
    assert failed_record.status == "failed"
    assert "translator down" in failed_record.error
    assert not pipeline.store.artifact_paths(video_id).analysis.exists()


def test_process_job_translation_best_effort_warns_and_uses_source_fallback(tmp_path, monkeypatch):
    video_id = "abc123def45"
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    enable_translation_settings(pipeline, best_effort=True)
    metadata = VideoMetadata(video_id=video_id, canonical_url=f"https://youtu.be/{video_id}", title="Talk")
    transcript = Transcript(
        video_id=video_id,
        source="youtube",
        language="en",
        segments=[TranscriptSegment(0, 5, "Graph text")],
    )
    analysis = sample_analysis(language="en", abstract="Source analysis")

    class FailingTranslationProvider:
        def translate_transcript(self, transcript, target_language):
            raise RuntimeError("translator down")

    class SourceAnalyzer:
        def __init__(self):
            self.transcripts = []

        def analyze(self, metadata, transcript):
            self.transcripts.append(transcript)
            assert transcript.language == "en"
            return analysis

    monkeypatch.setattr("yt_sub_hub.pipeline.fetch_video_metadata", lambda url, settings=None: metadata)
    monkeypatch.setattr(
        "yt_sub_hub.pipeline.resolve_transcript_bundle",
        lambda video_id, transcript_provider=None, whisper_provider=None: type(
            "Bundle",
            (),
            {"video_id": video_id, "source": "youtube", "raw_payload": transcript.to_dict(), "normalized": transcript},
        )(),
    )
    pipeline.translation_provider = FailingTranslationProvider()
    pipeline.analyzer = SourceAnalyzer()

    record = pipeline.process(f"https://youtu.be/{video_id}")
    paths = pipeline.store.artifact_paths(video_id)

    assert record.status == "completed_with_graph_warning"
    assert "Translation failed" in record.warning
    assert pipeline.analyzer.transcripts == [transcript]
    assert not paths.transcript_ru.exists()
    assert pipeline.store.load_analysis(video_id)["language"] == "en"


def test_submit_marks_stale_running_job_failed_then_requeues_same_url(tmp_path):
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    url = "https://youtu.be/abc123def45"
    job_id = job_id_for_url(url)
    pipeline.store.save_job(
        JobRecord(job_id=job_id, url=url, video_id="abc123def45", status="running", stage="running")
    )
    with sqlite3.connect(pipeline.store.db_path) as conn:
        conn.execute("UPDATE jobs SET updated_at = ? WHERE job_id = ?", (time.time() - 7200, job_id))

    record = pipeline.submit(url)
    loaded = pipeline.store.load_jobs()[job_id]
    history = pipeline.store.load_job_history(job_id)

    assert record.status == "queued"
    assert loaded.status == "queued"
    assert any(item["status"] == "failed" and item["stage"] == "failed" for item in history)
    assert any(item["message"] == "Stale running job marked failed during recovery" for item in history)


def test_submit_preserves_non_stale_running_job(tmp_path):
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    url = "https://youtu.be/abc123def45"
    job_id = job_id_for_url(url)
    pipeline.store.save_job(
        JobRecord(job_id=job_id, url=url, video_id="abc123def45", status="running", stage="metadata_fetched")
    )

    record = pipeline.submit(url)

    assert record.status == "running"
    assert record.stage == "metadata_fetched"
    assert pipeline.store.load_jobs()[job_id].status == "running"


def test_process_returns_non_stale_running_job_without_processing(tmp_path, monkeypatch):
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    url = "https://youtu.be/abc123def45"
    job_id = job_id_for_url(url)
    pipeline.store.save_job(
        JobRecord(job_id=job_id, url=url, video_id="abc123def45", status="running", stage="metadata_fetched")
    )

    def fail_process_job(job_id):
        raise AssertionError("process_job should not run for an existing non-stale running job")

    monkeypatch.setattr(pipeline, "process_job", fail_process_job)

    record = pipeline.process(url)

    assert record.status == "running"
    assert record.stage == "metadata_fetched"


def test_process_job_uses_cached_video_when_refreshing_missing_deep_graph(tmp_path, monkeypatch):
    video_id = "abc123def45"
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    pipeline.settings = pipeline.settings.__class__(
        **{**pipeline.settings.__dict__, "enable_deep_graph_llm": True, "deep_graph_best_effort": True}
    )
    metadata = VideoMetadata(
        video_id=video_id,
        canonical_url=f"https://www.youtube.com/watch?v={video_id}",
        title="Cached Talk",
    )
    transcript = Transcript(
        video_id=video_id,
        source="youtube",
        segments=[TranscriptSegment(0, 10, "Support breakout text.")],
    )
    analysis = Analysis(
        language="en",
        cleaned_title="Cached Talk",
        abstract="Cached",
        tags=[],
        outline=[],
        sections=[],
        concepts=[],
        claims=[],
        limitations=[],
        open_questions=[],
        related_topics=[],
    )
    pipeline.store.save_video_artifacts(metadata, transcript, transcript, analysis, "cached note")
    pipeline.store.save_graph_extraction(
        video_id,
        {
            "schema_version": "graph-extraction-v1",
            "video_id": video_id,
            "chunks": [],
            "nodes": [],
            "relationships": [],
            "quality_report": {},
        },
    )

    class CachedAnalyzer:
        def analyze(self, metadata, transcript):
            raise AssertionError("analysis should be loaded from cache when refreshing deep graph")

        def extract_deep_graph_chunks(self, metadata, transcript, max_chunk_chars):
            return [deep_graph_chunk_with_relationships(metadata.video_id)]

    def fail_metadata(url, settings=None):
        raise AssertionError("metadata fetch should not run when refreshing cached deep graph")

    def fail_transcript(video_id, whisper_provider=None):
        raise AssertionError("transcript resolver should not run when cached transcript exists")

    monkeypatch.setattr("yt_sub_hub.pipeline.fetch_video_metadata", fail_metadata)
    monkeypatch.setattr("yt_sub_hub.pipeline.resolve_transcript_bundle", fail_transcript)
    pipeline.analyzer = CachedAnalyzer()

    record = pipeline.process(f"https://youtu.be/{video_id}")
    extraction = pipeline.store.load_graph_extraction(video_id)

    assert record.status == "completed_with_graph_warning"
    assert "Used cached metadata to refresh deep graph extraction" in record.warning
    assert "Used cached analysis to refresh deep graph extraction" in record.warning
    assert extraction["chunks"]
    assert any(rel.get("extraction_method") == "llm_deep_graph" for rel in extraction["relationships"])


def test_translation_deep_graph_refresh_does_not_use_cached_source_analysis(tmp_path, monkeypatch):
    video_id = "abc123def45"
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    enable_translation_settings(pipeline)
    object.__setattr__(pipeline.settings, "enable_deep_graph_llm", True)
    metadata = VideoMetadata(video_id=video_id, canonical_url=f"https://youtu.be/{video_id}", title="Cached Talk")
    source_transcript = Transcript(video_id, "youtube", [TranscriptSegment(0, 10, "Support breakout text.")], language="en")
    russian_transcript = Transcript(video_id, "youtube", [TranscriptSegment(0, 10, "Русский пробой.")], language="ru")
    source_analysis = sample_analysis(language="en", abstract="Source analysis", concept="Support", claim_text="Support helps.")
    pipeline.persist(metadata, source_transcript, source_transcript, source_analysis)
    pipeline.store.save_translated_transcript(video_id, russian_transcript)
    pipeline.store.save_graph_extraction(
        video_id,
        {
            "schema_version": "graph-extraction-v1",
            "video_id": video_id,
            "chunks": [],
            "nodes": [],
            "relationships": [],
            "quality_report": {},
        },
    )

    class Analyzer:
        def analyze(self, metadata, transcript):
            assert transcript.language == "ru"
            return sample_analysis(language="ru", abstract="Русский анализ", concept="Пробой", claim_text="Пробой важен.")

        def extract_deep_graph_chunks(self, metadata, transcript, max_chunk_chars):
            assert transcript.language == "ru"
            return [deep_graph_chunk_with_relationships(metadata.video_id)]

    def fail_metadata(url, settings=None):
        raise AssertionError("metadata fetch should not run when refreshing cached deep graph")

    def fail_transcript(video_id, whisper_provider=None):
        raise AssertionError("transcript resolver should not run when cached transcript exists")

    monkeypatch.setattr("yt_sub_hub.pipeline.fetch_video_metadata", fail_metadata)
    monkeypatch.setattr("yt_sub_hub.pipeline.resolve_transcript_bundle", fail_transcript)
    pipeline.analyzer = Analyzer()

    record = pipeline.process(f"https://youtu.be/{video_id}")
    analysis_ru = pipeline.store.load_analysis(video_id, variant="ru")

    assert record.status == "completed_with_graph_warning"
    assert analysis_ru["language"] == "ru"
    assert analysis_ru["abstract"] == "Русский анализ"


def test_already_processed_detects_same_video_from_different_url_forms(tmp_path):
    video_id = "abc123def45"
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    metadata = VideoMetadata(
        video_id=video_id,
        canonical_url=f"https://www.youtube.com/watch?v={video_id}",
        title="Cached Talk",
    )
    transcript = Transcript(video_id=video_id, source="whisper", segments=[TranscriptSegment(0, 1, "Hello")])
    analysis = Analysis(
        language="en",
        cleaned_title="Cached Talk",
        abstract="Cached",
        tags=[],
        outline=[],
        sections=[NoteSection(heading="Intro", summary="Cached", timestamp_seconds=0, bullets=[])],
        concepts=[],
        claims=[],
        limitations=[],
        open_questions=[],
        related_topics=[],
    )
    pipeline.store.save_video_artifacts(metadata, transcript, transcript, analysis, "cached note")

    existing = pipeline.already_processed(f"https://youtu.be/{video_id}")

    assert existing == (video_id, str(tmp_path / "videos" / video_id / "note.md"))


def test_already_processed_translation_mode_requires_russian_artifacts(tmp_path):
    video_id = "abc123def45"
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    enable_translation_settings(pipeline)
    metadata = VideoMetadata(
        video_id=video_id,
        canonical_url=f"https://youtu.be/{video_id}",
        title="Cached Talk",
    )
    transcript = Transcript(video_id=video_id, source="youtube", language="en", segments=[TranscriptSegment(0, 1, "Hello")])
    analysis = sample_analysis(language="en", abstract="Source analysis")
    pipeline.store.save_video_artifacts(metadata, transcript, transcript, analysis, "source note")

    assert pipeline.already_processed(f"https://youtu.be/{video_id}") is None

    pipeline.store.save_translated_transcript(
        video_id,
        Transcript(video_id=video_id, source="youtube", language="ru", segments=[TranscriptSegment(0, 1, "Привет")]),
    )
    pipeline.store.save_analysis_artifact(
        video_id,
        sample_analysis(language="ru", abstract="Русский анализ", concept="Приветствие", claim_text="Приветствие важно."),
        variant="ru",
    )

    assert pipeline.already_processed(f"https://youtu.be/{video_id}") == (
        video_id,
        str(tmp_path / "videos" / video_id / "note.md"),
    )


def test_rebuild_graph_prefers_saved_graph_extraction(tmp_path):
    graph_store = RecordingGraphStore()
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    metadata = VideoMetadata(video_id="abc123", canonical_url="https://youtu.be/abc123", title="Talk")
    transcript = Transcript(video_id="abc123", source="youtube", segments=[TranscriptSegment(0, 1, "Graph text")])
    analysis = Analysis(
        language="en",
        cleaned_title="Talk",
        abstract="Abstract",
        tags=[],
        outline=[],
        sections=[NoteSection(heading="Intro", summary="Summary", timestamp_seconds=0, bullets=[])],
        concepts=[Concept(name="Graph", definition="A structure.")],
        claims=[],
        limitations=[],
        open_questions=[],
        related_topics=[],
    )
    pipeline.persist(metadata, transcript, analysis)

    count = pipeline.rebuild_graph(graph_store)

    assert count == 1
    assert graph_store.payloads
    assert any(node["label"] == "SourceSegment" for node in graph_store.payloads[0].nodes)


def test_rebuild_graph_deletes_video_relationships_before_upsert_for_video_id(tmp_path):
    graph_store = RecordingGraphStore()
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    metadata = VideoMetadata(video_id="abc123", canonical_url="https://youtu.be/abc123", title="Talk")
    transcript = Transcript(video_id="abc123", source="youtube", segments=[TranscriptSegment(0, 1, "Graph text")])
    analysis = Analysis(
        language="en",
        cleaned_title="Talk",
        abstract="Abstract",
        tags=[],
        outline=[],
        sections=[NoteSection(heading="Intro", summary="Summary", timestamp_seconds=0, bullets=[])],
        concepts=[Concept(name="Graph", definition="A structure.")],
        claims=[],
        limitations=[],
        open_questions=[],
        related_topics=[],
    )
    pipeline.persist(metadata, transcript, analysis)

    count = pipeline.rebuild_graph(graph_store, video_id="abc123")

    assert count == 1
    assert graph_store.deleted_video_ids == ["abc123"]
    assert [operation[0] for operation in graph_store.operations] == [
        "delete_video_relationships",
        "upsert_payload",
    ]


def test_rebuild_graph_dry_run_does_not_require_or_call_graph_store(tmp_path):
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    metadata = VideoMetadata(video_id="abc123", canonical_url="https://youtu.be/abc123", title="Talk")
    transcript = Transcript(video_id="abc123", source="youtube", segments=[TranscriptSegment(0, 1, "Graph text")])
    analysis = Analysis(
        language="en",
        cleaned_title="Talk",
        abstract="Abstract",
        tags=[],
        outline=[],
        sections=[],
        concepts=[],
        claims=[],
        limitations=[],
        open_questions=[],
        related_topics=[],
    )
    pipeline.store.save_video_artifacts(metadata, transcript, transcript, analysis, "note")

    count = pipeline.rebuild_graph(None, dry_run=True, clear=True)

    assert count == 1
    assert (tmp_path / "videos" / "abc123" / "graph_extraction.json").exists()


def test_rebuild_graph_translation_mode_rebuilds_from_russian_artifacts(tmp_path):
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    enable_translation_settings(pipeline)
    metadata = VideoMetadata(video_id="abc123", canonical_url="https://youtu.be/abc123", title="Talk")
    source_transcript = Transcript(
        video_id="abc123",
        source="youtube",
        language="en",
        segments=[TranscriptSegment(0, 1, "Graph text")],
    )
    russian_transcript = Transcript(
        video_id="abc123",
        source="youtube",
        language="ru",
        segments=[TranscriptSegment(0, 1, "Русский граф")],
    )
    source_analysis = sample_analysis(language="en", abstract="Source analysis")
    russian_analysis = sample_analysis(
        language="ru",
        abstract="Русский анализ",
        concept="Граф",
        claim_text="Граф помогает.",
    )
    pipeline.persist(metadata, source_transcript, source_transcript, source_analysis)
    pipeline.store.save_translated_transcript("abc123", russian_transcript)
    pipeline.store.save_analysis_artifact("abc123", russian_analysis, variant="ru", compatibility=True)

    count = pipeline.rebuild_graph(None, dry_run=True)
    extraction = pipeline.store.load_graph_extraction("abc123")

    assert count == 1
    assert extraction["generated_from"]["transcript"] == "transcript.ru.json"
    assert extraction["generated_from"]["analysis"] == "analysis.ru.json"
    assert any(node.get("text") == "Русский граф" for node in extraction["nodes"])


def test_rebuild_graph_translation_mode_rebuilds_stale_source_extraction(tmp_path):
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    enable_translation_settings(pipeline)
    metadata = VideoMetadata(video_id="abc123", canonical_url="https://youtu.be/abc123", title="Talk")
    source_transcript = Transcript(
        video_id="abc123",
        source="youtube",
        language="en",
        segments=[TranscriptSegment(0, 1, "Graph text")],
    )
    russian_transcript = Transcript(
        video_id="abc123",
        source="youtube",
        language="ru",
        segments=[TranscriptSegment(0, 1, "Русский граф")],
    )
    source_analysis = sample_analysis(language="en", abstract="Source analysis")
    russian_analysis = sample_analysis(language="ru", abstract="Русский анализ", concept="Граф", claim_text="Граф помогает.")
    pipeline.persist(metadata, source_transcript, source_transcript, source_analysis)
    stale = pipeline.store.load_graph_extraction("abc123")
    assert stale["generated_from"]["transcript"] == "transcript.normalized.json"
    pipeline.store.save_translated_transcript("abc123", russian_transcript)
    pipeline.store.save_analysis_artifact("abc123", russian_analysis, variant="ru", compatibility=True)

    pipeline.rebuild_graph(None, dry_run=True)
    extraction = pipeline.store.load_graph_extraction("abc123")

    assert extraction["generated_from"]["transcript"] == "transcript.ru.json"
    assert any(node.get("text") == "Русский граф" for node in extraction["nodes"])
    assert not any(node.get("text") == "Graph text" for node in extraction["nodes"])


def test_rebuild_graph_translation_mode_requires_complete_russian_artifacts_when_present(tmp_path):
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    enable_translation_settings(pipeline)
    metadata = VideoMetadata(video_id="abc123", canonical_url="https://youtu.be/abc123", title="Talk")
    source_transcript = Transcript(
        video_id="abc123",
        source="youtube",
        language="en",
        segments=[TranscriptSegment(0, 1, "Graph text")],
    )
    russian_transcript = Transcript(
        video_id="abc123",
        source="youtube",
        language="ru",
        segments=[TranscriptSegment(0, 1, "Русский граф")],
    )
    pipeline.store.save_video_artifacts(
        metadata,
        source_transcript,
        source_transcript,
        sample_analysis(language="en", abstract="Source analysis"),
        "source note",
    )
    pipeline.store.save_translated_transcript("abc123", russian_transcript)

    with pytest.raises(RuntimeError, match="analysis.ru.json"):
        pipeline.rebuild_graph(None, dry_run=True)


def test_targeted_rebuild_ignores_unrelated_incomplete_translation_artifacts(tmp_path):
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    enable_translation_settings(pipeline)
    complete = VideoMetadata(video_id="complete", canonical_url="https://youtu.be/complete", title="Complete")
    incomplete = VideoMetadata(video_id="incomplete", canonical_url="https://youtu.be/incomplete", title="Incomplete")
    source_transcript = Transcript("complete", "youtube", [TranscriptSegment(0, 1, "Source")], language="en")
    ru_transcript = Transcript("complete", "youtube", [TranscriptSegment(0, 1, "Русский")], language="ru")
    pipeline.store.save_video_artifacts(complete, source_transcript, source_transcript, sample_analysis(), "note")
    pipeline.store.save_translated_transcript("complete", ru_transcript)
    pipeline.store.save_analysis_artifact(
        "complete",
        sample_analysis(language="ru", abstract="Русский", concept="Концепт", claim_text="Концепт важен."),
        variant="ru",
    )
    incomplete_source = Transcript("incomplete", "youtube", [TranscriptSegment(0, 1, "Source")], language="en")
    incomplete_ru = Transcript("incomplete", "youtube", [TranscriptSegment(0, 1, "Русский")], language="ru")
    pipeline.store.save_video_artifacts(incomplete, incomplete_source, incomplete_source, sample_analysis(), "note")
    pipeline.store.save_translated_transcript("incomplete", incomplete_ru)

    assert pipeline.rebuild_graph(None, video_id="complete", dry_run=True) == 1


def test_rebuild_graph_requires_store_when_not_dry_run(tmp_path):
    pipeline = VideoProcessingPipeline.for_test(tmp_path)

    try:
        pipeline.rebuild_graph(None, dry_run=False)
    except ValueError as exc:
        assert "graph_store is required unless dry_run=True" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_persist_merges_deep_graph_chunks_when_enabled(tmp_path):
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    pipeline.settings = pipeline.settings.__class__(
        **{**pipeline.settings.__dict__, "enable_deep_graph_llm": True, "deep_graph_min_confidence": 0.55}
    )

    class DeepAnalyzer:
        calls = []

        def extract_deep_graph_chunks(self, metadata, transcript, max_chunk_chars):
            self.calls.append((metadata.video_id, max_chunk_chars))
            return [deep_graph_chunk_with_relationships(metadata.video_id)]

    pipeline.analyzer = DeepAnalyzer()
    metadata = VideoMetadata(video_id="abc123", canonical_url="https://youtu.be/abc123", title="Talk")
    transcript = Transcript(video_id="abc123", source="youtube", segments=[TranscriptSegment(0, 1, "Graph text")])
    analysis = Analysis(
        language="en",
        cleaned_title="Talk",
        abstract="Abstract",
        tags=[],
        outline=[],
        sections=[],
        concepts=[],
        claims=[],
        limitations=[],
        open_questions=[],
        related_topics=[],
    )

    paths, warning = pipeline.persist(metadata, transcript, analysis)
    extraction = pipeline.store.load_graph_extraction("abc123")

    assert paths.note.exists()
    assert warning is None
    assert extraction["chunks"]
    assert pipeline.analyzer.calls == [("abc123", 2500)]
    llm_rels = [rel for rel in extraction["relationships"] if rel.get("extraction_method") == "llm_deep_graph"]
    assert llm_rels
    assert {"RELATED_TO", "CAUSES", "SUPPORTS", "CONTRASTS_WITH"} <= {rel["type"] for rel in llm_rels}


def test_persist_warns_when_deep_graph_chunks_partially_fail(tmp_path):
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    pipeline.settings = pipeline.settings.__class__(
        **{**pipeline.settings.__dict__, "enable_deep_graph_llm": True, "deep_graph_min_confidence": 0.55}
    )

    class PartiallyFailingAnalyzer:
        def extract_deep_graph_chunks(self, metadata, transcript, max_chunk_chars):
            return [
                deep_graph_chunk_with_relationships(metadata.video_id),
                rejected_deep_graph_chunk(metadata.video_id, "chunk-2", "Invalid deep graph JSON"),
                rejected_deep_graph_chunk(metadata.video_id, "chunk-3", "Invalid deep graph JSON"),
            ]

    pipeline.analyzer = PartiallyFailingAnalyzer()
    metadata = VideoMetadata(video_id="abc123", canonical_url="https://youtu.be/abc123", title="Talk")
    transcript = Transcript(video_id="abc123", source="youtube", segments=[TranscriptSegment(0, 1, "Graph text")])
    analysis = Analysis(
        language="en",
        cleaned_title="Talk",
        abstract="Abstract",
        tags=[],
        outline=[],
        sections=[],
        concepts=[],
        claims=[],
        limitations=[],
        open_questions=[],
        related_topics=[],
    )

    paths, warning = pipeline.persist(metadata, transcript, analysis)
    extraction = pipeline.store.load_graph_extraction("abc123")

    assert paths.note.exists()
    assert warning == "Deep graph partial failure: 2/3 chunks failed"
    assert extraction["quality_report"]["chunks_total"] == 3
    assert extraction["quality_report"]["chunks_succeeded"] == 1
    assert extraction["quality_report"]["chunks_failed"] == 2
    assert any(rel.get("extraction_method") == "llm_deep_graph" for rel in extraction["relationships"])


def test_process_job_marks_partial_deep_graph_failure_as_warning_status(tmp_path, monkeypatch):
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    pipeline.settings = pipeline.settings.__class__(
        **{**pipeline.settings.__dict__, "enable_deep_graph_llm": True, "deep_graph_min_confidence": 0.55}
    )
    metadata = VideoMetadata(video_id="abc123def45", canonical_url="https://youtu.be/abc123def45", title="Talk")
    transcript = Transcript(
        video_id="abc123def45",
        source="youtube",
        segments=[TranscriptSegment(0, 1, "Graph text")],
    )

    class PartiallyFailingAnalyzer:
        def analyze(self, metadata, transcript):
            return Analysis(
                language="en",
                cleaned_title="Talk",
                abstract="Abstract",
                tags=[],
                outline=[],
                sections=[],
                concepts=[],
                claims=[],
                limitations=[],
                open_questions=[],
                related_topics=[],
            )

        def extract_deep_graph_chunks(self, metadata, transcript, max_chunk_chars):
            return [
                deep_graph_chunk_with_relationships(metadata.video_id),
                rejected_deep_graph_chunk(metadata.video_id, "chunk-2", "Invalid deep graph JSON"),
                rejected_deep_graph_chunk(metadata.video_id, "chunk-3", "Invalid deep graph JSON"),
            ]

    monkeypatch.setattr("yt_sub_hub.pipeline.fetch_video_metadata", lambda url, settings=None: metadata)
    monkeypatch.setattr(
        "yt_sub_hub.pipeline.resolve_transcript_bundle",
        lambda video_id, transcript_provider=None, whisper_provider=None: type(
            "Bundle",
            (),
            {"video_id": video_id, "source": "youtube", "raw_payload": transcript.to_dict(), "normalized": transcript},
        )(),
    )
    pipeline.analyzer = PartiallyFailingAnalyzer()

    record = pipeline.process("https://youtu.be/abc123def45")
    extraction = pipeline.store.load_graph_extraction("abc123def45")

    assert record.status == "completed_with_graph_warning"
    assert record.warning == "Deep graph partial failure: 2/3 chunks failed"
    assert extraction["quality_report"]["chunks_failed"] == 2


def test_persist_deep_graph_failure_is_best_effort(tmp_path):
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    pipeline.settings = pipeline.settings.__class__(
        **{**pipeline.settings.__dict__, "enable_deep_graph_llm": True, "deep_graph_best_effort": True}
    )

    class FailingDeepAnalyzer:
        def extract_deep_graph_chunks(self, metadata, transcript, max_chunk_chars):
            raise RuntimeError("llm unavailable")

    pipeline.analyzer = FailingDeepAnalyzer()
    metadata = VideoMetadata(video_id="abc123", canonical_url="https://youtu.be/abc123", title="Talk")
    transcript = Transcript(video_id="abc123", source="youtube", segments=[TranscriptSegment(0, 1, "Graph text")])
    analysis = Analysis(
        language="en",
        cleaned_title="Talk",
        abstract="Abstract",
        tags=[],
        outline=[],
        sections=[],
        concepts=[],
        claims=[],
        limitations=[],
        open_questions=[],
        related_topics=[],
    )

    paths, warning = pipeline.persist(metadata, transcript, analysis)
    extraction = pipeline.store.load_graph_extraction("abc123")

    assert paths.note.exists()
    assert "Deep graph extraction failed" in warning
    assert extraction["chunks"] == []


def test_persist_empty_deep_graph_chunks_records_quality_warning(tmp_path):
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    pipeline.settings = pipeline.settings.__class__(
        **{**pipeline.settings.__dict__, "enable_deep_graph_llm": True, "deep_graph_best_effort": True}
    )

    class EmptyDeepAnalyzer:
        def extract_deep_graph_chunks(self, metadata, transcript, max_chunk_chars):
            return []

    pipeline.analyzer = EmptyDeepAnalyzer()
    metadata = VideoMetadata(video_id="abc123", canonical_url="https://youtu.be/abc123", title="Talk")
    transcript = Transcript(video_id="abc123", source="youtube", segments=[TranscriptSegment(0, 1, "Graph text")])
    analysis = Analysis(
        language="en",
        cleaned_title="Talk",
        abstract="Abstract",
        tags=[],
        outline=[],
        sections=[],
        concepts=[],
        claims=[],
        limitations=[],
        open_questions=[],
        related_topics=[],
    )

    paths, warning = pipeline.persist(metadata, transcript, analysis)
    extraction = pipeline.store.load_graph_extraction("abc123")

    assert paths.note.exists()
    assert "Deep graph extraction failed" in warning
    assert extraction["quality_report"]["chunks_total"] == 0
    assert extraction["quality_report"]["chunks_failed"] == 0
    assert extraction["quality_report"]["deep_graph_error"]


def test_persist_preserves_previous_llm_extraction_when_best_effort_refresh_fails(tmp_path):
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    pipeline.settings = pipeline.settings.__class__(
        **{**pipeline.settings.__dict__, "enable_deep_graph_llm": True, "deep_graph_best_effort": True}
    )

    class DeepAnalyzer:
        def __init__(self):
            self.fail = False

        def extract_deep_graph_chunks(self, metadata, transcript, max_chunk_chars):
            if self.fail:
                raise RuntimeError("llm unavailable")
            return [deep_graph_chunk_with_relationships(metadata.video_id)]

    analyzer = DeepAnalyzer()
    pipeline.analyzer = analyzer
    metadata = VideoMetadata(video_id="abc123", canonical_url="https://youtu.be/abc123", title="Talk")
    transcript = Transcript(video_id="abc123", source="youtube", segments=[TranscriptSegment(0, 1, "Support breakout")])
    analysis = Analysis(
        language="en",
        cleaned_title="Talk",
        abstract="Abstract",
        tags=[],
        outline=[],
        sections=[],
        concepts=[],
        claims=[],
        limitations=[],
        open_questions=[],
        related_topics=[],
    )
    pipeline.persist(metadata, transcript, analysis)
    analyzer.fail = True

    paths, warning = pipeline.persist(metadata, transcript, analysis)
    extraction = pipeline.store.load_graph_extraction("abc123")

    assert paths.note.exists()
    assert "llm unavailable" in warning
    assert extraction["chunks"]
    assert any(rel.get("extraction_method") == "llm_deep_graph" for rel in extraction["relationships"])


def test_deep_graph_source_fallback_is_not_relabelled_as_russian(tmp_path):
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    enable_translation_settings(pipeline, best_effort=True)
    object.__setattr__(pipeline.settings, "enable_deep_graph_llm", True)
    object.__setattr__(pipeline.settings, "deep_graph_best_effort", True)

    class DeepAnalyzer:
        def __init__(self):
            self.fail = False

        def extract_deep_graph_chunks(self, metadata, transcript, max_chunk_chars):
            if self.fail:
                raise RuntimeError("llm unavailable")
            return [deep_graph_chunk_with_relationships(metadata.video_id)]

    analyzer = DeepAnalyzer()
    pipeline.analyzer = analyzer
    metadata = VideoMetadata(video_id="abc123", canonical_url="https://youtu.be/abc123", title="Talk")
    source_transcript = Transcript("abc123", "youtube", [TranscriptSegment(0, 1, "Source support")], language="en")
    russian_transcript = Transcript("abc123", "youtube", [TranscriptSegment(0, 1, "Русская поддержка")], language="ru")
    source_analysis = sample_analysis(language="en", abstract="Source analysis", concept="Support", claim_text="Support matters.")
    russian_analysis = sample_analysis(
        language="ru",
        abstract="Русский анализ.",
        concept="Поддержка",
        claim_text="Поддержка важна.",
    )

    pipeline.persist(
        metadata,
        source_transcript,
        source_transcript,
        source_analysis,
        working_transcript=source_transcript,
        transcript_translated=False,
        analysis_translated=False,
    )
    analyzer.fail = True

    with pytest.raises(RuntimeError, match="RU artifact review hard failures: graph_language_not_ru"):
        pipeline.persist(
            metadata,
            source_transcript,
            source_transcript,
            russian_analysis,
            working_transcript=russian_transcript,
            transcript_translated=True,
            analysis_translated=True,
        )
    paths = pipeline.store.artifact_paths("abc123")
    extraction = pipeline.store.load_graph_extraction("abc123")

    assert paths.note.exists()
    assert paths.ru_artifact_review.exists()
    assert extraction["generated_from"]["transcript"] == "transcript.normalized.json"
    assert extraction["generated_from"]["analysis"] == "analysis.source.json"
    assert extraction["generated_from"]["language"] != "ru"
    assert any(node.get("text") == "Source support" for node in extraction["nodes"])
    assert not any(node.get("text") == "Русская поддержка" for node in extraction["nodes"])


def test_process_uses_translated_transcript_for_analysis_and_graph(tmp_path, monkeypatch):
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    pipeline.settings = pipeline.settings.__class__(
        **{**pipeline.settings.__dict__, "enable_translation": True, "target_language": "ru"}
    )
    metadata = VideoMetadata(video_id="abc123def45", canonical_url="https://youtu.be/abc123def45", title="Talk")
    source_transcript = Transcript(
        video_id="abc123def45",
        source="youtube",
        language="es",
        segments=[TranscriptSegment(0, 3, "Soporte y ruptura.")],
    )
    russian_transcript = Transcript(
        video_id="abc123def45",
        source="youtube",
        language="ru",
        segments=[TranscriptSegment(0, 3, "Поддержка и пробой.")],
    )

    class Translator:
        def translate_transcript(self, transcript, target_language):
            assert transcript.segments[0].text == "Soporte y ruptura."
            return russian_transcript

    class Analyzer:
        def analyze(self, metadata, transcript):
            assert transcript.language == "ru"
            assert transcript.segments[0].text == "Поддержка и пробой."
            return Analysis(
                language="ru",
                cleaned_title="Разбор",
                abstract="Русский обзор",
                tags=["рынок"],
                outline=["Пробой"],
                sections=[NoteSection("Пробой", "Русское резюме", 0, ["Деталь"])],
                concepts=[Concept("Поддержка", "Зона спроса")],
                claims=[Claim("Поддержка важна.", "Поддержка и пробой.")],
                limitations=[],
                open_questions=[],
                related_topics=["Трейдинг"],
            )

    monkeypatch.setattr("yt_sub_hub.pipeline.fetch_video_metadata", lambda url, settings=None: metadata)
    monkeypatch.setattr(
        "yt_sub_hub.pipeline.resolve_transcript_bundle",
        lambda video_id, transcript_provider=None, whisper_provider=None: type(
            "Bundle",
            (),
            {
                "video_id": video_id,
                "source": "youtube",
                "raw_payload": source_transcript.to_dict(),
                "normalized": source_transcript,
            },
        )(),
    )
    pipeline.translation_provider = Translator()
    pipeline.analyzer = Analyzer()

    record = pipeline.process("https://youtu.be/abc123def45")
    extraction = pipeline.store.load_graph_extraction("abc123def45")

    assert record.status == "completed"
    assert (tmp_path / "videos" / "abc123def45" / "transcript.normalized.json").exists()
    assert (tmp_path / "videos" / "abc123def45" / "transcript.ru.json").exists()
    assert (tmp_path / "videos" / "abc123def45" / "analysis.ru.json").exists()
    assert "Русский обзор" in (tmp_path / "videos" / "abc123def45" / "note.md").read_text(encoding="utf-8")
    assert any(node["label"] == "Concept" and node["name"] == "Поддержка" for node in extraction["nodes"])
    assert any(node["label"] == "SourceSegment" and node["text"] == "Поддержка и пробой." for node in extraction["nodes"])
    assert extraction["generated_from"]["transcript"] == "transcript.ru.json"
    assert extraction["generated_from"]["analysis"] == "analysis.ru.json"


def test_translation_failure_fails_pipeline_when_not_best_effort(tmp_path, monkeypatch):
    from yt_sub_hub.translation import TranslationError

    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    pipeline.settings = pipeline.settings.__class__(
        **{
            **pipeline.settings.__dict__,
            "enable_translation": True,
            "target_language": "ru",
            "translation_best_effort": False,
        }
    )
    metadata = VideoMetadata(video_id="abc123def45", canonical_url="https://youtu.be/abc123def45", title="Talk")
    source_transcript = Transcript(
        video_id="abc123def45",
        source="youtube",
        language="es",
        segments=[TranscriptSegment(0, 3, "Soporte.")],
    )

    class Translator:
        def translate_transcript(self, transcript, target_language):
            raise TranslationError("model unavailable")

    monkeypatch.setattr("yt_sub_hub.pipeline.fetch_video_metadata", lambda url, settings=None: metadata)
    monkeypatch.setattr(
        "yt_sub_hub.pipeline.resolve_transcript_bundle",
        lambda video_id, transcript_provider=None, whisper_provider=None: type(
            "Bundle",
            (),
            {"video_id": video_id, "source": "youtube", "raw_payload": {}, "normalized": source_transcript},
        )(),
    )
    pipeline.translation_provider = Translator()

    try:
        pipeline.process("https://youtu.be/abc123def45")
    except TranslationError as exc:
        assert "model unavailable" in str(exc)
    else:
        raise AssertionError("Expected translation failure")

    failed_record = next(iter(pipeline.store.load_jobs().values()))
    assert failed_record.status == "failed"
    assert failed_record.stage == "failed"
    assert "model unavailable" in failed_record.error
    paths = pipeline.store.artifact_paths("abc123def45")
    assert not paths.analysis_ru.exists()
    assert not paths.graph_extraction.exists()


def test_translation_timeout_fails_pipeline_without_russian_artifacts(tmp_path, monkeypatch):
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    pipeline.settings = pipeline.settings.__class__(
        **{
            **pipeline.settings.__dict__,
            "enable_translation": True,
            "target_language": "ru",
            "translation_best_effort": False,
        }
    )
    metadata = VideoMetadata(video_id="abc123def45", canonical_url="https://youtu.be/abc123def45", title="Talk")
    source_transcript = Transcript(
        video_id="abc123def45",
        source="youtube",
        language="es",
        segments=[TranscriptSegment(0, 3, "Soporte.")],
    )

    class Translator:
        def translate_transcript(self, transcript, target_language):
            raise TimeoutError("LM Studio translation request failed: timed out")

    class Analyzer:
        def analyze(self, metadata, transcript):
            raise AssertionError("analysis should not run after strict translation timeout")

    monkeypatch.setattr("yt_sub_hub.pipeline.fetch_video_metadata", lambda url, settings=None: metadata)
    monkeypatch.setattr(
        "yt_sub_hub.pipeline.resolve_transcript_bundle",
        lambda video_id, transcript_provider=None, whisper_provider=None: type(
            "Bundle",
            (),
            {"video_id": video_id, "source": "youtube", "raw_payload": {}, "normalized": source_transcript},
        )(),
    )
    pipeline.translation_provider = Translator()
    pipeline.analyzer = Analyzer()

    with pytest.raises(TimeoutError, match="LM Studio translation request failed"):
        pipeline.process("https://youtu.be/abc123def45")

    failed_record = next(iter(pipeline.store.load_jobs().values()))
    assert failed_record.status == "failed"
    assert failed_record.stage == "failed"
    assert "LM Studio translation request failed" in failed_record.error
    paths = pipeline.store.artifact_paths("abc123def45")
    assert not paths.analysis_ru.exists()
    assert not paths.graph_extraction.exists()


def test_analyzer_timeout_after_ru_transcript_fails_without_ru_analysis_or_graph(tmp_path, monkeypatch):
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    pipeline.settings = pipeline.settings.__class__(
        **{
            **pipeline.settings.__dict__,
            "enable_translation": True,
            "target_language": "ru",
            "translation_best_effort": False,
        }
    )
    metadata = VideoMetadata(video_id="abc123def45", canonical_url="https://youtu.be/abc123def45", title="Talk")
    source_transcript = Transcript(
        video_id="abc123def45",
        source="youtube",
        language="es",
        segments=[TranscriptSegment(0, 3, "Soporte.")],
    )
    russian_transcript = Transcript(
        video_id="abc123def45",
        source="youtube",
        language="ru",
        segments=[TranscriptSegment(0, 3, "Поддержка.")],
    )

    class Translator:
        def translate_transcript(self, transcript, target_language):
            return russian_transcript

        def translate_analysis(self, analysis, target_language):
            raise AssertionError("analysis translation should not run after analyzer timeout")

    class Analyzer:
        def analyze(self, metadata, transcript):
            assert transcript.language == "ru"
            raise TimeoutError("LM Studio request failed: timed out")

    monkeypatch.setattr("yt_sub_hub.pipeline.fetch_video_metadata", lambda url, settings=None: metadata)
    monkeypatch.setattr(
        "yt_sub_hub.pipeline.resolve_transcript_bundle",
        lambda video_id, transcript_provider=None, whisper_provider=None: type(
            "Bundle",
            (),
            {"video_id": video_id, "source": "youtube", "raw_payload": {}, "normalized": source_transcript},
        )(),
    )
    pipeline.translation_provider = Translator()
    pipeline.analyzer = Analyzer()

    with pytest.raises(TimeoutError, match="LM Studio request failed"):
        pipeline.process("https://youtu.be/abc123def45")

    failed_record = next(iter(pipeline.store.load_jobs().values()))
    assert failed_record.status == "failed"
    assert failed_record.stage == "failed"
    assert "LM Studio request failed" in failed_record.error
    paths = pipeline.store.artifact_paths("abc123def45")
    assert paths.transcript_ru.exists()
    assert not paths.analysis_ru.exists()
    assert not paths.graph_extraction.exists()


def test_translation_failure_warns_when_best_effort(tmp_path, monkeypatch):
    from yt_sub_hub.translation import TranslationError

    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    pipeline.settings = pipeline.settings.__class__(
        **{
            **pipeline.settings.__dict__,
            "enable_translation": True,
            "target_language": "ru",
            "translation_best_effort": True,
        }
    )
    metadata = VideoMetadata(video_id="abc123def45", canonical_url="https://youtu.be/abc123def45", title="Talk")
    source_transcript = Transcript(
        video_id="abc123def45",
        source="youtube",
        language="es",
        segments=[TranscriptSegment(0, 3, "Soporte.")],
    )

    class Translator:
        def translate_transcript(self, transcript, target_language):
            raise TranslationError("model unavailable")

    class Analyzer:
        def analyze(self, metadata, transcript):
            assert transcript.language == "es"
            return Analysis(
                language="es",
                cleaned_title="Talk",
                abstract="Source analysis",
                tags=[],
                outline=[],
                sections=[],
                concepts=[Concept("Soporte", "Zona")],
                claims=[],
                limitations=[],
                open_questions=[],
                related_topics=[],
            )

    monkeypatch.setattr("yt_sub_hub.pipeline.fetch_video_metadata", lambda url, settings=None: metadata)
    monkeypatch.setattr(
        "yt_sub_hub.pipeline.resolve_transcript_bundle",
        lambda video_id, transcript_provider=None, whisper_provider=None: type(
            "Bundle",
            (),
            {"video_id": video_id, "source": "youtube", "raw_payload": {}, "normalized": source_transcript},
        )(),
    )
    pipeline.translation_provider = Translator()
    pipeline.analyzer = Analyzer()

    record = pipeline.process("https://youtu.be/abc123def45")

    assert record.status == "completed_with_graph_warning"
    assert "Translation failed; graph and note generated from source language" in record.warning
    assert not (tmp_path / "videos" / "abc123def45" / "transcript.ru.json").exists()
