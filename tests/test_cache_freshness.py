import os
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from yt_sub_hub import cli
from yt_sub_hub.library import LibraryStore
from yt_sub_hub.models import Analysis, Transcript, TranscriptBundle, TranscriptSegment, VideoMetadata
from yt_sub_hub.pipeline import VideoProcessingPipeline
from yt_sub_hub.runtime import RuntimeStore
from yt_sub_hub.stage_pipeline import StagePipeline
from yt_sub_hub.transcripts import TranscriptResolutionError


VIDEO_ID = "abc123def45"
URL = f"https://youtu.be/{VIDEO_ID}"


def _analysis(text="old analysis"):
    return Analysis(language="fr", cleaned_title=text, abstract=text, tags=[], outline=[], sections=[], concepts=[], claims=[], limitations=[], open_questions=[], related_topics=[])


def _transcript(text="old source", language="en"):
    return Transcript(VIDEO_ID, "youtube", [TranscriptSegment(0, 1, text)], language=language)


def _seed(store):
    metadata = VideoMetadata(VIDEO_ID, URL, "old title")
    transcript = _transcript()
    paths = store.save_video_artifacts(metadata, transcript, transcript, _analysis(), "My handwritten note")
    store.save_translated_transcript(VIDEO_ID, _transcript("ancienne traduction", "fr"))
    store.save_graph_extraction(VIDEO_ID, {"old": True})
    return paths


def _expire(*paths):
    old = time.time() - 31 * 86400
    for path in paths:
        os.utime(path, (old, old))


def _live(monkeypatch):
    metadata = VideoMetadata(VIDEO_ID, URL, "new title")
    source = _transcript("new source")
    monkeypatch.setattr("yt_sub_hub.pipeline.fetch_video_metadata", lambda *args, **kwargs: metadata)
    monkeypatch.setattr("yt_sub_hub.pipeline.build_transcript_provider", lambda *args, **kwargs: None)
    monkeypatch.setattr("yt_sub_hub.pipeline.resolve_transcript_bundle", lambda *args, **kwargs: TranscriptBundle(VIDEO_ID, "youtube", source.to_dict(), source))


def test_expired_inputs_are_unavailable_and_completed_video_is_not_reused(tmp_path):
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    paths = _seed(pipeline.store)
    _expire(paths.metadata, paths.raw_transcript, paths.normalized_transcript)

    assert pipeline.store.load_video_metadata(VIDEO_ID) is None
    assert pipeline.store.load_raw_transcript(VIDEO_ID) is None
    assert pipeline.store.load_normalized_transcript(VIDEO_ID) is None
    assert pipeline.already_processed(URL) is None
    assert paths.note.read_text() == "My handwritten note"


def test_refresh_reloads_sources_and_rebuilds_translation_analysis_and_graph(tmp_path, monkeypatch):
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    pipeline.settings = replace(pipeline.settings, enable_translation=True, target_language="fr")
    paths = _seed(pipeline.store)
    _live(monkeypatch)
    pipeline.translation_provider = SimpleNamespace(translate_transcript=lambda source, target: _transcript("nouvelle traduction de " + source.full_text(), target))
    pipeline.analyzer = SimpleNamespace(analyze=lambda metadata, transcript: _analysis(metadata.title + ": " + transcript.full_text()))

    result = pipeline.process(URL, refresh=True)

    assert result.status.startswith("completed")
    assert pipeline.store.load_video_metadata(VIDEO_ID).title == "new title"
    assert pipeline.store.load_normalized_transcript(VIDEO_ID).full_text() == "new source"
    assert pipeline.store.load_translated_transcript(VIDEO_ID, "fr").full_text() == "nouvelle traduction de new source"
    assert pipeline.store.load_analysis(VIDEO_ID)["cleaned_title"] == "new title: nouvelle traduction de new source"
    assert pipeline.store.load_graph_extraction(VIDEO_ID).get("old") is None
    assert any(path.read_text() == "My handwritten note" for path in paths.video_dir.glob("note.before-refresh.*.md"))


@pytest.mark.parametrize("failure_stage", ["metadata", "transcript"])
def test_failed_explicit_refresh_cannot_fall_back_to_old_inputs_after_restart(tmp_path, monkeypatch, failure_stage):
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    paths = _seed(pipeline.store)
    _live(monkeypatch)

    def fail(*args, **kwargs):
        raise TranscriptResolutionError("live source unavailable")

    target = "fetch_video_metadata" if failure_stage == "metadata" else "resolve_transcript_bundle"
    monkeypatch.setattr(f"yt_sub_hub.pipeline.{target}", fail)
    with pytest.raises(TranscriptResolutionError, match="live source unavailable"):
        pipeline.process(URL, refresh=True)

    restarted = VideoProcessingPipeline.for_test(tmp_path)
    assert restarted.already_processed(URL) is None
    assert restarted.store.load_normalized_transcript(VIDEO_ID) is None
    assert restarted.store.load_analysis(VIDEO_ID) is None
    assert paths.note.read_text() == "My handwritten note"


def test_reanalysis_does_not_extend_cached_transcript_acquisition_age(tmp_path, monkeypatch):
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    paths = _seed(pipeline.store)
    earlier = time.time() - 10 * 86400
    os.utime(paths.normalized_transcript, (earlier, earlier))
    os.utime(paths.raw_transcript, (earlier, earlier))
    pipeline.analyzer = SimpleNamespace(analyze=lambda *args: _analysis("new analysis"))
    _live(monkeypatch)

    pipeline.process(URL)

    assert paths.normalized_transcript.stat().st_mtime == earlier
    assert paths.raw_transcript.stat().st_mtime == earlier


def test_staged_expired_transcript_reloads_and_invalidates_derived_cache_across_restart(tmp_path, monkeypatch):
    direct = VideoProcessingPipeline.for_test(tmp_path)
    paths = _seed(direct.store)
    _expire(paths.raw_transcript, paths.normalized_transcript)
    source = _transcript("new source")
    monkeypatch.setattr("yt_sub_hub.stage_pipeline.build_transcript_provider", lambda *args, **kwargs: SimpleNamespace(fetch_bundle=lambda video_id: (source.to_dict(), source)))
    runtime = RuntimeStore(tmp_path / "index.sqlite3", settings=direct.settings)
    job = runtime.create_video_job(job_id="job", video_id=VIDEO_ID, canonical_url=URL)
    runtime.ensure_stage_plan(job.id)
    stage = next(item for item in runtime.list_stage_plan(job.id) if item.stage_name == "transcript_source")
    staged = StagePipeline(settings=direct.settings, pipeline=direct)

    result = staged.run(staged.context(job, stage, runtime=runtime))

    assert result.status == "completed"
    restarted = LibraryStore(tmp_path)
    assert restarted.load_normalized_transcript(VIDEO_ID).full_text() == "new source"
    assert restarted.load_translated_transcript(VIDEO_ID, "fr") is None
    assert restarted.load_analysis(VIDEO_ID) is None
    assert restarted.load_graph_extraction(VIDEO_ID) is None


def test_cleanup_defaults_to_preview_and_apply_only_removes_expired_raw_inputs(tmp_path, monkeypatch):
    store = LibraryStore(tmp_path)
    paths = _seed(store)
    _expire(paths.metadata, paths.raw_transcript, paths.normalized_transcript)
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))

    preview = CliRunner().invoke(cli.app, ["cache-cleanup"])
    assert preview.exit_code == 0, preview.output
    assert "metadata.json" in preview.output
    assert paths.metadata.exists()
    applied = CliRunner().invoke(cli.app, ["cache-cleanup", "--apply"])
    assert applied.exit_code == 0, applied.output
    assert not paths.metadata.exists()
    assert not paths.raw_transcript.exists()
    assert not paths.normalized_transcript.exists()
    assert paths.analysis.exists()
    assert paths.note.read_text() == "My handwritten note"
    assert store.load_analysis(VIDEO_ID) is None


def test_cleanup_ignores_symlinked_video_directories_and_source_files(tmp_path, monkeypatch):
    root = tmp_path / "library"
    paths = _seed(LibraryStore(root))
    outside = tmp_path / "outside.json"
    outside.write_text('{"private": true}')
    _expire(outside)
    paths.metadata.unlink()
    paths.metadata.symlink_to(outside)
    (root / "videos" / "other1234567").symlink_to(paths.video_dir, target_is_directory=True)
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(root))

    result = CliRunner().invoke(cli.app, ["cache-cleanup", "--apply"])

    assert result.exit_code == 0, result.output
    assert outside.read_text() == '{"private": true}'
    assert paths.metadata.is_symlink()


def test_cli_refresh_bypasses_completed_prompt_and_fetches_new_input(tmp_path, monkeypatch):
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    _seed(pipeline.store)
    _live(monkeypatch)
    pipeline.analyzer = SimpleNamespace(analyze=lambda metadata, transcript: _analysis(transcript.full_text()))
    monkeypatch.setattr(cli, "_pipeline", lambda: pipeline)

    result = CliRunner().invoke(cli.app, ["process", URL, "--refresh"])

    assert result.exit_code == 0, result.output
    assert "уже анализировалось" not in result.output
    assert pipeline.store.load_analysis(VIDEO_ID)["cleaned_title"] == "new source"


def test_refresh_refuses_to_replace_sources_while_another_worker_owns_video(tmp_path, monkeypatch):
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    paths = _seed(pipeline.store)
    _live(monkeypatch)

    with LibraryStore(tmp_path).video_lock(VIDEO_ID):
        with pytest.raises(RuntimeError, match="busy"):
            pipeline.process(URL, refresh=True)

    assert pipeline.store.load_normalized_transcript(VIDEO_ID).full_text() == "old source"
    assert paths.note.read_text() == "My handwritten note"


def test_cleanup_refuses_to_delete_inputs_held_by_a_running_worker(tmp_path):
    store = LibraryStore(tmp_path)
    paths = _seed(store)
    _expire(paths.metadata, paths.raw_transcript, paths.normalized_transcript)

    with LibraryStore(tmp_path).video_lock(VIDEO_ID):
        with pytest.raises(RuntimeError, match="busy"):
            store.cleanup_expired_raw_cache(apply=True)

    assert paths.metadata.exists()
    assert paths.raw_transcript.exists()


def test_refresh_requests_atomic_video_core_replacement(tmp_path, monkeypatch):
    class Graph:
        def __init__(self):
            self.events = []

        def replace_video_core_payload(self, video_id, payload):
            self.events.append(("replace_core", video_id, payload))

    graph = Graph()
    pipeline = VideoProcessingPipeline.for_test(tmp_path, graph_store=graph)
    _seed(pipeline.store)
    _live(monkeypatch)
    pipeline.analyzer = SimpleNamespace(analyze=lambda *args: _analysis("new analysis"))

    pipeline.process(URL, refresh=True)

    assert [event[0] for event in graph.events] == ["replace_core"]
    assert graph.events[0][1] == VIDEO_ID


def test_resumed_translation_merge_rejects_chunks_from_before_refresh_even_if_source_text_matches(tmp_path):
    direct = VideoProcessingPipeline.for_test(tmp_path)
    direct.settings = replace(direct.settings, enable_translation=True, target_language="fr")
    direct.store.save_source_metadata(VideoMetadata(VIDEO_ID, URL, "title"))
    direct.store.save_source_transcript(VIDEO_ID, _transcript().to_dict(), _transcript())
    direct.translation_provider = SimpleNamespace(translate_transcript=lambda source, target: _transcript("traduction", target))
    staged = StagePipeline(settings=direct.settings, pipeline=direct)
    runtime = RuntimeStore(tmp_path / "index.sqlite3", settings=direct.settings)
    job = runtime.create_video_job(job_id="job", video_id=VIDEO_ID, canonical_url=URL)
    runtime.ensure_stage_plan(job.id)

    def stage(name):
        return next(item for item in runtime.list_stage_plan(job.id) if item.stage_name == name)

    staged.run(staged.context(job, stage("transcript_translation_expand"), runtime=runtime))
    staged.run(staged.context(job, stage("transcript_translation_chunk_000"), runtime=runtime))
    direct.store.invalidate_source_cache(VIDEO_ID)

    with pytest.raises(RuntimeError, match="stale translation chunk"):
        staged.run(staged.context(job, stage("transcript_translation_merge"), runtime=runtime))


def test_graph_rebuild_does_not_consume_expired_raw_inputs(tmp_path):
    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    paths = _seed(pipeline.store)
    _expire(paths.metadata, paths.raw_transcript, paths.normalized_transcript)

    assert pipeline.rebuild_graph(dry_run=True) == 0
    assert pipeline.store.load_analysis_files() == []
    assert pipeline.store.load_graph_extraction_files() == []


def test_staged_expired_metadata_is_fetched_live_and_blocks_older_analysis(tmp_path, monkeypatch):
    direct = VideoProcessingPipeline.for_test(tmp_path)
    paths = _seed(direct.store)
    _expire(paths.metadata)
    _live(monkeypatch)
    staged = StagePipeline(settings=direct.settings, pipeline=direct)
    runtime = RuntimeStore(tmp_path / "index.sqlite3", settings=direct.settings)
    job = runtime.create_video_job(job_id="job", video_id=VIDEO_ID, canonical_url=URL)
    runtime.ensure_stage_plan(job.id)
    stage = next(item for item in runtime.list_stage_plan(job.id) if item.stage_name == "metadata")

    result = staged.run(staged.context(job, stage, runtime=runtime))

    assert result.status == "completed"
    assert direct.store.load_video_metadata(VIDEO_ID).title == "new title"
    assert direct.store.load_analysis(VIDEO_ID) is None


def test_visual_frame_is_regenerated_with_current_transcript_after_source_refresh(tmp_path):
    from yt_sub_hub.visual import VisualAnalysisService

    pipeline = VideoProcessingPipeline.for_test(tmp_path)
    _seed(pipeline.store)
    pipeline.store.save_visual_frame(
        VIDEO_ID, "000000", raw={},
        normalized={"video_id": VIDEO_ID, "timestamp_seconds": 0, "summary": "old visual summary", "transcript_context": {"text": "old source"}},
    )
    pipeline.store.invalidate_source_cache(VIDEO_ID, refresh_inputs=True)
    pipeline.store.save_source_metadata(VideoMetadata(VIDEO_ID, URL, "new title"))
    source = _transcript("new source")
    pipeline.store.save_source_transcript(VIDEO_ID, source.to_dict(), source)

    class Extractor:
        def extract_frame(self, metadata, output_path, timestamp_seconds, force=False):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"jpg")
            return output_path

    class Provider:
        def analyze_frame(self, frame_path, metadata, context, timestamp_seconds, question=None):
            return {"summary": "Visual explanation of " + context["text"]}

    service = VisualAnalysisService(
        settings=replace(pipeline.settings, vision_model="vision-test"),
        store=pipeline.store, provider=Provider(), extractor=Extractor(),
    )

    result = service.analyze_frame(VIDEO_ID, 0)

    assert result["summary"] == "Visual explanation of new source"
    assert result["transcript_context"]["text"] == "new source"
    assert pipeline.store.load_visual_frame(VIDEO_ID, "000000")["summary"] == result["summary"]
