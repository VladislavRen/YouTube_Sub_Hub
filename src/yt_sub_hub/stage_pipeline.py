from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import re
from typing import Any

from yt_sub_hub.artifact_review import load_ru_artifact_review, parse_frontmatter, review_ru_artifacts
from yt_sub_hub.config import Settings, get_settings, normalize_acceptance_mode, normalize_quality_mode
from yt_sub_hub.graph import Neo4jGraphStore
from yt_sub_hub.graph_extraction import build_deterministic_graph_extraction, graph_payload_from_extraction
from yt_sub_hub.language_quality import contains_cyrillic, validate_ru_analysis, validate_ru_markdown_note, validate_ru_text_field
from yt_sub_hub.library import LibraryStore
from yt_sub_hub.models import Analysis, Claim, Concept, JobRecord, NoteSection, Transcript, TranscriptSegment, slugify
from yt_sub_hub.notes import render_markdown_note
from yt_sub_hub.pipeline import VideoProcessingPipeline, _analysis_matches_target_language
from yt_sub_hub.product_quality import evaluate_product_quality
from yt_sub_hub.quality import run_quality_gate
from yt_sub_hub.runtime import StagePlanRecord, VideoJobRecord
from yt_sub_hub.transcripts import (
    EXPECTED_MEDIA_FALLBACK_REASONS,
    SKIPPED_REQUIRES_MEDIA_REASON,
    TranscriptResolutionError,
    TranscriptSourceError,
    WhisperAudioProvider,
    build_transcript_provider,
    explain_full_processing_media_download_disabled,
)
from yt_sub_hub.whisper_backends import build_whisper_backend


@dataclass(frozen=True)
class StageContext:
    job: VideoJobRecord
    stage: StagePlanRecord
    settings: Settings
    store: LibraryStore
    pipeline: VideoProcessingPipeline
    runtime: Any | None = None


@dataclass(frozen=True)
class StageResult:
    status: str = "completed"
    warning: str | None = None
    error: str | None = None
    artifacts: list[Path] = field(default_factory=list)
    skip_stages: list[str] = field(default_factory=list)

    @classmethod
    def completed(cls, *, warning: str | None = None, skip_stages: list[str] | None = None) -> "StageResult":
        return cls(status="completed", warning=warning, skip_stages=skip_stages or [])

    @classmethod
    def skipped(cls, warning: str | None = None) -> "StageResult":
        return cls(status="skipped", warning=warning)

    @classmethod
    def failed(cls, error: str) -> "StageResult":
        return cls(status="failed", error=error)


StageHandler = Callable[[StageContext], StageResult]


class StagePipeline:
    def __init__(
        self,
        *,
        library_dir: Path | str | None = None,
        settings: Settings | None = None,
        pipeline: VideoProcessingPipeline | None = None,
        handlers: dict[str, StageHandler] | None = None,
        discovery_provider: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        if library_dir is not None:
            self.settings = replace(self.settings, library_dir=Path(library_dir))
        self.store = LibraryStore(self.settings.library_dir)
        self.pipeline = pipeline or VideoProcessingPipeline(settings=self.settings, store=self.store)
        self.discovery_provider = discovery_provider
        self.handlers: dict[str, StageHandler] = {
            "metadata": self._metadata,
            "transcript_source": self._transcript_source,
            "audio_download": self._audio_download,
            "whisper_transcription": self._whisper_transcription,
            "transcript_translation_expand": self._transcript_translation_expand,
            "transcript_translation_merge": self._transcript_translation_merge,
            "scientific_analysis": self._scientific_analysis,
            "analysis_reduce": self._analysis_reduce,
            "analysis_translation": self._analysis_translation,
            "analysis_ru_review": self._analysis_ru_review,
            "markdown_render": self._markdown_render,
            "deterministic_graph": self._deterministic_graph,
            "deep_graph_llm": self._deep_graph_llm,
            "visual_frame_analysis": self._visual_frame_analysis,
            "ru_artifact_review": self._ru_artifact_review,
            "neo4j_sync": self._neo4j_sync,
            "quality_gate": self._quality_gate,
            "youtube_discovery_seed_extract": self._youtube_discovery_seed_extract,
            "youtube_discovery_query_expand": self._youtube_discovery_query_expand,
            "youtube_discovery_candidate_fetch": self._youtube_discovery_candidate_fetch,
            "youtube_discovery_candidate_score": self._youtube_discovery_candidate_score,
            "youtube_discovery_review": self._youtube_discovery_review,
            "youtube_discovery_graph": self._youtube_discovery_graph,
            "youtube_discovery_report": self._youtube_discovery_report,
        }
        if handlers:
            self.handlers.update(handlers)

    def context(self, job: VideoJobRecord, stage: StagePlanRecord, runtime: Any | None = None) -> StageContext:
        return StageContext(job=job, stage=stage, settings=self.settings, store=self.store, pipeline=self.pipeline, runtime=runtime)

    def run(self, ctx: StageContext) -> StageResult:
        if _quality_mode(ctx) == "draft" and ctx.stage.stage_name in {
            "deterministic_graph",
            "deep_graph_llm",
            "ru_artifact_review",
            "neo4j_sync",
        }:
            return StageResult.skipped("draft mode skips graph sync and production validation")
        handler = self.handlers.get(ctx.stage.stage_name)
        if handler is None and ctx.stage.stage_name.startswith("transcript_translation_chunk_"):
            handler = self._transcript_translation_chunk
        if handler is None and ctx.stage.stage_name.startswith("analysis_chunk_"):
            handler = self._analysis_chunk
        if handler is None and ctx.stage.stage_name.startswith("youtube_discovery_query_"):
            handler = self._youtube_discovery_query
        if handler is None:
            raise KeyError(f"Unknown pipeline stage: {ctx.stage.stage_name}")
        return handler(ctx)

    def _legacy_record(self, job: VideoJobRecord) -> JobRecord:
        return JobRecord(
            job_id=job.id,
            url=job.canonical_url,
            video_id=job.video_id,
            status="queued",
            stage="queued",
        )

    def _metadata(self, ctx: StageContext) -> StageResult:
        cached = ctx.store.load_video_metadata(ctx.job.video_id)
        if cached is not None:
            return StageResult.completed(warning="metadata cache hit")
        metadata = ctx.pipeline._fetch_or_load_metadata(self._legacy_record(ctx.job), [])
        paths = ctx.store.artifact_paths(ctx.job.video_id)
        paths.video_dir.mkdir(parents=True, exist_ok=True)
        ctx.store._write_json(paths.metadata, metadata.to_dict())
        return StageResult.completed()

    def _transcript_source(self, ctx: StageContext) -> StageResult:
        if ctx.store.load_normalized_transcript(ctx.job.video_id) is not None:
            return StageResult.completed(warning="transcript cache hit", skip_stages=["audio_download", "whisper_transcription"])
        acceptance_mode = _acceptance_mode(ctx)
        try:
            raw, normalized = build_transcript_provider(
                ctx.settings,
                allow_unofficial_provider=acceptance_mode != "data_api_only",
            ).fetch_bundle(ctx.job.video_id)
        except TranscriptSourceError as exc:
            if acceptance_mode == "data_api_only":
                if exc.reason in {"captions_unavailable", "captions_forbidden"}:
                    return StageResult.skipped(SKIPPED_REQUIRES_MEDIA_REASON)
                return StageResult.failed(str(exc))
            if exc.reason not in EXPECTED_MEDIA_FALLBACK_REASONS:
                return StageResult.failed(str(exc))
            if not ctx.settings.media_download_enabled:
                return StageResult.failed(explain_full_processing_media_download_disabled())
            return StageResult.completed(warning="needs whisper fallback")
        except TranscriptResolutionError as exc:
            if acceptance_mode == "data_api_only":
                return StageResult.failed(str(exc))
            if not ctx.settings.media_download_enabled:
                return StageResult.failed(explain_full_processing_media_download_disabled())
            return StageResult.completed(warning="needs whisper fallback")
        except Exception as exc:
            return StageResult.failed(str(exc))
        paths = ctx.store.artifact_paths(ctx.job.video_id)
        paths.video_dir.mkdir(parents=True, exist_ok=True)
        ctx.store._write_json(paths.raw_transcript, raw)
        ctx.store._write_json(paths.normalized_transcript, normalized.to_dict())
        return StageResult.completed(skip_stages=["audio_download", "whisper_transcription"])

    def _audio_download(self, ctx: StageContext) -> StageResult:
        if ctx.store.load_normalized_transcript(ctx.job.video_id) is not None:
            return StageResult.skipped("transcript already available")
        provider = WhisperAudioProvider(
            ctx.job.canonical_url,
            ctx.settings.library_dir / "assets" / "audio",
            keep_audio=True,
            settings=ctx.settings,
        )
        provider._download_audio(ctx.job.video_id)
        return StageResult.completed()

    def _whisper_transcription(self, ctx: StageContext) -> StageResult:
        if ctx.store.load_normalized_transcript(ctx.job.video_id) is not None:
            return StageResult.skipped("transcript already available")
        audio_path = ctx.settings.library_dir / "assets" / "audio" / f"{ctx.job.video_id}.mp3"
        if not audio_path.exists() or audio_path.stat().st_size <= 0:
            return StageResult.failed(f"audio artifact is required before whisper_transcription: {audio_path}")
        try:
            bundle = build_whisper_backend(ctx.settings).transcribe(audio_path, video_id=ctx.job.video_id)
            paths = ctx.store.artifact_paths(ctx.job.video_id)
            paths.video_dir.mkdir(parents=True, exist_ok=True)
            ctx.store._write_json(paths.raw_transcript, bundle.raw_payload)
            ctx.store._write_json(paths.normalized_transcript, bundle.normalized.to_dict())
            return StageResult.completed()
        finally:
            if not ctx.settings.keep_audio:
                audio_path.unlink(missing_ok=True)

    def _transcript_translation_expand(self, ctx: StageContext) -> StageResult:
        if not ctx.settings.enable_translation:
            return StageResult.skipped("translation disabled")
        transcript = ctx.store.load_normalized_transcript(ctx.job.video_id)
        if transcript is None:
            raise RuntimeError("normalized transcript is required")
        target = _target_language(ctx)
        cached = ctx.store.load_translated_transcript(ctx.job.video_id, target)
        if _transcript_matches_target_language(cached, target):
            return StageResult.completed(warning="translation cache hit")
        chunks = _plan_translation_chunks(transcript, ctx.settings, target)
        if ctx.runtime is None:
            raise RuntimeError("runtime is required for transcript translation chunk expansion")
        ctx.runtime.expand_translation_chunk_plan(ctx.job.id, chunks)
        return StageResult.completed()

    def _transcript_translation_chunk(self, ctx: StageContext) -> StageResult:
        if not ctx.settings.enable_translation:
            return StageResult.skipped("translation disabled")
        transcript = ctx.store.load_normalized_transcript(ctx.job.video_id)
        if transcript is None:
            raise RuntimeError("normalized transcript is required")
        payload = _stage_payload(ctx.stage)
        target = str(payload.get("target_language") or _target_language(ctx))
        chunk_index = int(payload["chunk_index"])
        chunk_path = _translation_chunk_path(ctx.store, ctx.job.video_id, chunk_index)
        if _valid_translation_chunk_artifact(chunk_path, transcript, payload, target):
            return StageResult.completed(warning="translation chunk cache hit")
        start_index = int(payload["source_segment_start_index"])
        end_index = int(payload["source_segment_end_index"])
        source_chunk = slice_translation_chunk(transcript, payload)
        provider = ctx.pipeline.translation_provider or _default_chunk_translation_provider(ctx)
        translated = provider.translate_transcript(source_chunk, target)
        if (translated.language or "").strip().lower() != target.strip().lower():
            raise RuntimeError(f"translation chunk {chunk_index:03d} did not produce {target}")
        artifact = _build_translation_chunk_artifact(
            ctx,
            payload,
            transcript,
            translated,
            target,
            provider,
        )
        _validate_translation_chunk_artifact(artifact, transcript, payload, target)
        _commit_translation_chunk(ctx, chunk_path, artifact)
        return StageResult.completed()

    def _transcript_translation_merge(self, ctx: StageContext) -> StageResult:
        if not ctx.settings.enable_translation:
            return StageResult.skipped("translation disabled")
        transcript = ctx.store.load_normalized_transcript(ctx.job.video_id)
        if transcript is None:
            raise RuntimeError("normalized transcript is required")
        target = _target_language(ctx)
        cached = ctx.store.load_translated_transcript(ctx.job.video_id, target)
        if _transcript_matches_target_language(cached, target):
            return StageResult.completed(warning="translation cache hit")
        chunk_payloads = _translation_chunk_payloads(ctx, target)
        translated_segments = _merge_translation_chunks(ctx, transcript, chunk_payloads, target)
        merged = Transcript(
            video_id=transcript.video_id,
            source=transcript.source,
            language=target,
            segments=translated_segments,
        )
        if not _transcript_matches_target_language(merged, target):
            raise RuntimeError(f"transcript.{target}.json did not pass {target} language gate")
        ctx.store.save_translated_transcript(ctx.job.video_id, merged)
        return StageResult.completed()

    def _scientific_analysis(self, ctx: StageContext) -> StageResult:
        metadata = ctx.store.load_video_metadata(ctx.job.video_id)
        transcript, source_transcript, language = _analysis_input_transcript(ctx)
        if metadata is None or transcript is None:
            raise RuntimeError("metadata and transcript are required")
        use_map_reduce = bool(getattr(ctx.settings, "analysis_map_reduce_enabled", True))
        direct_max_segments = max(1, int(getattr(ctx.settings, "analysis_direct_max_segments", 180)))
        is_long = len(transcript.segments) > direct_max_segments
        cached = ctx.store.load_analysis(ctx.job.video_id, variant="source")
        if cached is not None and not is_long:
            return StageResult.completed(warning="analysis cache hit", skip_stages=["analysis_reduce"])
        if not ctx.settings.enable_translation and ctx.store.load_analysis(ctx.job.video_id) is not None and not is_long:
            return StageResult.completed(warning="analysis cache hit", skip_stages=["analysis_reduce"])
        if use_map_reduce and is_long:
            if ctx.runtime is None:
                raise RuntimeError("runtime is required for analysis chunk expansion")
            chunks = _plan_analysis_chunks(transcript, ctx.settings, source_transcript, language)
            ctx.runtime.expand_analysis_chunk_plan(ctx.job.id, chunks)
            return StageResult.completed()
        analysis = ctx.pipeline._analyze_or_load_cached(metadata, transcript, [])
        ctx.store.save_analysis_artifact(ctx.job.video_id, analysis, variant="source", compatibility=not ctx.settings.enable_translation)
        return StageResult.completed(skip_stages=["analysis_reduce"])

    def _analysis_chunk(self, ctx: StageContext) -> StageResult:
        metadata = ctx.store.load_video_metadata(ctx.job.video_id)
        if metadata is None:
            raise RuntimeError("metadata is required")
        payload = _stage_payload(ctx.stage)
        transcript = _load_analysis_payload_transcript(ctx, payload)
        current_hash = _transcript_hash(transcript)
        if payload.get("source_transcript_hash") != current_hash:
            raise RuntimeError("analysis chunk payload transcript hash does not match current transcript")
        chunk_index = int(payload["chunk_index"])
        chunk_path = _analysis_chunk_path(ctx.store, ctx.job.video_id, chunk_index)
        if _valid_analysis_chunk_artifact(chunk_path, transcript, payload):
            return StageResult.completed(warning="analysis chunk cache hit")
        start_index = int(payload["source_segment_start_index"])
        end_index = int(payload["source_segment_end_index"])
        source_segments = transcript.segments[start_index : end_index + 1]
        source_chunk = Transcript(
            video_id=transcript.video_id,
            source=transcript.source,
            language=transcript.language,
            segments=[
                TranscriptSegment(segment.start, segment.duration, segment.text, segment.raw_text)
                for segment in source_segments
            ],
        )
        analyzer = ctx.pipeline.analyzer
        if not hasattr(analyzer, "analyze_chunk"):
            raise RuntimeError("analyzer must support analyze_chunk for staged analysis")
        chunk_kwargs = {
            "chunk_index": chunk_index,
            "total_chunks": int(payload.get("total_chunks") or 1),
        }
        if "timeout_seconds" in inspect.signature(analyzer.analyze_chunk).parameters:
            chunk_kwargs["timeout_seconds"] = ctx.settings.analysis_chunk_timeout_seconds
        raw_chunk = analyzer.analyze_chunk(
            metadata,
            source_chunk,
            **chunk_kwargs,
        )
        artifact = _build_analysis_chunk_artifact(ctx, payload, transcript, raw_chunk, analyzer)
        _validate_analysis_chunk_artifact(artifact, transcript, payload)
        _commit_analysis_chunk(chunk_path, artifact)
        return StageResult.completed()

    def _analysis_reduce(self, ctx: StageContext) -> StageResult:
        metadata = ctx.store.load_video_metadata(ctx.job.video_id)
        if metadata is None:
            raise RuntimeError("metadata is required")
        payloads = _analysis_chunk_payloads(ctx)
        if not payloads:
            if ctx.store.load_analysis(ctx.job.video_id, variant="source") is not None:
                return StageResult.skipped("analysis reduce skipped for direct analysis")
            raise RuntimeError("analysis chunks are required before analysis_reduce")
        first_payload = payloads[0]
        transcript = _load_analysis_payload_transcript(ctx, first_payload)
        chunk_artifacts = _load_analysis_chunk_artifacts(ctx, transcript, payloads)
        analysis = _reduce_analysis_chunks(metadata, transcript, chunk_artifacts)
        ctx.store.save_analysis_artifact(ctx.job.video_id, analysis, variant="source", compatibility=not ctx.settings.enable_translation)
        return StageResult.completed()

    def _analysis_translation(self, ctx: StageContext) -> StageResult:
        if not ctx.settings.enable_translation:
            source = ctx.store.load_analysis(ctx.job.video_id, variant="source")
            if source is not None:
                from yt_sub_hub.llm import parse_analysis_response

            ctx.store.save_analysis_artifact(ctx.job.video_id, parse_analysis_response(source), compatibility=True)
            return StageResult.skipped("translation disabled")
        source = ctx.store.load_analysis(ctx.job.video_id, variant="source") or ctx.store.load_analysis(ctx.job.video_id)
        if source is None:
            raise RuntimeError("analysis source is required")
        from yt_sub_hub.llm import parse_analysis_response

        target = _target_language(ctx)
        target_transcript = ctx.store.load_translated_transcript(ctx.job.video_id, target)
        transcript_translated = _transcript_matches_target_language(target_transcript, target)
        if not transcript_translated and not ctx.settings.translation_best_effort:
            raise RuntimeError(f"transcript.{target}.json must exist and pass {target} language gate before analysis_translation")
        cached = ctx.store.load_analysis(ctx.job.video_id, variant=target)
        if _analysis_cache_matches_source(cached, source, target_transcript if transcript_translated else None, target):
            return StageResult.completed(warning="analysis translation cache hit")
        analysis = parse_analysis_response(source)
        fallback_warning = None
        try:
            translated, did_translate = ctx.pipeline._translate_analysis_or_source(analysis, transcript_translated, [])
        except Exception as exc:
            if not ctx.settings.translation_best_effort:
                _write_analysis_translation_diagnostics(ctx, source, target_transcript if transcript_translated else None, target, exc)
                return StageResult.failed(ctx.pipeline._analysis_translation_failure_message())
            translated = _deterministic_target_analysis_fallback(analysis, target, target_transcript if transcript_translated else None)
            if not _analysis_matches_target_language(translated, target):
                return StageResult.failed(f"analysis.{target}.json did not pass {target} language gate")
            did_translate = True
            fallback_warning = f"analysis.{target}.json used deterministic sparse-analysis fallback"
        if not did_translate and ctx.settings.translation_best_effort and transcript_translated:
            translated = _deterministic_target_analysis_fallback(analysis, target, target_transcript)
            if not _analysis_matches_target_language(translated, target):
                return StageResult.failed(f"analysis.{target}.json did not pass {target} language gate")
            did_translate = True
            fallback_warning = f"analysis.{target}.json used deterministic sparse-analysis fallback"
        if did_translate and not _analysis_matches_target_language(translated, target):
            if not ctx.settings.translation_best_effort:
                exc = RuntimeError(f"analysis.{target}.json did not pass {target} language gate")
                _write_analysis_translation_diagnostics(ctx, source, target_transcript if transcript_translated else None, target, exc)
                return StageResult.failed(str(exc))
            translated = _deterministic_target_analysis_fallback(analysis, target, target_transcript if transcript_translated else None)
            if _analysis_matches_target_language(translated, target):
                fallback_warning = f"analysis.{target}.json used deterministic sparse-analysis fallback"
        if did_translate and _analysis_matches_target_language(translated, target):
            ctx.store.save_analysis_artifact(
                ctx.job.video_id,
                translated,
                variant=target,
                compatibility=True,
                extra_fields=_analysis_translation_provenance(
                    source,
                    target_transcript if transcript_translated else None,
                    target,
                    ctx.settings,
                    translated,
                    strategy="deterministic_sparse_fallback" if fallback_warning else "llm_analysis_translation",
                ),
            )
        elif did_translate:
            raise RuntimeError(f"analysis.{target}.json did not pass {target} language gate")
        else:
            ctx.store.save_analysis_artifact(ctx.job.video_id, analysis, variant="source", compatibility=True)
        return StageResult.completed(warning=fallback_warning if did_translate else "analysis translation skipped or source fallback")

    def _analysis_ru_review(self, ctx: StageContext) -> StageResult:
        if not ctx.settings.enable_translation:
            return StageResult.skipped("translation disabled")
        target = _target_language(ctx).strip().lower()
        if target != "ru":
            return StageResult.skipped("analysis_ru_review only supports ru")
        analysis_data = ctx.store.load_analysis(ctx.job.video_id, variant="ru")
        if analysis_data is None:
            return StageResult.failed("analysis.ru.json must exist")
        from yt_sub_hub.llm import parse_analysis_response

        try:
            analysis = parse_analysis_response(analysis_data)
        except Exception as exc:
            return StageResult.failed(f"analysis.ru.json must be valid analysis JSON: {exc}")
        issues = validate_ru_analysis(analysis)
        if issues:
            return StageResult.failed("; ".join(f"analysis.ru.json {issue}" for issue in issues))
        return StageResult.completed()

    def _markdown_render(self, ctx: StageContext) -> StageResult:
        metadata = ctx.store.load_video_metadata(ctx.job.video_id)
        target = _target_language(ctx)
        _require_analysis_ru_review_passed(ctx, "markdown_render")
        if ctx.settings.enable_translation:
            target_transcript = ctx.store.load_translated_transcript(ctx.job.video_id, target)
            if not _transcript_matches_target_language(target_transcript, target) and not ctx.settings.translation_best_effort:
                raise RuntimeError(f"transcript.{target}.json must exist and pass {target} language gate before markdown_render")
            analysis_data = ctx.store.load_analysis(ctx.job.video_id, variant=target)
            if not _analysis_data_matches_target_language(analysis_data, target):
                if not ctx.settings.translation_best_effort:
                    raise RuntimeError(f"analysis.{target}.json must exist and pass {target} language gate before markdown_render")
                analysis_data = ctx.store.load_analysis(ctx.job.video_id, variant="source") or ctx.store.load_analysis(ctx.job.video_id)
        else:
            analysis_data = ctx.store.load_analysis(ctx.job.video_id)
        if metadata is None or analysis_data is None:
            raise RuntimeError("metadata and analysis are required")
        note_path = ctx.store.note_path(ctx.job.video_id)
        if note_path.exists() and note_path.read_text(encoding="utf-8").strip():
            note_text = note_path.read_text(encoding="utf-8")
            if (
                not ctx.settings.enable_translation
                or target.strip().lower() != "ru"
                or (not validate_ru_markdown_note(note_text) and _note_matches_target(note_text, analysis_data, target_transcript if ctx.settings.enable_translation else None))
            ):
                return StageResult.completed(warning="note cache hit")
        from yt_sub_hub.llm import parse_analysis_response

        markdown = render_markdown_note(
            metadata,
            parse_analysis_response(analysis_data),
            provenance=_note_provenance(
                analysis_data,
                target_transcript if ctx.settings.enable_translation else None,
                target,
                quality_mode=_quality_mode(ctx),
            )
            if ctx.settings.enable_translation and target.strip().lower() == "ru"
            else None,
        )
        ctx.store.save_note_artifact(ctx.job.video_id, markdown)
        return StageResult.completed()

    def _deterministic_graph(self, ctx: StageContext) -> StageResult:
        target = _target_language(ctx)
        _require_analysis_ru_review_passed(ctx, "deterministic_graph")
        target_analysis_for_cache = ctx.store.load_analysis(ctx.job.video_id, variant=target) if ctx.settings.enable_translation else None
        target_transcript_for_cache = ctx.store.load_translated_transcript(ctx.job.video_id, target) if ctx.settings.enable_translation else None
        source_analysis_for_cache = (
            ctx.store.load_analysis(ctx.job.video_id, variant="source") or ctx.store.load_analysis(ctx.job.video_id)
            if not ctx.settings.enable_translation
            else None
        )
        source_transcript_for_cache = ctx.store.load_normalized_transcript(ctx.job.video_id) if not ctx.settings.enable_translation else None
        cached = ctx.store.load_graph_extraction(ctx.job.video_id)
        if (
            _graph_matches_target(cached, target, target_analysis_for_cache, target_transcript_for_cache)
            if ctx.settings.enable_translation
            else _graph_matches_source(cached, source_analysis_for_cache, source_transcript_for_cache)
        ):
            return StageResult.completed(warning="graph cache hit")
        metadata = ctx.store.load_video_metadata(ctx.job.video_id)
        transcript = None
        analysis_data = None
        generated_transcript = "transcript.normalized.json"
        generated_analysis = "analysis.source.json" if ctx.settings.enable_translation else "analysis.json"
        generated_language: str | None = None
        if ctx.settings.enable_translation:
            target_transcript = ctx.store.load_translated_transcript(ctx.job.video_id, target)
            target_analysis = ctx.store.load_analysis(ctx.job.video_id, variant=target)
            if _transcript_matches_target_language(target_transcript, target) and _analysis_data_matches_target_language(target_analysis, target):
                transcript = target_transcript
                analysis_data = target_analysis
                generated_transcript = f"transcript.{target}.json"
                generated_analysis = f"analysis.{target}.json"
                generated_language = target
            elif not ctx.settings.translation_best_effort:
                raise RuntimeError(f"transcript.{target}.json and analysis.{target}.json must pass {target} gates before deterministic_graph")
        if transcript is None:
            transcript = ctx.store.load_normalized_transcript(ctx.job.video_id)
        if analysis_data is None:
            analysis_data = ctx.store.load_analysis(ctx.job.video_id, variant="source") or ctx.store.load_analysis(ctx.job.video_id)
        if metadata is None or transcript is None or analysis_data is None:
            raise RuntimeError("metadata, transcript, and analysis are required")
        from yt_sub_hub.llm import parse_analysis_response

        analysis = parse_analysis_response(analysis_data)
        extraction = build_deterministic_graph_extraction(metadata, transcript, analysis)
        generated_from = extraction.setdefault("generated_from", {})
        generated_from["transcript"] = generated_transcript
        generated_from["analysis"] = generated_analysis
        generated_from["language"] = generated_language or _source_graph_language(analysis, transcript, target)
        generated_from["analysis_hash"] = _analysis_data_hash(analysis_data)
        generated_from["transcript_hash"] = _transcript_hash(transcript)
        visual_summary = ctx.store.load_visual_summary(ctx.job.video_id)
        if visual_summary is not None:
            from yt_sub_hub.graph_extraction import merge_visual_summary_into_extraction

            extraction = merge_visual_summary_into_extraction(extraction, visual_summary)
        ctx.store.save_graph_extraction(ctx.job.video_id, extraction)
        return StageResult.completed()

    def _deep_graph_llm(self, ctx: StageContext) -> StageResult:
        if not ctx.settings.enable_deep_graph_llm:
            return StageResult.skipped("deep graph disabled")
        _require_analysis_ru_review_passed(ctx, "deep_graph_llm")
        metadata = ctx.store.load_video_metadata(ctx.job.video_id)
        target = _target_language(ctx)
        generated_transcript = "transcript.normalized.json"
        generated_analysis = "analysis.source.json" if ctx.settings.enable_translation else "analysis.json"
        generated_language: str | None = None
        if ctx.settings.enable_translation:
            target_transcript = ctx.store.load_translated_transcript(ctx.job.video_id, target)
            target_analysis = ctx.store.load_analysis(ctx.job.video_id, variant=target)
            if _transcript_matches_target_language(target_transcript, target) and _analysis_data_matches_target_language(target_analysis, target):
                transcript = target_transcript
                analysis_data = target_analysis
                generated_transcript = f"transcript.{target}.json"
                generated_analysis = f"analysis.{target}.json"
                generated_language = target
            elif ctx.settings.translation_best_effort:
                transcript = ctx.store.load_normalized_transcript(ctx.job.video_id)
                analysis_data = ctx.store.load_analysis(ctx.job.video_id, variant="source") or ctx.store.load_analysis(ctx.job.video_id)
            else:
                raise RuntimeError(f"transcript.{target}.json and analysis.{target}.json must pass {target} gates before deep_graph_llm")
        else:
            transcript = ctx.store.load_normalized_transcript(ctx.job.video_id)
            analysis_data = ctx.store.load_analysis(ctx.job.video_id)
        if metadata is None or transcript is None or analysis_data is None:
            raise RuntimeError("metadata, transcript, and analysis are required")
        from yt_sub_hub.llm import parse_analysis_response

        extraction, warning, _ = ctx.pipeline._build_graph_extraction(
            metadata,
            transcript,
            parse_analysis_response(analysis_data),
            previous_extraction=ctx.store.load_graph_extraction(ctx.job.video_id),
        )
        generated_from = extraction.setdefault("generated_from", {})
        generated_from["transcript"] = generated_transcript
        generated_from["analysis"] = generated_analysis
        generated_from["language"] = generated_language or _source_graph_language(parse_analysis_response(analysis_data), transcript, target)
        if generated_analysis == f"analysis.{target}.json":
            generated_from["analysis_hash"] = _analysis_data_hash(analysis_data)
        generated_from["transcript_hash"] = _transcript_hash(transcript)
        ctx.store.save_graph_extraction(ctx.job.video_id, extraction)
        return StageResult.completed(warning=warning)

    def _visual_frame_analysis(self, ctx: StageContext) -> StageResult:
        return StageResult.skipped("visual frame analysis is on-demand")

    def _ru_artifact_review(self, ctx: StageContext) -> StageResult:
        if _quality_mode(ctx) == "draft":
            return StageResult.skipped("draft mode skips graph sync and production validation")
        if not ctx.settings.enable_translation:
            return StageResult.skipped("translation disabled")
        target = _target_language(ctx).strip().lower()
        if target != "ru":
            return StageResult.skipped("ru_artifact_review only supports ru")
        report = review_ru_artifacts(ctx.job.video_id, ctx.store, ctx.settings)
        hard = [issue for issue in report.issues if issue.severity == "hard"]
        if hard:
            return StageResult.failed("RU artifact review hard failures: " + "; ".join(f"{issue.code}: {issue.message}" for issue in hard))
        warnings = [issue for issue in report.issues if issue.severity == "warning"]
        if warnings:
            return StageResult.completed(warning="RU artifact review warnings: " + "; ".join(f"{issue.code}: {issue.message}" for issue in warnings))
        return StageResult.completed()

    def _neo4j_sync(self, ctx: StageContext) -> StageResult:
        if _quality_mode(ctx) == "draft":
            return StageResult.skipped("draft mode skips graph sync and production validation")
        if not ctx.settings.enable_graph:
            return StageResult.skipped("graph sync disabled")
        _require_analysis_ru_review_passed(ctx, "neo4j_sync")
        _require_ru_artifact_review_passed(ctx, "neo4j_sync")
        extraction = ctx.store.load_graph_extraction(ctx.job.video_id)
        if extraction is None:
            raise RuntimeError("graph extraction is required")
        target = _target_language(ctx)
        target_analysis = ctx.store.load_analysis(ctx.job.video_id, variant=target) if ctx.settings.enable_translation else None
        target_transcript = ctx.store.load_translated_transcript(ctx.job.video_id, target) if ctx.settings.enable_translation else None
        if ctx.settings.enable_translation and not _graph_matches_target(extraction, target, target_analysis, target_transcript):
            raise RuntimeError(f"graph_extraction.json must be generated from {target} transcript and analysis before neo4j_sync")
        if ctx.settings.enable_translation and target.strip().lower() == "ru":
            quality = run_quality_gate(ctx.store, ctx.job.video_id, target_language=target)
            product_quality = evaluate_product_quality(
                _quality_mode(ctx),
                quality,
                load_ru_artifact_review(ctx.store, ctx.job.video_id),
                ctx.runtime.list_stage_plan(ctx.job.id) if ctx.runtime is not None else [],
            )
            if not product_quality["graph_sync_allowed"]:
                raise RuntimeError(
                    f"graph sync is not allowed in {_quality_mode(ctx)} mode: "
                    + "; ".join(product_quality["reasons"])
                )
        graph_store = Neo4jGraphStore(ctx.settings.neo4j_uri, ctx.settings.neo4j_user, ctx.settings.neo4j_password)
        try:
            graph_store.delete_video_relationships(ctx.job.video_id)
            graph_store.upsert_payload(graph_payload_from_extraction(extraction))
        finally:
            graph_store.close()
        return StageResult.completed()

    def _quality_gate(self, ctx: StageContext) -> StageResult:
        if _quality_mode(ctx) == "draft":
            return StageResult.completed(warning="draft mode: not production validated")
        _require_ru_artifact_review_passed(ctx, "quality_gate")
        result = run_quality_gate(ctx.store, ctx.job.video_id, target_language=ctx.settings.target_language)
        if result.hard_failures:
            return StageResult.failed("Quality gate hard failures: " + "; ".join(result.hard_failures))
        if result.warnings:
            return StageResult.completed(warning="Quality gate warnings: " + "; ".join(result.warnings))
        return StageResult.completed()

    def _youtube_discovery_seed_extract(self, ctx: StageContext) -> StageResult:
        from yt_sub_hub import youtube_discovery as discovery

        options = discovery.read_discovery_stage_options(ctx.stage.stage_payload_json)
        discovery.extract_discovery_seeds_stage(
            ctx.store,
            ctx.job.video_id,
            limit=options["limit"],
            max_quota_units=options["max_quota_units"],
        )
        return StageResult.completed()

    def _youtube_discovery_query_expand(self, ctx: StageContext) -> StageResult:
        from yt_sub_hub import youtube_discovery as discovery

        if ctx.runtime is None:
            raise RuntimeError("runtime is required for youtube discovery query expansion")
        options = discovery.read_discovery_stage_options(ctx.stage.stage_payload_json)
        queries = discovery.expand_discovery_queries_stage(
            ctx.store,
            ctx.job.video_id,
            max_quota_units=options["max_quota_units"],
        )
        ctx.runtime.expand_youtube_discovery_query_plan(ctx.job.id, queries)
        return StageResult.completed()

    def _youtube_discovery_query(self, ctx: StageContext) -> StageResult:
        from yt_sub_hub import youtube_discovery as discovery

        payload = _stage_payload(ctx.stage)
        query = payload.get("query") or {}
        options = discovery.read_discovery_stage_options(_discovery_base_stage_payload(ctx))
        discovery.execute_discovery_query_stage(
            ctx.store,
            ctx.job.video_id,
            query,
            provider=self.discovery_provider,
            settings=ctx.settings,
            limit=options["limit"],
            max_quota_units=options["max_quota_units"],
        )
        return StageResult.completed()

    def _youtube_discovery_candidate_fetch(self, ctx: StageContext) -> StageResult:
        from yt_sub_hub import youtube_discovery as discovery

        query_statuses: dict[str, str] = {}
        if ctx.runtime is not None:
            for stage in ctx.runtime.list_stage_plan(ctx.job.id):
                if not stage.stage_name.startswith("youtube_discovery_query_") or stage.stage_name == "youtube_discovery_query_expand":
                    continue
                payload = _stage_payload(stage)
                query_id = str(payload.get("query_id") or "")
                if query_id:
                    query_statuses[query_id] = stage.status
        discovery.merge_discovery_query_results_stage(ctx.store, ctx.job.video_id, query_statuses)
        return StageResult.completed()

    def _youtube_discovery_candidate_score(self, ctx: StageContext) -> StageResult:
        from yt_sub_hub import youtube_discovery as discovery

        options = discovery.read_discovery_stage_options(_discovery_base_stage_payload(ctx))
        discovery.score_discovery_candidates_stage(ctx.store, ctx.job.video_id, limit=options["limit"])
        return StageResult.completed()

    def _youtube_discovery_review(self, ctx: StageContext) -> StageResult:
        from yt_sub_hub import youtube_discovery as discovery

        try:
            review = discovery.review_discovery_candidates_stage(ctx.store, ctx.job.video_id)
        except discovery.DiscoveryValidationError as exc:
            return StageResult.failed(str(exc))
        if review.get("status") == "warning":
            return StageResult.completed(warning="youtube discovery review warnings")
        return StageResult.completed()

    def _youtube_discovery_graph(self, ctx: StageContext) -> StageResult:
        from yt_sub_hub import youtube_discovery as discovery

        discovery.build_discovery_graph_stage(ctx.store, ctx.job.video_id)
        return StageResult.completed()

    def _youtube_discovery_report(self, ctx: StageContext) -> StageResult:
        from yt_sub_hub import youtube_discovery as discovery

        discovery.render_discovery_report_stage(ctx.store, ctx.job.video_id)
        return StageResult.completed()


def _analysis_input_transcript(ctx: StageContext) -> tuple[Transcript | None, str, str]:
    target = _target_language(ctx)
    target_transcript = ctx.store.load_translated_transcript(ctx.job.video_id, target) if ctx.settings.enable_translation else None
    if ctx.settings.enable_translation:
        if _transcript_matches_target_language(target_transcript, target):
            return target_transcript, f"transcript.{target}.json", target
        if not ctx.settings.translation_best_effort:
            raise RuntimeError(f"transcript.{target}.json must exist and pass {target} language gate before scientific_analysis")
    return ctx.store.load_normalized_transcript(ctx.job.video_id), "transcript.normalized.json", "source"


def _plan_analysis_chunks(transcript: Transcript, settings: Settings, source_transcript: str, language: str) -> list[dict[str, Any]]:
    max_segments = max(1, int(getattr(settings, "analysis_chunk_max_segments", 120)))
    max_chars = max(1, int(getattr(settings, "analysis_chunk_max_chars", 4500)))
    chunks: list[dict[str, Any]] = []
    start_index = 0
    current_chars = 0
    source_hash = _transcript_hash(transcript)
    for index, segment in enumerate(transcript.segments):
        size = len(segment.text or "") + 24
        too_many_segments = index - start_index >= max_segments
        too_many_chars = index > start_index and current_chars + size > max_chars
        if too_many_segments or too_many_chars:
            chunks.append(
                _analysis_chunk_payload(
                    transcript,
                    source_transcript,
                    language,
                    source_hash,
                    len(chunks),
                    start_index,
                    index - 1,
                    settings,
                )
            )
            start_index = index
            current_chars = 0
        current_chars += size
    if transcript.segments:
        chunks.append(
            _analysis_chunk_payload(
                transcript,
                source_transcript,
                language,
                source_hash,
                len(chunks),
                start_index,
                len(transcript.segments) - 1,
                settings,
            )
        )
    for chunk in chunks:
        chunk["total_chunks"] = len(chunks)
    return chunks


def _analysis_chunk_payload(
    transcript: Transcript,
    source_transcript: str,
    language: str,
    source_hash: str,
    chunk_index: int,
    start_index: int,
    end_index: int,
    settings: Settings,
) -> dict[str, Any]:
    return {
        "video_id": transcript.video_id,
        "chunk_index": chunk_index,
        "source_transcript": source_transcript,
        "source_segment_start_index": start_index,
        "source_segment_end_index": end_index,
        "source_transcript_hash": source_hash,
        "language": language,
        "analysis_chunk_max_segments": int(getattr(settings, "analysis_chunk_max_segments", 120)),
        "analysis_chunk_max_chars": int(getattr(settings, "analysis_chunk_max_chars", 4500)),
    }


def _load_analysis_payload_transcript(ctx: StageContext, payload: dict[str, Any]) -> Transcript:
    source_transcript = str(payload.get("source_transcript") or "transcript.normalized.json")
    target = _target_language(ctx)
    if source_transcript == f"transcript.{target}.json":
        transcript = ctx.store.load_translated_transcript(ctx.job.video_id, target)
    else:
        transcript = ctx.store.load_normalized_transcript(ctx.job.video_id)
    if transcript is None:
        raise RuntimeError(f"{source_transcript} is required")
    return transcript


def _analysis_chunk_path(store: LibraryStore, video_id: str, chunk_index: int) -> Path:
    return store.video_dir(video_id) / "analysis" / "chunks" / f"{chunk_index:03d}.json"


def _valid_analysis_chunk_artifact(path: Path, transcript: Transcript, payload: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        _validate_analysis_chunk_artifact(artifact, transcript, payload)
    except Exception:
        return False
    return True


def _build_analysis_chunk_artifact(
    ctx: StageContext,
    payload: dict[str, Any],
    source_transcript: Transcript,
    raw_chunk: dict[str, Any],
    analyzer: Any,
) -> dict[str, Any]:
    start_index = int(payload["source_segment_start_index"])
    end_index = int(payload["source_segment_end_index"])
    chunk_index = int(payload["chunk_index"])
    total_chunks = int(payload.get("total_chunks", 1))
    raw_outline = raw_chunk.get("outline")
    if isinstance(raw_outline, list):
        outline = [str(item) for item in raw_outline if str(item).strip()]
    else:
        heading = str(raw_chunk.get("heading") or f"Chunk {chunk_index + 1}")
        outline = [heading]
    summary = str(raw_chunk.get("summary") or "")
    if not summary.strip():
        raise RuntimeError(f"analysis chunk {chunk_index:03d} summary is empty")
    evidence_indexes = _normalize_evidence_indexes(raw_chunk.get("evidence_segment_indexes"), start_index, end_index)
    model = (
        raw_chunk.get("model")
        or getattr(analyzer, "model", None)
        or getattr(analyzer, "gemma_model", None)
        or getattr(ctx.settings, "gemma_model", None)
    )
    return {
        "video_id": ctx.job.video_id,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "source_transcript": payload.get("source_transcript") or "transcript.normalized.json",
        "source_segment_start_index": start_index,
        "source_segment_end_index": end_index,
        "source_transcript_hash": payload.get("source_transcript_hash") or _transcript_hash(source_transcript),
        "language": payload.get("language") or "source",
        "model": model,
        "prompt_version": str(raw_chunk.get("prompt_version") or "analysis-chunk-v1"),
        "summary": summary,
        "outline": outline,
        "concepts": _objects_with_string_fields(raw_chunk.get("concepts"), ["name", "definition"]),
        "claims": _objects_with_string_fields(raw_chunk.get("claims"), ["text", "evidence"]),
        "evidence_segment_indexes": evidence_indexes,
        "limitations": _string_list(raw_chunk.get("limitations")),
        "open_questions": _string_list(raw_chunk.get("open_questions")),
        "related_topics": _string_list(raw_chunk.get("related_topics")),
    }


def _normalize_evidence_indexes(values: Any, start_index: int, end_index: int) -> list[int]:
    result: list[int] = []
    if not isinstance(values, list):
        return list(range(start_index, end_index + 1))
    for value in values:
        try:
            index = int(value)
        except Exception:
            continue
        if start_index <= index <= end_index:
            normalized = index
        elif 0 <= index <= end_index - start_index:
            normalized = start_index + index
        else:
            continue
        if normalized not in result:
            result.append(normalized)
    return result or list(range(start_index, end_index + 1))


def _string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(item).strip() for item in values if str(item).strip()]


def _objects_with_string_fields(values: Any, fields: list[str]) -> list[dict[str, str]]:
    if not isinstance(values, list):
        return []
    result: list[dict[str, str]] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        item = {field: str(value.get(field) or "").strip() for field in fields}
        if all(item.values()):
            result.append(item)
    return result


def _validate_analysis_chunk_artifact(artifact: dict[str, Any], transcript: Transcript, payload: dict[str, Any]) -> None:
    if artifact.get("video_id") != transcript.video_id:
        raise RuntimeError("analysis chunk video_id mismatch")
    if int(artifact.get("chunk_index", -1)) != int(payload["chunk_index"]):
        raise RuntimeError("analysis chunk index mismatch")
    if int(artifact.get("total_chunks", -1)) != int(payload.get("total_chunks", 1)):
        raise RuntimeError("analysis chunk total mismatch")
    if str(artifact.get("source_transcript") or "") != str(payload.get("source_transcript") or "transcript.normalized.json"):
        raise RuntimeError("analysis chunk source transcript mismatch")
    if artifact.get("source_transcript_hash") != payload.get("source_transcript_hash"):
        raise RuntimeError("analysis chunk source transcript hash mismatch")
    if artifact.get("source_transcript_hash") != _transcript_hash(transcript):
        raise RuntimeError("analysis chunk source transcript hash is stale")
    if str(artifact.get("language") or "") != str(payload.get("language") or "source"):
        raise RuntimeError("analysis chunk language mismatch")
    start_index = int(payload["source_segment_start_index"])
    end_index = int(payload["source_segment_end_index"])
    if int(artifact.get("source_segment_start_index", -1)) != start_index:
        raise RuntimeError("analysis chunk start index mismatch")
    if int(artifact.get("source_segment_end_index", -1)) != end_index:
        raise RuntimeError("analysis chunk end index mismatch")
    if end_index < start_index or start_index < 0 or end_index >= len(transcript.segments):
        raise RuntimeError("analysis chunk range outside transcript")
    if not str(artifact.get("summary") or "").strip():
        raise RuntimeError("analysis chunk summary is required")
    if not isinstance(artifact.get("outline"), list):
        raise RuntimeError("analysis chunk outline must be a list")
    if not isinstance(artifact.get("concepts"), list):
        raise RuntimeError("analysis chunk concepts must be a list")
    if not isinstance(artifact.get("claims"), list):
        raise RuntimeError("analysis chunk claims must be a list")
    evidence = artifact.get("evidence_segment_indexes")
    if not isinstance(evidence, list) or any(int(index) < start_index or int(index) > end_index for index in evidence):
        raise RuntimeError("analysis chunk evidence indexes outside chunk range")
    if str(payload.get("language") or "").strip().lower() == "ru" and not _contains_cyrillic(
        " ".join([str(artifact.get("summary") or ""), *[str(item) for item in artifact.get("outline", [])]])
    ):
        raise RuntimeError("analysis chunk text must contain Cyrillic")


def _commit_analysis_chunk(path: Path, artifact: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _analysis_chunk_payloads(ctx: StageContext) -> list[dict[str, Any]]:
    if ctx.runtime is None:
        raise RuntimeError("runtime is required for analysis reduce")
    plans = [stage for stage in ctx.runtime.list_stage_plan(ctx.job.id) if stage.stage_name.startswith("analysis_chunk_")]
    payloads = [_stage_payload(stage) for stage in plans]
    payloads.sort(key=lambda item: int(item["chunk_index"]))
    return payloads


def _load_analysis_chunk_artifacts(
    ctx: StageContext,
    transcript: Transcript,
    payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen: set[int] = set()
    artifacts: list[dict[str, Any]] = []
    for payload in payloads:
        chunk_index = int(payload["chunk_index"])
        path = _analysis_chunk_path(ctx.store, ctx.job.video_id, chunk_index)
        if not path.exists():
            raise RuntimeError(f"missing analysis chunk {chunk_index:03d}")
        artifact = json.loads(path.read_text(encoding="utf-8"))
        _validate_analysis_chunk_artifact(artifact, transcript, payload)
        start_index = int(payload["source_segment_start_index"])
        end_index = int(payload["source_segment_end_index"])
        for index in range(start_index, end_index + 1):
            if index in seen:
                raise RuntimeError(f"duplicate analysis chunk coverage at segment {index}")
            seen.add(index)
        artifacts.append(artifact)
    expected = set(range(len(transcript.segments)))
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        if missing:
            raise RuntimeError(f"analysis chunks do not cover source transcript; missing indexes: {missing[:5]}")
        raise RuntimeError(f"analysis chunks cover indexes outside source transcript: {extra[:5]}")
    return artifacts


def _reduce_analysis_chunks(metadata: Any, transcript: Transcript, artifacts: list[dict[str, Any]]) -> Analysis:
    sections: list[NoteSection] = []
    concepts_by_slug: dict[str, Concept] = {}
    claims_by_slug: dict[str, Claim] = {}
    limitations: list[str] = []
    open_questions: list[str] = []
    related_topics: list[str] = []
    language = "ru" if (transcript.language or "").strip().lower() == "ru" else (transcript.language or "source")
    for artifact in artifacts:
        start_index = int(artifact["source_segment_start_index"])
        heading = next((str(item).strip() for item in artifact.get("outline", []) if str(item).strip()), f"Chunk {int(artifact['chunk_index']) + 1}")
        sections.append(
            NoteSection(
                heading=heading,
                summary=str(artifact.get("summary") or ""),
                timestamp_seconds=int(transcript.segments[start_index].start) if transcript.segments else 0,
                bullets=[str(item) for item in artifact.get("outline", [])[1:4]],
            )
        )
        for concept_data in artifact.get("concepts", []):
            concept = Concept(name=str(concept_data.get("name") or ""), definition=str(concept_data.get("definition") or ""))
            if concept.slug:
                concepts_by_slug.setdefault(concept.slug, concept)
        for claim_data in artifact.get("claims", []):
            claim = Claim(text=str(claim_data.get("text") or ""), evidence=str(claim_data.get("evidence") or ""))
            if claim.slug:
                claims_by_slug.setdefault(claim.slug, claim)
        limitations.extend(_string_list(artifact.get("limitations")))
        open_questions.extend(_string_list(artifact.get("open_questions")))
        related_topics.extend(_string_list(artifact.get("related_topics")))
    concepts = list(concepts_by_slug.values())
    claims = list(claims_by_slug.values())
    _fill_minimum_analysis_items(language, sections, concepts, claims, related_topics)
    summaries = [section.summary for section in sections if section.summary]
    abstract = _trim_text(" ".join(summaries), 900) or metadata.title
    return Analysis(
        language=language,
        cleaned_title=metadata.title,
        abstract=abstract,
        tags=_dedupe_texts(related_topics, limit=8),
        outline=[section.heading for section in sections],
        sections=sections,
        concepts=concepts,
        claims=claims,
        limitations=_dedupe_texts(limitations) or _default_reduce_list(language, "limitations"),
        open_questions=_dedupe_texts(open_questions) or _default_reduce_list(language, "open_questions"),
        related_topics=_dedupe_texts(related_topics),
        prompt_version="scientific-note-map-reduce-v1",
        model=next((str(artifact.get("model")) for artifact in artifacts if artifact.get("model")), None),
    )


def _fill_minimum_analysis_items(
    language: str,
    sections: list[NoteSection],
    concepts: list[Concept],
    claims: list[Claim],
    related_topics: list[str],
) -> None:
    ru = language.strip().lower() == "ru"
    while sections and len(concepts) < 3:
        section = sections[min(len(concepts), len(sections) - 1)]
        name = section.heading if section.heading else ("Основная тема" if ru else "Main topic")
        definition = section.summary or ("Тема выделена из фрагмента расшифровки." if ru else "Topic extracted from the transcript chunk.")
        concept = Concept(name=name, definition=_trim_text(definition, 260))
        if concept.slug and all(existing.slug != concept.slug for existing in concepts):
            concepts.append(concept)
        else:
            break
    while sections and len(claims) < 3:
        section = sections[min(len(claims), len(sections) - 1)]
        text = (
            f"Раздел «{section.heading}» раскрывает важную часть видео."
            if ru
            else f"The section '{section.heading}' captures an important part of the video."
        )
        evidence = section.summary or ("Сводка раздела взята из расшифровки." if ru else "The section summary is extracted from the transcript.")
        claim = Claim(text=text, evidence=_trim_text(evidence, 320))
        if claim.slug and all(existing.slug != claim.slug for existing in claims):
            claims.append(claim)
        else:
            break
    if not related_topics:
        related_topics.extend([concept.name for concept in concepts[:5]])


def _default_reduce_list(language: str, field: str) -> list[str]:
    ru = language.strip().lower() == "ru"
    if field == "limitations":
        return ["Ограничения явно не указаны в фрагментах анализа."] if ru else ["Limitations were not explicitly stated in the chunk analyses."]
    if field == "open_questions":
        return ["Открытые вопросы явно не указаны в фрагментах анализа."] if ru else ["Open questions were not explicitly stated in the chunk analyses."]
    return []


def _dedupe_texts(values: list[str], limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        key = slugify(text) or text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
        if limit is not None and len(result) >= limit:
            break
    return result


def _trim_text(value: str, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] or text[:max_chars]


def plan_translation_chunks(transcript: Transcript, settings: Settings, target: str) -> list[dict[str, Any]]:
    max_segments = max(1, int(getattr(settings, "translation_chunk_max_segments", 30)))
    max_chars = max(1, int(getattr(settings, "translation_chunk_max_chars", 2000)))
    chunks: list[dict[str, Any]] = []
    start_index = 0
    current_chars = 0
    source_hash = _transcript_hash(transcript)
    for index, segment in enumerate(transcript.segments):
        size = len(segment.text or "") + 24
        too_many_segments = index - start_index >= max_segments
        too_many_chars = index > start_index and current_chars + size > max_chars
        if too_many_segments or too_many_chars:
            chunks.append(
                _translation_chunk_payload(
                    transcript,
                    target,
                    source_hash,
                    len(chunks),
                    start_index,
                    index - 1,
                    settings,
                )
            )
            start_index = index
            current_chars = 0
        current_chars += size
    if transcript.segments:
        chunks.append(
            _translation_chunk_payload(
                transcript,
                target,
                source_hash,
                len(chunks),
                start_index,
                len(transcript.segments) - 1,
                settings,
            )
        )
    for chunk in chunks:
        chunk["total_chunks"] = len(chunks)
    return chunks


def _plan_translation_chunks(transcript: Transcript, settings: Settings, target: str) -> list[dict[str, Any]]:
    return plan_translation_chunks(transcript, settings, target)


def slice_translation_chunk(transcript: Transcript, payload: dict[str, Any]) -> Transcript:
    start_index = int(payload["source_segment_start_index"])
    end_index = int(payload["source_segment_end_index"])
    source_segments = transcript.segments[start_index : end_index + 1]
    return Transcript(
        video_id=transcript.video_id,
        source=transcript.source,
        language=transcript.language,
        segments=[
            TranscriptSegment(segment.start, segment.duration, segment.text, segment.raw_text)
            for segment in source_segments
        ],
    )


def _translation_chunk_payload(
    transcript: Transcript,
    target: str,
    source_hash: str,
    chunk_index: int,
    start_index: int,
    end_index: int,
    settings: Settings,
) -> dict[str, Any]:
    return {
        "video_id": transcript.video_id,
        "target_language": target,
        "chunk_index": chunk_index,
        "source_segment_start_index": start_index,
        "source_segment_end_index": end_index,
        "source_transcript_hash": source_hash,
        "translation_chunk_max_segments": int(getattr(settings, "translation_chunk_max_segments", 30)),
        "translation_chunk_max_chars": int(getattr(settings, "translation_chunk_max_chars", 2000)),
    }


def _stage_payload(stage: StagePlanRecord) -> dict[str, Any]:
    if not stage.stage_payload_json:
        raise RuntimeError(f"stage payload is required for {stage.stage_name}")
    data = json.loads(stage.stage_payload_json)
    if not isinstance(data, dict):
        raise RuntimeError(f"stage payload must be an object for {stage.stage_name}")
    return data


def _discovery_base_stage_payload(ctx: StageContext) -> str | None:
    if ctx.runtime is None:
        return ctx.stage.stage_payload_json
    for stage in ctx.runtime.list_stage_plan(ctx.job.id):
        if stage.stage_name == "youtube_discovery_seed_extract":
            return stage.stage_payload_json
    return ctx.stage.stage_payload_json


def _translation_chunk_path(store: LibraryStore, video_id: str, chunk_index: int) -> Path:
    return store.video_dir(video_id) / "translation" / "chunks" / f"{chunk_index:03d}.json"


def _valid_translation_chunk_artifact(path: Path, transcript: Transcript, payload: dict[str, Any], target: str) -> bool:
    if not path.exists():
        return False
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        _validate_translation_chunk_artifact(artifact, transcript, payload, target)
    except Exception:
        return False
    return True


def _build_translation_chunk_artifact(
    ctx: StageContext,
    payload: dict[str, Any],
    source_transcript: Transcript,
    translated: Transcript,
    target: str,
    provider: Any,
) -> dict[str, Any]:
    start_index = int(payload["source_segment_start_index"])
    end_index = int(payload["source_segment_end_index"])
    source_segments = source_transcript.segments[start_index : end_index + 1]
    if len(translated.segments) != len(source_segments):
        raise RuntimeError("translation chunk segment count does not match source segment count")
    segments: list[dict[str, Any]] = []
    for offset, (source, translated_segment) in enumerate(zip(source_segments, translated.segments, strict=True)):
        index = start_index + offset
        segments.append(
            {
                "index": index,
                "start": source.start,
                "duration": source.duration,
                "source_text": source.text,
                "text": translated_segment.text,
                "raw_text": source.raw_text,
            }
        )
    model = getattr(provider, "model", None) or getattr(ctx.settings, "translation_model", None)
    prompt_version = None
    raw_response = getattr(provider, "last_raw_response", None)
    if isinstance(raw_response, dict):
        prompt_version = raw_response.get("prompt_version")
    return {
        "video_id": ctx.job.video_id,
        "target_language": target,
        "language": target,
        "chunk_index": int(payload["chunk_index"]),
        "total_chunks": int(payload.get("total_chunks", 1)),
        "source_segment_start_index": start_index,
        "source_segment_end_index": end_index,
        "source_transcript_hash": payload.get("source_transcript_hash") or _transcript_hash(source_transcript),
        "model": model,
        "prompt_version": prompt_version or "translation-v1",
        "segments": segments,
    }


def _validate_translation_chunk_artifact(
    artifact: dict[str, Any],
    transcript: Transcript,
    payload: dict[str, Any],
    target: str,
) -> None:
    if artifact.get("video_id") != transcript.video_id:
        raise RuntimeError("translation chunk video_id mismatch")
    if str(artifact.get("target_language") or artifact.get("language") or "").strip().lower() != target.strip().lower():
        raise RuntimeError("translation chunk language mismatch")
    if int(artifact.get("chunk_index", -1)) != int(payload["chunk_index"]):
        raise RuntimeError("translation chunk index mismatch")
    if artifact.get("source_transcript_hash") and payload.get("source_transcript_hash"):
        if artifact.get("source_transcript_hash") != payload.get("source_transcript_hash"):
            raise RuntimeError("translation chunk source transcript hash mismatch")
    start_index = int(payload["source_segment_start_index"])
    end_index = int(payload["source_segment_end_index"])
    if int(artifact.get("source_segment_start_index", -1)) != start_index:
        raise RuntimeError("translation chunk start index mismatch")
    if int(artifact.get("source_segment_end_index", -1)) != end_index:
        raise RuntimeError("translation chunk end index mismatch")
    segments = artifact.get("segments")
    if not isinstance(segments, list):
        raise RuntimeError("translation chunk segments must be a list")
    if len(segments) != end_index - start_index + 1:
        raise RuntimeError("translation chunk segment count mismatch")
    translated_texts: list[str] = []
    for offset, item in enumerate(segments):
        if not isinstance(item, dict):
            raise RuntimeError("translation chunk segment must be an object")
        expected_index = start_index + offset
        source = transcript.segments[expected_index]
        if int(item.get("index", -1)) != expected_index:
            raise RuntimeError("translation chunk segment index mismatch")
        if float(item.get("start")) != float(source.start) or float(item.get("duration")) != float(source.duration):
            raise RuntimeError("translation chunk timestamp changed")
        translated_texts.append(str(item.get("text") or ""))
    if target.strip().lower() == "ru" and not _contains_cyrillic("\n".join(translated_texts)):
        raise RuntimeError("translation chunk text must contain Cyrillic")


def _commit_translation_chunk(ctx: StageContext, path: Path, artifact: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _translation_chunk_payloads(ctx: StageContext, target: str) -> list[dict[str, Any]]:
    if ctx.runtime is None:
        raise RuntimeError("runtime is required for transcript translation merge")
    plans = [
        stage
        for stage in ctx.runtime.list_stage_plan(ctx.job.id)
        if stage.stage_name.startswith("transcript_translation_chunk_")
    ]
    if not plans:
        raise RuntimeError("translation chunks are required before transcript translation merge")
    payloads = [_stage_payload(stage) for stage in plans]
    payloads.sort(key=lambda item: int(item["chunk_index"]))
    for payload in payloads:
        if str(payload.get("target_language") or "").strip().lower() != target.strip().lower():
            raise RuntimeError("translation chunk target language mismatch")
    return payloads


def _merge_translation_chunks(
    ctx: StageContext,
    transcript: Transcript,
    payloads: list[dict[str, Any]],
    target: str,
) -> list[TranscriptSegment]:
    seen: set[int] = set()
    translated: dict[int, TranscriptSegment] = {}
    for payload in payloads:
        chunk_index = int(payload["chunk_index"])
        path = _translation_chunk_path(ctx.store, ctx.job.video_id, chunk_index)
        if not path.exists():
            raise RuntimeError(f"missing translation chunk {chunk_index:03d}")
        artifact = json.loads(path.read_text(encoding="utf-8"))
        _validate_translation_chunk_artifact(artifact, transcript, payload, target)
        for segment in artifact["segments"]:
            index = int(segment["index"])
            if index in seen:
                raise RuntimeError(f"duplicate translated segment index {index}")
            seen.add(index)
            source = transcript.segments[index]
            translated[index] = TranscriptSegment(
                source.start,
                source.duration,
                str(segment.get("text") or ""),
                source.raw_text,
            )
    expected = set(range(len(transcript.segments)))
    if seen != expected:
        missing = sorted(expected - seen)
        raise RuntimeError(f"translation chunks do not cover source transcript; missing indexes: {missing[:5]}")
    return [translated[index] for index in range(len(transcript.segments))]


def _default_chunk_translation_provider(ctx: StageContext):
    from yt_sub_hub.translation import LMStudioTranslationProvider

    return LMStudioTranslationProvider(
        ctx.settings.lm_studio_base_url,
        ctx.settings.lm_studio_api_key,
        ctx.settings.translation_model,
        target_language=_target_language(ctx),
        max_chars=max(1, int(getattr(ctx.settings, "translation_chunk_max_chars", ctx.settings.translation_max_chars))),
        timeout_seconds=float(getattr(ctx.settings, "translation_chunk_timeout_seconds", ctx.settings.lm_studio_timeout_seconds)),
    )


def _transcript_hash(transcript: Transcript) -> str:
    payload = json.dumps(transcript.to_dict(), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _target_language(ctx: StageContext) -> str:
    return str(getattr(ctx.settings, "target_language", None) or "ru").strip() or "ru"


def _transcript_matches_target_language(transcript: Any | None, target: str) -> bool:
    if transcript is None:
        return False
    if (transcript.language or "").strip().lower() != target.strip().lower():
        return False
    if target.strip().lower() != "ru":
        return True
    return _contains_cyrillic(transcript.full_text())


def _analysis_data_matches_target_language(analysis_data: dict[str, Any] | None, target: str) -> bool:
    if analysis_data is None:
        return False
    try:
        from yt_sub_hub.llm import parse_analysis_response

        analysis = parse_analysis_response(analysis_data)
    except Exception:
        return False
    return _analysis_matches_target_language(analysis, target)


def _analysis_translation_provenance(
    source_analysis_data: dict[str, Any],
    target_transcript: Transcript | None,
    target: str,
    settings: Settings,
    translated_analysis: Analysis | None = None,
    *,
    strategy: str = "llm_analysis_translation",
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "source_analysis_hash": _analysis_data_hash(source_analysis_data),
        "translation_model": getattr(settings, "translation_model", None),
        "translation_prompt_version": "analysis-translation-v2",
        "generated_from": {
            "analysis": "analysis.source.json",
            "target_language": target,
            "strategy": strategy,
            "source_model": source_analysis_data.get("model"),
            "source_prompt_version": source_analysis_data.get("prompt_version"),
        },
    }
    if translated_analysis is not None:
        data["model"] = translated_analysis.model
        data["prompt_version"] = translated_analysis.prompt_version
    if target_transcript is not None:
        transcript_hash = _transcript_hash(target_transcript)
        data["source_transcript_hash"] = transcript_hash
        data["generated_from"]["transcript"] = f"transcript.{target}.json"
        data["generated_from"]["transcript_hash"] = transcript_hash
    return data


def _write_analysis_translation_diagnostics(
    ctx: StageContext,
    source_analysis_data: dict[str, Any],
    target_transcript: Transcript | None,
    target: str,
    exc: Exception,
) -> None:
    diagnostics_dir = ctx.store.video_dir(ctx.job.video_id) / "translation"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "stage": "analysis_translation",
        "video_id": ctx.job.video_id,
        "target_language": target,
        "source_analysis_hash": _analysis_data_hash(source_analysis_data),
        "model": getattr(ctx.settings, "translation_model", None),
        "prompt_version": "analysis-translation-v2",
        "error": str(exc.__cause__ or exc),
        "error_class": exc.__class__.__name__,
    }
    if target_transcript is not None:
        payload["source_transcript_hash"] = _transcript_hash(target_transcript)
    path = diagnostics_dir / "analysis_translation_error.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _analysis_cache_matches_source(
    analysis_data: dict[str, Any] | None,
    source_analysis_data: dict[str, Any],
    target_transcript: Transcript | None,
    target: str,
) -> bool:
    if not _analysis_data_matches_target_language(analysis_data, target):
        return False
    if analysis_data is None:
        return False
    generated_from = analysis_data.get("generated_from") if isinstance(analysis_data.get("generated_from"), dict) else {}
    if generated_from.get("strategy") != "llm_analysis_translation":
        return False
    if analysis_data.get("translation_prompt_version") != "analysis-translation-v2":
        return False
    if analysis_data.get("source_analysis_hash") != _analysis_data_hash(source_analysis_data):
        return False
    if target_transcript is not None and analysis_data.get("source_transcript_hash") != _transcript_hash(target_transcript):
        return False
    required_sections, required_concepts, required_claims = _minimum_target_analysis_counts(source_analysis_data, target_transcript)
    if (
        len(analysis_data.get("sections") or []) < required_sections
        or len(analysis_data.get("concepts") or []) < required_concepts
        or len(analysis_data.get("claims") or []) < required_claims
    ):
        return False
    return True


def _minimum_target_analysis_counts(source_analysis_data: dict[str, Any], target_transcript: Transcript | None) -> tuple[int, int, int]:
    source_sections = len(source_analysis_data.get("sections") or [])
    source_concepts = len(source_analysis_data.get("concepts") or [])
    source_claims = len(source_analysis_data.get("claims") or [])
    required_sections = source_sections
    required_concepts = source_concepts
    required_claims = source_claims
    return required_sections, required_concepts, required_claims


def _note_provenance(
    analysis_data: dict[str, Any],
    transcript: Transcript | None,
    target: str,
    *,
    quality_mode: str = "strict",
) -> dict[str, str]:
    provenance = {
        "analysis_hash": _analysis_data_hash(analysis_data),
        "generated_from_analysis": f"analysis.{target}.json",
        "quality_mode": normalize_quality_mode(quality_mode),
        "quality_status": "draft" if normalize_quality_mode(quality_mode) == "draft" else "processing",
    }
    if transcript is not None:
        provenance["transcript_hash"] = _transcript_hash(transcript)
    return provenance


def _quality_mode(ctx: StageContext) -> str:
    return normalize_quality_mode(getattr(ctx.job, "quality_mode", None) or getattr(ctx.settings, "quality_mode", "strict"))


def _acceptance_mode(ctx: StageContext) -> str:
    return normalize_acceptance_mode(
        getattr(ctx.job, "acceptance_mode", None) or getattr(ctx.settings, "acceptance_mode", "data_api_only")
    )


def _note_matches_target(note_text: str, analysis_data: dict[str, Any] | None, transcript: Transcript | None) -> bool:
    if analysis_data is None:
        return False
    frontmatter = parse_frontmatter(note_text)
    if frontmatter.get("generated_from_analysis") != "analysis.ru.json":
        return False
    if frontmatter.get("analysis_hash") != _analysis_data_hash(analysis_data):
        return False
    if transcript is not None and frontmatter.get("transcript_hash") != _transcript_hash(transcript):
        return False
    return True


def _require_analysis_ru_review_passed(ctx: StageContext, downstream_stage: str) -> None:
    if not ctx.settings.enable_translation or _target_language(ctx).strip().lower() != "ru" or ctx.runtime is None:
        return
    try:
        stages = ctx.runtime.list_stage_plan(ctx.job.id)
    except Exception:
        return
    review = next((stage for stage in stages if stage.stage_name == "analysis_ru_review"), None)
    if review is None:
        raise RuntimeError(f"analysis_ru_review must be planned before {downstream_stage}")
    if review.status not in {"completed", "completed_with_warning", "skipped"}:
        raise RuntimeError(f"analysis_ru_review must complete before {downstream_stage}")


def _require_ru_artifact_review_passed(ctx: StageContext, downstream_stage: str) -> None:
    if not ctx.settings.enable_translation or _target_language(ctx).strip().lower() != "ru" or ctx.runtime is None:
        return
    try:
        stages = ctx.runtime.list_stage_plan(ctx.job.id)
    except Exception:
        return
    review = next((stage for stage in stages if stage.stage_name == "ru_artifact_review"), None)
    if review is None:
        raise RuntimeError(f"ru_artifact_review must be planned before {downstream_stage}")
    if review.status not in {"completed", "completed_with_warning", "skipped"}:
        raise RuntimeError(f"ru_artifact_review must complete before {downstream_stage}")


def _analysis_data_hash(analysis_data: dict[str, Any]) -> str:
    payload = json.dumps(analysis_data, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _deterministic_target_analysis_fallback(
    analysis: Analysis,
    target: str,
    transcript: Transcript | None = None,
) -> Analysis:
    if target.strip().lower() != "ru":
        return analysis

    if transcript is None or not _transcript_matches_target_language(transcript, target):
        raise RuntimeError("valid transcript.ru.json is required for deterministic Russian analysis fallback")

    sections = _fallback_sections_from_transcript(transcript)
    if not sections:
        raise RuntimeError("valid transcript.ru.json is required for deterministic Russian analysis fallback")

    representative = _representative_transcript_segments(transcript, count=4)
    abstract_excerpt = _ru_excerpt(" ".join(segment.text for segment in representative), max_chars=520)
    abstract = f"Русский конспект по расшифровке: {abstract_excerpt}"
    if not _valid_ru_field("abstract", abstract, 0.70):
        abstract = f"Русский конспект по расшифровке: {sections[0].summary}"

    source_concepts = list(getattr(analysis, "concepts", []) or [])
    concept_limit = max(3, min(20, len(source_concepts)))
    if len(source_concepts) >= 10:
        concept_limit = max(10, concept_limit)
    concepts = [_fallback_concept_from_source(concept) for concept in source_concepts[:concept_limit]]
    concepts = [
        concept
        for index, concept in enumerate(concepts)
        if _valid_ru_field(f"concepts[{index}].name", concept.name, 0.50, short_label=True)
        and _valid_ru_field(f"concepts[{index}].definition", concept.definition, 0.65)
    ]
    if not concepts:
        concepts = [
            Concept(name="Основная тема", definition=f"Главная линия видео выделена из русской расшифровки: {_ru_excerpt(sections[0].summary, 220)}"),
            Concept(name="Контекст видео", definition="Сведения о ходе выпуска и обсуждаемых эпизодах взяты из русской расшифровки."),
            Concept(name="Ключевой эпизод", definition=f"Один из важных фрагментов описан в разделе: {_ru_excerpt(sections[min(1, len(sections) - 1)].summary, 220)}"),
        ]

    valid_source_claims = [
        Claim(text=claim.text.strip(), evidence=claim.evidence.strip())
        for index, claim in enumerate(getattr(analysis, "claims", []) or [])
        if _valid_ru_field(f"claims[{index}].text", claim.text, 0.65)
        and _valid_ru_field(f"claims[{index}].evidence", claim.evidence, 0.65)
    ][:10]
    claims = valid_source_claims
    if len(claims) < 10 and len(concepts) >= 10:
        claims.extend(_fallback_claims_from_concepts(concepts, transcript, limit=10 - len(claims)))
    if not claims:
        for section in sections[:3]:
            claims.append(
                Claim(
                    text=f"Раздел «{section.heading}» раскрывает важную часть темы видео.",
                    evidence=section.summary,
                )
            )
    source_tags = [
        item.strip()
        for item in getattr(analysis, "tags", []) or []
        if _valid_ru_field("tags[]", item, 0.50, short_label=True)
    ][:5]
    source_limitations = [
        item.strip()
        for item in getattr(analysis, "limitations", []) or []
        if _valid_ru_field("limitations[]", item, 0.65)
    ][:3]
    source_open_questions = [
        item.strip()
        for item in getattr(analysis, "open_questions", []) or []
        if _valid_ru_field("open_questions[]", item, 0.65)
    ][:3]
    source_related_topics = [
        item.strip()
        for item in getattr(analysis, "related_topics", []) or []
        if _valid_ru_field("related_topics[]", item, 0.50, short_label=True)
    ][:5]
    return Analysis(
        language="ru",
        cleaned_title=(
            _safe_ru_text(getattr(analysis, "cleaned_title", ""), "Конспект видео", 0.50, short_label=True)
            if _contains_cyrillic(getattr(analysis, "cleaned_title", ""))
            else "Конспект видео"
        ),
        abstract=abstract,
        tags=source_tags or ["видео", "конспект", "русская расшифровка"],
        outline=[section.heading for section in sections],
        sections=sections,
        concepts=concepts,
        claims=claims,
        limitations=source_limitations or ["Детали основаны на автоматической русской расшифровке."],
        open_questions=source_open_questions or ["Какие темы требуют дополнительной проверки после просмотра видео?"],
        related_topics=source_related_topics or ["Основная тема", "Контекст видео", "Русская расшифровка"],
        prompt_version=getattr(analysis, "prompt_version", None),
        model=getattr(analysis, "model", None),
    )


def _fallback_concept_from_source(concept: Concept) -> Concept:
    raw_name = " ".join(str(concept.name or "").split()).strip()
    name = _ru_concept_name(raw_name)
    definition = " ".join(str(concept.definition or "").split()).strip()
    if not _valid_ru_field("concept.definition", definition, 0.65):
        if raw_name:
            definition = f"Русское описание проекта {raw_name} связывает этот пункт с темами выпуска и важными анонсами."
        else:
            definition = "Русское описание связывает этот пункт с темами выпуска и важными анонсами."
    return Concept(name=name, definition=definition)


def _ru_concept_name(name: str) -> str:
    clean = " ".join(str(name or "").split()).strip()
    if _valid_ru_field("concept.name", clean, 0.50, short_label=True):
        return clean
    if clean:
        for candidate in (
            f"Игра {clean} в выпуске",
            f"Игровой проект {clean} в этом выпуске",
        ):
            if _valid_ru_field("concept.name", candidate, 0.50, short_label=True):
                return candidate
    return "Игровой проект выпуска"


def _fallback_claims_from_concepts(concepts: list[Concept], transcript: Transcript, *, limit: int) -> list[Claim]:
    excerpt = _ru_excerpt(" ".join(segment.text for segment in _representative_transcript_segments(transcript, count=2)), max_chars=260)
    claims: list[Claim] = []
    for concept in concepts:
        if len(claims) >= limit:
            break
        claims.append(
            Claim(
                text=f"{concept.name} занимает отдельное место в подборке выпуска.",
                evidence=f"Русская расшифровка связывает этот пункт с обсуждением релизов и игровых особенностей: {excerpt}",
            )
        )
    return claims


def _fallback_sections_from_transcript(transcript: Transcript) -> list[NoteSection]:
    segments = [
        segment
        for segment in transcript.segments
        if _valid_ru_field("transcript.segments[].text", segment.text, 0.65)
    ]
    if not segments:
        return []
    if len(segments) < 3:
        section_count = len(segments)
    else:
        section_count = min(6, max(3, len(segments)))
    sections: list[NoteSection] = []
    for index in range(section_count):
        anchor_index = round(index * (len(segments) - 1) / max(1, section_count - 1))
        excerpt = _ru_excerpt(_segment_window_text(segments, anchor_index), max_chars=760)
        summary = f"В этом фрагменте расшифровки: {excerpt}"
        if not _valid_ru_field(f"sections[{index}].summary", summary, 0.65):
            summary = "В этом разделе приведен русский фрагмент расшифровки о содержании видео и основных событиях."
        sections.append(
            NoteSection(
                heading=f"Раздел {index + 1}",
                summary=summary,
                timestamp_seconds=segments[anchor_index].timestamp_url_offset,
                bullets=_fallback_bullets_from_text(excerpt),
            )
        )
    return sections


def _representative_transcript_segments(transcript: Transcript, count: int = 4) -> list[TranscriptSegment]:
    segments = [
        segment
        for segment in transcript.segments
        if _valid_ru_field("transcript.segments[].text", segment.text, 0.65)
    ]
    if not segments or count <= 0:
        return []
    if len(segments) <= count:
        return segments
    selected: list[TranscriptSegment] = []
    seen: set[int] = set()
    for index in range(count):
        segment_index = round(index * (len(segments) - 1) / max(1, count - 1))
        if segment_index not in seen:
            selected.append(segments[segment_index])
            seen.add(segment_index)
    return selected


def _safe_ru_text(value: str | None, fallback: str, min_ratio: float, short_label: bool = False) -> str:
    clean = " ".join(str(value or "").split())
    return clean if _valid_ru_field("safe_ru_text", clean, min_ratio, short_label=short_label) else fallback


def _ru_excerpt(segment_text: str | None, max_chars: int = 360) -> str:
    text = " ".join(str(segment_text or "").split())
    if not text:
        return "Русская расшифровка содержит основной фрагмент видео."
    if not _valid_ru_field("excerpt", text, 0.65):
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
        valid_sentences = [sentence for sentence in sentences if _valid_ru_field("excerpt.sentence", sentence, 0.65)]
        if valid_sentences:
            text = " ".join(valid_sentences)
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rsplit(" ", 1)[0].strip()
    return clipped.rstrip(".,;:") + "."


def _segment_window_text(segments: list[TranscriptSegment], anchor_index: int, *, min_chars: int = 180) -> str:
    texts = [segments[anchor_index].text]
    cursor = anchor_index + 1
    while cursor < len(segments) and len(" ".join(texts)) < min_chars:
        texts.append(segments[cursor].text)
        cursor += 1
    return " ".join(texts)


def _fallback_bullets_from_text(text: str) -> list[str]:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
    bullets: list[str] = []
    for sentence in sentences:
        bullet = _ru_excerpt(sentence, max_chars=220)
        if _valid_ru_field("section.bullet", bullet, 0.65):
            bullets.append(bullet)
        if len(bullets) >= 3:
            break
    if bullets:
        return bullets
    fallback = f"Фрагмент расшифровки: {_ru_excerpt(text, max_chars=220)}"
    if _valid_ru_field("section.bullet", fallback, 0.65):
        return [fallback]
    return ["В этом пункте выделен важный эпизод русской расшифровки."]


def _valid_ru_field(name: str, value: str | None, min_ratio: float, *, short_label: bool = False) -> bool:
    return not validate_ru_text_field(name, value, min_ratio, True, short_label=short_label)


def _graph_matches_target(
    extraction: dict[str, Any] | None,
    target: str,
    analysis_data: dict[str, Any] | None = None,
    transcript: Transcript | None = None,
) -> bool:
    if not extraction:
        return False
    generated_from = extraction.get("generated_from", {})
    matches_target = (
        str(generated_from.get("language") or "").strip().lower() == target.strip().lower()
        and generated_from.get("transcript") == f"transcript.{target}.json"
        and generated_from.get("analysis") == f"analysis.{target}.json"
    )
    if not matches_target:
        return False
    if analysis_data is None:
        analysis_matches = True
    else:
        analysis_matches = generated_from.get("analysis_hash") == _analysis_data_hash(analysis_data)
    if not analysis_matches:
        return False
    if transcript is not None and generated_from.get("transcript_hash") not in {None, _transcript_hash(transcript)}:
        return False
    return True


def _graph_matches_source(
    extraction: dict[str, Any] | None,
    analysis_data: dict[str, Any] | None = None,
    transcript: Transcript | None = None,
) -> bool:
    if not extraction:
        return False
    generated_from = extraction.get("generated_from", {})
    if generated_from.get("transcript") != "transcript.normalized.json":
        return False
    if generated_from.get("analysis") not in {"analysis.json", "analysis.source.json"}:
        return False
    if analysis_data is None:
        analysis_matches = True
    else:
        analysis_matches = generated_from.get("analysis_hash") == _analysis_data_hash(analysis_data)
    if not analysis_matches:
        return False
    if transcript is not None and generated_from.get("transcript_hash") not in {None, _transcript_hash(transcript)}:
        return False
    return True


def _source_graph_language(analysis: Any, transcript: Any, target: str) -> str | None:
    language = (getattr(analysis, "language", None) or getattr(transcript, "language", None) or "").strip()
    if not language:
        return None
    if language.lower() == target.strip().lower():
        return "source"
    return language


def _contains_cyrillic(value: str | None) -> bool:
    return contains_cyrillic(value)
