from __future__ import annotations

import hashlib
from collections.abc import Callable

from yt_sub_hub.artifact_review import analysis_data_hash, load_ru_artifact_review, review_ru_artifacts, transcript_hash
from yt_sub_hub.config import Settings, get_settings, normalize_quality_mode
from yt_sub_hub.graph import Neo4jGraphStore, build_graph_payload
from yt_sub_hub.graph_extraction import build_deterministic_graph_extraction, graph_payload_from_extraction
from yt_sub_hub.ingestion import fetch_video_metadata, parse_youtube_url
from yt_sub_hub.language_quality import contains_cyrillic, validate_ru_analysis
from yt_sub_hub.library import LibraryStore
from yt_sub_hub.llm import LMStudioAnalyzer, parse_analysis_response
from yt_sub_hub.models import Analysis, JobRecord, Transcript, TranscriptBundle, TranscriptSegment, VideoMetadata
from yt_sub_hub.notes import render_markdown_note
from yt_sub_hub.transcripts import TranscriptResolutionError, WhisperAudioProvider, build_transcript_provider, resolve_transcript_bundle
from yt_sub_hub.translation import LMStudioTranslationProvider, TranslationError


def job_id_for_url(url: str) -> str:
    return hashlib.sha1(url.strip().encode("utf-8")).hexdigest()[:16]


class JobCancelledError(RuntimeError):
    pass


def _contains_cyrillic(value: str | None) -> bool:
    return contains_cyrillic(value)


def _word_tokens(value: str | None) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    for char in value or "":
        if char.isalpha():
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def _cyrillic_word_ratio(values: list[str]) -> float:
    tokens = [token for value in values for token in _word_tokens(value)]
    if not tokens:
        return 0.0
    cyrillic_tokens = sum(1 for token in tokens if _contains_cyrillic(token))
    return cyrillic_tokens / len(tokens)


def _analysis_matches_target_language(analysis: Analysis, target: str) -> bool:
    normalized_target = target.strip().lower()
    if normalized_target != "ru":
        return (analysis.language or "").strip().lower() == normalized_target
    return not validate_ru_analysis(analysis)


class VideoProcessingPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        store: LibraryStore | None = None,
        analyzer: LMStudioAnalyzer | None = None,
        graph_store: Neo4jGraphStore | None = None,
        translation_provider=None,
        enable_default_graph: bool = True,
        cancel_checker: Callable[[str], bool] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or LibraryStore(self.settings.library_dir)
        self.analyzer = analyzer or LMStudioAnalyzer(
            self.settings.lm_studio_base_url,
            self.settings.lm_studio_api_key,
            self.settings.qwen_model,
            self.settings.gemma_model,
            max_prompt_chars=self.settings.max_prompt_chars,
            timeout_seconds=self.settings.lm_studio_timeout_seconds,
            deep_graph_max_output_tokens=self.settings.deep_graph_max_output_tokens,
        )
        self.graph_store = graph_store
        self.translation_provider = translation_provider
        self.enable_default_graph = enable_default_graph
        self.cancel_checker = cancel_checker

    @classmethod
    def for_test(cls, library_dir, graph_store=None, cancel_checker: Callable[[str], bool] | None = None):
        settings = Settings(
            library_dir=library_dir,
            lm_studio_base_url="http://localhost:1234/v1",
            lm_studio_api_key="test",
            qwen_model="qwen",
            gemma_model="gemma",
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            keep_audio=True,
            enable_graph=graph_store is not None,
            graph_best_effort=True,
            enable_deep_graph_llm=False,
            deep_graph_min_confidence=0.55,
            deep_graph_max_chunk_chars=2500,
            deep_graph_max_output_tokens=1500,
            deep_graph_best_effort=True,
            lm_studio_timeout_seconds=120,
            max_prompt_chars=24000,
            ytdlp_cookies_from_browser=None,
            ytdlp_proxy=None,
            ytdlp_js_runtime=None,
            ytdlp_socket_timeout=60,
            ytdlp_retries=10,
        )
        return cls(settings=settings, graph_store=graph_store, enable_default_graph=False, cancel_checker=cancel_checker)

    def submit(self, url: str) -> JobRecord:
        parsed = parse_youtube_url(url)
        self.store.mark_stale_running_jobs_failed()
        existing = self.store.load_jobs().get(job_id_for_url(url))
        if existing is not None and existing.status in {"queued", "running"}:
            return existing
        record = JobRecord(
            job_id=job_id_for_url(url),
            url=url,
            video_id=parsed.video_id,
            status="queued",
            stage="queued",
        )
        self.store.save_job(record)
        return record

    def process(self, url: str) -> JobRecord:
        record = self.submit(url)
        if record.status == "running":
            return record
        return self.process_job(record.job_id)

    def already_processed(self, url: str) -> tuple[str, str] | None:
        parsed = parse_youtube_url(url)
        if not self.store.has_completed_video(parsed.video_id):
            return None
        if any(
            job.video_id == parsed.video_id
            and job.status == "failed"
            and str(job.error or "").startswith("RU artifact review hard failures:")
            for job in self.store.load_jobs().values()
        ):
            return None
        if self.settings.enable_translation and self._target_language().strip().lower() == "ru":
            review = load_ru_artifact_review(self.store, parsed.video_id)
            if review is not None and review.get("status") == "failed":
                return None
        if self.settings.enable_translation and not self._target_artifacts_ready(parsed.video_id):
            return None
        if self.settings.enable_deep_graph_llm:
            extraction = self.store.load_graph_extraction(parsed.video_id)
            if not self._has_llm_deep_graph(extraction):
                return None
        return parsed.video_id, str(self.store.note_path(parsed.video_id))

    def process_job(self, job_id: str) -> JobRecord:
        jobs = self.store.load_jobs()
        if job_id not in jobs:
            raise KeyError(f"Unknown job: {job_id}")
        record = jobs[job_id]
        warnings: list[str] = []
        try:
            self.store.update_job_stage(job_id, status="running", stage="running")
            self._raise_if_cancelled(job_id)
            metadata = self._fetch_or_load_metadata(record, warnings)
            self.store.update_job_stage(job_id, status="running", stage="metadata_fetched")
            self._raise_if_cancelled(job_id)
            bundle = self._resolve_or_load_transcript(metadata, warnings, prefer_cached=bool(warnings))
            self.store.update_job_stage(job_id, status="running", stage="transcript_resolved")
            self._raise_if_cancelled(job_id)
            working_transcript, transcript_translated = self._translate_transcript_or_source(bundle.normalized, warnings)
            self._raise_if_cancelled(job_id)
            if transcript_translated:
                self.store.save_translated_transcript(metadata.video_id, working_transcript)
            self._raise_if_cancelled(job_id)
            analysis = self._analyze_or_load_cached(metadata, working_transcript, warnings)
            self._raise_if_cancelled(job_id)
            analysis, analysis_translated = self._translate_analysis_or_source(
                analysis,
                transcript_translated,
                warnings,
            )
            self._raise_if_cancelled(job_id)
            target_language_ready = (
                transcript_translated
                and analysis_translated
                and (working_transcript.language or "").strip().lower() == self._target_language().lower()
                and _analysis_matches_target_language(analysis, self._target_language())
            )
            self.store.update_job_stage(job_id, status="running", stage="analysis_completed")
            self._raise_if_cancelled(job_id)
            paths, graph_warning = self.persist(
                metadata,
                bundle.raw_payload,
                bundle.normalized,
                analysis,
                working_transcript=working_transcript,
                transcript_translated=transcript_translated,
                analysis_translated=target_language_ready,
            )
            if graph_warning:
                warnings.append(graph_warning)
            warning = "; ".join(warnings) if warnings else None
            self.store.update_job_stage(job_id, status="running", stage="artifacts_saved", note_path=str(paths.note))
            final_status = "completed_with_graph_warning" if warning else "completed"
            final_stage = "artifacts_saved" if warning else "graph_synced"
            return self.store.update_job_stage(
                job_id,
                status=final_status,
                stage=final_stage,
                note_path=str(paths.note),
                warning=warning,
            )
        except JobCancelledError:
            return self._mark_cancelled(job_id)
        except Exception as exc:
            self.store.update_job_stage(job_id, status="failed", stage="failed", error=str(exc))
            raise

    def _raise_if_cancelled(self, job_id: str) -> None:
        if self.cancel_checker is not None and self.cancel_checker(job_id):
            raise JobCancelledError("cancel requested")

    def _mark_cancelled(self, job_id: str) -> JobRecord:
        return self.store.update_job_stage(
            job_id,
            status="skipped",
            stage="cancelled",
            warning="cancel requested",
        )

    def _fetch_or_load_metadata(self, record: JobRecord, warnings: list[str]) -> VideoMetadata:
        cached = self.store.load_video_metadata(record.video_id)
        if (
            self.settings.enable_deep_graph_llm
            and cached is not None
            and not self._has_llm_deep_graph(self.store.load_graph_extraction(record.video_id))
        ):
            warnings.append("Used cached metadata to refresh deep graph extraction")
            return cached
        if (
            cached is not None
            and self.settings.enable_translation
            and (
                self.store.load_translated_transcript(record.video_id, self._target_language()) is None
                or not self._target_analysis_artifact_ready(record.video_id)
            )
        ):
            return cached
        try:
            return fetch_video_metadata(record.url, settings=self.settings)
        except Exception as exc:
            if cached is None:
                raise
            warnings.append(f"Used cached metadata because YouTube metadata fetch failed: {exc}")
            return cached

    def _resolve_or_load_transcript(
        self,
        metadata: VideoMetadata,
        warnings: list[str],
        prefer_cached: bool = False,
    ) -> TranscriptBundle:
        cached = self._cached_transcript_bundle(metadata.video_id)
        if cached is not None:
            if prefer_cached:
                if warnings and warnings[-1] == "Used cached metadata to refresh translation artifacts":
                    warnings.append("Used cached transcript to refresh translation artifacts")
                else:
                    warnings.append("Used cached transcript because live metadata fetch already failed")
            return cached
        try:
            whisper_provider = None
            if self.settings.acceptance_mode == "full_processing":
                whisper_provider = WhisperAudioProvider(
                    metadata.canonical_url,
                    self.settings.library_dir / "assets" / "audio",
                    keep_audio=self.settings.keep_audio,
                    settings=self.settings,
                )
            return resolve_transcript_bundle(
                metadata.video_id,
                transcript_provider=build_transcript_provider(
                    self.settings,
                    allow_unofficial_provider=self.settings.acceptance_mode != "data_api_only",
                ),
                whisper_provider=whisper_provider,
            )
        except TranscriptResolutionError as exc:
            cached = self._cached_transcript_bundle(metadata.video_id)
            if cached is None:
                raise
            warnings.append(f"Used cached transcript because transcript resolution failed: {exc}")
            return cached

    def _cached_transcript_bundle(self, video_id: str) -> TranscriptBundle | None:
        normalized = self.store.load_normalized_transcript(video_id)
        if normalized is None:
            return None
        raw_payload = self.store.load_raw_transcript(video_id) or normalized.to_dict()
        return TranscriptBundle(
            video_id=video_id,
            source=normalized.source,
            raw_payload=raw_payload,
            normalized=normalized,
        )

    def _analyze_or_load_cached(
        self,
        metadata: VideoMetadata,
        transcript: Transcript,
        warnings: list[str],
    ) -> Analysis:
        if self.settings.enable_deep_graph_llm and not self._has_llm_deep_graph(
            self.store.load_graph_extraction(metadata.video_id)
        ):
            cached_analysis = (
                self.store.load_analysis(metadata.video_id, variant=self._target_language())
                if self.settings.enable_translation and transcript.language == self._target_language()
                else self.store.load_analysis(metadata.video_id)
            )
            if cached_analysis is not None:
                if (
                    self.settings.enable_translation
                    and transcript.language == self._target_language()
                    and not self._analysis_data_matches_target_language(cached_analysis)
                ):
                    return self.analyzer.analyze(metadata, transcript)
                warnings.append("Used cached analysis to refresh deep graph extraction")
                return parse_analysis_response(cached_analysis)
        return self.analyzer.analyze(metadata, transcript)

    def persist(
        self,
        metadata: VideoMetadata,
        raw_transcript,
        normalized_transcript,
        analysis: Analysis | None = None,
        *,
        working_transcript: Transcript | None = None,
        transcript_translated: bool = False,
        analysis_translated: bool = False,
        translated: bool | None = None,
    ):
        if translated is not None:
            transcript_translated = translated
            analysis_translated = translated
        if analysis is None:
            analysis = normalized_transcript
            normalized_transcript = raw_transcript
        working_transcript = working_transcript or normalized_transcript
        if analysis_translated and not _analysis_matches_target_language(analysis, self._target_language()):
            if not self.settings.translation_best_effort:
                raise TranslationError(self._analysis_translation_failure_message())
            analysis_translated = False
        if transcript_translated and (working_transcript.language or "").strip().lower() != self._target_language().lower():
            if not self.settings.translation_best_effort:
                raise TranslationError(f"Transcript translation did not produce {self._target_language()}")
            transcript_translated = False
        markdown = render_markdown_note(metadata, analysis, provenance=self._quality_note_provenance())
        paths = self.store.save_video_artifacts(metadata, raw_transcript, normalized_transcript, analysis, markdown)
        if transcript_translated:
            self.store.save_translated_transcript(metadata.video_id, working_transcript)
        if analysis_translated:
            self.store.save_analysis_artifact(
                metadata.video_id,
                analysis,
                variant=self._target_language(),
                compatibility=True,
                extra_fields={
                    "source_transcript_hash": transcript_hash(working_transcript) if transcript_translated else None,
                    "generated_from": {
                        "transcript": f"transcript.{self._target_language()}.json" if transcript_translated else "transcript.normalized.json",
                        "target_language": self._target_language(),
                    },
                },
            )
            analysis_data = self.store.load_analysis(metadata.video_id, variant=self._target_language())
            self.store.save_note_artifact(
                metadata.video_id,
                render_markdown_note(
                    metadata,
                    analysis,
                    provenance={
                        "analysis_hash": analysis_data_hash(analysis_data),
                        "transcript_hash": transcript_hash(working_transcript),
                        "generated_from_analysis": f"analysis.{self._target_language()}.json",
                        **self._quality_note_provenance(),
                    },
                ),
            )
        elif self.settings.enable_translation:
            self.store.save_analysis_artifact(metadata.video_id, analysis, variant="source", compatibility=True)
        if self._quality_mode() == "draft":
            return paths, "draft mode skips graph sync and production validation"
        previous_extraction = self.store.load_graph_extraction(metadata.video_id)
        graph_extraction, warning, preserved_previous_extraction = self._build_graph_extraction(
            metadata,
            working_transcript,
            analysis,
            previous_extraction=previous_extraction,
        )
        if not preserved_previous_extraction:
            generated_from = graph_extraction.setdefault("generated_from", {})
            generated_from["transcript"] = (
                f"transcript.{self._target_language()}.json" if transcript_translated else "transcript.normalized.json"
            )
            generated_from["analysis"] = f"analysis.{self._target_language()}.json" if analysis_translated else (
                "analysis.source.json" if self.settings.enable_translation else "analysis.json"
            )
            generated_from["language"] = self._target_language() if analysis_translated else self._source_generated_language(
                analysis,
                normalized_transcript,
            )
            if transcript_translated:
                generated_from["translated_from"] = normalized_transcript.language
                generated_from["translation_model"] = self.settings.translation_model
                generated_from["transcript_hash"] = transcript_hash(working_transcript)
            if analysis_translated:
                analysis_data = self.store.load_analysis(metadata.video_id, variant=self._target_language())
                generated_from["analysis_hash"] = analysis_data_hash(analysis_data)
        visual_summary = self.store.load_visual_summary(metadata.video_id)
        if visual_summary is not None:
            from yt_sub_hub.graph_extraction import merge_visual_summary_into_extraction

            graph_extraction = merge_visual_summary_into_extraction(graph_extraction, visual_summary)
        self.store.save_graph_extraction(metadata.video_id, graph_extraction)
        if self.settings.enable_translation and analysis_translated and transcript_translated and self._target_language().strip().lower() == "ru":
            review_report = review_ru_artifacts(metadata.video_id, self.store, self.settings)
            hard_issues = [issue for issue in review_report.issues if issue.severity == "hard"]
            if review_report.status == "failed" or hard_issues:
                blocking_issues = hard_issues or review_report.issues
                message = "; ".join(f"{issue.code}: {issue.message}" for issue in blocking_issues)
                if not message:
                    message = f"status={review_report.status}"
                raise RuntimeError(f"RU artifact review hard failures: {message}")
            if review_report.status == "warning":
                warning_issues = [issue for issue in review_report.issues if issue.severity == "warning"] or review_report.issues
                review_warning = "RU artifact review warnings: " + "; ".join(
                    f"{issue.code}: {issue.message}" for issue in warning_issues
                )
                warning = f"{warning}; {review_warning}" if warning else review_warning
                if self._quality_mode() == "strict":
                    warning = f"{warning}; Graph sync skipped because strict mode has review warnings"
                    return paths, warning
        payload = graph_payload_from_extraction(graph_extraction)
        if self._graph_enabled():
            try:
                graph_store = self.graph_store or Neo4jGraphStore(
                    self.settings.neo4j_uri,
                    self.settings.neo4j_user,
                    self.settings.neo4j_password,
                )
                graph_store.upsert_payload(payload)
            except Exception as exc:
                if not self.settings.graph_best_effort:
                    raise
                graph_warning = f"Graph sync failed: {exc}"
                warning = f"{warning}; {graph_warning}" if warning else graph_warning
        return paths, warning

    def _translate_transcript_or_source(
        self,
        transcript: Transcript,
        warnings: list[str],
    ) -> tuple[Transcript, bool]:
        if not self.settings.enable_translation:
            return transcript, False
        target = self._target_language()
        if (transcript.language or "").strip().lower() == target.lower():
            return Transcript(
                video_id=transcript.video_id,
                source=transcript.source,
                language=target,
                segments=[
                    TranscriptSegment(segment.start, segment.duration, segment.text, segment.raw_text)
                    for segment in transcript.segments
                ],
            ), True
        cached = self.store.load_translated_transcript(transcript.video_id, target)
        if cached is not None and (cached.language or "").strip().lower() == target.lower():
            return cached, True
        try:
            provider = self.translation_provider or self._default_translation_provider()
            translated = provider.translate_transcript(transcript, target)
            if (translated.language or "").strip().lower() != target.lower():
                raise TranslationError(f"Transcript translation did not produce {target}")
            return translated, True
        except Exception:
            if not self.settings.translation_best_effort:
                raise
            warnings.append("Translation failed; graph and note generated from source language")
            return transcript, False

    def _translate_analysis_or_source(
        self,
        analysis: Analysis,
        transcript_translated: bool,
        warnings: list[str],
    ) -> tuple[Analysis, bool]:
        if not transcript_translated or not self.settings.enable_translation:
            return analysis, False
        target = self._target_language()
        if _analysis_matches_target_language(analysis, target):
            return analysis, True
        try:
            provider = self.translation_provider or self._default_translation_provider()
            translated_analysis = provider.translate_analysis(analysis, target)
            if not _analysis_matches_target_language(translated_analysis, target):
                return self._handle_analysis_translation_failure(analysis, warnings)
            return translated_analysis, True
        except Exception as exc:
            return self._handle_analysis_translation_failure(analysis, warnings, exc)

    def _handle_analysis_translation_failure(
        self,
        analysis: Analysis,
        warnings: list[str],
        exc: Exception | None = None,
    ) -> tuple[Analysis, bool]:
        message = self._analysis_translation_failure_message()
        if not self.settings.translation_best_effort:
            if exc is not None:
                raise TranslationError(message) from exc
            raise TranslationError(message)
        if exc is not None:
            warnings.append(f"{message}: {exc}; graph and note generated from analyzer language")
        else:
            warnings.append(f"{message}; graph and note generated from analyzer language")
        return analysis, False

    def _analysis_translation_failure_message(self) -> str:
        if self._target_language().strip().lower() == "ru":
            return "Russian analysis was not produced"
        return f"Target-language analysis was not produced for {self._target_language()}"

    def _source_generated_language(self, analysis: Analysis, normalized_transcript: Transcript) -> str | None:
        target = self._target_language().strip().lower()
        language = (analysis.language or normalized_transcript.language or "").strip()
        if language.lower() != target:
            return language or None
        source_language = (normalized_transcript.language or "").strip()
        if source_language and source_language.lower() != target:
            return source_language
        return "source"

    def _default_translation_provider(self):
        return LMStudioTranslationProvider(
            self.settings.lm_studio_base_url,
            self.settings.lm_studio_api_key,
            self.settings.translation_model,
            target_language=self._target_language(),
            max_chars=self.settings.translation_max_chars,
            timeout_seconds=self.settings.lm_studio_timeout_seconds,
        )

    def _target_language(self) -> str:
        return str(
            getattr(self.settings, "target_language", None)
            or getattr(self.settings, "translation_target_language", None)
            or "ru"
        )

    def _quality_mode(self) -> str:
        return normalize_quality_mode(getattr(self.settings, "quality_mode", "strict"))

    def _quality_note_provenance(self) -> dict[str, str]:
        mode = self._quality_mode()
        return {
            "quality_mode": mode,
            "quality_status": "draft" if mode == "draft" else "processing",
        }

    @staticmethod
    def _analysis_language(analysis_data: dict | None) -> str | None:
        if not analysis_data:
            return None
        language = analysis_data.get("language")
        if language is None:
            return None
        return str(language).strip().lower()

    def _target_artifacts_ready(self, video_id: str) -> bool:
        target = self._target_language()
        transcript = self.store.load_translated_transcript(video_id, target)
        return (
            transcript is not None
            and (transcript.language or "").strip().lower() == target.lower()
            and self._target_analysis_artifact_ready(video_id)
        )

    def _target_analysis_artifact_ready(self, video_id: str) -> bool:
        return self._analysis_data_matches_target_language(
            self.store.load_analysis(video_id, variant=self._target_language())
        )

    def _analysis_data_matches_target_language(self, analysis_data: dict | None) -> bool:
        if not analysis_data:
            return False
        try:
            analysis = parse_analysis_response(analysis_data)
        except Exception:
            return False
        return _analysis_matches_target_language(analysis, self._target_language())

    def _build_graph_extraction(
        self,
        metadata: VideoMetadata,
        transcript: Transcript,
        analysis: Analysis,
        previous_extraction: dict | None = None,
    ) -> tuple[dict, str | None, bool]:
        graph_extraction = build_deterministic_graph_extraction(metadata, transcript, analysis)
        graph_extraction.setdefault("generated_from", {})["language"] = analysis.language or transcript.language
        warning = None
        if not self.settings.enable_deep_graph_llm:
            return graph_extraction, warning, False

        try:
            from yt_sub_hub.graph_extraction import merge_deep_graph_chunks

            chunks = self.analyzer.extract_deep_graph_chunks(
                metadata,
                transcript,
                max_chunk_chars=self.settings.deep_graph_max_chunk_chars,
            )
            if not chunks:
                graph_extraction = self._mark_deep_graph_failure(graph_extraction, "LLM returned no deep graph chunks")
                raise RuntimeError("LLM returned no deep graph chunks")
            graph_extraction = merge_deep_graph_chunks(
                metadata,
                transcript,
                graph_extraction,
                chunks,
                min_confidence=self.settings.deep_graph_min_confidence,
                model=self.settings.qwen_model,
            )
            quality = graph_extraction.get("quality_report", {})
            chunks_total = int(quality.get("chunks_total") or 0)
            chunks_failed = int(quality.get("chunks_failed") or 0)
            if chunks_total > 0 and chunks_failed == chunks_total:
                graph_extraction = self._mark_deep_graph_failure(
                    graph_extraction,
                    f"LLM deep graph failed for all chunks: {chunks_failed}/{chunks_total} chunks failed",
                )
                raise RuntimeError(f"LLM deep graph failed for all chunks: {chunks_failed}/{chunks_total} chunks failed")
            if not self._has_llm_deep_graph(graph_extraction):
                graph_extraction = self._mark_deep_graph_failure(
                    graph_extraction,
                    "LLM deep graph produced no accepted llm_deep_graph relationships",
                )
                raise RuntimeError("LLM deep graph produced no accepted relationships")
            if chunks_failed > 0:
                warning = f"Deep graph partial failure: {chunks_failed}/{chunks_total} chunks failed"
        except Exception as exc:
            if not self.settings.deep_graph_best_effort:
                raise
            warning = f"Deep graph extraction failed: {exc}"
            if self._has_llm_deep_graph(previous_extraction):
                preserved = dict(previous_extraction or {})
                quality = preserved.setdefault("quality_report", {})
                quality["deep_graph_error"] = str(exc)
                return preserved, warning, True
            graph_extraction = self._mark_deep_graph_failure(graph_extraction, str(exc))
        return graph_extraction, warning, False

    @staticmethod
    def _has_llm_deep_graph(extraction: dict | None) -> bool:
        if not extraction or not extraction.get("chunks"):
            return False
        return any(
            rel.get("audit_status", "accepted") == "accepted"
            and (rel.get("extraction_method") == "llm_deep_graph" or rel.get("reliability") == "llm_extracted")
            for rel in extraction.get("relationships", [])
        )

    @staticmethod
    def _mark_deep_graph_failure(extraction: dict, error: str) -> dict:
        quality = extraction.setdefault("quality_report", {})
        chunks = extraction.get("chunks", [])
        failed_chunks = [chunk for chunk in chunks if chunk.get("rejected")]
        quality["deep_graph_error"] = error
        quality["chunks_total"] = len(chunks)
        quality["chunks_succeeded"] = len(chunks) - len(failed_chunks)
        quality["chunks_failed"] = len(failed_chunks)
        quality["relationships_accepted"] = sum(
            1
            for rel in extraction.get("relationships", [])
            if rel.get("audit_status", "accepted") == "accepted"
            and (rel.get("extraction_method") == "llm_deep_graph" or rel.get("reliability") == "llm_extracted")
        )
        return extraction

    def _graph_enabled(self) -> bool:
        return self.graph_store is not None or (self.enable_default_graph and self.settings.enable_graph)

    def rebuild_graph(
        self,
        graph_store: Neo4jGraphStore | None = None,
        video_id: str | None = None,
        dry_run: bool = False,
        clear: bool = False,
        min_confidence: float = 0.0,
    ) -> int:
        if not dry_run and graph_store is None:
            raise ValueError("graph_store is required unless dry_run=True")
        if clear and not dry_run and hasattr(graph_store, "clear"):
            graph_store.clear()
        count = 0
        for metadata_data, transcript_data, analysis_data in self.store.load_graph_source_files(
            prefer_translation=self.settings.enable_translation,
            target_language=self._target_language(),
            video_id=video_id,
        ):
            if video_id is not None and metadata_data.get("video_id") != video_id:
                continue
            metadata = VideoMetadata(**metadata_data)
            analysis = parse_analysis_response(analysis_data)
            saved_extraction = self.store.load_graph_extraction(metadata.video_id)
            transcript = Transcript(
                video_id=transcript_data["video_id"],
                source=transcript_data["source"],
                language=transcript_data.get("language"),
                segments=[TranscriptSegment(**segment) for segment in transcript_data.get("segments", [])],
            )
            if (
                self.settings.enable_translation
                and (transcript.language or "").strip().lower() == self._target_language().lower()
                and not _analysis_matches_target_language(analysis, self._target_language())
            ):
                raise RuntimeError(
                    f"analysis.{self._target_language()}.json does not match {self._target_language()} language gate"
                )
            if self._saved_extraction_stale_for_translation(saved_extraction, transcript, analysis) or saved_extraction is None or (
                self.settings.enable_deep_graph_llm and not self._has_llm_deep_graph(saved_extraction)
            ):
                saved_extraction, _, preserved_previous_extraction = self._build_graph_extraction(
                    metadata,
                    transcript,
                    analysis,
                    previous_extraction=saved_extraction,
                )
                if (
                    self.settings.enable_translation
                    and transcript.language == self._target_language()
                    and _analysis_matches_target_language(analysis, self._target_language())
                    and not preserved_previous_extraction
                ):
                    generated_from = saved_extraction.setdefault("generated_from", {})
                    generated_from["transcript"] = f"transcript.{self._target_language()}.json"
                    generated_from["analysis"] = f"analysis.{self._target_language()}.json"
                    generated_from["language"] = self._target_language()
            visual_summary = self.store.load_visual_summary(metadata.video_id)
            if visual_summary is not None:
                from yt_sub_hub.graph_extraction import merge_visual_summary_into_extraction

                saved_extraction = merge_visual_summary_into_extraction(saved_extraction, visual_summary)
            self.store.save_graph_extraction(metadata.video_id, saved_extraction)
            if not dry_run:
                if video_id is not None:
                    graph_store.delete_video_relationships(metadata.video_id)
                graph_store.upsert_payload(graph_payload_from_extraction(saved_extraction, min_confidence=min_confidence))
            count += 1
        return count

    def _saved_extraction_stale_for_translation(
        self,
        extraction: dict | None,
        transcript: Transcript,
        analysis: Analysis,
    ) -> bool:
        if not self.settings.enable_translation or extraction is None:
            return False
        if transcript.language != self._target_language() or not _analysis_matches_target_language(
            analysis, self._target_language()
        ):
            return False
        generated_from = extraction.get("generated_from", {})
        return (
            generated_from.get("transcript") != f"transcript.{self._target_language()}.json"
            or generated_from.get("analysis") != f"analysis.{self._target_language()}.json"
        )
