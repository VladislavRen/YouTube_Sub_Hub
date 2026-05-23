from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from yt_sub_hub.language_quality import (
    contains_blocked_english_phrases,
    validate_ru_analysis,
    validate_ru_markdown_note,
)
from yt_sub_hub.library import LibraryStore
from yt_sub_hub.llm import parse_analysis_response
from yt_sub_hub.models import Analysis, Transcript, slugify


REVIEW_VERSION = "ru-artifact-review-v1"
GENERIC_RU_CONCEPTS = {"Основная тема", "Контекст видео", "Ключевой эпизод"}


@dataclass(frozen=True)
class ReviewIssue:
    severity: str
    code: str
    artifact: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReviewReport:
    video_id: str
    target_language: str
    status: str
    reviewer: dict[str, str]
    checked_artifacts: dict[str, str]
    hashes: dict[str, str | None]
    coverage: dict[str, Any]
    issues: list[ReviewIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["issues"] = [issue.to_dict() for issue in self.issues]
        return data


class RuArtifactReviewer:
    def __init__(self, store: LibraryStore, settings: Any | None = None) -> None:
        self.store = store
        self.settings = settings

    def review(self, video_id: str) -> ReviewReport:
        target_language = str(getattr(self.settings, "target_language", "ru") or "ru").strip().lower()
        checked_artifacts = {
            "transcript": "transcript.ru.json",
            "analysis": "analysis.ru.json",
            "markdown": "note.md",
            "graph": "graph_extraction.json",
        }
        hashes: dict[str, str | None] = {
            "transcript_ru_hash": None,
            "analysis_ru_hash": None,
            "note_hash": None,
            "graph_hash": None,
        }
        coverage: dict[str, Any] = {
            "transcript_segments_total": None,
            "analysis_sections": None,
            "analysis_concepts": None,
            "analysis_claims": None,
            "coverage_source": None,
            "covered_segments_count": None,
            "covered_segment_ratio": None,
            "covered_segment_ranges": [],
            "chunk_artifacts_total": 0,
            "chunk_artifacts_valid": 0,
            "chunk_artifacts_stale": 0,
            "timestamp_span_seconds": None,
            "transcript_duration_seconds": None,
            "timestamp_span_ratio": None,
        }
        issues: list[ReviewIssue] = []
        paths = self.store.artifact_paths(video_id)

        transcript = self._load_required_transcript(video_id, issues)
        analysis_data = self._load_required_analysis(video_id, issues)
        note_text = self._load_required_text(paths.note, "note.md", issues)
        graph = self._load_required_graph(video_id, issues)
        source_analysis_data = self.store.load_analysis(video_id, variant="source")

        if transcript is not None:
            hashes["transcript_ru_hash"] = transcript_hash(transcript)
            coverage["transcript_segments_total"] = len(transcript.segments)
        if analysis_data is not None:
            hashes["analysis_ru_hash"] = analysis_data_hash(analysis_data)
        if note_text is not None:
            hashes["note_hash"] = text_hash(note_text)
        if graph is not None:
            hashes["graph_hash"] = canonical_hash(graph)

        analysis = self._check_analysis(video_id, analysis_data, source_analysis_data, transcript, coverage, issues)
        if note_text is not None:
            self._check_markdown(note_text, analysis, source_analysis_data, hashes, issues)
        if graph is not None:
            self._check_graph(graph, analysis, hashes, issues)

        hard_issues = [issue for issue in issues if issue.severity == "hard"]
        status = "failed" if hard_issues else ("warning" if issues else "passed")
        report = ReviewReport(
            video_id=video_id,
            target_language=target_language,
            status=status,
            reviewer={"kind": "deterministic", "version": REVIEW_VERSION},
            checked_artifacts=checked_artifacts,
            hashes=hashes,
            coverage=coverage,
            issues=issues,
        )
        self.save_report(video_id, report)
        return report

    def save_report(self, video_id: str, report: ReviewReport) -> Path:
        path = self.store.artifact_paths(video_id).ru_artifact_review
        path.parent.mkdir(parents=True, exist_ok=True)
        LibraryStore._write_json(path, report.to_dict())
        return path

    def _load_required_transcript(self, video_id: str, issues: list[ReviewIssue]) -> Transcript | None:
        transcript = self.store.load_translated_transcript(video_id, "ru")
        if transcript is None:
            issues.append(_hard("missing_transcript_ru", "transcript.ru.json", "transcript.ru.json must exist"))
            return None
        if (transcript.language or "").strip().lower() != "ru":
            issues.append(_hard("invalid_transcript_language", "transcript.ru.json", "transcript.ru.json language must be ru"))
        return transcript

    def _load_required_analysis(self, video_id: str, issues: list[ReviewIssue]) -> dict[str, Any] | None:
        analysis_data = self.store.load_analysis(video_id, variant="ru")
        if analysis_data is None:
            issues.append(_hard("missing_analysis_ru", "analysis.ru.json", "analysis.ru.json must exist"))
        return analysis_data

    def _load_required_text(self, path: Path, artifact: str, issues: list[ReviewIssue]) -> str | None:
        if not path.exists() or not path.read_text(encoding="utf-8").strip():
            issues.append(_hard(f"missing_{artifact.replace('.', '_')}", artifact, f"{artifact} must exist and be non-empty"))
            return None
        return path.read_text(encoding="utf-8")

    def _load_required_graph(self, video_id: str, issues: list[ReviewIssue]) -> dict[str, Any] | None:
        graph = self.store.load_graph_extraction(video_id)
        if graph is None:
            issues.append(_hard("missing_graph", "graph_extraction.json", "graph_extraction.json must exist"))
        return graph

    def _check_analysis(
        self,
        video_id: str,
        analysis_data: dict[str, Any] | None,
        source_analysis_data: dict[str, Any] | None,
        transcript: Transcript | None,
        coverage: dict[str, Any],
        issues: list[ReviewIssue],
    ) -> Analysis | None:
        if analysis_data is None:
            return None
        try:
            analysis = parse_analysis_response(analysis_data)
        except Exception as exc:
            issues.append(_hard("invalid_analysis_json", "analysis.ru.json", f"analysis.ru.json must be valid analysis JSON: {exc}"))
            return None
        for issue in validate_ru_analysis(analysis):
            issues.append(_hard("ru_analysis_language_quality", "analysis.ru.json", f"analysis.ru.json {issue}"))

        coverage["analysis_sections"] = len(analysis.sections)
        coverage["analysis_concepts"] = len(analysis.concepts)
        coverage["analysis_claims"] = len(analysis.claims)
        source_counts = _analysis_counts(source_analysis_data)
        source_rich = source_counts["concepts"] >= 10 and source_counts["claims"] >= 10
        if source_rich:
            self._check_count_ratio("sections", len(analysis.sections), source_counts["sections"], 3, issues)
            self._check_count_ratio("concepts", len(analysis.concepts), source_counts["concepts"], 5, issues)
            self._check_count_ratio("claims", len(analysis.claims), source_counts["claims"], 5, issues)
            expected_source_hash = analysis_data_hash(source_analysis_data) if source_analysis_data is not None else None
            if expected_source_hash and analysis_data.get("source_analysis_hash") != expected_source_hash:
                issues.append(
                    _hard(
                        "stale_analysis_ru",
                        "analysis.ru.json",
                        "analysis.ru.json source_analysis_hash must match current analysis.source.json",
                    )
                )

        if transcript is not None:
            total = len(transcript.segments)
            coverage["transcript_duration_seconds"] = _transcript_duration_seconds(transcript)
            self._check_transcript_coverage(video_id, analysis_data, analysis, transcript, coverage, issues)
            if total > 500 and len(analysis.sections) < 5:
                issues.append(_hard("low_transcript_coverage_sections", "analysis.ru.json", "long transcript analysis must contain at least 5 sections"))
            for label, count in (("concepts", len(analysis.concepts)), ("claims", len(analysis.claims))):
                if total > 500 and count < 10:
                    severity = "hard" if source_rich else "warning"
                    issues.append(
                        ReviewIssue(
                            severity=severity,
                            code=f"low_transcript_coverage_{label}",
                            artifact="analysis.ru.json",
                            message=f"long transcript analysis should contain at least 10 {label}",
                        )
                    )
            expected_transcript_hash = transcript_hash(transcript)
            if analysis_data.get("source_transcript_hash") not in {None, expected_transcript_hash}:
                issues.append(
                    _hard(
                        "stale_analysis_ru_transcript",
                        "analysis.ru.json",
                        "analysis.ru.json source_transcript_hash must match current transcript.ru.json",
                    )
                )
        return analysis

    def _check_transcript_coverage(
        self,
        video_id: str,
        analysis_data: dict[str, Any],
        analysis: Analysis,
        transcript: Transcript,
        coverage: dict[str, Any],
        issues: list[ReviewIssue],
    ) -> None:
        total = len(transcript.segments)
        expected_transcript_hash = transcript_hash(transcript)

        evidence_ranges = _merge_segment_ranges(_collect_evidence_segment_ranges(analysis_data), total)
        if evidence_ranges:
            covered = _covered_count(evidence_ranges)
            ratio = covered / max(1, total)
            coverage.update(
                {
                    "coverage_source": "evidence_indexes",
                    "covered_segments_count": covered,
                    "covered_segment_ratio": ratio,
                    "covered_segment_ranges": [list(item) for item in evidence_ranges],
                }
            )
            if total > 500:
                if ratio < 0.55:
                    issues.append(_hard("low_transcript_coverage_evidence", "analysis.ru.json", "evidence segment references cover too little of the long transcript"))
                elif ratio < 0.75:
                    issues.append(_warning("low_transcript_coverage_evidence", "analysis.ru.json", "evidence segment references cover a limited part of the long transcript"))
            return

        chunk_ranges, chunk_total, chunk_valid, chunk_stale = self._load_analysis_chunk_ranges(video_id, expected_transcript_hash, analysis_data, total)
        coverage["chunk_artifacts_total"] = chunk_total
        coverage["chunk_artifacts_valid"] = chunk_valid
        coverage["chunk_artifacts_stale"] = chunk_stale
        if chunk_ranges:
            merged_chunks = _merge_segment_ranges(chunk_ranges, total)
            covered = _covered_count(merged_chunks)
            ratio = covered / max(1, total)
            coverage.update(
                {
                    "coverage_source": "analysis_chunks",
                    "covered_segments_count": covered,
                    "covered_segment_ratio": ratio,
                    "covered_segment_ranges": [list(item) for item in merged_chunks],
                }
            )
            if total > 500:
                if ratio < 0.80:
                    issues.append(_hard("low_transcript_coverage_chunks", "analysis/chunks", "analysis chunk artifacts cover too little of the long transcript"))
                elif ratio < 0.90:
                    issues.append(_warning("low_transcript_coverage_chunks", "analysis/chunks", "analysis chunk artifacts cover a limited part of the long transcript"))
            return

        timestamp_coverage = _timestamp_span_coverage(analysis, transcript)
        coverage.update(timestamp_coverage)
        if timestamp_coverage["coverage_source"] == "section_timestamps":
            ratio = timestamp_coverage["timestamp_span_ratio"]
            if total > 500 and ratio is not None:
                if ratio < 0.25:
                    issues.append(_hard("low_transcript_coverage_timestamps", "analysis.ru.json", "section timestamps are concentrated in too small a part of the long transcript"))
                    if chunk_stale > 0:
                        issues.append(_hard("stale_analysis_chunk_coverage", "analysis/chunks", "analysis chunk artifacts are stale and cannot prove transcript coverage"))
                elif ratio < 0.50:
                    issues.append(_warning("low_transcript_coverage_timestamps", "analysis.ru.json", "section timestamps cover a limited span of the long transcript"))
            return

        if chunk_stale > 0:
            issues.append(_hard("stale_analysis_chunk_coverage", "analysis/chunks", "analysis chunk artifacts are stale and cannot prove transcript coverage"))
            return

        severity = "hard" if total > 500 else "warning"
        issues.append(
            ReviewIssue(
                severity=severity,
                code="missing_transcript_coverage_signal",
                artifact="analysis.ru.json",
                message="analysis.ru.json has no evidence, chunk, or timestamp coverage signal",
            )
        )

    def _load_analysis_chunk_ranges(
        self,
        video_id: str,
        transcript_hash_value: str,
        analysis_data: dict[str, Any],
        total_segments: int,
    ) -> tuple[list[tuple[int, int]], int, int, int]:
        chunk_dir = self.store.video_dir(video_id) / "analysis" / "chunks"
        if not chunk_dir.exists():
            return [], 0, 0, 0

        generated_from = analysis_data.get("generated_from") if isinstance(analysis_data.get("generated_from"), dict) else {}
        expected_source_transcript = str(generated_from.get("transcript") or "transcript.ru.json")
        expected_language = str(generated_from.get("target_language") or analysis_data.get("language") or "ru").strip().lower()
        ranges: list[tuple[int, int]] = []
        total = 0
        valid = 0
        stale = 0
        for path in sorted(chunk_dir.glob("*.json")):
            total += 1
            try:
                artifact = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                stale += 1
                continue
            if not isinstance(artifact, dict):
                stale += 1
                continue
            start = _int_or_none(artifact.get("source_segment_start_index"))
            end = _int_or_none(artifact.get("source_segment_end_index"))
            language = str(artifact.get("language") or "").strip().lower()
            source_transcript = str(artifact.get("source_transcript") or "")
            chunk_hash = artifact.get("source_transcript_hash")
            if (
                start is None
                or end is None
                or start < 0
                or end < start
                or start >= total_segments
                or end >= total_segments
                or chunk_hash != transcript_hash_value
                or source_transcript != expected_source_transcript
                or language != expected_language
            ):
                stale += 1
                continue
            ranges.append((start, end))
            valid += 1
        return ranges, total, valid, stale

    def _check_count_ratio(
        self,
        label: str,
        actual: int,
        source: int,
        floor_value: int,
        issues: list[ReviewIssue],
    ) -> None:
        required = min(floor_value, math.floor(source * 0.5))
        if actual < required:
            issues.append(
                _hard(
                    f"analysis_ru_sparse_{label}",
                    "analysis.ru.json",
                    f"analysis.ru.json has {actual} {label}, below required {required} from source {source}",
                )
            )

    def _check_markdown(
        self,
        markdown: str,
        analysis: Analysis | None,
        source_analysis_data: dict[str, Any] | None,
        hashes: dict[str, str | None],
        issues: list[ReviewIssue],
    ) -> None:
        for issue in validate_ru_markdown_note(markdown):
            issues.append(_hard("markdown_language_quality", "note.md", f"note.md {issue}"))
        for phrase in ("Evidence:", "Claim:"):
            if phrase in markdown:
                issues.append(_hard("markdown_source_language_leakage", "note.md", f"note.md contains blocked English phrase: {phrase}"))

        frontmatter = parse_frontmatter(markdown)
        if frontmatter.get("generated_from_analysis") != "analysis.ru.json":
            issues.append(_hard("stale_markdown_generated_from", "note.md", "note.md must be generated from analysis.ru.json"))
        if hashes["analysis_ru_hash"] and frontmatter.get("analysis_hash") != hashes["analysis_ru_hash"]:
            issues.append(_hard("stale_markdown_analysis_hash", "note.md", "note.md was built from stale analysis hash"))
        if hashes["transcript_ru_hash"] and frontmatter.get("transcript_hash") != hashes["transcript_ru_hash"]:
            issues.append(_hard("stale_markdown_transcript_hash", "note.md", "note.md was built from stale transcript hash"))

        if analysis is None:
            return
        concept_names = [concept.name for concept in analysis.concepts if concept.name.strip()]
        meaningful = [name for name in concept_names if name not in GENERIC_RU_CONCEPTS]
        source_counts = _analysis_counts(source_analysis_data)
        if source_counts["concepts"] >= 10 and len(meaningful) < 5:
            issues.append(_hard("generic_markdown_concepts", "note.md", "note.md is based on generic concepts for a rich source analysis"))
        missing = _missing_terms(markdown, meaningful)
        required_missing_limit = 0 if len(meaningful) <= 5 else math.floor(len(meaningful) * 0.5)
        if meaningful and len(missing) > required_missing_limit:
            issues.append(
                _hard(
                    "markdown_missing_analysis_concepts",
                    "note.md",
                    "note.md does not include enough key concepts from analysis.ru.json",
                )
            )

    def _check_graph(
        self,
        graph: dict[str, Any],
        analysis: Analysis | None,
        hashes: dict[str, str | None],
        issues: list[ReviewIssue],
    ) -> None:
        generated_from = graph.get("generated_from") if isinstance(graph.get("generated_from"), dict) else {}
        if str(generated_from.get("language") or "").strip().lower() != "ru":
            issues.append(_hard("graph_language_not_ru", "graph_extraction.json", "graph_extraction.generated_from.language must be ru"))
        if generated_from.get("analysis") != "analysis.ru.json":
            issues.append(_hard("graph_analysis_source_not_ru", "graph_extraction.json", "graph_extraction.generated_from.analysis must be analysis.ru.json"))
        if generated_from.get("transcript") != "transcript.ru.json":
            issues.append(_hard("graph_transcript_source_not_ru", "graph_extraction.json", "graph_extraction.generated_from.transcript must be transcript.ru.json"))
        if hashes["analysis_ru_hash"] and generated_from.get("analysis_hash") != hashes["analysis_ru_hash"]:
            issues.append(_hard("stale_graph_analysis_hash", "graph_extraction.json", "graph_extraction.generated_from.analysis_hash must match analysis.ru.json"))
        if hashes["transcript_ru_hash"] and generated_from.get("transcript_hash") not in {None, hashes["transcript_ru_hash"]}:
            issues.append(_hard("stale_graph_transcript_hash", "graph_extraction.json", "graph_extraction.generated_from.transcript_hash must match transcript.ru.json"))

        if analysis is None:
            return
        analysis_concepts = [concept.name for concept in analysis.concepts if concept.name and concept.name not in GENERIC_RU_CONCEPTS]
        graph_concepts = [
            str(node.get("name") or "")
            for node in graph.get("nodes", [])
            if isinstance(node, dict) and node.get("label") == "Concept"
        ]
        missing = _missing_terms(" ".join(graph_concepts), analysis_concepts)
        allowed_missing = 0 if len(analysis_concepts) <= 5 else math.floor(len(analysis_concepts) * 0.5)
        if analysis_concepts and len(missing) > allowed_missing:
            issues.append(
                _hard(
                    "graph_missing_analysis_concepts",
                    "graph_extraction.json",
                    "graph Concept nodes do not cover enough concepts from analysis.ru.json",
                )
            )


def review_ru_artifacts(video_id: str, store: LibraryStore, settings: Any | None = None) -> ReviewReport:
    return RuArtifactReviewer(store, settings).review(video_id)


def load_ru_artifact_review(store: LibraryStore, video_id: str) -> dict[str, Any] | None:
    path = store.artifact_paths(video_id).ru_artifact_review
    if not path.exists():
        return None
    return LibraryStore._read_json(path)


def analysis_data_hash(analysis_data: dict[str, Any]) -> str:
    payload = json.dumps(analysis_data, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def transcript_hash(transcript: Transcript) -> str:
    payload = json.dumps(transcript.to_dict(), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_hash(data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_frontmatter(markdown: str) -> dict[str, str]:
    lines = markdown.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    result: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip().strip('"').strip("'")
    return result


def _analysis_counts(analysis_data: dict[str, Any] | None) -> dict[str, int]:
    if not analysis_data:
        return {"sections": 0, "concepts": 0, "claims": 0}
    return {
        "sections": len(analysis_data.get("sections") or []),
        "concepts": len(analysis_data.get("concepts") or []),
        "claims": len(analysis_data.get("claims") or []),
    }


def _collect_evidence_segment_ranges(analysis_data: dict[str, Any]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for start_key, end_key in (
                ("source_segment_start_index", "source_segment_end_index"),
                ("start_segment_index", "end_segment_index"),
            ):
                start = _int_or_none(value.get(start_key))
                end = _int_or_none(value.get(end_key))
                if start is not None and end is not None:
                    ranges.append((start, end))
            for key, item in value.items():
                if key in {
                    "evidence_segment_indexes",
                    "source_segment_indexes",
                    "segment_indexes",
                    "evidence_segments",
                    "source_segments",
                }:
                    ranges.extend(_collect_ranges_from_value(item))
                elif key == "segment_range":
                    ranges.extend(_collect_ranges_from_value(item, list_as_range=True))
                elif isinstance(item, dict | list):
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict | list):
                    walk(item)

    walk(analysis_data)
    return ranges


def _collect_ranges_from_value(value: Any, *, list_as_range: bool = False) -> list[tuple[int, int]]:
    direct_index = _int_or_none(value)
    if direct_index is not None:
        return [(direct_index, direct_index)]

    if isinstance(value, list):
        if list_as_range and len(value) == 2:
            start = _int_or_none(value[0])
            end = _int_or_none(value[1])
            if start is not None and end is not None:
                return [(start, end)]
        ranges: list[tuple[int, int]] = []
        for item in value:
            ranges.extend(_collect_ranges_from_value(item))
        return ranges

    if isinstance(value, dict):
        for start_key, end_key in (
            ("source_segment_start_index", "source_segment_end_index"),
            ("start_segment_index", "end_segment_index"),
            ("start", "end"),
        ):
            start = _int_or_none(value.get(start_key))
            end = _int_or_none(value.get(end_key))
            if start is not None and end is not None:
                return [(start, end)]
        for index_key in ("index", "segment_index", "source_segment_index"):
            index = _int_or_none(value.get(index_key))
            if index is not None:
                return [(index, index)]
        ranges: list[tuple[int, int]] = []
        for item in value.values():
            if isinstance(item, dict | list):
                ranges.extend(_collect_ranges_from_value(item))
        return ranges

    return []


def _merge_segment_ranges(ranges: list[tuple[int, int]], total: int) -> list[tuple[int, int]]:
    if total <= 0:
        return []
    normalized: list[tuple[int, int]] = []
    for start, end in ranges:
        if start < 0 or end < start:
            continue
        clamped_start = max(0, min(start, total - 1))
        clamped_end = max(0, min(end, total - 1))
        if clamped_end < clamped_start:
            continue
        normalized.append((clamped_start, clamped_end))
    if not normalized:
        return []
    normalized.sort()
    merged = [normalized[0]]
    for start, end in normalized[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _covered_count(ranges: list[tuple[int, int]]) -> int:
    return sum(end - start + 1 for start, end in ranges)


def _timestamp_span_coverage(analysis: Analysis, transcript: Transcript) -> dict[str, Any]:
    timestamps = [
        timestamp
        for timestamp in (_int_or_none(section.timestamp_seconds) for section in analysis.sections)
        if timestamp is not None and timestamp >= 0
    ]
    duration = _transcript_duration_seconds(transcript)
    if not timestamps or duration is None or duration <= 0:
        return {
            "coverage_source": None,
            "timestamp_span_seconds": None,
            "transcript_duration_seconds": duration,
            "timestamp_span_ratio": None,
        }
    span = max(timestamps) - min(timestamps)
    return {
        "coverage_source": "section_timestamps",
        "timestamp_span_seconds": span,
        "transcript_duration_seconds": duration,
        "timestamp_span_ratio": span / duration,
    }


def _transcript_duration_seconds(transcript: Transcript) -> float | None:
    if not transcript.segments:
        return None
    ends = [segment.start + segment.duration for segment in transcript.segments]
    return max(ends) if ends else None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _missing_terms(text: str, terms: list[str]) -> list[str]:
    normalized_text = _normalized_search_text(text)
    return [term for term in terms if slugify(term) not in normalized_text and _normalized_search_text(term) not in normalized_text]


def _normalized_search_text(text: str) -> str:
    return " ".join([slugify(text), " ".join(str(text or "").lower().split())])


def _hard(code: str, artifact: str, message: str) -> ReviewIssue:
    return ReviewIssue(severity="hard", code=code, artifact=artifact, message=message)


def _warning(code: str, artifact: str, message: str) -> ReviewIssue:
    return ReviewIssue(severity="warning", code=code, artifact=artifact, message=message)
