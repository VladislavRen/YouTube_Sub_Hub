from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from yt_sub_hub.artifact_review import canonical_hash, load_ru_artifact_review, text_hash, transcript_hash
from yt_sub_hub.language_quality import contains_cyrillic, validate_ru_analysis, validate_ru_markdown_note
from yt_sub_hub.library import LibraryStore
from yt_sub_hub.llm import parse_analysis_response
from yt_sub_hub.pipeline import _analysis_matches_target_language


@dataclass
class QualityGateResult:
    video_id: str
    status: str
    hard_failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_quality_gate(store: LibraryStore, video_id: str, *, target_language: str = "ru") -> QualityGateResult:
    target_language = target_language.strip().lower()
    hard_failures: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {}

    transcript = store.load_translated_transcript(video_id, target_language)
    if transcript is None:
        hard_failures.append(f"transcript.{target_language}.json must exist")
    elif (transcript.language or "").strip().lower() != target_language:
        hard_failures.append(f"transcript.{target_language}.json language must be {target_language}")
    elif target_language == "ru" and not _contains_cyrillic(transcript.full_text()):
        hard_failures.append(f"transcript.{target_language}.json must contain Cyrillic text")

    analysis = store.load_analysis(video_id, variant=target_language)
    if analysis is None:
        hard_failures.append(f"analysis.{target_language}.json must exist")
    elif str(analysis.get("language") or "").strip().lower() != target_language:
        hard_failures.append(f"analysis.{target_language}.json language must be {target_language}")
    else:
        try:
            parsed_analysis = parse_analysis_response(analysis)
        except Exception:
            hard_failures.append(f"analysis.{target_language}.json must be valid analysis JSON")
        else:
            if target_language == "ru":
                analysis_issues = validate_ru_analysis(parsed_analysis)
                if analysis_issues:
                    hard_failures.extend(f"analysis.{target_language}.json {issue}" for issue in analysis_issues)
            elif not _analysis_matches_target_language(parsed_analysis, target_language):
                hard_failures.append(f"analysis.{target_language}.json must pass {target_language} language gate")

    note_path = store.note_path(video_id)
    if not note_path.exists() or not note_path.read_text(encoding="utf-8").strip():
        hard_failures.append("note.md must exist and be non-empty")
    else:
        note_text = note_path.read_text(encoding="utf-8")
        if target_language == "ru":
            if not _contains_cyrillic(note_text):
                hard_failures.append("note.md must contain Cyrillic text")
            hard_failures.extend(f"note.md {issue}" for issue in validate_ru_markdown_note(note_text))

    extraction = store.load_graph_extraction(video_id)
    if extraction is None:
        hard_failures.append("graph_extraction.json must exist")
    else:
        generated_from = extraction.get("generated_from")
        if not isinstance(generated_from, dict):
            generated_from = {}
        generated_language = str(generated_from.get("language") or "").strip().lower()
        if generated_language != target_language:
            hard_failures.append(f"graph_extraction.generated_from.language must be {target_language}")
        expected_transcript = f"transcript.{target_language}.json"
        if generated_from.get("transcript") != expected_transcript:
            hard_failures.append(f"graph_extraction.generated_from.transcript must be {expected_transcript}")
        expected_analysis = f"analysis.{target_language}.json"
        if generated_from.get("analysis") != expected_analysis:
            hard_failures.append(f"graph_extraction.generated_from.analysis must be {expected_analysis}")
        analysis_hash = generated_from.get("analysis_hash")
        if analysis_hash is not None and analysis is not None and analysis_hash != _analysis_data_hash(analysis):
            hard_failures.append(f"graph_extraction.generated_from.analysis_hash must match analysis.{target_language}.json")
        _check_graph_shape(extraction, hard_failures, warnings, details)

    if target_language == "ru":
        _check_ru_artifact_review_report(
            store,
            video_id,
            transcript,
            analysis,
            note_path.read_text(encoding="utf-8") if note_path.exists() else None,
            extraction,
            hard_failures,
            warnings,
            details,
        )

    status = "failed" if hard_failures else ("completed_with_warning" if warnings else "completed")
    return QualityGateResult(
        video_id=video_id,
        status=status,
        hard_failures=hard_failures,
        warnings=warnings,
        details=details,
    )


def _check_ru_artifact_review_report(
    store: LibraryStore,
    video_id: str,
    transcript: Any | None,
    analysis: dict[str, Any] | None,
    note_text: str | None,
    extraction: dict[str, Any] | None,
    hard_failures: list[str],
    warnings: list[str],
    details: dict[str, Any],
) -> None:
    report = load_ru_artifact_review(store, video_id)
    if report is None:
        hard_failures.append("review/ru_artifact_review.json must exist")
        return
    details["ru_artifact_review"] = report
    status = report.get("status")
    if status == "failed":
        hard_failures.append("ru_artifact_review status must not be failed")
    elif status not in {"passed", "warning"}:
        hard_failures.append("ru_artifact_review status must be passed or warning")
    hashes = report.get("hashes") if isinstance(report.get("hashes"), dict) else {}
    if transcript is not None and hashes.get("transcript_ru_hash") != transcript_hash(transcript):
        hard_failures.append("ru_artifact_review transcript_ru_hash must match transcript.ru.json")
    if analysis is not None and hashes.get("analysis_ru_hash") != _analysis_data_hash(analysis):
        hard_failures.append("ru_artifact_review analysis_ru_hash must match analysis.ru.json")
    if note_text is not None and hashes.get("note_hash") != text_hash(note_text):
        hard_failures.append("ru_artifact_review note_hash must match note.md")
    if extraction is not None and hashes.get("graph_hash") != canonical_hash(extraction):
        hard_failures.append("ru_artifact_review graph_hash must match graph_extraction.json")
    for issue in report.get("issues") or []:
        if not isinstance(issue, dict):
            continue
        message = str(issue.get("message") or issue.get("code") or "review issue")
        if issue.get("severity") == "hard":
            hard_failures.append(f"ru_artifact_review hard failure: {message}")
        elif issue.get("severity") == "warning":
            warnings.append(f"ru_artifact_review warning: {message}")


def _check_graph_shape(
    extraction: dict[str, Any],
    hard_failures: list[str],
    warnings: list[str],
    details: dict[str, Any],
) -> None:
    nodes = extraction.get("nodes", [])
    relationships = extraction.get("relationships", [])
    labels = [node.get("label") for node in nodes if isinstance(node, dict)]
    details["node_count"] = len(nodes)
    details["relationship_count"] = len(relationships)
    details["labels"] = sorted({str(label) for label in labels if label})

    if "Video" not in labels:
        hard_failures.append("graph must contain a Video node")
    if "SourceSegment" not in labels:
        warnings.append("graph should contain SourceSegment nodes")
    if not {"Concept", "Claim", "Topic"} & set(labels):
        warnings.append("graph should contain at least one Concept, Claim, or Topic node")
    for relationship in relationships:
        if not isinstance(relationship, dict):
            continue
        if not relationship.get("source_video_id") and not relationship.get("source_channel_id"):
            warnings.append("graph relationships should include source_video_id or source_channel_id provenance")
            break


def _contains_cyrillic(value: str) -> bool:
    return contains_cyrillic(value)


def _analysis_data_hash(analysis_data: dict[str, Any]) -> str:
    payload = json.dumps(analysis_data, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
