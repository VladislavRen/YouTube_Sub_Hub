from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from yt_sub_hub.artifact_review import analysis_data_hash, transcript_hash
from yt_sub_hub.config import normalize_quality_mode
from yt_sub_hub.language_quality import (
    contains_blocked_english_phrases,
    ru_word_ratio,
    validate_ru_analysis,
    validate_ru_markdown_note,
)
from yt_sub_hub.library import LibraryStore
from yt_sub_hub.llm import parse_analysis_response
from yt_sub_hub.models import Analysis


@dataclass
class ArtifactQualityScore:
    language_score: float
    coverage_score: float
    graph_score: float | None
    source_leakage_score: float
    completeness_score: float
    overall_score: float
    status: str
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_artifact_quality_score(
    store: LibraryStore,
    video_id: str,
    *,
    quality_mode: str,
    quality_gate: Any | None = None,
    ru_artifact_review: dict[str, Any] | None = None,
    stages: list[Any] | None = None,
) -> dict[str, Any]:
    mode = normalize_quality_mode(quality_mode)
    quality_gate_payload = _as_dict(quality_gate) or {"hard_failures": [], "warnings": [], "details": {}}
    review = ru_artifact_review or {}
    stages = stages or []
    reasons: list[str] = []

    analysis_data = _safe_load_analysis(store, video_id)
    note_text = _safe_read_text(store.artifact_paths(video_id).note)
    transcript = _safe_load_transcript(store, video_id)

    language_score, language_metrics, language_reasons, language_hard = _language_score(analysis_data, note_text)
    coverage_score, coverage_metrics, coverage_reasons = _coverage_score(review.get("coverage") if isinstance(review, dict) else None)
    graph_score, graph_metrics, graph_reasons = _graph_score(
        store,
        video_id,
        mode=mode,
        analysis_data=analysis_data,
        transcript=transcript,
        stages=stages,
    )
    source_leakage_score, leakage_metrics, leakage_reasons, leakage_hard = _source_leakage_score(
        note_text,
        language_metrics,
        review,
        quality_gate_payload,
    )
    completeness_score, completeness_metrics, completeness_reasons, missing_required = _completeness_score(
        store,
        video_id,
        mode=mode,
        analysis_data=analysis_data,
        transcript=transcript,
    )

    reasons.extend(language_reasons)
    reasons.extend(coverage_reasons)
    reasons.extend(graph_reasons)
    reasons.extend(leakage_reasons)
    reasons.extend(completeness_reasons)

    hard_failures = [str(item) for item in quality_gate_payload.get("hard_failures") or []]
    quality_warnings = [str(item) for item in quality_gate_payload.get("warnings") or []]
    review_hard, review_warnings = _review_issue_messages(review)
    if hard_failures:
        reasons.extend(f"quality_gate hard failure: {message}" for message in hard_failures)
    if review_hard:
        reasons.extend(f"ru_artifact_review hard failure: {message}" for message in review_hard)
    if quality_warnings:
        reasons.extend(f"quality_gate warning: {message}" for message in quality_warnings)
    if review_warnings:
        reasons.extend(f"ru_artifact_review warning: {message}" for message in review_warnings)
    if review.get("status") == "failed":
        reasons.append("ru_artifact_review failed")

    weights = _weights_for_mode(mode)
    component_scores = {
        "language": language_score,
        "coverage": coverage_score,
        "source_leakage": source_leakage_score,
        "completeness": completeness_score,
    }
    if mode != "draft":
        component_scores["graph"] = graph_score if graph_score is not None else 0.0
    overall_score = _round_score(sum(component_scores[name] * weight for name, weight in weights.items()))

    failed = (
        bool(hard_failures)
        or review.get("status") == "failed"
        or bool(review_hard)
        or missing_required
        or language_hard
        or leakage_hard
        or overall_score < 0.60
    )
    warning = (
        mode == "draft"
        or 0.60 <= overall_score < 0.85
        or bool(quality_warnings)
        or bool(review_warnings)
        or bool(graph_reasons)
    )
    if failed:
        status = "failed"
    elif warning:
        status = "warning"
    else:
        status = "ready"

    metrics = {
        "weights": weights,
        "language": language_metrics,
        "coverage": coverage_metrics,
        "graph": graph_metrics,
        "source_leakage": leakage_metrics,
        "completeness": completeness_metrics,
    }
    return ArtifactQualityScore(
        language_score=_round_score(language_score),
        coverage_score=_round_score(coverage_score),
        graph_score=None if graph_score is None else _round_score(graph_score),
        source_leakage_score=_round_score(source_leakage_score),
        completeness_score=_round_score(completeness_score),
        overall_score=overall_score,
        status=status,
        reasons=_dedupe(reasons),
        metrics=metrics,
    ).as_dict()


def _language_score(
    analysis_data: dict[str, Any] | None,
    note_text: str | None,
) -> tuple[float, dict[str, Any], list[str], bool]:
    reasons: list[str] = []
    metrics: dict[str, Any] = {
        "analysis_ru_word_ratio": 0.0,
        "note_ru_word_ratio": 0.0,
        "analysis_validator_issue_count": 0,
        "note_validator_issue_count": 0,
        "blocked_phrases": [],
    }
    hard = False
    analysis: Analysis | None = None
    analysis_issues: list[str] = []
    note_issues: list[str] = []

    if not analysis_data:
        reasons.append("analysis.ru.json is missing or empty")
        hard = True
    else:
        try:
            analysis = parse_analysis_response(analysis_data)
        except Exception as exc:
            reasons.append(f"analysis.ru.json is not valid analysis JSON: {exc}")
            hard = True
        else:
            analysis_issues = validate_ru_analysis(analysis)
            metrics["analysis_validator_issue_count"] = len(analysis_issues)
            metrics["analysis_ru_word_ratio"] = _round_score(ru_word_ratio(_analysis_text(analysis)))
            if analysis_issues:
                hard = True
                reasons.extend(f"analysis.ru.json {issue}" for issue in analysis_issues)

    if note_text is None or not note_text.strip():
        reasons.append("note.md is missing or empty")
        hard = True
    else:
        note_issues = validate_ru_markdown_note(note_text)
        metrics["note_validator_issue_count"] = len(note_issues)
        metrics["note_ru_word_ratio"] = _round_score(ru_word_ratio(_markdown_body_text(note_text)))
        if note_issues:
            hard = True
            reasons.extend(f"note.md {issue}" for issue in note_issues)

    blocked = []
    if analysis_data:
        blocked.extend(contains_blocked_english_phrases(_text_values(analysis_data)))
    if note_text:
        blocked.extend(contains_blocked_english_phrases(note_text))
    metrics["blocked_phrases"] = sorted(set(blocked))

    if hard:
        return 0.0, metrics, reasons, True
    ratio_score = min(metrics["analysis_ru_word_ratio"], metrics["note_ru_word_ratio"]) / 0.85
    return _clamp(ratio_score), metrics, reasons, False


def _coverage_score(coverage: Any | None) -> tuple[float, dict[str, Any], list[str]]:
    if not isinstance(coverage, dict) or not coverage:
        return 0.0, {}, ["coverage metrics are missing"]
    source = coverage.get("coverage_source")
    if source == "section_timestamps":
        ratio = _float_or_none(coverage.get("timestamp_span_ratio"))
        target = 0.50
    else:
        ratio = _float_or_none(coverage.get("covered_segment_ratio"))
        target = 0.75 if source == "evidence_indexes" else 0.90 if source == "analysis_chunks" else None
    metrics = dict(coverage)
    if ratio is None or target is None:
        return 0.0, metrics, ["coverage ratio is missing"]
    score = min(1.0, ratio / target)
    stale = _int_or_zero(coverage.get("chunk_artifacts_stale"))
    total = _int_or_zero(coverage.get("chunk_artifacts_total"))
    reasons: list[str] = []
    if stale > 0:
        denominator = max(1, total)
        penalty = min(0.2, 0.05 * stale, stale / denominator * 0.25)
        score = max(0.0, score - penalty)
        if score < 0.85:
            reasons.append(f"coverage has stale analysis chunks: {stale}")
    if score < 0.60:
        reasons.append(f"coverage score is low for {source or 'unknown'}")
    return _clamp(score), metrics, reasons


def _graph_score(
    store: LibraryStore,
    video_id: str,
    *,
    mode: str,
    analysis_data: dict[str, Any] | None,
    transcript: Any | None,
    stages: list[Any],
) -> tuple[float | None, dict[str, Any], list[str]]:
    neo4j_status = _stage_status(stages, "neo4j_sync")
    metrics: dict[str, Any] = {"neo4j_sync_stage_status": neo4j_status}
    if mode == "draft":
        return None, metrics, ["draft mode skips graph sync and production validation"]

    graph = store.load_graph_extraction(video_id)
    if graph is None:
        return 0.0, metrics, ["graph_extraction.json is missing"]

    nodes = graph.get("nodes") if isinstance(graph.get("nodes"), list) else []
    relationships = graph.get("relationships") if isinstance(graph.get("relationships"), list) else []
    labels = sorted({str(node.get("label")) for node in nodes if isinstance(node, dict) and node.get("label")})
    generated_from = graph.get("generated_from") if isinstance(graph.get("generated_from"), dict) else {}
    analysis_hash_match = None
    transcript_hash_match = None
    if analysis_data is not None and generated_from.get("analysis_hash") is not None:
        analysis_hash_match = generated_from.get("analysis_hash") == analysis_data_hash(analysis_data)
    if transcript is not None and generated_from.get("transcript_hash") is not None:
        transcript_hash_match = generated_from.get("transcript_hash") == transcript_hash(transcript)

    metrics.update(
        {
            "node_count": len(nodes),
            "relationship_count": len(relationships),
            "labels": labels,
            "generated_from": generated_from,
            "analysis_hash_matches": analysis_hash_match,
            "transcript_hash_matches": transcript_hash_match,
        }
    )

    checks = [
        (len(nodes) > 0, "graph has no nodes"),
        (len(relationships) > 0, "graph has no relationships"),
        ("Video" in labels, "graph is missing Video label"),
        ("SourceSegment" in labels, "graph is missing SourceSegment label"),
        (bool({"Concept", "Claim"} & set(labels)), "graph is missing Concept/Claim labels"),
        (str(generated_from.get("language") or "").strip().lower() == "ru", "graph language provenance is not ru"),
        (generated_from.get("analysis") == "analysis.ru.json", "graph analysis provenance is not analysis.ru.json"),
        (generated_from.get("transcript") == "transcript.ru.json", "graph transcript provenance is not transcript.ru.json"),
        (analysis_hash_match is not False, "graph analysis hash is stale"),
        (transcript_hash_match is not False, "graph transcript hash is stale"),
    ]
    passed = sum(1 for ok, _ in checks if ok)
    reasons = [reason for ok, reason in checks if not ok]
    score = passed / len(checks)
    if neo4j_status in {"failed", "skipped"}:
        score = min(score, 0.75)
        reasons.append(f"neo4j_sync stage is {neo4j_status}")
    elif neo4j_status in {"pending", "running"}:
        score = min(score, 0.85)
        reasons.append(f"neo4j_sync stage is {neo4j_status}")
    return _clamp(score), metrics, reasons


def _source_leakage_score(
    note_text: str | None,
    language_metrics: dict[str, Any],
    review: dict[str, Any],
    quality_gate: dict[str, Any],
) -> tuple[float, dict[str, Any], list[str], bool]:
    blocked = list(language_metrics.get("blocked_phrases") or [])
    if note_text:
        blocked.extend(contains_blocked_english_phrases(note_text))
    review_messages = [
        str(issue.get("message") or issue.get("code") or "")
        for issue in review.get("issues") or []
        if isinstance(issue, dict)
    ]
    gate_messages = [str(item) for item in (quality_gate.get("hard_failures") or []) + (quality_gate.get("warnings") or [])]
    leakage_messages = [
        message
        for message in [*review_messages, *gate_messages]
        if any(term in message.lower() for term in ("leakage", "blocked english", "source-language", "source language"))
    ]
    metrics = {
        "blocked_phrases": sorted(set(blocked)),
        "blocked_phrase_count": len(set(blocked)),
        "leakage_issue_count": len(leakage_messages),
    }
    hard = any(
        isinstance(issue, dict)
        and issue.get("severity") == "hard"
        and any(term in str(issue.get("message") or issue.get("code") or "").lower() for term in ("leakage", "blocked english", "language"))
        for issue in review.get("issues") or []
    )
    hard = hard or any("blocked english" in message.lower() or "leakage" in message.lower() for message in gate_messages)
    if hard:
        return 0.0, metrics, leakage_messages or ["source-language leakage detected"], True
    penalty = min(1.0, 0.2 * metrics["blocked_phrase_count"] + 0.15 * metrics["leakage_issue_count"])
    reasons = leakage_messages[:]
    if blocked:
        reasons.append("blocked English phrases detected")
    return _clamp(1.0 - penalty), metrics, reasons, False


def _completeness_score(
    store: LibraryStore,
    video_id: str,
    *,
    mode: str,
    analysis_data: dict[str, Any] | None,
    transcript: Any | None,
) -> tuple[float, dict[str, Any], list[str], bool]:
    paths = store.artifact_paths(video_id)
    required: dict[str, Path] = {
        "transcript.ru.json": paths.transcript_ru,
        "analysis.ru.json": paths.analysis_ru,
        "note.md": paths.note,
        "review/ru_artifact_review.json": paths.ru_artifact_review,
    }
    if mode != "draft":
        required["graph_extraction.json"] = paths.graph_extraction

    presence = {name: path.exists() and path.is_file() and path.stat().st_size > 0 for name, path in required.items()}
    missing = [name for name, exists in presence.items() if not exists]
    reasons = [f"{name} is missing or empty" for name in missing]
    presence_score = sum(1 for exists in presence.values() if exists) / max(1, len(presence))

    sections = concepts = claims = 0
    if analysis_data:
        sections = len(analysis_data.get("sections") or [])
        concepts = len(analysis_data.get("concepts") or [])
        claims = len(analysis_data.get("claims") or [])
    segment_count = len(transcript.segments) if transcript is not None else 0
    count_targets = {"sections": 3, "concepts": 5, "claims": 5}
    if segment_count > 500:
        count_targets = {"sections": 5, "concepts": 10, "claims": 10}
    count_score = (
        min(1.0, sections / count_targets["sections"])
        + min(1.0, concepts / count_targets["concepts"])
        + min(1.0, claims / count_targets["claims"])
    ) / 3
    for name, value in (("sections", sections), ("concepts", concepts), ("claims", claims)):
        if value < count_targets[name]:
            reasons.append(f"analysis.ru.json has low {name} count: {value}/{count_targets[name]}")
    metrics = {
        "required_artifacts": presence,
        "missing_required_artifacts": missing,
        "analysis_sections": sections,
        "analysis_concepts": concepts,
        "analysis_claims": claims,
        "transcript_segments_total": segment_count,
        "count_targets": count_targets,
    }
    return _clamp(presence_score * 0.6 + count_score * 0.4), metrics, reasons, bool(missing)


def _weights_for_mode(mode: str) -> dict[str, float]:
    if mode == "draft":
        return {"language": 0.4, "coverage": 0.35, "completeness": 0.25}
    return {"language": 0.30, "coverage": 0.25, "graph": 0.20, "source_leakage": 0.15, "completeness": 0.10}


def _review_issue_messages(review: dict[str, Any]) -> tuple[list[str], list[str]]:
    hard: list[str] = []
    warnings: list[str] = []
    for issue in review.get("issues") or []:
        if not isinstance(issue, dict):
            continue
        message = str(issue.get("message") or issue.get("code") or "review issue")
        if issue.get("severity") == "hard":
            hard.append(message)
        elif issue.get("severity") == "warning":
            warnings.append(message)
    return hard, warnings


def _analysis_text(analysis: Analysis) -> str:
    parts: list[str] = [
        analysis.cleaned_title,
        analysis.abstract,
        *analysis.tags,
        *analysis.outline,
        *analysis.limitations,
        *analysis.open_questions,
        *analysis.related_topics,
    ]
    for section in analysis.sections:
        parts.extend([section.heading, section.summary, *section.bullets])
    for concept in analysis.concepts:
        parts.extend([concept.name, concept.definition])
    for claim in analysis.claims:
        parts.extend([claim.text, claim.evidence])
    return "\n".join(part for part in parts if part)


def _text_values(value: Any) -> str:
    parts: list[str] = []

    def walk(item: Any) -> None:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, list):
            for child in item:
                walk(child)
        elif isinstance(item, dict):
            for child in item.values():
                walk(child)

    walk(value)
    return "\n".join(parts)


def _markdown_body_text(markdown: str) -> str:
    lines = markdown.splitlines()
    body = markdown
    if lines and lines[0].strip() == "---":
        for index in range(1, len(lines)):
            if lines[index].strip() == "---":
                body = "\n".join(lines[index + 1 :])
                break
    body = re.sub(r"https?://\S+", " ", body)
    body = re.sub(r"`[^`]*`", " ", body)
    body = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", body)
    body = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", body)
    body = re.sub(r"[*_>#|:-]+", " ", body)
    return body


def _safe_load_analysis(store: LibraryStore, video_id: str) -> dict[str, Any] | None:
    try:
        return store.load_analysis(video_id, variant="ru")
    except Exception:
        return None


def _safe_load_transcript(store: LibraryStore, video_id: str) -> Any | None:
    try:
        return store.load_translated_transcript(video_id, "ru")
    except Exception:
        return None


def _safe_read_text(path: Path) -> str | None:
    try:
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _stage_status(stages: list[Any], stage_name: str) -> str | None:
    for stage in stages:
        name = stage.get("stage_name") if isinstance(stage, dict) else getattr(stage, "stage_name", None)
        if name != stage_name:
            continue
        return stage.get("status") if isinstance(stage, dict) else getattr(stage, "status", None)
    return None


def _as_dict(value: Any | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if isinstance(value, dict):
        return value
    return None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_zero(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _round_score(value: float) -> float:
    return round(_clamp(value), 3)


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
