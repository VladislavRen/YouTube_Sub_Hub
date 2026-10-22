from __future__ import annotations

from pathlib import Path

from test_artifact_review import (
    _settings,
    _transcript_with_duration,
    _write_analysis_chunk,
    _write_review_video,
    _write_review_video_with_transcript,
)
from yt_sub_hub.artifact_quality_score import compute_artifact_quality_score
from yt_sub_hub.artifact_review import review_ru_artifacts, transcript_hash
from yt_sub_hub.library import LibraryStore
from yt_sub_hub.quality import run_quality_gate


def _reviewed_valid_video(tmp_path: Path) -> tuple[LibraryStore, dict, dict]:
    store = LibraryStore(tmp_path)
    _write_review_video(store)
    review = review_ru_artifacts("abc123def45", store, _settings(tmp_path)).to_dict()
    quality_gate = run_quality_gate(store, "abc123def45").as_dict()
    return store, quality_gate, review


def test_valid_ru_artifacts_score_ready_with_high_overall(tmp_path):
    store, quality_gate, review = _reviewed_valid_video(tmp_path)

    score = compute_artifact_quality_score(
        store,
        "abc123def45",
        quality_mode="strict",
        quality_gate=quality_gate,
        ru_artifact_review=review,
        stages=[{"stage_name": "neo4j_sync", "status": "completed"}],
    )

    assert score["status"] == "ready"
    assert score["overall_score"] >= 0.85
    assert score["language_score"] == 1.0
    assert score["source_leakage_score"] == 1.0
    assert score["graph_score"] > 0.9
    assert score["metrics"]["coverage"]["coverage_source"] in {"section_timestamps", "analysis_chunks", "evidence_indexes"}


def test_missing_analysis_ru_fails_with_low_completeness(tmp_path):
    store, _, review = _reviewed_valid_video(tmp_path)
    store.artifact_paths("abc123def45").analysis_ru.unlink()
    quality_gate = run_quality_gate(store, "abc123def45").as_dict()

    score = compute_artifact_quality_score(
        store,
        "abc123def45",
        quality_mode="strict",
        quality_gate=quality_gate,
        ru_artifact_review=review,
        stages=[],
    )

    assert score["status"] == "failed"
    assert score["completeness_score"] < 0.8
    assert any("analysis.ru.json" in reason for reason in score["reasons"])


def test_low_coverage_score_warns_without_hard_gate_failure(tmp_path):
    store, quality_gate, review = _reviewed_valid_video(tmp_path)
    review["coverage"] = {
        **review["coverage"],
        "coverage_source": "analysis_chunks",
        "covered_segment_ratio": 0.2,
        "covered_segments_count": 2,
        "transcript_segments_total": 10,
        "chunk_artifacts_total": 2,
        "chunk_artifacts_valid": 2,
        "chunk_artifacts_stale": 0,
    }

    score = compute_artifact_quality_score(
        store,
        "abc123def45",
        quality_mode="strict",
        quality_gate=quality_gate,
        ru_artifact_review=review,
        stages=[],
    )

    assert score["status"] == "warning"
    assert 0.0 < score["coverage_score"] < 0.5
    assert score["overall_score"] >= 0.60
    assert any("coverage" in reason for reason in score["reasons"])


def test_english_leakage_in_note_fails_source_leakage(tmp_path):
    store = LibraryStore(tmp_path)
    _write_review_video(store)
    note_path = store.note_path("abc123def45")
    note_path.write_text(note_path.read_text(encoding="utf-8") + "\n\nEvidence: The video explains this.\n", encoding="utf-8")
    review = review_ru_artifacts("abc123def45", store, _settings(tmp_path)).to_dict()
    quality_gate = run_quality_gate(store, "abc123def45").as_dict()

    score = compute_artifact_quality_score(
        store,
        "abc123def45",
        quality_mode="strict",
        quality_gate=quality_gate,
        ru_artifact_review=review,
        stages=[],
    )

    assert score["status"] == "failed"
    assert score["source_leakage_score"] == 0.0
    assert any("leakage" in reason.lower() or "blocked" in reason.lower() for reason in score["reasons"])


def test_missing_graph_in_strict_fails_graph_score(tmp_path):
    store, _, review = _reviewed_valid_video(tmp_path)
    store.artifact_paths("abc123def45").graph_extraction.unlink()
    quality_gate = run_quality_gate(store, "abc123def45").as_dict()

    score = compute_artifact_quality_score(
        store,
        "abc123def45",
        quality_mode="strict",
        quality_gate=quality_gate,
        ru_artifact_review=review,
        stages=[],
    )

    assert score["status"] == "failed"
    assert score["graph_score"] == 0.0
    assert any("graph_extraction.json" in reason for reason in score["reasons"])


def test_draft_mode_excludes_graph_and_explains_skipped_validation(tmp_path):
    store, _, review = _reviewed_valid_video(tmp_path)

    score = compute_artifact_quality_score(
        store,
        "abc123def45",
        quality_mode="draft",
        quality_gate=None,
        ru_artifact_review=review,
        stages=[],
    )

    assert score["status"] == "warning"
    assert score["graph_score"] is None
    assert score["metrics"]["weights"] == {"language": 0.4, "coverage": 0.35, "completeness": 0.25}
    assert any("draft mode skips graph sync" in reason for reason in score["reasons"])


def test_live_like_pl6_fixture_reports_ready_with_full_chunk_coverage(tmp_path):
    store = LibraryStore(tmp_path)
    transcript = _transcript_with_duration(1415, 3823.36)
    _write_review_video_with_transcript(store, transcript, source_count=40, ru_count=12)
    source_hash = transcript_hash(transcript)
    for index, (start, end) in enumerate([(0, 54), (55, 124), (125, 210), (211, 1414)]):
        _write_analysis_chunk(store, "abc123def45", index=index, total=4, start=start, end=end, source_hash=source_hash)
    review = review_ru_artifacts("abc123def45", store, _settings(tmp_path)).to_dict()
    quality_gate = run_quality_gate(store, "abc123def45").as_dict()

    score = compute_artifact_quality_score(
        store,
        "abc123def45",
        quality_mode="strict",
        quality_gate=quality_gate,
        ru_artifact_review=review,
        stages=[{"stage_name": "neo4j_sync", "status": "completed"}],
    )

    assert score["status"] == "ready"
    assert score["coverage_score"] == 1.0
    assert score["metrics"]["coverage"]["covered_segment_ratio"] == 1.0
    assert score["overall_score"] >= 0.85
    assert score["metrics"]["graph"]["neo4j_sync_stage_status"] == "completed"
