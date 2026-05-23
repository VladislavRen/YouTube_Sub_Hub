from test_artifact_review import _settings, _write_review_video
from yt_sub_hub.artifact_review import review_ru_artifacts
from yt_sub_hub.library import LibraryStore
from yt_sub_hub.product_quality import build_product_quality_payload, evaluate_product_quality
from yt_sub_hub.quality import run_quality_gate


VIDEO_ID = "abc123def45"
NEO4J_COMPLETED = [{"stage_name": "neo4j_sync", "status": "completed"}]


def _ready_product_quality_fixture(tmp_path):
    store = LibraryStore(tmp_path)
    _write_review_video(store)
    review = review_ru_artifacts(VIDEO_ID, store, _settings(tmp_path)).to_dict()
    quality_gate = run_quality_gate(store, VIDEO_ID).as_dict()
    return store, quality_gate, review


def _degrade_review_coverage(review):
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
    return review


def test_product_quality_ignores_operational_cache_optional_stage_warnings_in_strict():
    status = evaluate_product_quality(
        "strict",
        quality_gate={"hard_failures": [], "warnings": []},
        ru_artifact_review={"status": "passed", "issues": []},
        stages=[
            {"stage_name": "transcript_source", "status": "completed_with_warning", "warning": "needs whisper"},
            {
                "stage_name": "analysis_translation",
                "status": "completed_with_warning",
                "warning": "analysis translation cache hit",
            },
            {"stage_name": "markdown_render", "status": "completed_with_warning", "warning": "note cache hit"},
            {
                "stage_name": "deterministic_graph",
                "status": "completed_with_warning",
                "warning": "graph cache hit",
            },
            {
                "stage_name": "analysis_reduce",
                "status": "skipped",
                "warning": "skipped by dependency branch",
            },
            {
                "stage_name": "visual_frame_analysis",
                "status": "skipped",
                "warning": "visual frame analysis is on-demand",
                "hard_failure": False,
            },
        ],
    )

    assert status["mode"] == "strict"
    assert status["status"] == "ready"
    assert status["production_ready"] is True
    assert status["graph_sync_allowed"] is True
    assert status["reasons"] == []


def test_strict_requires_clean_gate_and_review_for_production_ready():
    status = evaluate_product_quality(
        "strict",
        quality_gate={"hard_failures": [], "warnings": ["graph should contain SourceSegment nodes"]},
        ru_artifact_review={"status": "passed", "issues": []},
        stages=[],
    )

    assert status["mode"] == "strict"
    assert status["status"] == "warning"
    assert status["production_ready"] is False
    assert status["graph_sync_allowed"] is False
    assert "quality_gate warning: graph should contain SourceSegment nodes" in status["reasons"]


def test_product_quality_strict_still_blocks_real_quality_warning():
    status = evaluate_product_quality(
        "strict",
        quality_gate={"hard_failures": [], "warnings": ["source-language leakage detected"]},
        ru_artifact_review={"status": "passed", "issues": []},
        stages=[],
    )

    assert status["status"] == "warning"
    assert status["production_ready"] is False
    assert status["graph_sync_allowed"] is False
    assert "quality_gate warning: source-language leakage detected" in status["reasons"]


def test_best_effort_allows_warning_quality_without_hiding_reasons():
    status = evaluate_product_quality(
        "best_effort",
        quality_gate={"hard_failures": [], "warnings": ["small graph"]},
        ru_artifact_review={
            "status": "warning",
            "issues": [{"severity": "warning", "code": "chunk_artifacts_stale", "message": "chunk_artifacts_stale=2"}],
        },
        stages=[],
    )

    assert status["status"] == "warning"
    assert status["production_ready"] is True
    assert status["graph_sync_allowed"] is True
    assert "ru_artifact_review warning: chunk_artifacts_stale=2" in status["reasons"]


def test_product_quality_best_effort_allows_real_quality_warning():
    status = evaluate_product_quality(
        "best_effort",
        quality_gate={"hard_failures": [], "warnings": ["low RU ratio"]},
        ru_artifact_review={"status": "passed", "issues": []},
        stages=[],
    )

    assert status["status"] == "warning"
    assert status["production_ready"] is True
    assert status["graph_sync_allowed"] is True
    assert "quality_gate warning: low RU ratio" in status["reasons"]


def test_draft_is_never_production_ready_or_graph_syncable():
    status = evaluate_product_quality(
        "draft",
        quality_gate=None,
        ru_artifact_review=None,
        stages=[],
    )

    assert status == {
        "mode": "draft",
        "status": "draft",
        "production_ready": False,
        "graph_sync_allowed": False,
        "reasons": ["draft mode skips graph sync and production validation"],
    }


def test_product_quality_builder_adds_artifact_quality_score_without_changing_old_fields(tmp_path):
    store, quality_gate, review = _ready_product_quality_fixture(tmp_path)

    payload = build_product_quality_payload(
        store,
        VIDEO_ID,
        "strict",
        quality_gate,
        review,
        NEO4J_COMPLETED,
    )

    assert {key: payload[key] for key in ("mode", "status", "production_ready", "graph_sync_allowed", "reasons")} == {
        "mode": "strict",
        "status": "ready",
        "production_ready": True,
        "graph_sync_allowed": True,
        "reasons": [],
    }
    assert payload["artifact_quality_score"]["status"] == "ready"
    assert payload["artifact_quality_score"]["overall_score"] >= 0.85


def test_product_quality_builder_downgrades_ready_when_artifact_score_warns(tmp_path):
    store, quality_gate, review = _ready_product_quality_fixture(tmp_path)
    _degrade_review_coverage(review)

    payload = build_product_quality_payload(
        store,
        VIDEO_ID,
        "strict",
        quality_gate,
        review,
        NEO4J_COMPLETED,
    )

    assert payload["artifact_quality_score"]["status"] == "warning"
    assert payload["status"] == "warning"
    assert payload["production_ready"] is False
    assert payload["graph_sync_allowed"] is False
    assert any(reason.startswith("artifact quality score warning: overall_score=") for reason in payload["reasons"])


def test_product_quality_builder_allows_score_warning_in_best_effort(tmp_path):
    store, quality_gate, review = _ready_product_quality_fixture(tmp_path)
    _degrade_review_coverage(review)

    payload = build_product_quality_payload(
        store,
        VIDEO_ID,
        "best_effort",
        quality_gate,
        review,
        NEO4J_COMPLETED,
    )

    assert payload["artifact_quality_score"]["status"] == "warning"
    assert payload["status"] == "warning"
    assert payload["production_ready"] is True
    assert payload["graph_sync_allowed"] is True
    assert any(reason.startswith("artifact quality score warning: overall_score=") for reason in payload["reasons"])


def test_product_quality_builder_downgrades_ready_when_artifact_score_fails(tmp_path):
    store, quality_gate, review = _ready_product_quality_fixture(tmp_path)
    store.artifact_paths(VIDEO_ID).graph_extraction.unlink()

    payload = build_product_quality_payload(
        store,
        VIDEO_ID,
        "strict",
        quality_gate,
        review,
        NEO4J_COMPLETED,
    )

    assert payload["artifact_quality_score"]["status"] == "failed"
    assert payload["status"] == "failed"
    assert payload["production_ready"] is False
    assert payload["graph_sync_allowed"] is False
    assert any(reason.startswith("artifact quality score failed: overall_score=") for reason in payload["reasons"])


def test_product_quality_builder_preserves_base_failed_even_if_score_ready(tmp_path):
    store, quality_gate, review = _ready_product_quality_fixture(tmp_path)

    payload = build_product_quality_payload(
        store,
        VIDEO_ID,
        "strict",
        quality_gate,
        review,
        NEO4J_COMPLETED
        + [{"stage_name": "legacy_gate", "status": "failed", "hard_failure": True, "error": "manual hard failure"}],
    )

    assert payload["artifact_quality_score"]["status"] == "ready"
    assert payload["status"] == "failed"
    assert payload["production_ready"] is False
    assert payload["graph_sync_allowed"] is False
    assert "legacy_gate hard failure: manual hard failure" in payload["reasons"]


def test_product_quality_builder_preserves_base_failed_and_keeps_failed_score_reasons(tmp_path):
    store, quality_gate, review = _ready_product_quality_fixture(tmp_path)
    store.artifact_paths(VIDEO_ID).graph_extraction.unlink()

    payload = build_product_quality_payload(
        store,
        VIDEO_ID,
        "strict",
        quality_gate,
        review,
        NEO4J_COMPLETED
        + [{"stage_name": "legacy_gate", "status": "failed", "hard_failure": True, "error": "manual hard failure"}],
    )

    assert payload["artifact_quality_score"]["status"] == "failed"
    assert payload["status"] == "failed"
    assert payload["production_ready"] is False
    assert payload["graph_sync_allowed"] is False
    assert "legacy_gate hard failure: manual hard failure" in payload["reasons"]
    assert any(reason.startswith("artifact quality score failed: overall_score=") for reason in payload["reasons"])


def test_product_quality_builder_keeps_draft_not_syncable_when_artifact_score_fails(tmp_path):
    payload = build_product_quality_payload(
        LibraryStore(tmp_path),
        VIDEO_ID,
        "draft",
        None,
        None,
        [],
    )

    assert payload["mode"] == "draft"
    assert payload["status"] == "draft"
    assert payload["artifact_quality_score"]["status"] == "failed"
    assert payload["production_ready"] is False
    assert payload["graph_sync_allowed"] is False
    assert "draft mode skips graph sync and production validation" in payload["reasons"]
    assert any(reason.startswith("artifact quality score failed: overall_score=") for reason in payload["reasons"])


def test_product_quality_builder_keeps_valid_live_like_ready(tmp_path):
    store, quality_gate, review = _ready_product_quality_fixture(tmp_path)

    payload = build_product_quality_payload(
        store,
        VIDEO_ID,
        "strict",
        quality_gate,
        review,
        NEO4J_COMPLETED,
    )

    assert payload["status"] == "ready"
    assert payload["artifact_quality_score"]["status"] == "ready"
    assert payload["production_ready"] is True
    assert payload["graph_sync_allowed"] is True
    assert payload["reasons"] == []
