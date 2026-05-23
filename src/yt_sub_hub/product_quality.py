from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from yt_sub_hub.config import normalize_quality_mode


_QUALITY_WARNING_STAGES = {"analysis_ru_review", "quality_gate", "ru_artifact_review"}
_QUALITY_WARNING_TERMS = {
    "graph mismatch",
    "low ru ratio",
    "missing ru",
    "missing ru artifact",
    "provenance",
    "review",
    "reviewer",
    "source-language",
    "source language",
    "stale",
}


@dataclass
class ProductQualityStatus:
    mode: str
    status: str
    production_ready: bool
    graph_sync_allowed: bool
    reasons: list[str] = field(default_factory=list)
    artifact_quality_score: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.artifact_quality_score is None:
            payload.pop("artifact_quality_score")
        return payload


def evaluate_product_quality(
    mode: str | None,
    quality_gate: Any | None = None,
    ru_artifact_review: dict[str, Any] | None = None,
    stages: list[Any] | None = None,
) -> dict[str, Any]:
    mode = normalize_quality_mode(mode)
    if mode == "draft":
        return ProductQualityStatus(
            mode="draft",
            status="draft",
            production_ready=False,
            graph_sync_allowed=False,
            reasons=["draft mode skips graph sync and production validation"],
        ).as_dict()

    hard_failures, warnings = _collect_quality_gate_reasons(quality_gate)
    review_hard, review_warnings = _collect_review_reasons(ru_artifact_review)
    stage_hard, stage_warnings = _collect_stage_reasons(stages or [])
    hard_failures.extend(review_hard)
    hard_failures.extend(stage_hard)
    warnings.extend(review_warnings)
    warnings.extend(stage_warnings)

    if hard_failures:
        return ProductQualityStatus(
            mode=mode,
            status="failed",
            production_ready=False,
            graph_sync_allowed=False,
            reasons=hard_failures + warnings,
        ).as_dict()

    if warnings:
        return ProductQualityStatus(
            mode=mode,
            status="warning",
            production_ready=mode == "best_effort",
            graph_sync_allowed=mode == "best_effort",
            reasons=warnings,
        ).as_dict()

    return ProductQualityStatus(
        mode=mode,
        status="ready",
        production_ready=True,
        graph_sync_allowed=True,
        reasons=[],
        ).as_dict()


def build_product_quality_payload(
    store: Any,
    video_id: str,
    mode: str | None,
    quality_gate: Any | None = None,
    ru_artifact_review: dict[str, Any] | None = None,
    stages: list[Any] | None = None,
) -> dict[str, Any]:
    from yt_sub_hub.artifact_quality_score import compute_artifact_quality_score

    payload = evaluate_product_quality(mode, quality_gate, ru_artifact_review, stages)
    score = compute_artifact_quality_score(
        store,
        video_id,
        quality_mode=payload["mode"],
        quality_gate=quality_gate,
        ru_artifact_review=ru_artifact_review,
        stages=stages,
    )
    return _merge_artifact_quality_score(payload, score)


def _merge_artifact_quality_score(base_payload: dict[str, Any], score_payload: dict[str, Any]) -> dict[str, Any]:
    payload = {**base_payload, "artifact_quality_score": score_payload}
    score_status = score_payload.get("status")
    if payload.get("status") == "failed":
        if score_status in {"failed", "warning"}:
            payload["reasons"] = _dedupe_reasons(
                list(payload.get("reasons") or []) + _artifact_quality_score_reasons(str(score_status), score_payload)
            )
        return payload

    mode = normalize_quality_mode(payload.get("mode"))
    if mode == "draft" and score_status in {"failed", "warning"}:
        payload["production_ready"] = False
        payload["graph_sync_allowed"] = False
        payload["reasons"] = _dedupe_reasons(
            list(payload.get("reasons") or []) + _artifact_quality_score_reasons(str(score_status), score_payload)
        )
        return payload

    if score_status == "failed":
        payload["status"] = "failed"
        payload["production_ready"] = False
        payload["graph_sync_allowed"] = False
        payload["reasons"] = _dedupe_reasons(
            list(payload.get("reasons") or []) + _artifact_quality_score_reasons("failed", score_payload)
        )
    elif score_status == "warning":
        payload["status"] = "warning"
        payload["production_ready"] = mode == "best_effort"
        payload["graph_sync_allowed"] = mode == "best_effort"
        payload["reasons"] = _dedupe_reasons(
            list(payload.get("reasons") or []) + _artifact_quality_score_reasons("warning", score_payload)
        )
    return payload


def _artifact_quality_score_reasons(status: str, score_payload: dict[str, Any]) -> list[str]:
    overall_score = score_payload.get("overall_score", "unknown")
    reasons = [f"artifact quality score {status}: overall_score={overall_score}"]
    reasons.extend(f"artifact quality score reason: {reason}" for reason in score_payload.get("reasons") or [])
    return reasons


def _dedupe_reasons(reasons: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        if reason in seen:
            continue
        seen.add(reason)
        deduped.append(reason)
    return deduped


def is_operational_warning(stage_name: str | None, warning: Any | None) -> bool:
    if not warning:
        return False
    name = (stage_name or "").strip().lower()
    message = str(warning).strip().lower()
    if "cache hit" in message:
        return True
    if name == "transcript_source" and "needs whisper" in message:
        return True
    if name == "visual_frame_analysis" and "on-demand" in message:
        return True
    if name == "deep_graph_llm" and any(term in message for term in ("disabled", "skipped", "not enabled")):
        return True
    if name == "analysis_reduce" and "skipped by dependency branch" in message:
        return True
    return False


def is_product_quality_warning(stage_name: str | None, warning: Any | None) -> bool:
    if not warning or is_operational_warning(stage_name, warning):
        return False
    name = (stage_name or "").strip().lower()
    message = str(warning).strip().lower()
    if name in _QUALITY_WARNING_STAGES:
        return True
    if any(term in message for term in _QUALITY_WARNING_TERMS):
        return True
    return True


def _collect_quality_gate_reasons(quality_gate: Any | None) -> tuple[list[str], list[str]]:
    if quality_gate is None:
        return [], []
    if hasattr(quality_gate, "as_dict"):
        quality_gate = quality_gate.as_dict()
    hard = [f"quality_gate hard failure: {reason}" for reason in (quality_gate.get("hard_failures") or [])]
    warnings = [f"quality_gate warning: {reason}" for reason in (quality_gate.get("warnings") or [])]
    return hard, warnings


def _collect_review_reasons(review: dict[str, Any] | None) -> tuple[list[str], list[str]]:
    if not review:
        return [], []
    hard: list[str] = []
    warnings: list[str] = []
    if review.get("status") == "failed":
        hard.append("ru_artifact_review failed")
    for issue in review.get("issues") or []:
        if not isinstance(issue, dict):
            continue
        message = str(issue.get("message") or issue.get("code") or "review issue")
        if issue.get("severity") == "hard":
            hard.append(f"ru_artifact_review hard failure: {message}")
        elif issue.get("severity") == "warning":
            warnings.append(f"ru_artifact_review warning: {message}")
    return hard, warnings


def _collect_stage_reasons(stages: list[Any]) -> tuple[list[str], list[str]]:
    hard: list[str] = []
    warnings: list[str] = []
    for stage in stages:
        name = stage.get("stage_name") if isinstance(stage, dict) else getattr(stage, "stage_name", None)
        normalized_name = (name or "").strip().lower()
        if normalized_name.startswith("research_") or normalized_name in {"research_enrichment", "neo4j_research_sync"}:
            continue
        status = stage.get("status") if isinstance(stage, dict) else getattr(stage, "status", None)
        warning = stage.get("warning") if isinstance(stage, dict) else getattr(stage, "warning", None)
        error = stage.get("error") if isinstance(stage, dict) else getattr(stage, "error", None)
        hard_failure = getattr(stage, "hard_failure", None)
        if isinstance(stage, dict):
            hard_failure = stage.get("hard_failure", hard_failure)
        if status == "failed" and hard_failure:
            hard.append(f"{name or 'stage'} hard failure: {error or 'failed'}")
        if is_product_quality_warning(name, warning):
            warnings.append(f"{name or 'stage'} warning: {warning}")
    return hard, warnings
