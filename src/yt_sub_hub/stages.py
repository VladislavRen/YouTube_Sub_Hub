from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageDefinition:
    name: str
    resource_lock: str | None
    max_attempts: int
    hard_failure: bool = True


PIPELINE_STAGES = [
    StageDefinition("metadata", "youtube_metadata", 3),
    StageDefinition("transcript_source", "youtube_metadata", 2),
    StageDefinition("audio_download", "audio_download", 2),
    StageDefinition("whisper_transcription", "whisper_cpu", 1),
    StageDefinition("transcript_translation_expand", "local_validation", 1),
    StageDefinition("transcript_translation_merge", "local_validation", 1),
    StageDefinition("scientific_analysis", "lm_studio_analysis", 2),
    StageDefinition("analysis_reduce", "local_validation", 1),
    StageDefinition("analysis_translation", "lm_studio_translation", 2),
    StageDefinition("analysis_ru_review", "local_validation", 1),
    StageDefinition("markdown_render", "markdown_render", 1),
    StageDefinition("deterministic_graph", "local_validation", 1),
    StageDefinition("deep_graph_llm", "deep_graph_llm", 1, hard_failure=False),
    StageDefinition("visual_frame_analysis", "lm_studio_analysis", 1, hard_failure=False),
    StageDefinition("ru_artifact_review", "critic_review", 1),
    StageDefinition("neo4j_sync", "neo4j_sync", 2, hard_failure=False),
    StageDefinition("quality_gate", "local_validation", 1),
]

_STAGE_ALIASES = {
    "lm_studio_analysis": "scientific_analysis",
    "transcript_translation": "transcript_translation_expand",
}


def stage_by_name(name: str) -> StageDefinition:
    if name.startswith("youtube_discovery_query_"):
        return StageDefinition(name, "youtube_data_api", 2)
    if name in {"youtube_discovery_seed_extract", "youtube_discovery_query_expand", "youtube_discovery_candidate_fetch"}:
        return StageDefinition(name, "local_validation", 1)
    if name == "youtube_discovery_candidate_score":
        return StageDefinition(name, "local_candidate_scoring", 1)
    if name == "youtube_discovery_review":
        return StageDefinition(name, "local_validation", 1)
    if name in {"youtube_discovery_graph", "youtube_discovery_report"}:
        return StageDefinition(name, "local_validation", 1)
    if name == "youtube_discovery_enqueue":
        return StageDefinition(name, "queue_mutation", 1)
    if name.startswith("transcript_translation_chunk_"):
        return StageDefinition(name, "lm_studio_translation", 2)
    if name.startswith("analysis_chunk_"):
        return StageDefinition(name, "lm_studio_analysis", 2)
    if name == "analysis_reduce":
        return StageDefinition(name, "local_validation", 1)
    lookup_name = _STAGE_ALIASES.get(name, name)
    for stage in PIPELINE_STAGES:
        if stage.name == lookup_name:
            return stage
    raise KeyError(f"Unknown pipeline stage: {name}")


def retry_delay_seconds(attempt: int, *, base_seconds: float = 30, max_seconds: float = 900) -> float:
    normalized_attempt = max(1, int(attempt))
    return min(float(max_seconds), float(base_seconds) * (2 ** (normalized_attempt - 1)))
