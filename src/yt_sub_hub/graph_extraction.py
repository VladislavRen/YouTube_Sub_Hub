from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from typing import Any

from yt_sub_hub.graph import GraphPayload
from yt_sub_hub.models import Analysis, Transcript, VideoMetadata, slugify

SCHEMA_VERSION = "graph-extraction-v1"
GRAPH_PROMPT_VERSION = "deep-graph-chunk-v1"
LLM_DEEP_GRAPH_METHOD = "llm_deep_graph"

ALLOWED_RELATIONSHIP_TYPES = {
    "PUBLISHED",
    "PART_OF",
    "MENTIONS",
    "EXPLAINS",
    "HAS_SEGMENT",
    "EVIDENCE_FOR",
    "ABOUT",
    "SUPPORTED_BY",
    "CONTRADICTS",
    "SUPPORTS",
    "RELATED_TO",
    "CAUSES",
    "CONTRASTS_WITH",
    "MEASURED_BY",
    "USES",
    "APPLIES_TO",
    "HAS_VISUAL_FRAME",
    "SHOWN_IN",
    "VISIBLE_AT",
    "EXPLAINS_VISUAL",
}


def canonical_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").casefold()
    normalized = re.sub(r"[^\w\s]+", " ", normalized, flags=re.UNICODE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def claim_hash(text: str) -> str:
    return hashlib.sha256(canonical_text(text).encode("utf-8")).hexdigest()[:16]


def _stable_hash(*parts: Any, length: int = 24) -> str:
    raw = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def relationship_uid(
    from_id: str,
    rel_type: str,
    to_id: str,
    source_video_id: str | None,
    source_segment_id: str | None,
    extraction_method: str,
) -> str:
    return _stable_hash(from_id, rel_type, to_id, source_video_id, source_segment_id, extraction_method)


def _add_node(nodes: dict[str, dict[str, Any]], node: dict[str, Any]) -> None:
    existing = nodes.get(node["id"])
    if existing is None:
        nodes[node["id"]] = node
        return
    for key, value in node.items():
        if key == "aliases":
            aliases = list(dict.fromkeys([*existing.get("aliases", []), *value]))
            existing["aliases"] = aliases
        elif existing.get(key) in (None, "", []):
            existing[key] = value


def _add_relationship(
    relationships: dict[str, dict[str, Any]],
    from_id: str,
    to_id: str,
    rel_type: str,
    *,
    source_video_id: str,
    source_segment_id: str | None = None,
    timestamp_start: float | None = None,
    timestamp_end: float | None = None,
    extraction_method: str = "analysis_v1",
    prompt_version: str | None = None,
    model: str | None = None,
    evidence_text: str | None = None,
    confidence: float = 1.0,
    reliability: str = "deterministic",
    audit_status: str = "accepted",
) -> None:
    if rel_type not in ALLOWED_RELATIONSHIP_TYPES:
        return
    rel_id = relationship_uid(from_id, rel_type, to_id, source_video_id, source_segment_id, extraction_method)
    relationships.setdefault(
        rel_id,
        {
            "id": rel_id,
            "from": from_id,
            "to": to_id,
            "type": rel_type,
            "confidence": confidence,
            "reliability": reliability,
            "source_video_id": source_video_id,
            "source_segment_id": source_segment_id,
            "timestamp_start": timestamp_start,
            "timestamp_end": timestamp_end,
            "extraction_method": extraction_method,
            "prompt_version": prompt_version,
            "model": model,
            "evidence_text": evidence_text,
            "audit_status": audit_status,
        },
    )


def _matching_segment(transcript: Transcript, text: str) -> tuple[int, Any] | None:
    needle = canonical_text(text)
    if not needle:
        return None
    for index, segment in enumerate(transcript.segments):
        haystack = canonical_text(segment.text)
        if needle in haystack or haystack in needle:
            return index, segment
    words = [word for word in needle.split() if len(word) > 3]
    if not words:
        return None
    best: tuple[int, Any, int] | None = None
    for index, segment in enumerate(transcript.segments):
        haystack = canonical_text(segment.text)
        score = sum(1 for word in words if word in haystack)
        if score and (best is None or score > best[2]):
            best = (index, segment, score)
    if best is None:
        return None
    return best[0], best[1]


def _segment_id(video_id: str, index: int) -> str:
    return f"segment:{video_id}:{index}"


def _segment_for_timestamp(transcript: Transcript, timestamp: float | int | None) -> tuple[str | None, float | None, float | None]:
    if timestamp is None:
        return None, None, None
    value = float(timestamp)
    for index, segment in enumerate(transcript.segments):
        start = float(segment.start)
        end = start + float(segment.duration)
        if start <= value <= end:
            return _segment_id(transcript.video_id, index), start, end
    return None, value, value


def _recalculate_quality(extraction: dict[str, Any]) -> None:
    node_counts = Counter(node["label"] for node in extraction.get("nodes", []))
    relationship_counts = Counter(rel["type"] for rel in extraction.get("relationships", []))
    chunks = extraction.get("chunks", [])
    failed_chunks = [chunk for chunk in chunks if chunk.get("rejected")]
    llm_relationships = [
        rel
        for rel in extraction.get("relationships", [])
        if rel.get("audit_status", "accepted") == "accepted"
        and (rel.get("extraction_method") == LLM_DEEP_GRAPH_METHOD or rel.get("reliability") == "llm_extracted")
    ]
    quality = extraction.setdefault("quality_report", {})
    quality["node_count_by_label"] = dict(sorted(node_counts.items()))
    quality["relationship_count_by_type"] = dict(sorted(relationship_counts.items()))
    quality["chunks_total"] = len(chunks)
    quality["chunks_succeeded"] = len(chunks) - len(failed_chunks)
    quality["chunks_failed"] = len(failed_chunks)
    quality["relationships_accepted"] = len(llm_relationships)
    graph_bloat = quality.setdefault("graph_bloat", {})
    graph_bloat["node_count"] = len(extraction.get("nodes", []))
    graph_bloat["relationship_count"] = len(extraction.get("relationships", []))


def visual_frame_node_id(video_id: str, timestamp_seconds: float | int) -> str:
    from yt_sub_hub.visual import timestamp_key

    return f"visual_frame:{video_id}:{timestamp_key(timestamp_seconds)}"


def visual_evidence_node_id(frame_id: str, summary_text: str) -> str:
    return f"visual_evidence:{_stable_hash(frame_id, summary_text, length=16)}"


def _visual_item_node_id(prefix: str, value: str) -> str:
    return f"{prefix}:{slugify(canonical_text(value))}"


def build_visual_graph_payload(visual_summary: dict[str, Any]) -> GraphPayload:
    extraction = merge_visual_summary_into_extraction(
        {
            "schema_version": SCHEMA_VERSION,
            "video_id": visual_summary.get("video_id"),
            "nodes": [],
            "relationships": [],
            "quality_report": {},
        },
        visual_summary,
    )
    return GraphPayload(nodes=extraction["nodes"], relationships=extraction["relationships"])


def merge_visual_summary_into_extraction(extraction: dict[str, Any], visual_summary: dict[str, Any]) -> dict[str, Any]:
    video_id = str(visual_summary.get("video_id") or extraction.get("video_id") or "")
    if not video_id:
        return extraction
    merged = {
        **extraction,
        "nodes": [dict(node) for node in extraction.get("nodes", [])],
        "relationships": [dict(relationship) for relationship in extraction.get("relationships", [])],
        "quality_report": {**extraction.get("quality_report", {})},
    }
    nodes = _node_lookup(merged)
    relationships = _relationship_lookup(merged)
    video_node_id = f"video:{video_id}"
    _add_node(
        nodes,
        {
            "id": video_node_id,
            "label": "Video",
            "video_id": video_id,
        },
    )

    def add_visual_relationship(from_id: str, to_id: str, rel_type: str, frame: dict[str, Any], frame_id: str) -> None:
        confidence = _confidence(frame) if "confidence" in frame else 1.0
        timestamp = float(frame.get("timestamp_seconds", 0))
        rel_id = relationship_uid(from_id, rel_type, to_id, video_id, frame_id, "vision_frame_v1")
        relationships.setdefault(
            rel_id,
            {
                "id": rel_id,
                "from": from_id,
                "to": to_id,
                "type": rel_type,
                "source_video_id": video_id,
                "source_frame_id": frame_id,
                "timestamp_start": timestamp,
                "timestamp_end": timestamp,
                "extraction_method": "vision_frame_v1",
                "confidence": confidence,
                "audit_status": "accepted",
                "evidence_text": frame.get("summary"),
            },
        )

    for frame in visual_summary.get("frames", []):
        if not isinstance(frame, dict):
            continue
        timestamp = float(frame.get("timestamp_seconds", 0))
        frame_id = visual_frame_node_id(video_id, timestamp)
        timestamp_key = frame_id.rsplit(":", 1)[-1]
        confidence = _confidence(frame) if "confidence" in frame else 1.0
        _add_node(
            nodes,
            {
                "id": frame_id,
                "label": "VisualFrame",
                "video_id": video_id,
                "timestamp_seconds": timestamp,
                "frame_path": frame.get("frame_path"),
                "screen_type": frame.get("screen_type"),
                "summary": frame.get("summary"),
                "confidence": confidence,
                "timestamp_key": timestamp_key,
            },
        )
        add_visual_relationship(video_node_id, frame_id, "HAS_VISUAL_FRAME", frame, frame_id)

        summary = str(frame.get("summary") or "")
        if summary:
            observation_id = visual_evidence_node_id(frame_id, summary)
            _add_node(
                nodes,
                {
                    "id": observation_id,
                    "label": "VisualObservation",
                    "text": summary,
                    "source_video_id": video_id,
                    "source_frame_id": frame_id,
                    "confidence": confidence,
                },
            )
            add_visual_relationship(observation_id, frame_id, "SHOWN_IN", frame, frame_id)
            add_visual_relationship(observation_id, video_node_id, "EXPLAINS_VISUAL", frame, frame_id)

        chart_elements = frame.get("chart_elements") if isinstance(frame.get("chart_elements"), dict) else {}
        for pattern in chart_elements.get("patterns", []) or []:
            node_id = _visual_item_node_id("chart_pattern", str(pattern))
            _add_node(nodes, {"id": node_id, "label": "ChartPattern", "name": str(pattern), "slug": slugify(str(pattern))})
            add_visual_relationship(node_id, frame_id, "SHOWN_IN", frame, frame_id)
        for level in chart_elements.get("levels", []) or []:
            node_id = _visual_item_node_id("chart_level", str(level))
            _add_node(nodes, {"id": node_id, "label": "ChartLevel", "name": str(level), "slug": slugify(str(level))})
            add_visual_relationship(node_id, frame_id, "SHOWN_IN", frame, frame_id)
        for indicator in chart_elements.get("indicators", []) or []:
            node_id = _visual_item_node_id("indicator", str(indicator))
            _add_node(nodes, {"id": node_id, "label": "Indicator", "name": str(indicator), "slug": slugify(str(indicator))})
            add_visual_relationship(node_id, frame_id, "SHOWN_IN", frame, frame_id)

    merged["nodes"] = list(nodes.values())
    merged["relationships"] = list(relationships.values())
    _recalculate_quality(merged)
    return merged


def build_deterministic_graph_extraction(
    metadata: VideoMetadata,
    transcript: Transcript,
    analysis: Analysis,
) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    relationships: dict[str, dict[str, Any]] = {}
    video_id = metadata.video_id
    video_node_id = f"video:{video_id}"
    prompt_version = analysis.prompt_version
    model = analysis.model
    duplicate_concept_candidates: list[dict[str, Any]] = []
    concept_slugs_seen: dict[str, str] = {}
    orphan_claims_without_evidence: list[str] = []
    claims_without_source_segment: list[str] = []

    _add_node(
        nodes,
        {
            "id": video_node_id,
            "label": "Video",
            "video_id": video_id,
            "title": metadata.title,
            "url": metadata.canonical_url,
            "duration_seconds": metadata.duration_seconds,
            "language": analysis.language or transcript.language,
        },
    )

    if metadata.channel:
        channel_slug = slugify(metadata.channel)
        channel_id = f"channel:{channel_slug}"
        _add_node(nodes, {"id": channel_id, "label": "Channel", "name": metadata.channel, "slug": channel_slug})
        _add_relationship(
            relationships,
            channel_id,
            video_node_id,
            "PUBLISHED",
            source_video_id=video_id,
            extraction_method="metadata_v1",
        )

    for index, segment in enumerate(transcript.segments):
        segment_id = _segment_id(video_id, index)
        end = float(segment.start) + float(segment.duration)
        _add_node(
            nodes,
            {
                "id": segment_id,
                "label": "SourceSegment",
                "video_id": video_id,
                "segment_index": index,
                "start": segment.start,
                "end": end,
                "text": segment.text,
            },
        )
        _add_relationship(
            relationships,
            video_node_id,
            segment_id,
            "HAS_SEGMENT",
            source_video_id=video_id,
            source_segment_id=segment_id,
            timestamp_start=segment.start,
            timestamp_end=end,
            extraction_method="transcript_v1",
        )

    for source, topic_values in (("analysis.related_topics", analysis.related_topics), ("analysis.tags", analysis.tags)):
        for topic in topic_values:
            topic_slug = slugify(topic)
            if not topic_slug:
                continue
            topic_id = f"topic:{topic_slug}"
            _add_node(nodes, {"id": topic_id, "label": "Topic", "name": topic, "slug": topic_slug, "source": source})
            _add_relationship(
                relationships,
                video_node_id,
                topic_id,
                "PART_OF",
                source_video_id=video_id,
                extraction_method=source,
                prompt_version=prompt_version,
                model=model,
            )

    concept_ids: list[str] = []
    for concept in analysis.concepts:
        canonical_name = canonical_text(concept.name)
        concept_slug = slugify(canonical_name)
        if not concept_slug:
            continue
        if concept_slug in concept_slugs_seen and concept_slugs_seen[concept_slug] != concept.name:
            duplicate_concept_candidates.append(
                {"slug": concept_slug, "first": concept_slugs_seen[concept_slug], "duplicate": concept.name}
            )
        concept_slugs_seen.setdefault(concept_slug, concept.name)
        concept_id = f"concept:{concept_slug}"
        concept_ids.append(concept_id)
        _add_node(
            nodes,
            {
                "id": concept_id,
                "label": "Concept",
                "name": concept.name,
                "canonical_name": canonical_name,
                "slug": concept_slug,
                "definition": concept.definition,
                "language": analysis.language,
                "aliases": [concept.name],
                "conflict_status": "ok",
            },
        )
        _add_relationship(
            relationships,
            video_node_id,
            concept_id,
            "MENTIONS",
            source_video_id=video_id,
            extraction_method="analysis.concepts",
            prompt_version=prompt_version,
            model=model,
            evidence_text=concept.definition,
        )
        for index, segment in enumerate(transcript.segments):
            if canonical_name and canonical_name in canonical_text(segment.text):
                segment_id = _segment_id(video_id, index)
                _add_relationship(
                    relationships,
                    segment_id,
                    concept_id,
                    "MENTIONS",
                    source_video_id=video_id,
                    source_segment_id=segment_id,
                    timestamp_start=segment.start,
                    timestamp_end=float(segment.start) + float(segment.duration),
                    extraction_method="lexical_match_v1",
                    evidence_text=segment.text,
                )

    for claim in analysis.claims:
        normalized_claim = canonical_text(claim.text)
        hash_value = claim_hash(claim.text)
        claim_id = f"claim:{hash_value}"
        _add_node(
            nodes,
            {
                "id": claim_id,
                "label": "Claim",
                "text": claim.text,
                "canonical_text": normalized_claim,
                "claim_hash": hash_value,
                "language": analysis.language,
                "status": "accepted",
            },
        )
        _add_relationship(
            relationships,
            video_node_id,
            claim_id,
            "EXPLAINS",
            source_video_id=video_id,
            extraction_method="analysis.claims",
            prompt_version=prompt_version,
            model=model,
            evidence_text=claim.evidence,
        )
        for concept_id in concept_ids:
            concept_name = nodes[concept_id].get("canonical_name", "")
            normalized_evidence = canonical_text(claim.evidence)
            if concept_name and (concept_name in normalized_claim or concept_name in normalized_evidence):
                _add_relationship(
                    relationships,
                    claim_id,
                    concept_id,
                    "ABOUT",
                    source_video_id=video_id,
                    extraction_method="claim_text_match_v1",
                    prompt_version=prompt_version,
                    model=model,
                )
        if not claim.evidence:
            orphan_claims_without_evidence.append(claim_id)
            continue
        evidence_id = f"evidence:{_stable_hash(video_id, hash_value, claim.evidence, length=16)}"
        _add_node(
            nodes,
            {
                "id": evidence_id,
                "label": "Evidence",
                "text": claim.evidence,
                "kind": "paraphrase",
                "source_video_id": video_id,
                "confidence": 1.0,
            },
        )
        _add_relationship(
            relationships,
            claim_id,
            evidence_id,
            "SUPPORTED_BY",
            source_video_id=video_id,
            extraction_method="analysis.claim_evidence",
            prompt_version=prompt_version,
            model=model,
            evidence_text=claim.evidence,
        )
        match = _matching_segment(transcript, claim.evidence)
        if match is None:
            claims_without_source_segment.append(claim_id)
            continue
        segment_index, segment = match
        segment_id = _segment_id(video_id, segment_index)
        _add_relationship(
            relationships,
            segment_id,
            claim_id,
            "EVIDENCE_FOR",
            source_video_id=video_id,
            source_segment_id=segment_id,
            timestamp_start=segment.start,
            timestamp_end=float(segment.start) + float(segment.duration),
            extraction_method="evidence_text_match_v1",
            prompt_version=prompt_version,
            model=model,
            evidence_text=claim.evidence,
        )

    for question in analysis.open_questions:
        question_id = f"question:{_stable_hash(video_id, question, length=16)}"
        _add_node(nodes, {"id": question_id, "label": "Question", "text": question, "kind": "open", "status": "open"})
        normalized_question = canonical_text(question)
        for concept_id in concept_ids:
            concept_name = nodes[concept_id].get("canonical_name", "")
            if concept_name and concept_name in normalized_question:
                _add_relationship(
                    relationships,
                    question_id,
                    concept_id,
                    "ABOUT",
                    source_video_id=video_id,
                    extraction_method="question_text_match_v1",
                    prompt_version=prompt_version,
                    model=model,
                    evidence_text=question,
                )

    node_list = list(nodes.values())
    relationship_list = list(relationships.values())
    node_counts = Counter(node["label"] for node in node_list)
    relationship_counts = Counter(rel["type"] for rel in relationship_list)
    return {
        "schema_version": SCHEMA_VERSION,
        "video_id": video_id,
        "generated_from": {
            "metadata": "metadata.json",
            "transcript": "transcript.normalized.json",
            "analysis": "analysis.json",
            "analysis_prompt_version": prompt_version,
            "graph_prompt_version": GRAPH_PROMPT_VERSION,
            "model": model,
        },
        "chunks": [],
        "nodes": node_list,
        "relationships": relationship_list,
        "quality_report": {
            "node_count_by_label": dict(sorted(node_counts.items())),
            "relationship_count_by_type": dict(sorted(relationship_counts.items())),
            "rejected_relationships": [],
            "low_confidence_extractions": [],
            "duplicate_concept_candidates": duplicate_concept_candidates,
            "orphan_claims_without_evidence": orphan_claims_without_evidence,
            "claims_without_source_segment": claims_without_source_segment,
            "graph_bloat": {
                "node_count": len(node_list),
                "relationship_count": len(relationship_list),
            },
        },
    }


def _node_lookup(extraction: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {node["id"]: node for node in extraction.get("nodes", [])}


def _relationship_lookup(extraction: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        relationship.get("id") or f"{relationship.get('from')}|{relationship.get('type')}|{relationship.get('to')}": relationship
        for relationship in extraction.get("relationships", [])
    }


def _append_rejection(quality: dict[str, Any], bucket: str, reason: str, item: dict[str, Any]) -> None:
    quality.setdefault(bucket, []).append({"reason": reason, "item": item})


def _confidence(item: dict[str, Any]) -> float:
    try:
        return float(item.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0


def merge_deep_graph_chunks(
    metadata: VideoMetadata,
    transcript: Transcript,
    base_extraction: dict[str, Any],
    chunks: list[dict[str, Any]],
    min_confidence: float,
    model: str | None,
) -> dict[str, Any]:
    extraction = {
        **base_extraction,
        "chunks": list(chunks),
        "nodes": [dict(node) for node in base_extraction.get("nodes", [])],
        "relationships": [dict(relationship) for relationship in base_extraction.get("relationships", [])],
        "quality_report": {**base_extraction.get("quality_report", {})},
    }
    quality = extraction.setdefault("quality_report", {})
    quality.setdefault("rejected_relationships", [])
    quality.setdefault("low_confidence_extractions", [])
    quality.setdefault("rejected_nodes", [])

    nodes = _node_lookup(extraction)
    relationships = _relationship_lookup(extraction)

    def add_global_node(node: dict[str, Any]) -> None:
        existing = nodes.get(node["id"])
        if existing is None:
            nodes[node["id"]] = node
            return
        if node.get("label") == "Concept":
            aliases = list(dict.fromkeys([*existing.get("aliases", []), *node.get("aliases", [])]))
            existing["aliases"] = aliases
        for key, value in node.items():
            if existing.get(key) in (None, "", []):
                existing[key] = value

    def add_llm_relationship(
        from_id: str,
        to_id: str,
        rel_type: str,
        item: dict[str, Any],
        chunk: dict[str, Any],
        evidence_text: str,
    ) -> None:
        if from_id == to_id:
            _append_rejection(quality, "rejected_relationships", "self_relation", item)
            return
        confidence = _confidence(item)
        if confidence < min_confidence:
            _append_rejection(quality, "low_confidence_extractions", "low_confidence", item)
            return
        if rel_type not in ALLOWED_RELATIONSHIP_TYPES:
            _append_rejection(quality, "rejected_relationships", "unsupported_relationship_type", item)
            return
        timestamp_start = item.get("timestamp_start", chunk.get("timestamp_start"))
        timestamp_end = item.get("timestamp_end", chunk.get("timestamp_end"))
        source_segment_id, _, _ = _segment_for_timestamp(transcript, timestamp_start)
        rel_id = relationship_uid(
            from_id,
            rel_type,
            to_id,
            metadata.video_id,
            source_segment_id or chunk.get("chunk_id"),
            LLM_DEEP_GRAPH_METHOD,
        )
        relationships.setdefault(
            rel_id,
            {
                "id": rel_id,
                "from": from_id,
                "to": to_id,
                "type": rel_type,
                "confidence": confidence,
                "reliability": "llm_extracted",
                "source_video_id": metadata.video_id,
                "source_segment_id": source_segment_id,
                "chunk_id": chunk.get("chunk_id"),
                "timestamp_start": timestamp_start,
                "timestamp_end": timestamp_end,
                "extraction_method": LLM_DEEP_GRAPH_METHOD,
                "prompt_version": GRAPH_PROMPT_VERSION,
                "model": model,
                "evidence_text": evidence_text,
                "audit_status": "accepted",
            },
        )

    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id") or "chunk")
        if chunk.get("rejected"):
            _append_rejection(quality, "rejected_nodes", str(chunk.get("error") or "rejected_chunk"), chunk)
            continue
        local: dict[str, str] = {}
        local_owner: dict[str, str] = {}

        def register_local(local_id: Any, node_id: str, kind: str, item: dict[str, Any]) -> bool:
            key = str(local_id or "").strip()
            if not key:
                _append_rejection(quality, "rejected_nodes", "missing_local_id", item)
                return False
            if key in local:
                _append_rejection(
                    quality,
                    "rejected_nodes",
                    "duplicate_local_id",
                    {"kind": kind, "previous_kind": local_owner.get(key), "local_id": key, "item": item},
                )
                return False
            local[key] = node_id
            local_owner[key] = kind
            return True

        for concept in chunk.get("concepts", []):
            if _confidence(concept) < min_confidence:
                _append_rejection(quality, "low_confidence_extractions", "low_confidence_concept", concept)
                continue
            canonical_name = canonical_text(str(concept.get("name", "")))
            slug = slugify(canonical_name)
            if not slug:
                _append_rejection(quality, "rejected_nodes", "empty_concept_name", concept)
                continue
            node_id = f"concept:{slug}"
            aliases = [concept.get("name", ""), *concept.get("aliases", [])]
            add_global_node(
                {
                    "id": node_id,
                    "label": "Concept",
                    "name": concept.get("name", ""),
                    "canonical_name": canonical_name,
                    "slug": slug,
                    "definition": concept.get("definition", ""),
                    "language": base_extraction.get("generated_from", {}).get("language"),
                    "aliases": [alias for alias in dict.fromkeys(aliases) if alias],
                    "conflict_status": "ok",
                }
            )
            register_local(concept.get("local_id"), node_id, "concept", concept)

        for claim in chunk.get("claims", []):
            if _confidence(claim) < min_confidence:
                _append_rejection(quality, "low_confidence_extractions", "low_confidence_claim", claim)
                continue
            hash_value = claim_hash(str(claim.get("text", "")))
            node_id = f"claim:{hash_value}"
            add_global_node(
                {
                    "id": node_id,
                    "label": "Claim",
                    "text": claim.get("text", ""),
                    "canonical_text": canonical_text(str(claim.get("text", ""))),
                    "claim_hash": hash_value,
                    "language": None,
                    "status": "accepted",
                }
            )
            register_local(claim.get("local_id"), node_id, "claim", claim)

        for evidence in chunk.get("evidence", []):
            if _confidence(evidence) < min_confidence:
                _append_rejection(quality, "low_confidence_extractions", "low_confidence_evidence", evidence)
                continue
            node_id = f"evidence:{_stable_hash(metadata.video_id, chunk_id, evidence.get('local_id'), evidence.get('text'), length=16)}"
            add_global_node(
                {
                    "id": node_id,
                    "label": "Evidence",
                    "text": evidence.get("text", ""),
                    "kind": evidence.get("kind", "paraphrase"),
                    "source_video_id": metadata.video_id,
                    "confidence": _confidence(evidence),
                }
            )
            if not register_local(evidence.get("local_id"), node_id, "evidence", evidence):
                continue
            for claim_local_id in evidence.get("supports_claim_ids", []):
                claim_id = local.get(claim_local_id)
                if claim_id is None:
                    _append_rejection(quality, "rejected_relationships", "unknown_local_id", evidence)
                    continue
                add_llm_relationship(claim_id, node_id, "SUPPORTED_BY", evidence, chunk, evidence.get("quote") or evidence.get("text", ""))
                source_segment_id, _, _ = _segment_for_timestamp(transcript, evidence.get("timestamp_start"))
                if source_segment_id:
                    add_llm_relationship(source_segment_id, claim_id, "EVIDENCE_FOR", evidence, chunk, evidence.get("quote") or evidence.get("text", ""))

        for collection, label, id_prefix, name_key, text_key in (
            ("tools", "Tool", "tool", "name", "description"),
            ("methods", "Method", "method", "name", "description"),
            ("metrics", "Metric", "metric", "name", "definition"),
        ):
            for item in chunk.get(collection, []):
                if _confidence(item) < min_confidence:
                    _append_rejection(quality, "low_confidence_extractions", f"low_confidence_{collection}", item)
                    continue
                slug = slugify(canonical_text(str(item.get(name_key, ""))))
                if not slug:
                    _append_rejection(quality, "rejected_nodes", f"empty_{collection}_name", item)
                    continue
                node_id = f"{id_prefix}:{slug}"
                node = {
                    "id": node_id,
                    "label": label,
                    "name": item.get(name_key, ""),
                    "slug": slug,
                    "description": item.get(text_key, ""),
                    "confidence": _confidence(item),
                }
                if label == "Metric":
                    node["definition"] = item.get("definition", "")
                add_global_node(node)
                register_local(item.get("local_id"), node_id, collection, item)

        for risk in chunk.get("risks", []):
            if _confidence(risk) < min_confidence:
                _append_rejection(quality, "low_confidence_extractions", "low_confidence_risk", risk)
                continue
            node_id = f"risk:{_stable_hash(metadata.video_id, chunk_id, risk.get('text'), length=16)}"
            add_global_node(
                {
                    "id": node_id,
                    "label": "Risk",
                    "text": risk.get("text", ""),
                    "category": "llm",
                    "severity": None,
                    "confidence": _confidence(risk),
                }
            )
            register_local(risk.get("local_id"), node_id, "risk", risk)

        for question in chunk.get("questions", []):
            if _confidence(question) < min_confidence:
                _append_rejection(quality, "low_confidence_extractions", "low_confidence_question", question)
                continue
            node_id = f"question:{_stable_hash(metadata.video_id, chunk_id, question.get('text'), length=16)}"
            add_global_node(
                {
                    "id": node_id,
                    "label": "Question",
                    "text": question.get("text", ""),
                    "kind": "llm",
                    "status": "open",
                    "confidence": _confidence(question),
                }
            )
            register_local(question.get("local_id"), node_id, "question", question)

        for claim in chunk.get("claims", []):
            claim_id = local.get(claim.get("local_id"))
            if not claim_id:
                continue
            for concept_local_id, concept_id in local.items():
                if not concept_id.startswith("concept:"):
                    continue
                concept_name = nodes.get(concept_id, {}).get("canonical_name", "")
                text = canonical_text(f"{claim.get('text', '')} {claim.get('paraphrased_evidence', '')}")
                if concept_name and concept_name in text:
                    add_llm_relationship(claim_id, concept_id, "ABOUT", claim, chunk, claim.get("paraphrased_evidence", ""))

        for relation in chunk.get("concept_relations", []):
            from_id = local.get(relation.get("from_concept_id"))
            to_id = local.get(relation.get("to_concept_id"))
            if from_id is None or to_id is None:
                _append_rejection(quality, "rejected_relationships", "unknown_local_id", relation)
                continue
            rel_type = relation.get("type", "")
            if rel_type == "MEASURED_BY" and not to_id.startswith("metric:"):
                _append_rejection(quality, "rejected_relationships", "invalid_measured_by_target", relation)
                continue
            add_llm_relationship(from_id, to_id, rel_type, relation, chunk, relation.get("evidence_quote", ""))

        for relation in chunk.get("claim_relations", []):
            from_id = local.get(relation.get("from_claim_id"))
            to_id = local.get(relation.get("to_claim_id"))
            if from_id is None or to_id is None:
                _append_rejection(quality, "rejected_relationships", "unknown_local_id", relation)
                continue
            add_llm_relationship(from_id, to_id, relation.get("type", ""), relation, chunk, relation.get("evidence_quote", ""))

        for method in chunk.get("methods", []):
            method_id = local.get(method.get("local_id"))
            if method_id is None:
                continue
            for tool_local_id in method.get("uses_tool_ids", []):
                tool_id = local.get(tool_local_id)
                if tool_id is None:
                    _append_rejection(quality, "rejected_relationships", "unknown_local_id", method)
                    continue
                add_llm_relationship(method_id, tool_id, "USES", method, chunk, method.get("evidence_quote", ""))
            for concept_local_id in method.get("applies_to_concept_ids", []):
                concept_id = local.get(concept_local_id)
                if concept_id is None:
                    _append_rejection(quality, "rejected_relationships", "unknown_local_id", method)
                    continue
                add_llm_relationship(method_id, concept_id, "APPLIES_TO", method, chunk, method.get("evidence_quote", ""))

        for metric in chunk.get("metrics", []):
            metric_id = local.get(metric.get("local_id"))
            if metric_id is None:
                continue
            for concept_local_id in metric.get("measured_concept_ids", []):
                concept_id = local.get(concept_local_id)
                if concept_id is None:
                    _append_rejection(quality, "rejected_relationships", "unknown_local_id", metric)
                    continue
                add_llm_relationship(concept_id, metric_id, "MEASURED_BY", metric, chunk, metric.get("evidence_quote", ""))

        for risk in chunk.get("risks", []):
            risk_id = local.get(risk.get("local_id"))
            if risk_id is None:
                continue
            for concept_local_id in risk.get("related_concept_ids", []):
                concept_id = local.get(concept_local_id)
                if concept_id is None:
                    _append_rejection(quality, "rejected_relationships", "unknown_local_id", risk)
                    continue
                add_llm_relationship(risk_id, concept_id, "ABOUT", risk, chunk, risk.get("evidence_quote", ""))

        for question in chunk.get("questions", []):
            question_id = local.get(question.get("local_id"))
            if question_id is None:
                continue
            for concept_local_id in question.get("about_concept_ids", []):
                concept_id = local.get(concept_local_id)
                if concept_id is None:
                    _append_rejection(quality, "rejected_relationships", "unknown_local_id", question)
                    continue
                add_llm_relationship(question_id, concept_id, "ABOUT", question, chunk, question.get("evidence_quote", ""))

    extraction["nodes"] = list(nodes.values())
    extraction["relationships"] = list(relationships.values())
    _recalculate_quality(extraction)
    return extraction


def graph_payload_from_extraction(extraction: dict[str, Any], min_confidence: float = 0.0) -> GraphPayload:
    relationships = [
        rel
        for rel in extraction.get("relationships", [])
        if rel.get("audit_status", "accepted") == "accepted" and float(rel.get("confidence", 0.0)) >= min_confidence
    ]
    return GraphPayload(nodes=list(extraction.get("nodes", [])), relationships=relationships)
