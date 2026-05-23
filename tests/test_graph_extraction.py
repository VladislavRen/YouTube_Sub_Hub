from yt_sub_hub.graph_extraction import (
    ALLOWED_RELATIONSHIP_TYPES,
    build_deterministic_graph_extraction,
    canonical_text,
    claim_hash,
    merge_deep_graph_chunks,
    relationship_uid,
)
from yt_sub_hub.models import Analysis, Claim, Concept, NoteSection, Transcript, TranscriptSegment, VideoMetadata


def test_canonical_text_and_claim_hash_are_stable():
    assert canonical_text("  The MARKET   breaks structure. ") == "the market breaks structure"
    assert claim_hash("The market breaks structure.") == claim_hash(" the market breaks structure ")


def test_relationship_uid_includes_source_context():
    first = relationship_uid("a", "MENTIONS", "b", "video1", "segment:video1:0", "lexical_match_v1")
    second = relationship_uid("a", "MENTIONS", "b", "video1", "segment:video1:1", "lexical_match_v1")

    assert first != second
    assert len(first) == 24


def test_deterministic_graph_extraction_adds_segments_provenance_and_quality_report():
    metadata = VideoMetadata(
        video_id="abc123",
        canonical_url="https://www.youtube.com/watch?v=abc123",
        title="Graph Talk",
        channel="Research Channel",
        duration_seconds=12,
    )
    transcript = Transcript(
        video_id="abc123",
        source="youtube",
        language="en",
        segments=[
            TranscriptSegment(start=0, duration=5, text="Causal inference compares interventions."),
            TranscriptSegment(start=5, duration=7, text="Graphs help explain causal inference claims."),
        ],
    )
    analysis = Analysis(
        language="en",
        cleaned_title="Graph Talk",
        abstract="Abstract",
        tags=["science"],
        outline=[],
        sections=[NoteSection(heading="Intro", summary="Summary", timestamp_seconds=0, bullets=[])],
        concepts=[Concept(name="Causal Inference", definition="Comparing interventions.")],
        claims=[Claim(text="Graphs help causal inference.", evidence="Graphs help explain causal inference claims.")],
        limitations=[],
        open_questions=["How do we validate causal claims?"],
        related_topics=["Research Methods"],
        model="test-model",
    )

    extraction = build_deterministic_graph_extraction(metadata, transcript, analysis)

    assert extraction["schema_version"] == "graph-extraction-v1"
    labels = {node["label"] for node in extraction["nodes"]}
    assert {"Video", "Channel", "Topic", "Concept", "Claim", "Evidence", "Question", "SourceSegment"} <= labels
    segment_nodes = [node for node in extraction["nodes"] if node["label"] == "SourceSegment"]
    assert segment_nodes[0]["id"] == "segment:abc123:0"
    assert segment_nodes[0]["end"] == 5

    relationships = extraction["relationships"]
    assert all(rel["type"] in ALLOWED_RELATIONSHIP_TYPES for rel in relationships)
    assert any(rel["type"] == "HAS_SEGMENT" and rel["confidence"] == 1.0 for rel in relationships)
    assert any(rel["type"] == "EVIDENCE_FOR" and rel["source_segment_id"] == "segment:abc123:1" for rel in relationships)
    assert extraction["quality_report"]["node_count_by_label"]["SourceSegment"] == 2
    assert extraction["quality_report"]["orphan_claims_without_evidence"] == []


def test_deep_graph_chunks_merge_nodes_relations_and_rejections():
    metadata = VideoMetadata(video_id="abc123", canonical_url="https://youtu.be/abc123", title="Talk")
    transcript = Transcript(
        video_id="abc123",
        source="youtube",
        segments=[TranscriptSegment(0, 10, "Graph algorithms use causal inference tools.")],
    )
    analysis = Analysis(
        language="en",
        cleaned_title="Talk",
        abstract="Abstract",
        tags=[],
        outline=[],
        sections=[],
        concepts=[Concept(name="Graph Algorithms", definition="Algorithms on graphs.")],
        claims=[Claim(text="Graph algorithms help causal inference.", evidence="Graph algorithms use causal inference tools.")],
        limitations=[],
        open_questions=[],
        related_topics=[],
    )
    base = build_deterministic_graph_extraction(metadata, transcript, analysis)
    chunks = [
        {
            "chunk_id": "chunk-1",
            "video_id": "abc123",
            "timestamp_start": 0,
            "timestamp_end": 10,
            "concepts": [
                {
                    "local_id": "c1",
                    "name": "Graph Algorithms",
                    "definition": "Graph algorithms from the model.",
                    "aliases": ["Graph Algos"],
                    "confidence": 0.9,
                    "evidence_quote": "Graph algorithms",
                    "timestamp_start": 0,
                    "timestamp_end": 3,
                },
                {
                    "local_id": "c2",
                    "name": "Causal Inference",
                    "definition": "Reasoning about causes.",
                    "aliases": [],
                    "confidence": 0.9,
                    "evidence_quote": "causal inference",
                    "timestamp_start": 3,
                    "timestamp_end": 6,
                },
            ],
            "claims": [
                {
                    "local_id": "cl1",
                    "text": "Graph algorithms help causal inference.",
                    "paraphrased_evidence": "Graph algorithms use causal inference tools.",
                    "confidence": 0.92,
                    "timestamp_start": 0,
                    "timestamp_end": 10,
                },
                {
                    "local_id": "cl2",
                    "text": "Causal tools improve graph analysis.",
                    "paraphrased_evidence": "causal inference tools",
                    "confidence": 0.91,
                    "timestamp_start": 2,
                    "timestamp_end": 9,
                },
            ],
            "evidence": [
                {
                    "local_id": "e1",
                    "text": "Graph algorithms use causal inference tools.",
                    "kind": "quote",
                    "quote": "Graph algorithms use causal inference tools.",
                    "supports_claim_ids": ["cl1"],
                    "confidence": 0.95,
                    "timestamp_start": 0,
                    "timestamp_end": 10,
                }
            ],
            "concept_relations": [
                {
                    "from_concept_id": "c1",
                    "to_concept_id": "c2",
                    "type": "RELATED_TO",
                    "confidence": 0.8,
                    "evidence_quote": "Graph algorithms use causal inference tools.",
                    "timestamp_start": 0,
                    "timestamp_end": 10,
                },
                {
                    "from_concept_id": "c1",
                    "to_concept_id": "missing",
                    "type": "CAUSES",
                    "confidence": 0.8,
                    "evidence_quote": "missing endpoint",
                    "timestamp_start": 0,
                    "timestamp_end": 1,
                },
                {
                    "from_concept_id": "c1",
                    "to_concept_id": "c2",
                    "type": "CONTRASTS_WITH",
                    "confidence": 0.2,
                    "evidence_quote": "weak",
                    "timestamp_start": 0,
                    "timestamp_end": 1,
                },
            ],
            "claim_relations": [
                {
                    "from_claim_id": "cl1",
                    "to_claim_id": "cl2",
                    "type": "SUPPORTS",
                    "confidence": 0.85,
                    "evidence_quote": "supports",
                    "timestamp_start": 0,
                    "timestamp_end": 10,
                }
            ],
            "methods": [
                {
                    "local_id": "m1",
                    "name": "Graph Method",
                    "description": "Uses graph tooling.",
                    "uses_tool_ids": ["t1"],
                    "applies_to_concept_ids": ["c1"],
                    "confidence": 0.9,
                    "evidence_quote": "Graph algorithms use causal inference tools.",
                    "timestamp_start": 0,
                    "timestamp_end": 10,
                }
            ],
            "tools": [
                {
                    "local_id": "t1",
                    "name": "Causal Tool",
                    "description": "Tooling.",
                    "confidence": 0.9,
                    "evidence_quote": "tools",
                    "timestamp_start": 0,
                    "timestamp_end": 10,
                }
            ],
            "metrics": [
                {
                    "local_id": "mt1",
                    "name": "Accuracy",
                    "definition": "Correctness.",
                    "measured_concept_ids": ["c1"],
                    "confidence": 0.9,
                    "evidence_quote": "accuracy",
                    "timestamp_start": 0,
                    "timestamp_end": 10,
                }
            ],
            "risks": [
                {
                    "local_id": "r1",
                    "text": "Overfitting graph claims",
                    "related_concept_ids": ["c1"],
                    "confidence": 0.9,
                    "evidence_quote": "risk",
                    "timestamp_start": 0,
                    "timestamp_end": 10,
                }
            ],
            "questions": [
                {
                    "local_id": "q1",
                    "text": "How should graph algorithms be validated?",
                    "about_concept_ids": ["c1"],
                    "confidence": 0.9,
                    "evidence_quote": "validated",
                    "timestamp_start": 0,
                    "timestamp_end": 10,
                }
            ],
        }
    ]

    merged = merge_deep_graph_chunks(metadata, transcript, base, chunks, min_confidence=0.55, model="gemma")

    nodes = {node["id"]: node for node in merged["nodes"]}
    relationships = merged["relationships"]
    assert merged["chunks"] == chunks
    assert nodes["concept:graph-algorithms"]["aliases"] == ["Graph Algorithms", "Graph Algos"]
    assert "concept:causal-inference" in nodes
    assert any(node["label"] == "Method" for node in nodes.values())
    assert any(node["label"] == "Tool" for node in nodes.values())
    assert any(node["label"] == "Metric" for node in nodes.values())
    assert any(node["label"] == "Risk" for node in nodes.values())
    assert any(
        rel["type"] == "RELATED_TO"
        and rel["reliability"] == "llm_extracted"
        and rel["extraction_method"] == "llm_deep_graph"
        for rel in relationships
    )
    assert any(rel["type"] == "SUPPORTS" for rel in relationships)
    assert any(rel["type"] == "USES" for rel in relationships)
    assert any(rel["type"] == "APPLIES_TO" for rel in relationships)
    assert all("id" in rel and rel["audit_status"] == "accepted" for rel in relationships)
    assert merged["quality_report"]["rejected_relationships"]
    assert merged["quality_report"]["low_confidence_extractions"]
    assert merged["quality_report"]["chunks_total"] == 1
    assert merged["quality_report"]["chunks_succeeded"] == 1
    assert merged["quality_report"]["chunks_failed"] == 0
    assert merged["quality_report"]["relationships_accepted"] > 0

    merged_again = merge_deep_graph_chunks(metadata, transcript, base, chunks, min_confidence=0.55, model="gemma")
    assert {rel["id"] for rel in merged["relationships"]} == {rel["id"] for rel in merged_again["relationships"]}


def test_deep_graph_duplicate_local_ids_are_rejected_without_miswiring():
    metadata = VideoMetadata(video_id="abc123", canonical_url="https://youtu.be/abc123", title="Talk")
    transcript = Transcript(
        video_id="abc123",
        source="youtube",
        segments=[TranscriptSegment(0, 10, "Support zone causes breakout.")],
    )
    analysis = Analysis(
        language="en",
        cleaned_title="Talk",
        abstract="Abstract",
        tags=[],
        outline=[],
        sections=[],
        concepts=[],
        claims=[],
        limitations=[],
        open_questions=[],
        related_topics=[],
    )
    base = build_deterministic_graph_extraction(metadata, transcript, analysis)
    chunks = [
        {
            "chunk_id": "chunk-1",
            "video_id": "abc123",
            "timestamp_start": 0,
            "timestamp_end": 10,
            "concepts": [
                {
                    "local_id": "x1",
                    "name": "Support Zone",
                    "definition": "A support area.",
                    "aliases": [],
                    "confidence": 0.9,
                    "evidence_quote": "Support zone",
                    "timestamp_start": 0,
                    "timestamp_end": 2,
                }
            ],
            "claims": [
                {
                    "local_id": "x1",
                    "text": "Support causes breakout.",
                    "paraphrased_evidence": "Support zone causes breakout.",
                    "confidence": 0.9,
                    "timestamp_start": 0,
                    "timestamp_end": 10,
                }
            ],
            "evidence": [],
            "concept_relations": [],
            "claim_relations": [],
            "methods": [],
            "tools": [],
            "metrics": [],
            "risks": [],
            "questions": [],
        }
    ]

    merged = merge_deep_graph_chunks(metadata, transcript, base, chunks, min_confidence=0.55, model="gemma")

    assert any(item["reason"] == "duplicate_local_id" for item in merged["quality_report"]["rejected_nodes"])
