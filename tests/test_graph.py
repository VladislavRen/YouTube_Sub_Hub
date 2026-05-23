from yt_sub_hub.graph import RELATIONSHIP_TYPES
from yt_sub_hub.graph import build_graph_payload
from yt_sub_hub.graph import GraphPayload
from yt_sub_hub.graph import Neo4jGraphStore
from yt_sub_hub.models import Analysis, Claim, Concept, NoteSection, VideoMetadata


def test_build_graph_payload_deduplicates_concepts_by_slug():
    metadata = VideoMetadata(
        video_id="abc123",
        canonical_url="https://www.youtube.com/watch?v=abc123",
        title="Talk",
        channel="Channel",
    )
    analysis = Analysis(
        language="en",
        cleaned_title="Talk",
        abstract="Abstract",
        tags=["science"],
        outline=[],
        sections=[NoteSection(heading="Intro", summary="Summary", timestamp_seconds=0, bullets=[])],
        concepts=[
            Concept(name="Causal Inference", definition="Definition A"),
            Concept(name="causal inference", definition="Definition B"),
        ],
        claims=[Claim(text="Claim", evidence="Evidence")],
        limitations=[],
        open_questions=[],
        related_topics=["Research Methods"],
    )

    payload = build_graph_payload(metadata, analysis)

    concept_nodes = [node for node in payload.nodes if node["label"] == "Concept"]
    assert len(concept_nodes) == 1
    assert concept_nodes[0]["id"] == "concept:causal-inference"
    assert any(rel["type"] == "MENTIONS" for rel in payload.relationships)


def test_merge_relationship_persists_rich_properties():
    class RecordingTx:
        def __init__(self):
            self.calls = []

        def run(self, query, **kwargs):
            self.calls.append((query, kwargs))

    tx = RecordingTx()

    Neo4jGraphStore._merge_relationship(
        tx,
        {
            "id": "rel123",
            "from": "video:abc",
            "to": "concept:graph",
            "type": "MENTIONS",
            "confidence": 0.9,
            "source_video_id": "abc",
            "audit_status": "accepted",
        },
    )

    query, params = tx.calls[0]
    assert "MERGE (a)-[r:MENTIONS {id: $rel_id}]->(b)" in query
    assert params["rel_id"] == "rel123"
    assert params["props"]["confidence"] == 0.9
    assert params["props"]["source_video_id"] == "abc"


def test_ensure_schema_uses_typed_relationship_indexes(monkeypatch):
    statements = []

    class RecordingSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def run(self, statement):
            statements.append(statement)

    class RecordingDriver:
        def session(self):
            return RecordingSession()

    store = object.__new__(Neo4jGraphStore)
    store._driver = RecordingDriver()

    store.ensure_schema()

    assert not any("FOR ()-[r]-()" in statement for statement in statements)
    assert any("FOR ()-[r:MENTIONS]-() ON (r.source_video_id)" in statement for statement in statements)
    assert any("FOR ()-[r:EVIDENCE_FOR]-() ON (r.source_video_id)" in statement for statement in statements)


def test_ensure_schema_indexes_source_video_id_for_all_relationship_types():
    statements = []

    class RecordingSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def run(self, statement):
            statements.append(statement)

    class RecordingDriver:
        def session(self):
            return RecordingSession()

    store = object.__new__(Neo4jGraphStore)
    store._driver = RecordingDriver()

    store.ensure_schema()

    indexed_types = {
        statement.split("[r:", 1)[1].split("]", 1)[0]
        for statement in statements
        if "ON (r.source_video_id)" in statement and "[r:" in statement
    }
    assert RELATIONSHIP_TYPES <= indexed_types


def test_delete_video_relationships_deletes_only_relationships_for_source_video():
    calls = []

    class RecordingSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def run(self, statement, **kwargs):
            calls.append((statement, kwargs))

    class RecordingDriver:
        def session(self):
            return RecordingSession()

    store = object.__new__(Neo4jGraphStore)
    store._driver = RecordingDriver()

    store.delete_video_relationships("abc123")

    assert calls == [
        (
            "MATCH ()-[r]->() "
            "WHERE r.source_video_id = $video_id "
            "DELETE r",
            {"video_id": "abc123"},
        )
    ]
    assert "DETACH DELETE" not in calls[0][0]


def test_delete_research_for_video_deletes_only_research_owned_graph():
    calls = []

    class RecordingSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def run(self, statement, **kwargs):
            calls.append((statement, kwargs))

    class RecordingDriver:
        def session(self):
            return RecordingSession()

    store = object.__new__(Neo4jGraphStore)
    store._driver = RecordingDriver()

    store.delete_research_for_video("abc123")

    assert calls == [
        (
            "MATCH ()-[r]->() "
            "WHERE r.source_video_id = $video_id AND r.research_run_id IS NOT NULL "
            "DELETE r",
            {"video_id": "abc123"},
        ),
        (
            "MATCH (n) "
            "WHERE n.source_video_id = $video_id AND n.research_run_id IS NOT NULL "
            "DETACH DELETE n",
            {"video_id": "abc123"},
        ),
    ]


def test_research_consistency_ignores_shared_video_node_without_research_run_id():
    payload = GraphPayload(
        nodes=[
            {"id": "video:abc123", "label": "Video", "video_id": "abc123"},
            {"id": "research_finding:run-1:1", "label": "ResearchFinding", "research_run_id": "run-1"},
        ],
        relationships=[
            {
                "id": "research-rel-run-1",
                "from": "video:abc123",
                "to": "research_finding:run-1:1",
                "type": "RELATED_EXTERNAL",
                "research_run_id": "run-1",
            }
        ],
    )

    class RecordingSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def run(self, statement, **kwargs):
            if "MATCH (n)" in statement:
                return [{"id": "research_finding:run-1:1"}]
            return [{"id": "research-rel-run-1"}]

    class RecordingDriver:
        def session(self):
            return RecordingSession()

    store = object.__new__(Neo4jGraphStore)
    store._driver = RecordingDriver()

    assert store.research_consistency(payload) == {"checked": True, "missing": 0, "extra": 0}


def test_merge_node_rejects_invalid_label():
    class RecordingTx:
        def run(self, query, **kwargs):
            raise AssertionError("Cypher should not run for invalid labels")

    try:
        Neo4jGraphStore._merge_node(RecordingTx(), {"id": "bad", "label": "Bad`) DETACH DELETE n //"})
    except ValueError as exc:
        assert "Unsupported graph node label" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_merge_relationship_rejects_invalid_type():
    class RecordingTx:
        def run(self, query, **kwargs):
            raise AssertionError("Cypher should not run for invalid relationship types")

    try:
        Neo4jGraphStore._merge_relationship(
            RecordingTx(),
            {"id": "bad", "from": "a", "to": "b", "type": "BAD`) DELETE r //"},
        )
    except ValueError as exc:
        assert "Unsupported graph relationship type" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
