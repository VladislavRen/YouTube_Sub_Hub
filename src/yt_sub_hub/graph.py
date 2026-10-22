from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from yt_sub_hub.models import Analysis, Claim, Concept, VideoMetadata, slugify

NODE_LABELS = {
    "Video",
    "Channel",
    "Topic",
    "Concept",
    "Claim",
    "Evidence",
    "SourceSegment",
    "Method",
    "Tool",
    "Metric",
    "Risk",
    "Question",
    "VisualFrame",
    "VisualObservation",
    "ChartPattern",
    "ChartLevel",
    "Indicator",
    "ExternalSource",
    "ResearchFinding",
    "ResearchQuery",
    "Citation",
    "ResearchTopic",
    "YouTubeVideoCandidate",
    "YouTubeChannelCandidate",
    "DiscoveryQuery",
    "DiscoverySeed",
}

RELATIONSHIP_TYPES = {
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
    "FOUND_BY_QUERY",
    "CITES",
    "EXPANDS",
    "RELATED_EXTERNAL",
    "SOURCE_FOR",
    "MENTIONS_EXTERNAL",
    "DISCOVERED_BY_QUERY",
    "SUGGESTED_BY_RESEARCH_FINDING",
    "RELATED_TO_VIDEO",
    "CANDIDATE_FROM_SOURCE",
    "CHANNEL_CANDIDATE_FOR_TOPIC",
}


@dataclass(frozen=True)
class GraphPayload:
    nodes: list[dict[str, Any]]
    relationships: list[dict[str, Any]]


def _concept_node(concept: Concept) -> dict[str, Any]:
    return {
        "id": f"concept:{concept.slug}",
        "label": "Concept",
        "slug": concept.slug,
        "name": concept.name,
        "definition": concept.definition,
    }


def _claim_node(video_id: str, claim: Claim) -> dict[str, Any]:
    return {
        "id": f"claim:{video_id}:{claim.slug}",
        "label": "Claim",
        "text": claim.text,
        "evidence": claim.evidence,
    }


def build_graph_payload(metadata: VideoMetadata, analysis: Analysis) -> GraphPayload:
    nodes: dict[str, dict[str, Any]] = {
        f"video:{metadata.video_id}": {
            "id": f"video:{metadata.video_id}",
            "label": "Video",
            "video_id": metadata.video_id,
            "title": metadata.title,
            "url": metadata.canonical_url,
        }
    }
    relationships: list[dict[str, Any]] = []

    if metadata.channel:
        channel_id = f"channel:{slugify(metadata.channel)}"
        nodes[channel_id] = {"id": channel_id, "label": "Channel", "name": metadata.channel}
        relationships.append({"from": channel_id, "to": f"video:{metadata.video_id}", "type": "PUBLISHED"})

    for topic in analysis.related_topics + analysis.tags:
        topic_id = f"topic:{slugify(topic)}"
        nodes[topic_id] = {"id": topic_id, "label": "Topic", "name": topic}
        relationships.append({"from": f"video:{metadata.video_id}", "to": topic_id, "type": "PART_OF"})

    for concept in analysis.concepts:
        node = _concept_node(concept)
        nodes[node["id"]] = nodes.get(node["id"], node)
        relationships.append({"from": f"video:{metadata.video_id}", "to": node["id"], "type": "MENTIONS"})

    for claim in analysis.claims:
        node = _claim_node(metadata.video_id, claim)
        nodes[node["id"]] = node
        relationships.append({"from": f"video:{metadata.video_id}", "to": node["id"], "type": "EXPLAINS"})

    return GraphPayload(nodes=list(nodes.values()), relationships=relationships)


class Neo4jGraphStore:
    def __init__(self, uri: str, user: str, password: str) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError("neo4j package is required for graph persistence") from exc
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self._driver.close()

    def verify_connectivity(self) -> None:
        self._driver.verify_connectivity()

    def ensure_schema(self) -> None:
        statements = [
            "CREATE CONSTRAINT video_id IF NOT EXISTS FOR (n:Video) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT channel_id IF NOT EXISTS FOR (n:Channel) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT topic_id IF NOT EXISTS FOR (n:Topic) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (n:Concept) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (n:Claim) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT evidence_id IF NOT EXISTS FOR (n:Evidence) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT segment_id IF NOT EXISTS FOR (n:SourceSegment) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT method_id IF NOT EXISTS FOR (n:Method) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT tool_id IF NOT EXISTS FOR (n:Tool) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT metric_id IF NOT EXISTS FOR (n:Metric) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT risk_id IF NOT EXISTS FOR (n:Risk) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT question_id IF NOT EXISTS FOR (n:Question) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT visual_frame_id IF NOT EXISTS FOR (n:VisualFrame) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT visual_observation_id IF NOT EXISTS FOR (n:VisualObservation) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT chart_pattern_id IF NOT EXISTS FOR (n:ChartPattern) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT chart_level_id IF NOT EXISTS FOR (n:ChartLevel) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT indicator_id IF NOT EXISTS FOR (n:Indicator) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT external_source_id IF NOT EXISTS FOR (n:ExternalSource) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT research_finding_id IF NOT EXISTS FOR (n:ResearchFinding) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT research_query_id IF NOT EXISTS FOR (n:ResearchQuery) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT citation_id IF NOT EXISTS FOR (n:Citation) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT research_topic_id IF NOT EXISTS FOR (n:ResearchTopic) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT youtube_video_candidate_id IF NOT EXISTS FOR (n:YouTubeVideoCandidate) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT youtube_channel_candidate_id IF NOT EXISTS FOR (n:YouTubeChannelCandidate) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT discovery_query_id IF NOT EXISTS FOR (n:DiscoveryQuery) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT discovery_seed_id IF NOT EXISTS FOR (n:DiscoverySeed) REQUIRE n.id IS UNIQUE",
            "CREATE INDEX concept_slug IF NOT EXISTS FOR (n:Concept) ON (n.slug)",
            "CREATE INDEX concept_name IF NOT EXISTS FOR (n:Concept) ON (n.canonical_name)",
            "CREATE INDEX claim_hash IF NOT EXISTS FOR (n:Claim) ON (n.claim_hash)",
            "CREATE INDEX segment_video_time IF NOT EXISTS FOR (n:SourceSegment) ON (n.video_id, n.start)",
            "CREATE INDEX research_finding_run IF NOT EXISTS FOR (n:ResearchFinding) ON (n.research_run_id)",
            "CREATE INDEX external_source_url IF NOT EXISTS FOR (n:ExternalSource) ON (n.source_url)",
            "CREATE INDEX youtube_video_candidate_video_id IF NOT EXISTS FOR (n:YouTubeVideoCandidate) ON (n.youtube_video_id)",
            "CREATE INDEX youtube_channel_candidate_channel_id IF NOT EXISTS FOR (n:YouTubeChannelCandidate) ON (n.youtube_channel_id)",
        ]
        statements.extend(
            f"CREATE INDEX rel_{rel_type.lower()}_source_video IF NOT EXISTS "
            f"FOR ()-[r:{rel_type}]-() ON (r.source_video_id)"
            for rel_type in sorted(RELATIONSHIP_TYPES)
        )
        with self._driver.session() as session:
            for statement in statements:
                session.run(statement)

    def upsert_payload(self, payload: GraphPayload) -> None:
        self.ensure_schema()
        with self._driver.session() as session:
            for node in payload.nodes:
                session.execute_write(self._merge_node, node)
            for relationship in payload.relationships:
                session.execute_write(self._merge_relationship, relationship)

    def replace_video_core_payload(self, video_id: str, payload: GraphPayload) -> None:
        """Replace core edges atomically while retaining separately owned enrichment."""
        self.ensure_schema()
        with self._driver.session() as session:
            session.execute_write(self._replace_video_core_payload, video_id, payload)

    @staticmethod
    def _replace_video_core_payload(tx, video_id: str, payload: GraphPayload) -> None:
        tx.run(
            "MATCH ()-[r]->() "
            "WHERE r.source_video_id = $video_id "
            "AND r.research_run_id IS NULL AND r.discovery_run_id IS NULL "
            "DELETE r",
            video_id=video_id,
        )
        for node in payload.nodes:
            Neo4jGraphStore._merge_node(tx, node)
        for relationship in payload.relationships:
            Neo4jGraphStore._merge_relationship(tx, relationship)

    def clear(self) -> None:
        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def delete_video_relationships(self, video_id: str) -> None:
        with self._driver.session() as session:
            session.run(
                "MATCH ()-[r]->() "
                "WHERE r.source_video_id = $video_id "
                "DELETE r",
                video_id=video_id,
            )

    def delete_research_run(self, research_run_id: str) -> None:
        with self._driver.session() as session:
            session.run(
                "MATCH ()-[r]->() WHERE r.research_run_id = $research_run_id DELETE r",
                research_run_id=research_run_id,
            )
            session.run(
                "MATCH (n) WHERE n.research_run_id = $research_run_id AND NOT n:Video DETACH DELETE n",
                research_run_id=research_run_id,
            )

    def delete_research_for_video(self, video_id: str) -> None:
        with self._driver.session() as session:
            session.run(
                "MATCH ()-[r]->() "
                "WHERE r.source_video_id = $video_id AND r.research_run_id IS NOT NULL "
                "DELETE r",
                video_id=video_id,
            )
            session.run(
                "MATCH (n) "
                "WHERE n.source_video_id = $video_id AND n.research_run_id IS NOT NULL "
                "DETACH DELETE n",
                video_id=video_id,
            )

    def delete_discovery_for_video(self, video_id: str) -> None:
        with self._driver.session() as session:
            session.run(
                "MATCH ()-[r]->() "
                "WHERE r.source_video_id = $video_id AND r.discovery_run_id IS NOT NULL "
                "DELETE r",
                video_id=video_id,
            )
            session.run(
                "MATCH (n) "
                "WHERE n.source_video_id = $video_id AND n.discovery_run_id IS NOT NULL "
                "DETACH DELETE n",
                video_id=video_id,
            )

    def discovery_consistency(self, payload: GraphPayload) -> dict[str, Any]:
        expected_nodes = {node["id"] for node in payload.nodes if node.get("discovery_run_id")}
        expected_relationships = {
            relationship.get("id") or f"{relationship['from']}|{relationship['type']}|{relationship['to']}"
            for relationship in payload.relationships
        }
        discovery_run_ids = {
            value
            for item in [*payload.nodes, *payload.relationships]
            for value in [item.get("discovery_run_id")]
            if value
        }
        if not discovery_run_ids:
            return {"checked": False, "missing": None, "extra": None}
        with self._driver.session() as session:
            node_rows = session.run(
                "MATCH (n) WHERE n.discovery_run_id IN $run_ids RETURN n.id AS id",
                run_ids=list(discovery_run_ids),
            )
            actual_nodes = {str(record["id"]) for record in node_rows}
            rel_rows = session.run(
                "MATCH ()-[r]->() WHERE r.discovery_run_id IN $run_ids RETURN r.id AS id",
                run_ids=list(discovery_run_ids),
            )
            actual_relationships = {str(record["id"]) for record in rel_rows}
        missing = len((expected_nodes - actual_nodes) | (expected_relationships - actual_relationships))
        extra = len((actual_nodes - expected_nodes) | (actual_relationships - expected_relationships))
        return {"checked": True, "missing": missing, "extra": extra}

    def research_consistency(self, payload: GraphPayload) -> dict[str, Any]:
        expected_nodes = {node["id"] for node in payload.nodes if node.get("research_run_id")}
        expected_relationships = {
            relationship.get("id") or f"{relationship['from']}|{relationship['type']}|{relationship['to']}"
            for relationship in payload.relationships
        }
        research_run_ids = {
            value
            for item in [*payload.nodes, *payload.relationships]
            for value in [item.get("research_run_id")]
            if value
        }
        if not research_run_ids:
            return {"checked": False, "missing": None, "extra": None}
        with self._driver.session() as session:
            node_rows = session.run(
                "MATCH (n) WHERE n.research_run_id IN $run_ids RETURN n.id AS id",
                run_ids=list(research_run_ids),
            )
            actual_nodes = {str(record["id"]) for record in node_rows}
            rel_rows = session.run(
                "MATCH ()-[r]->() WHERE r.research_run_id IN $run_ids RETURN r.id AS id",
                run_ids=list(research_run_ids),
            )
            actual_relationships = {str(record["id"]) for record in rel_rows}
        missing = len((expected_nodes - actual_nodes) | (expected_relationships - actual_relationships))
        extra = len((actual_nodes - expected_nodes) | (actual_relationships - expected_relationships))
        return {"checked": True, "missing": missing, "extra": extra}

    def list_concepts(self, q: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        query = (
            "MATCH (c:Concept) "
            "WHERE $q IS NULL OR toLower(c.name) CONTAINS toLower($q) OR toLower(c.canonical_name) CONTAINS toLower($q) "
            "RETURN c.id AS id, c.name AS name, c.slug AS slug, c.definition AS definition "
            "ORDER BY c.name LIMIT $limit"
        )
        with self._driver.session() as session:
            return [dict(record) for record in session.run(query, q=q, limit=limit)]

    def get_neighbors(
        self,
        node_id: str,
        depth: int = 1,
        relation: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        max_depth = max(1, min(depth, 3))
        if relation is not None and not re.fullmatch(r"[A-Z_]+", relation):
            raise ValueError("relation must be an uppercase Neo4j relationship type")
        rel_filter = "" if relation is None else f":{relation}"
        query = (
            f"MATCH (c {{id: $id}})-[r{rel_filter}*1..{max_depth}]-(n) "
            "RETURN n.id AS id, labels(n) AS labels, "
            "coalesce(n.name, n.title, n.text) AS name, "
            "[rel IN r | type(rel)] AS relationship_types "
            "LIMIT $limit"
        )
        with self._driver.session() as session:
            return [dict(record) for record in session.run(query, id=node_id, limit=limit)]

    def find_path(self, from_id: str, to_id: str, max_depth: int = 4) -> list[dict[str, Any]]:
        depth = max(1, min(max_depth, 6))
        query = (
            f"MATCH path = shortestPath((a {{id: $from_id}})-[*1..{depth}]-(b {{id: $to_id}})) "
            "RETURN [n IN nodes(path) | {id: n.id, labels: labels(n), name: coalesce(n.name, n.title, n.text)}] AS nodes, "
            "[r IN relationships(path) | {id: r.id, type: type(r), confidence: r.confidence}] AS relationships"
        )
        with self._driver.session() as session:
            return [dict(record) for record in session.run(query, from_id=from_id, to_id=to_id)]

    def get_claims(self, concept: str | None = None, min_confidence: float = 0.0, limit: int = 100) -> list[dict[str, Any]]:
        if concept:
            query = (
                "MATCH (claim:Claim)-[r:ABOUT]->(concept:Concept) "
                "WHERE (concept.id = $concept OR concept.slug = $concept OR toLower(concept.name) = toLower($concept)) "
                "AND coalesce(r.confidence, 0.0) >= $min_confidence "
                "RETURN claim.id AS id, claim.text AS text, claim.status AS status, "
                "concept.id AS concept_id, concept.name AS concept_name, r.confidence AS confidence "
                "ORDER BY confidence DESC LIMIT $limit"
            )
            params = {"concept": concept, "min_confidence": min_confidence, "limit": limit}
        else:
            query = (
                "MATCH (claim:Claim) "
                "RETURN claim.id AS id, claim.text AS text, claim.status AS status "
                "ORDER BY claim.text LIMIT $limit"
            )
            params = {"limit": limit}
        with self._driver.session() as session:
            return [dict(record) for record in session.run(query, **params)]

    @staticmethod
    def _merge_node(tx, node: dict[str, Any]) -> None:
        label = node["label"]
        if label not in NODE_LABELS:
            raise ValueError(f"Unsupported graph node label: {label}")
        props = {key: value for key, value in node.items() if key != "label"}
        tx.run(f"MERGE (n:{label} {{id: $id}}) SET n += $props", id=node["id"], props=props)

    @staticmethod
    def _merge_relationship(tx, relationship: dict[str, Any]) -> None:
        rel_type = relationship["type"]
        if rel_type not in RELATIONSHIP_TYPES:
            raise ValueError(f"Unsupported graph relationship type: {rel_type}")
        props = {
            key: value
            for key, value in relationship.items()
            if key not in {"from", "to", "type"} and value is not None
        }
        rel_id = props.get("id") or f"{relationship['from']}|{rel_type}|{relationship['to']}"
        tx.run(
            (
                "MATCH (a {id: $from_id}) "
                "MATCH (b {id: $to_id}) "
                f"MERGE (a)-[r:{rel_type} {{id: $rel_id}}]->(b) "
                "SET r += $props"
            ),
            from_id=relationship["from"],
            to_id=relationship["to"],
            rel_id=rel_id,
            props=props,
        )
