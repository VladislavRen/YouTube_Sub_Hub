from __future__ import annotations

import importlib.util
import json
import xml.etree.ElementTree as ET

from yt_sub_hub.graph_export import export_graph_json, export_graphml
from yt_sub_hub.library import LibraryStore


def _write_extraction(root, video_id: str, concept_id: str) -> None:
    video_dir = root / "videos" / video_id
    video_dir.mkdir(parents=True)
    (video_dir / "graph_extraction.json").write_text(
        json.dumps(
            {
                "schema_version": "graph-extraction-v1",
                "video_id": video_id,
                "generated_from": {},
                "chunks": [],
                "nodes": [
                    {"id": f"video:{video_id}", "label": "Video", "title": f"Video {video_id}"},
                    {"id": concept_id, "label": "Concept", "name": "Graph"},
                ],
                "relationships": [
                    {
                        "id": f"rel:{video_id}",
                        "from": f"video:{video_id}",
                        "to": concept_id,
                        "type": "MENTIONS",
                        "confidence": 0.9,
                        "source_video_id": video_id,
                        "reliability": "llm_extracted",
                        "extraction_method": "llm_deep_graph",
                        "chunk_id": "chunk-1",
                        "model": "gpt-5-mini",
                        "prompt_version": "deep-graph-v1",
                        "audit_status": "accepted",
                    }
                ],
                "quality_report": {},
            }
        ),
        encoding="utf-8",
    )


def test_graph_export_filters_by_video_id(tmp_path):
    _write_extraction(tmp_path, "abc123", "concept:graph")
    _write_extraction(tmp_path, "xyz789", "concept:other")
    store = LibraryStore(tmp_path)

    output = export_graph_json(store, video_id="abc123")
    data = json.loads(output.read_text(encoding="utf-8"))

    assert output.name == "graph-abc123.json"
    assert {node["id"] for node in data["nodes"]} == {"video:abc123", "concept:graph"}
    assert data["relationships"][0]["source_video_id"] == "abc123"


def test_graphml_export_declares_keys_and_is_parseable(tmp_path):
    _write_extraction(tmp_path, "abc123", "concept:graph")
    store = LibraryStore(tmp_path)

    output = export_graphml(store, video_id="abc123")
    tree = ET.parse(output)
    root = tree.getroot()
    namespace = {"g": "http://graphml.graphdrawing.org/xmlns"}
    key_ids = {element.attrib["id"] for element in root.findall("g:key", namespace)}
    data_keys = {element.attrib["key"] for element in root.findall(".//g:data", namespace)}

    assert {"node_label", "node_name", "edge_type", "confidence", "source_video_id"} <= key_ids
    assert data_keys <= key_ids

    if importlib.util.find_spec("networkx") is not None:
        import networkx as nx

        graph = nx.read_graphml(output)
        assert "video:abc123" in graph.nodes


def test_graphml_export_preserves_llm_provenance_keys(tmp_path):
    _write_extraction(tmp_path, "abc123", "concept:graph")
    store = LibraryStore(tmp_path)

    output = export_graphml(store, video_id="abc123")
    tree = ET.parse(output)
    root = tree.getroot()
    namespace = {"g": "http://graphml.graphdrawing.org/xmlns"}
    key_ids = {element.attrib["id"] for element in root.findall("g:key", namespace)}
    edge_data = {
        element.attrib["key"]: element.text
        for element in root.findall(".//g:edge/g:data", namespace)
    }

    expected_keys = {
        "reliability",
        "extraction_method",
        "chunk_id",
        "model",
        "prompt_version",
        "audit_status",
    }
    assert expected_keys <= key_ids
    assert expected_keys <= edge_data.keys()
    assert edge_data["reliability"] == "llm_extracted"
    assert edge_data["extraction_method"] == "llm_deep_graph"
    assert edge_data["chunk_id"] == "chunk-1"
    assert edge_data["model"] == "gpt-5-mini"
    assert edge_data["prompt_version"] == "deep-graph-v1"
    assert edge_data["audit_status"] == "accepted"
