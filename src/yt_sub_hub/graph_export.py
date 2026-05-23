from __future__ import annotations

import html
import hashlib
from pathlib import Path
from typing import Any

from yt_sub_hub.library import LibraryStore

LLM_EDGE_PROVENANCE_KEYS = (
    "reliability",
    "extraction_method",
    "chunk_id",
    "model",
    "prompt_version",
    "audit_status",
)


def collect_local_graph(store: LibraryStore, video_id: str | None = None) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    relationships: dict[str, dict[str, Any]] = {}
    if video_id is not None:
        extraction = store.load_graph_extraction(video_id)
        if extraction is None:
            raise FileNotFoundError(f"graph extraction not found for {video_id}")
        extractions = [extraction]
    else:
        extractions = store.load_graph_extraction_files()
    for extraction in extractions:
        for node in extraction.get("nodes", []):
            nodes.setdefault(node["id"], node)
        for relationship in extraction.get("relationships", []):
            relationships.setdefault(relationship.get("id") or f"{relationship['from']}|{relationship['type']}|{relationship['to']}", relationship)
    return {"nodes": list(nodes.values()), "relationships": list(relationships.values())}


def export_graph_json(store: LibraryStore, output: Path | None = None, video_id: str | None = None) -> Path:
    store.ensure_layout()
    default_name = f"graph-{video_id}.json" if video_id else "graph.json"
    output_path = output or store.root / "graph" / "exports" / default_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    LibraryStore._write_json(output_path, collect_local_graph(store, video_id=video_id))
    return output_path


def export_graphml(store: LibraryStore, output: Path | None = None, video_id: str | None = None) -> Path:
    store.ensure_layout()
    default_name = f"graph-{video_id}.graphml" if video_id else "graph.graphml"
    output_path = output or store.root / "graph" / "exports" / default_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph = collect_local_graph(store, video_id=video_id)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
        '  <key id="node_label" for="node" attr.name="label" attr.type="string"/>',
        '  <key id="node_name" for="node" attr.name="name" attr.type="string"/>',
        '  <key id="edge_type" for="edge" attr.name="type" attr.type="string"/>',
        '  <key id="confidence" for="edge" attr.name="confidence" attr.type="double"/>',
        '  <key id="source_video_id" for="edge" attr.name="source_video_id" attr.type="string"/>',
        *[
            f'  <key id="{key}" for="edge" attr.name="{key}" attr.type="string"/>'
            for key in LLM_EDGE_PROVENANCE_KEYS
        ],
        '  <graph edgedefault="directed">',
    ]
    for node in graph["nodes"]:
        label = html.escape(str(node.get("label", "")))
        name = html.escape(str(node.get("name") or node.get("title") or node.get("text") or node["id"]))
        lines.extend(
            [
                f'    <node id="{html.escape(node["id"])}">',
                f'      <data key="node_label">{label}</data>',
                f'      <data key="node_name">{name}</data>',
                "    </node>",
            ]
        )
    for relationship in graph["relationships"]:
        fallback_id = hashlib.sha256(
            f"{relationship.get('from')}|{relationship.get('type')}|{relationship.get('to')}".encode("utf-8")
        ).hexdigest()[:24]
        rel_id = html.escape(str(relationship.get("id") or fallback_id))
        rel_type = html.escape(str(relationship.get("type") or ""))
        confidence = relationship.get("confidence")
        source_video_id = relationship.get("source_video_id")
        lines.extend(
            [
                f'    <edge id="{rel_id}" source="{html.escape(relationship["from"])}" target="{html.escape(relationship["to"])}">',
                f'      <data key="edge_type">{rel_type}</data>',
            ]
        )
        if confidence is not None:
            lines.append(f'      <data key="confidence">{html.escape(str(confidence))}</data>')
        if source_video_id is not None:
            lines.append(f'      <data key="source_video_id">{html.escape(str(source_video_id))}</data>')
        for key in LLM_EDGE_PROVENANCE_KEYS:
            value = relationship.get(key)
            if value is not None:
                lines.append(f'      <data key="{key}">{html.escape(str(value))}</data>')
        lines.append("    </edge>")
    lines.extend(["  </graph>", "</graphml>"])
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
