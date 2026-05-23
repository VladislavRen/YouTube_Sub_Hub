from __future__ import annotations

from yt_sub_hub.graph_extraction import (
    build_visual_graph_payload,
    visual_evidence_node_id,
    visual_frame_node_id,
)


def test_visual_frame_node_id_is_stable_for_video_and_timestamp():
    assert visual_frame_node_id("abc123", 15) == "visual_frame:abc123:000015"
    assert visual_frame_node_id("abc123", 15.9) == "visual_frame:abc123:000015_900"


def test_visual_evidence_node_id_is_stable_for_frame_and_summary_text():
    first = visual_evidence_node_id("visual_frame:abc123:000015", "Breakout above resistance.")
    second = visual_evidence_node_id("visual_frame:abc123:000015", "Breakout above resistance.")
    different = visual_evidence_node_id("visual_frame:abc123:000075", "Breakout above resistance.")

    assert first == second
    assert first.startswith("visual_evidence:")
    assert first != different


def test_build_visual_graph_payload_adds_frame_evidence_and_chart_concepts():
    visual_summary = {
        "video_id": "abc123",
        "frames": [
            {
                "timestamp_seconds": 15,
                "frame_path": "library/videos/abc123/visual/frames/000015.jpg",
                "screen_type": "candlestick_chart",
                "summary": "A chart shows a breakout above resistance.",
                "chart_elements": {
                    "ticker": "BTCUSDT",
                    "timeframe": "1h",
                    "patterns": ["breakout"],
                    "levels": ["resistance"],
                    "indicators": ["volume"],
                },
                "transcript_context": {"start": 10, "end": 20, "text": "Price breaks resistance here."},
                "confidence": 0.82,
            }
        ],
    }

    payload = build_visual_graph_payload(visual_summary)
    nodes = {node["id"]: node for node in payload.nodes}
    relationships = payload.relationships

    frame_id = "visual_frame:abc123:000015"
    assert nodes[frame_id]["label"] == "VisualFrame"
    assert nodes[frame_id]["video_id"] == "abc123"
    assert nodes[frame_id]["timestamp_seconds"] == 15
    assert nodes[frame_id]["summary"] == "A chart shows a breakout above resistance."
    assert nodes["chart_pattern:breakout"]["label"] == "ChartPattern"
    assert nodes["chart_pattern:breakout"]["name"] == "breakout"
    assert nodes["chart_level:resistance"]["label"] == "ChartLevel"
    assert nodes["indicator:volume"]["label"] == "Indicator"
    assert any(
        rel["from"] == "video:abc123"
        and rel["to"] == frame_id
        and rel["type"] == "HAS_VISUAL_FRAME"
        and rel["source_frame_id"] == frame_id
        and rel["extraction_method"] == "vision_frame_v1"
        for rel in relationships
    )
    assert any(
        rel["from"] == "chart_pattern:breakout"
        and rel["to"] == frame_id
        and rel["type"] == "SHOWN_IN"
        and rel["source_video_id"] == "abc123"
        and rel["source_frame_id"] == frame_id
        and rel["extraction_method"] == "vision_frame_v1"
        for rel in relationships
    )
    assert any(
        rel["from"].startswith("visual_evidence:")
        and rel["to"] == frame_id
        and rel["type"] == "SHOWN_IN"
        and rel["evidence_text"] == "A chart shows a breakout above resistance."
        for rel in relationships
    )
