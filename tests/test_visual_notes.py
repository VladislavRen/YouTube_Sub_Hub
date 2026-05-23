from __future__ import annotations

from yt_sub_hub.visual_notes import render_visual_note


def test_render_visual_note_from_summary_dict_orders_frames_and_links_timestamps():
    summary = {
        "video_id": "abc123",
        "generated_at": "2026-04-29T10:00:00Z",
        "frames": [
            {
                "timestamp_seconds": 75,
                "frame_path": "library/videos/abc123/visual/frames/000075.jpg",
                "screen_type": "slide",
                "summary": "A slide introduces causal graphs.",
                "visible_objects": ["title text", "node diagram"],
                "chart_elements": {"patterns": [], "levels": [], "indicators": []},
                "transcript_context": {"start": 70, "end": 80, "text": "Here is the causal graph."},
                "uncertainties": [],
                "confidence": 0.91,
            },
            {
                "timestamp_seconds": 15,
                "frame_path": "library/videos/abc123/visual/frames/000015.jpg",
                "screen_type": "candlestick_chart",
                "summary": "A chart shows a breakout above resistance.",
                "visible_objects": ["candles", "volume bars"],
                "chart_elements": {
                    "ticker": "BTCUSDT",
                    "timeframe": "1h",
                    "patterns": ["breakout"],
                    "levels": ["resistance"],
                    "indicators": ["volume"],
                },
                "explanation": "The visual supports the spoken breakout claim.",
                "transcript_context": {"start": 10, "end": 20, "text": "Price breaks resistance here."},
                "uncertainties": ["exchange name is not readable"],
                "confidence": 0.82,
            },
        ],
    }

    markdown = render_visual_note(summary)

    assert markdown.startswith("# Visual Analysis\n\nVideo: abc123\n\n## Frames\n")
    assert markdown.index("### 00:15") < markdown.index("### 01:15")
    assert "**Screen type:** candlestick_chart" in markdown
    assert "[open frame](library/videos/abc123/visual/frames/000015.jpg)" in markdown
    assert "**Summary:** A chart shows a breakout above resistance." in markdown
    assert "- Ticker: BTCUSDT" in markdown
    assert "- Patterns: breakout" in markdown
    assert "- Levels: resistance" in markdown
    assert "- Indicators: volume" in markdown
    assert "> Price breaks resistance here." in markdown
    assert "- exchange name is not readable" in markdown
    assert "Confidence: 0.82" in markdown


def test_render_visual_note_from_frame_list_uses_unknown_video_title_and_skips_empty_sections():
    markdown = render_visual_note(
        [
            {
                "timestamp_seconds": 0,
                "screen_type": "talking_head",
                "summary": "The presenter introduces the topic.",
                "visible_objects": [],
                "chart_elements": {},
                "transcript_context": {},
                "uncertainties": [],
                "confidence": None,
            }
        ]
    )

    assert markdown.startswith("# Visual Analysis\n\nVideo: unknown\n\n## Frames\n")
    assert "### 00:00" in markdown
    assert "**Screen type:** talking_head" in markdown
    assert "The presenter introduces the topic." in markdown
    assert "Visible Objects" not in markdown
    assert "Chart Elements" not in markdown
    assert "Uncertainties" not in markdown
    assert "Confidence:" not in markdown


def test_render_visual_note_empty_summary_returns_actionable_empty_note():
    markdown = render_visual_note({"video_id": "abc123", "frames": []})

    assert markdown == "# Visual Analysis\n\nVideo: abc123\n\n## Frames\n\nNo visual frames analyzed yet.\n"
