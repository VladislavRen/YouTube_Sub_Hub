from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def _frames_from_summary(visual_summary: Mapping[str, Any] | list[Mapping[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(visual_summary, Mapping):
        video_id = str(visual_summary.get("video_id") or "unknown")
        raw_frames = visual_summary.get("frames") or []
    else:
        video_id = "unknown"
        raw_frames = visual_summary
    frames = [dict(frame) for frame in raw_frames if isinstance(frame, Mapping)]
    frames.sort(key=lambda frame: float(frame.get("timestamp_seconds") or 0))
    return video_id, frames


def _timestamp(seconds: Any) -> str:
    total = max(0, int(float(seconds or 0)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _values(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Mapping):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, Iterable):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)]


def _append_list_section(lines: list[str], heading: str, values: Any) -> None:
    items = _values(values)
    if not items:
        return
    lines.extend([f"### {heading}", ""])
    lines.extend(f"- {item}" for item in items)
    lines.append("")


def _append_chart_elements(lines: list[str], chart_elements: Mapping[str, Any]) -> None:
    entries: list[str] = []
    for key, value in chart_elements.items():
        values = _values(value)
        if not values:
            continue
        label = str(key).replace("_", " ").title()
        entries.append(f"- {label}: {', '.join(values)}")
    if not entries:
        return
    lines.extend(["### Chart Elements", ""])
    lines.extend(entries)
    lines.append("")


def _append_transcript_context(lines: list[str], context: Mapping[str, Any]) -> None:
    text = str(context.get("text") or "").strip()
    if not text:
        return
    start = context.get("start")
    end = context.get("end")
    suffix = ""
    if start is not None and end is not None:
        suffix = f" ({_timestamp(start)}-{_timestamp(end)})"
    lines.extend([f"### Transcript Context{suffix}", "", f"> {text}", ""])


def render_visual_note(visual_summary: Mapping[str, Any] | list[Mapping[str, Any]]) -> str:
    video_id, frames = _frames_from_summary(visual_summary)
    lines = ["# Visual Analysis", "", f"Video: {video_id}", "", "## Frames", ""]
    if not frames:
        lines.append("No visual frames analyzed yet.")
        lines.append("")
        return "\n".join(lines)

    for frame in frames:
        timestamp = _timestamp(frame.get("timestamp_seconds"))
        screen_type = str(frame.get("screen_type") or "unknown")
        lines.extend([f"### {timestamp}", "", f"**Screen type:** {screen_type}", ""])

        frame_path = str(frame.get("frame_path") or "").strip()
        if frame_path:
            lines.extend([f"[open frame]({frame_path})", ""])

        summary = str(frame.get("summary") or "").strip()
        if summary:
            lines.extend([f"**Summary:** {summary}", ""])

        explanation = str(frame.get("explanation") or "").strip()
        if explanation:
            lines.extend([f"**Explanation:** {explanation}", ""])

        _append_list_section(lines, "Visible Objects", frame.get("visible_objects"))
        chart_elements = frame.get("chart_elements")
        if isinstance(chart_elements, Mapping):
            _append_chart_elements(lines, chart_elements)
        context = frame.get("transcript_context")
        if isinstance(context, Mapping):
            _append_transcript_context(lines, context)
        _append_list_section(lines, "Uncertainties", frame.get("uncertainties"))

        confidence = frame.get("confidence")
        if confidence is not None:
            lines.extend([f"Confidence: {confidence}", ""])

    return "\n".join(lines)
