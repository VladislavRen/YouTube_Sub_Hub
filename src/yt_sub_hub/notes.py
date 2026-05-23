from __future__ import annotations

from yt_sub_hub.models import Analysis, VideoMetadata


def _yaml_scalar(value: str) -> str:
    clean = " ".join(value.split())
    return '"' + clean.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _yaml_list(values: list[str]) -> str:
    if not values:
        return "[]"
    return "[" + ", ".join(_yaml_scalar(value) for value in values) + "]"


def _timestamp_link(metadata: VideoMetadata, seconds: int | None) -> str:
    if seconds is None:
        return ""
    mins, secs = divmod(max(0, int(seconds)), 60)
    return f" [{mins:02d}:{secs:02d}](https://youtu.be/{metadata.video_id}?t={int(seconds)})"


def render_markdown_note(metadata: VideoMetadata, analysis: Analysis, provenance: dict[str, str] | None = None) -> str:
    labels = _section_labels(analysis.language)
    provenance = provenance or {}
    lines: list[str] = [
        "---",
        f"video_id: {_yaml_scalar(metadata.video_id)}",
        f"url: {_yaml_scalar(metadata.canonical_url)}",
        f"title: {_yaml_scalar(metadata.title)}",
        f"channel: {_yaml_scalar(metadata.channel or '')}",
        f"language: {_yaml_scalar(analysis.language)}",
        f"tags: {_yaml_list(analysis.tags)}",
        f"prompt_version: {_yaml_scalar(analysis.prompt_version)}",
        f"model: {_yaml_scalar(analysis.model or '')}",
    ]
    for key, value in sorted(provenance.items()):
        lines.append(f"{key}: {_yaml_scalar(str(value))}")
    lines.extend(
        [
            "---",
            "",
            f"# {analysis.cleaned_title or metadata.title}",
            "",
            f"## {labels['abstract']}",
            "",
            analysis.abstract,
            "",
            f"## {labels['outline']}",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in analysis.outline)
    lines.extend(["", f"## {labels['detailed_notes']}", ""])
    for section in analysis.sections:
        lines.append(f"### {section.heading}{_timestamp_link(metadata, section.timestamp_seconds)}")
        lines.extend(["", section.summary, ""])
        lines.extend(f"- {bullet}" for bullet in section.bullets)
        lines.append("")

    lines.extend([f"## {labels['key_concepts']}", ""])
    lines.extend(f"- **{concept.name}**: {concept.definition}" for concept in analysis.concepts)
    lines.extend(["", f"## {labels['claims']}", ""])
    lines.extend(
        f"- **{labels['claim_label']}:** {claim.text}\n  - {labels['evidence_label']}: {claim.evidence}"
        for claim in analysis.claims
    )
    lines.extend(["", f"## {labels['limitations']}", ""])
    lines.extend(f"- {item}" for item in analysis.limitations)
    lines.extend(["", f"## {labels['open_questions']}", ""])
    lines.extend(f"- {item}" for item in analysis.open_questions)
    lines.extend(["", f"## {labels['related_topics']}", ""])
    lines.extend(f"- {item}" for item in analysis.related_topics)
    lines.append("")
    return "\n".join(lines)


def _section_labels(language: str) -> dict[str, str]:
    if (language or "").strip().lower() == "ru":
        return {
            "abstract": "Аннотация",
            "outline": "План",
            "detailed_notes": "Подробные заметки",
            "key_concepts": "Ключевые понятия",
            "claims": "Утверждения",
            "claim_label": "Утверждение",
            "evidence_label": "Доказательство",
            "limitations": "Ограничения",
            "open_questions": "Открытые вопросы",
            "related_topics": "Связанные темы",
        }
    return {
        "abstract": "Abstract",
        "outline": "Outline",
        "detailed_notes": "Detailed Notes",
        "key_concepts": "Key Concepts",
        "claims": "Claims And Arguments",
        "claim_label": "Claim",
        "evidence_label": "Evidence",
        "limitations": "Limitations",
        "open_questions": "Open Questions",
        "related_topics": "Related Topics",
    }
