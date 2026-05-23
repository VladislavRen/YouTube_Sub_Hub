from __future__ import annotations

import re

from yt_sub_hub.models import Analysis


BLOCKED_ENGLISH_PHRASES: tuple[str, ...] = (
    "Scientific summary",
    "The note is assembled",
    "Detailed Notes",
    "Abstract",
    "Outline",
    "Claims",
    "Key Concepts",
    "Related Topics",
    "Open Questions",
    "Limitations",
    "The video",
    "This video",
    "source transcript",
    "analysis generated",
    "summary of",
    "Evidence:",
    "Claim:",
)

ENGLISH_RU_MARKDOWN_HEADINGS: tuple[str, ...] = (
    "## Abstract",
    "## Outline",
    "## Detailed Notes",
    "## Key Concepts",
    "## Claims",
    "## Claims And Arguments",
    "## Limitations",
    "## Open Questions",
    "## Related Topics",
)

ENGLISH_RU_MARKDOWN_LABELS: tuple[str, ...] = ("**Claim:**", "Evidence:")


def contains_cyrillic(text: str | None) -> bool:
    return any("\u0400" <= char <= "\u04ff" for char in text or "")


def word_tokens(text: str | None) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    for char in text or "":
        if char.isalpha():
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def ru_word_ratio(text: str | None) -> float:
    tokens = word_tokens(text)
    if not tokens:
        return 0.0
    ru_tokens = sum(1 for token in tokens if contains_cyrillic(token))
    return ru_tokens / len(tokens)


def contains_blocked_english_phrases(text: str | None) -> list[str]:
    normalized = text or ""
    return [phrase for phrase in BLOCKED_ENGLISH_PHRASES if re.search(re.escape(phrase), normalized, re.IGNORECASE)]


def validate_ru_text_field(
    name: str,
    text: str | None,
    min_ratio: float,
    required: bool,
    short_label: bool = False,
) -> list[str]:
    value = (text or "").strip()
    issues: list[str] = []
    if not value:
        if required:
            issues.append(f"{name} is required")
        return issues

    for phrase in contains_blocked_english_phrases(value):
        issues.append(f"{name} contains blocked English phrase: {phrase}")

    tokens = word_tokens(value)
    if required and not tokens:
        issues.append(f"{name} must contain words")
        return issues

    if short_label and len(tokens) <= 2:
        non_ru = [token for token in tokens if not contains_cyrillic(token)]
        if non_ru:
            issues.append(f"{name} must use Cyrillic words")
        return issues

    ratio = ru_word_ratio(value)
    if ratio < min_ratio:
        issues.append(f"{name} ru_word_ratio {ratio:.2f} below {min_ratio:.2f}")
    return issues


def validate_ru_analysis(analysis: Analysis) -> list[str]:
    issues: list[str] = []
    if (analysis.language or "").strip().lower() != "ru":
        issues.append("language must be ru")
        return issues

    issues.extend(validate_ru_text_field("abstract", analysis.abstract, 0.70, True))
    for index, tag in enumerate(analysis.tags):
        issues.extend(validate_ru_text_field(f"tags[{index}]", tag, 0.50, True, short_label=True))

    if not analysis.outline:
        issues.append("outline must contain at least one item")
    for index, item in enumerate(analysis.outline):
        issues.extend(validate_ru_text_field(f"outline[{index}]", item, 0.50, True, short_label=True))

    if not analysis.sections:
        issues.append("sections must contain at least one item")
    for index, section in enumerate(analysis.sections):
        issues.extend(validate_ru_text_field(f"sections[{index}].heading", section.heading, 0.50, True, short_label=True))
        issues.extend(validate_ru_text_field(f"sections[{index}].summary", section.summary, 0.65, True))
        for bullet_index, bullet in enumerate(section.bullets):
            issues.extend(validate_ru_text_field(f"sections[{index}].bullets[{bullet_index}]", bullet, 0.65, True))

    if not analysis.concepts:
        issues.append("concepts must contain at least one item")
    for index, concept in enumerate(analysis.concepts):
        issues.extend(validate_ru_text_field(f"concepts[{index}].name", concept.name, 0.50, True, short_label=True))
        issues.extend(validate_ru_text_field(f"concepts[{index}].definition", concept.definition, 0.65, True))

    if not analysis.claims:
        issues.append("claims must contain at least one item")
    for index, claim in enumerate(analysis.claims):
        issues.extend(validate_ru_text_field(f"claims[{index}].text", claim.text, 0.65, True))
        issues.extend(validate_ru_text_field(f"claims[{index}].evidence", claim.evidence, 0.65, True))

    for index, item in enumerate(analysis.limitations):
        issues.extend(validate_ru_text_field(f"limitations[{index}]", item, 0.65, True))
    for index, item in enumerate(analysis.open_questions):
        issues.extend(validate_ru_text_field(f"open_questions[{index}]", item, 0.65, True))
    for index, item in enumerate(analysis.related_topics):
        issues.extend(validate_ru_text_field(f"related_topics[{index}]", item, 0.50, True, short_label=True))

    return issues


def validate_ru_markdown_note(markdown: str) -> list[str]:
    issues: list[str] = []
    for heading in ENGLISH_RU_MARKDOWN_HEADINGS:
        pattern = r"(?im)^\s*" + re.escape(heading) + r"\s*$"
        if re.search(pattern, markdown):
            issues.append(f"contains English markdown heading: {heading}")
    for label in ENGLISH_RU_MARKDOWN_LABELS:
        if label in markdown:
            issues.append(f"contains English markdown label: {label}")
    for phrase in contains_blocked_english_phrases(_markdown_body(markdown)):
        issues.append(f"contains blocked English phrase: {phrase}")
    issues.extend(
        issue
        for issue in validate_ru_text_field("body", _markdown_body_text(markdown), 0.65, True)
        if "contains blocked English phrase" not in issue
    )
    return issues


def _markdown_body(markdown: str) -> str:
    lines = markdown.splitlines()
    if lines and lines[0].strip() == "---":
        for index in range(1, len(lines)):
            if lines[index].strip() == "---":
                return "\n".join(lines[index + 1 :])
    return markdown


def _markdown_body_text(markdown: str) -> str:
    body = _markdown_body(markdown)
    body = re.sub(r"https?://\S+", " ", body)
    body = re.sub(r"`[^`]*`", " ", body)
    body = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", body)
    body = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", body)
    body = re.sub(r"[*_>#|:-]+", " ", body)
    return body
