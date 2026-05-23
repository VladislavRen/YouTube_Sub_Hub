from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def slugify(value: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    return "-".join(part for part in normalized.split("-") if part)


@dataclass(frozen=True)
class ParsedYouTubeUrl:
    video_id: str
    canonical_url: str


@dataclass
class VideoMetadata:
    video_id: str
    canonical_url: str
    title: str
    channel: str | None = None
    duration_seconds: int | None = None
    description: str | None = None
    source_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TranscriptSegment:
    start: float
    duration: float
    text: str
    raw_text: str | None = None

    @property
    def timestamp_url_offset(self) -> int:
        return max(0, int(self.start))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Transcript:
    video_id: str
    source: str
    segments: list[TranscriptSegment]
    language: str | None = None

    def full_text(self) -> str:
        return "\n".join(segment.text for segment in self.segments if segment.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "source": self.source,
            "language": self.language,
            "segments": [segment.to_dict() for segment in self.segments],
        }


@dataclass
class TranscriptBundle:
    video_id: str
    source: str
    raw_payload: Any
    normalized: Transcript

    def raw_to_dict(self) -> Any:
        return self.raw_payload


@dataclass
class NoteSection:
    heading: str
    summary: str
    timestamp_seconds: int | None = None
    bullets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Concept:
    name: str
    definition: str

    @property
    def slug(self) -> str:
        return slugify(self.name)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["slug"] = self.slug
        return data


@dataclass
class Claim:
    text: str
    evidence: str

    @property
    def slug(self) -> str:
        return slugify(self.text)[:96]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["slug"] = self.slug
        return data


@dataclass
class Analysis:
    language: str
    cleaned_title: str
    abstract: str
    tags: list[str]
    outline: list[str]
    sections: list[NoteSection]
    concepts: list[Concept]
    claims: list[Claim]
    limitations: list[str]
    open_questions: list[str]
    related_topics: list[str]
    prompt_version: str = "scientific-note-v1"
    model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "cleaned_title": self.cleaned_title,
            "abstract": self.abstract,
            "tags": self.tags,
            "outline": self.outline,
            "sections": [section.to_dict() for section in self.sections],
            "concepts": [concept.to_dict() for concept in self.concepts],
            "claims": [claim.to_dict() for claim in self.claims],
            "limitations": self.limitations,
            "open_questions": self.open_questions,
            "related_topics": self.related_topics,
            "prompt_version": self.prompt_version,
            "model": self.model,
        }


@dataclass
class JobRecord:
    job_id: str
    url: str
    video_id: str
    status: str
    stage: str | None = None
    note_path: str | None = None
    error: str | None = None
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
