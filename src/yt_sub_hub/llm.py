from __future__ import annotations

import json
import re
import time
from typing import Any

from yt_sub_hub.models import Analysis, Claim, Concept, NoteSection, Transcript, TranscriptSegment, VideoMetadata
from yt_sub_hub.processing import chunk_transcript

PROMPT_VERSION = "scientific-note-v1"
ANALYSIS_REQUIRED_FIELDS = [
    "language",
    "cleaned_title",
    "abstract",
    "tags",
    "outline",
    "sections",
    "concepts",
    "claims",
    "limitations",
    "open_questions",
    "related_topics",
]


class AnalysisValidationError(ValueError):
    """Raised when an LLM response does not match the analysis contract."""


DEEP_GRAPH_ARRAY_FIELDS = [
    "concepts",
    "claims",
    "evidence",
    "concept_relations",
    "claim_relations",
    "methods",
    "tools",
    "metrics",
    "risks",
    "questions",
]


def build_analysis_json_schema_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "scientific_note_analysis",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ANALYSIS_REQUIRED_FIELDS,
                "properties": {
                    "language": {"type": "string"},
                    "cleaned_title": {"type": "string"},
                    "abstract": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "outline": {"type": "array", "items": {"type": "string"}},
                    "sections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["heading", "summary", "timestamp_seconds", "bullets"],
                            "properties": {
                                "heading": {"type": "string"},
                                "summary": {"type": "string"},
                                "timestamp_seconds": {"type": "integer"},
                                "bullets": {"type": "array", "items": {"type": "string"}},
                            },
                        },
                    },
                    "concepts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["name", "definition"],
                            "properties": {"name": {"type": "string"}, "definition": {"type": "string"}},
                        },
                    },
                    "claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["text", "evidence"],
                            "properties": {"text": {"type": "string"}, "evidence": {"type": "string"}},
                        },
                    },
                    "limitations": {"type": "array", "items": {"type": "string"}},
                    "open_questions": {"type": "array", "items": {"type": "string"}},
                    "related_topics": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    }


def build_chunk_analysis_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "scientific_note_chunk",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "heading",
                    "summary",
                    "timestamp_seconds",
                    "bullets",
                    "concepts",
                    "claims",
                    "limitations",
                    "open_questions",
                    "related_topics",
                ],
                "properties": {
                    "heading": {"type": "string"},
                    "summary": {"type": "string"},
                    "timestamp_seconds": {"type": "integer"},
                    "bullets": {"type": "array", "items": {"type": "string"}},
                    "concepts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["name", "definition"],
                            "properties": {"name": {"type": "string"}, "definition": {"type": "string"}},
                        },
                    },
                    "claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["text", "evidence"],
                            "properties": {"text": {"type": "string"}, "evidence": {"type": "string"}},
                        },
                    },
                    "limitations": {"type": "array", "items": {"type": "string"}},
                    "open_questions": {"type": "array", "items": {"type": "string"}},
                    "related_topics": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    }


def build_deep_graph_chunk_response_format() -> dict[str, Any]:
    number_confidence = {"type": "number", "minimum": 0, "maximum": 1}
    timed_fields = {
        "timestamp_start": {"type": "number"},
        "timestamp_end": {"type": "number"},
    }
    def object_schema(required: list[str], properties: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": required,
            "properties": properties,
        }

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "deep_graph_chunk",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "chunk_id",
                    "video_id",
                    "timestamp_start",
                    "timestamp_end",
                    "concepts",
                    "claims",
                    "evidence",
                    "concept_relations",
                    "claim_relations",
                    "methods",
                    "tools",
                    "metrics",
                    "risks",
                    "questions",
                ],
                "properties": {
                    "chunk_id": {"type": "string"},
                    "video_id": {"type": "string"},
                    "timestamp_start": {"type": "number"},
                    "timestamp_end": {"type": "number"},
                    "concepts": {
                        "type": "array",
                        "items": object_schema(
                            [
                                "local_id",
                                "name",
                                "definition",
                                "aliases",
                                "confidence",
                                "evidence_quote",
                                "timestamp_start",
                                "timestamp_end",
                            ],
                            {
                                "local_id": {"type": "string"},
                                "name": {"type": "string"},
                                "definition": {"type": "string"},
                                "aliases": {"type": "array", "items": {"type": "string"}},
                                "confidence": number_confidence,
                                "evidence_quote": {"type": "string"},
                                **timed_fields,
                            },
                        ),
                    },
                    "claims": {
                        "type": "array",
                        "items": object_schema(
                            [
                                "local_id",
                                "text",
                                "paraphrased_evidence",
                                "confidence",
                                "timestamp_start",
                                "timestamp_end",
                            ],
                            {
                                "local_id": {"type": "string"},
                                "text": {"type": "string"},
                                "paraphrased_evidence": {"type": "string"},
                                "confidence": number_confidence,
                                **timed_fields,
                            },
                        ),
                    },
                    "evidence": {
                        "type": "array",
                        "items": object_schema(
                            [
                                "local_id",
                                "text",
                                "kind",
                                "quote",
                                "supports_claim_ids",
                                "confidence",
                                "timestamp_start",
                                "timestamp_end",
                            ],
                            {
                                "local_id": {"type": "string"},
                                "text": {"type": "string"},
                                "kind": {
                                    "type": "string",
                                    "enum": ["quote", "paraphrase", "example", "data_point", "citation"],
                                },
                                "quote": {"type": "string"},
                                "supports_claim_ids": {"type": "array", "items": {"type": "string"}},
                                "confidence": number_confidence,
                                **timed_fields,
                            },
                        ),
                    },
                    "concept_relations": {
                        "type": "array",
                        "items": object_schema(
                            [
                                "from_concept_id",
                                "to_concept_id",
                                "type",
                                "confidence",
                                "evidence_quote",
                                "timestamp_start",
                                "timestamp_end",
                            ],
                            {
                                "from_concept_id": {"type": "string"},
                                "to_concept_id": {"type": "string"},
                                "type": {
                                    "type": "string",
                                    "enum": ["RELATED_TO", "PART_OF", "CAUSES", "CONTRASTS_WITH", "MEASURED_BY"],
                                },
                                "confidence": number_confidence,
                                "evidence_quote": {"type": "string"},
                                **timed_fields,
                            },
                        ),
                    },
                    "claim_relations": {
                        "type": "array",
                        "items": object_schema(
                            [
                                "from_claim_id",
                                "to_claim_id",
                                "type",
                                "confidence",
                                "evidence_quote",
                                "timestamp_start",
                                "timestamp_end",
                            ],
                            {
                                "from_claim_id": {"type": "string"},
                                "to_claim_id": {"type": "string"},
                                "type": {"type": "string", "enum": ["SUPPORTS", "CONTRADICTS"]},
                                "confidence": number_confidence,
                                "evidence_quote": {"type": "string"},
                                **timed_fields,
                            },
                        ),
                    },
                    "methods": {
                        "type": "array",
                        "items": object_schema(
                            [
                                "local_id",
                                "name",
                                "description",
                                "uses_tool_ids",
                                "applies_to_concept_ids",
                                "confidence",
                                "evidence_quote",
                                "timestamp_start",
                                "timestamp_end",
                            ],
                            {
                                "local_id": {"type": "string"},
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "uses_tool_ids": {"type": "array", "items": {"type": "string"}},
                                "applies_to_concept_ids": {"type": "array", "items": {"type": "string"}},
                                "confidence": number_confidence,
                                "evidence_quote": {"type": "string"},
                                **timed_fields,
                            },
                        ),
                    },
                    "tools": {
                        "type": "array",
                        "items": object_schema(
                            [
                                "local_id",
                                "name",
                                "description",
                                "confidence",
                                "evidence_quote",
                                "timestamp_start",
                                "timestamp_end",
                            ],
                            {
                                "local_id": {"type": "string"},
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "confidence": number_confidence,
                                "evidence_quote": {"type": "string"},
                                **timed_fields,
                            },
                        ),
                    },
                    "metrics": {
                        "type": "array",
                        "items": object_schema(
                            [
                                "local_id",
                                "name",
                                "definition",
                                "measured_concept_ids",
                                "confidence",
                                "evidence_quote",
                                "timestamp_start",
                                "timestamp_end",
                            ],
                            {
                                "local_id": {"type": "string"},
                                "name": {"type": "string"},
                                "definition": {"type": "string"},
                                "measured_concept_ids": {"type": "array", "items": {"type": "string"}},
                                "confidence": number_confidence,
                                "evidence_quote": {"type": "string"},
                                **timed_fields,
                            },
                        ),
                    },
                    "risks": {
                        "type": "array",
                        "items": object_schema(
                            [
                                "local_id",
                                "text",
                                "related_concept_ids",
                                "confidence",
                                "evidence_quote",
                                "timestamp_start",
                                "timestamp_end",
                            ],
                            {
                                "local_id": {"type": "string"},
                                "text": {"type": "string"},
                                "related_concept_ids": {"type": "array", "items": {"type": "string"}},
                                "confidence": number_confidence,
                                "evidence_quote": {"type": "string"},
                                **timed_fields,
                            },
                        ),
                    },
                    "questions": {
                        "type": "array",
                        "items": object_schema(
                            [
                                "local_id",
                                "text",
                                "about_concept_ids",
                                "confidence",
                                "evidence_quote",
                                "timestamp_start",
                                "timestamp_end",
                            ],
                            {
                                "local_id": {"type": "string"},
                                "text": {"type": "string"},
                                "about_concept_ids": {"type": "array", "items": {"type": "string"}},
                                "confidence": number_confidence,
                                "evidence_quote": {"type": "string"},
                                **timed_fields,
                            },
                        ),
                    },
                },
            },
        },
    }


def extract_json_payload(content: str) -> str:
    stripped = content.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL | re.IGNORECASE)
    return fenced.group(1).strip() if fenced else stripped


def rejected_deep_graph_chunk(
    expected_video_id: str,
    expected_chunk_id: str,
    error: str,
    *,
    timestamp_start: float = 0,
    timestamp_end: float = 0,
) -> dict[str, Any]:
    return {
        "chunk_id": expected_chunk_id,
        "video_id": expected_video_id,
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
        "concepts": [],
        "claims": [],
        "evidence": [],
        "concept_relations": [],
        "claim_relations": [],
        "methods": [],
        "tools": [],
        "metrics": [],
        "risks": [],
        "questions": [],
        "rejected": True,
        "error": error,
    }


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_deep_graph_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _confidence_in_range(item: dict[str, Any]) -> bool:
    if "confidence" not in item:
        return True
    value = _number(item.get("confidence"), default=-1)
    return 0 <= value <= 1


def parse_deep_graph_chunk_response(
    payload: dict[str, Any] | str,
    *,
    expected_video_id: str,
    expected_chunk_id: str,
) -> dict[str, Any]:
    if isinstance(payload, str):
        try:
            data = json.loads(extract_json_payload(payload))
        except json.JSONDecodeError as exc:
            return rejected_deep_graph_chunk(
                expected_video_id,
                expected_chunk_id,
                f"Invalid deep graph JSON: {exc}",
            )
    else:
        data = payload
    if not isinstance(data, dict):
        return rejected_deep_graph_chunk(expected_video_id, expected_chunk_id, "Deep graph response must be a JSON object")

    normalized: dict[str, Any] = {
        "chunk_id": expected_chunk_id,
        "video_id": expected_video_id,
        "timestamp_start": _number(data.get("timestamp_start")),
        "timestamp_end": _number(data.get("timestamp_end")),
    }
    invalid_fields: list[str] = []
    for field in DEEP_GRAPH_ARRAY_FIELDS:
        items = _normalize_deep_graph_items(data.get(field, []))
        if any(not _confidence_in_range(item) for item in items):
            invalid_fields.append(field)
        normalized[field] = items
    if invalid_fields:
        return rejected_deep_graph_chunk(
            expected_video_id,
            expected_chunk_id,
            f"Deep graph response has invalid confidence values in: {', '.join(sorted(invalid_fields))}",
            timestamp_start=normalized["timestamp_start"],
            timestamp_end=normalized["timestamp_end"],
        )
    return normalized


def _require(data: dict[str, Any], key: str, expected_type: type) -> Any:
    if key not in data:
        raise AnalysisValidationError(f"Missing required analysis field: {key}")
    value = data[key]
    if not isinstance(value, expected_type):
        raise AnalysisValidationError(f"Field {key} must be {expected_type.__name__}")
    return value


def parse_analysis_response(payload: dict[str, Any] | str, model: str | None = None) -> Analysis:
    if isinstance(payload, str):
        try:
            data = json.loads(extract_json_payload(payload))
        except json.JSONDecodeError as exc:
            raise AnalysisValidationError("Analysis response is not valid JSON") from exc
    else:
        data = payload

    sections = [
        NoteSection(
            heading=str(_require(section, "heading", str)),
            summary=str(_require(section, "summary", str)),
            timestamp_seconds=section.get("timestamp_seconds"),
            bullets=[str(item) for item in section.get("bullets", [])],
        )
        for section in _require(data, "sections", list)
    ]
    concepts = [
        Concept(name=str(_require(concept, "name", str)), definition=str(_require(concept, "definition", str)))
        for concept in _require(data, "concepts", list)
    ]
    claims = [
        Claim(text=str(_require(claim, "text", str)), evidence=str(_require(claim, "evidence", str)))
        for claim in _require(data, "claims", list)
    ]
    return Analysis(
        language=str(_require(data, "language", str)),
        cleaned_title=str(_require(data, "cleaned_title", str)),
        abstract=str(_require(data, "abstract", str)),
        tags=[str(item) for item in _require(data, "tags", list)],
        outline=[str(item) for item in _require(data, "outline", list)],
        sections=sections,
        concepts=concepts,
        claims=claims,
        limitations=[str(item) for item in _require(data, "limitations", list)],
        open_questions=[str(item) for item in _require(data, "open_questions", list)],
        related_topics=[str(item) for item in _require(data, "related_topics", list)],
        prompt_version=str(data.get("prompt_version", PROMPT_VERSION)),
        model=model or data.get("model"),
    )


def _dedupe_strings(values: list[str], limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(value)
        if limit is not None and len(result) >= limit:
            break
    return result


def merge_chunk_analyses(metadata: VideoMetadata, chunks: list[dict[str, Any]], model: str | None = None) -> Analysis:
    sections: list[NoteSection] = []
    concepts_by_slug: dict[str, Concept] = {}
    claims_by_slug: dict[str, Claim] = {}
    limitations: list[str] = []
    open_questions: list[str] = []
    related_topics: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        heading = str(chunk.get("heading") or f"Part {index}")
        summary = str(chunk.get("summary") or "")
        sections.append(
            NoteSection(
                heading=heading,
                summary=summary,
                timestamp_seconds=int(chunk.get("timestamp_seconds") or 0),
                bullets=[str(item) for item in chunk.get("bullets", [])],
            )
        )
        for concept_data in chunk.get("concepts", []):
            concept = Concept(
                name=str(concept_data.get("name", "")),
                definition=str(concept_data.get("definition", "")),
            )
            if concept.slug:
                concepts_by_slug.setdefault(concept.slug, concept)
        for claim_data in chunk.get("claims", []):
            claim = Claim(text=str(claim_data.get("text", "")), evidence=str(claim_data.get("evidence", "")))
            if claim.slug:
                claims_by_slug.setdefault(claim.slug, claim)
        limitations.extend(str(item) for item in chunk.get("limitations", []))
        open_questions.extend(str(item) for item in chunk.get("open_questions", []))
        related_topics.extend(str(item) for item in chunk.get("related_topics", []))

    outline = [section.heading for section in sections]
    abstract = (
        f"Scientific summary of '{metadata.title}'. The note is assembled from {len(sections)} transcript "
        "segments to preserve coverage across the full video."
    )
    return Analysis(
        language="unknown",
        cleaned_title=metadata.title,
        abstract=abstract,
        tags=_dedupe_strings(related_topics, limit=8),
        outline=outline,
        sections=sections,
        concepts=list(concepts_by_slug.values()),
        claims=list(claims_by_slug.values()),
        limitations=_dedupe_strings(limitations),
        open_questions=_dedupe_strings(open_questions),
        related_topics=_dedupe_strings(related_topics),
        model=model,
    )


def build_extractive_chunk_analysis(chunk: str, index: int) -> dict[str, Any]:
    lines = [line.strip() for line in chunk.splitlines() if line.strip()]
    first_timestamp = 0
    if lines:
        match = re.match(r"^\[(\d+)s\]", lines[0])
        if match:
            first_timestamp = int(match.group(1))
    cleaned_lines = [re.sub(r"^\[\d+s\]\s*", "", line).strip() for line in lines]
    summary_text = " ".join(cleaned_lines)
    bullets = [line for line in cleaned_lines[:8] if line]
    return {
        "heading": f"Transcript fragment {index}",
        "summary": summary_text[:900],
        "timestamp_seconds": first_timestamp,
        "bullets": bullets,
        "concepts": [],
        "claims": [],
        "limitations": ["LLM chunk response was invalid JSON; this section uses extractive transcript coverage."],
        "open_questions": [],
        "related_topics": [],
    }


class LMStudioAnalyzer:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        qwen_model: str,
        gemma_model: str,
        max_prompt_chars: int = 3500,
        timeout_seconds: float = 120,
        retries: int = 1,
        deep_graph_max_output_tokens: int = 1500,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.qwen_model = qwen_model
        self.gemma_model = gemma_model
        self.max_prompt_chars = max_prompt_chars
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.deep_graph_max_output_tokens = deep_graph_max_output_tokens

    def analyze(self, metadata: VideoMetadata, transcript: Transcript) -> Analysis:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for LM Studio integration") from exc

        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx package is required for LM Studio integration") from exc

        client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout_seconds,
            http_client=httpx.Client(trust_env=False),
        )
        if len(transcript.full_text()) > self.max_prompt_chars:
            return self._analyze_long_transcript(client, metadata, transcript)
        messages = [
            {
                "role": "system",
                "content": (
                    "Return only valid JSON for a scientific Markdown note. "
                    "Required keys: language, cleaned_title, abstract, tags, outline, "
                    "sections, concepts, claims, limitations, open_questions, related_topics. "
                    "Use timestamp_seconds=0 when a timestamp is unknown."
                ),
            },
            {
                "role": "user",
                "content": self._prompt(metadata, transcript),
            },
        ]
        response = self._create_completion(
            client,
            model=self.qwen_model,
            messages=messages,
            temperature=0.2,
            response_format=build_analysis_json_schema_response_format(),
        )
        content = response.choices[0].message.content or "{}"
        return parse_analysis_response(content, model=self.qwen_model)

    def analyze_chunk(
        self,
        metadata: VideoMetadata,
        transcript: Transcript,
        *,
        chunk_index: int,
        total_chunks: int,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for LM Studio integration") from exc

        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx package is required for LM Studio integration") from exc

        client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=timeout_seconds or self.timeout_seconds,
            http_client=httpx.Client(trust_env=False),
        )
        lines = [
            f"[{index}] [{int(segment.start)}s] {segment.text}"
            for index, segment in enumerate(transcript.segments)
        ]
        prompt_prefix = (
            f"Video title: {metadata.title}\n"
            f"Transcript language: {transcript.language or 'unknown'}\n"
            f"Chunk: {chunk_index + 1}/{total_chunks}\n"
            "Analyze only this transcript slice. Return concrete details, not a global summary. "
            "Write the analysis in the same language as the transcript.\n\n"
        )
        budget = max(500, self.max_prompt_chars - len(prompt_prefix) - 200)
        transcript_text = "\n".join(lines)
        if len(transcript_text) > budget:
            transcript_text = transcript_text[:budget].rsplit("\n", 1)[0] or transcript_text[:budget]
        messages = [
            {
                "role": "system",
                "content": (
                    "Analyze this transcript chunk as one section of a scientific note. "
                    "Return only valid JSON matching the schema. Extract concrete details; do not over-compress. "
                    "Use the transcript language for every human-readable field; for Russian transcripts, write Russian."
                ),
            },
            {"role": "user", "content": f"{prompt_prefix}{transcript_text}"},
        ]
        response = self._create_completion(
            client,
            model=self.gemma_model,
            messages=messages,
            temperature=0.2,
            max_tokens=1000,
            response_format=build_chunk_analysis_response_format(),
        )
        content = response.choices[0].message.content or "{}"
        try:
            data = json.loads(extract_json_payload(content))
        except json.JSONDecodeError as exc:
            raise AnalysisValidationError("Chunk analysis response is not valid JSON") from exc
        summary = str(data.get("summary") or "").strip()
        heading = str(data.get("heading") or f"Chunk {chunk_index + 1}").strip()
        if not summary:
            raise AnalysisValidationError("Chunk analysis summary is empty")
        return {
            "summary": summary,
            "outline": [heading, *[str(item) for item in data.get("bullets", []) if str(item).strip()]],
            "concepts": data.get("concepts", []),
            "claims": data.get("claims", []),
            "evidence_segment_indexes": list(range(len(transcript.segments))),
            "limitations": data.get("limitations", []),
            "open_questions": data.get("open_questions", []),
            "related_topics": data.get("related_topics", []),
            "prompt_version": "analysis-chunk-v1",
            "model": self.gemma_model,
        }

    def _create_completion(self, client, **kwargs):
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                return client.chat.completions.create(**kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt < self.retries:
                    time.sleep(1)
        raise RuntimeError(f"LM Studio request failed: {last_exc}") from last_exc

    def extract_deep_graph_chunks(
        self,
        metadata: VideoMetadata,
        transcript: Transcript,
        max_chunk_chars: int = 2500,
    ) -> list[dict[str, Any]]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for LM Studio integration") from exc

        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx package is required for LM Studio integration") from exc

        client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout_seconds,
            http_client=httpx.Client(trust_env=False),
        )
        effective_chunk_chars = max(300, min(max_chunk_chars, max(300, self.max_prompt_chars - 1000)))
        chunks = chunk_transcript(transcript, max_chars=effective_chunk_chars)
        results: list[dict[str, Any]] = []
        for index, chunk in enumerate(chunks, start=1):
            chunk_id = f"chunk-{index}"
            results.extend(self._extract_deep_graph_chunk(client, metadata, chunk, chunk_id))
        return results

    def _extract_deep_graph_chunk(self, client, metadata: VideoMetadata, chunk: str, chunk_id: str) -> list[dict[str, Any]]:
        prompt_prefix = f"Video id: {metadata.video_id}\nVideo title: {metadata.title}\nChunk id: {chunk_id}\n"
        budget = max(300, self.max_prompt_chars - len(prompt_prefix) - 200)
        prompt_chunk = chunk if len(chunk) <= budget else chunk[:budget].rsplit("\n", 1)[0] or chunk[:budget]
        last_error: str | None = None
        compact_prompt = False
        for attempt in range(self.retries + 1):
            messages = [
                {
                    "role": "system",
                    "content": self._deep_graph_system_prompt(compact=compact_prompt),
                },
                {
                    "role": "user",
                    "content": f"{prompt_prefix}{prompt_chunk}",
                },
            ]
            try:
                response = self._create_completion(
                    client,
                    model=self.qwen_model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=self.deep_graph_max_output_tokens,
                    response_format=build_deep_graph_chunk_response_format(),
                )
            except Exception as exc:
                if self._is_context_length_error(exc) and len(prompt_chunk) > 600:
                    return self._split_and_extract_deep_graph_chunk(client, metadata, prompt_chunk, chunk_id)
                return [
                    rejected_deep_graph_chunk(
                        metadata.video_id,
                        chunk_id,
                        f"Deep graph request failed: {exc}",
                    )
                ]
            content = response.choices[0].message.content or "{}"
            parsed = parse_deep_graph_chunk_response(
                content,
                expected_video_id=metadata.video_id,
                expected_chunk_id=chunk_id,
            )
            if not parsed.get("rejected"):
                return [parsed]
            last_error = str(parsed.get("error") or "invalid deep graph response")
            if attempt < self.retries:
                prompt_chunk = self._shrink_prompt_chunk(prompt_chunk)
                compact_prompt = True
        return [
            rejected_deep_graph_chunk(
                metadata.video_id,
                chunk_id,
                last_error or "invalid deep graph response",
            )
        ]

    @staticmethod
    def _deep_graph_system_prompt(compact: bool = False) -> str:
        if compact:
            return (
                "Extract a compact deep knowledge graph from this transcript chunk. "
                "Return only valid JSON matching the schema. "
                "Prefer fewer high-confidence items over many items. "
                "Use at most 3 concepts, 3 claims, 3 evidence items, 2 relations, "
                "1 method, 1 tool, 1 metric, 1 risk, and 1 question."
            )
        return (
            "Extract a deep knowledge graph from this transcript chunk. "
            "Return only valid JSON matching the schema. Use local ids inside the chunk."
        )

    @staticmethod
    def _shrink_prompt_chunk(chunk: str) -> str:
        if not chunk:
            return " "
        stripped = chunk.strip()
        if len(stripped) <= 1:
            return stripped or chunk[:1]
        target = max(600, len(stripped) // 2) if len(stripped) > 600 else max(1, len(stripped) // 2)
        target = min(target, len(stripped) - 1)
        candidate = stripped[:target]
        if "\n" in candidate:
            candidate = candidate.rsplit("\n", 1)[0].strip()
        candidate = candidate.strip()
        if candidate:
            return candidate
        return stripped[:target].strip() or stripped[:1]

    def _split_and_extract_deep_graph_chunk(
        self,
        client,
        metadata: VideoMetadata,
        chunk: str,
        chunk_id: str,
    ) -> list[dict[str, Any]]:
        lines = chunk.splitlines()
        if len(lines) < 2:
            half = max(1, len(chunk) // 2)
            parts = [chunk[:half], chunk[half:]]
        else:
            midpoint = max(1, len(lines) // 2)
            parts = ["\n".join(lines[:midpoint]), "\n".join(lines[midpoint:])]
        results: list[dict[str, Any]] = []
        for index, part in enumerate(parts, start=1):
            if part.strip():
                results.extend(self._extract_deep_graph_chunk(client, metadata, part, f"{chunk_id}.{index}"))
        return results

    @staticmethod
    def _is_context_length_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return any(token in message for token in ("context length", "context_length", "maximum context", "too many tokens"))

    def _analyze_long_transcript(self, client, metadata: VideoMetadata, transcript: Transcript) -> Analysis:
        chunk_size = max(1500, self.max_prompt_chars - 1000)
        chunks = chunk_transcript(transcript, max_chars=chunk_size)
        chunk_analyses: list[dict[str, Any]] = []
        for index, chunk in enumerate(chunks, start=1):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Analyze this transcript chunk as one section of a scientific note. "
                        "Return only valid JSON matching the schema. Extract concrete details; do not over-compress."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Video: {metadata.title}\nChunk {index}/{len(chunks)}:\n{chunk}",
                },
            ]
            response = self._create_completion(
                client,
                model=self.qwen_model,
                messages=messages,
                temperature=0.2,
                max_tokens=1000,
                response_format=build_chunk_analysis_response_format(),
            )
            content = response.choices[0].message.content or "{}"
            try:
                chunk_analyses.append(json.loads(extract_json_payload(content)))
            except json.JSONDecodeError:
                chunk_analyses.append(build_extractive_chunk_analysis(chunk, index))
        return merge_chunk_analyses(metadata, chunk_analyses, model=self.qwen_model)

    def _prompt(self, metadata: VideoMetadata, transcript: Transcript) -> str:
        chunks = chunk_transcript(transcript, max_chars=max(20, self.max_prompt_chars // 4))
        chunk_texts = [f"Chunk {index}:\n{chunk}" for index, chunk in enumerate(chunks, start=1)]
        if sum(len(chunk) for chunk in chunks) > self.max_prompt_chars:
            selected: list[str] = []
            size = 0
            for chunk in chunk_texts:
                if len(selected) >= 2 and size + len(chunk) > self.max_prompt_chars:
                    break
                selected.append(chunk)
                size += len(chunk)
            transcript_text = "\n\n".join(selected)
            transcript_header = "Transcript chunks:"
        else:
            transcript_text = transcript.full_text()
            transcript_header = "Transcript:"
        prompt_prefix = (
            f"Video title: {metadata.title}\n"
            f"Channel: {metadata.channel or 'unknown'}\n"
            f"URL: {metadata.canonical_url}\n\n"
            "Analyze the transcript in a scientific style. Preserve timestamps by using "
            "timestamp_seconds for each section where possible.\n\n"
        )
        budget = max(500, self.max_prompt_chars - len(prompt_prefix) - len(transcript_header) - 2)
        if len(transcript_text) > budget:
            transcript_text = transcript_text[:budget].rsplit("\n", 1)[0] or transcript_text[:budget]
        return f"{prompt_prefix}{transcript_header}\n{transcript_text}"
