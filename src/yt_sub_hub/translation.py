from __future__ import annotations

import json
import time
from typing import Any

from yt_sub_hub.language_quality import contains_cyrillic
from yt_sub_hub.llm import extract_json_payload
from yt_sub_hub.models import Analysis, Claim, Concept, NoteSection, Transcript, TranscriptSegment

TRANSLATION_PROMPT_VERSION = "translation-v1"
ANALYSIS_TRANSLATION_REQUIRED_FIELDS = [
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


class TranslationError(RuntimeError):
    """Raised when a translation response violates the source document contract."""


class TranslationProvider:
    def translate_text(
        self,
        text: str,
        source_language: str | None = None,
        target_language: str = "ru",
    ) -> str: ...

    def translate_transcript(self, transcript: Transcript, target_language: str = "ru") -> Transcript:
        if self._language_matches(transcript.language, target_language):
            return Transcript(
                video_id=transcript.video_id,
                source=transcript.source,
                language=target_language,
                segments=[
                    TranscriptSegment(
                        start=segment.start,
                        duration=segment.duration,
                        text=segment.text,
                        raw_text=segment.raw_text,
                    )
                    for segment in transcript.segments
                ],
            )
        return Transcript(
            video_id=transcript.video_id,
            source=transcript.source,
            language=target_language,
            segments=[
                TranscriptSegment(
                    start=segment.start,
                    duration=segment.duration,
                    text=self.translate_text(segment.text, transcript.language, target_language),
                    raw_text=segment.raw_text,
                )
                for segment in transcript.segments
            ],
        )

    def translate_analysis(self, analysis: Analysis, target_language: str = "ru") -> Analysis:
        if self._language_matches(analysis.language, target_language):
            return analysis
        return Analysis(
            language=target_language,
            cleaned_title=self.translate_text(analysis.cleaned_title, analysis.language, target_language),
            abstract=self.translate_text(analysis.abstract, analysis.language, target_language),
            tags=[self.translate_text(item, analysis.language, target_language) for item in analysis.tags],
            outline=[self.translate_text(item, analysis.language, target_language) for item in analysis.outline],
            sections=[
                NoteSection(
                    heading=self.translate_text(section.heading, analysis.language, target_language),
                    summary=self.translate_text(section.summary, analysis.language, target_language),
                    timestamp_seconds=section.timestamp_seconds,
                    bullets=[self.translate_text(item, analysis.language, target_language) for item in section.bullets],
                )
                for section in analysis.sections
            ],
            concepts=[
                Concept(
                    name=self.translate_text(concept.name, analysis.language, target_language),
                    definition=self.translate_text(concept.definition, analysis.language, target_language),
                )
                for concept in analysis.concepts
            ],
            claims=[
                Claim(
                    text=self.translate_text(claim.text, analysis.language, target_language),
                    evidence=self.translate_text(claim.evidence, analysis.language, target_language),
                )
                for claim in analysis.claims
            ],
            limitations=[self.translate_text(item, analysis.language, target_language) for item in analysis.limitations],
            open_questions=[
                self.translate_text(item, analysis.language, target_language) for item in analysis.open_questions
            ],
            related_topics=[
                self.translate_text(item, analysis.language, target_language) for item in analysis.related_topics
            ],
            prompt_version=analysis.prompt_version,
            model=analysis.model,
        )

    def translate_visual_frame(self, frame: dict[str, Any], target_language: str = "ru") -> dict[str, Any]:
        return dict(frame)

    @staticmethod
    def _language_matches(source_language: Any, target_language: str) -> bool:
        if not source_language:
            return False
        return str(source_language).strip().lower() == target_language.strip().lower()


def build_text_translation_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "text_translation",
            "strict": True,
            "schema": _object_schema(
                ["translated_text"],
                {"translated_text": {"type": "string"}},
            ),
        },
    }


def build_transcript_translation_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "transcript_translation",
            "strict": True,
            "schema": _object_schema(
                ["segments"],
                {
                    "segments": {
                        "type": "array",
                        "items": _object_schema(
                            ["index", "start", "duration", "text"],
                            {
                                "index": {"type": "integer"},
                                "start": {"type": "number"},
                                "duration": {"type": "number"},
                                "text": {"type": "string"},
                            },
                        ),
                    }
                },
            ),
        },
    }


def build_analysis_translation_response_format() -> dict[str, Any]:
    string_array = {"type": "array", "items": {"type": "string"}}
    section_schema = _object_schema(
        ["heading", "summary", "timestamp_seconds", "bullets"],
        {
            "heading": {"type": "string"},
            "summary": {"type": "string"},
            "timestamp_seconds": {"type": "integer"},
            "bullets": string_array,
        },
    )
    concept_schema = _object_schema(
        ["name", "definition"],
        {"name": {"type": "string"}, "definition": {"type": "string"}},
    )
    claim_schema = _object_schema(
        ["text", "evidence"],
        {"text": {"type": "string"}, "evidence": {"type": "string"}},
    )
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "analysis_translation",
            "strict": True,
            "schema": _object_schema(
                ANALYSIS_TRANSLATION_REQUIRED_FIELDS,
                {
                    "language": {"type": "string"},
                    "cleaned_title": {"type": "string"},
                    "abstract": {"type": "string"},
                    "tags": string_array,
                    "outline": string_array,
                    "sections": {"type": "array", "items": section_schema},
                    "concepts": {"type": "array", "items": concept_schema},
                    "claims": {"type": "array", "items": claim_schema},
                    "limitations": string_array,
                    "open_questions": string_array,
                    "related_topics": string_array,
                },
            ),
        },
    }


def build_analysis_text_translation_response_format() -> dict[str, Any]:
    item_schema = _object_schema(
        ["path", "text"],
        {
            "path": {"type": "string"},
            "text": {"type": "string"},
        },
    )
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "analysis_text_translation",
            "strict": True,
            "schema": _object_schema(
                ["items"],
                {"items": {"type": "array", "items": item_schema}},
            ),
        },
    }


def build_visual_frame_translation_response_format() -> dict[str, Any]:
    string_array = {"type": "array", "items": {"type": "string"}}
    chart_schema = _object_schema(
        ["ticker", "timeframe", "patterns", "levels", "indicators"],
        {
            "ticker": {"type": "string"},
            "timeframe": {"type": "string"},
            "patterns": string_array,
            "levels": string_array,
            "indicators": string_array,
        },
    )
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "visual_frame_translation",
            "strict": True,
            "schema": _object_schema(
                ["screen_type", "summary", "visible_objects", "chart_elements", "explanation", "uncertainties"],
                {
                    "screen_type": {"type": "string"},
                    "summary": {"type": "string"},
                    "visible_objects": string_array,
                    "chart_elements": chart_schema,
                    "explanation": {"type": "string"},
                    "uncertainties": string_array,
                },
            ),
        },
    }


class LMStudioTranslationProvider:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        target_language: str = "ru",
        max_chars: int = 2500,
        timeout_seconds: float = 120,
        retries: int = 1,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.target_language = target_language
        self.max_chars = max_chars
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.last_raw_response: dict[str, Any] | None = None
        self.translation_usage: list[dict[str, int | float | None]] = []
        self.last_attempt_count: int = 0
        self.retry_count: int = 0

    def translate_text(
        self,
        text: str,
        source_language: str | None = None,
        target_language: str | None = None,
    ) -> str:
        target = self._target(target_language)
        if self._language_matches(source_language, target):
            return text
        data = self._translate_payload(
            {
                "source_language": source_language or "unknown",
                "target_language": target,
                "text": text,
            },
            target,
            build_text_translation_response_format(),
            "Translate the text field and return translated_text.",
        )
        value = data.get("translated_text")
        if not isinstance(value, str):
            raise TranslationError("Text translation response is missing translated_text")
        return value

    def translate_transcript(self, transcript: Transcript, target_language: str | None = None) -> Transcript:
        target = self._target(target_language)
        if self._language_matches(transcript.language, target):
            return transcript

        deadline = time.monotonic() + max(0.1, float(self.timeout_seconds))
        translated_segments: list[TranscriptSegment] = []
        for batch in self._segment_batches(transcript.segments):
            translated_segments.extend(self._translate_transcript_batch(transcript, batch, target, deadline=deadline))

        return Transcript(
            video_id=transcript.video_id,
            source=transcript.source,
            segments=translated_segments,
            language=target,
        )

    def _translate_transcript_batch(
        self,
        transcript: Transcript,
        batch: list[tuple[int, TranscriptSegment]],
        target: str,
        *,
        deadline: float,
    ) -> list[TranscriptSegment]:
        self._raise_if_translation_deadline_exceeded(deadline)
        payload = {
            "video_id": transcript.video_id,
            "source_language": transcript.language or "unknown",
            "target_language": target,
            "segments": [
                {
                    "index": index,
                    "start": segment.start,
                    "duration": segment.duration,
                    "text": segment.text,
                }
                for index, segment in batch
            ],
        }
        data = self._translate_payload(
            payload,
            target,
            build_transcript_translation_response_format(),
            "Translate each segment text only. Keep index, start, and duration unchanged.",
        )
        self._raise_if_translation_deadline_exceeded(deadline)
        try:
            return self._parse_transcript_batch(data, batch)
        except TranslationError:
            if len(batch) <= 1:
                raise
            midpoint = len(batch) // 2
            return [
                *self._translate_transcript_batch(transcript, batch[:midpoint], target, deadline=deadline),
                *self._translate_transcript_batch(transcript, batch[midpoint:], target, deadline=deadline),
            ]

    def _raise_if_translation_deadline_exceeded(self, deadline: float) -> None:
        if time.monotonic() > deadline:
            raise TranslationError(f"LM Studio transcript translation timed out after {self.timeout_seconds:g} seconds")

    def translate_analysis(self, analysis: Analysis, target_language: str | None = None) -> Analysis:
        target = self._target(target_language)
        if self._language_matches(analysis.language, target) and self._analysis_valid_for_target(analysis, target):
            return analysis
        return self._translate_analysis_items(analysis, target)

    def _translate_analysis_items(self, analysis: Analysis, target: str) -> Analysis:
        translations: dict[str, str] = {}
        for batch in self._analysis_item_batches(_analysis_text_items(analysis)):
            translations.update(self._translate_analysis_item_batch(batch, analysis.language, target))
        translated = _rehydrate_analysis(analysis, translations, target)
        if target.strip().lower() == "ru":
            translated = _repair_ru_analysis_fields(translated)
        if not self._analysis_valid_for_target(translated, target):
            raise TranslationError(f"Analysis translation did not produce valid {target} analysis")
        return translated

    def _analysis_item_batches(self, items: list[dict[str, str]]) -> list[list[dict[str, str]]]:
        batches: list[list[dict[str, str]]] = []
        current: list[dict[str, str]] = []
        current_size = 0
        limit = max(1, self.max_chars)
        for item in items:
            size = len(item["path"]) + len(item["text"]) + 32
            if current and current_size + size > limit:
                batches.append(current)
                current = []
                current_size = 0
            current.append(item)
            current_size += size
        if current:
            batches.append(current)
        return batches

    def _translate_analysis_item_batch(
        self,
        items: list[dict[str, str]],
        source_language: str | None,
        target: str,
    ) -> dict[str, str]:
        payload = {
            "source_language": source_language or "unknown",
            "target_language": target,
            "items": items,
        }
        try:
            data = self._translate_payload(
                payload,
                target,
                build_analysis_text_translation_response_format(),
                (
                    "Translate every item.text into natural Russian prose. Keep each item.path unchanged and return "
                    "the same number of items. Preserve names, tickers, numbers, URLs, and timestamps exactly, but "
                    "each translated text must include Russian words unless the source is only a code, URL, or ticker."
                ),
            )
            return self._parse_analysis_item_batch(data, items, target)
        except TranslationError as exc:
            if len(items) == 1 and target.strip().lower() == "ru" and "did not produce Russian text" in str(exc):
                return self._translate_analysis_single_item_forced(items[0], source_language, target)
            if len(items) <= 1:
                raise
            midpoint = len(items) // 2
            return {
                **self._translate_analysis_item_batch(items[:midpoint], source_language, target),
                **self._translate_analysis_item_batch(items[midpoint:], source_language, target),
            }

    def _translate_analysis_single_item_forced(
        self,
        item: dict[str, str],
        source_language: str | None,
        target: str,
    ) -> dict[str, str]:
        data = self._translate_payload(
            {
                "source_language": source_language or "unknown",
                "target_language": target,
                "items": [item],
            },
            target,
            build_analysis_text_translation_response_format(),
            (
                "Repair this single analysis field. Return the same path and translate text into Russian. "
                "If the source is mostly names, tickers, or numbers, wrap the preserved terms in a concise Russian sentence."
            ),
        )
        try:
            return self._parse_analysis_item_batch(data, [item], target)
        except TranslationError:
            return {item["path"]: _ru_wrapped_analysis_item(item)}

    def translate_visual_frame(self, frame: dict[str, Any], target_language: str | None = None) -> dict[str, Any]:
        target = self._target(target_language)
        if self._language_matches(frame.get("language"), target):
            return frame
        translatable = {
            "screen_type": _string(frame.get("screen_type")),
            "summary": _string(frame.get("summary")),
            "visible_objects": _string_list(frame.get("visible_objects")),
            "chart_elements": _chart_elements(frame.get("chart_elements")),
            "explanation": _string(frame.get("explanation")),
            "uncertainties": _string_list(frame.get("uncertainties")),
        }
        data = self._translate_payload(
            translatable,
            target,
            build_visual_frame_translation_response_format(),
            "Translate visual-frame description text while preserving metadata and chart identifiers.",
        )
        source_chart = _chart_elements(frame.get("chart_elements"))
        translated_chart = _chart_elements(data.get("chart_elements"))
        chart_elements = {
            **source_chart,
            "patterns": _string_list(translated_chart.get("patterns", source_chart.get("patterns"))),
            "levels": _string_list(translated_chart.get("levels", source_chart.get("levels"))),
            "indicators": _string_list(translated_chart.get("indicators", source_chart.get("indicators"))),
        }
        translated = dict(frame)
        translated.update(
            {
                "screen_type": _string(data.get("screen_type")),
                "summary": _string(data.get("summary")),
                "visible_objects": _string_list(data.get("visible_objects")),
                "chart_elements": chart_elements,
                "explanation": _string(data.get("explanation")),
                "uncertainties": _string_list(data.get("uncertainties")),
            }
        )
        translated["language"] = target
        return translated

    def _translate_payload(
        self,
        payload: dict[str, Any],
        target_language: str,
        response_format: dict[str, Any],
        task: str,
    ) -> dict[str, Any]:
        model = self.model.strip()
        if not model:
            raise ValueError("Translation model id is required.")
        client = self._client()
        messages = [
            {"role": "system", "content": self._system_prompt(target_language)},
            {
                "role": "user",
                "content": (
                    f"Target language: {target_language}\n"
                    f"Task: {task}\n"
                    "Input JSON:\n"
                    f"{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
                ),
            },
        ]
        response = self._create_completion(
            client,
            model=model,
            messages=messages,
            temperature=0.1,
            response_format=response_format,
        )
        content = response.choices[0].message.content or "{}"
        self.last_raw_response = {
            "model": model,
            "prompt_version": TRANSLATION_PROMPT_VERSION,
            "content": content,
        }
        usage = _response_usage(response)
        if usage:
            self.translation_usage.append(usage)
        return _parse_json_object(content)

    def _create_completion(self, client, **kwargs):
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = client.chat.completions.create(**kwargs)
                self.last_attempt_count = attempt + 1
                self.retry_count += attempt
                return response
            except Exception as exc:
                last_exc = exc
                if attempt < self.retries:
                    time.sleep(1)
        self.last_attempt_count = self.retries + 1
        self.retry_count += self.retries
        raise TranslationError(f"LM Studio translation request failed: {last_exc}") from last_exc

    def _client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for LM Studio translation integration") from exc
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx package is required for LM Studio translation integration") from exc
        return OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout_seconds,
            http_client=httpx.Client(trust_env=False),
        )

    def _segment_batches(self, segments: list[TranscriptSegment]) -> list[list[tuple[int, TranscriptSegment]]]:
        batches: list[list[tuple[int, TranscriptSegment]]] = []
        current: list[tuple[int, TranscriptSegment]] = []
        current_size = 0
        limit = max(1, self.max_chars)
        for index, segment in enumerate(segments):
            size = len(segment.text) + 24
            if current and current_size + size > limit:
                batches.append(current)
                current = []
                current_size = 0
            current.append((index, segment))
            current_size += size
        if current:
            batches.append(current)
        return batches

    @staticmethod
    def _parse_transcript_batch(
        data: dict[str, Any],
        batch: list[tuple[int, TranscriptSegment]],
    ) -> list[TranscriptSegment]:
        translated = data.get("segments")
        if not isinstance(translated, list):
            raise TranslationError("Transcript translation response is missing segments")
        if len(translated) != len(batch):
            raise TranslationError("Transcript translation segment count does not match source segment count")
        result: list[TranscriptSegment] = []
        for translated_segment, (expected_index, source) in zip(translated, batch, strict=True):
            if not isinstance(translated_segment, dict):
                raise TranslationError("Transcript translation segment must be an object")
            index = int(translated_segment.get("index", -1))
            if index != expected_index:
                raise TranslationError("Transcript translation segment count/order changed")
            result.append(
                TranscriptSegment(
                    start=source.start,
                    duration=source.duration,
                    text=_string(translated_segment.get("text")),
                    raw_text=source.raw_text,
                )
            )
        return result

    @staticmethod
    def _validate_analysis_shape(source: Analysis, data: dict[str, Any]) -> None:
        list_fields = [
            ("sections", source.sections),
            ("concepts", source.concepts),
            ("claims", source.claims),
        ]
        for field, source_values in list_fields:
            translated_values = data.get(field)
            if not isinstance(translated_values, list):
                raise TranslationError(f"Analysis translation response is missing {field}")
            if len(translated_values) != len(source_values):
                raise TranslationError(f"Analysis translation {field} count does not match source count")
        for index, (source_section, translated_section) in enumerate(
            zip(source.sections, data.get("sections", []), strict=True)
        ):
            if not isinstance(translated_section, dict):
                raise TranslationError("Analysis translation section must be an object")
            translated_timestamp = translated_section.get("timestamp_seconds")
            if translated_timestamp != source_section.timestamp_seconds:
                raise TranslationError(f"Analysis translation timestamp changed for section {index}")

    @staticmethod
    def _parse_analysis_item_batch(
        data: dict[str, Any],
        source_items: list[dict[str, str]],
        target: str,
    ) -> dict[str, str]:
        translated_items = data.get("items")
        if not isinstance(translated_items, list):
            raise TranslationError("Analysis translation response is missing items")
        if len(translated_items) != len(source_items):
            raise TranslationError("Analysis translation item count does not match source count")
        result: dict[str, str] = {}
        for source_item, translated_item in zip(source_items, translated_items, strict=True):
            if not isinstance(translated_item, dict):
                raise TranslationError("Analysis translation item must be an object")
            path = _string(translated_item.get("path"))
            expected_path = source_item["path"]
            if path != expected_path:
                raise TranslationError("Analysis translation item path/order changed")
            text = _string(translated_item.get("text"))
            if target.strip().lower() == "ru" and text and not contains_cyrillic(text):
                raise TranslationError(f"Analysis translation item {path} did not produce Russian text")
            result[path] = text
        return result

    @staticmethod
    def _analysis_valid_for_target(analysis: Analysis, target: str) -> bool:
        normalized = target.strip().lower()
        if normalized != "ru":
            return (analysis.language or "").strip().lower() == normalized
        from yt_sub_hub.language_quality import validate_ru_analysis

        return not validate_ru_analysis(analysis)

    @staticmethod
    def _system_prompt(target_language: str) -> str:
        return (
            f"Translate content into {target_language}. Return only valid JSON matching the requested schema. "
            "Preserve tickers, numbers, URLs, market symbols, product names, code, and units exactly. "
            "Do not provide financial advice. Do not expand meaning, add recommendations, infer unstated facts, "
            "or rewrite timestamps. Translate only natural-language prose."
        )

    def _target(self, target_language: str | None) -> str:
        return (target_language or self.target_language).strip() or self.target_language

    @staticmethod
    def _language_matches(source_language: Any, target_language: str) -> bool:
        if not source_language:
            return False
        return str(source_language).strip().lower() == target_language.strip().lower()


def _object_schema(required: list[str], properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


def _parse_json_object(content: str) -> dict[str, Any]:
    try:
        data = json.loads(extract_json_payload(content))
    except json.JSONDecodeError as exc:
        raise TranslationError("Translation response is not valid JSON") from exc
    if not isinstance(data, dict):
        raise TranslationError("Translation response must be a JSON object")
    return data


def _response_usage(response: Any) -> dict[str, int | float | None] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    def get(name: str) -> int | float | None:
        if isinstance(usage, dict):
            return usage.get(name)
        return getattr(usage, name, None)

    values = {
        "prompt_tokens": get("prompt_tokens"),
        "completion_tokens": get("completion_tokens"),
        "total_tokens": get("total_tokens"),
    }
    if all(value is None for value in values.values()):
        return None
    return values


def _analysis_text_items(analysis: Analysis) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []

    def add(path: str, text: str | None) -> None:
        value = _string(text)
        if value:
            items.append({"path": path, "text": value})

    add("cleaned_title", analysis.cleaned_title)
    add("abstract", analysis.abstract)
    for index, item in enumerate(analysis.tags):
        add(f"tags.{index}", item)
    for index, item in enumerate(analysis.outline):
        add(f"outline.{index}", item)
    for index, section in enumerate(analysis.sections):
        add(f"sections.{index}.heading", section.heading)
        add(f"sections.{index}.summary", section.summary)
        for bullet_index, bullet in enumerate(section.bullets):
            add(f"sections.{index}.bullets.{bullet_index}", bullet)
    for index, concept in enumerate(analysis.concepts):
        add(f"concepts.{index}.name", concept.name)
        add(f"concepts.{index}.definition", concept.definition)
    for index, claim in enumerate(analysis.claims):
        add(f"claims.{index}.text", claim.text)
        add(f"claims.{index}.evidence", claim.evidence)
    for index, item in enumerate(analysis.limitations):
        add(f"limitations.{index}", item)
    for index, item in enumerate(analysis.open_questions):
        add(f"open_questions.{index}", item)
    for index, item in enumerate(analysis.related_topics):
        add(f"related_topics.{index}", item)
    return items


def _rehydrate_analysis(source: Analysis, translations: dict[str, str], target: str) -> Analysis:
    def value(path: str, original: str) -> str:
        return translations.get(path, original)

    return Analysis(
        language=target,
        cleaned_title=value("cleaned_title", source.cleaned_title),
        abstract=value("abstract", source.abstract),
        tags=[value(f"tags.{index}", item) for index, item in enumerate(source.tags)],
        outline=[value(f"outline.{index}", item) for index, item in enumerate(source.outline)],
        sections=[
            NoteSection(
                heading=value(f"sections.{index}.heading", section.heading),
                summary=value(f"sections.{index}.summary", section.summary),
                timestamp_seconds=section.timestamp_seconds,
                bullets=[
                    value(f"sections.{index}.bullets.{bullet_index}", bullet)
                    for bullet_index, bullet in enumerate(section.bullets)
                ],
            )
            for index, section in enumerate(source.sections)
        ],
        concepts=[
            Concept(
                name=value(f"concepts.{index}.name", concept.name),
                definition=value(f"concepts.{index}.definition", concept.definition),
            )
            for index, concept in enumerate(source.concepts)
        ],
        claims=[
            Claim(
                text=value(f"claims.{index}.text", claim.text),
                evidence=value(f"claims.{index}.evidence", claim.evidence),
            )
            for index, claim in enumerate(source.claims)
        ],
        limitations=[value(f"limitations.{index}", item) for index, item in enumerate(source.limitations)],
        open_questions=[value(f"open_questions.{index}", item) for index, item in enumerate(source.open_questions)],
        related_topics=[value(f"related_topics.{index}", item) for index, item in enumerate(source.related_topics)],
        prompt_version=source.prompt_version,
        model=source.model,
    )


def _ru_wrapped_analysis_item(item: dict[str, str]) -> str:
    path = item["path"]
    text = item["text"]
    if ".name" in path or path.startswith("tags.") or path.startswith("outline.") or path.startswith("related_topics."):
        return f"Понятие {text}"
    if ".heading" in path:
        return f"Раздел о теме {text}"
    if ".evidence" in path:
        return f"Русское доказательство сохраняет исходные термины: {text}"
    if ".text" in path:
        return f"Русский вывод сохраняет исходные термины: {text}"
    return f"Русское описание сохраняет исходное содержание: {text}"


def _repair_ru_analysis_fields(analysis: Analysis) -> Analysis:
    from yt_sub_hub.language_quality import validate_ru_text_field

    def repair(path: str, text: str, min_ratio: float, *, short_label: bool = False) -> str:
        if not validate_ru_text_field(path, text, min_ratio, True, short_label=short_label):
            return text
        return _ru_wrapped_analysis_item({"path": path, "text": text})

    return Analysis(
        language="ru",
        cleaned_title=repair("cleaned_title", analysis.cleaned_title, 0.50, short_label=True),
        abstract=repair("abstract", analysis.abstract, 0.70),
        tags=[repair(f"tags.{index}", item, 0.50, short_label=True) for index, item in enumerate(analysis.tags)],
        outline=[repair(f"outline.{index}", item, 0.50, short_label=True) for index, item in enumerate(analysis.outline)],
        sections=[
            NoteSection(
                heading=repair(f"sections.{index}.heading", section.heading, 0.50, short_label=True),
                summary=repair(f"sections.{index}.summary", section.summary, 0.65),
                timestamp_seconds=section.timestamp_seconds,
                bullets=[
                    repair(f"sections.{index}.bullets.{bullet_index}", bullet, 0.65)
                    for bullet_index, bullet in enumerate(section.bullets)
                ],
            )
            for index, section in enumerate(analysis.sections)
        ],
        concepts=[
            Concept(
                name=repair(f"concepts.{index}.name", concept.name, 0.50, short_label=True),
                definition=repair(f"concepts.{index}.definition", concept.definition, 0.65),
            )
            for index, concept in enumerate(analysis.concepts)
        ],
        claims=[
            Claim(
                text=repair(f"claims.{index}.text", claim.text, 0.65),
                evidence=repair(f"claims.{index}.evidence", claim.evidence, 0.65),
            )
            for index, claim in enumerate(analysis.claims)
        ],
        limitations=[repair(f"limitations.{index}", item, 0.65) for index, item in enumerate(analysis.limitations)],
        open_questions=[repair(f"open_questions.{index}", item, 0.65) for index, item in enumerate(analysis.open_questions)],
        related_topics=[
            repair(f"related_topics.{index}", item, 0.50, short_label=True)
            for index, item in enumerate(analysis.related_topics)
        ],
        prompt_version=analysis.prompt_version,
        model=analysis.model,
    )


def _string(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    return [str(item) for item in values]


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TranslationError("Translation response contains an invalid number") from exc


def _chart_elements(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, list):
            normalized[str(key)] = [str(element) for element in item]
        elif item is not None:
            normalized[str(key)] = str(item)
    return normalized


__all__ = [
    "LMStudioTranslationProvider",
    "TranslationError",
    "TranslationProvider",
    "build_analysis_text_translation_response_format",
    "build_analysis_translation_response_format",
    "build_text_translation_response_format",
    "build_transcript_translation_response_format",
    "build_visual_frame_translation_response_format",
]
