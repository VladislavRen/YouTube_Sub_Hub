from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from yt_sub_hub.llm import extract_json_payload
from yt_sub_hub.models import VideoMetadata

VISUAL_PROMPT_VERSION = "vision-frame-v1"
VISUAL_FRAME_ANALYSIS_REQUIRED_FIELDS = [
    "screen_type",
    "summary",
    "visible_objects",
    "chart_elements",
    "explanation",
    "uncertainties",
    "confidence",
]


@runtime_checkable
class VisionProvider(Protocol):
    def analyze_frame(
        self,
        image_path: Path | str,
        metadata: VideoMetadata | dict[str, Any],
        transcript_context: dict[str, Any] | str,
        timestamp_seconds: float,
        question: str | None = None,
    ) -> dict[str, Any]: ...


def build_visual_frame_response_format() -> dict[str, Any]:
    string_array = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "visual_frame_analysis",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": VISUAL_FRAME_ANALYSIS_REQUIRED_FIELDS,
                "properties": {
                    "screen_type": {"type": "string"},
                    "summary": {"type": "string"},
                    "visible_objects": string_array,
                    "chart_elements": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["ticker", "timeframe", "patterns", "levels", "indicators"],
                        "properties": {
                            "ticker": {"type": "string"},
                            "timeframe": {"type": "string"},
                            "patterns": string_array,
                            "levels": string_array,
                            "indicators": string_array,
                        },
                    },
                    "explanation": {"type": "string"},
                    "uncertainties": string_array,
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
    }


def build_visual_frame_analysis_response_format() -> dict[str, Any]:
    return build_visual_frame_response_format()


def parse_visual_frame_response(payload: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(payload, str):
        try:
            data = json.loads(extract_json_payload(payload))
        except json.JSONDecodeError as exc:
            raise ValueError("Vision response is not valid JSON") from exc
    else:
        data = payload
    if not isinstance(data, dict):
        raise ValueError("Vision response must be a JSON object")
    return data


def parse_visual_frame_analysis_response(
    payload: dict[str, Any] | str,
    *,
    video_id: str,
    timestamp_seconds: float,
    frame_path: Path | str,
    model: str,
    transcript_context: dict[str, Any] | str,
) -> dict[str, Any]:
    raw = parse_visual_frame_response(payload)
    return {
        "video_id": str(video_id),
        "timestamp_seconds": float(timestamp_seconds),
        "frame_path": str(frame_path),
        "model": str(model),
        "screen_type": _clean_string(raw.get("screen_type")) or "unknown",
        "summary": _clean_string(raw.get("summary")),
        "visible_objects": _string_list(raw.get("visible_objects")),
        "chart_elements": _normalize_chart_elements(raw.get("chart_elements")),
        "explanation": _clean_string(raw.get("explanation")),
        "transcript_context": transcript_context,
        "uncertainties": _string_list(raw.get("uncertainties")),
        "confidence": _confidence(raw.get("confidence")),
    }


class LMStudioVisionProvider:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int = 1200,
        timeout_seconds: float = 120,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.last_raw_response: dict[str, Any] | None = None

    def analyze_frame(
        self,
        image_path: Path | str,
        metadata: VideoMetadata | dict[str, Any],
        transcript_context: dict[str, Any] | str,
        timestamp_seconds: float,
        question: str | None = None,
    ) -> dict[str, Any]:
        model = self.model.strip()
        if not model:
            raise ValueError("Vision model id is required for frame analysis.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for LM Studio vision integration") from exc
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx package is required for LM Studio vision integration") from exc

        client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout_seconds,
            http_client=httpx.Client(trust_env=False),
        )
        frame_path = Path(image_path)
        prompt = self._prompt(metadata, transcript_context, timestamp_seconds, question)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return only valid JSON that matches the visual_frame_analysis schema. "
                        "Describe only what is visible in this single frame and use the transcript only as nearby context. "
                        "Do not provide financial advice."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": self._image_data_url(frame_path)},
                        },
                    ],
                },
            ],
            temperature=0.1,
            max_tokens=self.max_tokens,
            response_format=build_visual_frame_analysis_response_format(),
        )
        content = response.choices[0].message.content or "{}"
        self.last_raw_response = {
            "model": model,
            "prompt_version": VISUAL_PROMPT_VERSION,
            "content": content,
        }
        return parse_visual_frame_analysis_response(
            content,
            video_id=_metadata_value(metadata, "video_id"),
            timestamp_seconds=timestamp_seconds,
            frame_path=frame_path,
            model=model,
            transcript_context=transcript_context,
        )

    @staticmethod
    def _image_data_url(image_path: Path) -> str:
        suffix = image_path.suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    @staticmethod
    def _prompt(
        metadata: VideoMetadata | dict[str, Any],
        transcript_context: dict[str, Any] | str,
        timestamp_seconds: float,
        question: str | None,
    ) -> str:
        question_block = f"\nQuestion: {question}" if question else ""
        context = (
            json.dumps(transcript_context, ensure_ascii=True, sort_keys=True)
            if isinstance(transcript_context, dict)
            else transcript_context
        )
        return (
            f"Video id: {_metadata_value(metadata, 'video_id')}\n"
            f"Video title: {_metadata_value(metadata, 'title')}\n"
            f"Channel: {_metadata_value(metadata, 'channel') or 'unknown'}\n"
            f"Timestamp seconds: {timestamp_seconds}\n\n"
            f"Transcript context:\n{context}\n"
            f"{question_block}\n\n"
            "Return a compact structured description with screen_type, summary, visible_objects, "
            "chart_elements, explanation, uncertainties, and confidence."
        )


def _metadata_value(metadata: VideoMetadata | dict[str, Any], key: str) -> str:
    value = metadata.get(key, "") if isinstance(metadata, dict) else getattr(metadata, key, "")
    return "" if value is None else str(value)


def _normalize_chart_elements(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, list):
            normalized[str(key)] = [str(element) for element in item]
        elif item is not None:
            normalized[str(key)] = str(item)
    return normalized


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    return [str(item) for item in values]


def _clean_string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


__all__ = [
    "LMStudioVisionProvider",
    "VISUAL_FRAME_ANALYSIS_REQUIRED_FIELDS",
    "VISUAL_PROMPT_VERSION",
    "VisionProvider",
    "build_visual_frame_analysis_response_format",
    "build_visual_frame_response_format",
    "parse_visual_frame_analysis_response",
    "parse_visual_frame_response",
]
