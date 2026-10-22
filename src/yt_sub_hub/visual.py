from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Callable

from yt_sub_hub.config import Settings, get_settings
from yt_sub_hub.ingestion import ytdlp_options
from yt_sub_hub.library import LibraryStore
from yt_sub_hub.models import Transcript, VideoMetadata


Runner = Callable[..., Any]


VISION_MODEL_NOT_CONFIGURED = (
    "Vision model is not configured. Set YT_SUB_HUB_VISION_MODEL to a vision-capable LM Studio model."
)


class VisualAnalysisError(RuntimeError):
    def __init__(self, message: str, *, action: str | None = None) -> None:
        self.action = action
        if action:
            super().__init__(f"{message} Action: {action}")
        else:
            super().__init__(message)


@dataclass(frozen=True)
class TranscriptWindow:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VisualFrameAnalysis:
    video_id: str
    timestamp_seconds: float
    frame_path: str
    model: str
    screen_type: str = "unknown"
    summary: str = ""
    visible_objects: list[str] = field(default_factory=list)
    chart_elements: dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    transcript_context: TranscriptWindow | None = None
    uncertainties: list[str] = field(default_factory=list)
    confidence: float | None = None

    @property
    def timestamp_key(self) -> str:
        return timestamp_key(self.timestamp_seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "timestamp_seconds": self.timestamp_seconds,
            "timestamp_key": self.timestamp_key,
            "frame_path": self.frame_path,
            "model": self.model,
            "screen_type": self.screen_type,
            "summary": self.summary,
            "visible_objects": self.visible_objects,
            "chart_elements": self.chart_elements,
            "explanation": self.explanation,
            "transcript_context": self.transcript_context.to_dict() if self.transcript_context else None,
            "uncertainties": self.uncertainties,
            "confidence": self.confidence,
        }


def timestamp_key(seconds: float | int) -> str:
    value = Decimal(str(seconds))
    if value < 0:
        raise ValueError("timestamp must be non-negative")
    whole = int(value)
    millis = int(((value - Decimal(whole)) * Decimal("1000")).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    if millis == 1000:
        whole += 1
        millis = 0
    key = f"{whole:06d}"
    if millis:
        key = f"{key}_{millis:03d}"
    return key


def transcript_window(transcript: Transcript, *, at_seconds: float | int, context_seconds: float | int) -> TranscriptWindow:
    start = max(0.0, float(at_seconds) - float(context_seconds))
    end = float(at_seconds) + float(context_seconds)
    texts = []
    for segment in transcript.segments:
        segment_start = float(segment.start)
        segment_end = segment_start + float(segment.duration)
        if segment_start <= end and segment_end >= start and segment.text:
            texts.append(segment.text)
    return TranscriptWindow(start=_clean_number(start), end=_clean_number(end), text="\n".join(texts))


def transcript_context_window(
    transcript: Transcript,
    timestamp_seconds: float | int,
    context_seconds: float | int,
) -> TranscriptWindow:
    return transcript_window(transcript, at_seconds=timestamp_seconds, context_seconds=context_seconds)


def normalize_visual_analysis(
    raw: dict[str, Any],
    *,
    video_id: str,
    timestamp_seconds: float,
    frame_path: Path | str,
    model: str,
    transcript_context: TranscriptWindow | None = None,
) -> VisualFrameAnalysis:
    chart_elements = _normalize_chart_elements(raw.get("chart_elements"))
    return VisualFrameAnalysis(
        video_id=video_id,
        timestamp_seconds=timestamp_seconds,
        frame_path=str(frame_path),
        model=model,
        screen_type=_clean_string(raw.get("screen_type")) or "unknown",
        summary=_clean_string(raw.get("summary")),
        visible_objects=_string_list(raw.get("visible_objects")),
        chart_elements=chart_elements,
        explanation=_clean_string(raw.get("explanation")),
        transcript_context=transcript_context,
        uncertainties=_string_list(raw.get("uncertainties")),
        confidence=_optional_float(raw.get("confidence")),
    )


def normalize_visual_frame(
    raw: dict[str, Any],
    *,
    video_id: str,
    timestamp_seconds: float,
    frame_path: Path | str,
    model: str,
    transcript_context: TranscriptWindow | dict[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(transcript_context, dict):
        transcript_context = TranscriptWindow(
            start=float(transcript_context.get("start", 0)),
            end=float(transcript_context.get("end", 0)),
            text=str(transcript_context.get("text") or ""),
        )
    return normalize_visual_analysis(
        raw,
        video_id=video_id,
        timestamp_seconds=timestamp_seconds,
        frame_path=frame_path,
        model=model,
        transcript_context=transcript_context,
    ).to_dict()


def build_ffmpeg_frame_command(video_path: Path | str, timestamp_seconds: float | int, output_path: Path | str) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-ss",
        str(timestamp_seconds),
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        str(output_path),
    ]


def extract_frame_with_ffmpeg(
    video_path: Path | str,
    timestamp_seconds: float | int,
    output_path: Path | str,
    *,
    runner: Runner = subprocess.run,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = build_ffmpeg_frame_command(video_path, timestamp_seconds, output)
    try:
        runner(command, check=True, capture_output=True, text=True)
    except Exception as exc:  # pragma: no cover - exact runner exception depends on caller
        raise VisualAnalysisError(
            f"ffmpeg frame extraction failed: {exc}",
            action="Install ffmpeg, verify the video cache path, and retry frame extraction.",
        ) from exc
    if not output.exists():
        raise VisualAnalysisError(
            f"ffmpeg frame extraction failed: expected frame at {output}",
            action="Check ffmpeg output permissions and retry frame extraction.",
        )
    return output


def build_ytdlp_video_command(video_url: str, output_path: Path | str) -> list[str]:
    return [
        "yt-dlp",
        "-f",
        "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "-o",
        str(output_path),
        video_url,
    ]


def download_video_with_ytdlp(video_url: str, output_path: Path | str, *, runner: Runner = subprocess.run) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = build_ytdlp_video_command(video_url, output)
    try:
        runner(command, check=True, capture_output=True, text=True)
    except Exception as exc:  # pragma: no cover - exact runner exception depends on caller
        raise VisualAnalysisError(
            f"yt-dlp video download failed: {exc}",
            action="Install yt-dlp, verify the video URL is reachable, and retry visual analysis.",
        ) from exc
    if not output.exists():
        raise VisualAnalysisError(
            f"yt-dlp video download failed: expected video at {output}",
            action="Check yt-dlp output permissions and retry visual analysis.",
        )
    return output


def analyze_visual_frame(
    video_id: str,
    timestamp_seconds: float,
    *,
    store: Any = None,
    extractor: Callable[..., Path | str] | None = None,
    provider: Callable[..., dict[str, Any]] | None = None,
    force: bool = False,
    model: str = "",
    transcript: Transcript | None = None,
    context_seconds: float = 20,
    question: str | None = None,
    video_url: str | None = None,
) -> dict[str, Any]:
    key = timestamp_key(timestamp_seconds)
    if store is not None and not force and hasattr(store, "load_visual_frame"):
        cached = store.load_visual_frame(video_id, key)
        if cached is not None:
            return cached

    paths = _visual_paths(store, video_id, key)
    frame_path = _path_value(paths, "frame") or Path(f"{key}.jpg")
    video_path = _path_value(paths, "video_cache")
    context = (
        transcript_window(transcript, at_seconds=timestamp_seconds, context_seconds=context_seconds)
        if transcript is not None
        else TranscriptWindow(
            start=_clean_number(max(0.0, float(timestamp_seconds) - float(context_seconds))),
            end=_clean_number(float(timestamp_seconds) + float(context_seconds)),
            text="",
        )
    )

    try:
        if extractor is not None:
            extracted = extractor(
                video_id=video_id,
                timestamp_seconds=timestamp_seconds,
                timestamp_key=key,
                frame_path=frame_path,
                video_path=video_path,
                video_url=video_url,
                paths=paths,
            )
            frame_path = Path(extracted)
        if provider is None:
            raise VisualAnalysisError(
                "visual analysis provider is required",
                action="Pass a provider callback or configure a vision provider before analyzing a frame.",
            )
        raw = provider(
            video_id=video_id,
            timestamp_seconds=timestamp_seconds,
            timestamp_key=key,
            frame_path=frame_path,
            model=model,
            question=question,
            transcript_context=context,
        )
    except VisualAnalysisError:
        raise
    except Exception as exc:
        raise VisualAnalysisError(
            str(exc),
            action="Check the visual analysis provider, model availability, and frame path.",
        ) from exc

    normalized = normalize_visual_analysis(
        raw,
        video_id=video_id,
        timestamp_seconds=timestamp_seconds,
        frame_path=frame_path,
        model=model,
        transcript_context=context,
    ).to_dict()
    if store is not None and hasattr(store, "save_visual_frame"):
        store.save_visual_frame(video_id, key, raw=raw, normalized=normalized)
    return normalized


class FfmpegFrameExtractor:
    def __init__(self, store: LibraryStore, settings: Settings | None = None) -> None:
        self.store = store
        self.settings = settings or get_settings()

    def extract_frame(
        self,
        metadata: VideoMetadata,
        output_path: Path,
        timestamp_seconds: float,
        force: bool = False,
    ) -> Path:
        if metadata.duration_seconds is not None and timestamp_seconds > metadata.duration_seconds:
            raise VisualAnalysisError(
                f"timestamp outside video duration: {timestamp_seconds}s > {metadata.duration_seconds}s",
                action="Choose a timestamp inside the video duration.",
            )
        if output_path.exists() and output_path.stat().st_size > 0 and not force:
            return output_path
        video_path = self.ensure_video_cached(metadata)
        if shutil.which("ffmpeg") is None:
            raise VisualAnalysisError(
                "ffmpeg missing",
                action="Install ffmpeg before extracting video frames.",
            )
        return extract_frame_with_ffmpeg(video_path, timestamp_seconds, output_path)

    def ensure_video_cached(self, metadata: VideoMetadata) -> Path:
        video_path = self.store.visual_artifact_paths(metadata.video_id, "000000").video_cache
        if video_path.exists() and video_path.stat().st_size > 0:
            return video_path
        try:
            import yt_dlp
        except ImportError as exc:
            raise VisualAnalysisError(
                "yt-dlp is required to download video for frame extraction",
                action="Install yt-dlp in the project environment and retry.",
            ) from exc
        video_path.parent.mkdir(parents=True, exist_ok=True)
        output_template = str(video_path.parent / f"{metadata.video_id}.%(ext)s")
        options = ytdlp_options(
            self.settings,
            format="best[height<=480][ext=mp4]/best[ext=mp4]/best[height<=480]/best",
            outtmpl=output_template,
            merge_output_format="mp4",
            quiet=True,
        )
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.download([metadata.canonical_url])
        except Exception as exc:
            raise VisualAnalysisError(
                f"video download failed for {metadata.video_id}: {exc}",
                action="Check YouTube access, cookies settings, and network connectivity.",
            ) from exc
        if not video_path.exists() or video_path.stat().st_size == 0:
            raise VisualAnalysisError(
                f"video download failed: expected {video_path}",
                action="Check yt-dlp output format and permissions.",
            )
        return video_path


class VisualAnalysisService:
    def __init__(
        self,
        settings: Settings | None = None,
        store: LibraryStore | None = None,
        provider: Any | None = None,
        extractor: Any | None = None,
        translation_provider: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or LibraryStore(self.settings.library_dir)
        self.provider = provider
        self.extractor = extractor or FfmpegFrameExtractor(self.store, self.settings)
        self.translation_provider = translation_provider

    def normalized_path(self, video_id: str, timestamp_seconds: float) -> Path:
        return self.store.visual_artifact_paths(video_id, timestamp_key(timestamp_seconds)).normalized

    def visual_note_path(self, video_id: str) -> Path:
        return self.store.video_dir(video_id) / "visual" / "visual_note.md"

    def analyze_frame(
        self,
        video_id: str,
        timestamp_seconds: float,
        question: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        key = timestamp_key(timestamp_seconds)
        cached = self.store.load_visual_frame(video_id, key)
        if cached is not None and self.settings.enable_translation and not self.settings.translation_best_effort:
            if cached.get("language") != self.settings.target_language:
                cached = None
        if cached is not None and not force:
            return cached
        if not self.settings.vision_model:
            raise VisualAnalysisError(VISION_MODEL_NOT_CONFIGURED)
        metadata = self.store.load_video_metadata(video_id)
        if metadata is None:
            raise VisualAnalysisError(
                f"video metadata not found for {video_id}",
                action="Process the video before running analyze-frame.",
            )
        transcript = (
            self.store.load_translated_transcript(video_id, self.settings.target_language)
            if self.settings.enable_translation
            else None
        ) or self.store.load_normalized_transcript(video_id)
        if transcript is None:
            raise VisualAnalysisError(
                f"transcript not found for {video_id}",
                action="Process the video before running analyze-frame.",
            )
        paths = self.store.visual_artifact_paths(video_id, key)
        frame_path = self.extractor.extract_frame(metadata, paths.frame, float(timestamp_seconds), force=force)
        context = transcript_window(
            transcript,
            at_seconds=float(timestamp_seconds),
            context_seconds=self.settings.frame_context_seconds,
        )
        provider = self.provider or self._default_provider()
        analysis_payload = provider.analyze_frame(
            frame_path,
            metadata,
            context.to_dict(),
            float(timestamp_seconds),
            question=question,
        )
        raw_payload = getattr(provider, "last_raw_response", None)
        if raw_payload is None:
            raw_payload = analysis_payload if isinstance(analysis_payload, dict) else {"response": analysis_payload}
        normalized = normalize_visual_frame(
            analysis_payload,
            video_id=video_id,
            timestamp_seconds=float(timestamp_seconds),
            frame_path=frame_path,
            model=self.settings.vision_model,
            transcript_context=context,
        )
        normalized_source = dict(normalized)
        if self.settings.enable_translation:
            try:
                translator = self.translation_provider or self._default_translation_provider()
                normalized = translator.translate_visual_frame(normalized_source, self.settings.target_language)
                normalized.setdefault("language", self.settings.target_language)
                normalized.setdefault("translated_from", normalized_source.get("language") or "unknown")
            except Exception as exc:
                self.store.save_visual_frame(
                    video_id,
                    key,
                    raw=raw_payload,
                    normalized=normalized_source,
                    normalized_source=normalized_source,
                    summary_variant="source",
                )
                if not self.settings.translation_best_effort:
                    raise VisualAnalysisError(str(exc), action="Check the translation provider and retry visual analysis.") from exc
                normalized = normalized_source
            else:
                self.store.save_visual_frame(
                    video_id,
                    key,
                    raw=raw_payload,
                    normalized=normalized_source,
                    normalized_source=normalized_source,
                    summary_variant="source",
                )
        saved = self.store.save_visual_frame(
            video_id,
            key,
            raw=raw_payload,
            normalized=normalized,
            normalized_source=normalized_source if self.settings.enable_translation else None,
        )
        from yt_sub_hub.visual_notes import render_visual_note

        saved.note.write_text(render_visual_note(self.store.load_visual_summary(video_id) or {}), encoding="utf-8")
        self._merge_graph_best_effort(video_id)
        return normalized

    def _default_provider(self):
        from yt_sub_hub.vision import LMStudioVisionProvider

        return LMStudioVisionProvider(
            base_url=self.settings.lm_studio_base_url,
            api_key=self.settings.lm_studio_api_key,
            model=self.settings.vision_model,
            max_tokens=self.settings.visual_max_output_tokens,
            timeout_seconds=self.settings.lm_studio_timeout_seconds,
        )

    def _default_translation_provider(self):
        from yt_sub_hub.translation import LMStudioTranslationProvider

        return LMStudioTranslationProvider(
            self.settings.lm_studio_base_url,
            self.settings.lm_studio_api_key,
            self.settings.translation_model,
            target_language=self.settings.target_language,
            max_chars=self.settings.translation_max_chars,
            timeout_seconds=self.settings.lm_studio_timeout_seconds,
        )

    def _merge_graph_best_effort(self, video_id: str) -> None:
        try:
            from yt_sub_hub.graph_extraction import merge_visual_summary_into_extraction
            from yt_sub_hub.artifact_review import review_ru_artifacts

            with self.store.video_lock(video_id):
                extraction = self.store.load_graph_extraction(video_id)
                summary = self.store.load_visual_summary(video_id)
                if extraction is None or summary is None:
                    return
                paths = self.store.artifact_paths(video_id)
                previous_graph = paths.graph_extraction.read_bytes()
                previous_review = paths.ru_artifact_review.read_bytes() if paths.ru_artifact_review.exists() else None
                try:
                    self.store.save_graph_extraction(video_id, merge_visual_summary_into_extraction(extraction, summary))
                    if previous_review is not None:
                        # The existing review belongs to RU core artifacts,
                        # independently of the current vision translation mode.
                        review = review_ru_artifacts(video_id, self.store)
                        if review.status == "failed":
                            raise VisualAnalysisError("Visual graph merge failed RU artifact review")
                except Exception:
                    paths.graph_extraction.write_bytes(previous_graph)
                    if previous_review is not None:
                        paths.ru_artifact_review.write_bytes(previous_review)
                    raise
        except Exception:
            if not self.settings.visual_best_effort:
                raise


def _visual_paths(store: Any, video_id: str, key: str) -> Any:
    if store is not None and hasattr(store, "visual_artifact_paths"):
        return store.visual_artifact_paths(video_id, key)
    return None


def _path_value(paths: Any, name: str) -> Path | None:
    if paths is None:
        return None
    if isinstance(paths, dict):
        value = paths.get(name)
    else:
        value = getattr(paths, name, None)
    return Path(value) if value is not None else None


def _normalize_chart_elements(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    return {
        "ticker": _clean_string(source.get("ticker")),
        "timeframe": _clean_string(source.get("timeframe")),
        "patterns": _string_list(source.get("patterns")),
        "levels": _string_list(source.get("levels")),
        "indicators": _string_list(source.get("indicators")),
    }


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        values = [value]
    else:
        try:
            values = list(value)
        except TypeError:
            values = [value]
    return [cleaned for item in values if (cleaned := _clean_string(item))]


def _clean_string(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_number(value: float) -> float | int:
    return int(value) if value.is_integer() else value
