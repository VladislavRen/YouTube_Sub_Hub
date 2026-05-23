from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yt_sub_hub.models import Analysis, JobRecord, Transcript, TranscriptSegment, VideoMetadata


@dataclass(frozen=True)
class VideoArtifactPaths:
    video_dir: Path
    note: Path
    metadata: Path
    raw_transcript: Path
    normalized_transcript: Path
    transcript_ru: Path
    analysis: Path
    analysis_source: Path
    analysis_ru: Path
    note_source: Path
    graph_extraction: Path
    graph_extraction_source: Path
    ru_artifact_review: Path


@dataclass(frozen=True)
class VisualArtifactPaths:
    video_cache: Path
    visual_dir: Path
    frame_dir: Path
    frame: Path
    raw: Path
    normalized_source: Path
    normalized: Path
    summary_source: Path
    summary: Path
    note: Path


class LibraryStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def ensure_layout(self) -> None:
        for path in [
            self.root / "videos",
            self.root / "topics",
            self.root / "graph" / "exports",
            self.root / "assets" / "audio",
            self.root / "assets" / "video",
        ]:
            path.mkdir(parents=True, exist_ok=True)
        self._ensure_db()

    @property
    def db_path(self) -> Path:
        return self.root / "index.sqlite3"

    def _connect(self):
        self.root.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path)

    def _ensure_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    video_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT,
                    note_path TEXT,
                    error TEXT,
                    warning TEXT,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS job_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT,
                    message TEXT,
                    created_at REAL NOT NULL
                )
                """
            )

    def video_dir(self, video_id: str) -> Path:
        return self.root / "videos" / video_id

    def artifact_paths(self, video_id: str) -> VideoArtifactPaths:
        video_dir = self.video_dir(video_id)
        return VideoArtifactPaths(
            video_dir=video_dir,
            note=video_dir / "note.md",
            metadata=video_dir / "metadata.json",
            raw_transcript=video_dir / "transcript.raw.json",
            normalized_transcript=video_dir / "transcript.normalized.json",
            transcript_ru=video_dir / "transcript.ru.json",
            analysis=video_dir / "analysis.json",
            analysis_source=video_dir / "analysis.source.json",
            analysis_ru=video_dir / "analysis.ru.json",
            note_source=video_dir / "note.source.md",
            graph_extraction=video_dir / "graph_extraction.json",
            graph_extraction_source=video_dir / "graph_extraction.source.json",
            ru_artifact_review=video_dir / "review" / "ru_artifact_review.json",
        )

    def visual_artifact_paths(self, video_id: str, timestamp_key: str) -> VisualArtifactPaths:
        visual_dir = self.video_dir(video_id) / "visual"
        frame_dir = visual_dir / "frames"
        return VisualArtifactPaths(
            video_cache=self.root / "assets" / "video" / f"{video_id}.mp4",
            visual_dir=visual_dir,
            frame_dir=frame_dir,
            frame=frame_dir / f"{timestamp_key}.jpg",
            raw=visual_dir / f"frame_{timestamp_key}.raw.json",
            normalized_source=visual_dir / f"frame_{timestamp_key}.normalized.source.json",
            normalized=visual_dir / f"frame_{timestamp_key}.normalized.json",
            summary_source=visual_dir / "visual_summary.source.json",
            summary=visual_dir / "visual_summary.json",
            note=visual_dir / "visual_note.md",
        )

    def has_completed_video(self, video_id: str) -> bool:
        paths = self.artifact_paths(video_id)
        return all(
            path.exists()
            for path in [
                paths.note,
                paths.metadata,
                paths.normalized_transcript,
                paths.analysis,
            ]
        )

    def save_video_artifacts(
        self,
        metadata: VideoMetadata,
        raw_transcript: Transcript,
        normalized_transcript: Transcript,
        analysis: Analysis,
        markdown: str,
    ) -> VideoArtifactPaths:
        self.ensure_layout()
        video_dir = self.video_dir(metadata.video_id)
        video_dir.mkdir(parents=True, exist_ok=True)
        paths = self.artifact_paths(metadata.video_id)
        paths.note.write_text(markdown, encoding="utf-8")
        self._write_json(paths.metadata, metadata.to_dict())
        raw_data = raw_transcript.to_dict() if hasattr(raw_transcript, "to_dict") else raw_transcript
        self._write_json(paths.raw_transcript, raw_data)
        self._write_json(paths.normalized_transcript, normalized_transcript.to_dict())
        self._write_json(paths.analysis, analysis.to_dict())
        return paths

    def save_translated_transcript(self, video_id: str, transcript: Transcript, language: str | None = None) -> Path:
        self.ensure_layout()
        video_dir = self.video_dir(video_id)
        video_dir.mkdir(parents=True, exist_ok=True)
        target_language = language or transcript.language or "ru"
        path = video_dir / f"transcript.{target_language}.json"
        self._write_json(path, transcript.to_dict())
        return path

    def load_translated_transcript(self, video_id: str, language: str = "ru") -> Transcript | None:
        path = self.video_dir(video_id) / f"transcript.{language}.json"
        if not path.exists():
            return None
        return self._load_transcript_path(path)

    def save_analysis_artifact(
        self,
        video_id: str,
        analysis: Analysis,
        *,
        variant: str = "default",
        compatibility: bool = False,
        extra_fields: dict[str, Any] | None = None,
    ) -> Path:
        self.ensure_layout()
        video_dir = self.video_dir(video_id)
        video_dir.mkdir(parents=True, exist_ok=True)
        path = self._analysis_path(video_id, variant)
        data = analysis.to_dict()
        if extra_fields:
            data.update(extra_fields)
        self._write_json(path, data)
        if compatibility and path != self.artifact_paths(video_id).analysis:
            self._write_json(self.artifact_paths(video_id).analysis, data)
        return path

    def save_note_artifact(self, video_id: str, markdown: str, *, variant: str = "default") -> Path:
        self.ensure_layout()
        video_dir = self.video_dir(video_id)
        video_dir.mkdir(parents=True, exist_ok=True)
        path = self.artifact_paths(video_id).note_source if variant == "source" else self.artifact_paths(video_id).note
        path.write_text(markdown, encoding="utf-8")
        return path

    def note_path(self, video_id: str) -> Path:
        return self.video_dir(video_id) / "note.md"

    def load_video_metadata(self, video_id: str) -> VideoMetadata | None:
        path = self.video_dir(video_id) / "metadata.json"
        if not path.exists():
            return None
        return VideoMetadata(**self._read_json(path))

    def load_raw_transcript(self, video_id: str) -> Any | None:
        path = self.video_dir(video_id) / "transcript.raw.json"
        if not path.exists():
            return None
        return self._read_json(path)

    def load_normalized_transcript(self, video_id: str) -> Transcript | None:
        path = self.video_dir(video_id) / "transcript.normalized.json"
        if not path.exists():
            return None
        return self._load_transcript_path(path)

    def load_analysis(self, video_id: str, variant: str = "default") -> dict[str, Any] | None:
        path = self._analysis_path(video_id, variant)
        if not path.exists():
            return None
        return self._read_json(path)

    def load_analysis_files(self) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for video_dir in sorted((self.root / "videos").glob("*")):
            metadata_path = video_dir / "metadata.json"
            analysis_path = video_dir / "analysis.json"
            if metadata_path.exists() and analysis_path.exists():
                pairs.append((self._read_json(metadata_path), self._read_json(analysis_path)))
        return pairs

    def save_graph_extraction(self, video_id: str, extraction: dict[str, Any], variant: str = "default") -> Path:
        self.ensure_layout()
        video_dir = self.video_dir(video_id)
        video_dir.mkdir(parents=True, exist_ok=True)
        path = self._graph_extraction_path(video_id, variant)
        self._write_json(path, extraction)
        return path

    def load_graph_extraction(self, video_id: str, variant: str = "default") -> dict[str, Any] | None:
        path = self._graph_extraction_path(video_id, variant)
        if not path.exists():
            return None
        return self._read_json(path)

    def load_graph_extraction_files(self) -> list[dict[str, Any]]:
        extractions: list[dict[str, Any]] = []
        for video_dir in sorted((self.root / "videos").glob("*")):
            path = video_dir / "graph_extraction.json"
            if path.exists():
                extractions.append(self._read_json(path))
        return extractions

    def save_visual_frame(
        self,
        video_id: str,
        timestamp_key: str,
        raw: dict[str, Any],
        normalized: dict[str, Any],
        normalized_source: dict[str, Any] | None = None,
        summary_variant: str = "default",
    ) -> VisualArtifactPaths:
        self.ensure_layout()
        paths = self.visual_artifact_paths(video_id, timestamp_key)
        paths.frame_dir.mkdir(parents=True, exist_ok=True)
        paths.visual_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(paths.raw, raw)
        if normalized_source is not None:
            self._write_json(paths.normalized_source, normalized_source)
        self._write_json(paths.normalized, normalized)
        summary_path = paths.summary_source if summary_variant == "source" else paths.summary
        summary = self.load_visual_summary(video_id, variant=summary_variant) or {
            "schema_version": "visual-summary-v1",
            "video_id": video_id,
            "frames": [],
        }
        frames = [
            frame
            for frame in summary.get("frames", [])
            if str(frame.get("timestamp_key") or "") != timestamp_key
            and float(frame.get("timestamp_seconds", -1)) != float(normalized.get("timestamp_seconds", -2))
        ]
        frame_entry = dict(normalized)
        frame_entry["timestamp_key"] = timestamp_key
        frames.append(frame_entry)
        frames.sort(key=lambda item: float(item.get("timestamp_seconds", 0)))
        summary["video_id"] = video_id
        summary["frames"] = frames
        self._write_json(summary_path, summary)
        return paths

    def load_visual_frame(self, video_id: str, timestamp_key: str) -> dict[str, Any] | None:
        path = self.visual_artifact_paths(video_id, timestamp_key).normalized
        if not path.exists():
            return None
        return self._read_json(path)

    def load_visual_summary(self, video_id: str, variant: str = "default") -> dict[str, Any] | None:
        name = "visual_summary.source.json" if variant == "source" else "visual_summary.json"
        path = self.video_dir(video_id) / "visual" / name
        if not path.exists():
            return None
        return self._read_json(path)

    def load_visual_summaries(self) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for video_dir in sorted((self.root / "videos").glob("*")):
            path = video_dir / "visual" / "visual_summary.json"
            if path.exists():
                summaries.append(self._read_json(path))
        return summaries

    def load_graph_source_files(
        self,
        *,
        prefer_translation: bool = False,
        target_language: str = "ru",
        video_id: str | None = None,
    ) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
        triples: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        for video_dir in sorted((self.root / "videos").glob("*")):
            if video_id is not None and video_dir.name != video_id:
                continue
            metadata_path = video_dir / "metadata.json"
            source_transcript_path = video_dir / "transcript.normalized.json"
            translated_transcript_path = video_dir / f"transcript.{target_language}.json"
            translated_analysis_path = video_dir / f"analysis.{target_language}.json"
            if prefer_translation and translated_transcript_path.exists():
                if not translated_analysis_path.exists():
                    raise RuntimeError(f"Missing {translated_analysis_path.name} for translated graph rebuild")
                transcript_path = translated_transcript_path
                analysis_path = translated_analysis_path
            else:
                transcript_path = source_transcript_path
                analysis_path = video_dir / "analysis.json"
            if metadata_path.exists() and transcript_path.exists() and analysis_path.exists():
                triples.append(
                    (
                        self._read_json(metadata_path),
                        self._read_json(transcript_path),
                        self._read_json(analysis_path),
                    )
                )
        return triples

    @staticmethod
    def _load_transcript_path(path: Path) -> Transcript:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Transcript(
            video_id=data["video_id"],
            source=data["source"],
            language=data.get("language"),
            segments=[TranscriptSegment(**segment) for segment in data.get("segments", [])],
        )

    def _analysis_path(self, video_id: str, variant: str) -> Path:
        paths = self.artifact_paths(video_id)
        if variant == "default":
            return paths.analysis
        if variant == "source":
            return paths.analysis_source
        if variant == "ru":
            return paths.analysis_ru
        return paths.video_dir / f"analysis.{variant}.json"

    def _graph_extraction_path(self, video_id: str, variant: str) -> Path:
        paths = self.artifact_paths(video_id)
        if variant == "source":
            return paths.graph_extraction_source
        return paths.graph_extraction

    def load_jobs(self) -> dict[str, JobRecord]:
        self.ensure_layout()
        if not self.db_path.exists():
            return {}
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT job_id, url, video_id, status, stage, note_path, error, warning FROM jobs"
            ).fetchall()
        return {
            row[0]: JobRecord(
                job_id=row[0],
                url=row[1],
                video_id=row[2],
                status=row[3],
                stage=row[4],
                note_path=row[5],
                error=row[6],
                warning=row[7],
            )
            for row in rows
        }

    def save_job(self, record: JobRecord) -> None:
        self.ensure_layout()
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (job_id, url, video_id, status, stage, note_path, error, warning, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    url=excluded.url,
                    video_id=excluded.video_id,
                    status=excluded.status,
                    stage=excluded.stage,
                    note_path=excluded.note_path,
                    error=excluded.error,
                    warning=excluded.warning,
                    updated_at=excluded.updated_at
                """,
                (
                    record.job_id,
                    record.url,
                    record.video_id,
                    record.status,
                    record.stage,
                    record.note_path,
                    record.error,
                    record.warning,
                    now,
                ),
            )
            conn.execute(
                "INSERT INTO job_history (job_id, status, stage, message, created_at) VALUES (?, ?, ?, ?, ?)",
                (record.job_id, record.status, record.stage, record.error or record.warning, now),
            )

    def mark_stale_running_jobs_failed(self, max_age_seconds: int = 3600) -> list[JobRecord]:
        self.ensure_layout()
        now = time.time()
        cutoff = now - max_age_seconds
        error = "Stale running job marked failed during recovery"
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT job_id, url, video_id, note_path, warning
                FROM jobs
                WHERE status = ? AND updated_at < ?
                ORDER BY updated_at, job_id
                """,
                ("running", cutoff),
            ).fetchall()
            recovered = [
                JobRecord(
                    job_id=row[0],
                    url=row[1],
                    video_id=row[2],
                    status="failed",
                    stage="failed",
                    note_path=row[3],
                    error=error,
                    warning=row[4],
                )
                for row in rows
            ]
            for record in recovered:
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = ?, stage = ?, error = ?, updated_at = ?
                    WHERE job_id = ?
                    """,
                    ("failed", "failed", error, now, record.job_id),
                )
                conn.execute(
                    "INSERT INTO job_history (job_id, status, stage, message, created_at) VALUES (?, ?, ?, ?, ?)",
                    (record.job_id, "failed", "failed", error, now),
                )
        return recovered

    def update_job_stage(
        self,
        job_id: str,
        status: str,
        stage: str | None = None,
        note_path: str | None = None,
        error: str | None = None,
        warning: str | None = None,
    ) -> JobRecord:
        jobs = self.load_jobs()
        if job_id not in jobs:
            raise KeyError(f"Unknown job: {job_id}")
        record = jobs[job_id]
        record.status = status
        record.stage = stage
        record.note_path = note_path if note_path is not None else record.note_path
        record.error = error
        record.warning = warning
        self.save_job(record)
        return record

    def load_job_history(self, job_id: str) -> list[dict[str, Any]]:
        self.ensure_layout()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, stage, message, created_at FROM job_history WHERE job_id = ? ORDER BY id",
                (job_id,),
            ).fetchall()
        return [
            {"status": row[0], "stage": row[1], "message": row[2], "created_at": row[3]}
            for row in rows
        ]

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def _read_json(path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))
