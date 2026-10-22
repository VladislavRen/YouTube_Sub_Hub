from __future__ import annotations

import fcntl
import json
import os
import re
import sqlite3
import tempfile
import time
from contextlib import contextmanager
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
    RAW_CACHE_MAX_AGE_SECONDS = 30 * 24 * 60 * 60
    RAW_CACHE_FILENAMES = ("metadata.json", "transcript.raw.json", "transcript.normalized.json")

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
        return (
            self.derived_cache_valid(video_id, paths.note)
            and self.load_video_metadata(video_id) is not None
            and self.load_normalized_transcript(video_id) is not None
            and self.load_analysis(video_id) is not None
        )

    def _cache_state(self, video_id: str) -> dict[str, int]:
        path = self.video_dir(video_id) / "source_cache.json"
        return self._read_json(path) if path.exists() else {}

    @contextmanager
    def video_lock(self, video_id: str, *, shared: bool = False):
        """Coordinate raw-input refresh/cleanup with direct and staged writers on this host."""
        if not re.fullmatch(r"[A-Za-z0-9_-]+", video_id):
            raise ValueError("Invalid video id for cache lock")
        directory = self.root / ".cache-locks"
        if directory.is_symlink():
            raise ValueError("Cache lock directory must not be a symlink")
        directory.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(directory / f"{video_id}.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            try:
                fcntl.flock(descriptor, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Video {video_id} is busy; retry after the active worker finishes") from exc
            yield
        finally:
            os.close(descriptor)

    def raw_cache_valid(self, video_id: str, path: Path) -> bool:
        if not path.is_file() or path.is_symlink():
            return False
        stat = path.stat()
        return (
            stat.st_mtime > time.time() - self.RAW_CACHE_MAX_AGE_SECONDS
            and stat.st_mtime_ns >= self._cache_state(video_id).get("inputs_invalidated_at_ns", 0)
        )

    def has_expired_inputs(self, video_id: str) -> bool:
        return any(
            path.exists() and not self.raw_cache_valid(video_id, path)
            for path in (self.video_dir(video_id) / name for name in self.RAW_CACHE_FILENAMES)
        )

    def derived_cache_valid(self, video_id: str, path: Path) -> bool:
        return (
            path.is_file()
            and not self.has_expired_inputs(video_id)
            and path.stat().st_mtime_ns >= self._cache_state(video_id).get("derived_invalidated_at_ns", 0)
        )

    def invalidate_source_cache(self, video_id: str, *, refresh_inputs: bool = False) -> None:
        """Persist invalidation before network work so failed refreshes stay invalid after restart."""
        directory = self.video_dir(video_id)
        directory.mkdir(parents=True, exist_ok=True)
        state = self._cache_state(video_id)
        now = time.time_ns()
        state["derived_invalidated_at_ns"] = now
        if refresh_inputs:
            state["inputs_invalidated_at_ns"] = now
        self._write_json(directory / "source_cache.json", state)
        # Notes can contain user edits. Preserve them before a subsequent render replaces them.
        for name in ("note", "note.source"):
            note = directory / f"{name}.md"
            if note.is_file() and not note.is_symlink():
                (directory / f"{name}.before-refresh.{now}.md").write_bytes(note.read_bytes())

    def save_source_metadata(self, metadata: VideoMetadata) -> None:
        paths = self.artifact_paths(metadata.video_id)
        paths.video_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(paths.metadata, metadata.to_dict())

    def save_source_transcript(self, video_id: str, raw: Any, normalized: Transcript) -> None:
        paths = self.artifact_paths(video_id)
        paths.video_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(paths.raw_transcript, raw)
        self._write_json(paths.normalized_transcript, normalized.to_dict())

    def cleanup_expired_raw_cache(self, *, apply: bool = False) -> list[Path]:
        """Only direct, regular raw-input files under validated video directories are candidates."""
        videos = self.root / "videos"
        if not videos.is_dir() or videos.is_symlink():
            return []
        candidates: list[Path] = []
        cutoff = time.time() - self.RAW_CACHE_MAX_AGE_SECONDS
        for directory in sorted(videos.iterdir()):
            if directory.is_symlink() or not directory.is_dir() or not re.fullmatch(r"[A-Za-z0-9_-]{11}", directory.name):
                continue
            expired = [
                directory / name for name in self.RAW_CACHE_FILENAMES
                if (directory / name).is_file() and not (directory / name).is_symlink()
                and (directory / name).stat().st_mtime <= cutoff
            ]
            if apply and expired:
                with self.video_lock(directory.name):
                    if directory.is_symlink() or videos.is_symlink():
                        continue
                    self.invalidate_source_cache(directory.name)
                    for path in expired:
                        # Recheck immediately before unlinking; fresh writes must survive a cleanup run.
                        if path.is_file() and not path.is_symlink() and path.stat().st_mtime <= cutoff:
                            path.unlink()
            candidates.extend(expired)
        return candidates

    def _write_source_if_changed(self, path: Path, data: Any) -> None:
        # Reanalysis must not renew the acquisition date of a cached source.
        if not path.exists() or self._read_json(path) != data:
            self._write_json(path, data)

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
        self._write_source_if_changed(paths.metadata, metadata.to_dict())
        raw_data = raw_transcript.to_dict() if hasattr(raw_transcript, "to_dict") else raw_transcript
        self._write_source_if_changed(paths.raw_transcript, raw_data)
        self._write_source_if_changed(paths.normalized_transcript, normalized_transcript.to_dict())
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
        if not self.derived_cache_valid(video_id, path):
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
        if not self.raw_cache_valid(video_id, path):
            return None
        return VideoMetadata(**self._read_json(path))

    def load_raw_transcript(self, video_id: str) -> Any | None:
        path = self.video_dir(video_id) / "transcript.raw.json"
        if not self.raw_cache_valid(video_id, path):
            return None
        return self._read_json(path)

    def load_normalized_transcript(self, video_id: str) -> Transcript | None:
        path = self.video_dir(video_id) / "transcript.normalized.json"
        raw_path = self.video_dir(video_id) / "transcript.raw.json"
        if not self.raw_cache_valid(video_id, path) or (raw_path.exists() and not self.raw_cache_valid(video_id, raw_path)):
            return None
        return self._load_transcript_path(path)

    def load_analysis(self, video_id: str, variant: str = "default") -> dict[str, Any] | None:
        path = self._analysis_path(video_id, variant)
        if not self.derived_cache_valid(video_id, path):
            return None
        return self._read_json(path)

    def load_analysis_files(self) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for video_dir in sorted((self.root / "videos").glob("*")):
            metadata_path = video_dir / "metadata.json"
            analysis_path = video_dir / "analysis.json"
            if self.raw_cache_valid(video_dir.name, metadata_path) and self.derived_cache_valid(video_dir.name, analysis_path):
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
        if not self.derived_cache_valid(video_id, path):
            return None
        return self._read_json(path)

    def load_graph_extraction_files(self) -> list[dict[str, Any]]:
        extractions: list[dict[str, Any]] = []
        for video_dir in sorted((self.root / "videos").glob("*")):
            path = video_dir / "graph_extraction.json"
            if self.derived_cache_valid(video_dir.name, path):
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
        if summary_variant != "source":
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
        if not self.derived_cache_valid(video_id, path):
            return None
        return self._read_json(path)

    def load_visual_summary(self, video_id: str, variant: str = "default") -> dict[str, Any] | None:
        name = "visual_summary.source.json" if variant == "source" else "visual_summary.json"
        path = self.video_dir(video_id) / "visual" / name
        if not self.derived_cache_valid(video_id, path):
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
            if self.load_video_metadata(video_dir.name) is None or self.load_normalized_transcript(video_dir.name) is None:
                continue
            metadata_path = video_dir / "metadata.json"
            source_transcript_path = video_dir / "transcript.normalized.json"
            translated_transcript_path = video_dir / f"transcript.{target_language}.json"
            translated_analysis_path = video_dir / f"analysis.{target_language}.json"
            if prefer_translation and self.derived_cache_valid(video_dir.name, translated_transcript_path):
                if not self.derived_cache_valid(video_dir.name, translated_analysis_path):
                    raise RuntimeError(f"Missing {translated_analysis_path.name} for translated graph rebuild")
                transcript_path = translated_transcript_path
                analysis_path = translated_analysis_path
            else:
                transcript_path = source_transcript_path
                analysis_path = video_dir / "analysis.json"
            if metadata_path.exists() and transcript_path.exists() and self.derived_cache_valid(video_dir.name, analysis_path):
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
        # Readers in other workers must never observe partially written JSON.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def _read_json(path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))
