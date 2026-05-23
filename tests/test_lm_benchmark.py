from __future__ import annotations

import json
import re
import socket
import subprocess
import sys
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from yt_sub_hub.config import get_settings
from yt_sub_hub.library import LibraryStore
from yt_sub_hub.models import Transcript, TranscriptSegment
from yt_sub_hub.translation import TranslationError


def _write_source_transcript(store: LibraryStore, video_id: str = "abc123def45", *, segments: int = 12) -> None:
    paths = store.artifact_paths(video_id)
    paths.video_dir.mkdir(parents=True, exist_ok=True)
    transcript = Transcript(
        video_id=video_id,
        source="youtube",
        language="en",
        segments=[
            TranscriptSegment(index * 5, 5, f"English source segment {index} with enough text for benchmark.")
            for index in range(segments)
        ],
    )
    store._write_json(paths.normalized_transcript, transcript.to_dict())


class SuccessfulProvider:
    active = 0
    lock = threading.Lock()

    def __init__(self, base_url, api_key, model, target_language="ru", max_chars=2500, timeout_seconds=120):
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.translation_usage = [{"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}]
        self.retry_count = 0

    def translate_transcript(self, transcript, target_language):
        with self.lock:
            type(self).active += 1
            active = type(self).active
        try:
            time.sleep(0.01 if active <= 2 else 0.02)
            return Transcript(
                transcript.video_id,
                transcript.source,
                [
                    TranscriptSegment(segment.start, segment.duration, f"Русский перевод {segment.text}", segment.raw_text)
                    for segment in transcript.segments
                ],
                language=target_language,
            )
        finally:
            with self.lock:
                type(self).active -= 1


class _TranslationStubState:
    def __init__(self, *, mode: str = "success") -> None:
        self.mode = mode
        self.active = 0
        self.request_count = 0
        self.lock = threading.Lock()


class _TranslationStubHandler(BaseHTTPRequestHandler):
    state: _TranslationStubState

    def log_message(self, format, *args):  # noqa: A002 - BaseHTTPRequestHandler API.
        return

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API.
        with self.state.lock:
            self.state.request_count += 1
            self.state.active += 1
            active = self.state.active
            request_count = self.state.request_count
        try:
            if self.state.mode == "hang":
                time.sleep(30)
                return
            length = int(self.headers.get("Content-Length") or "0")
            body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            if self.state.mode == "parse_error":
                content = '{"segments": []}'
            else:
                if self.state.mode == "speedup":
                    time.sleep(0.03 if active <= 2 else 1.0)
                elif self.state.mode == "hang_parallel" and active > 1:
                    time.sleep(30)
                    return
                elif self.state.mode == "hang_parallel":
                    time.sleep(0.2)
                payload = _extract_input_payload(body)
                translated = {
                    "segments": [
                        {
                            "index": segment["index"],
                            "start": segment["start"],
                            "duration": segment["duration"],
                            "text": f"Русский перевод {segment['text']}",
                        }
                        for segment in payload.get("segments", [])
                    ]
                }
                content = json.dumps(translated, ensure_ascii=False)
            response = {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": body.get("model", "test-model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            }
            raw = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        finally:
            with self.state.lock:
                self.state.active -= 1


def _extract_input_payload(body: dict) -> dict:
    messages = body.get("messages") or []
    content = next((item.get("content", "") for item in messages if item.get("role") == "user"), "")
    match = re.search(r"Input JSON:\n(?P<payload>\{.*\})", content, re.DOTALL)
    return json.loads(match.group("payload")) if match else {}


class _TranslationStubServer:
    def __init__(self, *, mode: str = "success") -> None:
        self.state = _TranslationStubState(mode=mode)
        handler = type("Handler", (_TranslationStubHandler,), {"state": self.state})
        self.server = ThreadingHTTPServer(("127.0.0.1", _free_port()), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _benchmark_settings(tmp_path: Path, base_url: str):
    return replace(
        get_settings(env_file=Path("/tmp/non-existent-yt-sub-hub.env")),
        library_dir=tmp_path,
        lm_studio_base_url=base_url,
        translation_model="test-model",
        translation_chunk_max_segments=1,
    )


def test_benchmark_aggregates_runs_and_recommends_two(tmp_path, monkeypatch):
    from yt_sub_hub import lm_benchmark

    _write_source_transcript(LibraryStore(tmp_path), segments=12)
    monkeypatch.setattr(lm_benchmark, "collect_memory_snapshot", lambda: {"rss_mb": 100.0, "pageouts": 0, "pressure": None, "total_memory_mb": 32000.0})
    with _TranslationStubServer(mode="speedup") as server:
        settings = _benchmark_settings(tmp_path, server.base_url)

        report = lm_benchmark.benchmark_lm_studio_translation(
            LibraryStore(tmp_path),
            settings,
            lm_benchmark.BenchmarkOptions(
                video_id="abc123def45",
                concurrencies=[1, 2, 4],
                sample_chunks=12,
                timeout_seconds=30,
                output=tmp_path / "report.json",
                chunk_selection="first",
                run_timeout_seconds=10,
            ),
        )

    assert report["recommendation"]["safe_default"] == 2
    assert report["sample"]["chunks_requested"] == 12
    assert [run["concurrency"] for run in report["runs"]] == [1, 2, 4]
    for run in report["runs"]:
        assert run["chunks_total"] == 12
        assert run["chunks_succeeded"] == 12
        assert run["chunks_failed"] == 0
        assert run["error_rate"] == 0
        assert run["timeout_rate"] == 0
        assert run["parse_error_rate"] == 0
        assert run["latency_seconds"]["p50"] is not None
        assert run["throughput"]["chars_per_second"] > 0
        assert run["throughput"]["tokens_per_second"] is not None
    assert (tmp_path / "report.json").exists()


def test_benchmark_counts_provider_timeout_and_rejects_run(tmp_path, monkeypatch):
    from yt_sub_hub import lm_benchmark

    _write_source_transcript(LibraryStore(tmp_path), segments=8)
    monkeypatch.setattr(lm_benchmark, "collect_memory_snapshot", lambda: {"rss_mb": 100.0, "pageouts": 0, "pressure": None, "total_memory_mb": 32000.0})
    with _TranslationStubServer(mode="hang") as server:
        settings = _benchmark_settings(tmp_path, server.base_url)

        report = lm_benchmark.benchmark_lm_studio_translation(
            LibraryStore(tmp_path),
            settings,
            lm_benchmark.BenchmarkOptions(
                "abc123def45",
                [1],
                2,
                30,
                tmp_path / "report.json",
                "first",
                run_timeout_seconds=0.2,
            ),
        )

    run = report["runs"][0]
    assert run["run_status"] == "completed_with_timeouts"
    assert run["timeout_rate"] == 1.0
    assert run["chunks_succeeded"] == 0
    assert report["recommendation"]["safe_default"] == 1
    assert "baseline failed" in report["recommendation"]["reason"]


def test_provider_timeout_result_appears_in_timed_out_chunk_indexes(tmp_path, monkeypatch):
    from yt_sub_hub import lm_benchmark

    transcript = Transcript("abc123def45", "youtube", [], language="en")
    settings = _benchmark_settings(tmp_path, "http://127.0.0.1:1234/v1")

    def fake_run_chunks(*args, **kwargs):
        return [
            {
                "chunk_index": 25,
                "source_segment_start_index": 25,
                "source_segment_end_index": 25,
                "source_chars": 100,
                "attempt": 1,
                "retry_count": 0,
                "started_at": "2026-05-06T00:00:00Z",
                "finished_at": "2026-05-06T00:00:01Z",
                "latency_seconds": 1.0,
                "status": "timeout",
                "error_class": "TranslationError",
                "error_message": "LM Studio transcript translation timed out after 60 seconds",
                "prompt_tokens": None,
                "response_tokens": None,
                "total_tokens": None,
                "chars_per_second": None,
            }
        ], []

    monkeypatch.setattr(lm_benchmark, "collect_memory_snapshot", lambda: {"rss_mb": None, "pageouts": None, "pressure": None, "total_memory_mb": None})
    monkeypatch.setattr(lm_benchmark, "_run_chunks_in_processes", fake_run_chunks)

    run = lm_benchmark._run_concurrency(
        transcript,
        settings,
        "ru",
        [{"chunk_index": 25}],
        concurrency=1,
        timeout_seconds=60,
        run_timeout_seconds=120,
    )

    assert run["timeout_rate"] > 0
    assert run["timed_out_chunk_indexes"] == [25]


def test_timeout_indexes_from_process_and_status_are_sorted_and_deduped(tmp_path, monkeypatch):
    from yt_sub_hub import lm_benchmark

    transcript = Transcript("abc123def45", "youtube", [], language="en")
    settings = _benchmark_settings(tmp_path, "http://127.0.0.1:1234/v1")

    def timeout_result(chunk_index: int, error_class: str) -> dict:
        return {
            "chunk_index": chunk_index,
            "source_segment_start_index": chunk_index,
            "source_segment_end_index": chunk_index,
            "source_chars": 100,
            "attempt": 1,
            "retry_count": 0,
            "started_at": "2026-05-06T00:00:00Z",
            "finished_at": "2026-05-06T00:00:01Z",
            "latency_seconds": 1.0,
            "status": "timeout",
            "error_class": error_class,
            "error_message": "timed out",
            "prompt_tokens": None,
            "response_tokens": None,
            "total_tokens": None,
            "chars_per_second": None,
        }

    def fake_run_chunks(*args, **kwargs):
        return [
            timeout_result(25, "TranslationError"),
            timeout_result(2, "BenchmarkRunTimeout"),
        ], [25, 2, 2]

    monkeypatch.setattr(lm_benchmark, "collect_memory_snapshot", lambda: {"rss_mb": None, "pageouts": None, "pressure": None, "total_memory_mb": None})
    monkeypatch.setattr(lm_benchmark, "_run_chunks_in_processes", fake_run_chunks)

    run = lm_benchmark._run_concurrency(
        transcript,
        settings,
        "ru",
        [{"chunk_index": 25}, {"chunk_index": 2}],
        concurrency=2,
        timeout_seconds=60,
        run_timeout_seconds=120,
    )

    assert run["timed_out_chunk_indexes"] == [2, 25]


def test_timeout_result_with_invalid_chunk_index_does_not_break_report(tmp_path, monkeypatch):
    from yt_sub_hub import lm_benchmark

    transcript = Transcript("abc123def45", "youtube", [], language="en")
    settings = _benchmark_settings(tmp_path, "http://127.0.0.1:1234/v1")

    def fake_run_chunks(*args, **kwargs):
        return [
            {
                "chunk_index": "bad-index",
                "source_segment_start_index": 0,
                "source_segment_end_index": 0,
                "source_chars": 100,
                "attempt": 1,
                "retry_count": 0,
                "started_at": "2026-05-06T00:00:00Z",
                "finished_at": "2026-05-06T00:00:01Z",
                "latency_seconds": 1.0,
                "status": "timeout",
                "error_class": "TranslationError",
                "error_message": "LM Studio transcript translation timed out after 60 seconds",
                "prompt_tokens": None,
                "response_tokens": None,
                "total_tokens": None,
                "chars_per_second": None,
            }
        ], []

    monkeypatch.setattr(lm_benchmark, "collect_memory_snapshot", lambda: {"rss_mb": None, "pageouts": None, "pressure": None, "total_memory_mb": None})
    monkeypatch.setattr(lm_benchmark, "_run_chunks_in_processes", fake_run_chunks)

    run = lm_benchmark._run_concurrency(
        transcript,
        settings,
        "ru",
        [{"chunk_index": "bad-index"}],
        concurrency=1,
        timeout_seconds=60,
        run_timeout_seconds=120,
    )

    assert run["timeout_rate"] == 1.0
    assert run["timed_out_chunk_indexes"] == []


def test_benchmark_counts_parse_errors(tmp_path, monkeypatch):
    from yt_sub_hub import lm_benchmark

    class ParseFailingProvider(SuccessfulProvider):
        def translate_transcript(self, transcript, target_language):
            raise TranslationError("Translation response is not valid JSON")

    _write_source_transcript(LibraryStore(tmp_path), segments=3)
    monkeypatch.setattr(lm_benchmark, "collect_memory_snapshot", lambda: {"rss_mb": None, "pageouts": None, "pressure": None, "total_memory_mb": None})
    with _TranslationStubServer(mode="parse_error") as server:
        settings = _benchmark_settings(tmp_path, server.base_url)

        report = lm_benchmark.benchmark_lm_studio_translation(
            LibraryStore(tmp_path),
            settings,
            lm_benchmark.BenchmarkOptions("abc123def45", [1], 3, 30, None, "first", run_timeout_seconds=10),
        )

    run = report["runs"][0]
    assert run["parse_error_rate"] == 1.0
    assert {item["status"] for item in run["chunk_results"]} == {"parse_error"}


def test_benchmark_does_not_write_production_artifacts(tmp_path, monkeypatch):
    from yt_sub_hub import lm_benchmark

    store = LibraryStore(tmp_path)
    _write_source_transcript(store, segments=6)
    paths = store.artifact_paths("abc123def45")
    before = _snapshot_tree(paths.video_dir)
    monkeypatch.setattr(lm_benchmark, "collect_memory_snapshot", lambda: {"rss_mb": None, "pageouts": None, "pressure": None, "total_memory_mb": None})
    with _TranslationStubServer(mode="success") as server:
        settings = _benchmark_settings(tmp_path, server.base_url)

        lm_benchmark.benchmark_lm_studio_translation(
            store,
            settings,
            lm_benchmark.BenchmarkOptions(
                "abc123def45",
                [1],
                4,
                30,
                tmp_path / "benchmarks" / "report.json",
                "first",
                run_timeout_seconds=10,
            ),
        )

    assert _snapshot_tree(paths.video_dir) == before
    assert not paths.transcript_ru.exists()
    assert not (paths.video_dir / "translation" / "chunks").exists()
    assert not (tmp_path / "index.sqlite3").exists()
    assert (tmp_path / "benchmarks" / "report.json").exists()


def test_report_schema_and_memory_sampler_tolerance(tmp_path, monkeypatch):
    from yt_sub_hub import lm_benchmark

    samples = iter(
        [
            {"rss_mb": 100.0, "pageouts": None, "pressure": None, "total_memory_mb": None},
            {"rss_mb": None, "pageouts": None, "pressure": None, "total_memory_mb": None},
        ]
    )
    _write_source_transcript(LibraryStore(tmp_path), segments=2)
    monkeypatch.setattr(lm_benchmark, "collect_memory_snapshot", lambda: next(samples, {"rss_mb": None, "pageouts": None, "pressure": None, "total_memory_mb": None}))
    with _TranslationStubServer(mode="success") as server:
        settings = _benchmark_settings(tmp_path, server.base_url)

        report = lm_benchmark.benchmark_lm_studio_translation(
            LibraryStore(tmp_path),
            settings,
            lm_benchmark.BenchmarkOptions("abc123def45", [1], 2, 30, None, "evenly-spaced", run_timeout_seconds=10),
        )

    assert set(report) >= {"video_id", "model", "base_url", "created_at", "sample", "runs", "recommendation"}
    run = report["runs"][0]
    assert set(run) >= {
        "concurrency",
        "started_at",
        "finished_at",
        "total_elapsed_seconds",
        "chunks_total",
        "chunks_succeeded",
        "chunks_failed",
        "error_rate",
        "timeout_rate",
        "parse_error_rate",
        "latency_seconds",
        "throughput",
        "memory",
        "chunk_results",
        "run_status",
        "run_timeout_seconds",
        "timed_out_chunk_indexes",
        "skipped_reason",
    }
    assert run["memory"]["rss_mb_before"] == 100.0
    assert run["memory"]["rss_mb_peak"] == 100.0


def test_hung_provider_writes_timeout_report_and_returns_quickly(tmp_path):
    script = f"""
from dataclasses import replace
from pathlib import Path
from yt_sub_hub import lm_benchmark
from yt_sub_hub.config import get_settings
from yt_sub_hub.library import LibraryStore
from tests.test_lm_benchmark import _TranslationStubServer, _write_source_transcript, _benchmark_settings

root = Path({str(tmp_path)!r})
store = LibraryStore(root)
_write_source_transcript(store, segments=2)
with _TranslationStubServer(mode="hang") as server:
    settings = _benchmark_settings(root, server.base_url)
    report = lm_benchmark.benchmark_lm_studio_translation(
        store,
        settings,
        lm_benchmark.BenchmarkOptions(
            "abc123def45",
            [1],
            2,
            30,
            root / "benchmarks" / "hung.json",
            "first",
            run_timeout_seconds=0.2,
        ),
    )
print(report["report_path"])
"""
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert time.perf_counter() - started < 5
    report_path = tmp_path / "benchmarks" / "hung.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    run = report["runs"][0]
    assert run["run_status"] == "completed_with_timeouts"
    assert run["timeout_rate"] > 0
    assert run["timed_out_chunk_indexes"] == [0, 1]
    assert run["chunk_results"][0]["status"] == "timeout"
    assert run["chunk_results"][0]["error_class"] == "BenchmarkRunTimeout"
    assert report["recommendation"]["safe_default"] == 1


def test_partial_report_written_after_success_before_later_timeout(tmp_path, monkeypatch):
    from yt_sub_hub import lm_benchmark

    _write_source_transcript(LibraryStore(tmp_path), segments=3)
    monkeypatch.setattr(lm_benchmark, "collect_memory_snapshot", lambda: {"rss_mb": None, "pageouts": None, "pressure": None, "total_memory_mb": None})
    report_path = tmp_path / "benchmarks" / "partial.json"
    with _TranslationStubServer(mode="hang_parallel") as server:
        settings = _benchmark_settings(tmp_path, server.base_url)

        report = lm_benchmark.benchmark_lm_studio_translation(
            LibraryStore(tmp_path),
            settings,
            lm_benchmark.BenchmarkOptions(
                "abc123def45",
                [1, 2],
                2,
                30,
                report_path,
                "first",
                run_timeout_seconds=3,
            ),
        )

    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert [run["concurrency"] for run in written["runs"]] == [1, 2]
    assert written["runs"][0]["run_status"] == "completed"
    assert written["runs"][1]["run_status"] == "completed_with_timeouts"
    assert report == written


def test_higher_concurrency_is_skipped_after_timeout_by_default(tmp_path, monkeypatch):
    from yt_sub_hub import lm_benchmark

    _write_source_transcript(LibraryStore(tmp_path), segments=4)
    monkeypatch.setattr(lm_benchmark, "collect_memory_snapshot", lambda: {"rss_mb": None, "pageouts": None, "pressure": None, "total_memory_mb": None})
    with _TranslationStubServer(mode="hang_parallel") as server:
        settings = _benchmark_settings(tmp_path, server.base_url)

        report = lm_benchmark.benchmark_lm_studio_translation(
            LibraryStore(tmp_path),
            settings,
            lm_benchmark.BenchmarkOptions(
                "abc123def45",
                [1, 2, 4],
                2,
                30,
                tmp_path / "benchmarks" / "skipped.json",
                "first",
                run_timeout_seconds=3,
            ),
        )

    assert [run["run_status"] for run in report["runs"]] == ["completed", "completed_with_timeouts", "skipped"]
    assert report["runs"][2]["skipped_reason"] == "lower concurrency timed out"
    assert report["recommendation"]["safe_default"] == 1


def test_continue_after_timeout_runs_later_concurrency_but_rejects_it(tmp_path, monkeypatch):
    from yt_sub_hub import lm_benchmark

    _write_source_transcript(LibraryStore(tmp_path), segments=4)
    monkeypatch.setattr(lm_benchmark, "collect_memory_snapshot", lambda: {"rss_mb": None, "pageouts": None, "pressure": None, "total_memory_mb": None})
    with _TranslationStubServer(mode="hang_parallel") as server:
        settings = _benchmark_settings(tmp_path, server.base_url)

        report = lm_benchmark.benchmark_lm_studio_translation(
            LibraryStore(tmp_path),
            settings,
            lm_benchmark.BenchmarkOptions(
                "abc123def45",
                [1, 2, 4],
                2,
                30,
                tmp_path / "benchmarks" / "continued.json",
                "first",
                run_timeout_seconds=3,
                continue_after_timeout=True,
            ),
        )

    assert [run["concurrency"] for run in report["runs"]] == [1, 2, 4]
    assert report["runs"][1]["run_status"] == "completed_with_timeouts"
    assert report["runs"][2]["run_status"] == "completed_with_timeouts"
    assert report["recommendation"]["safe_default"] == 1
    assert "lower concurrency timed out" in report["recommendation"]["reason"]


def test_exception_after_first_run_preserves_partial_report(tmp_path, monkeypatch):
    from yt_sub_hub import lm_benchmark

    _write_source_transcript(LibraryStore(tmp_path), segments=2)
    report_path = tmp_path / "benchmarks" / "error.json"
    settings = replace(
        get_settings(env_file=Path("/tmp/non-existent-yt-sub-hub.env")),
        library_dir=tmp_path,
        translation_chunk_max_segments=1,
    )
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(kwargs.get("concurrency") if "concurrency" in kwargs else args[4])
        if len(calls) == 2:
            raise RuntimeError("boom after first run")
        return {
            "concurrency": calls[-1],
            "started_at": "2026-05-06T00:00:00Z",
            "finished_at": "2026-05-06T00:00:01Z",
            "total_elapsed_seconds": 1,
            "chunks_total": 1,
            "chunks_succeeded": 1,
            "chunks_failed": 0,
            "error_rate": 0,
            "timeout_rate": 0,
            "parse_error_rate": 0,
            "retry_count": 0,
            "latency_seconds": {"p50": 1, "p90": 1, "p95": 1, "max": 1},
            "throughput": {"chunks_per_minute": 60, "tokens_per_second": None, "chars_per_second": None},
            "memory": {},
            "chunk_results": [],
            "run_status": "completed",
            "run_timeout_seconds": 10,
            "timed_out_chunk_indexes": [],
            "skipped_reason": None,
        }

    monkeypatch.setattr(lm_benchmark, "_run_concurrency", fake_run)

    with pytest.raises(RuntimeError, match="boom after first run"):
        lm_benchmark.benchmark_lm_studio_translation(
            LibraryStore(tmp_path),
            settings,
            lm_benchmark.BenchmarkOptions("abc123def45", [1, 2], 2, 30, report_path, "first", run_timeout_seconds=10),
        )

    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(written["runs"]) == 1
    assert written["benchmark_error"] == "boom after first run"


def test_memory_summary_does_not_treat_normal_memory_pressure_text_as_failure():
    from yt_sub_hub import lm_benchmark

    summary = lm_benchmark._memory_summary(
        [
            {
                "rss_mb": 100.0,
                "pageouts": 10,
                "pressure": "System-wide memory free percentage: 59%",
                "total_memory_mb": 32000.0,
            },
            {
                "rss_mb": 120.0,
                "pageouts": 10,
                "pressure": "System-wide memory free percentage: 58%",
                "total_memory_mb": 32000.0,
            },
        ]
    )

    assert summary["memory_pressure"] is False


def _snapshot_tree(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    result = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        result[str(path.relative_to(root))] = path.read_bytes().hex()
    return result
