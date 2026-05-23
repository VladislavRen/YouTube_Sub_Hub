from __future__ import annotations

import json
import math
import multiprocessing
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from yt_sub_hub.config import Settings
from yt_sub_hub.library import LibraryStore
from yt_sub_hub.models import Transcript
from yt_sub_hub.stage_pipeline import plan_translation_chunks, slice_translation_chunk
from yt_sub_hub.translation import LMStudioTranslationProvider, TranslationError


@dataclass(frozen=True)
class BenchmarkOptions:
    video_id: str
    concurrencies: list[int]
    sample_chunks: int
    timeout_seconds: float
    output: Path | None
    chunk_selection: str = "evenly-spaced"
    run_timeout_seconds: float | None = None
    continue_after_timeout: bool = False


class BenchmarkReportError(RuntimeError):
    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


def benchmark_lm_studio_translation(
    store: LibraryStore,
    settings: Settings,
    options: BenchmarkOptions,
    *,
    _timeout_by_concurrency: dict[int, float] | None = None,
) -> dict[str, Any]:
    transcript = store.load_normalized_transcript(options.video_id)
    if transcript is None:
        raise RuntimeError(f"transcript.normalized.json is required for benchmark: {options.video_id}")
    target = str(getattr(settings, "target_language", None) or "ru").strip() or "ru"
    chunks = plan_translation_chunks(transcript, settings, target)
    selected_payloads = _select_chunks(chunks, options.sample_chunks, options.chunk_selection)
    created_at = _utc_now()
    report_path = options.output or _default_output_path(settings.library_dir, options.video_id, created_at)
    report: dict[str, Any] = {
        "video_id": options.video_id,
        "model": settings.translation_model,
        "base_url": settings.lm_studio_base_url,
        "created_at": created_at,
        "sample": {
            "chunks_requested": int(options.sample_chunks),
            "chunks_available": len(chunks),
            "chunks_selected": len(selected_payloads),
            "chunk_selection": options.chunk_selection,
            "max_chunk_chars": int(getattr(settings, "translation_chunk_max_chars", 2000)),
            "prompt_size_chars": sum(_chunk_source_chars(transcript, payload) for payload in selected_payloads),
        },
        "runs": [],
        "recommendation": {"safe_default": 1, "reason": "no benchmark runs were completed"},
        "report_path": str(report_path),
    }
    writer = BenchmarkReportWriter(report_path, report)
    concurrencies = _normalize_concurrencies(options.concurrencies)
    run_timeout_setting = (
        options.run_timeout_seconds
        if options.run_timeout_seconds is not None
        else float(getattr(settings, "lm_studio_benchmark_run_timeout_seconds", 0) or 0)
    )
    try:
        for index, concurrency in enumerate(concurrencies):
            timeout_seconds = float((_timeout_by_concurrency or {}).get(concurrency, options.timeout_seconds))
            run = _run_concurrency(
                transcript,
                settings,
                target,
                selected_payloads,
                concurrency,
                timeout_seconds,
                run_timeout_seconds=run_timeout_setting,
            )
            report["runs"].append(run)
            report["recommendation"] = recommend_safe_default(report["runs"])
            writer.write(report)
            if _run_timed_out(run) and not options.continue_after_timeout:
                for skipped_concurrency in concurrencies[index + 1 :]:
                    skipped = _skipped_run(
                        skipped_concurrency,
                        selected_payloads,
                        timeout_seconds,
                        run_timeout_setting,
                        "lower concurrency timed out",
                    )
                    report["runs"].append(skipped)
                report["recommendation"] = recommend_safe_default(report["runs"])
                writer.write(report)
                break
    except KeyboardInterrupt as exc:
        writer.mark_interrupted(str(exc) or "KeyboardInterrupt")
        raise
    except Exception as exc:
        report["benchmark_error"] = _short_error(str(exc))
        writer.write(report)
        raise BenchmarkReportError(str(exc), report) from exc
    writer.finalize(recommend_safe_default(report["runs"]))
    return report


class BenchmarkReportWriter:
    def __init__(self, path: Path, initial_report: dict[str, Any]) -> None:
        self.path = path
        self.report = initial_report
        self.write(initial_report)

    def update_run(self, run: dict[str, Any]) -> None:
        self.report.setdefault("runs", []).append(run)
        self.report["recommendation"] = recommend_safe_default(self.report["runs"])
        self.write(self.report)

    def finalize(self, recommendation: dict[str, Any]) -> None:
        self.report["recommendation"] = recommendation
        self.write(self.report)

    def mark_interrupted(self, error: str) -> None:
        self.report["interrupted"] = True
        self.report["benchmark_error"] = _short_error(error)
        self.write(self.report)

    def write(self, report: dict[str, Any]) -> None:
        self.report = report
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(f"{self.path.name}.tmp")
        tmp_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.path)


def _run_concurrency(
    transcript: Transcript,
    settings: Settings,
    target: str,
    payloads: list[dict[str, Any]],
    concurrency: int,
    timeout_seconds: float,
    *,
    run_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    started_at = _utc_now()
    memory_sampler = MemorySampler()
    memory_sampler.start()
    start = time.perf_counter()
    results: list[dict[str, Any]] = []
    run_timeout = _resolve_run_timeout_seconds(len(payloads), concurrency, timeout_seconds, run_timeout_seconds)
    timed_out_chunk_indexes: list[int] = []
    try:
        results, timed_out_chunk_indexes = _run_chunks_in_processes(
            transcript,
            settings,
            target,
            payloads,
            concurrency,
            timeout_seconds,
            run_timeout,
            start,
        )
    finally:
        total_elapsed = time.perf_counter() - start
        memory = memory_sampler.stop()
    finished_at = _utc_now()
    results.sort(key=_chunk_sort_key)
    succeeded = [item for item in results if item["status"] == "success"]
    timeouts = [item for item in results if item["status"] == "timeout"]
    parse_errors = [item for item in results if item["status"] == "parse_error"]
    errors = [item for item in results if item["status"] == "error"]
    process_timeout_indexes = set(timed_out_chunk_indexes)
    status_timeout_indexes = {
        chunk_index
        for item in timeouts
        for chunk_index in [_safe_chunk_index(item)]
        if chunk_index is not None
    }
    all_timeout_indexes = sorted(process_timeout_indexes | status_timeout_indexes)
    latencies = [float(item["latency_seconds"]) for item in succeeded]
    source_chars = sum(int(item["source_chars"]) for item in succeeded)
    total_tokens = sum(_int_or_zero(item.get("total_tokens")) for item in succeeded)
    retry_count = sum(_int_or_zero(item.get("retry_count")) for item in results)
    chunks_total = len(results)
    run_status = "completed_with_timeouts" if timeouts else "completed"
    return {
        "concurrency": int(concurrency),
        "run_status": run_status,
        "run_timeout_seconds": round(run_timeout, 3),
        "timed_out_chunk_indexes": all_timeout_indexes,
        "skipped_reason": None,
        "started_at": started_at,
        "finished_at": finished_at,
        "total_elapsed_seconds": round(total_elapsed, 3),
        "chunks_total": chunks_total,
        "chunks_succeeded": len(succeeded),
        "chunks_failed": chunks_total - len(succeeded),
        "error_rate": _rate(len(errors), chunks_total),
        "timeout_rate": _rate(len(timeouts), chunks_total),
        "parse_error_rate": _rate(len(parse_errors), chunks_total),
        "retry_count": retry_count,
        "latency_seconds": {
            "p50": _round_or_none(_percentile(latencies, 50)),
            "p90": _round_or_none(_percentile(latencies, 90)),
            "p95": _round_or_none(_percentile(latencies, 95)),
            "max": _round_or_none(max(latencies) if latencies else None),
        },
        "throughput": {
            "chunks_per_minute": round((len(succeeded) / total_elapsed) * 60, 3) if total_elapsed > 0 else None,
            "tokens_per_second": round(total_tokens / total_elapsed, 3) if total_tokens and total_elapsed > 0 else None,
            "chars_per_second": round(source_chars / total_elapsed, 3) if source_chars and total_elapsed > 0 else None,
        },
        "memory": memory,
        "chunk_results": results,
    }


def _safe_chunk_index(item: dict[str, Any]) -> int | None:
    try:
        return int(item["chunk_index"])
    except (KeyError, TypeError, ValueError):
        return None


def _chunk_sort_key(item: dict[str, Any]) -> tuple[bool, int]:
    chunk_index = _safe_chunk_index(item)
    return chunk_index is None, chunk_index if chunk_index is not None else 0


def _run_chunks_in_processes(
    transcript: Transcript,
    settings: Settings,
    target: str,
    payloads: list[dict[str, Any]],
    concurrency: int,
    timeout_seconds: float,
    run_timeout_seconds: float,
    run_started: float,
) -> tuple[list[dict[str, Any]], list[int]]:
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    transcript_payload = transcript.to_dict()
    settings_payload = _benchmark_settings_payload(settings)
    max_workers = max(1, int(concurrency))
    deadline = run_started + max(0.001, float(run_timeout_seconds))
    pending = list(payloads)
    active: dict[int, tuple[multiprocessing.Process, dict[str, Any], float]] = {}
    completed_indexes: set[int] = set()
    results: list[dict[str, Any]] = []

    def launch_available() -> None:
        while pending and len(active) < max_workers and time.perf_counter() < deadline:
            payload = pending.pop(0)
            chunk_index = int(payload["chunk_index"])
            launched_at = time.perf_counter()
            process = ctx.Process(
                target=translate_chunk_worker,
                args=(settings_payload, transcript_payload, target, payload, timeout_seconds, result_queue),
            )
            process.daemon = True
            process.start()
            active[chunk_index] = (process, payload, launched_at)

    try:
        launch_available()
        while len(completed_indexes) < len(payloads) and time.perf_counter() < deadline:
            drained = False
            while True:
                try:
                    result = result_queue.get_nowait()
                except queue.Empty:
                    break
                drained = True
                chunk_index = int(result["chunk_index"])
                if chunk_index in completed_indexes:
                    continue
                completed_indexes.add(chunk_index)
                results.append(result)
                process_info = active.pop(chunk_index, None)
                if process_info is not None:
                    process_info[0].join(timeout=0.2)
            for chunk_index, (process, payload, launched_at) in list(active.items()):
                if process.is_alive():
                    continue
                process.join(timeout=0.2)
                if chunk_index not in completed_indexes:
                    completed_indexes.add(chunk_index)
                    active.pop(chunk_index, None)
                    results.append(
                        _chunk_process_error_result(
                            transcript,
                            payload,
                            launched_at,
                            f"worker process exited before returning result (exitcode={process.exitcode})",
                        )
                    )
            launch_available()
            if not drained:
                time.sleep(0.02)
    finally:
        timed_out_payloads = [payload for payload in payloads if int(payload["chunk_index"]) not in completed_indexes]
        timeout_finished_at = time.perf_counter()
        for process, _payload, _launched_at in list(active.values()):
            if process.is_alive():
                process.terminate()
        for process, _payload, _launched_at in list(active.values()):
            process.join(timeout=1)
        for payload in timed_out_payloads:
            results.append(_chunk_timeout_result(transcript, payload, run_started, timeout_finished_at))
        result_queue.close()
        result_queue.join_thread()
    timed_out_chunk_indexes = sorted(int(payload["chunk_index"]) for payload in payloads if int(payload["chunk_index"]) not in completed_indexes)
    return results, timed_out_chunk_indexes


def _resolve_run_timeout_seconds(
    chunks_total: int,
    concurrency: int,
    timeout_seconds: float,
    run_timeout_seconds: float | None,
) -> float:
    explicit = float(run_timeout_seconds or 0)
    if explicit > 0:
        return explicit
    batches = max(1, math.ceil(max(0, int(chunks_total)) / max(1, int(concurrency))))
    provider_timeout = max(0.1, float(timeout_seconds))
    return provider_timeout * batches + max(10.0, provider_timeout * 0.1)


def _benchmark_settings_payload(settings: Settings) -> dict[str, Any]:
    return {
        "lm_studio_base_url": settings.lm_studio_base_url,
        "lm_studio_api_key": settings.lm_studio_api_key,
        "translation_model": settings.translation_model,
        "translation_max_chars": int(getattr(settings, "translation_max_chars", 2500)),
        "translation_chunk_max_chars": int(getattr(settings, "translation_chunk_max_chars", 2000)),
    }


def _transcript_from_payload(data: dict[str, Any]) -> Transcript:
    from yt_sub_hub.models import TranscriptSegment

    return Transcript(
        video_id=str(data["video_id"]),
        source=str(data["source"]),
        language=data.get("language"),
        segments=[TranscriptSegment(**segment) for segment in data.get("segments", [])],
    )


def translate_chunk_worker(
    settings_payload: dict[str, Any],
    transcript_payload: dict[str, Any],
    target: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    result_queue: Any,
) -> None:
    transcript = _transcript_from_payload(transcript_payload)
    settings = SimpleNamespace(**settings_payload)
    try:
        result = _translate_chunk(transcript, settings, target, payload, timeout_seconds)
    except Exception as exc:  # noqa: BLE001 - parent needs a chunk-shaped worker failure.
        result = _chunk_error_result(transcript, payload, type(exc).__name__, _short_error(str(exc)))
    result_queue.put(result)


def _chunk_timeout_result(
    transcript: Transcript,
    payload: dict[str, Any],
    started_at: float,
    finished_at: float,
) -> dict[str, Any]:
    result = _chunk_error_result(
        transcript,
        payload,
        "BenchmarkRunTimeout",
        "benchmark run timed out before chunk completed",
    )
    result["latency_seconds"] = round(max(0.0, finished_at - started_at), 3)
    return result


def _chunk_process_error_result(
    transcript: Transcript,
    payload: dict[str, Any],
    started_at: float,
    message: str,
) -> dict[str, Any]:
    result = _chunk_error_result(transcript, payload, "BenchmarkWorkerError", message)
    result["latency_seconds"] = round(max(0.0, time.perf_counter() - started_at), 3)
    return result


def _chunk_error_result(
    transcript: Transcript,
    payload: dict[str, Any],
    error_class: str,
    error_message: str,
) -> dict[str, Any]:
    source_chars = _chunk_source_chars(transcript, payload)
    now = _utc_now()
    status = "timeout" if "timeout" in error_class.lower() or "timed out" in error_message.lower() else "error"
    return {
        "chunk_index": int(payload["chunk_index"]),
        "source_segment_start_index": int(payload["source_segment_start_index"]),
        "source_segment_end_index": int(payload["source_segment_end_index"]),
        "source_chars": source_chars,
        "attempt": 1,
        "retry_count": 0,
        "started_at": now,
        "finished_at": now,
        "latency_seconds": 0.0,
        "status": status,
        "error_class": error_class,
        "error_message": _short_error(error_message),
        "prompt_tokens": None,
        "response_tokens": None,
        "total_tokens": None,
        "chars_per_second": None,
    }


def _translate_chunk(
    transcript: Transcript,
    settings: Settings,
    target: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    chunk = slice_translation_chunk(transcript, payload)
    source_chars = sum(len(segment.text or "") for segment in chunk.segments)
    started_at = _utc_now()
    start = time.perf_counter()
    provider = LMStudioTranslationProvider(
        settings.lm_studio_base_url,
        settings.lm_studio_api_key,
        settings.translation_model,
        target_language=target,
        max_chars=max(1, int(getattr(settings, "translation_chunk_max_chars", settings.translation_max_chars))),
        timeout_seconds=timeout_seconds,
    )
    status = "success"
    error_class = None
    error_message = None
    prompt_tokens = completion_tokens = total_tokens = None
    try:
        translated = provider.translate_transcript(chunk, target)
        if (translated.language or "").strip().lower() != target.strip().lower():
            raise TranslationError(f"Transcript translation did not produce {target}")
        if len(translated.segments) != len(chunk.segments):
            raise TranslationError("Transcript translation segment count does not match source segment count")
    except Exception as exc:  # noqa: BLE001 - benchmark records provider failures per chunk.
        status = _status_for_exception(exc)
        error_class = type(exc).__name__
        error_message = _short_error(str(exc))
    finally:
        latency = time.perf_counter() - start
    usage = _sum_usage(getattr(provider, "translation_usage", []))
    if usage:
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
    return {
        "chunk_index": int(payload["chunk_index"]),
        "source_segment_start_index": int(payload["source_segment_start_index"]),
        "source_segment_end_index": int(payload["source_segment_end_index"]),
        "source_chars": source_chars,
        "attempt": int(getattr(provider, "last_attempt_count", None) or 1),
        "retry_count": int(getattr(provider, "retry_count", 0) or 0),
        "started_at": started_at,
        "finished_at": _utc_now(),
        "latency_seconds": round(latency, 3),
        "status": status,
        "error_class": error_class,
        "error_message": error_message,
        "prompt_tokens": prompt_tokens,
        "response_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "chars_per_second": round(source_chars / latency, 3) if status == "success" and latency > 0 else None,
    }


class MemorySampler:
    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = interval_seconds
        self._samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._samples.append(collect_memory_snapshot())
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        self._samples.append(collect_memory_snapshot())
        return _memory_summary(self._samples)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._samples.append(collect_memory_snapshot())


def collect_memory_snapshot() -> dict[str, Any]:
    ps = _collect_ps()
    vm = _collect_vm_stat()
    pressure = _collect_memory_pressure()
    total_memory_mb = _total_memory_mb()
    return {
        "rss_mb": ps.get("rss_mb"),
        "cpu_percent": ps.get("cpu_percent"),
        "processes": ps.get("processes"),
        "free_pages": vm.get("free_pages"),
        "speculative_pages": vm.get("speculative_pages"),
        "pageouts": vm.get("pageouts"),
        "pressure": pressure,
        "total_memory_mb": total_memory_mb,
    }


def recommend_safe_default(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {"safe_default": 1, "reason": "no benchmark runs were completed"}
    by_concurrency = {int(run["concurrency"]): run for run in runs}
    baseline = by_concurrency.get(1) or runs[0]
    baseline_reject = _rejection_reason(baseline, baseline)
    if baseline_reject:
        return {"safe_default": 1, "reason": f"insufficient evidence; baseline failed: {baseline_reject}"}
    lower_timeout = _lowest_timed_out_concurrency(runs)
    accepted = {
        int(run["concurrency"]): run
        for run in runs
        if _rejection_reason(run, baseline) is None and _lower_timeout_rejection(run, lower_timeout) is None
    }
    rejected = [
        f"concurrency={int(run['concurrency'])} rejected: {_combined_rejection_reason(run, baseline, lower_timeout)}"
        for run in runs
        if _combined_rejection_reason(run, baseline, lower_timeout) is not None
    ]
    safe = 1
    one = accepted.get(1, baseline)
    two = accepted.get(2)
    four = accepted.get(4)
    if two is not None and _throughput(two) >= _throughput(one) * 1.2:
        safe = 2
    if safe == 2 and four is not None and _throughput(four) >= _throughput(two) * 1.2:
        if _rate_value(four, "timeout_rate") == 0 and _rate_value(four, "error_rate") == 0 and _rate_value(four, "parse_error_rate") == 0:
            safe = 4
    reason_parts = [
        f"safe_default={safe} selected from accepted concurrencies {sorted(accepted)}",
    ]
    if safe == 1 and two is not None:
        reason_parts.append("concurrency=2 did not improve chunks/min by at least 20%")
    if safe == 2 and four is not None:
        reason_parts.append("concurrency=4 did not improve over 2 by at least 20%")
    reason_parts.extend(rejected)
    return {"safe_default": safe, "reason": "; ".join(reason_parts)}


def _normalize_concurrencies(values: list[int]) -> list[int]:
    normalized = []
    for value in values:
        concurrency = max(1, int(value))
        if concurrency not in normalized:
            normalized.append(concurrency)
    return normalized or [1]


def _lowest_timed_out_concurrency(runs: list[dict[str, Any]]) -> int | None:
    timed_out = [
        int(run["concurrency"])
        for run in runs
        if _run_timed_out(run)
    ]
    return min(timed_out) if timed_out else None


def _lower_timeout_rejection(run: dict[str, Any], lower_timeout: int | None) -> str | None:
    if lower_timeout is None:
        return None
    if int(run["concurrency"]) >= lower_timeout:
        return "lower concurrency timed out"
    return None


def _combined_rejection_reason(
    run: dict[str, Any],
    baseline: dict[str, Any],
    lower_timeout: int | None,
) -> str | None:
    reasons = [
        reason
        for reason in [
            _rejection_reason(run, baseline),
            _lower_timeout_rejection(run, lower_timeout),
        ]
        if reason
    ]
    if not reasons:
        return None
    deduped = list(dict.fromkeys(reasons))
    return "; ".join(deduped)


def _run_timed_out(run: dict[str, Any]) -> bool:
    return bool(run.get("timed_out_chunk_indexes")) or float(run.get("timeout_rate") or 0) > 0


def _skipped_run(
    concurrency: int,
    payloads: list[dict[str, Any]],
    timeout_seconds: float,
    run_timeout_seconds: float | None,
    reason: str,
) -> dict[str, Any]:
    now = _utc_now()
    chunks_total = len(payloads)
    return {
        "concurrency": int(concurrency),
        "run_status": "skipped",
        "run_timeout_seconds": round(
            _resolve_run_timeout_seconds(chunks_total, concurrency, timeout_seconds, run_timeout_seconds),
            3,
        ),
        "timed_out_chunk_indexes": [],
        "skipped_reason": reason,
        "started_at": now,
        "finished_at": now,
        "total_elapsed_seconds": 0,
        "chunks_total": chunks_total,
        "chunks_succeeded": 0,
        "chunks_failed": chunks_total,
        "error_rate": 0,
        "timeout_rate": 0,
        "parse_error_rate": 0,
        "retry_count": 0,
        "latency_seconds": {"p50": None, "p90": None, "p95": None, "max": None},
        "throughput": {"chunks_per_minute": None, "tokens_per_second": None, "chars_per_second": None},
        "memory": {},
        "chunk_results": [],
    }


def _select_chunks(chunks: list[dict[str, Any]], sample_chunks: int, selection: str) -> list[dict[str, Any]]:
    if sample_chunks <= 0 or sample_chunks >= len(chunks):
        return list(chunks)
    if selection == "first":
        return list(chunks[:sample_chunks])
    if selection != "evenly-spaced":
        raise ValueError("chunk_selection must be first or evenly-spaced")
    if sample_chunks == 1:
        return [chunks[0]]
    indexes: list[int] = []
    for index in range(sample_chunks):
        selected = round(index * (len(chunks) - 1) / (sample_chunks - 1))
        if selected not in indexes:
            indexes.append(selected)
    return [chunks[index] for index in indexes]


def _chunk_source_chars(transcript: Transcript, payload: dict[str, Any]) -> int:
    start_index = int(payload["source_segment_start_index"])
    end_index = int(payload["source_segment_end_index"])
    return sum(len(segment.text or "") for segment in transcript.segments[start_index : end_index + 1])


def _status_for_exception(exc: Exception) -> str:
    message = str(exc).lower()
    if isinstance(exc, TimeoutError) or "timed out" in message or "timeout" in message:
        return "timeout"
    if isinstance(exc, TranslationError) and (
        "json" in message or "segment" in message or "missing" in message or "response" in message
    ):
        return "parse_error"
    return "error"


def _short_error(message: str, limit: int = 240) -> str:
    text = " ".join(message.split())
    return text if len(text) <= limit else text[:limit].rstrip()


def _sum_usage(usages: list[dict[str, Any]]) -> dict[str, int] | None:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    found = False
    for usage in usages:
        if not isinstance(usage, dict):
            continue
        for key in totals:
            value = usage.get(key)
            if isinstance(value, (int, float)):
                totals[key] += int(value)
                found = True
    return totals if found else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def _rate(count: int, total: int) -> float:
    return round(count / total, 6) if total else 0.0


def _rate_value(run: dict[str, Any], key: str) -> float:
    return float(run.get(key) or 0)


def _throughput(run: dict[str, Any]) -> float:
    throughput = run.get("throughput") or {}
    return float(throughput.get("chunks_per_minute") or 0)


def _rejection_reason(run: dict[str, Any], baseline: dict[str, Any]) -> str | None:
    if run.get("run_status") == "skipped":
        return str(run.get("skipped_reason") or "skipped")
    if _rate_value(run, "timeout_rate") > 0:
        return "timeout_rate > 0"
    if _rate_value(run, "error_rate") > 0.05:
        return "error_rate > 0.05"
    if _rate_value(run, "parse_error_rate") > 0.05:
        return "parse_error_rate > 0.05"
    if int(run.get("chunks_succeeded") or 0) < int(run.get("chunks_total") or 0):
        return "successful chunks < chunks_total"
    p95 = ((run.get("latency_seconds") or {}).get("p95"))
    baseline_p95 = ((baseline.get("latency_seconds") or {}).get("p95"))
    if p95 is not None and baseline_p95 not in {None, 0} and float(p95) > float(baseline_p95) * 2.5:
        return "p95 latency > 2.5x concurrency=1 p95"
    memory = run.get("memory") or {}
    if memory.get("memory_pressure"):
        return "memory pressure detected"
    return None


def _memory_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    rss_values = [float(sample["rss_mb"]) for sample in samples if isinstance(sample.get("rss_mb"), (int, float))]
    pageouts = [int(sample["pageouts"]) for sample in samples if isinstance(sample.get("pageouts"), int)]
    total_values = [
        float(sample["total_memory_mb"]) for sample in samples if isinstance(sample.get("total_memory_mb"), (int, float))
    ]
    pressure_values = [str(sample.get("pressure") or "") for sample in samples]
    pageouts_delta = pageouts[-1] - pageouts[0] if len(pageouts) >= 2 else None
    rss_before = rss_values[0] if rss_values else None
    rss_peak = max(rss_values) if rss_values else None
    rss_after = rss_values[-1] if rss_values else None
    total_memory_mb = total_values[-1] if total_values else None
    pressure_text = " ".join(value for value in pressure_values if value).strip().lower()
    pressure_bad = _pressure_text_indicates_pressure(pressure_text)
    rss_bad = bool(rss_peak is not None and total_memory_mb and rss_peak >= total_memory_mb * 0.9)
    pageouts_bad = bool(pageouts_delta is not None and pageouts_delta > 0)
    return {
        "rss_mb_before": _round_or_none(rss_before),
        "rss_mb_peak": _round_or_none(rss_peak),
        "rss_mb_after": _round_or_none(rss_after),
        "pageouts_delta": pageouts_delta,
        "pressure": next((value for value in pressure_values if value), None),
        "memory_pressure": pressure_bad or rss_bad or pageouts_bad,
        "sample_count": len(samples),
    }


def _pressure_text_indicates_pressure(pressure_text: str) -> bool:
    if not pressure_text:
        return False
    lowered = pressure_text.lower()
    bad_markers = [
        "critical",
        "warning",
        "warn:",
        "memory pressure: high",
        "memory pressure: critical",
        "pressure level: high",
        "pressure level: critical",
    ]
    return any(marker in lowered for marker in bad_markers)


def _collect_ps() -> dict[str, Any]:
    completed = _run_command(["ps", "-axo", "pid,comm,rss,%cpu"])
    if completed is None:
        return {"rss_mb": None, "cpu_percent": None, "processes": []}
    processes = []
    rss_total = 0.0
    cpu_total = 0.0
    for line in completed.stdout.splitlines()[1:]:
        parts = line.split(None, 3)
        if len(parts) != 4:
            continue
        pid, command, rss, cpu = parts
        lowered = command.lower()
        if not any(marker in lowered for marker in ["lm studio", "lmstudio", "llmworker", "node"]):
            continue
        try:
            rss_mb = int(rss) / 1024
            cpu_percent = float(cpu)
        except ValueError:
            continue
        rss_total += rss_mb
        cpu_total += cpu_percent
        processes.append({"pid": int(pid), "command": command, "rss_mb": round(rss_mb, 3), "cpu_percent": cpu_percent})
    return {
        "rss_mb": round(rss_total, 3) if processes else None,
        "cpu_percent": round(cpu_total, 3) if processes else None,
        "processes": processes,
    }


def _collect_vm_stat() -> dict[str, Any]:
    completed = _run_command(["vm_stat"])
    if completed is None:
        return {"free_pages": None, "speculative_pages": None, "pageouts": None}
    values: dict[str, int | None] = {"free_pages": None, "speculative_pages": None, "pageouts": None}
    for line in completed.stdout.splitlines():
        normalized = line.strip().rstrip(".")
        if ":" not in normalized:
            continue
        key, raw = normalized.split(":", 1)
        try:
            value = int(raw.strip().replace(".", ""))
        except ValueError:
            continue
        lowered = key.lower()
        if lowered == "pages free":
            values["free_pages"] = value
        elif lowered == "pages speculative":
            values["speculative_pages"] = value
        elif lowered == "pageouts":
            values["pageouts"] = value
    return values


def _collect_memory_pressure() -> str | None:
    completed = _run_command(["memory_pressure"], timeout=2)
    if completed is None:
        return None
    text = " ".join(line.strip() for line in completed.stdout.splitlines() if line.strip())
    return text[:240] or None


def _total_memory_mb() -> float | None:
    completed = _run_command(["sysctl", "-n", "hw.memsize"])
    if completed is None:
        return None
    try:
        return round(int(completed.stdout.strip()) / 1024 / 1024, 3)
    except ValueError:
        return None


def _run_command(args: list[str], timeout: float = 1) -> subprocess.CompletedProcess[str] | None:
    try:
        completed = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed


def _default_output_path(library_dir: Path, video_id: str, created_at: str) -> Path:
    safe_timestamp = created_at.replace(":", "").replace("-", "").replace("+", "Z")
    return library_dir / "benchmarks" / f"lm_studio_translation_{video_id}_{safe_timestamp}.json"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _round_or_none(value: float | None) -> float | None:
    return round(float(value), 3) if value is not None else None


def _int_or_zero(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) else 0
