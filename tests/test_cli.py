from typer.testing import CliRunner
import json
from pathlib import Path

from yt_sub_hub import cli
from yt_sub_hub.ingestion import YouTubeAuthRequiredError
from yt_sub_hub.models import JobRecord


class FakePipeline:
    def __init__(self, existing=None):
        self.existing = existing
        self.process_calls = []

    def already_processed(self, url):
        return self.existing

    def process(self, url):
        self.process_calls.append(url)
        return JobRecord(
            job_id="job123",
            url=url,
            video_id="abc123def45",
            status="completed",
            note_path="library/videos/abc123def45/note.md",
        )


class FakeBenchmarkOptions:
    def __init__(
        self,
        *,
        video_id,
        concurrencies,
        sample_chunks,
        timeout_seconds,
        run_timeout_seconds,
        continue_after_timeout,
        output,
        chunk_selection,
    ):
        self.video_id = video_id
        self.concurrencies = concurrencies
        self.sample_chunks = sample_chunks
        self.timeout_seconds = timeout_seconds
        self.run_timeout_seconds = run_timeout_seconds
        self.continue_after_timeout = continue_after_timeout
        self.output = output
        self.chunk_selection = chunk_selection


def test_process_prompts_and_skips_existing_video(monkeypatch):
    fake = FakePipeline(existing=("abc123def45", "library/videos/abc123def45/note.md"))
    monkeypatch.setattr(cli, "_pipeline", lambda: fake)

    result = CliRunner().invoke(cli.app, ["process", "https://youtu.be/abc123def45"], input="n\n")

    assert result.exit_code == 0
    assert "уже анализировалось" in result.output
    assert "skipped_existing" in result.output
    assert fake.process_calls == []


def test_process_reanalyzes_existing_video_after_confirmation(monkeypatch):
    fake = FakePipeline(existing=("abc123def45", "library/videos/abc123def45/note.md"))
    monkeypatch.setattr(cli, "_pipeline", lambda: fake)

    result = CliRunner().invoke(cli.app, ["process", "https://youtu.be/abc123def45"], input="y\n")

    assert result.exit_code == 0
    assert fake.process_calls == ["https://youtu.be/abc123def45"]
    assert "completed: abc123def45" in result.output


def test_process_force_bypasses_existing_video_prompt(monkeypatch):
    fake = FakePipeline(existing=("abc123def45", "library/videos/abc123def45/note.md"))
    monkeypatch.setattr(cli, "_pipeline", lambda: fake)

    result = CliRunner().invoke(cli.app, ["process", "--force", "https://youtu.be/abc123def45"])

    assert result.exit_code == 0
    assert "уже анализировалось" not in result.output
    assert fake.process_calls == ["https://youtu.be/abc123def45"]


def test_process_reports_youtube_auth_error_without_traceback(monkeypatch):
    class AuthErrorPipeline(FakePipeline):
        def process(self, url):
            raise YouTubeAuthRequiredError(
                "Видео требует авторизованные cookies. Включите cookies явно через "
                "YT_SUB_HUB_YTDLP_COOKIES_FILE=/secure/path/youtube-cookies.txt. "
                "Browser cookies are disabled by default."
            )

    monkeypatch.setattr(cli, "_pipeline", lambda: AuthErrorPipeline())

    result = CliRunner().invoke(cli.app, ["process", "--force", "https://youtu.be/abc123def45"])

    assert result.exit_code == 1
    assert "YT_SUB_HUB_YTDLP_COOKIES_FILE=/secure/path/youtube-cookies.txt" in result.output
    assert "Traceback" not in result.output


def test_setup_whisper_cpp_downloads_model_and_prints_env(tmp_path, monkeypatch):
    binary = tmp_path / "whisper-cli"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    monkeypatch.setenv("YT_SUB_HUB_WHISPER_MODEL", "base")
    monkeypatch.setattr("shutil.which", lambda name: str(binary) if name == "whisper-cli" else None)

    def fake_urlretrieve(url, filename):
        Path(filename).write_bytes(b"model")
        return filename, None

    from pathlib import Path

    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)

    result = CliRunner().invoke(cli.app, ["setup-whisper-cpp", "--download-model"])

    model = tmp_path / "assets" / "whisper.cpp" / "models" / "ggml-base.bin"
    assert result.exit_code == 0
    assert model.exists()
    assert f"binary: {binary}" in result.output
    assert f"model: {model}" in result.output
    assert "model_ready: True" in result.output
    assert f"YT_SUB_HUB_WHISPER_CPP_MODEL={model}" in result.output


def test_inspect_graph_reads_local_graph_extraction(tmp_path, monkeypatch):
    video_dir = tmp_path / "videos" / "abc123"
    video_dir.mkdir(parents=True)
    (video_dir / "graph_extraction.json").write_text(
        '{"schema_version":"graph-extraction-v1","video_id":"abc123","generated_from":{},'
        '"chunks":[],"nodes":[{"id":"concept:graph","label":"Concept"}],'
        '"relationships":[{"id":"rel1","type":"MENTIONS"}],'
        '"quality_report":{"node_count_by_label":{"Concept":1},"relationship_count_by_type":{"MENTIONS":1},'
        '"rejected_relationships":[],"duplicate_concept_candidates":[]}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))

    result = CliRunner().invoke(cli.app, ["inspect-graph", "abc123"])

    assert result.exit_code == 0
    assert "graph-extraction-v1" in result.output
    assert "Concept: 1" in result.output
    assert "MENTIONS: 1" in result.output


def test_graph_export_writes_json_export(tmp_path, monkeypatch):
    video_dir = tmp_path / "videos" / "abc123"
    video_dir.mkdir(parents=True)
    (video_dir / "graph_extraction.json").write_text(
        '{"schema_version":"graph-extraction-v1","video_id":"abc123","generated_from":{},'
        '"chunks":[],"nodes":[{"id":"video:abc123","label":"Video"}],"relationships":[],"quality_report":{}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))

    result = CliRunner().invoke(cli.app, ["graph-export", "--format", "json"])

    assert result.exit_code == 0
    export_path = tmp_path / "graph" / "exports" / "graph.json"
    assert export_path.exists()
    assert "exported graph" in result.output


def test_youtube_discovery_cli_commands_delegate_to_service(tmp_path, monkeypatch):
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    calls = {}

    def fake_run(store, video_id, **kwargs):
        calls["run"] = (video_id, kwargs)
        return {"status": {"status": "completed"}, "video_id": video_id}

    def fake_status(store, video_id):
        return {"video_id": video_id, "status": {"status": "completed"}}

    def fake_enqueue(runtime, store, video_id, **kwargs):
        calls["enqueue"] = (video_id, kwargs)
        return {"queued_count": 2}

    monkeypatch.setattr("yt_sub_hub.youtube_discovery.run_youtube_discovery", fake_run)
    monkeypatch.setattr("yt_sub_hub.youtube_discovery.build_discovery_payload", fake_status)
    monkeypatch.setattr("yt_sub_hub.youtube_discovery.enqueue_discovered_videos", fake_enqueue)

    discover = CliRunner().invoke(cli.app, ["discover-youtube", "abc123", "--limit", "7", "--max-quota-units", "300"])
    status = CliRunner().invoke(cli.app, ["discovery-status", "abc123", "--json"])
    enqueue = CliRunner().invoke(cli.app, ["enqueue-discovered-videos", "abc123", "--min-score", "0.5", "--limit", "2"])

    assert discover.exit_code == 0
    assert calls["run"][0] == "abc123"
    assert calls["run"][1]["limit"] == 7
    assert calls["run"][1]["max_quota_units"] == 300
    assert status.exit_code == 0
    assert json.loads(status.output)["status"]["status"] == "completed"
    assert enqueue.exit_code == 0
    assert calls["enqueue"][1]["min_score"] == 0.5
    assert calls["enqueue"][1]["limit"] == 2


def test_rebuild_graph_dry_run_does_not_create_neo4j_store(monkeypatch):
    class FakePipeline:
        def __init__(self):
            self.calls = []

        def rebuild_graph(self, graph_store=None, **kwargs):
            self.calls.append((graph_store, kwargs))
            assert graph_store is None
            assert kwargs["dry_run"] is True
            return 3

    fake = FakePipeline()

    def fail_graph_store(*args, **kwargs):
        raise AssertionError("dry-run must not construct Neo4jGraphStore")

    monkeypatch.setattr(cli, "_pipeline", lambda: fake)
    monkeypatch.setattr("yt_sub_hub.graph.Neo4jGraphStore", fail_graph_store)

    result = CliRunner().invoke(cli.app, ["rebuild-graph", "--dry-run", "--clear"])

    assert result.exit_code == 0
    assert "rebuilt graph from 3 videos (dry run)" in result.output


def test_inspect_graph_prints_samples_and_low_confidence_count(tmp_path, monkeypatch):
    video_dir = tmp_path / "videos" / "abc123"
    video_dir.mkdir(parents=True)
    (video_dir / "graph_extraction.json").write_text(
        '{"schema_version":"graph-extraction-v1","video_id":"abc123","generated_from":{},'
        '"chunks":[{"chunk_id":"chunk-1"}],'
        '"nodes":[{"id":"concept:graph","label":"Concept","name":"Graph"}],'
        '"relationships":[{"id":"rel1","type":"MENTIONS","from":"video:abc123","to":"concept:graph","confidence":1.0}],'
        '"quality_report":{"node_count_by_label":{"Concept":1},"relationship_count_by_type":{"MENTIONS":1},'
        '"rejected_relationships":[{"reason":"unknown_local_id"}],"low_confidence_extractions":[{"reason":"low_confidence"}],'
        '"duplicate_concept_candidates":[]}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))

    result = CliRunner().invoke(cli.app, ["inspect-graph", "abc123", "--limit", "1"])

    assert result.exit_code == 0
    assert "chunks: 1" in result.output
    assert "low_confidence_extractions: 1" in result.output
    assert "Concept concept:graph Graph" in result.output
    assert "MENTIONS video:abc123 -> concept:graph confidence=1.0" in result.output


def test_inspect_graph_summarizes_llm_relationships_and_rejection_reasons(tmp_path, monkeypatch):
    video_dir = tmp_path / "videos" / "abc123"
    video_dir.mkdir(parents=True)
    (video_dir / "graph_extraction.json").write_text(
        json.dumps(
            {
                "schema_version": "graph-extraction-v1",
                "video_id": "abc123",
                "generated_from": {},
                "chunks": [{"chunk_id": "chunk-1", "start": 0, "end": 120}],
                "nodes": [
                    {"id": "concept:graph", "label": "Concept", "name": "Graph"},
                    {"id": "concept:llm", "label": "Concept", "name": "LLM"},
                    {"id": "concept:notes", "label": "Concept", "name": "Notes"},
                ],
                "relationships": [
                    {
                        "id": "rel-llm-1",
                        "type": "RELATED_TO",
                        "from": "concept:graph",
                        "to": "concept:llm",
                        "confidence": 0.92,
                        "reliability": "llm_extracted",
                        "chunk_id": "chunk-1",
                        "audit_status": "accepted",
                    },
                    {
                        "id": "rel-llm-2",
                        "type": "CAUSES",
                        "from": "concept:llm",
                        "to": "concept:notes",
                        "confidence": 0.81,
                        "extraction_method": "llm_deep_graph",
                        "audit_status": "accepted",
                    },
                    {
                        "id": "rel-template",
                        "type": "MENTIONS",
                        "from": "video:abc123",
                        "to": "concept:graph",
                        "confidence": 1.0,
                    },
                    {
                        "id": "rel-rejected",
                        "type": "CONTRADICTS",
                        "from": "concept:notes",
                        "to": "concept:graph",
                        "confidence": 0.5,
                        "reliability": "llm_extracted",
                        "audit_status": "rejected",
                    },
                ],
                "quality_report": {
                    "rejected_relationships": [
                        {"reason": "unknown_local_id"},
                        {"reason": "unsupported_relationship_type"},
                        {"reason": "unknown_local_id"},
                    ],
                    "low_confidence_extractions": [
                        {"reason": "low_confidence"},
                        {"reason": "low_confidence_concept"},
                    ],
                    "duplicate_concept_candidates": [],
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))

    result = CliRunner().invoke(cli.app, ["inspect-graph", "abc123", "--limit", "3"])

    assert result.exit_code == 0
    assert "chunks: 1" in result.output
    assert "sample chunks:" in result.output
    assert "chunk-1" in result.output
    assert "llm_relationships: 2" in result.output
    assert "RELATED_TO: 1" in result.output
    assert "CAUSES: 1" in result.output
    assert "CONTRADICTS: 1" in result.output
    assert "rejection reasons:" in result.output
    assert "unknown_local_id: 2" in result.output
    assert "low_confidence_concept: 1" in result.output
    assert "sample llm relationships:" in result.output
    llm_samples = result.output.split("sample llm relationships:", 1)[1]
    assert "RELATED_TO concept:graph -> concept:llm confidence=0.92" in llm_samples
    assert "CAUSES concept:llm -> concept:notes confidence=0.81" in llm_samples
    assert "MENTIONS video:abc123 -> concept:graph" not in llm_samples


def test_graph_export_returns_1_when_video_id_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))

    result = CliRunner().invoke(cli.app, ["graph-export", "--format", "json", "--video-id", "missing"])

    assert result.exit_code == 1
    assert "graph extraction not found" in result.output


def test_benchmark_whisper_accepts_audio_path_and_does_not_write_transcript_artifacts(tmp_path, monkeypatch):
    audio_path = tmp_path / "sample.mp3"
    audio_path.write_bytes(b"audio")
    video_dir = tmp_path / "videos" / "sample"
    video_dir.mkdir(parents=True)
    calls = []

    def fake_benchmark(target, *, settings, seconds=30):
        calls.append((target, settings.library_dir, seconds))
        return {
            "backend": "openai-whisper-cpu",
            "model": "base",
            "device": "cpu",
            "elapsed_seconds": 0.25,
            "language": "en",
            "segments": 1,
            "sample": "Hello world",
        }

    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    monkeypatch.setattr("yt_sub_hub.cli.run_whisper_benchmark", fake_benchmark)

    result = CliRunner().invoke(cli.app, ["benchmark-whisper", str(audio_path)])

    assert result.exit_code == 0
    assert calls == [(str(audio_path), tmp_path, 30)]
    assert "backend: openai-whisper-cpu" in result.output
    assert "elapsed_seconds: 0.25" in result.output
    assert not (video_dir / "transcript.raw.json").exists()
    assert not (video_dir / "transcript.normalized.json").exists()


def test_captions_probe_human_output_contains_decision_and_blocker(monkeypatch):
    payload = {
        "schema_version": "captions_probe.v1",
        "target": "abc123def45",
        "target_kind": "video",
        "limit": 3,
        "target_language": "ru",
        "media_download_mode": "disabled",
        "summary": {
            "total_videos": 1,
            "data_api_caption_ready_count": 0,
            "needs_whisper_fallback_count": 1,
            "unavailable_count": 0,
            "can_run_data_api_only": False,
            "can_run_full_processing": False,
        },
        "videos": [
            {
                "video_id": "abc123def45",
                "metadata": {"status": "ok", "title": "Video", "channel": "Channel"},
                "captions_count": 0,
                "downloadable": False,
                "transcript_source_decision": "needs_whisper_fallback",
                "pipeline_blocker": "captions_unavailable_media_download_disabled",
            }
        ],
    }
    monkeypatch.setattr("yt_sub_hub.captions_probe.probe_captions", lambda target, limit=None, settings=None: payload)

    result = CliRunner().invoke(cli.app, ["captions-probe", "abc123def45"])

    assert result.exit_code == 0
    assert "Video abc123def45" in result.output
    assert "decision: needs_whisper_fallback" in result.output
    assert "blocker: captions_unavailable_media_download_disabled" in result.output


def test_captions_probe_json_is_valid_and_secret_free(monkeypatch):
    payload = {
        "schema_version": "captions_probe.v1",
        "target": "abc123def45",
        "target_kind": "video",
        "limit": 3,
        "target_language": "ru",
        "media_download_mode": "enabled",
        "summary": {
            "total_videos": 1,
            "data_api_caption_ready_count": 1,
            "needs_whisper_fallback_count": 0,
            "unavailable_count": 0,
            "can_run_data_api_only": True,
            "can_run_full_processing": True,
        },
        "videos": [
            {
                "video_id": "abc123def45",
                "metadata": {"status": "ok"},
                "captions_count": 1,
                "captions": [{"caption_id_masked": "cap...123", "language": "ru"}],
                "downloadable": True,
                "transcript_source_decision": "data_api_captions_ready",
                "pipeline_blocker": None,
            }
        ],
    }
    monkeypatch.setattr("yt_sub_hub.captions_probe.probe_captions", lambda target, limit=None, settings=None: payload)

    result = CliRunner().invoke(cli.app, ["captions-probe", "abc123def45", "--json"])

    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["schema_version"] == "captions_probe.v1"
    dumped = result.output.lower()
    assert "client_secret" not in dumped
    assert "access_token" not in dumped
    assert "refresh_token" not in dumped
    assert "caption-ru-secret" not in dumped


def test_captions_probe_cli_unavailable_video_json_returns_controlled_payload(tmp_path, monkeypatch):
    class FakeRequest:
        def execute(self):
            return {"items": []}

    class FakeYouTube:
        def videos(self):
            return self

        def list(self, **kwargs):
            return FakeRequest()

    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    monkeypatch.setenv("YT_SUB_HUB_YOUTUBE_API_TOKEN_FILE", str(tmp_path / "token.json"))
    (tmp_path / "token.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("yt_sub_hub.captions_probe.build_youtube_client", lambda settings, persist_refresh=False: FakeYouTube())

    result = CliRunner().invoke(cli.app, ["captions-probe", "zzzzzzzzzzz", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["target_kind"] == "video"
    assert payload["summary"]["unavailable_count"] == 1
    assert payload["videos"][0]["metadata"]["status"] == "error"
    assert payload["videos"][0]["transcript_source_decision"] == "unavailable"


def test_captions_probe_cli_unavailable_video_human_output(tmp_path, monkeypatch):
    class FakeRequest:
        def execute(self):
            return {"items": []}

    class FakeYouTube:
        def videos(self):
            return self

        def list(self, **kwargs):
            return FakeRequest()

    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    monkeypatch.setenv("YT_SUB_HUB_YOUTUBE_API_TOKEN_FILE", str(tmp_path / "token.json"))
    (tmp_path / "token.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("yt_sub_hub.captions_probe.build_youtube_client", lambda settings, persist_refresh=False: FakeYouTube())

    result = CliRunner().invoke(cli.app, ["captions-probe", "zzzzzzzzzzz"])

    assert result.exit_code == 0
    assert "metadata: error" in result.output
    assert "decision: unavailable" in result.output


def test_benchmark_lm_studio_translation_parses_options_and_writes_report(tmp_path, monkeypatch):
    calls = []
    report_path = tmp_path / "benchmarks" / "report.json"

    def fake_benchmark(store, settings, options):
        calls.append((store.root, settings.library_dir, options))
        report = {
            "video_id": options.video_id,
            "model": settings.translation_model,
            "base_url": settings.lm_studio_base_url,
            "created_at": "2026-05-06T00:00:00Z",
            "sample": {"chunks_requested": options.sample_chunks, "chunks_available": 12, "chunk_selection": options.chunk_selection},
            "runs": [
                {
                    "concurrency": 1,
                    "chunks_total": 3,
                    "chunks_succeeded": 3,
                    "chunks_failed": 0,
                    "timeout_rate": 0,
                    "error_rate": 0,
                    "parse_error_rate": 0,
                    "latency_seconds": {"p50": 1.2, "p95": 1.8},
                    "throughput": {"chunks_per_minute": 12.0},
                },
                {
                    "concurrency": 2,
                    "chunks_total": 3,
                    "chunks_succeeded": 3,
                    "chunks_failed": 0,
                    "timeout_rate": 0,
                    "error_rate": 0,
                    "parse_error_rate": 0,
                    "latency_seconds": {"p50": 1.1, "p95": 1.6},
                    "throughput": {"chunks_per_minute": 20.0},
                },
                {
                    "concurrency": 4,
                    "chunks_total": 3,
                    "chunks_succeeded": 3,
                    "chunks_failed": 0,
                    "timeout_rate": 0,
                    "error_rate": 0,
                    "parse_error_rate": 0,
                    "latency_seconds": {"p50": 1.1, "p95": 1.7},
                    "throughput": {"chunks_per_minute": 21.0},
                },
            ],
            "recommendation": {"safe_default": 2, "reason": "concurrency=2 is stable"},
        }
        report_path.parent.mkdir(parents=True)
        report_path.write_text(json.dumps(report), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report

    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    monkeypatch.setenv("YT_SUB_HUB_LM_STUDIO_BENCHMARK_RUN_TIMEOUT_SECONDS", "800")
    monkeypatch.setattr("yt_sub_hub.cli.BenchmarkOptions", FakeBenchmarkOptions)
    monkeypatch.setattr("yt_sub_hub.cli.benchmark_lm_studio_translation", fake_benchmark)

    result = CliRunner().invoke(
        cli.app,
        [
            "benchmark-lm-studio-translation",
            "abc123def45",
            "--concurrency",
            "1,2,4",
            "--sample-chunks",
            "3",
            "--timeout-seconds",
            "45",
            "--run-timeout-seconds",
            "300",
            "--chunk-selection",
            "first",
            "--output",
            str(report_path),
        ],
    )

    assert result.exit_code == 0
    assert len(calls) == 1
    options = calls[0][2]
    assert options.concurrencies == [1, 2, 4]
    assert options.sample_chunks == 3
    assert options.timeout_seconds == 45
    assert options.run_timeout_seconds == 300
    assert options.continue_after_timeout is False
    assert options.output == report_path
    assert options.chunk_selection == "first"
    assert "concurrency=1 success=3/3 timeout=0 error=0 parse_error=0 p50=1.2 p95=1.8 chunks/min=12.0" in result.output
    assert "recommendation safe_default=2 reason=concurrency=2 is stable" in result.output
    assert f"report: {report_path}" in result.output
    assert report_path.exists()


def test_benchmark_lm_studio_translation_json_output(tmp_path, monkeypatch):
    def fake_benchmark(store, settings, options):
        return {"video_id": options.video_id, "runs": [], "recommendation": {"safe_default": 1, "reason": "test"}}

    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    monkeypatch.setattr("yt_sub_hub.cli.BenchmarkOptions", FakeBenchmarkOptions)
    monkeypatch.setattr("yt_sub_hub.cli.benchmark_lm_studio_translation", fake_benchmark)

    result = CliRunner().invoke(cli.app, ["benchmark-lm-studio-translation", "abc123def45", "--concurrency", "2", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.output)["recommendation"]["safe_default"] == 1


def test_artifacts_cli_prints_browser_payload_json_for_video(tmp_path, monkeypatch):
    video_dir = tmp_path / "videos" / "abc123"
    video_dir.mkdir(parents=True)
    (video_dir / "metadata.json").write_text('{"video_id":"abc123","title":"CLI Video"}', encoding="utf-8")
    (video_dir / "note.md").write_text("# CLI Note\n", encoding="utf-8")
    (video_dir / "graph_extraction.json").write_text('{"nodes":[]}', encoding="utf-8")
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))

    result = CliRunner().invoke(cli.app, ["artifacts", "abc123", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["video_id"] == "abc123"
    assert payload["title"] == "CLI Video"
    assert any(artifact["key"] == "note_md" for artifact in payload["artifacts"])


def test_open_artifact_cli_prints_resolved_artifact_without_gui(tmp_path, monkeypatch):
    video_dir = tmp_path / "videos" / "abc123"
    video_dir.mkdir(parents=True)
    note_path = video_dir / "note.md"
    note_path.write_text("# CLI Note\n", encoding="utf-8")
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))

    result = CliRunner().invoke(cli.app, ["open-artifact", "abc123", "note_md"])

    assert result.exit_code == 0
    assert f"path: {note_path}" in result.output
    assert "api_url: /videos/abc123/artifacts/note_md" in result.output
    assert "download_url: /videos/abc123/artifacts/note_md/download" in result.output


def test_benchmark_lm_studio_translation_preserves_explicit_zero_sample_chunks(tmp_path, monkeypatch):
    calls = []

    def fake_benchmark(store, settings, options):
        calls.append(options)
        return {"video_id": options.video_id, "runs": [], "recommendation": {"safe_default": 1, "reason": "test"}}

    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    monkeypatch.setattr("yt_sub_hub.cli.BenchmarkOptions", FakeBenchmarkOptions)
    monkeypatch.setattr("yt_sub_hub.cli.benchmark_lm_studio_translation", fake_benchmark)

    result = CliRunner().invoke(
        cli.app,
        ["benchmark-lm-studio-translation", "abc123def45", "--sample-chunks", "0", "--timeout-seconds", "0", "--json"],
    )

    assert result.exit_code == 0
    assert calls[0].sample_chunks == 0
    assert calls[0].timeout_seconds == 0


def test_benchmark_lm_studio_translation_uses_config_run_timeout_by_default(tmp_path, monkeypatch):
    calls = []

    def fake_benchmark(store, settings, options):
        calls.append(options)
        return {"video_id": options.video_id, "runs": [], "recommendation": {"safe_default": 1, "reason": "test"}}

    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    monkeypatch.setenv("YT_SUB_HUB_LM_STUDIO_BENCHMARK_RUN_TIMEOUT_SECONDS", "650")
    monkeypatch.setattr("yt_sub_hub.cli.BenchmarkOptions", FakeBenchmarkOptions)
    monkeypatch.setattr("yt_sub_hub.cli.benchmark_lm_studio_translation", fake_benchmark)

    result = CliRunner().invoke(cli.app, ["benchmark-lm-studio-translation", "abc123def45", "--json"])

    assert result.exit_code == 0
    assert calls[0].run_timeout_seconds == 650
    assert calls[0].continue_after_timeout is False


def test_benchmark_lm_studio_translation_stop_after_timeout_overrides_default(tmp_path, monkeypatch):
    calls = []

    def fake_benchmark(store, settings, options):
        calls.append(options)
        return {"video_id": options.video_id, "runs": [], "recommendation": {"safe_default": 1, "reason": "test"}}

    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    monkeypatch.setattr("yt_sub_hub.cli.BenchmarkOptions", FakeBenchmarkOptions)
    monkeypatch.setattr("yt_sub_hub.cli.benchmark_lm_studio_translation", fake_benchmark)

    result = CliRunner().invoke(cli.app, ["benchmark-lm-studio-translation", "abc123def45", "--stop-after-timeout", "--json"])

    assert result.exit_code == 0
    assert calls[0].continue_after_timeout is False


def test_benchmark_lm_studio_translation_continue_after_timeout_can_be_explicit(tmp_path, monkeypatch):
    calls = []

    def fake_benchmark(store, settings, options):
        calls.append(options)
        return {"video_id": options.video_id, "runs": [], "recommendation": {"safe_default": 1, "reason": "test"}}

    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    monkeypatch.setattr("yt_sub_hub.cli.BenchmarkOptions", FakeBenchmarkOptions)
    monkeypatch.setattr("yt_sub_hub.cli.benchmark_lm_studio_translation", fake_benchmark)

    result = CliRunner().invoke(cli.app, ["benchmark-lm-studio-translation", "abc123def45", "--continue-after-timeout", "--json"])

    assert result.exit_code == 0
    assert calls[0].continue_after_timeout is True


def test_benchmark_lm_studio_translation_exits_zero_when_partial_error_report_exists(tmp_path, monkeypatch):
    from yt_sub_hub.lm_benchmark import BenchmarkReportError

    report_path = tmp_path / "benchmarks" / "partial.json"
    report = {
        "video_id": "abc123def45",
        "runs": [],
        "recommendation": {"safe_default": 1, "reason": "partial"},
        "benchmark_error": "boom",
        "report_path": str(report_path),
    }

    def fake_benchmark(store, settings, options):
        report_path.parent.mkdir(parents=True)
        report_path.write_text(json.dumps(report), encoding="utf-8")
        raise BenchmarkReportError("boom", report)

    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    monkeypatch.setattr("yt_sub_hub.cli.BenchmarkOptions", FakeBenchmarkOptions)
    monkeypatch.setattr("yt_sub_hub.cli.benchmark_lm_studio_translation", fake_benchmark)

    result = CliRunner().invoke(cli.app, ["benchmark-lm-studio-translation", "abc123def45"])

    assert result.exit_code == 0
    assert "recommendation safe_default=1 reason=partial" in result.output
    assert f"report: {report_path}" in result.output


def test_benchmark_lm_studio_translation_reports_errors_without_traceback(tmp_path, monkeypatch):
    def fake_benchmark(store, settings, options):
        raise RuntimeError("normalized transcript is required")

    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    monkeypatch.setattr("yt_sub_hub.cli.BenchmarkOptions", FakeBenchmarkOptions)
    monkeypatch.setattr("yt_sub_hub.cli.benchmark_lm_studio_translation", fake_benchmark)

    result = CliRunner().invoke(cli.app, ["benchmark-lm-studio-translation", "missing"])

    assert result.exit_code == 1
    assert "normalized transcript is required" in result.output
    assert "Traceback" not in result.output
