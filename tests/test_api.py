import sqlite3

from yt_sub_hub.api import create_job_record
from yt_sub_hub.models import JobRecord
from yt_sub_hub.runtime import RuntimeStore


def test_create_job_record_returns_queued_record_without_processing(tmp_path):
    record = create_job_record("https://youtu.be/abc123", tmp_path)

    assert record.status == "queued"
    assert record.video_id == "abc123"
    assert record.note_path is None


def test_create_job_record_is_idempotent_and_does_not_create_second_job(tmp_path):
    first = create_job_record("https://youtu.be/abc123", tmp_path)
    second = create_job_record("https://youtu.be/abc123", tmp_path)

    assert second.job_id == first.job_id


def test_create_job_api_accepts_quality_mode_override(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import yt_sub_hub.api as api_module
    from yt_sub_hub.api import create_app
    from yt_sub_hub.runtime import RuntimeStore

    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    monkeypatch.setattr(api_module, "process_existing_job", lambda job_id, library_dir=None: None)
    client = TestClient(create_app())

    response = client.post("/jobs", json={"url": "https://youtu.be/abc123", "quality_mode": "draft"})

    assert response.status_code == 200
    assert response.json()["quality_mode"] == "draft"
    assert RuntimeStore(tmp_path / "index.sqlite3").load_video_job(response.json()["job_id"]).quality_mode == "draft"


def test_create_job_api_accepts_acceptance_mode_override(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import yt_sub_hub.api as api_module
    from yt_sub_hub.api import create_app
    from yt_sub_hub.runtime import RuntimeStore

    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    monkeypatch.setattr(api_module, "process_existing_job", lambda job_id, library_dir=None: None)
    client = TestClient(create_app())

    response = client.post("/jobs", json={"url": "https://youtu.be/abc123", "acceptance_mode": "full_processing"})

    assert response.status_code == 200
    assert response.json()["acceptance_mode"] == "full_processing"
    assert RuntimeStore(tmp_path / "index.sqlite3").load_video_job(response.json()["job_id"]).acceptance_mode == "full_processing"


def test_process_existing_job_passes_runtime_quality_mode_to_pipeline_settings(tmp_path, monkeypatch):
    import yt_sub_hub.api as api_module

    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    monkeypatch.setenv("YT_SUB_HUB_QUALITY_MODE", "strict")
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.create_video_job(
        job_id="runtime-draft-job",
        video_id="abc123def45",
        canonical_url="https://www.youtube.com/watch?v=abc123def45",
        quality_mode="draft",
        acceptance_mode="full_processing",
    )
    observed = {}

    class RecordingPipeline:
        def __init__(self, *, store=None, cancel_checker=None, settings=None, **kwargs):
            observed["settings_quality_mode"] = settings.quality_mode
            observed["settings_acceptance_mode"] = settings.acceptance_mode
            observed["settings_library_dir"] = settings.library_dir
            observed["store"] = store
            observed["cancel_checker"] = cancel_checker

        def process_job(self, job_id):
            observed["job_id"] = job_id
            return JobRecord(job_id=job_id, url=job.canonical_url, video_id=job.video_id, status="completed")

    class InlineStageWorker:
        def __init__(self, runtime_arg, **kwargs):
            self.runtime = runtime_arg

        def process_job(self, job_id, processor):
            processor(self.runtime.load_video_job(job_id))
            return self.runtime.load_video_job(job_id)

    monkeypatch.setattr(api_module, "VideoProcessingPipeline", RecordingPipeline)
    monkeypatch.setattr("yt_sub_hub.worker.StageWorker", InlineStageWorker)

    api_module.process_existing_job(job.id, tmp_path)

    assert observed["settings_quality_mode"] == "draft"
    assert observed["settings_acceptance_mode"] == "full_processing"
    assert observed["settings_library_dir"] == tmp_path
    assert observed["job_id"] == job.id


def test_get_video_graph_returns_local_graph_extraction(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from yt_sub_hub.api import create_app

    video_dir = tmp_path / "videos" / "abc123"
    video_dir.mkdir(parents=True)
    (video_dir / "graph_extraction.json").write_text(
        '{"schema_version":"graph-extraction-v1","video_id":"abc123","generated_from":{},'
        '"chunks":[],"nodes":[],"relationships":[],"quality_report":{}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))

    response = TestClient(create_app()).get("/videos/abc123/graph")

    assert response.status_code == 200
    assert response.json()["schema_version"] == "graph-extraction-v1"


def test_queue_video_youtube_discovery_api_creates_stage_plan(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from yt_sub_hub.api import create_app

    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))

    response = TestClient(create_app()).post(
        "/videos/srcVideo123/discovery/youtube/queue",
        json={"limit": 7, "max_quota_units": 300},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["video_id"] == "srcVideo123"
    assert payload["job_kind"] == "youtube_discovery"
    assert payload["stage_count"] == 7
    stages = RuntimeStore(tmp_path / "index.sqlite3").list_stage_plan(payload["job_id"])
    assert stages[0].stage_name == "youtube_discovery_seed_extract"


def test_get_video_youtube_discovery_stages_api_reports_query_group(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from yt_sub_hub.api import create_app

    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    job = runtime.queue_youtube_discovery("srcVideo123", limit=7, max_quota_units=300)
    core = runtime.create_video_job(
        video_id="srcVideo123",
        canonical_url="https://www.youtube.com/watch?v=srcVideo123",
    )
    with sqlite3.connect(runtime.db_path) as conn:
        conn.execute("UPDATE video_jobs SET updated_at = ? WHERE id = ?", (9999999999, core.id))

    response = TestClient(create_app()).get("/videos/srcVideo123/discovery/youtube/stages")

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] == job.id
    assert payload["job_kind"] == "youtube_discovery"
    assert payload["chunks"]["youtube_discovery_queries"]["total"] == 0


def test_get_video_graph_returns_404_when_missing(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from yt_sub_hub.api import create_app

    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))

    response = TestClient(create_app()).get("/videos/missing/graph")

    assert response.status_code == 404


def test_graph_api_delegates_to_graph_store(monkeypatch):
    from fastapi.testclient import TestClient

    import yt_sub_hub.api as api_module
    from yt_sub_hub.api import create_app

    class FakeGraphStore:
        def __init__(self, uri, user, password):
            pass

        def close(self):
            pass

        def list_concepts(self, q=None, limit=100):
            return [{"id": "concept:graph", "name": q, "limit": limit}]

        def get_neighbors(self, node_id, depth=1, relation=None, limit=100):
            if relation == "bad":
                raise ValueError("bad relation")
            return [{"id": node_id, "depth": depth, "relation": relation, "limit": limit}]

        def find_path(self, from_id, to_id, max_depth=4):
            return [{"from": from_id, "to": to_id, "max_depth": max_depth}]

        def get_claims(self, concept=None, min_confidence=0.0, limit=100):
            return [{"concept": concept, "min_confidence": min_confidence, "limit": limit}]

    monkeypatch.setattr(api_module, "Neo4jGraphStore", FakeGraphStore)
    client = TestClient(create_app())

    assert client.get("/graph/concepts?q=graph&limit=5").json()[0]["limit"] == 5
    assert client.get("/graph/concepts/concept:graph/neighbors?depth=2&relation=MENTIONS&limit=7").json()[0]["depth"] == 2
    assert client.get("/graph/path?from=a&to=b&max_depth=3").json()[0]["max_depth"] == 3
    assert client.get("/graph/claims?concept=graph&min_confidence=0.7&limit=4").json()[0]["min_confidence"] == 0.7
    assert client.get("/graph/concepts/concept:graph/neighbors?relation=bad").status_code == 400


def test_artifact_browser_api_exposes_video_and_job_payloads(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from yt_sub_hub.api import create_app
    from yt_sub_hub.runtime import RuntimeStore

    video_dir = tmp_path / "videos" / "abc123"
    video_dir.mkdir(parents=True)
    (video_dir / "metadata.json").write_text('{"video_id":"abc123","title":"API Video"}', encoding="utf-8")
    (video_dir / "transcript.normalized.json").write_text('{"segments":[]}', encoding="utf-8")
    (video_dir / "transcript.ru.json").write_text('{"segments":[]}', encoding="utf-8")
    (video_dir / "analysis.ru.json").write_text('{"summary":"ok"}', encoding="utf-8")
    (video_dir / "note.md").write_text("# API Note\n", encoding="utf-8")
    (video_dir / "graph_extraction.json").write_text('{"nodes":[]}', encoding="utf-8")
    RuntimeStore(tmp_path / "index.sqlite3").create_video_job(
        job_id="job-api",
        video_id="abc123",
        canonical_url="https://www.youtube.com/watch?v=abc123",
        status="completed",
    )
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))

    client = TestClient(create_app())

    video_response = client.get("/videos/abc123/artifact-browser")
    job_response = client.get("/jobs/job-api/artifact-browser")

    assert video_response.status_code == 200
    assert video_response.json()["title"] == "API Video"
    assert video_response.json()["product_quality"]["mode"] == "strict"
    assert job_response.status_code == 200
    assert job_response.json()["job_id"] == "job-api"
    assert job_response.json()["quality_mode"] == "strict"


def test_youtube_discovery_api_delegates_to_service(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from yt_sub_hub.api import create_app

    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    calls = {}

    def fake_run(store, video_id, **kwargs):
        calls["run"] = (video_id, kwargs)
        return {"video_id": video_id, "status": {"status": "completed"}}

    def fake_payload(store, video_id):
        return {"video_id": video_id, "status": {"status": "completed"}}

    def fake_review(store, video_id):
        return {"video_id": video_id, "status": "passed"}

    def fake_sync(store, video_id, *, settings):
        calls["sync"] = video_id
        return {"video_id": video_id, "checked": True, "missing": 0, "extra": 0}

    def fake_enqueue(runtime, store, video_id, **kwargs):
        calls["enqueue"] = (video_id, kwargs)
        return {"queued_count": 1}

    monkeypatch.setattr("yt_sub_hub.youtube_discovery.run_youtube_discovery", fake_run)
    monkeypatch.setattr("yt_sub_hub.youtube_discovery.build_discovery_payload", fake_payload)
    monkeypatch.setattr("yt_sub_hub.youtube_discovery.load_discovery_review", fake_review)
    monkeypatch.setattr("yt_sub_hub.youtube_discovery.sync_discovery_graph_to_neo4j", fake_sync)
    monkeypatch.setattr("yt_sub_hub.youtube_discovery.enqueue_discovered_videos", fake_enqueue)
    client = TestClient(create_app())

    assert client.post("/videos/abc123/discovery/youtube/run", json={"limit": 4}).json()["status"]["status"] == "completed"
    assert client.get("/videos/abc123/discovery/youtube").json()["video_id"] == "abc123"
    assert client.get("/videos/abc123/discovery/youtube/review").json()["status"] == "passed"
    assert client.post("/videos/abc123/discovery/youtube/sync-neo4j").json()["missing"] == 0
    assert client.post("/videos/abc123/discovery/youtube/enqueue", json={"min_score": 0.5, "limit": 1}).json()["queued_count"] == 1
    assert calls["run"][1]["limit"] == 4
    assert calls["enqueue"][1]["min_score"] == 0.5


def test_artifact_content_api_returns_json_markdown_and_rejects_invalid_key(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from yt_sub_hub.api import create_app

    video_dir = tmp_path / "videos" / "abc123"
    video_dir.mkdir(parents=True)
    (video_dir / "graph_extraction.json").write_text('{"nodes":[]}', encoding="utf-8")
    (video_dir / "note.md").write_text("# API Note\n", encoding="utf-8")
    (tmp_path / "token.json").write_text('{"secret":true}', encoding="utf-8")
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    client = TestClient(create_app())

    graph = client.get("/videos/abc123/artifacts/graph_extraction")
    note = client.get("/videos/abc123/artifacts/note_md")
    invalid = client.get("/videos/abc123/artifacts/..%2F..%2Ftoken.json")

    assert graph.status_code == 200
    assert graph.json() == {"nodes": []}
    assert note.status_code == 200
    assert note.json() == {"content": "# API Note\n", "content_type": "text/markdown"}
    assert invalid.status_code == 404
    assert "secret" not in invalid.text


def test_artifact_download_api_returns_file_response(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from yt_sub_hub.api import create_app

    video_dir = tmp_path / "videos" / "abc123"
    video_dir.mkdir(parents=True)
    (video_dir / "note.md").write_text("# API Note\n", encoding="utf-8")
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))

    response = TestClient(create_app()).get("/videos/abc123/artifacts/note_md/download")

    assert response.status_code == 200
    assert response.text == "# API Note\n"
    assert response.headers["content-type"].startswith("text/markdown")
    assert 'filename="note.md"' in response.headers["content-disposition"]


def test_captions_probe_api_returns_shared_payload(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import yt_sub_hub.captions_probe as captions_probe_module
    from yt_sub_hub.api import create_app

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
        "videos": [{"video_id": "abc123def45", "transcript_source_decision": "needs_whisper_fallback"}],
    }
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    monkeypatch.setattr(captions_probe_module, "probe_captions", lambda target, limit=None, settings=None: payload)

    response = TestClient(create_app()).get("/captions-probe?target=abc123def45&limit=3")

    assert response.status_code == 200
    assert response.json() == payload


def test_captions_probe_api_invalid_target_returns_controlled_400(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import yt_sub_hub.captions_probe as captions_probe_module
    from yt_sub_hub.api import create_app
    from yt_sub_hub.captions_probe import CaptionsProbeError

    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))

    def fail_probe(target, limit=None, settings=None):
        raise CaptionsProbeError("Unsupported captions probe target: nope")

    monkeypatch.setattr(captions_probe_module, "probe_captions", fail_probe)

    response = TestClient(create_app()).get("/captions-probe?target=nope&limit=3")

    assert response.status_code == 400
    assert "Unsupported captions probe target" in response.json()["detail"]


def test_captions_probe_api_unavailable_video_returns_200_payload(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from yt_sub_hub.api import create_app

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

    response = TestClient(create_app()).get("/captions-probe?target=zzzzzzzzzzz")

    assert response.status_code == 200
    payload = response.json()
    assert payload["target_kind"] == "video"
    assert payload["summary"]["unavailable_count"] == 1
    assert payload["videos"][0]["metadata"]["status"] == "error"
    assert payload["videos"][0]["transcript_source_decision"] == "unavailable"
