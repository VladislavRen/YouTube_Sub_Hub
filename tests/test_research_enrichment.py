from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

from typer.testing import CliRunner


def _research():
    return importlib.import_module("yt_sub_hub.research_enrichment")


def test_search_results_canonicalize_and_dedupe_equivalent_urls():
    research = _research()
    results = [
        research.SearchResult(
            title="HTTPX docs",
            url="https://www.Example.com/docs/?utm_source=newsletter&id=42#install",
            snippet="Primary docs result.",
            rank=1,
        ),
        research.SearchResult(
            title="Duplicate docs",
            url="https://example.com/docs?id=42&utm_campaign=spring",
            snippet="Same canonical source.",
            rank=2,
        ),
        research.SearchResult(
            title="Release notes",
            url="https://example.com/releases?id=42",
            snippet="A distinct source.",
            rank=3,
        ),
    ]

    assert research.canonicalize_url(results[0].url) == "https://example.com/docs?id=42"
    assert research.dedupe_search_results(results) == [
        research.SearchResult(
            title="HTTPX docs",
            url="https://example.com/docs?id=42",
            snippet="Primary docs result.",
            rank=1,
        ),
        research.SearchResult(
            title="Release notes",
            url="https://example.com/releases?id=42",
            snippet="A distinct source.",
            rank=3,
        ),
    ]


def test_quote_validation_reports_exact_fuzzy_and_failed_matches():
    research = _research()
    source = research.FetchedSource(
        url="https://example.com/report",
        title="Report",
        text="The API requires every rendered claim to include a citation from fetched source text.",
    )

    exact = research.validate_quote(
        "every rendered claim to include a citation",
        source,
    )
    fuzzy = research.validate_quote(
        "every rendered claim includes citations from fetched source text",
        source,
    )
    failed = research.validate_quote(
        "the product automatically validates unrelated market forecasts",
        source,
    )

    assert exact.status == "exact"
    assert exact.source_url == "https://example.com/report"
    assert exact.confidence == 1.0
    assert fuzzy.status == "fuzzy"
    assert fuzzy.confidence >= 0.8
    assert fuzzy.source_url == "https://example.com/report"
    assert failed.status == "failed"
    assert failed.confidence < 0.8
    assert failed.source_url == "https://example.com/report"


def test_research_review_fails_hallucinated_url_or_failed_quote():
    research = _research()
    fetched_sources = [
        research.FetchedSource(
            url="https://example.com/source-a",
            title="Source A",
            text="Source A confirms that the MVP must cite fetched source text.",
        )
    ]
    report = research.ResearchReport(
        title="Research enrichment",
        claims=[
            research.ResearchClaim(
                text="The MVP must cite fetched source text.",
                citations=[
                    research.Citation(
                        url="https://example.com/source-a",
                        quote="the MVP must cite fetched source text",
                    )
                ],
            ),
            research.ResearchClaim(
                text="This claim points to a hallucinated source.",
                citations=[
                    research.Citation(
                        url="https://example.com/not-fetched",
                        quote="This quote cannot be checked against fetched sources.",
                    )
                ],
            ),
            research.ResearchClaim(
                text="This claim has a fetched URL but unsupported quote.",
                citations=[
                    research.Citation(
                        url="https://example.com/source-a",
                        quote="unrelated market forecasts are automatically validated",
                    )
                ],
            ),
        ],
    )

    review = research.review_research_enrichment(report, fetched_sources)

    assert review.status == "failed"
    assert review.ready_for_report is False
    assert {
        (issue.code, issue.severity)
        for issue in review.issues
    } >= {
        ("hallucinated_url", "hard"),
        ("quote_validation_failed", "hard"),
    }


def test_report_rendering_requires_citations_from_fetched_sources():
    research = _research()
    fetched_sources = [
        research.FetchedSource(
            url="https://example.com/source-a",
            title="Source A",
            text="The release policy requires citations from fetched source text.",
        )
    ]
    report = research.ResearchReport(
        title="Release policy",
        claims=[
            research.ResearchClaim(
                text="The release policy requires citations.",
                citations=[
                    research.Citation(
                        url="https://example.com/source-a",
                        quote="release policy requires citations",
                    )
                ],
            )
        ],
    )

    rendered = research.render_research_report(report, fetched_sources)

    assert "## Release policy" in rendered
    assert "The release policy requires citations." in rendered
    assert "[Source A](https://example.com/source-a)" in rendered
    assert "release policy requires citations" in rendered


def test_report_rendering_rejects_uncited_or_unfetched_claims():
    research = _research()
    fetched_sources = [
        research.FetchedSource(
            url="https://example.com/source-a",
            title="Source A",
            text="Fetched source text.",
        )
    ]

    uncited = research.ResearchReport(
        title="No citations",
        claims=[research.ResearchClaim(text="This claim has no citation.", citations=[])],
    )
    hallucinated = research.ResearchReport(
        title="Bad URL",
        claims=[
            research.ResearchClaim(
                text="This claim cites a URL that was not fetched.",
                citations=[
                    research.Citation(
                        url="https://example.com/not-fetched",
                        quote="Fetched source text.",
                    )
                ],
            )
        ],
    )

    for report in (uncited, hallucinated):
        try:
            research.render_research_report(report, fetched_sources)
        except research.ResearchValidationError as exc:
            assert "citation" in str(exc).lower() or "fetched" in str(exc).lower()
        else:
            raise AssertionError("render_research_report must reject reports without fetched-source citations")


def test_research_status_does_not_change_existing_product_quality_contract():
    research = _research()
    from yt_sub_hub.product_quality import evaluate_product_quality

    ready = evaluate_product_quality(
        "strict",
        quality_gate={"hard_failures": [], "warnings": []},
        ru_artifact_review={"status": "passed", "issues": []},
        stages=[
            {"stage_name": "research_enrichment", "status": research.ResearchStatus.FAILED.value, "warning": "research failed"},
        ],
    )
    warning = evaluate_product_quality(
        "strict",
        quality_gate={"hard_failures": [], "warnings": ["source-language leakage detected"]},
        ru_artifact_review={"status": "passed", "issues": []},
        stages=[
            {"stage_name": "research_enrichment", "status": research.ResearchStatus.PASSED.value},
        ],
    )

    assert ready == {
        "mode": "strict",
        "status": "ready",
        "production_ready": True,
        "graph_sync_allowed": True,
        "reasons": [],
    }
    assert warning["status"] == "warning"
    assert warning["production_ready"] is False
    assert warning["graph_sync_allowed"] is False
    assert warning["reasons"] == ["quality_gate warning: source-language leakage detected"]


def _write_ready_research_video(root, video_id="abc123def45"):
    video_dir = root / "videos" / video_id
    (video_dir / "review").mkdir(parents=True)
    (video_dir / "metadata.json").write_text(
        json.dumps(
            {
                "video_id": video_id,
                "canonical_url": f"https://www.youtube.com/watch?v={video_id}",
                "title": "Research Video",
                "channel": "Research Channel",
            }
        ),
        encoding="utf-8",
    )
    (video_dir / "transcript.ru.json").write_text(
        json.dumps({"video_id": video_id, "source": "fixture", "language": "ru", "segments": []}),
        encoding="utf-8",
    )
    (video_dir / "transcript.normalized.json").write_text(
        json.dumps({"video_id": video_id, "source": "fixture", "language": "ru", "segments": []}),
        encoding="utf-8",
    )
    (video_dir / "analysis.ru.json").write_text(
        json.dumps(
            {
                "language": "ru",
                "cleaned_title": "Research Video",
                "abstract": "Видео о проверяемых тезисах.",
                "tags": ["исследование"],
                "outline": ["Проверить тезисы"],
                "sections": [],
                "concepts": [{"name": "Цитирование", "definition": "Проверка цитат по источникам."}],
                "claims": [{"text": "MVP должен цитировать скачанный текст источника.", "evidence": "analysis"}],
                "limitations": [],
                "open_questions": ["Какие источники подтверждают тезис?"],
                "related_topics": ["качество источников"],
                "prompt_version": "fixture",
                "model": "fixture",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (video_dir / "analysis.json").write_text((video_dir / "analysis.ru.json").read_text(encoding="utf-8"), encoding="utf-8")
    (video_dir / "note.md").write_text("# Research Video\n\nГотовая заметка.\n", encoding="utf-8")
    (video_dir / "graph_extraction.json").write_text(json.dumps({"nodes": [], "relationships": []}), encoding="utf-8")
    (video_dir / "review" / "ru_artifact_review.json").write_text(json.dumps({"status": "passed", "issues": []}), encoding="utf-8")
    return video_dir


def test_research_video_cli_creates_reviewed_artifacts(tmp_path, monkeypatch):
    research = _research()
    import yt_sub_hub.cli as cli

    _write_ready_research_video(tmp_path)
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))

    class FakeProvider:
        def search(self, query, limit):
            return [
                research.SearchResult(
                    title="Source A",
                    url="https://example.com/source-a?utm_source=ignore",
                    snippet="Fixture source.",
                    rank=1,
                )
            ]

    class FakeFetcher:
        def fetch(self, result, source_id):
            return research.FetchedSource(
                source_id=source_id,
                result_ids=[result.result_id],
                url=result.url,
                title="Source A",
                text="Source A confirms that MVP должен цитировать скачанный текст источника.",
            )

    monkeypatch.setattr(research, "DuckDuckGoSearchProvider", lambda: FakeProvider())
    monkeypatch.setattr(research, "SourceFetcher", lambda *args, **kwargs: FakeFetcher())

    result = CliRunner().invoke(cli.app, ["research-video", "abc123def45"])

    assert result.exit_code == 0
    research_dir = tmp_path / "videos" / "abc123def45" / "research"
    assert (research_dir / "research_report.md").exists()
    assert json.loads((research_dir / "review" / "research_artifact_review.json").read_text(encoding="utf-8"))["status"] == "passed"
    assert json.loads((research_dir / "status.json").read_text(encoding="utf-8"))["status"] == "completed"


def test_research_status_cli_reads_separate_status(tmp_path, monkeypatch):
    import yt_sub_hub.cli as cli

    research_dir = tmp_path / "videos" / "abc123def45" / "research"
    research_dir.mkdir(parents=True)
    (research_dir / "status.json").write_text('{"video_id":"abc123def45","status":"failed"}', encoding="utf-8")
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))

    result = CliRunner().invoke(cli.app, ["research-status", "abc123def45", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.output)["status"] == "failed"


def test_research_api_exposes_report_review_and_retry(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    research = _research()
    import yt_sub_hub.research_enrichment as research_module
    from yt_sub_hub.api import create_app

    _write_ready_research_video(tmp_path)
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))

    class FakeProvider:
        def search(self, query, limit):
            return [research.SearchResult(title="Source A", url="https://example.com/source-a", snippet="Fixture", rank=1)]

    class FakeFetcher:
        def fetch(self, result, source_id):
            return research.FetchedSource(
                source_id=source_id,
                result_ids=[result.result_id],
                url=result.url,
                title="Source A",
                text="Fixture confirms that MVP должен цитировать скачанный текст источника.",
            )

    monkeypatch.setattr(research_module, "DuckDuckGoSearchProvider", lambda: FakeProvider())
    monkeypatch.setattr(research_module, "SourceFetcher", lambda *args, **kwargs: FakeFetcher())
    client = TestClient(create_app())

    assert client.post("/videos/abc123def45/research/run").status_code == 200
    assert client.get("/videos/abc123def45/research").json()["status"]["status"] == "completed"
    assert "Research Enrichment" in client.get("/videos/abc123def45/research/report").json()["markdown"]
    assert client.get("/videos/abc123def45/research/review").json()["status"] == "passed"
    assert client.post("/videos/abc123def45/research/retry").status_code == 200


def test_artifact_browser_lists_research_artifacts(tmp_path):
    from yt_sub_hub.artifact_browser import build_artifact_browser_for_video
    from yt_sub_hub.library import LibraryStore

    _write_ready_research_video(tmp_path)
    research_dir = tmp_path / "videos" / "abc123def45" / "research"
    (research_dir / "review").mkdir(parents=True)
    (research_dir / "research_report.md").write_text("# Research\n", encoding="utf-8")
    (research_dir / "findings.ru.json").write_text("[]", encoding="utf-8")
    (research_dir / "research_graph.json").write_text('{"nodes":[],"relationships":[]}', encoding="utf-8")
    (research_dir / "review" / "research_artifact_review.json").write_text('{"status":"passed"}', encoding="utf-8")

    payload = build_artifact_browser_for_video(LibraryStore(tmp_path), "abc123def45")
    keys = {artifact["key"] for artifact in payload["artifacts"]}

    assert {"research_report", "research_findings", "research_review", "research_graph"}.issubset(keys)


def test_source_fetcher_blocks_private_and_non_http_urls_without_network(monkeypatch):
    research = _research()
    unsafe_urls = [
        "http://localhost:8000/x",
        "http://127.0.0.1/x",
        "http://0.0.0.0/x",
        "http://[::1]/x",
        "http://169.254.169.254/latest/meta-data",
        "http://10.0.0.1/x",
        "http://172.16.0.1/x",
        "http://192.168.1.1/x",
        "file:///etc/passwd",
    ]
    network_calls = []

    def fail_network(*args, **kwargs):
        network_calls.append(args)
        raise AssertionError("unsafe URLs must be rejected before network I/O")

    monkeypatch.setattr(research.httpx, "get", fail_network)
    fetcher = research.SourceFetcher()

    for url in unsafe_urls:
        safe = research.validate_fetch_url(url)
        assert safe.allowed is False
        assert "/etc/passwd" not in safe.reason
        result = fetcher.fetch(research.SearchResult(title="Unsafe", url=url, snippet="", rank=1), "source-unsafe")
        assert result.fetched is False
        assert result.fetch_error
        assert "blocked" in result.fetch_error.lower() or "unsafe" in result.fetch_error.lower()

    assert network_calls == []


def test_source_fetcher_allows_public_url_with_trust_env_disabled(monkeypatch):
    research = _research()
    observed = {"stream_bytes_read": False}

    class FakeResponse:
        url = "https://example.com/article"
        status_code = 200
        headers = {"content-type": "text/html; charset=utf-8"}

        @property
        def text(self):
            raise AssertionError("SourceFetcher must read final response body via iter_bytes()")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def iter_bytes(self):
            observed["stream_bytes_read"] = True
            yield b"<html><title>Article</title><p>Public article text with citation evidence.</p></html>"

    class FakeClient:
        def __init__(self, **kwargs):
            observed.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url):
            observed["redirect_probe_url"] = url
            return FakeResponse()

        def stream(self, method, url):
            observed["method"] = method
            observed["url"] = url
            return FakeResponse()

    monkeypatch.setattr(research.httpx, "Client", FakeClient)
    monkeypatch.setattr(research.socket, "getaddrinfo", lambda host, port, type=0: [(None, None, None, None, ("93.184.216.34", 443))])

    source = research.SourceFetcher().fetch(
        research.SearchResult(title="Article", url="https://example.com/article", snippet="", rank=1),
        "source-001",
    )

    assert observed["trust_env"] is False
    assert observed["follow_redirects"] is False
    assert observed.get("method") == "GET"
    assert observed.get("url") == "https://example.com/article"
    assert observed["stream_bytes_read"] is True
    assert source.fetched is True
    assert source.title == "Article"


def test_source_fetcher_rejects_redirect_to_private_url(monkeypatch):
    research = _research()
    requested_urls = []
    body_read = False

    class FakeResponse:
        url = "https://example.com/redirect"
        status_code = 302
        headers = {"location": "http://127.0.0.1/private"}
        text = ""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def iter_bytes(self):
            nonlocal body_read
            body_read = True
            raise AssertionError("redirect target must be blocked before body read")

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url):
            raise AssertionError("SourceFetcher must inspect redirects without non-streaming body reads")

        def stream(self, method, url):
            assert method == "GET"
            requested_urls.append(url)
            return FakeResponse()

    monkeypatch.setattr(research.httpx, "Client", FakeClient)
    monkeypatch.setattr(research.socket, "getaddrinfo", lambda host, port, type=0: [(None, None, None, None, ("93.184.216.34", 443))])

    source = research.SourceFetcher().fetch(
        research.SearchResult(title="Redirect", url="https://example.com/redirect", snippet="", rank=1),
        "source-redirect",
    )

    assert source.fetched is False
    assert source.fetch_error
    assert "redirect" in source.fetch_error.lower() or "final url" in source.fetch_error.lower()
    assert requested_urls == ["https://example.com/redirect"]
    assert body_read is False


def test_source_fetcher_enforces_max_content_chars(monkeypatch):
    research = _research()

    class FakeResponse:
        url = "https://example.com/large"
        status_code = 200
        headers = {"content-type": "text/plain"}

        @property
        def text(self):
            raise AssertionError("SourceFetcher must read final response body via iter_bytes()")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def iter_bytes(self):
            yield b"0123456789" * 20

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url):
            return FakeResponse()

        def stream(self, method, url):
            assert method == "GET"
            assert url == "https://example.com/large"
            return FakeResponse()

    monkeypatch.setattr(research.httpx, "Client", FakeClient)
    monkeypatch.setattr(research.socket, "getaddrinfo", lambda host, port, type=0: [(None, None, None, None, ("93.184.216.34", 443))])

    source = research.SourceFetcher(max_content_chars=50).fetch(
        research.SearchResult(title="Large", url="https://example.com/large", snippet="", rank=1),
        "source-large",
    )

    assert source.fetched is True
    assert len(source.text) == 50


def test_source_fetcher_stops_streaming_after_max_content_chars(monkeypatch):
    research = _research()
    chunks = [b"abcdefghij"] * 30
    observed = {"chunks_requested": 0, "bytes_requested": 0}

    class FakeResponse:
        url = "https://example.com/huge-text"
        status_code = 200
        headers = {"content-type": "text/plain"}

        @property
        def text(self):
            raise AssertionError("SourceFetcher must read final response body via iter_bytes()")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def iter_bytes(self):
            for chunk in chunks:
                if observed["bytes_requested"] >= 200:
                    raise AssertionError("SourceFetcher requested a stream chunk beyond the byte cap")
                observed["chunks_requested"] += 1
                observed["bytes_requested"] += len(chunk)
                yield chunk

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url):
            return FakeResponse()

        def stream(self, method, url):
            assert method == "GET"
            assert url == "https://example.com/huge-text"
            return FakeResponse()

    monkeypatch.setattr(research.httpx, "Client", FakeClient)
    monkeypatch.setattr(research.socket, "getaddrinfo", lambda host, port, type=0: [(None, None, None, None, ("93.184.216.34", 443))])

    source = research.SourceFetcher(max_content_chars=50).fetch(
        research.SearchResult(title="Huge", url="https://example.com/huge-text", snippet="", rank=1),
        "source-huge",
    )

    assert source.fetched is True
    assert source.truncated is True
    assert len(source.text) <= 50
    assert observed["chunks_requested"] < len(chunks)
    assert observed["bytes_requested"] < sum(len(chunk) for chunk in chunks)


def test_source_fetcher_rejects_unsupported_content_type_before_body_read(monkeypatch):
    research = _research()
    observed = {"iter_bytes_called": False}

    class FakeResponse:
        url = "https://example.com/report.pdf"
        status_code = 200
        headers = {"content-type": "application/pdf"}

        @property
        def text(self):
            raise AssertionError("SourceFetcher must reject unsupported content-type before reading body")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def iter_bytes(self):
            observed["iter_bytes_called"] = True
            raise AssertionError("unsupported content-type body must not be read")

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url):
            return FakeResponse()

        def stream(self, method, url):
            assert method == "GET"
            assert url == "https://example.com/report.pdf"
            return FakeResponse()

    monkeypatch.setattr(research.httpx, "Client", FakeClient)
    monkeypatch.setattr(research.socket, "getaddrinfo", lambda host, port, type=0: [(None, None, None, None, ("93.184.216.34", 443))])

    source = research.SourceFetcher().fetch(
        research.SearchResult(title="PDF", url="https://example.com/report.pdf", snippet="", rank=1),
        "source-pdf",
    )

    assert source.fetched is False
    assert source.fetch_error
    assert "unsupported content-type" in source.fetch_error.lower()
    assert observed["iter_bytes_called"] is False


def _write_research_sync_artifacts(root, *, video_id="abc123def45", review_status="passed", run_id="run-current"):
    research_dir = root / "videos" / video_id / "research"
    (research_dir / "review").mkdir(parents=True)
    graph = {
        "schema_version": "research-enrichment-v1",
        "video_id": video_id,
        "research_run_id": run_id,
        "nodes": [
            {"id": f"research_finding:{run_id}:1", "label": "ResearchFinding", "source_video_id": video_id, "research_run_id": run_id}
        ],
        "relationships": [
            {
                "id": f"research-rel-{run_id}",
                "from": f"video:{video_id}",
                "to": f"research_finding:{run_id}:1",
                "type": "RELATED_EXTERNAL",
                "source_video_id": video_id,
                "source_url": "https://example.com/source-a",
                "provider": "duckduckgo",
                "research_run_id": run_id,
                "confidence": 0.9,
                "evidence_quote": "validated evidence quote",
                "quote_validation": "exact",
                "language": "ru",
            }
        ],
    }
    (research_dir / "research_graph.json").write_text(json.dumps(graph), encoding="utf-8")
    (research_dir / "review" / "research_artifact_review.json").write_text(
        json.dumps({"status": review_status, "issues": []}),
        encoding="utf-8",
    )
    return graph


def test_research_sync_neo4j_cli_cleans_previous_runs_and_prints_consistency(tmp_path, monkeypatch):
    research = _research()
    import yt_sub_hub.cli as cli

    graph = _write_research_sync_artifacts(tmp_path)
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))
    calls = []

    class FakeGraphStore:
        def __init__(self, uri, user, password):
            calls.append(("init", uri))

        def delete_research_for_video(self, video_id):
            calls.append(("delete_video", video_id))

        def upsert_payload(self, payload):
            calls.append(("upsert", [node["id"] for node in payload.nodes], [rel["id"] for rel in payload.relationships]))

        def research_consistency(self, payload):
            calls.append(("consistency", len(payload.nodes), len(payload.relationships)))
            return {"checked": True, "missing": 0, "extra": 0}

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr(research, "Neo4jGraphStore", FakeGraphStore)

    result = CliRunner().invoke(cli.app, ["research-sync-neo4j", "abc123def45", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload == {
        "video_id": "abc123def45",
        "checked": True,
        "missing": 0,
        "extra": 0,
        "node_count": len(graph["nodes"]),
        "relationship_count": len(graph["relationships"]),
    }
    assert ("delete_video", "abc123def45") in calls
    assert any(call[0] == "upsert" for call in calls)


def test_research_sync_neo4j_cli_blocks_failed_review_before_upsert(tmp_path, monkeypatch):
    research = _research()
    import yt_sub_hub.cli as cli

    _write_research_sync_artifacts(tmp_path, review_status="failed")
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))

    class FailGraphStore:
        def __init__(self, *args, **kwargs):
            raise AssertionError("failed review must block Neo4j construction")

    monkeypatch.setattr(research, "Neo4jGraphStore", FailGraphStore)

    result = CliRunner().invoke(cli.app, ["research-sync-neo4j", "abc123def45", "--json"])

    assert result.exit_code != 0
    assert "passed" in result.output.lower() or "review" in result.output.lower()


def test_research_sync_neo4j_api_returns_consistency_payload(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    research = _research()
    import yt_sub_hub.research_enrichment as research_module
    from yt_sub_hub.api import create_app

    _write_research_sync_artifacts(tmp_path)
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))

    class FakeGraphStore:
        def __init__(self, uri, user, password):
            pass

        def delete_research_for_video(self, video_id):
            self.video_id = video_id

        def upsert_payload(self, payload):
            self.payload = payload

        def research_consistency(self, payload):
            return {"checked": True, "missing": 0, "extra": 0}

        def close(self):
            pass

    monkeypatch.setattr(research_module, "Neo4jGraphStore", FakeGraphStore)

    response = TestClient(create_app()).post("/videos/abc123def45/research/sync-neo4j")

    assert response.status_code == 200
    assert response.json() == {
        "video_id": "abc123def45",
        "checked": True,
        "missing": 0,
        "extra": 0,
        "node_count": 1,
        "relationship_count": 1,
    }


def test_research_sync_neo4j_api_blocks_failed_review(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    _write_research_sync_artifacts(tmp_path, review_status="failed")
    monkeypatch.setenv("YT_SUB_HUB_LIBRARY_DIR", str(tmp_path))

    response = TestClient(__import__("yt_sub_hub.api", fromlist=["create_app"]).create_app()).post(
        "/videos/abc123def45/research/sync-neo4j"
    )

    assert response.status_code == 409


def test_research_sync_rejects_graph_relationship_without_valid_provenance(tmp_path):
    research = _research()
    from yt_sub_hub.library import LibraryStore

    graph = _write_research_sync_artifacts(tmp_path)
    graph["relationships"][0]["quote_validation"] = "failed"
    (tmp_path / "videos" / "abc123def45" / "research" / "research_graph.json").write_text(json.dumps(graph), encoding="utf-8")

    try:
        research.sync_research_graph_to_neo4j(
            LibraryStore(tmp_path),
            "abc123def45",
            settings=SimpleNamespace(neo4j_uri="bolt://unused", neo4j_user="neo4j", neo4j_password="password"),
        )
    except research.ResearchValidationError as exc:
        assert "quote_validation" in str(exc)
    else:
        raise AssertionError("sync must reject research graph relationships with failed quote validation")
