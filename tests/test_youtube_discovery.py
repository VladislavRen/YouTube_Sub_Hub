from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner
from yt_sub_hub.research_enrichment import _research_artifact_hashes, text_hash


def _write_core_and_research(root: Path, video_id: str = "srcVideo123") -> None:
    video_dir = root / "videos" / video_id
    research_dir = video_dir / "research"
    (research_dir / "review").mkdir(parents=True)
    (video_dir / "review").mkdir(parents=True)
    (video_dir / "metadata.json").write_text(
        json.dumps(
            {
                "video_id": video_id,
                "title": "Graph research lecture",
                "canonical_url": f"https://www.youtube.com/watch?v={video_id}",
            }
        ),
        encoding="utf-8",
    )
    (video_dir / "analysis.ru.json").write_text(
        json.dumps(
            {
                "concepts": [
                    {"name": "Graph databases", "definition": "Neo4j graph sync and claims."},
                    {"name": "Research enrichment", "definition": "External source review."},
                ],
                "claims": [
                    {"text": "Graph discovery should preserve provenance.", "evidence": "analysis"},
                ],
                "related_topics": ["knowledge graph"],
                "tags": ["Neo4j", "YouTube"],
            }
        ),
        encoding="utf-8",
    )
    (video_dir / "note.md").write_text("# Graph research lecture\n\nKnowledge graph discovery.\n", encoding="utf-8")
    (video_dir / "review" / "ru_artifact_review.json").write_text(json.dumps({"status": "passed"}), encoding="utf-8")
    (research_dir / "research_plan.json").write_text(
        json.dumps({"schema_version": "research-enrichment-v1", "research_run_id": "rr-1", "video_id": video_id,
                    "source_note_hash": text_hash((video_dir / "note.md").read_text()),
                    "source_analysis_hash": text_hash(json.dumps(json.loads((video_dir / "analysis.ru.json").read_text()), ensure_ascii=False, sort_keys=True))}),
        encoding="utf-8",
    )
    (research_dir / "findings.ru.json").write_text(
        json.dumps(
            [
                {
                    "finding_id": "finding-1",
                    "accepted": True,
                    "claim_ru": "Neo4j graph sync requires explicit provenance.",
                    "source_ids": ["source-001"],
                    "citations": [
                        {
                            "source_id": "source-001",
                            "url": "https://example.com/neo4j-provenance",
                            "quote": "Neo4j graph sync requires explicit provenance.",
                            "quote_validation": "exact",
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    (research_dir / "source_summaries.json").write_text(
        json.dumps(
            [
                {
                    "source_id": "source-001",
                    "summary_ru": "Neo4j provenance and graph review.",
                    "key_points": ["graph provenance", "review gates"],
                    "relevant_query_ids": ["rq-001"],
                }
            ]
        ),
        encoding="utf-8",
    )
    (research_dir / "queries.json").write_text(
        json.dumps([{"query_id": "rq-001", "query": "Neo4j provenance", "language": "ru"}]),
        encoding="utf-8",
    )
    (research_dir / "search_results.normalized.json").write_text(
        json.dumps(
            [
                {
                    "result_id": "sr-1",
                    "canonical_url": "https://example.com/neo4j-provenance",
                    "title": "Neo4j provenance",
                    "query_id": "rq-001",
                }
            ]
        ),
        encoding="utf-8",
    )
    (research_dir / "research_report.md").write_text("# Research\n\nNeo4j provenance.\n", encoding="utf-8")
    (research_dir / "research_graph.json").write_text(
        json.dumps(
            {
                "video_id": video_id,
                "research_run_id": "rr-1",
                "nodes": [{"id": "research_finding:rr-1:finding-1", "label": "ResearchFinding"}],
                "relationships": [],
            }
        ),
        encoding="utf-8",
    )
    (research_dir / "review" / "research_artifact_review.json").write_text(
        json.dumps({"status": "passed", "research_run_id": "rr-1",
                    "artifact_hashes": _research_artifact_hashes(research_dir)}),
        encoding="utf-8",
    )
    (research_dir / "status.json").write_text(json.dumps({"status": "completed", "research_run_id": "rr-1"}))


class FakeDiscoveryProvider:
    def search_videos(self, query: str, limit: int):
        return [
            {
                "youtube_video_id": "vidAlpha001",
                "title": "Neo4j provenance tutorial",
                "description": "Graph provenance and discovery review.",
                "channel_id": "UCgraphAlpha",
                "channel_title": "Graph Lab",
                "published_at": "2026-01-02T00:00:00Z",
                "raw": {"id": {"videoId": "vidAlpha001"}},
            },
            {
                "youtube_video_id": "srcVideo123",
                "title": "Self video",
                "description": "Should be excluded.",
                "channel_id": "UCself",
                "channel_title": "Self",
                "raw": {"id": {"videoId": "srcVideo123"}},
            },
        ][:limit]

    def search_channels(self, query: str, limit: int):
        return [
            {
                "youtube_channel_id": "UCgraphAlpha",
                "title": "Graph Lab",
                "description": "Neo4j graph systems.",
                "raw": {"id": {"channelId": "UCgraphAlpha"}},
            }
        ][:limit]

    def get_video_details(self, video_ids):
        return {
            "vidAlpha001": {
                "youtube_video_id": "vidAlpha001",
                "title": "Neo4j provenance tutorial",
                "description": "Graph provenance and discovery review.",
                "channel_id": "UCgraphAlpha",
                "channel_title": "Graph Lab",
                "published_at": "2026-01-02T00:00:00Z",
                "duration": "PT10M",
                "view_count": 1200,
                "language": "en",
                "tags": ["neo4j", "graph", "provenance"],
                "thumbnails": {"default": {"url": "https://i.ytimg.com/vi/vidAlpha001/default.jpg"}},
            }
        }

    def get_channel_details(self, channel_ids):
        return {
            "UCgraphAlpha": {
                "youtube_channel_id": "UCgraphAlpha",
                "title": "Graph Lab",
                "description": "Neo4j graph systems.",
                "published_at": "2024-01-01T00:00:00Z",
                "subscriber_count": 10000,
                "video_count": 100,
                "view_count": 500000,
                "language": "en",
                "thumbnails": {},
            }
        }


def _plan_for_seed(seed_text: str, *, max_queries: int = 20, seed_id: str = "seed-001"):
    from yt_sub_hub.youtube_discovery import _plan_queries

    seeds = {
        "input_hashes": {"fixture": "hash-1"},
        "items": [
            {
                "seed_id": seed_id,
                "kind": "concept",
                "text": seed_text,
                "confidence": 0.9,
                "selected": True,
            }
        ],
    }
    return _plan_queries("srcVideo123", "rr-1", "dr-1", seeds, max_queries=max_queries)


def test_plan_queries_marks_russian_seed_as_ru():
    queries = _plan_for_seed("процедурная генерация уровней")

    assert queries["schema_version"] == "youtube-discovery-v1"
    assert queries["artifact_type"] == "queries"
    assert queries["items"]
    assert all(item["language"] == "ru" for item in queries["items"] if item["rejected_reason"] is None)
    assert any("процедурная" in item["query"] for item in queries["items"])
    assert not [
        item
        for item in queries["items"]
        if item["language"] == "en" and any("а" <= char.lower() <= "я" or char == "ё" for char in item["query"])
    ]


def test_plan_queries_extracts_latin_entity_for_en_variant():
    queries = _plan_for_seed("Игра Mortal Shell 2 в выпуске")
    executable = [item for item in queries["items"] if item["rejected_reason"] is None]

    assert any(item["language"] == "ru" for item in executable)
    en_queries = [item for item in executable if item["language"] == "en" and "Mortal Shell 2" in item["query"]]
    assert en_queries
    assert {item["query_source"] for item in en_queries} <= {"latin_entity", "alias_map"}


def test_plan_queries_marks_english_seed_as_en():
    queries = _plan_for_seed("graph provenance review")

    assert queries["items"]
    assert all(item["language"] == "en" for item in queries["items"] if item["rejected_reason"] is None)
    assert any(item["query"] == "graph provenance review" for item in queries["items"])


def test_plan_queries_does_not_label_cyrillic_query_as_en_without_alias():
    queries = _plan_for_seed("сложная русская фраза без латинских сущностей")

    assert not [
        item
        for item in queries["items"]
        if item["language"] == "en" and any("а" <= char.lower() <= "я" or char == "ё" for char in item["query"])
    ]


def test_plan_queries_uses_language_appropriate_suffixes():
    queries = _plan_for_seed("Игра Mortal Shell 2 в выпуске", max_queries=20)
    executable = [item for item in queries["items"] if item["rejected_reason"] is None]

    assert any(item["language"] == "ru" and item["intent"] == "deep_dive" and "разбор" in item["query"] for item in executable)
    assert any(item["language"] == "ru" and item["intent"] == "channel_discovery" and "канал" in item["query"] for item in executable)
    assert any(item["language"] == "ru" and item["intent"] == "counterpoint" and "критика" in item["query"] for item in executable)
    assert any(item["language"] == "en" and item["intent"] == "deep_dive" and item["query"].endswith("explained") for item in executable)
    assert any(item["language"] == "en" and item["intent"] == "channel_discovery" and item["query"].endswith("channel") for item in executable)
    assert any(item["language"] == "en" and item["intent"] == "counterpoint" and item["query"].endswith("critique") for item in executable)


def test_plan_queries_includes_query_artifact_contract_fields():
    queries = _plan_for_seed("Игра Artificial Detective в выпуске")

    for item in queries["items"]:
        assert {
            "query_id",
            "seed_ids",
            "intent",
            "language",
            "query",
            "priority",
            "dedupe_key",
            "query_source",
            "rejected_reason",
            "quota_estimate",
        } <= set(item)
        assert item["quota_estimate"]["total_units"] > 0


def test_run_youtube_discovery_writes_artifacts_and_does_not_enqueue(tmp_path):
    from yt_sub_hub.library import LibraryStore
    from yt_sub_hub.runtime import RuntimeStore
    from yt_sub_hub.youtube_discovery import build_discovery_payload, run_youtube_discovery

    _write_core_and_research(tmp_path)
    runtime = RuntimeStore(tmp_path / "index.sqlite3")

    payload = run_youtube_discovery(LibraryStore(tmp_path), "srcVideo123", provider=FakeDiscoveryProvider(), limit=5)

    assert payload["status"]["status"] == "completed_with_warning"
    discovery_dir = tmp_path / "videos" / "srcVideo123" / "youtube_discovery"
    assert (discovery_dir / "seeds.json").exists()
    assert (discovery_dir / "queries.json").exists()
    video_candidates = json.loads((discovery_dir / "video_candidates.json").read_text(encoding="utf-8"))
    channel_candidates = json.loads((discovery_dir / "channel_candidates.json").read_text(encoding="utf-8"))
    review = json.loads((discovery_dir / "review.json").read_text(encoding="utf-8"))
    assert video_candidates["items"][0]["youtube_video_id"] == "vidAlpha001"
    assert video_candidates["items"][0]["canonical_url"] == "https://www.youtube.com/watch?v=vidAlpha001"
    assert all(item["youtube_video_id"] != "srcVideo123" for item in video_candidates["items"])
    assert channel_candidates["items"][0]["youtube_channel_id"] == "UCgraphAlpha"
    assert review["accepted_candidate_ids"]
    assert runtime.list_video_jobs() == []
    assert build_discovery_payload(LibraryStore(tmp_path), "srcVideo123")["artifacts"]["review"]["exists"] is True


def test_run_youtube_discovery_skips_rejected_queries(tmp_path, monkeypatch):
    from yt_sub_hub.library import LibraryStore
    from yt_sub_hub.youtube_discovery import run_youtube_discovery

    class RecordingProvider(FakeDiscoveryProvider):
        def __init__(self):
            self.video_search_queries = []
            self.channel_search_queries = []

        def search_videos(self, query: str, limit: int):
            self.video_search_queries.append(query)
            return super().search_videos(query, limit)

        def search_channels(self, query: str, limit: int):
            self.channel_search_queries.append(query)
            return super().search_channels(query, limit)

    def fake_expand_queries(store, video_id, *, max_quota_units=900):
        discovery_dir = store.video_dir(video_id) / "youtube_discovery"
        seeds = json.loads((discovery_dir / "seeds.json").read_text(encoding="utf-8"))
        payload = {
            "schema_version": "youtube-discovery-v1",
            "artifact_type": "queries",
            "video_id": video_id,
            "research_run_id": seeds["research_run_id"],
            "discovery_run_id": seeds["discovery_run_id"],
            "input_hashes": seeds.get("input_hashes") or {},
            "artifact_hash": "queries-with-rejected",
            "items": [
                {
                    "query_id": "ydq-active",
                    "seed_ids": ["seed-001"],
                    "intent": "same_topic",
                    "language": "en",
                    "query": "graph provenance",
                    "priority": 1,
                    "dedupe_key": "graph provenance:same_topic:en",
                    "query_source": "seed_text",
                    "quota_estimate": {"total_units": 202},
                    "rejected_reason": None,
                },
                {
                    "query_id": "ydq-rejected",
                    "seed_ids": ["seed-001"],
                    "intent": "same_topic",
                    "language": "en",
                    "query": "rejected query text",
                    "priority": 2,
                    "dedupe_key": "rejected query text:same_topic:en",
                    "query_source": "seed_text",
                    "quota_estimate": {"total_units": 202},
                    "rejected_reason": "query_language_mismatch",
                },
            ],
        }
        (discovery_dir / "queries.json").write_text(json.dumps(payload), encoding="utf-8")
        return payload

    _write_core_and_research(tmp_path)
    provider = RecordingProvider()
    monkeypatch.setattr("yt_sub_hub.youtube_discovery.expand_discovery_queries_stage", fake_expand_queries)

    run_youtube_discovery(LibraryStore(tmp_path), "srcVideo123", provider=provider, limit=5)

    assert "graph provenance" in provider.video_search_queries
    assert "graph provenance" in provider.channel_search_queries
    assert "rejected query text" not in provider.video_search_queries
    assert "rejected query text" not in provider.channel_search_queries


def test_discovery_requires_passed_research_review(tmp_path):
    from yt_sub_hub.library import LibraryStore
    from yt_sub_hub.youtube_discovery import DiscoveryValidationError, run_youtube_discovery

    _write_core_and_research(tmp_path)
    review_path = tmp_path / "videos" / "srcVideo123" / "research" / "review" / "research_artifact_review.json"
    review_path.write_text(json.dumps({"status": "failed"}), encoding="utf-8")

    try:
        run_youtube_discovery(LibraryStore(tmp_path), "srcVideo123", provider=FakeDiscoveryProvider())
    except DiscoveryValidationError as exc:
        assert "research_artifact_review status passed" in str(exc)
    else:
        raise AssertionError("discovery must require passed research review")


def test_review_blocks_candidates_without_verified_ids_or_provenance():
    from yt_sub_hub.youtube_discovery import review_youtube_discovery

    review = review_youtube_discovery(
        video_id="srcVideo123",
        research_run_id="rr-1",
        discovery_run_id="dr-1",
        seeds={"items": [{"seed_id": "seed-1"}]},
        queries={"items": [{"query_id": "ydq-001"}]},
        video_candidates={
            "items": [
                {"candidate_id": "bad-1", "canonical_url": "https://www.youtube.com/watch?v=madeUp", "query_ids": []}
            ]
        },
        channel_candidates={"items": []},
    )

    assert review["status"] == "failed"
    assert {issue["code"] for issue in review["issues"]} >= {"missing_verified_youtube_id", "missing_provenance"}


def test_review_rejects_video_canonical_url_mismatch():
    from yt_sub_hub.youtube_discovery import review_youtube_discovery

    review = review_youtube_discovery(
        video_id="srcVideo123",
        research_run_id="rr-1",
        discovery_run_id="dr-1",
        seeds={"items": [{"seed_id": "seed-1"}]},
        queries={"items": [{"query_id": "ydq-001"}]},
        video_candidates={
            "items": [
                {
                    "candidate_id": "video-bad-url",
                    "kind": "video",
                    "youtube_video_id": "vidAlpha001",
                    "canonical_url": "https://www.youtube.com/watch?v=OTHER999",
                    "query_ids": ["ydq-001"],
                    "seed_ids": ["seed-1"],
                    "discovery_score": 0.9,
                }
            ]
        },
        channel_candidates={"items": []},
    )

    assert review["status"] == "failed"
    assert "canonical_url_mismatch" in {issue["code"] for issue in review["issues"]}
    assert "video-bad-url" not in review["accepted_candidate_ids"]


def test_review_rejects_channel_canonical_url_mismatch():
    from yt_sub_hub.youtube_discovery import review_youtube_discovery

    review = review_youtube_discovery(
        video_id="srcVideo123",
        research_run_id="rr-1",
        discovery_run_id="dr-1",
        seeds={"items": [{"seed_id": "seed-1"}]},
        queries={"items": [{"query_id": "ydq-001"}]},
        video_candidates={"items": []},
        channel_candidates={
            "items": [
                {
                    "candidate_id": "channel-bad-url",
                    "kind": "channel",
                    "youtube_channel_id": "UCgood",
                    "canonical_url": "https://www.youtube.com/channel/UCbad",
                    "query_ids": ["ydq-001"],
                    "seed_ids": ["seed-1"],
                    "discovery_score": 0.9,
                }
            ]
        },
    )

    assert review["status"] == "failed"
    assert "canonical_url_mismatch" in {issue["code"] for issue in review["issues"]}
    assert "channel-bad-url" not in review["accepted_candidate_ids"]


def test_review_short_acceptance_uses_requested_limit_and_budgeted_query_coverage():
    from yt_sub_hub.youtube_discovery import review_youtube_discovery

    seeds = {
        "items": [
            {"seed_id": "seed-001", "selected": True},
            {"seed_id": "seed-002", "selected": True},
            {"seed_id": "seed-003", "selected": True},
        ]
    }
    queries = {
        "max_queries": 2,
        "items": [
            {"query_id": "ydq-001", "seed_ids": ["seed-001"], "rejected_reason": None},
            {"query_id": "ydq-002", "seed_ids": ["seed-002"], "rejected_reason": None},
        ],
    }
    video_candidates = {
        "requested_limit": 5,
        "items": [
            {
                "candidate_id": f"video-{index}",
                "kind": "video",
                "youtube_video_id": f"vidGood{index:03d}",
                "canonical_url": f"https://www.youtube.com/watch?v=vidGood{index:03d}",
                "query_ids": ["ydq-001"],
                "seed_ids": ["seed-001"],
                "discovery_score": 0.75,
            }
            for index in range(5)
        ],
    }
    channel_candidates = {
        "requested_limit": 5,
        "items": [
            {
                "candidate_id": f"channel-{index}",
                "kind": "channel",
                "youtube_channel_id": f"UCgood{index}",
                "canonical_url": f"https://www.youtube.com/channel/UCgood{index}",
                "query_ids": ["ydq-002"],
                "seed_ids": ["seed-002"],
                "discovery_score": 0.45,
            }
            for index in range(5)
        ],
    }

    review = review_youtube_discovery(
        video_id="srcVideo123",
        research_run_id="rr-1",
        discovery_run_id="dr-1",
        seeds=seeds,
        queries=queries,
        video_candidates=video_candidates,
        channel_candidates=channel_candidates,
    )

    assert review["status"] == "passed"
    assert review["issues"] == []
    assert review["completeness"]["minimum_video_candidates"] == 5
    assert review["completeness"]["minimum_channel_candidates"] == 3
    assert review["completeness"]["query_budget_exhausted"] is True


def test_enqueue_discovered_videos_requires_review_and_is_idempotent(tmp_path):
    from yt_sub_hub.library import LibraryStore
    from yt_sub_hub.runtime import RuntimeStore
    from yt_sub_hub.youtube_discovery import enqueue_discovered_videos, run_youtube_discovery

    _write_core_and_research(tmp_path)
    store = LibraryStore(tmp_path)
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    run_youtube_discovery(store, "srcVideo123", provider=FakeDiscoveryProvider(), limit=5)

    first = enqueue_discovered_videos(runtime, store, "srcVideo123", min_score=0.1, limit=5)
    second = enqueue_discovered_videos(runtime, store, "srcVideo123", min_score=0.1, limit=5)

    assert first["queued_count"] == 1
    assert second["queued_count"] == 1
    assert len(runtime.list_video_jobs()) == 1
    assert runtime.list_video_jobs()[0].job_kind == "video_processing"
    assert (tmp_path / "videos" / "srcVideo123" / "youtube_discovery" / "enqueue_audit.json").exists()


def test_enqueue_discovered_videos_rechecks_canonical_url_for_old_review(tmp_path):
    from yt_sub_hub.library import LibraryStore
    from yt_sub_hub.runtime import RuntimeStore
    from yt_sub_hub.youtube_discovery import DiscoveryValidationError, enqueue_discovered_videos, run_youtube_discovery

    _write_core_and_research(tmp_path)
    store = LibraryStore(tmp_path)
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    run_youtube_discovery(store, "srcVideo123", provider=FakeDiscoveryProvider(), limit=5)
    candidates_path = tmp_path / "videos" / "srcVideo123" / "youtube_discovery" / "video_candidates.json"
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    candidates["items"][0]["canonical_url"] = "https://www.youtube.com/watch?v=OTHER999"
    candidates_path.write_text(json.dumps(candidates), encoding="utf-8")

    try:
        enqueue_discovered_videos(runtime, store, "srcVideo123", min_score=0.1, limit=5)
    except DiscoveryValidationError as exc:
        assert "canonical URL" in str(exc)
    else:
        raise AssertionError("enqueue must reject stale review artifacts with mismatched canonical URL")


def test_enqueue_discovered_videos_validates_all_candidates_before_creating_jobs(tmp_path):
    from yt_sub_hub.library import LibraryStore
    from yt_sub_hub.runtime import RuntimeStore
    from yt_sub_hub.youtube_discovery import DiscoveryValidationError, enqueue_discovered_videos, run_youtube_discovery

    _write_core_and_research(tmp_path)
    store = LibraryStore(tmp_path)
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    run_youtube_discovery(store, "srcVideo123", provider=FakeDiscoveryProvider(), limit=5)
    discovery_dir = tmp_path / "videos" / "srcVideo123" / "youtube_discovery"
    candidates_path = discovery_dir / "video_candidates.json"
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    bad_candidate = {
        **candidates["items"][0],
        "candidate_id": "video-bad-url",
        "youtube_video_id": "vidBeta002",
        "canonical_url": "https://www.youtube.com/watch?v=OTHER999",
    }
    candidates["items"].append(bad_candidate)
    candidates_path.write_text(json.dumps(candidates), encoding="utf-8")
    review_path = discovery_dir / "review.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["accepted_candidate_ids"].append("video-bad-url")
    review_path.write_text(json.dumps(review), encoding="utf-8")

    try:
        enqueue_discovered_videos(runtime, store, "srcVideo123", min_score=0.1, limit=5)
    except DiscoveryValidationError as exc:
        assert "canonical URL" in str(exc)
    else:
        raise AssertionError("enqueue must reject stale review artifacts with mismatched canonical URL")
    assert runtime.list_video_jobs() == []


def test_enqueue_discovered_channel_rechecks_canonical_url_for_old_review(tmp_path, monkeypatch):
    from yt_sub_hub.library import LibraryStore
    from yt_sub_hub.runtime import RuntimeStore
    from yt_sub_hub.youtube_discovery import DiscoveryValidationError, enqueue_discovered_channel, run_youtube_discovery

    _write_core_and_research(tmp_path)
    store = LibraryStore(tmp_path)
    runtime = RuntimeStore(tmp_path / "index.sqlite3")
    run_youtube_discovery(store, "srcVideo123", provider=FakeDiscoveryProvider(), limit=5)
    candidates_path = tmp_path / "videos" / "srcVideo123" / "youtube_discovery" / "channel_candidates.json"
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    channel_id = candidates["items"][0]["youtube_channel_id"]
    candidates["items"][0]["canonical_url"] = "https://www.youtube.com/channel/UCbad"
    candidates_path.write_text(json.dumps(candidates), encoding="utf-8")

    def fail_fetch_channel_listing(*args, **kwargs):
        raise AssertionError("canonical URL guard must run before channel listing fetch")

    monkeypatch.setattr("yt_sub_hub.youtube_discovery.fetch_channel_listing", fail_fetch_channel_listing)

    try:
        enqueue_discovered_channel(runtime, store, "srcVideo123", channel_id=channel_id, limit=5)
    except DiscoveryValidationError as exc:
        assert "canonical URL" in str(exc)
    else:
        raise AssertionError("channel enqueue must reject stale review artifacts with mismatched canonical URL")


def test_sync_discovery_graph_to_neo4j_requires_passed_review(tmp_path, monkeypatch):
    from yt_sub_hub.library import LibraryStore
    from yt_sub_hub.youtube_discovery import run_youtube_discovery, sync_discovery_graph_to_neo4j

    _write_core_and_research(tmp_path)
    run_youtube_discovery(LibraryStore(tmp_path), "srcVideo123", provider=FakeDiscoveryProvider(), limit=5)
    calls = {}

    class FakeGraphStore:
        def __init__(self, uri, user, password):
            calls["init"] = (uri, user, password)

        def delete_discovery_for_video(self, video_id):
            calls["deleted"] = video_id

        def upsert_payload(self, payload):
            calls["payload"] = payload

        def discovery_consistency(self, payload):
            return {"checked": True, "missing": 0, "extra": 0}

        def close(self):
            calls["closed"] = True

    monkeypatch.setattr("yt_sub_hub.youtube_discovery.Neo4jGraphStore", FakeGraphStore)

    result = sync_discovery_graph_to_neo4j(
        LibraryStore(tmp_path),
        "srcVideo123",
        settings=type("Settings", (), {"neo4j_uri": "bolt://test", "neo4j_user": "neo4j", "neo4j_password": "pw"})(),
    )

    assert result["missing"] == 0
    assert result["extra"] == 0
    assert calls["deleted"] == "srcVideo123"


def test_discovery_graph_sync_rejects_candidate_url_mismatch(tmp_path, monkeypatch):
    from yt_sub_hub.library import LibraryStore
    from yt_sub_hub.youtube_discovery import DiscoveryValidationError, run_youtube_discovery, sync_discovery_graph_to_neo4j

    _write_core_and_research(tmp_path)
    run_youtube_discovery(LibraryStore(tmp_path), "srcVideo123", provider=FakeDiscoveryProvider(), limit=5)
    graph_path = tmp_path / "videos" / "srcVideo123" / "youtube_discovery" / "graph.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    for node in graph["nodes"]:
        if node.get("label") == "YouTubeVideoCandidate":
            node["url"] = "https://www.youtube.com/watch?v=OTHER999"
            break
    graph_path.write_text(json.dumps(graph), encoding="utf-8")

    class FailGraphStore:
        def __init__(self, *args, **kwargs):
            raise AssertionError("invalid discovery graph must block Neo4j construction")

    monkeypatch.setattr("yt_sub_hub.youtube_discovery.Neo4jGraphStore", FailGraphStore)

    try:
        sync_discovery_graph_to_neo4j(
            LibraryStore(tmp_path),
            "srcVideo123",
            settings=type("Settings", (), {"neo4j_uri": "bolt://test", "neo4j_user": "neo4j", "neo4j_password": "pw"})(),
        )
    except DiscoveryValidationError as exc:
        assert "canonical URL" in str(exc)
    else:
        raise AssertionError("graph sync must reject candidate URL mismatch before Neo4j writes")


def test_discovery_graph_sync_rejects_candidate_relationship_id_mismatch(tmp_path, monkeypatch):
    from yt_sub_hub.library import LibraryStore
    from yt_sub_hub.youtube_discovery import DiscoveryValidationError, run_youtube_discovery, sync_discovery_graph_to_neo4j

    _write_core_and_research(tmp_path)
    run_youtube_discovery(LibraryStore(tmp_path), "srcVideo123", provider=FakeDiscoveryProvider(), limit=5)
    graph_path = tmp_path / "videos" / "srcVideo123" / "youtube_discovery" / "graph.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    for relationship in graph["relationships"]:
        if relationship.get("type") == "RELATED_TO_VIDEO":
            relationship["youtube_video_id"] = "OTHER999"
            break
    graph_path.write_text(json.dumps(graph), encoding="utf-8")

    class FailGraphStore:
        def __init__(self, *args, **kwargs):
            raise AssertionError("invalid discovery graph must block Neo4j construction")

    monkeypatch.setattr("yt_sub_hub.youtube_discovery.Neo4jGraphStore", FailGraphStore)

    try:
        sync_discovery_graph_to_neo4j(
            LibraryStore(tmp_path),
            "srcVideo123",
            settings=type("Settings", (), {"neo4j_uri": "bolt://test", "neo4j_user": "neo4j", "neo4j_password": "pw"})(),
        )
    except DiscoveryValidationError as exc:
        assert "relationship video ID" in str(exc)
    else:
        raise AssertionError("graph sync must reject relationship ID mismatch before Neo4j writes")


def test_discovery_graph_sync_rejects_candidate_relationship_missing_id(tmp_path, monkeypatch):
    from yt_sub_hub.library import LibraryStore
    from yt_sub_hub.youtube_discovery import DiscoveryValidationError, run_youtube_discovery, sync_discovery_graph_to_neo4j

    _write_core_and_research(tmp_path)
    run_youtube_discovery(LibraryStore(tmp_path), "srcVideo123", provider=FakeDiscoveryProvider(), limit=5)
    graph_path = tmp_path / "videos" / "srcVideo123" / "youtube_discovery" / "graph.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    for relationship in graph["relationships"]:
        if relationship.get("type") == "RELATED_TO_VIDEO":
            relationship.pop("youtube_video_id", None)
            break
    graph_path.write_text(json.dumps(graph), encoding="utf-8")

    class FailGraphStore:
        def __init__(self, *args, **kwargs):
            raise AssertionError("invalid discovery graph must block Neo4j construction")

    monkeypatch.setattr("yt_sub_hub.youtube_discovery.Neo4jGraphStore", FailGraphStore)

    try:
        sync_discovery_graph_to_neo4j(
            LibraryStore(tmp_path),
            "srcVideo123",
            settings=type("Settings", (), {"neo4j_uri": "bolt://test", "neo4j_user": "neo4j", "neo4j_password": "pw"})(),
        )
    except DiscoveryValidationError as exc:
        assert "relationship video ID" in str(exc)
    else:
        raise AssertionError("graph sync must reject missing relationship ID before Neo4j writes")


def test_discovery_rejects_failed_research_retry_and_mismatched_run(tmp_path):
    import pytest
    from yt_sub_hub.library import LibraryStore
    from yt_sub_hub.youtube_discovery import _require_research_ready, DiscoveryValidationError
    _write_core_and_research(tmp_path)
    root = tmp_path / 'videos/srcVideo123/research'
    (root / 'status.json').write_text(json.dumps({'status': 'failed', 'research_run_id': 'rr-1'}))
    with pytest.raises(DiscoveryValidationError, match='completed'):
        _require_research_ready(LibraryStore(tmp_path), 'srcVideo123')
    (root / 'status.json').write_text(json.dumps({'status': 'completed', 'research_run_id': 'new-run'}))
    with pytest.raises(DiscoveryValidationError, match='run'):
        _require_research_ready(LibraryStore(tmp_path), 'srcVideo123')


def test_search_intent_does_not_determine_semantic_relationship():
    from yt_sub_hub.youtube_discovery import _relationship_for_query
    for intent in ['counterpoint', 'deep_dive', 'recent_updates', 'channel_discovery', 'verification']:
        assert _relationship_for_query({'intent': intent}) == 'unclear'
