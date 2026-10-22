from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from contextlib import closing
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from yt_sub_hub.channel import fetch_channel_listing
from yt_sub_hub.graph import GraphPayload, Neo4jGraphStore
from yt_sub_hub.library import LibraryStore
from yt_sub_hub.runtime import RuntimeStore
from yt_sub_hub.youtube_data_api import build_youtube_client, google_api_error_class, sanitize_google_api_message

SCHEMA_VERSION = "youtube-discovery-v1"
SEARCH_LIST_QUOTA_UNITS = 1
DETAILS_LIST_QUOTA_UNITS = 1
DEFAULT_MAX_SEARCH_CALLS = 100
VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,32}$")
CHANNEL_ID_RE = re.compile(r"^[A-Za-z0-9_-]{3,64}$")
QUERY_INTENTS = ("same_topic", "counterpoint", "deep_dive", "source_author", "channel_discovery", "recent_updates")


class DiscoveryValidationError(ValueError):
    pass


class DiscoveryProviderError(RuntimeError):
    pass


class YouTubeDiscoveryProvider(Protocol):
    def search_videos(self, query: str, limit: int) -> list[dict[str, Any]]: ...

    def search_channels(self, query: str, limit: int) -> list[dict[str, Any]]: ...

    def get_video_details(self, video_ids: list[str]) -> dict[str, dict[str, Any]]: ...

    def get_channel_details(self, channel_ids: list[str]) -> dict[str, dict[str, Any]]: ...


@dataclass
class QuotaBudget:
    """Local run limits, never the project's remaining daily Google quota.

    Persistent reservations are committed before requests and survive retries or
    process exits. A crash before sending can conservatively overcount an attempt.
    The same discovery_run_id retains usage; its limits can only be tightened.
    """

    max_units: int = 900
    used_units: int = 0
    max_search_calls: int = DEFAULT_MAX_SEARCH_CALLS
    used_search_calls: int = 0
    ledger_path: Path | None = None
    run_id: str | None = None
    _lock: Any = field(default_factory=Lock, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("max_units", "used_units", "max_search_calls", "used_search_calls"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise DiscoveryValidationError(f"{name} must be a non-negative integer")
        if self.ledger_path is None:
            return
        if not self.run_id:
            raise DiscoveryValidationError("discovery_run_id is required for a persistent quota budget")
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.ledger_path, timeout=30)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "CREATE TABLE IF NOT EXISTS discovery_quota ("
                "run_id TEXT PRIMARY KEY, max_units INTEGER NOT NULL, used_units INTEGER NOT NULL, "
                "max_search_calls INTEGER NOT NULL, used_search_calls INTEGER NOT NULL)"
            )
            db.execute(
                "INSERT OR IGNORE INTO discovery_quota VALUES (?, ?, ?, ?, ?)",
                (self.run_id, self.max_units, self.used_units, self.max_search_calls, self.used_search_calls),
            )
            db.execute(
                "UPDATE discovery_quota SET max_units = MIN(max_units, ?), "
                "max_search_calls = MIN(max_search_calls, ?) WHERE run_id = ?",
                (self.max_units, self.max_search_calls, self.run_id),
            )
            self._load_usage(db)

    def _load_usage(self, db: sqlite3.Connection) -> None:
        row = db.execute(
            "SELECT max_units, used_units, max_search_calls, used_search_calls FROM discovery_quota WHERE run_id = ?",
            (self.run_id,),
        ).fetchone()
        if row is None:
            raise DiscoveryValidationError("discovery run quota ledger is missing")
        self.max_units, self.used_units, self.max_search_calls, self.used_search_calls = row

    def consume(self, units: int, action: str, *, bucket: str = "general_data") -> None:
        if not isinstance(units, int) or units < 0 or bucket not in {"general_data", "search"}:
            raise DiscoveryValidationError("quota consumption requires non-negative units and a known bucket")
        with self._lock:
            if self.ledger_path is None:
                self._consume(units, action, bucket)
                return
            with closing(sqlite3.connect(self.ledger_path, timeout=30)) as db, db:
                db.execute("BEGIN IMMEDIATE")
                self._load_usage(db)
                self._consume(units, action, bucket)
                db.execute(
                    "UPDATE discovery_quota SET used_units = ?, used_search_calls = ? WHERE run_id = ?",
                    (self.used_units, self.used_search_calls, self.run_id),
                )

    def _consume(self, units: int, action: str, bucket: str) -> None:
        used, maximum = (self.used_search_calls, self.max_search_calls) if bucket == "search" else (self.used_units, self.max_units)
        if used + units > maximum:
            raise DiscoveryValidationError(
                f"YouTube Data API {bucket} run quota budget exceeded before {action}: {used + units}>{maximum}"
            )
        if bucket == "search":
            self.used_search_calls += units
        else:
            self.used_units += units

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            if self.ledger_path is not None:
                with closing(sqlite3.connect(self.ledger_path, timeout=30)) as db:
                    self._load_usage(db)
            return {
                "scope": "discovery_run",
                "accounting": "local_request_reservations_including_failures",
                "discovery_run_id": self.run_id,
                "general_units_bucket": "general_data",
                "max_units": self.max_units,
                "used_units": self.used_units,
                "remaining_units": max(0, self.max_units - self.used_units),
                "max_search_calls": self.max_search_calls,
                "used_search_calls": self.used_search_calls,
                "remaining_search_calls": max(0, self.max_search_calls - self.used_search_calls),
                "project_daily_remaining": None,
                "project_quota_authority": "Google Cloud Console",
            }


class YouTubeDataApiDiscoveryProvider:
    def __init__(self, settings: Any, *, youtube_service: Any | None = None, quota_budget: QuotaBudget | None = None) -> None:
        self.settings = settings
        self.youtube = youtube_service or build_youtube_client(settings, persist_refresh=True)
        self.quota_budget = quota_budget or QuotaBudget()

    def search_videos(self, query: str, limit: int) -> list[dict[str, Any]]:
        self.quota_budget.consume(SEARCH_LIST_QUOTA_UNITS, "search.list videos", bucket="search")
        response = self._execute(
            self.youtube.search().list(
                part="snippet",
                q=query,
                type="video",
                maxResults=max(1, min(50, int(limit))),
            ),
            "search.list videos",
        )
        results = []
        for item in response.get("items") or []:
            video_id = ((item.get("id") or {}).get("videoId") or "").strip()
            snippet = item.get("snippet") or {}
            if not video_id:
                continue
            results.append(
                {
                    "youtube_video_id": video_id,
                    "title": snippet.get("title"),
                    "description": snippet.get("description"),
                    "channel_id": snippet.get("channelId"),
                    "channel_title": snippet.get("channelTitle"),
                    "published_at": snippet.get("publishedAt"),
                    "raw": item,
                }
            )
        return results

    def search_channels(self, query: str, limit: int) -> list[dict[str, Any]]:
        self.quota_budget.consume(SEARCH_LIST_QUOTA_UNITS, "search.list channels", bucket="search")
        response = self._execute(
            self.youtube.search().list(
                part="snippet",
                q=query,
                type="channel",
                maxResults=max(1, min(50, int(limit))),
            ),
            "search.list channels",
        )
        results = []
        for item in response.get("items") or []:
            channel_id = ((item.get("id") or {}).get("channelId") or "").strip()
            snippet = item.get("snippet") or {}
            if not channel_id:
                continue
            results.append(
                {
                    "youtube_channel_id": channel_id,
                    "title": snippet.get("title"),
                    "description": snippet.get("description"),
                    "published_at": snippet.get("publishedAt"),
                    "raw": item,
                }
            )
        return results

    def get_video_details(self, video_ids: list[str]) -> dict[str, dict[str, Any]]:
        ids = _dedupe([video_id for video_id in video_ids if _valid_video_id(video_id)])
        if not ids:
            return {}
        self.quota_budget.consume(DETAILS_LIST_QUOTA_UNITS, "videos.list")
        response = self._execute(
            self.youtube.videos().list(part="snippet,contentDetails,statistics,status", id=",".join(ids)),
            "videos.list",
        )
        details: dict[str, dict[str, Any]] = {}
        for item in response.get("items") or []:
            video_id = str(item.get("id") or "")
            snippet = item.get("snippet") or {}
            statistics = item.get("statistics") or {}
            content_details = item.get("contentDetails") or {}
            details[video_id] = {
                "youtube_video_id": video_id,
                "title": snippet.get("title"),
                "description": snippet.get("description"),
                "channel_id": snippet.get("channelId"),
                "channel_title": snippet.get("channelTitle"),
                "published_at": snippet.get("publishedAt"),
                "duration": content_details.get("duration"),
                "view_count": _int_or_none(statistics.get("viewCount")),
                "language": snippet.get("defaultAudioLanguage") or snippet.get("defaultLanguage"),
                "tags": snippet.get("tags") or [],
                "thumbnails": snippet.get("thumbnails") or {},
                "raw": item,
            }
        return details

    def get_channel_details(self, channel_ids: list[str]) -> dict[str, dict[str, Any]]:
        ids = _dedupe([channel_id for channel_id in channel_ids if _valid_channel_id(channel_id)])
        if not ids:
            return {}
        self.quota_budget.consume(DETAILS_LIST_QUOTA_UNITS, "channels.list")
        response = self._execute(
            self.youtube.channels().list(part="snippet,statistics,contentDetails", id=",".join(ids)),
            "channels.list",
        )
        details: dict[str, dict[str, Any]] = {}
        for item in response.get("items") or []:
            channel_id = str(item.get("id") or "")
            snippet = item.get("snippet") or {}
            statistics = item.get("statistics") or {}
            details[channel_id] = {
                "youtube_channel_id": channel_id,
                "title": snippet.get("title"),
                "description": snippet.get("description"),
                "published_at": snippet.get("publishedAt"),
                "subscriber_count": _int_or_none(statistics.get("subscriberCount")),
                "video_count": _int_or_none(statistics.get("videoCount")),
                "view_count": _int_or_none(statistics.get("viewCount")),
                "language": snippet.get("defaultLanguage"),
                "thumbnails": snippet.get("thumbnails") or {},
                "raw": item,
            }
        return details

    def _execute(self, request: Any, action: str) -> dict[str, Any]:
        try:
            payload = request.execute()
        except Exception as exc:
            raise DiscoveryProviderError(
                f"YouTube Data API {action} failed ({google_api_error_class(exc)}): {sanitize_google_api_message(exc)}"
            ) from exc
        return payload if isinstance(payload, dict) else {}


def stage_payload_hash(payload: dict[str, Any] | None) -> str:
    return _hash_json(payload or {})


def discovery_dir_for(store: LibraryStore, video_id: str) -> Path:
    return store.video_dir(video_id) / "youtube_discovery"


def _discovery_run_quota(discovery_dir: Path, run_id: str, max_units: int, max_search_calls: int) -> QuotaBudget:
    return QuotaBudget(
        max_units=max_units,
        max_search_calls=max_search_calls,
        ledger_path=discovery_dir / "quota.sqlite3",
        run_id=run_id,
    )


def discovery_status_payload(store: LibraryStore, video_id: str) -> dict[str, Any]:
    path = discovery_dir_for(store, video_id) / "status.json"
    return _read_json(path) if path.exists() else {"video_id": video_id, "status": "not_started"}


def read_discovery_stage_options(stage_payload_json: str | None) -> dict[str, Any]:
    payload = json.loads(stage_payload_json) if stage_payload_json else {}
    return {
        "limit": int(payload.get("limit") or 20),
        "max_quota_units": int(payload.get("max_quota_units", 900)),
    }


def extract_discovery_seeds_stage(store: LibraryStore, video_id: str, *, limit: int = 20, max_quota_units: int = 900) -> dict[str, Any]:
    discovery_dir = discovery_dir_for(store, video_id)
    discovery_dir.mkdir(parents=True, exist_ok=True)
    research_run_id = _require_research_ready(store, video_id)
    input_hashes = _input_hashes(store, video_id)
    cached_path = discovery_dir / "seeds.json"
    if cached_path.exists():
        cached = _read_json(cached_path)
        if cached.get("input_hashes") == input_hashes:
            _write_discovery_status(
                discovery_dir,
                video_id,
                status="running",
                research_run_id=research_run_id,
                discovery_run_id=str(cached.get("discovery_run_id") or ""),
                limit=limit,
                max_quota_units=max_quota_units,
            )
            return cached
    discovery_run_id = _stable_id("discovery-run", video_id, research_run_id, _hash_json(input_hashes), length=18)
    seeds = _extract_seeds(store, video_id, research_run_id, discovery_run_id, input_hashes)
    _write_json(cached_path, seeds)
    _write_discovery_status(
        discovery_dir,
        video_id,
        status="running",
        research_run_id=research_run_id,
        discovery_run_id=discovery_run_id,
        limit=limit,
        max_quota_units=max_quota_units,
    )
    return seeds


def expand_discovery_queries_stage(
    store: LibraryStore,
    video_id: str,
    *,
    max_quota_units: int = 900,
    max_search_calls: int = DEFAULT_MAX_SEARCH_CALLS,
) -> dict[str, Any]:
    discovery_dir = discovery_dir_for(store, video_id)
    seeds = _read_json(discovery_dir / "seeds.json")
    research_run_id = str(seeds["research_run_id"])
    discovery_run_id = str(seeds["discovery_run_id"])
    quota = _discovery_run_quota(discovery_dir, discovery_run_id, max_quota_units, max_search_calls)
    max_queries = min(quota.max_units // _estimated_query_quota_units(), quota.max_search_calls // 2)
    queries = _plan_queries(video_id, research_run_id, discovery_run_id, seeds, max_queries=max_queries)
    queries["quota_budget"] = quota.to_dict()
    _write_json(discovery_dir / "queries.json", queries)
    return queries


def execute_discovery_query_stage(
    store: LibraryStore,
    video_id: str,
    query: dict[str, Any],
    *,
    provider: YouTubeDiscoveryProvider | None = None,
    settings: Any | None = None,
    max_quota_units: int = 900,
    max_search_calls: int = DEFAULT_MAX_SEARCH_CALLS,
    limit: int = 20,
) -> dict[str, Any]:
    discovery_dir = discovery_dir_for(store, video_id)
    query_results_dir = discovery_dir / "query_results"
    query_results_dir.mkdir(parents=True, exist_ok=True)
    seeds = _read_json(discovery_dir / "seeds.json")
    quota = _discovery_run_quota(discovery_dir, str(seeds.get("discovery_run_id") or ""), max_quota_units, max_search_calls)
    if provider is None:
        provider = YouTubeDataApiDiscoveryProvider(settings or _settings(), quota_budget=quota)
    elif isinstance(provider, YouTubeDataApiDiscoveryProvider):
        # A provider can be shared by concurrently running stages. Never replace
        # another stage's budget on that shared instance.
        provider = copy(provider)
        provider.quota_budget = quota
    query_id = str(query.get("query_id") or "")
    per_query_limit = max(1, min(10, int(limit)))
    started_at = utc_now()
    try:
        video_results = provider.search_videos(str(query.get("query") or ""), per_query_limit)
        channel_results = provider.search_channels(str(query.get("query") or ""), min(5, per_query_limit))
        video_details = provider.get_video_details([str(item.get("youtube_video_id") or "") for item in video_results])
        channel_details = provider.get_channel_details(
            _dedupe(
                [
                    *[str(item.get("youtube_channel_id") or "") for item in channel_results],
                    *[str(item.get("channel_id") or "") for item in video_results],
                ]
            )
        )
    except Exception as exc:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "query_result",
            "video_id": video_id,
            "query_id": query_id,
            "query": query,
            "status": "failed",
            "error_class": exc.__class__.__name__,
            "error_message": str(exc),
            "quota_usage": _quota_usage(provider, quota, max_quota_units=max_quota_units),
            "started_at": started_at,
            "updated_at": utc_now(),
        }
        _write_json(query_results_dir / f"{query_id}.json", payload)
        raise
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "query_result",
        "video_id": video_id,
        "query_id": query_id,
        "query": query,
        "status": "completed",
        "raw_results": [
            {"query_id": query_id, "kind": "video", "items": video_results},
            {"query_id": query_id, "kind": "channel", "items": channel_results},
        ],
        "video_results": video_results,
        "channel_results": channel_results,
        "video_details": video_details,
        "channel_details": channel_details,
        "quota_usage": _quota_usage(provider, quota, max_quota_units=max_quota_units),
        "started_at": started_at,
        "updated_at": utc_now(),
    }
    _write_json(query_results_dir / f"{query_id}.json", payload)
    return payload


def merge_discovery_query_results_stage(store: LibraryStore, video_id: str, query_stage_statuses: dict[str, str] | None = None) -> dict[str, Any]:
    discovery_dir = discovery_dir_for(store, video_id)
    queries = _read_json(discovery_dir / "queries.json")
    results_by_query: dict[str, dict[str, Any]] = {}
    missing = []
    active_queries = _active_query_items(queries)
    for query in active_queries:
        query_id = str(query.get("query_id") or "")
        status = (query_stage_statuses or {}).get(query_id)
        path = discovery_dir / "query_results" / f"{query_id}.json"
        if status == "skipped":
            continue
        if not path.exists():
            missing.append(query_id)
            continue
        result = _read_json(path)
        if result.get("status") != "completed":
            missing.append(query_id)
            continue
        results_by_query[query_id] = result
    if missing:
        raise DiscoveryValidationError(f"missing active query results: {', '.join(missing)}")
    raw_results = []
    for query_id in [str(item.get("query_id") or "") for item in active_queries]:
        result = results_by_query.get(query_id)
        if result is not None:
            raw_results.extend(result.get("raw_results") or [])
    seeds = _read_json(discovery_dir / "seeds.json")
    payload = _artifact_root(
        "search_results",
        video_id,
        str(seeds["research_run_id"]),
        str(seeds["discovery_run_id"]),
        seeds.get("input_hashes") or {},
        raw_results,
    )
    _write_json(discovery_dir / "search_results.raw.json", payload)
    _write_json(
        discovery_dir / "candidate_pool.json",
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "candidate_pool",
            "video_id": video_id,
            "research_run_id": seeds["research_run_id"],
            "discovery_run_id": seeds["discovery_run_id"],
            "input_hashes": seeds.get("input_hashes") or {},
            "query_result_ids": list(results_by_query),
            "items": list(results_by_query.values()),
            "created_at": utc_now(),
            "updated_at": utc_now(),
        },
    )
    return payload


def score_discovery_candidates_stage(store: LibraryStore, video_id: str, *, limit: int = 20) -> tuple[dict[str, Any], dict[str, Any]]:
    discovery_dir = discovery_dir_for(store, video_id)
    seeds = _read_json(discovery_dir / "seeds.json")
    pool = _read_json(discovery_dir / "candidate_pool.json")
    video_candidates_by_id: dict[str, dict[str, Any]] = {}
    channel_candidates_by_id: dict[str, dict[str, Any]] = {}
    for result in pool.get("items") or []:
        query = result.get("query") or {}
        _merge_video_candidates(video_candidates_by_id, result.get("video_results") or [], result.get("video_details") or {}, query, seeds, video_id)
        _merge_channel_candidates(channel_candidates_by_id, result.get("channel_results") or [], result.get("channel_details") or {}, query, seeds)
    research_run_id = str(seeds["research_run_id"])
    discovery_run_id = str(seeds["discovery_run_id"])
    input_hashes = seeds.get("input_hashes") or {}
    video_candidates = _candidate_collection(
        "video_candidates",
        video_id,
        research_run_id,
        discovery_run_id,
        input_hashes,
        sorted(video_candidates_by_id.values(), key=lambda item: (-float(item["discovery_score"]), item["candidate_id"]))[: int(limit)],
    )
    video_candidates["requested_limit"] = int(limit)
    channel_candidates = _candidate_collection(
        "channel_candidates",
        video_id,
        research_run_id,
        discovery_run_id,
        input_hashes,
        sorted(channel_candidates_by_id.values(), key=lambda item: (-float(item["discovery_score"]), item["candidate_id"]))[: max(3, min(10, int(limit)))],
    )
    channel_candidates["requested_limit"] = max(3, min(10, int(limit)))
    _write_json(discovery_dir / "video_candidates.json", video_candidates)
    _write_json(discovery_dir / "channel_candidates.json", channel_candidates)
    return video_candidates, channel_candidates


def review_discovery_candidates_stage(store: LibraryStore, video_id: str) -> dict[str, Any]:
    discovery_dir = discovery_dir_for(store, video_id)
    seeds = _read_json(discovery_dir / "seeds.json")
    queries = _read_json(discovery_dir / "queries.json")
    video_candidates = _read_json(discovery_dir / "video_candidates.json")
    channel_candidates = _read_json(discovery_dir / "channel_candidates.json")
    review = review_youtube_discovery(
        video_id=video_id,
        research_run_id=str(seeds["research_run_id"]),
        discovery_run_id=str(seeds["discovery_run_id"]),
        seeds=seeds,
        queries=queries,
        video_candidates=video_candidates,
        channel_candidates=channel_candidates,
    )
    _write_json(discovery_dir / "video_candidates.json", video_candidates)
    _write_json(discovery_dir / "channel_candidates.json", channel_candidates)
    _write_json(discovery_dir / "review.json", review)
    if review.get("status") == "failed":
        _write_discovery_status_from_artifacts(store, video_id, status="failed", review=review)
        raise DiscoveryValidationError("youtube discovery review failed")
    _write_discovery_status_from_artifacts(store, video_id, status="completed_with_warning" if review.get("status") == "warning" else "running", review=review)
    return review


def build_discovery_graph_stage(store: LibraryStore, video_id: str) -> dict[str, Any]:
    discovery_dir = discovery_dir_for(store, video_id)
    seeds = _read_json(discovery_dir / "seeds.json")
    queries = _read_json(discovery_dir / "queries.json")
    video_candidates = _read_json(discovery_dir / "video_candidates.json")
    channel_candidates = _read_json(discovery_dir / "channel_candidates.json")
    graph = build_discovery_graph(
        video_id,
        str(seeds["research_run_id"]),
        str(seeds["discovery_run_id"]),
        seeds,
        queries,
        video_candidates,
        channel_candidates,
    )
    _write_json(discovery_dir / "graph.json", graph)
    return graph


def render_discovery_report_stage(store: LibraryStore, video_id: str) -> str:
    discovery_dir = discovery_dir_for(store, video_id)
    review = _read_json(discovery_dir / "review.json")
    video_candidates = _read_json(discovery_dir / "video_candidates.json")
    channel_candidates = _read_json(discovery_dir / "channel_candidates.json")
    report = render_discovery_report(video_id, review, video_candidates, channel_candidates)
    (discovery_dir / "discovery_report.md").write_text(report, encoding="utf-8")
    final_status = "completed" if review.get("status") == "passed" else "completed_with_warning"
    _write_discovery_status_from_artifacts(store, video_id, status=final_status, review=review)
    return report


def run_youtube_discovery(
    store: LibraryStore,
    video_id: str,
    *,
    provider: YouTubeDiscoveryProvider | None = None,
    settings: Any | None = None,
    limit: int = 20,
    max_quota_units: int = 900,
) -> dict[str, Any]:
    discovery_dir = discovery_dir_for(store, video_id)
    discovery_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    _write_json(discovery_dir / "status.json", {"video_id": video_id, "status": "running", "updated_at": started_at})
    try:
        seeds = extract_discovery_seeds_stage(store, video_id, limit=limit, max_quota_units=max_quota_units)
        queries = expand_discovery_queries_stage(store, video_id, max_quota_units=max_quota_units)
        for query in _active_query_items(queries):
            execute_discovery_query_stage(
                store,
                video_id,
                query,
                provider=provider,
                settings=settings,
                limit=limit,
                max_quota_units=max_quota_units,
            )
        merge_discovery_query_results_stage(store, video_id)
        score_discovery_candidates_stage(store, video_id, limit=limit)
        review_discovery_candidates_stage(store, video_id)
        build_discovery_graph_stage(store, video_id)
        render_discovery_report_stage(store, video_id)
        return build_discovery_payload(store, video_id)
    except Exception as exc:
        _write_json(discovery_dir / "status.json", {"video_id": video_id, "status": "failed", "error": str(exc), "updated_at": utc_now()})
        raise


def build_discovery_payload(store: LibraryStore, video_id: str) -> dict[str, Any]:
    root = discovery_dir_for(store, video_id)
    status_path = root / "status.json"
    status = _read_json(status_path) if status_path.exists() else {"video_id": video_id, "status": "not_started"}
    return {
        "video_id": video_id,
        "status": status,
        "artifacts": {
            "seeds": _artifact_info(root / "seeds.json", store.root),
            "queries": _artifact_info(root / "queries.json", store.root),
            "video_candidates": _artifact_info(root / "video_candidates.json", store.root),
            "channel_candidates": _artifact_info(root / "channel_candidates.json", store.root),
            "review": _artifact_info(root / "review.json", store.root),
            "graph": _artifact_info(root / "graph.json", store.root),
            "report": _artifact_info(root / "discovery_report.md", store.root),
        },
    }


def load_discovery_review(store: LibraryStore, video_id: str) -> dict[str, Any]:
    path = discovery_dir_for(store, video_id) / "review.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return _read_json(path)


def review_youtube_discovery(
    *,
    video_id: str,
    research_run_id: str,
    discovery_run_id: str,
    seeds: dict[str, Any],
    queries: dict[str, Any],
    video_candidates: dict[str, Any],
    channel_candidates: dict[str, Any],
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    query_ids = {str(query.get("query_id")) for query in queries.get("items") or []}
    seed_ids = {str(seed.get("seed_id")) for seed in seeds.get("items") or []}
    accepted: list[str] = []
    rejected: list[str] = []
    needs_review: list[str] = []
    for item in [*(video_candidates.get("items") or []), *(channel_candidates.get("items") or [])]:
        candidate_id = str(item.get("candidate_id") or "")
        kind = str(item.get("kind") or "")
        verified_id = item.get("youtube_video_id") if kind == "video" else item.get("youtube_channel_id")
        candidate_issues: list[dict[str, Any]] = []
        item_queries = {str(value) for value in item.get("query_ids") or [] if value}
        item_seeds = {str(value) for value in item.get("seed_ids") or [] if value}
        if kind not in {"video", "channel"}:
            candidate_issues.append(_issue("hard", "missing_verified_youtube_id", candidate_id, "candidate requires verified YouTube id and kind"))
        if kind == "video" and not _valid_video_id(str(verified_id or "")):
            candidate_issues.append(_issue("hard", "missing_verified_youtube_id", candidate_id, "video candidate requires verified YouTube video id"))
        if kind == "channel" and not _valid_channel_id(str(verified_id or "")):
            candidate_issues.append(_issue("hard", "missing_verified_youtube_id", candidate_id, "channel candidate requires verified YouTube channel id"))
        if not item_queries or not item_queries.issubset(query_ids) or not item_seeds or not item_seeds.issubset(seed_ids):
            candidate_issues.append(_issue("hard", "missing_provenance", candidate_id, "candidate requires seed/query provenance"))
        if kind == "video" and not _canonical_url_matches_verified_id(item):
            candidate_issues.append(
                _issue(
                    "hard",
                    "canonical_url_mismatch",
                    candidate_id,
                    "video canonical URL must match verified YouTube video ID",
                )
            )
        if kind == "channel" and not _canonical_url_matches_verified_id(item):
            candidate_issues.append(
                _issue(
                    "hard",
                    "canonical_url_mismatch",
                    candidate_id,
                    "channel canonical URL must match verified YouTube channel ID",
                )
            )
        issues.extend(candidate_issues)
        candidate_has_hard_issue = any(issue["severity"] == "hard" for issue in candidate_issues)
        if candidate_has_hard_issue:
            rejected.append(candidate_id)
            item["review_status"] = "rejected"
        elif float(item.get("discovery_score") or 0.0) >= 0.5 and verified_id and item_queries and item_seeds:
            accepted.append(candidate_id)
            item["review_status"] = "accepted"
        elif float(item.get("discovery_score") or 0.0) >= 0.35:
            needs_review.append(candidate_id)
            item["review_status"] = "needs_review"
        else:
            rejected.append(candidate_id)
            item["review_status"] = "rejected"
    video_candidate_count = len(video_candidates.get("items") or [])
    channel_candidate_count = len(channel_candidates.get("items") or [])
    minimum_video_candidates = _minimum_candidate_target(video_candidates, default=10)
    minimum_channel_candidates = _minimum_candidate_target(channel_candidates, default=3)
    video_target_met = video_candidate_count >= minimum_video_candidates
    channel_target_met = channel_candidate_count >= minimum_channel_candidates
    candidate_targets_met = video_target_met and channel_target_met
    if not video_target_met:
        issues.append(
            _issue(
                "warning",
                "min_video_candidates_not_met",
                "",
                f"fewer than {minimum_video_candidates} video candidates produced",
            )
        )
    if not channel_target_met:
        issues.append(
            _issue(
                "warning",
                "min_channel_candidates_not_met",
                "",
                f"fewer than {minimum_channel_candidates} channel candidates produced",
            )
        )
    active_queries = _active_query_items(queries)
    max_queries = _int_or_none(queries.get("max_queries"))
    query_budget_exhausted = max_queries is not None and len(active_queries) >= max(0, max_queries)
    active_seed_ids = {str(seed_id) for query in active_queries for seed_id in query.get("seed_ids") or []}
    for seed in seeds.get("items") or []:
        seed_id = str(seed.get("seed_id") or "")
        if seed.get("selected") and seed_id and seed_id not in active_seed_ids:
            if not (candidate_targets_met and query_budget_exhausted):
                issues.append(
                    _issue(
                        "warning",
                        "selected_seed_without_executable_query",
                        seed_id,
                        "selected seed produced no executable discovery query",
                    )
                )
    bilingual_capable = [
        seed
        for seed in seeds.get("items") or []
        if seed.get("selected")
        and _russian_aliases_for_seed(str(seed.get("text") or ""))
        and _english_aliases_for_seed(str(seed.get("text") or ""))
    ]
    active_languages = {str(query.get("language") or "") for query in active_queries}
    if bilingual_capable and not {"ru", "en"}.issubset(active_languages):
        issues.append(_issue("warning", "bilingual_query_coverage_low", "", "bilingual-capable seeds did not produce both RU and EN executable queries"))
    hard = [issue for issue in issues if issue["severity"] == "hard"]
    status = "failed" if hard else ("warning" if issues else "passed")
    return {
        "schema_version": SCHEMA_VERSION,
        "video_id": video_id,
        "research_run_id": research_run_id,
        "discovery_run_id": discovery_run_id,
        "status": status,
        "issues": issues,
        "accepted_candidate_ids": accepted,
        "rejected_candidate_ids": rejected,
        "needs_review_candidate_ids": needs_review,
        "completeness": {
            "video_candidate_count": video_candidate_count,
            "channel_candidate_count": channel_candidate_count,
            "minimum_video_candidates": minimum_video_candidates,
            "minimum_channel_candidates": minimum_channel_candidates,
            "requested_video_limit": video_candidates.get("requested_limit"),
            "requested_channel_limit": channel_candidates.get("requested_limit"),
            "active_query_count": len(active_queries),
            "max_queries": max_queries,
            "query_budget_exhausted": query_budget_exhausted,
        },
        "relevance": {"accepted_count": len(accepted), "needs_review_count": len(needs_review)},
        "non_contradiction": {"relations": _relation_counts(video_candidates, channel_candidates)},
        "created_at": utc_now(),
        "updated_at": utc_now(),
    }


def build_discovery_graph(
    video_id: str,
    research_run_id: str,
    discovery_run_id: str,
    seeds: dict[str, Any],
    queries: dict[str, Any],
    video_candidates: dict[str, Any],
    channel_candidates: dict[str, Any],
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = [
        {"id": f"video:{video_id}", "label": "Video", "video_id": video_id},
    ]
    relationships: list[dict[str, Any]] = []
    for seed in seeds.get("items") or []:
        nodes.append(
            {
                "id": f"discovery_seed:{discovery_run_id}:{seed['seed_id']}",
                "label": "DiscoverySeed",
                "text": seed.get("text"),
                "kind": seed.get("kind"),
                "source_video_id": video_id,
                "research_run_id": research_run_id,
                "discovery_run_id": discovery_run_id,
            }
        )
        for source_id in seed.get("source_ids") or []:
            relationships.append(
                _rel(
                    f"video:{video_id}",
                    f"discovery_seed:{discovery_run_id}:{seed['seed_id']}",
                    "SUGGESTED_BY_RESEARCH_FINDING",
                    video_id,
                    research_run_id,
                    discovery_run_id,
                    seed_id=seed["seed_id"],
                    score=seed.get("confidence", 0.0),
                    review_status="accepted",
                    source_id=source_id,
                )
            )
    for query in queries.get("items") or []:
        nodes.append(
            {
                "id": f"discovery_query:{discovery_run_id}:{query['query_id']}",
                "label": "DiscoveryQuery",
                "query": query.get("query"),
                "intent": query.get("intent"),
                "language": query.get("language"),
                "source_video_id": video_id,
                "research_run_id": research_run_id,
                "discovery_run_id": discovery_run_id,
            }
        )
        for seed_id in query.get("seed_ids") or []:
            relationships.append(
                _rel(
                    f"discovery_seed:{discovery_run_id}:{seed_id}",
                    f"discovery_query:{discovery_run_id}:{query['query_id']}",
                    "DISCOVERED_BY_QUERY",
                    video_id,
                    research_run_id,
                    discovery_run_id,
                    seed_id=seed_id,
                    query_id=query["query_id"],
                    score=1.0,
                    review_status="accepted",
                )
            )
    for candidate in video_candidates.get("items") or []:
        node_id = f"youtube_video_candidate:{candidate['youtube_video_id']}"
        nodes.append(
            {
                "id": node_id,
                "label": "YouTubeVideoCandidate",
                "youtube_video_id": candidate["youtube_video_id"],
                "url": candidate["canonical_url"],
                "title": candidate.get("title"),
                "source_video_id": video_id,
                "research_run_id": research_run_id,
                "discovery_run_id": discovery_run_id,
                "review_status": candidate.get("review_status"),
                "discovery_score": candidate.get("discovery_score"),
            }
        )
        relationships.append(
            _rel(
                f"video:{video_id}",
                node_id,
                "RELATED_TO_VIDEO",
                video_id,
                research_run_id,
                discovery_run_id,
                seed_id=(candidate.get("seed_ids") or [""])[0],
                query_id=(candidate.get("query_ids") or [""])[0],
                youtube_video_id=candidate["youtube_video_id"],
                score=candidate.get("discovery_score", 0.0),
                review_status=candidate.get("review_status"),
            )
        )
    for candidate in channel_candidates.get("items") or []:
        node_id = f"youtube_channel_candidate:{candidate['youtube_channel_id']}"
        nodes.append(
            {
                "id": node_id,
                "label": "YouTubeChannelCandidate",
                "youtube_channel_id": candidate["youtube_channel_id"],
                "url": candidate["canonical_url"],
                "title": candidate.get("title"),
                "source_video_id": video_id,
                "research_run_id": research_run_id,
                "discovery_run_id": discovery_run_id,
                "review_status": candidate.get("review_status"),
                "discovery_score": candidate.get("discovery_score"),
            }
        )
        relationships.append(
            _rel(
                f"video:{video_id}",
                node_id,
                "CHANNEL_CANDIDATE_FOR_TOPIC",
                video_id,
                research_run_id,
                discovery_run_id,
                seed_id=(candidate.get("seed_ids") or [""])[0],
                query_id=(candidate.get("query_ids") or [""])[0],
                youtube_channel_id=candidate["youtube_channel_id"],
                score=candidate.get("discovery_score", 0.0),
                review_status=candidate.get("review_status"),
            )
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "video_id": video_id,
        "research_run_id": research_run_id,
        "discovery_run_id": discovery_run_id,
        "nodes": nodes,
        "relationships": relationships,
        "created_at": utc_now(),
        "updated_at": utc_now(),
    }


def render_discovery_report(
    video_id: str,
    review: dict[str, Any],
    video_candidates: dict[str, Any],
    channel_candidates: dict[str, Any],
) -> str:
    lines = [
        f"# YouTube Discovery: {video_id}",
        "",
        "## Summary",
        "",
        f"- Review status: {review.get('status')}",
        f"- Video candidates: {len(video_candidates.get('items') or [])}",
        f"- Channel candidates: {len(channel_candidates.get('items') or [])}",
        "",
        "## Video Candidates",
        "",
        "| Status | Score | Title | URL | Reason |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for item in video_candidates.get("items") or []:
        lines.append(
            f"| {item.get('review_status')} | {float(item.get('discovery_score') or 0):.2f} | "
            f"{_md(item.get('title') or item.get('youtube_video_id'))} | {item.get('canonical_url')} | {_md(item.get('reason') or '')} |"
        )
    lines.extend(["", "## Channel Candidates", "", "| Status | Score | Title | URL | Reason |", "| --- | ---: | --- | --- | --- |"])
    for item in channel_candidates.get("items") or []:
        lines.append(
            f"| {item.get('review_status')} | {float(item.get('discovery_score') or 0):.2f} | "
            f"{_md(item.get('title') or item.get('youtube_channel_id'))} | {item.get('canonical_url')} | {_md(item.get('reason') or '')} |"
        )
    lines.extend(["", "## Review Issues", ""])
    if review.get("issues"):
        lines.extend(f"- {issue['severity']}: {issue['code']}: {issue['message']}" for issue in review["issues"])
    else:
        lines.append("- No review issues.")
    return "\n".join(lines) + "\n"


def enqueue_discovered_videos(
    runtime: RuntimeStore,
    store: LibraryStore,
    video_id: str,
    *,
    min_score: float = 0.75,
    limit: int = 5,
    quality_mode: str = "strict",
    acceptance_mode: str = "data_api_only",
) -> dict[str, Any]:
    review = load_discovery_review(store, video_id)
    if review.get("status") == "failed":
        raise DiscoveryValidationError("enqueue requires non-failed discovery review")
    accepted = set(review.get("accepted_candidate_ids") or [])
    candidates_path = discovery_dir_for(store, video_id) / "video_candidates.json"
    candidates = _read_json(candidates_path)
    selected = [
        item
        for item in candidates.get("items") or []
        if item.get("candidate_id") in accepted and float(item.get("discovery_score") or 0.0) >= min_score
    ][: max(0, int(limit))]
    for item in selected:
        if not _canonical_url_matches_verified_id(item):
            raise DiscoveryValidationError(
                f"video canonical URL must match verified YouTube video ID: {item.get('candidate_id') or item.get('youtube_video_id')}"
            )
    jobs = []
    for item in selected:
        job = runtime.create_video_job(
            video_id=item["youtube_video_id"],
            canonical_url=item["canonical_url"],
            quality_mode=quality_mode,
            acceptance_mode=acceptance_mode,
        )
        runtime.ensure_stage_plan(job.id)
        jobs.append(job.to_dict())
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_video_id": video_id,
        "queued_count": len(jobs),
        "jobs": jobs,
        "min_score": min_score,
        "limit": limit,
        "created_at": utc_now(),
    }
    _append_audit(discovery_dir_for(store, video_id) / "enqueue_audit.json", payload)
    return payload


def enqueue_discovered_channel(
    runtime: RuntimeStore,
    store: LibraryStore,
    video_id: str,
    *,
    channel_id: str,
    limit: int | None = None,
    quality_mode: str = "strict",
    acceptance_mode: str = "data_api_only",
    settings: Any | None = None,
) -> dict[str, Any]:
    review = load_discovery_review(store, video_id)
    if review.get("status") == "failed":
        raise DiscoveryValidationError("enqueue requires non-failed discovery review")
    channels = _read_json(discovery_dir_for(store, video_id) / "channel_candidates.json")
    accepted = set(review.get("accepted_candidate_ids") or [])
    candidate = next(
        (
            item
            for item in channels.get("items") or []
            if item.get("youtube_channel_id") == channel_id and item.get("candidate_id") in accepted
        ),
        None,
    )
    if candidate is None:
        raise DiscoveryValidationError(f"channel candidate is not accepted: {channel_id}")
    if not _canonical_url_matches_verified_id(candidate):
        raise DiscoveryValidationError(f"channel canonical URL must match verified YouTube channel ID: {channel_id}")
    from yt_sub_hub.channel import enqueue_channel_videos

    listing = fetch_channel_listing(channel_id, limit=limit, settings=settings or _settings())
    if listing.channel_id != channel_id:
        raise DiscoveryValidationError(f"channel listing ID mismatch: {listing.channel_id} != {channel_id}")
    channel_job, video_jobs = enqueue_channel_videos(
        runtime,
        store,
        channel_url=listing.channel_url,
        channel_id=listing.channel_id,
        uploads_playlist_id=listing.uploads_playlist_id,
        videos=listing.videos,
        limit=limit,
        quality_mode=quality_mode,
        acceptance_mode=acceptance_mode,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_video_id": video_id,
        "channel_job": channel_job.to_dict(),
        "video_jobs": [job.to_dict() for job in video_jobs],
        "created_at": utc_now(),
    }
    _append_audit(discovery_dir_for(store, video_id) / "enqueue_audit.json", payload)
    return payload


def sync_discovery_graph_to_neo4j(store: LibraryStore, video_id: str, *, settings: Any) -> dict[str, Any]:
    review = load_discovery_review(store, video_id)
    if review.get("status") == "failed":
        raise DiscoveryValidationError("neo4j discovery sync requires non-failed discovery review")
    path = discovery_dir_for(store, video_id) / "graph.json"
    if not path.exists():
        raise FileNotFoundError(path)
    graph_payload = _read_json(path)
    _validate_discovery_graph_candidate_urls(graph_payload)
    payload = GraphPayload(nodes=graph_payload.get("nodes") or [], relationships=graph_payload.get("relationships") or [])
    graph_store = Neo4jGraphStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    try:
        if hasattr(graph_store, "delete_discovery_for_video"):
            graph_store.delete_discovery_for_video(video_id)
        else:
            graph_store.delete_research_for_video(video_id)
        graph_store.upsert_payload(payload)
        if hasattr(graph_store, "discovery_consistency"):
            consistency = graph_store.discovery_consistency(payload)
        else:
            consistency = graph_store.research_consistency(payload)
    finally:
        graph_store.close()
    result = {
        "video_id": video_id,
        "checked": bool(consistency.get("checked")),
        "missing": consistency.get("missing"),
        "extra": consistency.get("extra"),
        "node_count": len(payload.nodes),
        "relationship_count": len(payload.relationships),
    }
    if result["checked"] and result["missing"] == 0 and result["extra"] == 0:
        status_path = discovery_dir_for(store, video_id) / "status.json"
        status = _read_json(status_path) if status_path.exists() else {"video_id": video_id}
        status["discovery_sync"] = {"neo4j": {**result, "synced_at": utc_now()}}
        status["updated_at"] = utc_now()
        _write_json(status_path, status)
    return result


def _require_research_ready(store: LibraryStore, video_id: str) -> str:
    research_dir = store.video_dir(video_id) / "research"
    review_path = research_dir / "review" / "research_artifact_review.json"
    if not review_path.exists():
        raise DiscoveryValidationError("discovery requires research_artifact_review status passed")
    review = _read_json(review_path)
    if review.get("status") != "passed":
        raise DiscoveryValidationError("discovery requires research_artifact_review status passed")
    from yt_sub_hub.research_enrichment import require_current_research_run, ResearchValidationError
    try:
        return require_current_research_run(store, video_id)
    except ResearchValidationError as exc:
        raise DiscoveryValidationError(str(exc)) from exc


def _extract_seeds(
    store: LibraryStore,
    video_id: str,
    research_run_id: str,
    discovery_run_id: str,
    input_hashes: dict[str, str],
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    video_dir = store.video_dir(video_id)
    analysis = _read_optional_json(video_dir / "analysis.ru.json") or {}
    for concept in analysis.get("concepts") or []:
        text = _trim_query_text(str(concept.get("name") or ""))
        if text:
            items.append(_seed(video_id, len(items) + 1, "concept", text, "analysis.ru.json", "concepts.name", [], 0.85))
    for claim in analysis.get("claims") or []:
        text = _trim_query_text(str(claim.get("text") or ""))
        if text:
            items.append(_seed(video_id, len(items) + 1, "claim", text, "analysis.ru.json", "claims.text", [], 0.75))
    research_dir = video_dir / "research"
    for finding in _read_optional_json(research_dir / "findings.ru.json") or []:
        if not isinstance(finding, dict) or finding.get("accepted") is False:
            continue
        text = _trim_query_text(str(finding.get("claim_ru") or ""))
        if text:
            items.append(
                _seed(
                    video_id,
                    len(items) + 1,
                    "finding",
                    text,
                    "research/findings.ru.json",
                    "claim_ru",
                    [str(finding.get("finding_id") or ""), *[str(source_id) for source_id in finding.get("source_ids") or []]],
                    0.9,
                )
            )
    for summary in _read_optional_json(research_dir / "source_summaries.json") or []:
        if not isinstance(summary, dict):
            continue
        for point in summary.get("key_points") or []:
            text = _trim_query_text(str(point or ""))
            if text:
                items.append(
                    _seed(
                        video_id,
                        len(items) + 1,
                        "external_context",
                        text,
                        "research/source_summaries.json",
                        "key_points",
                        [str(summary.get("source_id") or "")],
                        0.7,
                    )
                )
    deduped = []
    seen = set()
    for item in items:
        key = _normalize(item["text"])
        if key in seen:
            continue
        seen.add(key)
        item["selected"] = len(deduped) < 12
        deduped.append(item)
    return _artifact_root("seeds", video_id, research_run_id, discovery_run_id, input_hashes, deduped)


QUERY_PLAN_INTENTS = ("same_topic", "deep_dive", "channel_discovery", "counterpoint", "recent_updates")
ENGLISH_ALIAS_MAP = {
    "mortal shell 2": ["Mortal Shell 2"],
    "blight survival": ["Blight Survival"],
    "stranger than heaven": ["Stranger than Heaven"],
    "stranger zenheaven": ["Stranger than Heaven"],
    "artificial detective": ["Artificial Detective"],
}
RU_INTENT_SUFFIXES = {
    "same_topic": "",
    "deep_dive": "разбор",
    "channel_discovery": "канал",
    "counterpoint": "критика",
    "recent_updates": "2026",
}
EN_INTENT_SUFFIXES = {
    "same_topic": "",
    "deep_dive": "explained",
    "channel_discovery": "channel",
    "counterpoint": "critique",
    "recent_updates": "2026",
}


def _detect_query_language(text: str) -> str:
    cyrillic = len(re.findall(r"[А-Яа-яЁё]", text or ""))
    latin = len(re.findall(r"[A-Za-z]", text or ""))
    if cyrillic and latin:
        return "mixed"
    if cyrillic:
        return "ru"
    if latin:
        return "en"
    return "unknown"


def _extract_latin_entities(seed_text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9:'-]*|[А-Яа-яЁё]+", seed_text or "")
    entities: list[str] = []
    current: list[str] = []
    for token in tokens:
        if re.search(r"[A-Za-z0-9]", token):
            current.append(token)
            continue
        if current:
            entity = " ".join(current)
            if re.search(r"[A-Za-z]", entity):
                entities.append(entity)
            current = []
    if current:
        entity = " ".join(current)
        if re.search(r"[A-Za-z]", entity):
            entities.append(entity)
    return _dedupe([_trim_query_text(entity, max_words=8) for entity in entities if len(entity.strip()) > 1])


def _english_aliases_for_seed(seed_text: str) -> list[str]:
    normalized = _normalize(seed_text)
    aliases: list[str] = []
    for marker, values in ENGLISH_ALIAS_MAP.items():
        if marker in normalized:
            aliases.extend(values)
    aliases.extend(_extract_latin_entities(seed_text))
    return _dedupe([_trim_query_text(alias, max_words=8) for alias in aliases])


def _russian_aliases_for_seed(seed_text: str) -> list[str]:
    if _has_cyrillic(seed_text):
        return [_trim_query_text(seed_text, max_words=8)]
    return []


def _build_query_variants(seed: dict[str, Any]) -> list[dict[str, Any]]:
    seed_text = str(seed.get("text") or "")
    variants: list[dict[str, Any]] = []
    if _detect_query_language(seed_text) == "en":
        base = _trim_query_text(seed_text, max_words=8)
        if base:
            variants.append({"language": "en", "query": base, "query_source": "seed_text"})
    for alias in _russian_aliases_for_seed(seed_text):
        variants.append({"language": "ru", "query": alias, "query_source": "seed_text"})
    for alias in _english_aliases_for_seed(seed_text):
        source = "alias_map" if any(alias in values for values in ENGLISH_ALIAS_MAP.values()) else "latin_entity"
        variants.append({"language": "en", "query": alias, "query_source": source})
    deduped: list[dict[str, Any]] = []
    seen = set()
    for variant in variants:
        key = _normalize(f"{variant.get('language')}:{variant.get('query')}")
        if not variant.get("query") or key in seen:
            continue
        seen.add(key)
        deduped.append(variant)
    return deduped


def _query_language_is_valid(query: str, language: str) -> bool:
    detected = _detect_query_language(query)
    if language == "en":
        return detected == "en"
    if language == "ru":
        return detected in {"ru", "mixed"}
    return False


def _query_suffix(language: str, intent: str) -> str:
    if language == "ru":
        return RU_INTENT_SUFFIXES[intent]
    return EN_INTENT_SUFFIXES[intent]


def _query_dedupe_key(intent: str, language: str, query_text: str) -> str:
    return _normalize(f"{query_text}:{intent}:{language}")


def _quota_estimate() -> dict[str, Any]:
    return {
        "search_video_units": SEARCH_LIST_QUOTA_UNITS,
        "search_channel_units": SEARCH_LIST_QUOTA_UNITS,
        "search_calls": 2,
        "search_units_bucket": "search",
        "video_details_units": DETAILS_LIST_QUOTA_UNITS,
        "channel_details_units": DETAILS_LIST_QUOTA_UNITS,
        "general_units": _estimated_query_quota_units(),
        # Retained for existing artifact readers; search has its own bucket.
        "total_units": _estimated_query_quota_units(),
        "total_units_bucket": "general_data",
    }


def _rank_query_seed(seed: dict[str, Any]) -> tuple[int, float, int, str]:
    seed_text = str(seed.get("text") or "")
    return (
        0 if seed.get("selected") else 1,
        -float(seed.get("confidence") or 0.0),
        -len(_english_aliases_for_seed(seed_text)),
        str(seed.get("seed_id") or ""),
    )


def _active_query_items(queries: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in queries.get("items") or [] if isinstance(item, dict) and not item.get("rejected_reason")]


def _plan_queries(
    video_id: str,
    research_run_id: str,
    discovery_run_id: str,
    seeds: dict[str, Any],
    *,
    max_queries: int = 8,
) -> dict[str, Any]:
    if max_queries < 0:
        raise DiscoveryValidationError("max_queries must be non-negative")
    items: list[dict[str, Any]] = []
    if max_queries == 0:
        return _queries_artifact(video_id, research_run_id, discovery_run_id, seeds.get("input_hashes") or {}, items, max_queries=0)
    seen = set()
    priority = 1
    executable_count = 0
    ranked_seeds = sorted([seed for seed in seeds.get("items") or [] if isinstance(seed, dict)], key=_rank_query_seed)
    for seed in ranked_seeds:
        if not seed.get("selected"):
            continue
        variants = _build_query_variants(seed)
        for intent in QUERY_PLAN_INTENTS:
            for variant in variants:
                base = str(variant.get("query") or "")
                language = str(variant.get("language") or "")
                suffix = _query_suffix(language, intent)
                query_text = " ".join(part for part in [base, suffix] if part).strip()
                query_text = _trim_query_text(query_text, max_words=9)
                if not query_text:
                    continue
                rejected_reason = None
                if not _query_language_is_valid(query_text, language):
                    rejected_reason = "query_language_mismatch"
                dedupe_key = _query_dedupe_key(intent, language, query_text)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                if rejected_reason is None and executable_count >= int(max_queries):
                    return _queries_artifact(video_id, research_run_id, discovery_run_id, seeds.get("input_hashes") or {}, items, max_queries=max_queries)
                items.append(
                    {
                        "query_id": f"ydq-{priority:03d}",
                        "seed_ids": [seed["seed_id"]],
                        "intent": intent,
                        "language": language,
                        "query": query_text,
                        "priority": priority,
                        "dedupe_key": dedupe_key,
                        "query_source": variant.get("query_source") or "seed_text",
                        "quota_estimate": _quota_estimate(),
                        "rejected_reason": rejected_reason,
                    }
                )
                priority += 1
                if rejected_reason is None:
                    executable_count += 1
                if executable_count >= int(max_queries):
                    return _queries_artifact(video_id, research_run_id, discovery_run_id, seeds.get("input_hashes") or {}, items, max_queries=max_queries)
    return _queries_artifact(video_id, research_run_id, discovery_run_id, seeds.get("input_hashes") or {}, items, max_queries=max_queries)


def _estimated_query_quota_units() -> int:
    return DETAILS_LIST_QUOTA_UNITS * 2


def _merge_video_candidates(
    target: dict[str, dict[str, Any]],
    search_results: list[dict[str, Any]],
    details: dict[str, dict[str, Any]],
    query: dict[str, Any],
    seeds: dict[str, Any],
    source_video_id: str,
) -> None:
    seed_ids = list(query.get("seed_ids") or [])
    seed_text = _seed_text(seeds, seed_ids[0] if seed_ids else "")
    for result in search_results:
        video_id = str(result.get("youtube_video_id") or "")
        if video_id == source_video_id or not _valid_video_id(video_id):
            continue
        data = {**result, **details.get(video_id, {})}
        candidate = target.get(video_id) or _video_candidate(video_id, data, query, seed_ids, seed_text)
        _merge_provenance(candidate, query, seed_ids)
        _update_best_score(candidate, _score_candidate(candidate, seed_text))
        target[video_id] = candidate


def _merge_channel_candidates(
    target: dict[str, dict[str, Any]],
    search_results: list[dict[str, Any]],
    details: dict[str, dict[str, Any]],
    query: dict[str, Any],
    seeds: dict[str, Any],
) -> None:
    seed_ids = list(query.get("seed_ids") or [])
    seed_text = _seed_text(seeds, seed_ids[0] if seed_ids else "")
    for result in search_results:
        channel_id = str(result.get("youtube_channel_id") or "")
        if not _valid_channel_id(channel_id):
            continue
        data = {**result, **details.get(channel_id, {})}
        candidate = target.get(channel_id) or _channel_candidate(channel_id, data, query, seed_ids, seed_text)
        _merge_provenance(candidate, query, seed_ids)
        _update_best_score(candidate, _score_candidate(candidate, seed_text, channel=True))
        target[channel_id] = candidate


def _video_candidate(video_id: str, data: dict[str, Any], query: dict[str, Any], seed_ids: list[str], seed_text: str) -> dict[str, Any]:
    return {
        "candidate_id": _stable_id("youtube-video-candidate", video_id, length=18),
        "kind": "video",
        "youtube_video_id": video_id,
        "canonical_url": f"https://www.youtube.com/watch?v={video_id}",
        "title": data.get("title"),
        "description": data.get("description"),
        "channel_id": data.get("channel_id"),
        "channel_title": data.get("channel_title"),
        "published_at": data.get("published_at"),
        "duration": data.get("duration"),
        "view_count": data.get("view_count"),
        "language": data.get("language"),
        "tags": data.get("tags") or [],
        "thumbnails": data.get("thumbnails") or {},
        "query_ids": [query["query_id"]],
        "seed_ids": seed_ids,
        "source_ids": seed_ids,
        "relationship": _relationship_for_query(query),
        "query_intents": [query.get("intent") or "unknown"],
        "review_status": "needs_review",
        "processed_status": "new",
        "reason": f"Matched seed '{seed_text}' via query '{query['query']}'.",
    }


def _channel_candidate(channel_id: str, data: dict[str, Any], query: dict[str, Any], seed_ids: list[str], seed_text: str) -> dict[str, Any]:
    return {
        "candidate_id": _stable_id("youtube-channel-candidate", channel_id, length=18),
        "kind": "channel",
        "youtube_channel_id": channel_id,
        "canonical_url": f"https://www.youtube.com/channel/{channel_id}",
        "title": data.get("title"),
        "description": data.get("description"),
        "published_at": data.get("published_at"),
        "subscriber_count": data.get("subscriber_count"),
        "video_count": data.get("video_count"),
        "view_count": data.get("view_count"),
        "language": data.get("language"),
        "thumbnails": data.get("thumbnails") or {},
        "query_ids": [query["query_id"]],
        "seed_ids": seed_ids,
        "source_ids": seed_ids,
        "relationship": "unclear",
        "query_intents": [query.get("intent") or "unknown"],
        "review_status": "needs_review",
        "reason": f"Matched seed '{seed_text}' via channel query '{query['query']}'.",
    }


def _score_candidate(candidate: dict[str, Any], seed_text: str, *, channel: bool = False) -> dict[str, float]:
    haystack = " ".join(
        str(value or "")
        for value in [
            candidate.get("title"),
            candidate.get("description"),
            candidate.get("channel_title"),
            " ".join(candidate.get("tags") or []),
        ]
    )
    topic_overlap = _token_overlap(seed_text, haystack)
    concept_overlap = topic_overlap
    language = str(candidate.get("language") or "").lower()
    language_match = 1.0 if language in {"ru", "en", ""} else 0.4
    availability = 1.0 if candidate.get("canonical_url") else 0.0
    caption_readiness = 0.5 if channel else 0.6
    authority = min(1.0, (float(candidate.get("subscriber_count") or candidate.get("view_count") or 0) / 10000.0)) if channel else 0.5
    freshness = 0.6 if candidate.get("published_at") else 0.4
    relevance = max(0.45, topic_overlap)
    novelty = 0.7
    contradiction_risk = 0.25 if candidate.get("relationship") == "contradicts" else 0.05
    source_alignment = max(0.4, topic_overlap)
    score = (
        relevance * 0.26
        + topic_overlap * 0.15
        + concept_overlap * 0.12
        + source_alignment * 0.12
        + novelty * 0.08
        + language_match * 0.06
        + availability * 0.08
        + caption_readiness * 0.04
        + authority * 0.04
        + freshness * 0.05
        - contradiction_risk * 0.05
    )
    return {
        "relevance_score": round(relevance, 4),
        "topic_overlap_score": round(topic_overlap, 4),
        "concept_overlap_score": round(concept_overlap, 4),
        "source_alignment_score": round(source_alignment, 4),
        "novelty_score": round(novelty, 4),
        "contradiction_risk": round(contradiction_risk, 4),
        "language_match_score": round(language_match, 4),
        "availability_score": round(availability, 4),
        "caption_readiness_score": round(caption_readiness, 4),
        "channel_authority_score": round(authority, 4),
        "freshness_score": round(freshness, 4),
        "discovery_score": round(min(1.0, max(0.0, score)), 4),
    }


def _candidate_collection(
    name: str,
    video_id: str,
    research_run_id: str,
    discovery_run_id: str,
    input_hashes: dict[str, str],
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    return _artifact_root(name, video_id, research_run_id, discovery_run_id, input_hashes, items)


def _queries_artifact(
    video_id: str,
    research_run_id: str,
    discovery_run_id: str,
    input_hashes: dict[str, str],
    items: list[dict[str, Any]],
    *,
    max_queries: int,
) -> dict[str, Any]:
    payload = _artifact_root("queries", video_id, research_run_id, discovery_run_id, input_hashes, items)
    payload["max_queries"] = max(0, int(max_queries))
    payload["active_query_count"] = len(_active_query_items(payload))
    return payload


def _artifact_root(
    artifact_type: str,
    video_id: str,
    research_run_id: str,
    discovery_run_id: str,
    input_hashes: dict[str, str],
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": artifact_type,
        "video_id": video_id,
        "research_run_id": research_run_id,
        "discovery_run_id": discovery_run_id,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "input_hashes": input_hashes,
        "artifact_hash": _hash_json(items),
        "items": items,
    }


def _minimum_candidate_target(candidates: dict[str, Any], *, default: int) -> int:
    requested_limit = _int_or_none(candidates.get("requested_limit"))
    if requested_limit is None:
        return int(default)
    return max(1, min(int(default), int(requested_limit)))


def _seed(
    video_id: str,
    index: int,
    kind: str,
    text: str,
    source_artifact: str,
    source_field: str,
    source_ids: list[str],
    confidence: float,
) -> dict[str, Any]:
    return {
        "seed_id": f"seed-{index:03d}",
        "kind": kind,
        "text": text,
        "language": "ru" if _has_cyrillic(text) else "en",
        "source_artifact": source_artifact,
        "source_field": source_field,
        "source_ids": [source_id for source_id in source_ids if source_id],
        "confidence": confidence,
        "selected": True,
    }


def _rel(
    from_id: str,
    to_id: str,
    rel_type: str,
    source_video_id: str,
    research_run_id: str,
    discovery_run_id: str,
    **props: Any,
) -> dict[str, Any]:
    rel = {
        "id": _stable_id("discovery-rel", from_id, rel_type, to_id, research_run_id, discovery_run_id, length=22),
        "from": from_id,
        "to": to_id,
        "type": rel_type,
        "source_video_id": source_video_id,
        "research_run_id": research_run_id,
        "discovery_run_id": discovery_run_id,
        "score": props.pop("score", 0.0),
        "review_status": props.pop("review_status", None),
    }
    rel.update({key: value for key, value in props.items() if value not in {None, ""}})
    return rel


def _input_hashes(store: LibraryStore, video_id: str) -> dict[str, str]:
    paths = [
        store.video_dir(video_id) / "analysis.ru.json",
        store.video_dir(video_id) / "note.md",
        store.video_dir(video_id) / "research" / "research_report.md",
        store.video_dir(video_id) / "research" / "findings.ru.json",
        store.video_dir(video_id) / "research" / "source_summaries.json",
        store.video_dir(video_id) / "research" / "queries.json",
        store.video_dir(video_id) / "research" / "search_results.normalized.json",
        store.video_dir(video_id) / "research" / "research_graph.json",
    ]
    return {str(path.relative_to(store.root)): _hash_file(path) for path in paths if path.exists()}


def _quota_usage(provider: YouTubeDiscoveryProvider, quota: QuotaBudget, *, max_quota_units: int) -> dict[str, Any]:
    usage = quota.to_dict()
    if isinstance(provider, YouTubeDataApiDiscoveryProvider):
        return usage
    # Alternate providers do not expose their Google request attempts.
    usage.update(accounting="unavailable_for_provider", used_units=None, remaining_units=None,
                 used_search_calls=None, remaining_search_calls=None)
    return usage


def _write_discovery_status(
    discovery_dir: Path,
    video_id: str,
    *,
    status: str,
    research_run_id: str | None = None,
    discovery_run_id: str | None = None,
    limit: int | None = None,
    max_quota_units: int | None = None,
    review_status: str | None = None,
    video_candidate_count: int | None = None,
    channel_candidate_count: int | None = None,
) -> None:
    path = discovery_dir / "status.json"
    existing = _read_json(path) if path.exists() else {}
    payload = {
        **existing,
        "schema_version": SCHEMA_VERSION,
        "video_id": video_id,
        "status": status,
        "updated_at": utc_now(),
    }
    if "created_at" not in payload:
        payload["created_at"] = payload["updated_at"]
    if research_run_id:
        payload["research_run_id"] = research_run_id
    if discovery_run_id:
        payload["discovery_run_id"] = discovery_run_id
    if limit is not None:
        payload["limit"] = limit
    if max_quota_units is not None:
        payload["max_quota_units"] = max_quota_units
    if review_status is not None:
        payload["review_status"] = review_status
    if video_candidate_count is not None:
        payload["video_candidate_count"] = video_candidate_count
    if channel_candidate_count is not None:
        payload["channel_candidate_count"] = channel_candidate_count
    _write_json(path, payload)


def _write_discovery_status_from_artifacts(
    store: LibraryStore,
    video_id: str,
    *,
    status: str,
    review: dict[str, Any] | None = None,
) -> None:
    discovery_dir = discovery_dir_for(store, video_id)
    seeds = _read_optional_json(discovery_dir / "seeds.json") or {}
    video_candidates = _read_optional_json(discovery_dir / "video_candidates.json") or {"items": []}
    channel_candidates = _read_optional_json(discovery_dir / "channel_candidates.json") or {"items": []}
    _write_discovery_status(
        discovery_dir,
        video_id,
        status=status,
        research_run_id=seeds.get("research_run_id"),
        discovery_run_id=seeds.get("discovery_run_id"),
        review_status=(review or {}).get("status"),
        video_candidate_count=len(video_candidates.get("items") or []),
        channel_candidate_count=len(channel_candidates.get("items") or []),
    )


def _artifact_info(path: Path, root: Path) -> dict[str, Any]:
    exists = path.exists()
    return {
        "path": str(path.relative_to(root)) if path.is_absolute() or root in path.parents else str(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
        "updated_at": path.stat().st_mtime if exists else None,
    }


def _append_audit(path: Path, payload: dict[str, Any]) -> None:
    existing = _read_json(path) if path.exists() else {"schema_version": SCHEMA_VERSION, "items": []}
    existing.setdefault("items", []).append(payload)
    existing["updated_at"] = utc_now()
    _write_json(path, existing)


def _merge_provenance(candidate: dict[str, Any], query: dict[str, Any], seed_ids: list[str]) -> None:
    candidate["query_intents"] = _dedupe([*candidate.get("query_intents", []), query.get("intent") or "unknown"])
    candidate["query_ids"] = _dedupe([*candidate.get("query_ids", []), query["query_id"]])
    candidate["seed_ids"] = _dedupe([*candidate.get("seed_ids", []), *seed_ids])
    candidate["source_ids"] = _dedupe([*candidate.get("source_ids", []), *seed_ids])


def _update_best_score(candidate: dict[str, Any], scores: dict[str, float]) -> None:
    if float(scores.get("discovery_score") or 0.0) >= float(candidate.get("discovery_score") or -1.0):
        candidate.update(scores)


def _relationship_for_query(query: dict[str, Any]) -> str:
    # Search intent is provenance, not evidence of a video's semantic stance.
    return "unclear"


def _relation_counts(video_candidates: dict[str, Any], channel_candidates: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in [*(video_candidates.get("items") or []), *(channel_candidates.get("items") or [])]:
        relation = str(item.get("relationship") or "unclear")
        counts[relation] = counts.get(relation, 0) + 1
    return counts


def _issue(severity: str, code: str, candidate_id: str, message: str) -> dict[str, str]:
    return {"severity": severity, "code": code, "candidate_id": candidate_id, "message": message}


def _expected_video_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _expected_channel_url(channel_id: str) -> str:
    return f"https://www.youtube.com/channel/{channel_id}"


def _canonical_url_matches_verified_id(candidate: dict[str, Any]) -> bool:
    kind = str(candidate.get("kind") or "")
    canonical_url = str(candidate.get("canonical_url") or "")
    if kind == "video":
        video_id = str(candidate.get("youtube_video_id") or "")
        return _valid_video_id(video_id) and canonical_url == _expected_video_url(video_id)
    if kind == "channel":
        channel_id = str(candidate.get("youtube_channel_id") or "")
        return _valid_channel_id(channel_id) and canonical_url == _expected_channel_url(channel_id)
    return False


def _validate_discovery_graph_candidate_urls(graph_payload: dict[str, Any]) -> None:
    nodes = graph_payload.get("nodes") or []
    node_by_id = {str(node.get("id") or ""): node for node in nodes if isinstance(node, dict)}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        label = node.get("label")
        if label == "YouTubeVideoCandidate":
            video_id = str(node.get("youtube_video_id") or "")
            if not _valid_video_id(video_id) or str(node.get("url") or "") != _expected_video_url(video_id):
                raise DiscoveryValidationError("discovery graph video canonical URL must match verified YouTube video ID")
        if label == "YouTubeChannelCandidate":
            channel_id = str(node.get("youtube_channel_id") or "")
            if not _valid_channel_id(channel_id) or str(node.get("url") or "") != _expected_channel_url(channel_id):
                raise DiscoveryValidationError("discovery graph channel canonical URL must match verified YouTube channel ID")
    for relationship in graph_payload.get("relationships") or []:
        if not isinstance(relationship, dict):
            continue
        to_node = node_by_id.get(str(relationship.get("to") or ""))
        if relationship.get("youtube_video_id"):
            if not to_node or to_node.get("label") != "YouTubeVideoCandidate":
                raise DiscoveryValidationError("discovery graph relationship video ID must match candidate node")
            if str(relationship.get("youtube_video_id") or "") != str(to_node.get("youtube_video_id") or ""):
                raise DiscoveryValidationError("discovery graph relationship video ID must match candidate node")
        if relationship.get("youtube_channel_id"):
            if not to_node or to_node.get("label") != "YouTubeChannelCandidate":
                raise DiscoveryValidationError("discovery graph relationship channel ID must match candidate node")
            if str(relationship.get("youtube_channel_id") or "") != str(to_node.get("youtube_channel_id") or ""):
                raise DiscoveryValidationError("discovery graph relationship channel ID must match candidate node")
        if to_node and to_node.get("label") == "YouTubeVideoCandidate" and not relationship.get("youtube_video_id"):
            raise DiscoveryValidationError("discovery graph relationship video ID must match candidate node")
        if to_node and to_node.get("label") == "YouTubeChannelCandidate" and not relationship.get("youtube_channel_id"):
            raise DiscoveryValidationError("discovery graph relationship channel ID must match candidate node")


def _valid_video_id(value: str) -> bool:
    return bool(VIDEO_ID_RE.fullmatch(value or ""))


def _valid_channel_id(value: str) -> bool:
    return bool(CHANNEL_ID_RE.fullmatch(value or ""))


def _seed_text(seeds: dict[str, Any], seed_id: str) -> str:
    for seed in seeds.get("items") or []:
        if seed.get("seed_id") == seed_id:
            return str(seed.get("text") or "")
    return ""


def _token_overlap(a: str, b: str) -> float:
    left = {token for token in re.findall(r"[\wА-Яа-яЁё]+", a.lower()) if len(token) > 2}
    right = {token for token in re.findall(r"[\wА-Яа-яЁё]+", b.lower()) if len(token) > 2}
    if not left:
        return 0.0
    return min(1.0, len(left & right) / len(left))


def _trim_query_text(text: str, *, max_words: int = 12) -> str:
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^\wА-Яа-яЁё .:-]+", " ", text)
    words = [word.strip(" .:-") for word in text.split() if word.strip(" .:-")]
    return " ".join(words[:max_words]).strip()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _dedupe(values: list[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _has_cyrillic(text: str) -> bool:
    return bool(re.search(r"[А-Яа-яЁё]", text))


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _stable_id(*parts: Any, length: int = 16) -> str:
    return hashlib.sha256(":".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:length]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return _read_json(path)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _md(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _settings() -> Any:
    from yt_sub_hub.config import get_settings

    return get_settings()
