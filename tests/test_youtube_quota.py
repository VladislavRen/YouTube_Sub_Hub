import json
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import pytest

from yt_sub_hub.library import LibraryStore
from yt_sub_hub.youtube_discovery import (
    DiscoveryProviderError,
    DiscoveryValidationError,
    QuotaBudget,
    YouTubeDataApiDiscoveryProvider,
    execute_discovery_query_stage,
    expand_discovery_queries_stage,
    read_discovery_stage_options,
)


class RecordingYouTubeService:
    """Record actual external request attempts, including failed requests."""

    def __init__(self, *, fail=False, empty=False):
        self.fail = fail
        self.empty = empty
        self.attempts = []
        self.lock = Lock()

    def search(self):
        return self.Resource(self, "search")

    def videos(self):
        return self.Resource(self, "videos")

    def channels(self):
        return self.Resource(self, "channels")

    class Resource:
        def __init__(self, service, endpoint):
            self.service = service
            self.endpoint = endpoint

        def list(self, **params):
            service, endpoint = self.service, self.endpoint

            class Request:
                def execute(self):
                    with service.lock:
                        service.attempts.append((endpoint, params))
                    if service.fail:
                        raise RuntimeError("temporary upstream error")
                    if service.empty:
                        return {"items": []}
                    if endpoint == "search":
                        identifier = {"videoId": "vidAlpha001"} if params["type"] == "video" else {"channelId": "UCgraphAlpha"}
                        return {"items": [{"id": identifier, "snippet": {"title": "Graph", "channelId": "UCgraphAlpha"}}]}
                    return {"items": [{"id": "vidAlpha001" if endpoint == "videos" else "UCgraphAlpha", "snippet": {"title": "Graph"}, "statistics": {}, "contentDetails": {}}]}

            return Request()


def _provider(service, quota=None):
    return YouTubeDataApiDiscoveryProvider(object(), youtube_service=service, quota_budget=quota)


def _store_with_seeds(tmp_path, *, run_id="dr-1"):
    store = LibraryStore(tmp_path)
    root = store.video_dir("srcVideo123") / "youtube_discovery"
    root.mkdir(parents=True, exist_ok=True)
    (root / "seeds.json").write_text(
        json.dumps({
            "research_run_id": "rr-1", "discovery_run_id": run_id, "input_hashes": {},
            "items": [{"seed_id": "seed-1", "text": "graph provenance", "selected": True, "confidence": 0.9}],
        }),
        encoding="utf-8",
    )
    return store


def _query(number=1):
    return {"query_id": f"ydq-{number:03d}", "query": "graph provenance"}


def test_search_calls_use_separate_budget_even_when_general_data_budget_is_zero():
    service = RecordingYouTubeService()
    quota = QuotaBudget(max_units=0)
    provider = _provider(service, quota)

    assert provider.search_videos("graph", 1)[0]["youtube_video_id"] == "vidAlpha001"
    assert provider.search_channels("graph", 1)[0]["youtube_channel_id"] == "UCgraphAlpha"
    with pytest.raises(DiscoveryValidationError):
        provider.get_video_details(["vidAlpha001"])

    assert [endpoint for endpoint, _ in service.attempts] == ["search", "search"]
    usage = quota.to_dict()
    assert usage["used_units"] == 0
    assert usage["used_search_calls"] == 2
    assert usage["scope"] == "discovery_run"
    assert usage["project_daily_remaining"] is None


def test_default_search_budget_stops_the_101st_request_before_google():
    service = RecordingYouTubeService(empty=True)
    provider = _provider(service)
    for _ in range(100):
        assert provider.search_videos("graph", 1) == []

    with pytest.raises(DiscoveryValidationError, match="search"):
        provider.search_channels("graph", 1)

    assert len(service.attempts) == 100
    assert provider.quota_budget.to_dict()["used_units"] == 0
    assert provider.quota_budget.to_dict()["remaining_search_calls"] == 0


@pytest.mark.parametrize("endpoint", ["search", "videos"])
def test_failed_requests_consume_their_bucket_before_retry(endpoint):
    service = RecordingYouTubeService(fail=True)
    provider = _provider(service, QuotaBudget(max_units=1, max_search_calls=1))
    invoke = (lambda: provider.search_videos("graph", 1)) if endpoint == "search" else (lambda: provider.get_video_details(["vidAlpha001"]))

    with pytest.raises(DiscoveryProviderError):
        invoke()
    with pytest.raises(DiscoveryValidationError):
        invoke()

    assert [item[0] for item in service.attempts] == [endpoint]


@pytest.mark.parametrize("general,search,expected", [(2, 100, 1), (900, 3, 1), (0, 100, 0), (900, 0, 0), (1, 100, 0)])
def test_query_planning_respects_both_independent_budgets(tmp_path, general, search, expected):
    store = _store_with_seeds(tmp_path)
    plan = expand_discovery_queries_stage(store, "srcVideo123", max_quota_units=general, max_search_calls=search)

    assert len([item for item in plan["items"] if not item["rejected_reason"]]) == expected
    assert plan["max_queries"] == expected
    assert plan["quota_budget"]["max_units"] == general
    assert plan["quota_budget"]["max_search_calls"] == search
    if expected:
        assert plan["items"][0]["quota_estimate"]["search_calls"] == 2
        assert plan["items"][0]["quota_estimate"]["general_units"] == 2


@pytest.mark.parametrize("limits", [{"max_units": -1}, {"max_search_calls": -1}])
def test_negative_run_budgets_are_rejected(limits):
    with pytest.raises(DiscoveryValidationError, match="non-negative"):
        QuotaBudget(**limits)


def test_zero_stage_budget_is_not_replaced_by_default():
    assert read_discovery_stage_options('{"max_quota_units": 0}')["max_quota_units"] == 0


def test_zero_query_budget_is_reported_as_exhausted_in_review():
    from yt_sub_hub.youtube_discovery import review_youtube_discovery

    review = review_youtube_discovery(
        video_id="srcVideo123", research_run_id="rr-1", discovery_run_id="dr-1",
        seeds={"items": []}, queries={"max_queries": 0, "items": []},
        video_candidates={"items": []}, channel_candidates={"items": []},
    )

    assert review["completeness"]["query_budget_exhausted"] is True


def test_separate_query_stages_and_retries_share_persisted_run_consumption(tmp_path):
    store = _store_with_seeds(tmp_path)
    service = RecordingYouTubeService()
    first = execute_discovery_query_stage(store, "srcVideo123", _query(), provider=_provider(service), max_quota_units=2, max_search_calls=2)

    assert first["status"] == "completed"
    assert first["quota_usage"]["used_units"] == 2
    assert first["quota_usage"]["used_search_calls"] == 2
    for query in [_query(2), _query()]:
        with pytest.raises(DiscoveryValidationError):
            execute_discovery_query_stage(store, "srcVideo123", query, provider=_provider(service), max_quota_units=2, max_search_calls=2)

    assert [endpoint for endpoint, _ in service.attempts] == ["search", "search", "videos", "channels"]


def test_failed_stage_attempt_remains_charged_when_provider_is_recreated(tmp_path):
    store = _store_with_seeds(tmp_path)
    service = RecordingYouTubeService(fail=True)
    for expected_error in [DiscoveryProviderError, DiscoveryValidationError]:
        with pytest.raises(expected_error):
            execute_discovery_query_stage(store, "srcVideo123", _query(), provider=_provider(service), max_search_calls=1)
    assert len(service.attempts) == 1


def test_concurrent_stages_cannot_overspend_shared_search_budget(tmp_path):
    store = _store_with_seeds(tmp_path)
    service = RecordingYouTubeService(empty=True)

    def run(number):
        try:
            return execute_discovery_query_stage(store, "srcVideo123", _query(number), provider=_provider(service), max_search_calls=6)
        except DiscoveryValidationError:
            return None

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(run, range(12)))

    assert len(service.attempts) == 6
    assert sum(result is not None for result in results) <= 3


def test_new_discovery_run_has_independent_budget(tmp_path):
    store = _store_with_seeds(tmp_path)
    service = RecordingYouTubeService(empty=True)
    first = execute_discovery_query_stage(store, "srcVideo123", _query(), provider=_provider(service), max_search_calls=2)
    _store_with_seeds(tmp_path, run_id="dr-2")
    second = execute_discovery_query_stage(store, "srcVideo123", _query(), provider=_provider(service), max_search_calls=2)

    assert first["quota_usage"]["used_search_calls"] == second["quota_usage"]["used_search_calls"] == 2
    assert len(service.attempts) == 4
