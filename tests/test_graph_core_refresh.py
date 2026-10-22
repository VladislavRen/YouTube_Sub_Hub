"""Core graph refresh regressions at the Neo4j transaction boundary; no live DB."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from test_cache_freshness import VIDEO_ID, URL, _analysis, _live, _seed
from yt_sub_hub.graph import GraphPayload, Neo4jGraphStore
from yt_sub_hub.pipeline import VideoProcessingPipeline
from yt_sub_hub.runtime import RuntimeStore
from yt_sub_hub.stage_pipeline import StagePipeline


class MemoryTransaction:
    """Model the delete/merge subset used by GraphStore, retaining transaction isolation."""
    def __init__(self, state, *, fail_relationship=None):
        self.state = state
        self.fail_relationship = fail_relationship

    def run(self, query, **params):
        if query.startswith("CREATE "):
            return
        if "DELETE r" in query:
            for key, edge in list(self.state["relationships"].items()):
                if edge.get("source_video_id") != params["video_id"]:
                    continue
                if "r.research_run_id IS NULL" in query and edge.get("research_run_id") is not None:
                    continue
                if "r.discovery_run_id IS NULL" in query and edge.get("discovery_run_id") is not None:
                    continue
                del self.state["relationships"][key]
            return
        if query.startswith("MERGE (n:"):
            self.state["nodes"].setdefault(params["id"], {}).update(params["props"])
            return
        if "MERGE (a)-[r:" in query:
            if params["rel_id"] == self.fail_relationship:
                raise RuntimeError("simulated Neo4j write failure")
            self.state["relationships"].setdefault(params["rel_id"], {}).update(params["props"])
            return
        raise AssertionError(f"Unexpected Cypher: {query}")


class MemorySession:
    def __init__(self, driver):
        self.driver = driver

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def run(self, query, **params):
        return MemoryTransaction(self.driver.state).run(query, **params)

    def execute_write(self, callback, *args):
        pending = deepcopy(self.driver.state)
        tx = MemoryTransaction(pending, fail_relationship=self.driver.fail_relationship)
        result = callback(tx, *args)
        self.driver.state = pending
        return result


def _graph_store():
    class Driver:
        fail_relationship = None

        def __init__(self):
            self.state = {
                "nodes": {f"video:{VIDEO_ID}": {"id": f"video:{VIDEO_ID}", "title": "old title"}},
                "relationships": {
                    "old-core": {"id": "old-core", "source_video_id": VIDEO_ID},
                    "research": {"id": "research", "source_video_id": VIDEO_ID, "research_run_id": "research-1"},
                    "discovery": {"id": "discovery", "source_video_id": VIDEO_ID, "discovery_run_id": "discovery-1"},
                    "both": {"id": "both", "source_video_id": VIDEO_ID, "research_run_id": "research-1", "discovery_run_id": "discovery-1"},
                    "empty-owner": {"id": "empty-owner", "source_video_id": VIDEO_ID, "research_run_id": ""},
                    "other-video": {"id": "other-video", "source_video_id": "other123456"},
                },
            }

        def session(self):
            return MemorySession(self)

        def close(self):
            pass

    store = object.__new__(Neo4jGraphStore)
    store._driver = Driver()
    return store


def _core_payload():
    return GraphPayload(
        nodes=[
            {"id": f"video:{VIDEO_ID}", "label": "Video", "title": "updated title"},
            {"id": "concept:new", "label": "Concept", "name": "New"},
        ],
        relationships=[
            {"id": "new-core", "from": f"video:{VIDEO_ID}", "to": "concept:new", "type": "MENTIONS", "source_video_id": VIDEO_ID},
            {"id": "last-core", "from": f"video:{VIDEO_ID}", "to": "concept:new", "type": "EXPLAINS", "source_video_id": VIDEO_ID},
        ],
    )


def test_atomic_core_replacement_preserves_research_discovery_and_other_video_edges():
    store = _graph_store()
    protected = {key: deepcopy(value) for key, value in store._driver.state["relationships"].items() if key != "old-core"}

    store.replace_video_core_payload(VIDEO_ID, _core_payload())

    relationships = store._driver.state["relationships"]
    assert "old-core" not in relationships
    assert {"new-core", "last-core"} <= relationships.keys()
    assert {key: relationships[key] for key in protected} == protected
    assert store._driver.state["nodes"][f"video:{VIDEO_ID}"]["title"] == "updated title"


def test_failed_core_replacement_rolls_back_deletion_nodes_and_partial_relationship_writes():
    store = _graph_store()
    original = deepcopy(store._driver.state)
    store._driver.fail_relationship = "last-core"

    with pytest.raises(RuntimeError, match="simulated Neo4j write failure"):
        store.replace_video_core_payload(VIDEO_ID, _core_payload())

    assert store._driver.state == original


@pytest.mark.parametrize("mode", ["direct", "staged"])
def test_pipeline_core_sync_preserves_existing_enrichment(tmp_path, monkeypatch, mode):
    graph = _graph_store()
    protected = {key: deepcopy(value) for key, value in graph._driver.state["relationships"].items() if key != "old-core"}
    pipeline = VideoProcessingPipeline.for_test(tmp_path, graph_store=graph if mode == "direct" else None)
    _seed(pipeline.store)
    _live(monkeypatch)
    pipeline.analyzer = SimpleNamespace(analyze=lambda *args: _analysis("new analysis"))
    pipeline.process(URL, refresh=True)
    if mode == "staged":
        settings = replace(pipeline.settings, enable_graph=True)
        staged = StagePipeline(settings=settings)
        runtime = RuntimeStore(tmp_path / "index.sqlite3", settings=settings)
        job = runtime.create_video_job(job_id="staged", video_id=VIDEO_ID, canonical_url=URL)
        runtime.ensure_stage_plan(job.id)
        stage = next(item for item in runtime.list_stage_plan(job.id) if item.stage_name == "neo4j_sync")
        monkeypatch.setattr("yt_sub_hub.stage_pipeline.Neo4jGraphStore", lambda *args: graph)
        result = staged.run(staged.context(job, stage, runtime=runtime))
        assert result.status == "completed"

    relationships = graph._driver.state["relationships"]
    assert "old-core" not in relationships
    assert {key: relationships[key] for key in protected} == protected
    assert len(relationships) > len(protected)
