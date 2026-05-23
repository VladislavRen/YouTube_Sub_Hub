import sqlite3
import time

from yt_sub_hub.library import LibraryStore
from yt_sub_hub.models import JobRecord


def test_library_store_persists_jobs_in_sqlite_with_stage_history(tmp_path):
    store = LibraryStore(tmp_path)
    record = JobRecord(job_id="job1", url="https://youtu.be/abc123", video_id="abc123", status="queued")

    store.save_job(record)
    store.update_job_stage("job1", status="running", stage="metadata_fetched")

    loaded = store.load_jobs()["job1"]
    history = store.load_job_history("job1")
    assert (tmp_path / "index.sqlite3").exists()
    assert loaded.status == "running"
    assert loaded.stage == "metadata_fetched"
    assert history[-1]["stage"] == "metadata_fetched"


def test_mark_stale_running_jobs_failed_marks_only_old_running_jobs(tmp_path):
    store = LibraryStore(tmp_path)
    records = [
        JobRecord(job_id="old-running", url="https://youtu.be/oldrun", video_id="oldrun", status="running", stage="running"),
        JobRecord(job_id="fresh-running", url="https://youtu.be/freshrun", video_id="freshrun", status="running", stage="running"),
        JobRecord(job_id="old-queued", url="https://youtu.be/oldqueue", video_id="oldqueue", status="queued", stage="queued"),
        JobRecord(job_id="old-completed", url="https://youtu.be/oldcomplete", video_id="oldcomplete", status="completed", stage="graph_synced"),
    ]
    for record in records:
        store.save_job(record)
    old_time = time.time() - 7200
    with sqlite3.connect(store.db_path) as conn:
        conn.execute("UPDATE jobs SET updated_at = ? WHERE job_id != ?", (old_time, "fresh-running"))

    recovered = store.mark_stale_running_jobs_failed(max_age_seconds=3600)
    jobs = store.load_jobs()
    history = store.load_job_history("old-running")

    assert [record.job_id for record in recovered] == ["old-running"]
    assert jobs["old-running"].status == "failed"
    assert jobs["old-running"].stage == "failed"
    assert jobs["old-running"].error == "Stale running job marked failed during recovery"
    assert jobs["fresh-running"].status == "running"
    assert jobs["old-queued"].status == "queued"
    assert jobs["old-completed"].status == "completed"
    assert history[-1]["message"] == "Stale running job marked failed during recovery"


def test_mark_stale_running_jobs_failed_keeps_non_stale_running_job(tmp_path):
    store = LibraryStore(tmp_path)
    store.save_job(
        JobRecord(job_id="fresh-running", url="https://youtu.be/freshrun", video_id="freshrun", status="running", stage="running")
    )

    recovered = store.mark_stale_running_jobs_failed(max_age_seconds=3600)

    assert recovered == []
    assert store.load_jobs()["fresh-running"].status == "running"
