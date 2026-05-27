"""Integrity statuses in the DB are terminal but never counted as votes/misses."""

from nelson.db import Database


def _one_job_db(tmp_path):
    db = Database(tmp_path / "t.db")
    scan_id = db.create_scan("/x", commit_sha="abc")
    db.create_jobs_batch(scan_id, [("a.py", "CWE-89", "claude-haiku")])
    job = db.next_pending_job(scan_id)
    assert job is not None
    db.claim_job(job["id"])
    return db, scan_id, job["id"]


def test_mark_auth_failed_sets_status_and_excludes_from_coverage(tmp_path):
    db, scan_id, job_id = _one_job_db(tmp_path)

    db.mark_job_auth_failed(job_id, "Invalid API key")

    counts = db.job_counts(scan_id)
    assert counts == {"auth_failed": 1}

    # coverage counts only 'complete' jobs as a model having looked → auth_failed
    # must not register as a vote (and therefore can't become a miss).
    focused, open_ = db.coverage_for_scans([scan_id])
    assert focused == {}
    assert open_ == {}

    row = db.get_scan(scan_id)  # scan row still retrievable
    assert row is not None
    db.close()


def test_mark_infra_error_sets_status_and_excludes_from_coverage(tmp_path):
    db, scan_id, job_id = _one_job_db(tmp_path)

    db.mark_job_infra_error(job_id, "connection error")

    assert db.job_counts(scan_id) == {"infra_error": 1}
    focused, open_ = db.coverage_for_scans([scan_id])
    assert focused == {}
    assert open_ == {}
    db.close()


def test_complete_job_is_counted_as_coverage(tmp_path):
    # Contrast: a genuinely completed job DOES count as the model having looked.
    db, scan_id, job_id = _one_job_db(tmp_path)
    db.complete_job(job_id, tokens_in=1, tokens_out=1, cost_usd=0.0)

    focused, _ = db.coverage_for_scans([scan_id])
    assert focused == {("a.py", "CWE-89"): {"claude-haiku"}}
    db.close()
