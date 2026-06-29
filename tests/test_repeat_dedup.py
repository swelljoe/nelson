"""Repeat (--repeat / jobs.pass) and review-time de-duplication."""

import sqlite3
from collections import defaultdict

from nelson.db import Database
from nelson.inventory import SourceFile
from nelson.scanner import build_job_matrix


def test_build_job_matrix_repeat_emits_passes():
    files = [SourceFile("a.py", "python", 10), SourceFile("b.py", "python", 20)]
    models = ["claude:haiku", "claude:sonnet"]
    jobs = build_job_matrix(files, models, repeat=3)

    assert len(jobs) == 2 * 2 * 3
    assert all(cwe == "OPEN" for _f, cwe, _m, _p in jobs)

    by_fm: dict[tuple[str, str], set[int]] = defaultdict(set)
    for f, _cwe, m, p in jobs:
        by_fm[(f, m)].add(p)
    assert len(by_fm) == 4  # 2 files x 2 models
    assert all(passes == {0, 1, 2} for passes in by_fm.values())


def test_build_job_matrix_default_repeat_is_one():
    jobs = build_job_matrix([SourceFile("a.py", "python", 1)], ["claude:haiku"])
    assert len(jobs) == 1
    assert jobs[0][3] == 0  # pass


# -- migration 7: jobs gains `pass`, FKs survive the table rebuild -----------

_V6_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE scans (
    id INTEGER PRIMARY KEY, target_dir TEXT NOT NULL, commit_sha TEXT,
    started_at TEXT, completed_at TEXT, config TEXT
);
CREATE TABLE jobs (
    id INTEGER PRIMARY KEY,
    scan_id INTEGER NOT NULL REFERENCES scans(id),
    file_path TEXT NOT NULL, cwe_id TEXT NOT NULL, model_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    started_at TEXT, completed_at TEXT,
    tokens_in INTEGER, tokens_out INTEGER, cost_usd REAL, error_msg TEXT,
    UNIQUE(scan_id, file_path, cwe_id, model_id)
);
CREATE TABLE findings (
    id INTEGER PRIMARY KEY,
    job_id INTEGER NOT NULL REFERENCES jobs(id),
    line_number INTEGER, code_snippet TEXT, explanation TEXT, confidence TEXT,
    verified_by_model TEXT, verified_status TEXT, suggested_fix TEXT
);
"""


def _make_v6_db(path):
    conn = sqlite3.connect(str(path))
    conn.executescript(_V6_SCHEMA)
    conn.execute("INSERT INTO scans(id, target_dir) VALUES(1, '/x')")
    conn.execute(
        "INSERT INTO jobs(id, scan_id, file_path, cwe_id, model_id, status) "
        "VALUES(10, 1, 'a.py', 'OPEN', 'm', 'complete')"
    )
    conn.execute(
        "INSERT INTO findings(id, job_id, line_number, explanation) "
        "VALUES(100, 10, 5, 'boom')"
    )
    conn.execute("INSERT INTO meta(key, value) VALUES('schema_version', '6')")
    conn.commit()
    conn.close()


def test_migration_v6_to_v7_adds_pass_and_preserves_fks(tmp_path):
    p = tmp_path / "old.db"
    _make_v6_db(p)

    db = Database(p)  # opening runs the migration up to v7

    # legacy row keeps its id and gains pass=0
    row = db.conn.execute("SELECT id, pass FROM jobs WHERE id = 10").fetchone()
    assert row["pass"] == 0

    # findings.job_id FK still resolves to the same (preserved-id) job
    joined = db.conn.execute(
        "SELECT j.file_path FROM findings f JOIN jobs j ON f.job_id = j.id "
        "WHERE f.id = 100"
    ).fetchone()
    assert joined["file_path"] == "a.py"

    # the widened UNIQUE now admits repeated passes of the same (file, model)
    db.create_jobs_batch(1, [("a.py", "OPEN", "m", 1), ("a.py", "OPEN", "m", 2)])
    n = db.conn.execute(
        "SELECT COUNT(*) AS c FROM jobs WHERE file_path = 'a.py'"
    ).fetchone()["c"]
    assert n == 3
    db.close()


# -- dedup: one judge call per cluster, verdict propagated to all members ----


class _CountingAdapter:
    """Fake reviewer: returns a fixed verdict and counts how often it's called."""

    name = "fake-judge"
    needs_pacing = False

    def __init__(self):
        self.calls = 0

    def run(self, prompt, cancel_event=None):
        from nelson.agents import AgentResult

        self.calls += 1
        return AgentResult(
            findings=[],
            raw_output='{"verdict": "confirmed", "reasoning": "r", "severity": "high"}',
        )


def test_review_dedups_cluster_judges_once_propagates(tmp_path, monkeypatch):
    from nelson import review as review_mod

    (tmp_path / "a.py").write_text("line1\nrun(user)\nline3\n")
    db = Database(tmp_path / "t.db")
    scan_id = db.create_scan(str(tmp_path), config={"models": ["m1", "m2"]})

    # Same bug at the same line found by 2 models across 2 passes -> 4 findings,
    # one cluster.
    finding_ids = []
    for model in ("m1", "m2"):
        for p in (0, 1):
            db.create_jobs_batch(scan_id, [("a.py", "OPEN", model, p)])
            job = db.next_pending_job(scan_id, model_id=model)
            db.claim_job(job["id"])
            db.complete_job(job["id"])
            finding_ids.append(
                db.add_finding(
                    job["id"],
                    line_number=2,
                    code_snippet="run(user)",
                    explanation="[CWE-78] command injection",
                    confidence="high",
                )
            )

    adapter = _CountingAdapter()
    monkeypatch.setattr(review_mod, "create_adapter", lambda *a, **k: adapter)

    review_mod.run_review(db, scan_id, "fake:judge", str(tmp_path), delay=0)

    # exactly one judge call for the whole cluster...
    assert adapter.calls == 1
    # ...and every member row carries the propagated verdict.
    ph = ",".join("?" * len(finding_ids))
    rows = db.conn.execute(
        f"SELECT verified_status FROM findings WHERE id IN ({ph})",  # noqa: S608
        finding_ids,
    ).fetchall()
    assert [r["verified_status"] for r in rows] == ["confirmed"] * 4
    db.close()


def test_review_separate_bugs_judged_separately(tmp_path, monkeypatch):
    from nelson import review as review_mod

    (tmp_path / "a.py").write_text("\n".join(f"line{i}" for i in range(40)))
    db = Database(tmp_path / "t.db")
    scan_id = db.create_scan(str(tmp_path), config={"models": ["m1"]})

    # Two findings far apart in the same file -> two clusters -> two judge calls.
    for line in (2, 30):
        db.create_jobs_batch(scan_id, [("a.py", "OPEN", "m1", line)])
        job = db.next_pending_job(scan_id, model_id="m1")
        db.claim_job(job["id"])
        db.complete_job(job["id"])
        db.add_finding(
            job["id"],
            line_number=line,
            explanation="[CWE-78] injection",
            confidence="high",
        )

    adapter = _CountingAdapter()
    monkeypatch.setattr(review_mod, "create_adapter", lambda *a, **k: adapter)
    review_mod.run_review(db, scan_id, "fake:judge", str(tmp_path), delay=0)
    assert adapter.calls == 2
    db.close()
