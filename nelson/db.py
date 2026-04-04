"""SQLite database for scan state tracking."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 1

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS scans (
    id INTEGER PRIMARY KEY,
    target_dir TEXT NOT NULL,
    commit_sha TEXT,
    started_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    completed_at TEXT,
    config TEXT  -- JSON: model list, CWE set, filters
);

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY,
    scan_id INTEGER NOT NULL REFERENCES scans(id),
    file_path TEXT NOT NULL,
    cwe_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending, running, complete, error, skipped
    started_at TEXT,
    completed_at TEXT,
    tokens_in INTEGER,
    tokens_out INTEGER,
    cost_usd REAL,
    error_msg TEXT,
    UNIQUE(scan_id, file_path, cwe_id, model_id)
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(scan_id, status);
CREATE INDEX IF NOT EXISTS idx_jobs_file ON jobs(scan_id, file_path);

CREATE TABLE IF NOT EXISTS findings (
    id INTEGER PRIMARY KEY,
    job_id INTEGER NOT NULL REFERENCES jobs(id),
    line_number INTEGER,
    code_snippet TEXT,
    explanation TEXT,
    confidence TEXT,  -- high, medium, low
    verified_by_model TEXT,
    verified_status TEXT,  -- confirmed, false_positive, needs_review
    suggested_fix TEXT
);
"""


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class Database:
    def __init__(self, path: str | Path = "nelson.db"):
        self.path = Path(path)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        # Set schema version if not present
        self.conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self.conn.commit()

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- Scans --

    def create_scan(
        self, target_dir: str, commit_sha: str | None = None, config: dict | None = None
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO scans(target_dir, commit_sha, config) VALUES(?, ?, ?)",
            (target_dir, commit_sha, json.dumps(config) if config else None),
        )
        self.conn.commit()
        assert cur.lastrowid is not None
        return cur.lastrowid

    def complete_scan(self, scan_id: int):
        self.conn.execute(
            "UPDATE scans SET completed_at = ? WHERE id = ?", (_now(), scan_id)
        )
        self.conn.commit()

    def get_scan(self, scan_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM scans WHERE id = ?", (scan_id,)
        ).fetchone()

    def latest_scan(self) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM scans ORDER BY id DESC LIMIT 1"
        ).fetchone()

    def list_scans(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM scans ORDER BY id DESC").fetchall()

    # -- Jobs --

    def create_jobs_batch(self, scan_id: int, jobs: list[tuple[str, str, str]]):
        """Insert a batch of (file_path, cwe_id, model_id) jobs."""
        self.conn.executemany(
            "INSERT OR IGNORE INTO jobs(scan_id, file_path, cwe_id, model_id) VALUES(?, ?, ?, ?)",
            [(scan_id, f, c, m) for f, c, m in jobs],
        )
        self.conn.commit()

    def next_pending_job(
        self, scan_id: int, model_id: str | None = None
    ) -> sqlite3.Row | None:
        if model_id:
            return self.conn.execute(
                "SELECT * FROM jobs WHERE scan_id = ? AND status = 'pending' AND model_id = ? LIMIT 1",
                (scan_id, model_id),
            ).fetchone()
        return self.conn.execute(
            "SELECT * FROM jobs WHERE scan_id = ? AND status = 'pending' LIMIT 1",
            (scan_id,),
        ).fetchone()

    def claim_job(self, job_id: int) -> bool:
        """Atomically claim a pending job. Returns True if claimed."""
        cur = self.conn.execute(
            "UPDATE jobs SET status = 'running', started_at = ? WHERE id = ? AND status = 'pending'",
            (_now(), job_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def complete_job(
        self,
        job_id: int,
        *,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        cost_usd: float | None = None,
    ):
        self.conn.execute(
            """UPDATE jobs SET status = 'complete', completed_at = ?,
               tokens_in = ?, tokens_out = ?, cost_usd = ?
               WHERE id = ?""",
            (_now(), tokens_in, tokens_out, cost_usd, job_id),
        )
        self.conn.commit()

    def fail_job(self, job_id: int, error_msg: str):
        self.conn.execute(
            "UPDATE jobs SET status = 'error', completed_at = ?, error_msg = ? WHERE id = ?",
            (_now(), error_msg, job_id),
        )
        self.conn.commit()

    def job_counts(self, scan_id: int) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT status, COUNT(*) as cnt FROM jobs WHERE scan_id = ? GROUP BY status",
            (scan_id,),
        ).fetchall()
        return {r["status"]: r["cnt"] for r in rows}

    def usage_by_model(self, scan_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            """SELECT model_id,
                      COUNT(*) as jobs,
                      SUM(tokens_in) as total_tokens_in,
                      SUM(tokens_out) as total_tokens_out,
                      SUM(cost_usd) as total_cost_usd
               FROM jobs
               WHERE scan_id = ? AND status = 'complete'
               GROUP BY model_id""",
            (scan_id,),
        ).fetchall()

    # -- Findings --

    def add_finding(
        self,
        job_id: int,
        *,
        line_number: int | None = None,
        code_snippet: str | None = None,
        explanation: str | None = None,
        confidence: str | None = None,
    ) -> int:
        cur = self.conn.execute(
            """INSERT INTO findings(job_id, line_number, code_snippet, explanation, confidence)
               VALUES(?, ?, ?, ?, ?)""",
            (job_id, line_number, code_snippet, explanation, confidence),
        )
        self.conn.commit()
        assert cur.lastrowid is not None
        return cur.lastrowid

    def update_finding_review(
        self,
        finding_id: int,
        *,
        verified_by_model: str,
        verified_status: str,
        suggested_fix: str | None = None,
    ):
        self.conn.execute(
            """UPDATE findings SET verified_by_model = ?, verified_status = ?, suggested_fix = ?
               WHERE id = ?""",
            (verified_by_model, verified_status, suggested_fix, finding_id),
        )
        self.conn.commit()

    def unreviewed_findings(self, scan_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            """SELECT f.*, j.file_path, j.cwe_id, j.model_id, j.scan_id
               FROM findings f
               JOIN jobs j ON f.job_id = j.id
               WHERE j.scan_id = ? AND f.verified_status IS NULL
               ORDER BY j.file_path, f.line_number""",
            (scan_id,),
        ).fetchall()

    def findings_for_scan(self, scan_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            """SELECT f.*, j.file_path, j.cwe_id, j.model_id
               FROM findings f
               JOIN jobs j ON f.job_id = j.id
               WHERE j.scan_id = ?
               ORDER BY j.file_path, f.line_number""",
            (scan_id,),
        ).fetchall()

    def findings_summary(self, scan_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            """SELECT j.cwe_id, j.model_id, f.confidence,
                      COUNT(*) as count
               FROM findings f
               JOIN jobs j ON f.job_id = j.id
               WHERE j.scan_id = ?
               GROUP BY j.cwe_id, j.model_id, f.confidence
               ORDER BY count DESC""",
            (scan_id,),
        ).fetchall()

    def review_summary(self, scan_id: int) -> dict[str, int]:
        rows = self.conn.execute(
            """SELECT f.verified_status, COUNT(*) as cnt
               FROM findings f
               JOIN jobs j ON f.job_id = j.id
               WHERE j.scan_id = ?
               GROUP BY f.verified_status""",
            (scan_id,),
        ).fetchall()
        return {r["verified_status"] or "unreviewed": r["cnt"] for r in rows}
