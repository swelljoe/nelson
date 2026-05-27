"""SQLite database for scan state tracking."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Bumped to 2 for the benchmark corpus (`cases` table). New tables/columns go in
# MIGRATIONS keyed by their target version; _apply_migrations walks a stored DB
# up to SCHEMA_VERSION, so old scan databases upgrade in place.
SCHEMA_VERSION = 2

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
    -- pending, running, complete, skipped, error (genuine task error, e.g.
    -- unreadable file), and the integrity statuses auth_failed / infra_error.
    -- The integrity statuses mean the model never got a fair look at the code,
    -- so scoring must never count them as a "miss".
    status TEXT NOT NULL DEFAULT 'pending',
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

# Schema migrations, keyed by the version they bring the DB *to*. Each is run
# once, in order, on a database whose stored version is lower. Statements must be
# safe to re-run (IF NOT EXISTS) so a half-applied migration recovers cleanly.
MIGRATIONS: dict[int, str] = {
    2: """
    -- Benchmark corpus + ground truth. One row per upstream advisory
    -- (ext_id = the seed identifier, CVE-… or GHSA-…). A case moves
    -- candidate -> vetted (Opus pre-vet) -> retired (derivation unusable).
    CREATE TABLE IF NOT EXISTS cases (
        id INTEGER PRIMARY KEY,
        source TEXT NOT NULL,                 -- cvd / osv / manual
        ext_id TEXT NOT NULL UNIQUE,          -- seed identifier (CVE-… / GHSA-…)
        cve_id TEXT,
        ghsa_id TEXT,
        ant_id TEXT,                          -- Anthropic finding id(s)
        project TEXT,
        repo_url TEXT,
        vuln_commit TEXT,                     -- pre-patch SHA (= fix_commit parent)
        fix_commit TEXT,
        bug_class TEXT,
        cwe TEXT,
        disclosure_date TEXT,
        severity TEXT,
        description TEXT,
        gt_files TEXT,                        -- JSON: list[str]
        gt_hunks TEXT,                        -- JSON: list[{file,start,end}]
        build_recipe TEXT,
        status TEXT NOT NULL DEFAULT 'candidate',  -- candidate / vetted / retired
        vet_confidence REAL,
        vet_notes TEXT,
        added_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
        updated_at TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status);
    CREATE INDEX IF NOT EXISTS idx_cases_source ON cases(source);
    """,
}

# Columns a caller may set on a case. Used to validate dict keys before they reach
# SQL (column names can't be parameterized, so we never interpolate arbitrary ones).
CASE_COLUMNS: tuple[str, ...] = (
    "source",
    "ext_id",
    "cve_id",
    "ghsa_id",
    "ant_id",
    "project",
    "repo_url",
    "vuln_commit",
    "fix_commit",
    "bug_class",
    "cwe",
    "disclosure_date",
    "severity",
    "description",
    "gt_files",
    "gt_hunks",
    "build_recipe",
    "status",
    "vet_confidence",
    "vet_notes",
)
# These hold structured data; lists/dicts are JSON-encoded on the way in.
_CASE_JSON_COLUMNS = frozenset({"gt_files", "gt_hunks"})


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _encode_case_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """Validate column names and JSON-encode the structured columns.

    Raises ValueError on an unknown column so a typo fails loudly here rather
    than silently dropping ground-truth data.
    """
    unknown = set(fields) - set(CASE_COLUMNS)
    if unknown:
        raise ValueError(f"unknown case column(s): {', '.join(sorted(unknown))}")
    out: dict[str, Any] = {}
    for k, v in fields.items():
        if k in _CASE_JSON_COLUMNS and not isinstance(v, str | None):
            out[k] = json.dumps(v)
        else:
            out[k] = v
    return out


class Database:
    def __init__(self, path: str | Path = "nelson.db"):
        self.path = Path(path)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._apply_migrations()
        self.conn.commit()

    def _apply_migrations(self) -> None:
        """Walk the DB from its stored schema version up to SCHEMA_VERSION.

        A database with no recorded version is treated as the base schema
        (version 1), so a pre-corpus scan DB picks up the `cases` table the
        first time it is opened by this code.
        """
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        current = int(row["value"]) if row else 1
        for version in range(current + 1, SCHEMA_VERSION + 1):
            script = MIGRATIONS.get(version)
            if script:
                self.conn.executescript(script)
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(SCHEMA_VERSION),),
        )

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

    def mark_job_auth_failed(self, job_id: int, error_msg: str):
        """Terminal: the model could not authenticate / was not reachable due to
        credentials. Integrity rule: this is NOT a miss — it is excluded from the
        scoring denominator (coverage_for_scans only counts 'complete')."""
        self.conn.execute(
            "UPDATE jobs SET status = 'auth_failed', completed_at = ?, error_msg = ? WHERE id = ?",
            (_now(), error_msg, job_id),
        )
        self.conn.commit()

    def mark_job_infra_error(self, job_id: int, error_msg: str):
        """Terminal: infrastructure/transport failure (timeout, connection,
        unexplained non-zero exit). Like auth_failed, never counted as a miss."""
        self.conn.execute(
            "UPDATE jobs SET status = 'infra_error', completed_at = ?, error_msg = ? WHERE id = ?",
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

    def findings_for_scans(self, scan_ids: list[int]) -> list[sqlite3.Row]:
        """Like findings_for_scan but across multiple scans."""
        if not scan_ids:
            return []
        # ph is just `?, ?, ?` placeholders; values are bound below. sqlite3
        # has no native list binding for IN clauses.
        ph = ",".join("?" * len(scan_ids))
        return self.conn.execute(
            f"""SELECT f.*, j.file_path, j.cwe_id, j.model_id, j.scan_id
               FROM findings f
               JOIN jobs j ON f.job_id = j.id
               WHERE j.scan_id IN ({ph})
               ORDER BY j.file_path, f.line_number""",  # noqa: S608
            scan_ids,
        ).fetchall()

    def coverage_for_scans(
        self, scan_ids: list[int]
    ) -> tuple[dict[tuple[str, str], set[str]], dict[str, set[str]]]:
        """Distinct models that ran a *completed* job for each (file, cwe).

        Returns (focused, open) where focused maps (file, cwe) -> {model_id}
        and open maps file -> {model_id} for OPEN-mode jobs. A model that
        errored out on a job isn't counted as having 'voted no'.
        """
        if not scan_ids:
            return {}, {}
        ph = ",".join("?" * len(scan_ids))
        rows = self.conn.execute(
            f"""SELECT DISTINCT file_path, cwe_id, model_id
               FROM jobs
               WHERE scan_id IN ({ph}) AND status = 'complete'""",  # noqa: S608
            scan_ids,
        ).fetchall()
        focused: dict[tuple[str, str], set[str]] = {}
        open_: dict[str, set[str]] = {}
        for r in rows:
            if r["cwe_id"] == "OPEN":
                open_.setdefault(r["file_path"], set()).add(r["model_id"])
            else:
                focused.setdefault((r["file_path"], r["cwe_id"]), set()).add(
                    r["model_id"]
                )
        return focused, open_

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

    # -- Cases (benchmark corpus) --

    def upsert_case(self, fields: dict[str, Any]) -> int:
        """Insert a case, or update the existing row with the same ext_id.

        Requires at least ``source`` and ``ext_id`` (the natural key). Used by
        the seed importer, which always has both. On conflict, every *provided*
        column is overwritten — so re-importing a seed refreshes it without
        clobbering columns the import doesn't carry (enrichment, ground truth).
        """
        if "source" not in fields or "ext_id" not in fields:
            raise ValueError("upsert_case requires 'source' and 'ext_id'")
        data = _encode_case_fields(fields)
        cols = list(data)
        placeholders = ", ".join("?" for _ in cols)
        # On conflict, refresh the non-key columns the caller supplied.
        updatable = [c for c in cols if c != "ext_id"]
        set_clause = ", ".join(f"{c} = excluded.{c}" for c in updatable)
        set_clause = f"{set_clause}, updated_at = excluded.updated_at"
        sql = (
            f"INSERT INTO cases ({', '.join(cols)}, updated_at) "  # noqa: S608
            f"VALUES ({placeholders}, ?) "
            f"ON CONFLICT(ext_id) DO UPDATE SET {set_clause}"
        )
        self.conn.execute(sql, (*[data[c] for c in cols], _now()))
        self.conn.commit()
        # lastrowid is only the new id on insert; resolve via ext_id to be safe.
        row = self.conn.execute(
            "SELECT id FROM cases WHERE ext_id = ?", (fields["ext_id"],)
        ).fetchone()
        return row["id"]

    def update_case_fields(self, ext_id: str, fields: dict[str, Any]) -> None:
        """Partial update of an existing case (enrichment / derivation / vet).

        Unlike :meth:`upsert_case` this never inserts: a missing case is a no-op,
        and only the named columns change. ``ext_id`` itself cannot be updated.
        """
        data = _encode_case_fields(fields)
        data.pop("ext_id", None)
        if not data:
            return
        cols = list(data)
        set_clause = ", ".join(f"{c} = ?" for c in cols)
        self.conn.execute(
            f"UPDATE cases SET {set_clause}, updated_at = ? WHERE ext_id = ?",  # noqa: S608
            (*[data[c] for c in cols], _now(), ext_id),
        )
        self.conn.commit()

    def get_case(self, ext_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM cases WHERE ext_id = ?", (ext_id,)
        ).fetchone()

    def list_cases(
        self, status: str | None = None, source: str | None = None
    ) -> list[sqlite3.Row]:
        sql = "SELECT * FROM cases"
        clauses, params = [], []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY ext_id"
        return self.conn.execute(sql, params).fetchall()

    def count_cases_by_status(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT status, COUNT(*) AS cnt FROM cases GROUP BY status"
        ).fetchall()
        return {r["status"]: r["cnt"] for r in rows}
