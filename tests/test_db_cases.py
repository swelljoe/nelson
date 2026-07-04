"""Schema migration path and the `cases` table CRUD.

Covers two things the corpus depends on: (1) an existing pre-corpus scan DB
upgrades in place to gain the `cases` table, and (2) case rows round-trip,
including partial enrichment/vet updates that must not clobber other columns.
"""

import json
import sqlite3

import pytest

from nelson.db import SCHEMA, SCHEMA_VERSION, Database


def _table_exists(conn, name):
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )


def test_fresh_db_is_at_current_version_with_cases(tmp_path):
    db = Database(tmp_path / "fresh.db")
    ver = db.conn.execute(
        "SELECT value FROM meta WHERE key='schema_version'"
    ).fetchone()["value"]
    assert int(ver) == SCHEMA_VERSION
    assert _table_exists(db.conn, "cases")
    db.close()


def test_v1_database_upgrades_in_place(tmp_path):
    # Simulate a pre-corpus database: base schema + recorded version 1, no cases.
    path = tmp_path / "old.db"
    raw = sqlite3.connect(str(path))
    raw.executescript(SCHEMA)
    raw.execute("INSERT INTO meta(key, value) VALUES('schema_version', '1')")
    raw.commit()
    assert not _table_exists(raw, "cases")
    raw.close()

    # Opening it through Database runs the migration.
    db = Database(path)
    assert _table_exists(db.conn, "cases")
    ver = db.conn.execute(
        "SELECT value FROM meta WHERE key='schema_version'"
    ).fetchone()["value"]
    assert int(ver) == SCHEMA_VERSION
    db.close()


def test_v5_migration_is_idempotent_after_partial_apply(tmp_path):
    path = tmp_path / "partial-v5.db"
    db = Database(path)
    db.close()

    raw = sqlite3.connect(str(path))
    raw.execute("UPDATE meta SET value = '4' WHERE key = 'schema_version'")
    raw.commit()
    raw.close()

    reopened = Database(path)
    cols = {row["name"] for row in reopened.conn.execute("PRAGMA table_info(runs)")}
    assert "target_file" in cols
    ver = reopened.conn.execute(
        "SELECT value FROM meta WHERE key='schema_version'"
    ).fetchone()["value"]
    assert int(ver) == SCHEMA_VERSION
    reopened.close()


def _seed(ext_id="CVE-2026-27654"):
    return {
        "source": "cvd",
        "ext_id": ext_id,
        "ant_id": "ANT-2026-HY56VRSB",
        "project": "nginx",
        "bug_class": "heap-buffer-overflow",
        "severity": "high",
        "description": "Heap buffer overflow in DAV COPY/MOVE",
        "disclosure_date": "2026-05-20T07:40:37Z",
    }


def test_upsert_insert_then_update_is_idempotent(tmp_path):
    db = Database(tmp_path / "c.db")
    cid = db.upsert_case(_seed())
    assert cid > 0
    # Re-importing the same seed updates in place (same id, no duplicate row).
    cid2 = db.upsert_case({**_seed(), "severity": "critical"})
    assert cid2 == cid
    assert len(db.list_cases()) == 1
    row = db.get_case("CVE-2026-27654")
    assert row is not None
    assert row["severity"] == "critical"
    db.close()


def test_ground_truth_json_columns_round_trip(tmp_path):
    db = Database(tmp_path / "c.db")
    db.upsert_case(_seed())
    files = ["src/http/ngx_http.c", "src/core/ngx_string.c"]
    hunks = [{"file": "src/http/ngx_http.c", "start": 120, "end": 138}]
    db.update_case_fields(
        "CVE-2026-27654", {"gt_files": files, "gt_hunks": hunks, "fix_commit": "abc123"}
    )
    row = db.get_case("CVE-2026-27654")
    assert row is not None
    assert json.loads(row["gt_files"]) == files
    assert json.loads(row["gt_hunks"]) == hunks
    assert row["fix_commit"] == "abc123"
    # Partial update left the seed columns untouched.
    assert row["project"] == "nginx"
    db.close()


def test_verification_json_round_trip(tmp_path):
    db = Database(tmp_path / "c.db")
    db.upsert_case(_seed())
    verification = {
        "invariant": "unsafe input is rejected",
        "witnesses": [{"command": ["pytest", "test_security.py"]}],
    }
    db.update_case_fields("CVE-2026-27654", {"verification": verification})
    row = db.get_case("CVE-2026-27654")
    assert row is not None
    assert json.loads(row["verification"]) == verification
    db.close()


def test_update_missing_case_is_noop(tmp_path):
    db = Database(tmp_path / "c.db")
    db.update_case_fields("GHSA-does-not-exist", {"cwe": "CWE-787"})
    assert db.get_case("GHSA-does-not-exist") is None
    db.close()


def test_list_and_count_filter_by_status_and_source(tmp_path):
    db = Database(tmp_path / "c.db")
    db.upsert_case(_seed("CVE-1"))
    db.upsert_case({**_seed("CVE-2"), "status": "vetted"})
    db.upsert_case({**_seed("GHSA-x"), "source": "manual", "status": "vetted"})

    assert {r["ext_id"] for r in db.list_cases(status="vetted")} == {"CVE-2", "GHSA-x"}
    assert {r["ext_id"] for r in db.list_cases(source="manual")} == {"GHSA-x"}
    assert db.count_cases_by_status() == {"candidate": 1, "vetted": 2}
    db.close()


def test_upsert_requires_natural_key_and_rejects_unknown_columns(tmp_path):
    db = Database(tmp_path / "c.db")
    with pytest.raises(ValueError, match="source"):
        db.upsert_case({"ext_id": "CVE-9"})
    with pytest.raises(ValueError, match="unknown case column"):
        db.upsert_case({**_seed(), "not_a_column": 1})
    db.close()
