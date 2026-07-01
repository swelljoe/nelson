"""Tests for the `bench status` dashboard command."""

from click.testing import CliRunner

from nelson.cli import main
from nelson.corpus import Case
from nelson.db import Database
from nelson.runner import Competitor


def _case(ext_id, *, files=("Foo.java",)):
    return Case(
        source="cvd",
        ext_id=ext_id,
        status="vetted",
        repo_url="https://example.invalid/r",
        vuln_commit="deadbeef",
        fix_commit="cafef00d",
        gt_files=list(files),
        gt_hunks=[{"file": f, "start": 73, "end": 79} for f in files],
    )


def _seed(tmp_path):
    """One active + one retired competitor, one 2-trial matrix, mixed statuses."""
    db = Database(tmp_path / "t.db")
    case_id = db.upsert_case(_case("CVE-1").to_db_fields())
    active = db.upsert_competitor(
        Competitor(name="raw/active", model="m").to_db_fields()
    )
    retired_fields = Competitor(name="raw/retired", model="m").to_db_fields()
    retired_fields["status"] = "retired"
    retired = db.upsert_competitor(retired_fields)
    # active: trial 0 complete, trial 1 running.
    r0 = db.create_run(case_id, active, "Foo.java", 0)
    db.start_run(r0)
    db.complete_run(r0, cost_usd=0.1, tokens_out=100)
    r1 = db.create_run(case_id, active, "Foo.java", 1)
    db.start_run(r1)
    # retired: an orphaned running row (killed loop left it 'running').
    r2 = db.create_run(case_id, retired, "Foo.java", 0)
    db.start_run(r2)
    return db


def test_status_reports_matrix_completion_and_running(tmp_path):
    _seed(tmp_path).close()
    res = CliRunner().invoke(
        main,
        ["bench", "status", "--db", str(tmp_path / "t.db"), "--repeat", "2"],
    )
    assert res.exit_code == 0, res.output
    # One active competitor x one case x one file x 2 trials = 2 cells, 1 done.
    assert "cells 1/2" in res.output
    assert "raw/active" in res.output
    # The active running trial and the retired orphan both appear as running.
    assert "running now (2)" in res.output
    assert "orphan" in res.output  # retired competitor's stale running row flagged


def test_status_empty_db_is_clean(tmp_path):
    Database(tmp_path / "empty.db").close()
    res = CliRunner().invoke(
        main, ["bench", "status", "--db", str(tmp_path / "empty.db")]
    )
    assert res.exit_code == 0, res.output
    assert "cells 0/0" in res.output
