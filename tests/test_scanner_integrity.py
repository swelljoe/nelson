"""End-to-end (at the job level): scanner routes failure kinds to DB statuses.

Proves the wiring from a runtime's AgentResult.failure_kind through to the
terminal job status, with no subprocess or network — the adapter is faked.
"""

import threading

from nelson.agents import AgentResult, FailureKind, Finding
from nelson.cwe import CWE_OPEN
from nelson.db import Database
from nelson.scanner import _process_one_job


class _CannedAdapter:
    """Minimal runtime stand-in returning a fixed AgentResult."""

    name = "fake-model"
    needs_pacing = False

    def __init__(self, result: AgentResult):
        self._result = result

    def run(self, prompt, cancel_event=None):
        return self._result


def _setup(tmp_path):
    (tmp_path / "a.py").write_text("import os\nos.system(input())\n")
    db = Database(tmp_path / "t.db")
    scan_id = db.create_scan(str(tmp_path))
    db.create_jobs_batch(scan_id, [("a.py", "OPEN", "fake-model")])
    job = db.next_pending_job(scan_id)
    assert job is not None
    db.claim_job(job["id"])
    return db, scan_id, job


def _status(db, job_id):
    return db.conn.execute(
        "SELECT status FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()["status"]


def test_auth_failure_marks_auth_failed_not_error(tmp_path):
    db, _scan, job = _setup(tmp_path)
    adapter = _CannedAdapter(
        AgentResult(
            findings=[], raw_output="", failure_kind=FailureKind.AUTH, error="bad key"
        )
    )

    rate_limited, ok = _process_one_job(
        db, job, adapter, tmp_path, {"OPEN": CWE_OPEN}, threading.Event()
    )

    assert (rate_limited, ok) == (False, False)
    assert _status(db, job["id"]) == "auth_failed"
    db.close()


def test_infra_failure_marks_infra_error(tmp_path):
    db, _scan, job = _setup(tmp_path)
    adapter = _CannedAdapter(
        AgentResult(
            findings=[], raw_output="", failure_kind=FailureKind.INFRA, error="timeout"
        )
    )

    _process_one_job(db, job, adapter, tmp_path, {"OPEN": CWE_OPEN}, threading.Event())

    assert _status(db, job["id"]) == "infra_error"
    db.close()


def test_rate_limit_signals_retry_and_leaves_job_running(tmp_path):
    db, _scan, job = _setup(tmp_path)
    adapter = _CannedAdapter(
        AgentResult(findings=[], raw_output="", failure_kind=FailureKind.RATE_LIMIT)
    )

    rate_limited, ok = _process_one_job(
        db, job, adapter, tmp_path, {"OPEN": CWE_OPEN}, threading.Event()
    )

    # Caller (the worker) is responsible for releasing+backing off; the job is
    # not marked terminal here.
    assert rate_limited is True
    assert ok is False
    assert _status(db, job["id"]) == "running"
    db.close()


def test_clean_run_completes_and_records_finding(tmp_path):
    db, scan_id, job = _setup(tmp_path)
    adapter = _CannedAdapter(
        AgentResult(
            findings=[
                Finding(
                    line_number=2,
                    code_snippet="os.system(input())",
                    explanation="command injection",
                    confidence="high",
                    cwe_id="CWE-78",
                )
            ],
            raw_output="[...]",
        )
    )

    _process_one_job(db, job, adapter, tmp_path, {"OPEN": CWE_OPEN}, threading.Event())

    assert _status(db, job["id"]) == "complete"
    findings = db.findings_for_scan(scan_id)
    assert len(findings) == 1
    # OPEN-mode prefixes the model-reported CWE onto the explanation.
    assert "CWE-78" in findings[0]["explanation"]
    db.close()
