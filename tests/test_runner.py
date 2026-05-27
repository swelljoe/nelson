"""Container runner: output parsing, spec assembly, and the orchestrated run.

The container backend and auth are injectable, so the full run_case flow is
exercised end-to-end without podman, a network, or real credentials.
"""

import json
from pathlib import Path

import pytest

from nelson.corpus import Case
from nelson.db import Database
from nelson.runner import (
    AuthMaterial,
    BenchRunner,
    Competitor,
    ContainerExecResult,
    ContainerSpec,
    claude_code_spec,
    extract_result,
    parse_competitor_findings,
)

# -- Output parsing ----------------------------------------------------------


def _stream(*events: dict) -> str:
    return "\n".join(json.dumps(e) for e in events)


def test_extract_result_from_stream_json():
    stdout = _stream(
        {"type": "system", "subtype": "init"},
        {"type": "assistant", "message": {"content": "looking"}},
        {
            "type": "result",
            "subtype": "success",
            "result": '[{"file": "a.c", "line": 7}]',
            "usage": {"input_tokens": 1200, "output_tokens": 80},
            "total_cost_usd": 0.034,
        },
    )
    text, tin, tout, cost = extract_result(stdout)
    assert json.loads(text) == [{"file": "a.c", "line": 7}]
    assert (tin, tout, cost) == (1200, 80, 0.034)


def test_extract_result_from_plain_json_object():
    stdout = json.dumps(
        {
            "type": "result",
            "result": "[]",
            "usage": {"input_tokens": 5, "output_tokens": 1},
            "cost_usd": 0.001,  # non-stream key
        }
    )
    text, tin, tout, cost = extract_result(stdout)
    assert text == "[]"
    assert (tin, tout, cost) == (5, 1, 0.001)


def test_extract_result_sums_cached_input_tokens():
    # A real agentic run's input is dominated by cache reads/writes; tokens_in
    # must reflect the total, not just the final turn's fresh input.
    stdout = _stream(
        {
            "type": "result",
            "result": "[]",
            "usage": {
                "input_tokens": 12,
                "cache_creation_input_tokens": 103814,
                "cache_read_input_tokens": 721156,
                "output_tokens": 36708,
            },
            "total_cost_usd": 1.33,
        }
    )
    _, tin, tout, _ = extract_result(stdout)
    assert tin == 12 + 103814 + 721156
    assert tout == 36708


def test_extract_result_handles_no_result_event():
    text, tin, tout, cost = extract_result("garbage not json\nmore garbage")
    assert "garbage" in text
    assert (tin, tout, cost) == (None, None, None)


def test_parse_competitor_findings_keeps_file_field():
    text = (
        "Here is what I found:\n```json\n"
        '[{"file": "src/x.js", "line": 12, "explanation": "sqli", '
        '"confidence": "high", "cwe": "CWE-89"}]\n```'
    )
    findings = parse_competitor_findings(text)
    assert len(findings) == 1
    assert findings[0]["file"] == "src/x.js"
    assert findings[0]["cwe"] == "CWE-89"


def test_parse_competitor_findings_empty_array():
    assert parse_competitor_findings("[]") == []


def test_parse_competitor_findings_no_array():
    assert parse_competitor_findings("I could not find anything.") == []


def test_parse_competitor_findings_ignores_prose_arrays_before_the_answer():
    # Real failure mode (junrar, file-scoped): the model wrote an analysis
    # summary whose prose contained a *valid* JSON string-array
    # (["..","evil.txt"]) BEFORE the fenced findings array. The parser must skip
    # the prose array (no objects) and return the real findings.
    text = (
        "Analysis: makeFile splits on '\\\\' to get "
        '["..","evil.txt"], escaping the destination.\n\n'
        "```json\n"
        '[{"file": "Extractor.java", "line": 76, "explanation": "zip slip", '
        '"confidence": "high", "cwe": "CWE-22"}]\n```'
    )
    findings = parse_competitor_findings(text)
    assert len(findings) == 1
    assert findings[0]["line"] == 76
    assert findings[0]["cwe"] == "CWE-22"


def test_parse_competitor_findings_repairs_invalid_escapes():
    # Model quotes code with a lone backslash (invalid JSON escape); the verdict
    # must still be recovered rather than dropped.
    text = r'[{"file": "a.c", "line": 3, "code": "x = \q;", "confidence": "low"}]'
    findings = parse_competitor_findings(text)
    assert len(findings) == 1
    assert findings[0]["file"] == "a.c"


# -- Spec assembly -----------------------------------------------------------


def test_claude_code_spec_mounts_source_readonly_and_injects_auth(tmp_path):
    comp = Competitor(name="claude-code/sonnet", model="sonnet")
    auth = AuthMaterial(
        env={"CLAUDE_CONFIG_DIR": "/cfg"}, mounts=[("/h/cfg", "/cfg", "rw,U")]
    )
    spec = claude_code_spec(
        comp, "audit it", tmp_path / "src", Path("/usr/bin/claude"), auth, name="r1"
    )
    assert spec.argv[:2] == ["claude", "-p"]
    assert "--dangerously-skip-permissions" in spec.argv
    assert "sonnet" in spec.argv
    # /src is read-only; the auth config dir comes through; binary is mounted ro.
    modes = {dst: opts for _, dst, opts in spec.mounts}
    assert modes["/src"] == "ro"
    assert modes["/usr/local/bin/claude"] == "ro"
    assert modes["/cfg"] == "rw,U"
    assert spec.env["CLAUDE_CONFIG_DIR"] == "/cfg"
    assert spec.stdin == "audit it"


def test_claude_code_spec_passes_budget_cap_when_set(tmp_path):
    comp = Competitor(name="claude-code/sonnet", model="sonnet")
    auth = AuthMaterial(env={}, mounts=[])
    spec = claude_code_spec(
        comp,
        "audit it",
        tmp_path / "src",
        Path("/usr/bin/claude"),
        auth,
        name="r1",
        max_budget_usd=0.5,
    )
    assert "--max-budget-usd" in spec.argv
    assert spec.argv[spec.argv.index("--max-budget-usd") + 1] == "0.5"
    # Omitted when None.
    spec2 = claude_code_spec(
        comp, "x", tmp_path / "src", Path("/usr/bin/claude"), auth, name="r2"
    )
    assert "--max-budget-usd" not in spec2.argv


# -- File scoping ------------------------------------------------------------


def test_vulnerable_files_drops_tests_and_dedupes():
    from nelson.runner import vulnerable_files

    case = Case(
        source="cvd",
        ext_id="GHSA-y",
        gt_hunks=[
            {"file": "src/main/java/Foo.java", "start": 10, "end": 12},
            {"file": "src/main/java/Foo.java", "start": 40, "end": 42},  # dup file
            {"file": "src/test/java/FooTest.java", "start": 1, "end": 5},
            {"file": "lib/util.py", "start": 3, "end": 3},
        ],
    )
    assert vulnerable_files(case) == ["src/main/java/Foo.java", "lib/util.py"]


def test_vulnerable_files_falls_back_when_only_tests():
    from nelson.runner import vulnerable_files

    # If every hunk is in a test file, audit them rather than have nothing to run.
    case = Case(
        source="cvd",
        ext_id="GHSA-z",
        gt_hunks=[{"file": "tests/test_x.py", "start": 1, "end": 2}],
    )
    assert vulnerable_files(case) == ["tests/test_x.py"]


def test_build_competitor_prompt_names_file_and_hides_advisory():
    from nelson.runner import build_competitor_prompt

    case = Case(
        source="cvd",
        ext_id="GHSA-q",
        cwe="CWE-22",
        bug_class="path-traversal",
        description="backslash zip slip",
    )
    prompt = build_competitor_prompt(case, "src/Extractor.java")
    assert "src/Extractor.java" in prompt
    assert "review" in prompt.lower()
    # No advisory leak: nothing about the planted bug reaches the competitor.
    for leak in ("CWE-22", "path-traversal", "backslash", "zip slip", "GHSA-q"):
        assert leak not in prompt


# -- Orchestrated run (fake backend + fake auth) -----------------------------


class FakeBackend:
    """Records the spec it was handed and returns a canned exec result."""

    def __init__(self, result: ContainerExecResult):
        self._result = result
        self.spec: ContainerSpec | None = None
        self.ensured = False

    def ensure_image(self) -> None:
        self.ensured = True

    def run(self, spec, timeout, cancel_event=None):
        self.spec = spec
        return self._result


class FakeAuth:
    def prepare(self, staging):
        return AuthMaterial(env={"CLAUDE_CONFIG_DIR": "/cfg"}, mounts=[])


def _vetted_case() -> Case:
    return Case(
        source="cvd",
        ext_id="GHSA-test",
        project="acme/widget",
        repo_url="https://example.invalid/acme/widget",
        vuln_commit="deadbeef",
        fix_commit="cafef00d",
        bug_class="sql-injection",
        status="vetted",
    )


def _runner(db, backend, monkeypatch, tmp_path) -> BenchRunner:
    # Don't touch the network/git: pretend the checkout already exists.
    monkeypatch.setattr(
        "nelson.runner.prepare_checkout", lambda url, commit, dest, **k: Path(dest)
    )
    return BenchRunner(
        db,
        backend=backend,
        auth=FakeAuth(),
        claude_bin=Path("/usr/bin/claude"),
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs",
    )


def test_run_case_complete_persists_findings_and_usage(tmp_path, monkeypatch):
    db = Database(tmp_path / "t.db")
    stdout = _stream(
        {
            "type": "result",
            "subtype": "success",
            "result": '[{"file": "app/db.js", "line": 14, "explanation": "raw SQL", '
            '"confidence": "high", "cwe": "CWE-89"}]',
            "usage": {"input_tokens": 900, "output_tokens": 40},
            "total_cost_usd": 0.02,
        }
    )
    backend = FakeBackend(
        ContainerExecResult(0, stdout, "", container_id="nelson-run-1")
    )
    runner = _runner(db, backend, monkeypatch, tmp_path)

    result = runner.run_case(
        _vetted_case(),
        Competitor(name="claude-code/sonnet", model="sonnet"),
        "app/db.js",
    )

    assert result.status == "complete"
    assert backend.ensured
    assert len(result.findings) == 1
    assert (result.tokens_in, result.tokens_out, result.cost_usd) == (900, 40, 0.02)

    runs = db.list_runs()
    assert len(runs) == 1
    assert runs[0]["status"] == "complete"
    # The run records which file it was scoped to, and the prompt named it.
    assert runs[0]["target_file"] == "app/db.js"
    assert backend.spec is not None and backend.spec.stdin is not None
    assert "app/db.js" in backend.spec.stdin
    findings = db.run_findings(runs[0]["id"])
    assert findings[0]["file"] == "app/db.js"
    assert findings[0]["line_start"] == 14
    assert findings[0]["cwe"] == "CWE-89"
    # Transcript persisted to disk.
    assert Path(runs[0]["transcript_path"]).read_text() == stdout


def test_run_case_auth_failure_is_not_a_miss(tmp_path, monkeypatch):
    db = Database(tmp_path / "t.db")
    # Non-zero exit whose output carries an auth marker.
    backend = FakeBackend(
        ContainerExecResult(
            1, "", "Invalid API key · Please run /login", "nelson-run-1"
        )
    )
    runner = _runner(db, backend, monkeypatch, tmp_path)

    result = runner.run_case(
        _vetted_case(),
        Competitor(name="claude-code/sonnet", model="sonnet"),
        "app/db.js",
    )

    assert result.status == "auth_failed"
    assert db.list_runs()[0]["status"] == "auth_failed"


def test_run_case_nonzero_exit_without_marker_is_infra_error(tmp_path, monkeypatch):
    db = Database(tmp_path / "t.db")
    backend = FakeBackend(
        ContainerExecResult(2, "", "segfault in image", "nelson-run-1")
    )
    runner = _runner(db, backend, monkeypatch, tmp_path)

    result = runner.run_case(
        _vetted_case(),
        Competitor(name="claude-code/sonnet", model="sonnet"),
        "app/db.js",
    )

    assert result.status == "infra_error"
    assert db.list_runs()[0]["status"] == "infra_error"


def test_run_case_timeout_is_infra_error(tmp_path, monkeypatch):
    db = Database(tmp_path / "t.db")
    backend = FakeBackend(
        ContainerExecResult(124, "", "timeout", "nelson-run-1", timed_out=True)
    )
    runner = _runner(db, backend, monkeypatch, tmp_path)

    result = runner.run_case(
        _vetted_case(),
        Competitor(name="claude-code/sonnet", model="sonnet"),
        "app/db.js",
    )

    assert result.status == "infra_error"
    assert result.error == "timeout"


def test_run_case_requires_derived_case(tmp_path, monkeypatch):
    from nelson.runner import RunnerError

    db = Database(tmp_path / "t.db")
    backend = FakeBackend(ContainerExecResult(0, "[]", ""))
    runner = _runner(db, backend, monkeypatch, tmp_path)
    undrived = Case(source="cvd", ext_id="GHSA-x")  # no repo_url / vuln_commit

    with pytest.raises(RunnerError):
        runner.run_case(
            undrived, Competitor(name="claude-code/sonnet", model="sonnet"), "x.js"
        )


# -- DB run layer ------------------------------------------------------------


def test_competitor_upsert_is_idempotent(tmp_path):
    db = Database(tmp_path / "t.db")
    a = db.upsert_competitor({"name": "claude-code/sonnet", "model": "sonnet"})
    b = db.upsert_competitor({"name": "claude-code/sonnet", "model": "sonnet-4-6"})
    assert a == b
    comp = db.get_competitor("claude-code/sonnet")
    assert comp is not None
    assert comp["model"] == "sonnet-4-6"


def test_run_layer_round_trip(tmp_path):
    db = Database(tmp_path / "t.db")
    case_id = db.upsert_case({"source": "cvd", "ext_id": "GHSA-z"})
    comp_id = db.upsert_competitor({"name": "claude-code/sonnet", "model": "sonnet"})
    run_id = db.create_run(case_id, comp_id)
    db.start_run(run_id, container_id="nelson-run-x")
    db.complete_run(run_id, tokens_in=10, tokens_out=2, cost_usd=0.01, wall_clock_s=3.0)
    db.add_run_finding(run_id, file="x.c", line_start=1, line_end=1, confidence="high")
    row = db.get_run(run_id)
    assert row is not None
    assert row["status"] == "complete"
    assert row["container_id"] == "nelson-run-x"
    assert len(db.run_findings(run_id)) == 1
