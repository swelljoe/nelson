"""Failure classification and the integrity rule.

The core invariant under test: an auth / rate / infra failure is distinguishable
from a model that genuinely looked and found nothing. An empty findings list with
no failure_kind is a real (negative) result; an empty findings list with a
failure_kind is NOT.
"""

import pytest

from nelson.agents import (
    AgentResult,
    ClaudeCLIAdapter,
    FailureKind,
    PreflightResult,
    classify_failure,
)
from nelson.auth import AuthProfile

# -- classify_failure --------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Error: 429 Too Many Requests",
        "model is overloaded, try again",
        "RESOURCE_EXHAUSTED: quota exceeded",
        "You have hit the rate limit",
    ],
)
def test_rate_limit_markers(text):
    assert classify_failure(text, failed=True) is FailureKind.RATE_LIMIT


@pytest.mark.parametrize(
    "text",
    [
        "Invalid API key provided",
        "Not logged in. Please run /login",
        "OAuth token has expired",
        "401 Unauthorized",
        "error 403 forbidden",
        "authentication_error: missing api key",
    ],
)
def test_auth_markers(text):
    assert classify_failure(text, failed=True) is FailureKind.AUTH


def test_unexplained_failure_is_infra_never_none():
    # A non-zero exit we can't attribute is infra, not a silent success/miss.
    assert classify_failure("Segmentation fault (core dumped)", failed=True) is (
        FailureKind.INFRA
    )


def test_success_with_no_markers_is_none():
    assert classify_failure("here are your findings: []", failed=False) is None


def test_rate_limit_wins_over_auth():
    # A 429 body that also mentions a key must classify as rate-limit (retryable),
    # not auth (terminal).
    text = "429 too many requests for this api key"
    assert classify_failure(text, failed=True) is FailureKind.RATE_LIMIT


# -- AgentResult sync --------------------------------------------------------


def test_failure_kind_rate_limit_sets_legacy_bool():
    r = AgentResult(findings=[], raw_output="", failure_kind=FailureKind.RATE_LIMIT)
    assert r.rate_limited is True


def test_legacy_rate_limited_bool_backfills_kind():
    r = AgentResult(findings=[], raw_output="", rate_limited=True)
    assert r.failure_kind is FailureKind.RATE_LIMIT


def test_clean_empty_result_has_no_failure_kind():
    # The "found nothing" case: empty findings, no failure.
    r = AgentResult(findings=[], raw_output="[]")
    assert r.failure_kind is None
    assert r.rate_limited is False


# -- Missing secret short-circuits to AUTH without running a subprocess ------


def test_missing_secret_yields_auth_without_subprocess(monkeypatch):
    # If a configured secret is absent, run() must return auth_failed BEFORE
    # spawning the CLI — a missing key is never a miss, and we don't waste a call.
    import nelson.agents as agents

    def _boom(*args, **kwargs):
        raise AssertionError("subprocess must not be spawned when auth is missing")

    monkeypatch.setattr(agents, "_run_cli", _boom)

    adapter = ClaudeCLIAdapter(
        model="haiku",
        auth_profile=AuthProfile(name="p", env={"ANTHROPIC_API_KEY": "ABSENT_KEY"}),
    )
    # Ensure the secret really is absent in this environment.
    monkeypatch.delenv("ABSENT_KEY", raising=False)

    result = adapter.run("anything")

    assert result.failure_kind is FailureKind.AUTH
    assert result.findings == []
    assert "ABSENT_KEY" in (result.error or "")


def test_no_auth_profile_inherits_env(monkeypatch):
    # Default behavior unchanged: no profile -> _resolve_env returns None so the
    # child inherits the parent environment (env=None to Popen).
    adapter = ClaudeCLIAdapter(model="haiku")
    assert adapter.auth_profile is None
    assert adapter._resolve_env() is None


def test_runtime_dimensions_exposed():
    # The model/runtime/tool-profile split is visible as data.
    adapter = ClaudeCLIAdapter(model="sonnet")
    assert adapter.runtime == "claude-code"
    assert adapter.model_id == "sonnet"
    assert adapter.tool_profile == "single-shot"
    assert adapter.name == "claude-sonnet"  # unchanged DB identity


# -- Preflight ---------------------------------------------------------------


class _FakeAdapter(ClaudeCLIAdapter):
    """Adapter whose run() returns a canned result, for preflight tests."""

    def __init__(self, result: AgentResult):
        super().__init__(model="fake")
        self._result = result

    def run(self, prompt, cancel_event=None):
        return self._result


def test_preflight_ok_on_clean_result():
    pf = _FakeAdapter(AgentResult(findings=[], raw_output="ok")).preflight()
    assert pf == PreflightResult(True, "ok")


def test_preflight_reports_auth_failure():
    res = AgentResult(
        findings=[], raw_output="", failure_kind=FailureKind.AUTH, error="bad key"
    )
    pf = _FakeAdapter(res).preflight()
    assert pf.ok is False
    assert pf.status == "auth_failed"


def test_preflight_reports_infra_failure():
    res = AgentResult(
        findings=[], raw_output="", failure_kind=FailureKind.INFRA, error="timeout"
    )
    pf = _FakeAdapter(res).preflight()
    assert pf.ok is False
    assert pf.status == "infra_error"
