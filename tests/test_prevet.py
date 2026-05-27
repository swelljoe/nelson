"""Pre-vet judge: verdict parsing, prompt assembly, and the claude-CLI path."""

import json
import subprocess

import pytest

from nelson.corpus import Case
from nelson.prevet import (
    ClaudeCLIJudge,
    Judge,
    build_prevet_prompt,
    parse_verdict,
)


def test_claude_judge_satisfies_protocol():
    assert isinstance(ClaudeCLIJudge(), Judge)


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"confidence": 0.8, "reasoning": "clear UAF"}', (0.8, "clear UAF")),
        (
            'Sure!\n```json\n{"confidence": 0.2, "reasoning": "vague"}\n```',
            (0.2, "vague"),
        ),
        ('{"confidence": 1.7, "reasoning": "x"}', (1.0, "x")),  # clamped
        ('{"confidence": -3}', (0.0, "")),
    ],
)
def test_parse_verdict_extracts_and_clamps(text, expected):
    assert parse_verdict(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "no json here",
        "{not valid json}",
        '{"reasoning": "missing confidence"}',
    ],
)
def test_parse_verdict_returns_none_when_unusable(text):
    assert parse_verdict(text) is None


def test_prompt_includes_advisory_and_diff_and_omits_empty_fields():
    case = Case(
        source="cvd",
        ext_id="CVE-1",
        project="nginx",
        bug_class="heap-buffer-overflow",
        description="overflow in DAV",
        fix_commit="abc",
    )
    prompt = build_prevet_prompt(case, "diff --git a/x b/x\n@@ -1 +1 @@\n-a\n+b")
    assert "CVE-1" in prompt
    assert "overflow in DAV" in prompt
    assert "diff --git" in prompt
    # Empty fields (no CVE/CWE/severity here) are not rendered as blank lines.
    assert "CWE:" not in prompt
    assert "Severity:" not in prompt


def _completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(["claude"], returncode, stdout, stderr)


def _envelope(result_text, tin=10, tout=5, cost=0.01):
    return json.dumps(
        {
            "type": "result",
            "result": result_text,
            "usage": {"input_tokens": tin, "output_tokens": tout},
            "cost_usd": cost,
        }
    )


def test_claude_judge_parses_verdict_and_usage(monkeypatch):
    env = _envelope('{"confidence": 0.9, "reasoning": "sound"}')
    monkeypatch.setattr(
        "nelson.prevet._run_cli", lambda *a, **k: _completed(0, env)
    )
    v = ClaudeCLIJudge().vet(Case(source="x", ext_id="CVE-1"), "diff")
    assert v.error is None
    assert v.confidence == 0.9
    assert v.notes == "sound"
    assert (v.tokens_in, v.tokens_out, v.cost_usd) == (10, 5, 0.01)


def test_claude_judge_reports_nonzero_exit_as_error(monkeypatch):
    monkeypatch.setattr(
        "nelson.prevet._run_cli",
        lambda *a, **k: _completed(1, "", "Invalid API key"),
    )
    v = ClaudeCLIJudge().vet(Case(source="x", ext_id="CVE-1"), "diff")
    assert v.error is not None
    assert v.confidence == 0.0


def test_claude_judge_reports_timeout_as_error(monkeypatch):
    def _boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="claude", timeout=1)

    monkeypatch.setattr("nelson.prevet._run_cli", _boom)
    v = ClaudeCLIJudge().vet(Case(source="x", ext_id="CVE-1"), "diff")
    assert v.error == "timeout"


def test_claude_judge_unparseable_reply_is_error_not_low_confidence(monkeypatch):
    monkeypatch.setattr(
        "nelson.prevet._run_cli", lambda *a, **k: _completed(0, _envelope("I refuse"))
    )
    v = ClaudeCLIJudge().vet(Case(source="x", ext_id="CVE-1"), "diff")
    assert v.error == "unparseable judge reply"
    assert "I refuse" in v.notes
