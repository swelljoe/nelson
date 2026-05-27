"""Opus pre-vet: judge whether a derived case is a sound benchmark item.

Auto-derived ground truth is noisy — squashed fixes, refactors bundled with the
security change, advisories whose described bug doesn't match what the diff
actually touches. Before a case counts, Opus 4.7 reviews the advisory against the
derived fix and rates how confidently a model analyzing the *pre-patch* code
could find *this* vulnerability. This is the truth side of judging, so the judge
sees the description (the FP-judge, P4, deliberately will not).

A judge *failure* (timeout, auth, unparseable reply) is reported via
``VetVerdict.error`` and must never be read as "retire": the pipeline leaves such
a case a candidate to retry, exactly as auth/infra failures are never scored as a
miss elsewhere. Judge token cost is captured so it can be tracked separately from
competitor cost.

The judge runs ``claude -p`` directly on the host, where the CLI is already
signed in to the subscription. Token usage draws against the plan's included
budget — there is no API key and (unlike a competitor) no container, so the
judge needs no credential injection.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .agents import _run_cli, classify_failure

if TYPE_CHECKING:
    from .corpus import Case

# Bound the evidence we send: enough of the fix to judge coherence, not a whole
# vendored-dependency diff.
_MAX_DIFF_CHARS = 12000

PREVET_INSTRUCTIONS = """\
You are vetting a candidate case for a security benchmark. The benchmark gives a \
model the source code *before* a fix and checks whether it can find the \
vulnerability that the fix later addressed.

Judge whether this is a SOUND case: (1) the advisory and the fix describe the \
same, real security bug; (2) the vulnerability is actually present in the \
pre-patch code shown by the diff's removed/context lines (not, say, a pure \
hardening or version bump); and (3) a capable model reading the pre-patch code \
could plausibly identify it.

Reply with ONLY a JSON object:
{"confidence": <float 0.0-1.0>, "reasoning": "<one or two sentences>"}
where confidence is how strongly this is a sound, findable case."""


@dataclass
class VetVerdict:
    """Outcome of pre-vetting one case.

    ``confidence`` in [0,1] is meaningful only when ``error`` is None. When
    ``error`` is set the judge could not decide and the case must stay a
    candidate (do not retire on a judge failure).
    """

    confidence: float
    notes: str = ""
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    error: str | None = None


@runtime_checkable
class Judge(Protocol):
    name: str

    def vet(self, case: Case, diff: str) -> VetVerdict: ...


def build_prevet_prompt(case: Case, diff: str) -> str:
    """Assemble the advisory metadata + (truncated) fix diff into a judge prompt."""
    advisory = "\n".join(
        f"{label}: {value}"
        for label, value in (
            ("Identifier", case.ext_id),
            ("CVE", case.cve_id),
            ("GHSA", case.ghsa_id),
            ("Project", case.project),
            ("Bug class", case.bug_class),
            ("CWE", case.cwe),
            ("Severity", case.severity),
            ("Description", case.description),
        )
        if value
    )
    clipped = diff[:_MAX_DIFF_CHARS]
    if len(diff) > _MAX_DIFF_CHARS:
        clipped += "\n... [diff truncated] ..."
    return (
        f"{PREVET_INSTRUCTIONS}\n\n"
        f"## Advisory\n{advisory}\n\n"
        f"## Fix diff (parent -> fix commit {case.fix_commit})\n"
        f"```diff\n{clipped}\n```"
    )


def parse_verdict(text: str) -> tuple[float, str] | None:
    """Extract (confidence, reasoning) from the judge's reply, or None if absent.

    Tolerant of prose or code-fences around the JSON. Returns None (not a
    zero-confidence verdict) when nothing parseable is found, so the caller can
    distinguish "judge said low" from "judge reply unusable".
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict) or "confidence" not in data:
        return None
    try:
        confidence = max(0.0, min(1.0, float(data["confidence"])))
    except (TypeError, ValueError):
        return None
    return confidence, str(data.get("reasoning", ""))


class ClaudeCLIJudge:
    """Pre-vet judge backed by ``claude -p`` (Opus by default).

    Uses the host's already-authenticated claude CLI (subscription budget), so
    there is no API key or auth profile: the subprocess inherits this process's
    environment unchanged.
    """

    name = "claude-cli"

    def __init__(self, model: str = "opus", timeout: int = 180):
        self.model = model
        self.timeout = timeout

    def vet(self, case: Case, diff: str) -> VetVerdict:
        prompt = build_prevet_prompt(case, diff)
        cmd = [
            "claude",
            "-p",
            "--model",
            self.model,
            "--output-format",
            "json",
            "--no-session-persistence",
        ]
        try:
            # env=None: inherit the host's authenticated claude CLI session.
            result = _run_cli(cmd, self.timeout, input_text=prompt)
        except subprocess.TimeoutExpired:
            return VetVerdict(0.0, error="timeout")

        raw, stderr = result.stdout, result.stderr
        if result.returncode != 0:
            kind = classify_failure(raw + stderr, failed=True)
            return VetVerdict(
                0.0, error=f"claude exit {result.returncode} ({kind}): {stderr[:200]}"
            )

        text, tin, tout, cost = _unwrap_claude_json(raw)
        parsed = parse_verdict(text)
        if parsed is None:
            return VetVerdict(
                0.0,
                error="unparseable judge reply",
                notes=text[:300],
                tokens_in=tin,
                tokens_out=tout,
                cost_usd=cost,
            )
        confidence, reasoning = parsed
        return VetVerdict(
            confidence,
            notes=reasoning,
            tokens_in=tin,
            tokens_out=tout,
            cost_usd=cost,
        )


def _unwrap_claude_json(
    raw: str,
) -> tuple[str, int | None, int | None, float | None]:
    """Pull text + usage out of ``claude --output-format json`` (else raw text)."""
    try:
        envelope = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw, None, None, None
    if isinstance(envelope, dict) and "result" in envelope:
        usage = envelope.get("usage", {})
        return (
            str(envelope["result"]),
            usage.get("input_tokens"),
            usage.get("output_tokens"),
            envelope.get("cost_usd"),
        )
    return raw, None, None, None
