"""Run a patch-generation job for one benchmark detection finding."""

from __future__ import annotations

import json
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .agents import FailureKind, classify_failure
from .corpus import Case
from .runner import (
    Competitor,
    ContainerBackend,
    PodmanBackend,
    RunnerError,
    _safe_rmtree,
    prepare_checkout,
)
from .runtimes import RuntimeContext, auth_for_competitor, get_runtime
from .verify import CandidatePatchVerifier, PodmanCommandRunner

_FENCED_DIFF = re.compile(r"```(?:diff)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_git_diff(text: str) -> str | None:
    """Extract a model's Git patch while retaining strict application semantics."""
    candidates = [match.strip() for match in _FENCED_DIFF.findall(text)]
    candidates.append(text.strip())
    for candidate in candidates:
        start = candidate.find("diff --git ")
        if start >= 0:
            return candidate[start:].strip() + "\n"
    return None


def build_remediation_prompt(finding) -> str:
    location = finding["file"] or "(unspecified file)"
    if finding["line_start"] is not None:
        location += f":{finding['line_start']}"
    return f"""A separate security-audit job reported this vulnerability:

Location: {location}
Confidence: {finding['confidence'] or 'unspecified'}
CWE: {finding['cwe'] or 'unspecified'}

{finding['description'] or '(no description supplied)'}

Inspect the repository and implement a minimal fix for this reported vulnerability
without breaking legitimate behavior. Return only the Git unified diff required by
the system instructions.
"""


@dataclass
class RemediationResult:
    remediation_id: int
    status: str
    patch_applied: bool = False
    build_passed: bool = False
    witnesses_passed: bool = False
    controls_passed: bool = False
    verified: bool = False
    tokens_in: int | None = None
    tokens_out: int | None = None
    wall_clock_s: float | None = None
    error: str | None = None


class RemediationRunner:
    def __init__(
        self,
        db,
        *,
        backend: ContainerBackend | None = None,
        cache_dir: str | Path = "remediation-cache",
        runs_dir: str | Path = "remediation-runs",
        verification_dir: str | Path = "verification-cache",
        network: bool = True,
        timeout_s: float = 1800.0,
    ):
        self.db = db
        self.backend = backend or PodmanBackend()
        self.cache_dir = Path(cache_dir)
        self.runs_dir = Path(runs_dir)
        self.verification_dir = Path(verification_dir)
        self.network = network
        self.timeout_s = timeout_s

    def run(
        self,
        finding_id: int,
        competitor_name: str,
        harness_dir: str | Path,
        *,
        thinking: bool = True,
        max_output_tokens: int = 16384,
    ) -> RemediationResult:
        finding = self.db.get_run_finding_context(finding_id)
        if finding is None:
            raise RunnerError(f"finding {finding_id} not found")
        competitor_row = self.db.get_competitor(competitor_name)
        if competitor_row is None:
            raise RunnerError(f"competitor {competitor_name!r} not found")
        competitor = Competitor.from_row(competitor_row)
        if competitor.runtime != "raw-api-loop":
            raise RunnerError("remediation currently supports raw-api-loop competitors")
        case_row = self.db.get_case_by_id(finding["case_id"])
        if case_row is None:
            raise RunnerError(f"case {finding['case_id']} not found")
        case = Case.from_row(case_row)
        config = {
            "thinking": thinking,
            "max_output_tokens": max_output_tokens,
            "harness_dir": str(Path(harness_dir)),
        }
        remediation_id = self.db.create_remediation_run(
            finding_id, competitor_row["id"], config=config
        )
        outcome = RemediationResult(remediation_id, "pending")

        try:
            if not case.repo_url or not case.vuln_commit:
                raise RunnerError("case requires repo_url and vuln_commit")
            source = prepare_checkout(
                case.repo_url,
                case.vuln_commit,
                self.cache_dir / case.ext_id,
            )
            runtime = get_runtime(competitor.runtime)
            self.backend.ensure_image()
        except Exception as exc:
            return self._fail(outcome, "infra_error", str(exc))

        staging = Path(tempfile.mkdtemp(prefix="nelson-remediation-auth-"))
        try:
            try:
                auth = auth_for_competitor(competitor, runtime).prepare(staging)
            except RunnerError as exc:
                return self._fail(outcome, "auth_failed", str(exc))
            ctx = RuntimeContext(
                competitor=competitor,
                prompt=build_remediation_prompt(finding),
                src_dir=source,
                auth=auth,
                name=f"nelson-remediate-{remediation_id}",
                network=self.network,
            )
            spec = runtime.build_spec(ctx)
            spec.env["NELSON_TASK"] = "remediation"
            spec.env["NELSON_MAX_OUTPUT_TOKENS"] = str(max_output_tokens)
            extra = json.loads(spec.env.get("NELSON_EXTRA_BODY", "{}"))
            template = extra.setdefault("chat_template_kwargs", {})
            template["enable_thinking"] = thinking
            spec.env["NELSON_EXTRA_BODY"] = json.dumps(extra)
            self.db.start_remediation_run(remediation_id)
            started = time.monotonic()
            executed = self.backend.run(spec, self.timeout_s)
            wall = time.monotonic() - started
        finally:
            _safe_rmtree(staging)

        outcome.wall_clock_s = wall
        combined = executed.stdout + executed.stderr
        if executed.timed_out or executed.returncode != 0:
            kind = classify_failure(combined, failed=True)
            status = "auth_failed" if kind is FailureKind.AUTH else "infra_error"
            return self._fail(
                outcome,
                status,
                "remediation timed out" if executed.timed_out else combined[-1000:],
            )

        parsed = runtime.parse_output(executed, competitor)
        outcome.tokens_in = parsed.tokens_in
        outcome.tokens_out = parsed.tokens_out
        patch = extract_git_diff(parsed.final_text)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = self.runs_dir / f"remediation-{remediation_id}.jsonl"
        transcript_path.write_text(executed.stdout)
        patch_path = self.runs_dir / f"remediation-{remediation_id}.diff"
        if patch is not None:
            patch_path.write_text(patch)

        verification = None
        if patch is not None:
            spec_data = case.verification or {}
            verification = CandidatePatchVerifier(
                PodmanCommandRunner(
                    self.backend,
                    image=str(spec_data.get("image") or "nelson-bench:fedora-tools2"),
                    network=bool(spec_data.get("network", False)),
                    harness_dir=harness_dir,
                )
            ).verify(case, patch_path, self.verification_dir, harness_dir)

        checks = verification.checks if verification else []
        outcome.patch_applied = bool(
            verification and verification.patch and verification.patch.applied
        )
        outcome.build_passed = _kind_passed(checks, "build")
        outcome.witnesses_passed = _kind_passed(checks, "witness")
        outcome.controls_passed = _kind_passed(checks, "control")
        outcome.verified = bool(verification and verification.verified)
        outcome.status = "complete"
        if patch is None:
            outcome.error = "model returned no Git unified diff"
        elif verification and verification.error:
            outcome.error = verification.error
        elif verification and verification.patch and not verification.patch.applied:
            outcome.error = verification.patch.error
        elif verification and not verification.verified:
            failed_kinds = sorted(
                {check.kind for check in verification.checks if not check.passed}
            )
            outcome.error = (
                f"failed verification stages: {', '.join(failed_kinds)}"
                if failed_kinds
                else "candidate did not satisfy verification harness"
            )
        self.db.complete_remediation_run(
            remediation_id,
            tokens_in=parsed.tokens_in,
            tokens_out=parsed.tokens_out,
            cost_usd=parsed.cost,
            wall_clock_s=wall,
            transcript_path=str(transcript_path),
            raw_output=parsed.final_text,
            patch_text=patch,
            patch_applied=outcome.patch_applied,
            build_passed=outcome.build_passed,
            witnesses_passed=outcome.witnesses_passed,
            controls_passed=outcome.controls_passed,
            verified=outcome.verified,
            error_msg=outcome.error,
        )
        if verification and verification.error:
            self.db.fail_remediation_run(
                remediation_id, "infra_error", verification.error
            )
            outcome.status = "infra_error"
        return outcome

    def _fail(
        self, outcome: RemediationResult, status: str, error: str
    ) -> RemediationResult:
        self.db.fail_remediation_run(outcome.remediation_id, status, error)
        outcome.status = status
        outcome.error = error
        return outcome


def _kind_passed(checks, kind: str) -> bool:
    selected = [check for check in checks if check.kind == kind]
    return all(check.passed for check in selected) if selected else kind != "witness"
