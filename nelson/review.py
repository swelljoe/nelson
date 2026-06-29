"""Review pass: have a (usually smarter) model verify findings."""

import json
import logging
import time
from pathlib import Path

from . import compare as compare_mod
from .agents import create_adapter
from .db import Database

log = logging.getLogger(__name__)

# Lower rank sorts first when choosing a cluster's representative finding.
_CONF_RANK = {"high": 0, "medium": 1, "low": 2}


def _representative(members: list[dict]) -> dict:
    """Pick the finding that best represents a cluster for the judge.

    Highest confidence first, then earliest line, then lowest id — deterministic
    so a re-review sees the same representative.
    """

    def key(m: dict):
        conf = (m.get("confidence") or "").lower()
        line = m["line_number"] if m["line_number"] is not None else 10**9
        return (_CONF_RANK.get(conf, 3), line, m["id"])

    return min(members, key=key)


def build_review_prompt(
    file_path: str,
    file_content: str,
    language: str,
    finding: dict,
) -> str:
    """Build a prompt asking a model to verify a reported vulnerability."""
    cwe_id = finding.get("effective_cwe") or finding.get("cwe_id") or "OPEN"
    cwe_label = (
        cwe_id if cwe_id not in ("OPEN", "UNKNOWN") else "general vulnerability scan"
    )

    return f"""You are a senior security engineer reviewing a vulnerability report. A scanning tool has flagged the following potential vulnerability. Your job is to determine whether this is a real, exploitable vulnerability by analyzing the actual code and execution flow.

REPORTED FINDING:
- File: {file_path}
- Line: {finding["line_number"]}
- CWE: {cwe_label}
- Scanner's confidence: {finding["confidence"]}
- Code flagged: {finding["code_snippet"]}
- Scanner's explanation: {finding["explanation"]}

ANALYSIS INSTRUCTIONS:
1. Read the full source file below carefully.
2. Trace the execution flow: Is the flagged code actually reachable with attacker-controlled input?
3. Consider the context: Is this an internal tool, a network service, a library? Who controls the inputs?
4. Check for existing mitigations: Are there input validation, sanitization, or access controls that the scanner may have missed?
5. Assess realistic exploitability, not just theoretical possibility.

Return a JSON object (not an array) with these fields:
- "verdict": one of "confirmed", "false_positive", or "needs_review"
  - "confirmed": This is a real vulnerability that should be fixed
  - "false_positive": The scanner was wrong — this is not actually exploitable
  - "needs_review": You're not sure — a human should look at this
- "reasoning": Detailed explanation of your analysis, including the execution flow you traced (string)
- "severity": "critical", "high", "medium", "low", or "informational" (string)
- "suggested_fix": If confirmed, a brief description of how to fix it. If false_positive, why it's safe. (string)

Return ONLY the JSON object, no other text, no markdown fences.

Full source file ({file_path}):
```{language}
{file_content}
```"""


def parse_review_result(text: str) -> dict | None:
    """Parse the review model's JSON response."""
    text = text.strip()

    # Try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "verdict" in data:
            return data
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in text
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    data = json.loads(text[start : i + 1])
                    if isinstance(data, dict) and "verdict" in data:
                        return data
                except json.JSONDecodeError:
                    pass
                break

    return None


def run_review(
    db: Database,
    scan_id: int,
    model_spec: str,
    target_dir: str,
    delay: float = 2.0,
    max_backoff: float = 300.0,
    line_tolerance: int = 2,
    on_progress=None,
    tools: bool = False,
):
    """De-duplicate findings into clusters and review each cluster once.

    Repeated passes and multiple models surface the same bug many times. We
    cluster the unreviewed findings (same file/CWE within ``line_tolerance``
    lines), judge one representative per cluster, and write the verdict to every
    member — so the (often expensive) reviewer is paid once per unique bug, while
    all member rows are kept so cross-model agreement reporting still works.

    Args:
        db: Database instance
        scan_id: Scan whose findings to review
        model_spec: Model spec for the reviewer (e.g., "claude:sonnet")
        target_dir: Root directory of the scanned code
        delay: Seconds between reviews (for CLI pacing)
        max_backoff: Maximum backoff on rate limiting
        line_tolerance: Line window for treating findings as the same bug
        on_progress: Optional callback(reviewed_clusters, total_clusters)
        tools: Give an OpenAI-compatible reviewer a read-only tool loop over the
            scanned tree so it can read related files when tracing reachability.
            No-op for the CLI runtimes (Claude Code, Gemini CLI).
    """
    from .inventory import LANGUAGE_MAP

    target = Path(target_dir).resolve()
    adapter = create_adapter(model_spec, tools=tools, tools_root=str(target))

    findings = db.unreviewed_findings(scan_id)
    if not findings:
        log.info("No unreviewed findings for scan %d", scan_id)
        return

    clusters = compare_mod.cluster_findings(findings, line_tolerance=line_tolerance)
    total = len(clusters)
    log.info(
        "Reviewing %d findings in %d clusters with %s",
        len(findings),
        total,
        adapter.name,
    )

    def _apply(members: list[dict], *, status: str, by: str, fix: str | None):
        """Write one verdict to every member of a cluster."""
        for m in members:
            db.update_finding_review(
                m["id"],
                verified_by_model=by,
                verified_status=status,
                suggested_fix=fix,
            )

    backoff = 0.0

    try:
        for i, cluster in enumerate(clusters):
            members = cluster.findings
            rep = _representative(members)
            file_path = target / rep["file_path"]
            if not file_path.is_file():
                log.info("File deleted, marking resolved: %s", rep["file_path"])
                _apply(
                    members,
                    status="resolved",
                    by="nelson",
                    fix="File no longer exists in the source tree.",
                )
                if on_progress:
                    on_progress(i + 1, total)
                continue
            try:
                content = file_path.read_text(errors="replace")
            except OSError as e:
                log.warning("Cannot read %s: %s", rep["file_path"], e)
                continue

            # Determine language from file extension
            ext = file_path.suffix.lower()
            language = LANGUAGE_MAP.get(ext, "unknown")

            prompt = build_review_prompt(
                rep["file_path"],
                content,
                language,
                rep,
            )

            log.info(
                "Review %d/%d: %s line %s (%s, %d findings)",
                i + 1,
                total,
                rep["file_path"],
                rep["line_number"],
                rep.get("effective_cwe") or rep["cwe_id"],
                len(members),
            )

            result = adapter.run(prompt)

            if result.rate_limited:
                backoff = min(max(backoff * 2, 30.0), max_backoff)
                log.warning("Rate limited, backing off %.0fs", backoff)
                time.sleep(backoff)
                # Retry this cluster
                result = adapter.run(prompt)
                if result.rate_limited:
                    log.error(
                        "Still rate limited after backoff, skipping cluster at %s:%s",
                        rep["file_path"],
                        rep["line_number"],
                    )
                    continue

            backoff = 0.0

            if result.error:
                log.warning(
                    "Review error for %s:%s: %s",
                    rep["file_path"],
                    rep["line_number"],
                    result.error,
                )
                continue

            review = parse_review_result(result.raw_output)
            # For Claude CLI, the raw_output is the JSON envelope;
            # try parsing the result field
            if review is None:
                try:
                    envelope = json.loads(result.raw_output)
                    if isinstance(envelope, dict) and "result" in envelope:
                        review = parse_review_result(
                            str(envelope["result"]),
                        )
                except (json.JSONDecodeError, TypeError):
                    pass

            if review is None:
                log.warning(
                    "Could not parse review response for %s:%s",
                    rep["file_path"],
                    rep["line_number"],
                )
                continue

            verdict = review.get("verdict", "needs_review")
            if verdict not in (
                "confirmed",
                "false_positive",
                "needs_review",
            ):
                verdict = "needs_review"

            reasoning = review.get("reasoning", "")
            severity = review.get("severity", "")
            suggested_fix = review.get("suggested_fix", "")

            # Combine reasoning and severity into suggested_fix for storage
            fix_text = ""
            if severity:
                fix_text += f"Severity: {severity}\n"
            if reasoning:
                fix_text += f"Analysis: {reasoning}\n"
            if suggested_fix:
                fix_text += f"Fix: {suggested_fix}"

            _apply(
                members,
                status=verdict,
                by=adapter.name,
                fix=fix_text.strip() or None,
            )

            if on_progress:
                on_progress(i + 1, total)

            if delay > 0 and adapter.needs_pacing:
                time.sleep(delay)

    except KeyboardInterrupt:
        log.warning("Review interrupted by user")
        raise
