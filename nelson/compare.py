"""Cross-model comparison: cluster findings into 'same issue' groups,
score agreement, and render text/JSON reports.

A cluster is a maximal set of findings sharing the same file_path and
effective CWE, grouped by single-link line proximity: when ordered by
line number, each successive finding in the cluster is within
``line_tolerance`` of the previous finding. The voter denominator for a
cluster is the set of distinct models that ran a *completed* job
covering that (file, cwe) — either as a focused per-CWE job or as an
OPEN-mode job on that file.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import click

# In open mode, scanner.py prefixes the model-identified CWE onto the
# explanation text: "[CWE-89] User input flows to ..." We parse it back
# out so open-mode findings cluster on the actual vulnerability class
# rather than the literal job CWE of "OPEN".
_CWE_PREFIX = re.compile(r"^\[(CWE-\d+|unknown)\]\s*", re.IGNORECASE)


def effective_cwe(row) -> str:
    """The CWE that should be used for clustering this finding."""
    cwe = row["cwe_id"]
    if cwe:
        cwe = cwe.upper()
        if cwe != "OPEN":
            return cwe
    m = _CWE_PREFIX.match(row["explanation"] or "")
    return m.group(1).upper() if m else "UNKNOWN"


@dataclass
class Cluster:
    file_path: str
    cwe_id: str
    findings: list[dict] = field(default_factory=list)
    models_flagged: list[str] = field(default_factory=list)
    models_missed: list[str] = field(default_factory=list)

    @property
    def line_range(self) -> tuple[int | None, int | None]:
        lines = [
            f["line_number"] for f in self.findings if f["line_number"] is not None
        ]
        if not lines:
            return None, None
        return min(lines), max(lines)

    @property
    def agreement(self) -> int:
        return len(self.models_flagged)

    @property
    def eligible(self) -> int:
        return len(self.models_flagged) + len(self.models_missed)


def _row_to_dict(row) -> dict:
    """sqlite3.Row -> plain dict, with the effective CWE attached."""
    d = {k: row[k] for k in row}
    d["effective_cwe"] = effective_cwe(row)
    return d


def cluster_findings(
    findings: list,
    line_tolerance: int = 2,
) -> list[Cluster]:
    """Group findings into Clusters using single-link line proximity."""
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for raw in findings:
        d = _row_to_dict(raw)
        by_key[(d["file_path"], d["effective_cwe"])].append(d)

    clusters: list[Cluster] = []
    for (file_path, cwe), items in by_key.items():
        # None-line findings get their own cluster per (file, cwe).
        with_line = [it for it in items if it["line_number"] is not None]
        without_line = [it for it in items if it["line_number"] is None]

        with_line.sort(key=lambda x: x["line_number"])

        cur: list[dict] = []
        cur_max = -(10**9)
        for it in with_line:
            ln = it["line_number"]
            if cur and ln - cur_max <= line_tolerance:
                cur.append(it)
                if ln > cur_max:
                    cur_max = ln
            else:
                if cur:
                    clusters.append(Cluster(file_path, cwe, cur))
                cur = [it]
                cur_max = ln
        if cur:
            clusters.append(Cluster(file_path, cwe, cur))
        if without_line:
            clusters.append(Cluster(file_path, cwe, without_line))

    return clusters


def annotate_clusters(
    clusters: list[Cluster],
    focused_coverage: dict[tuple[str, str], set[str]],
    open_coverage: dict[str, set[str]],
) -> None:
    """Fill in models_flagged and models_missed on each cluster."""
    for c in clusters:
        flagged = {f["model_id"] for f in c.findings}
        eligible = set()
        eligible |= focused_coverage.get((c.file_path, c.cwe_id), set())
        eligible |= open_coverage.get(c.file_path, set())
        # Defensive: any model that flagged is by definition eligible.
        eligible |= flagged
        c.models_flagged = sorted(flagged)
        c.models_missed = sorted(eligible - flagged)


def all_models(
    focused_coverage: dict[tuple[str, str], set[str]],
    open_coverage: dict[str, set[str]],
) -> list[str]:
    seen: set[str] = set()
    for s in focused_coverage.values():
        seen |= s
    for s in open_coverage.values():
        seen |= s
    return sorted(seen)


def filter_clusters(
    clusters: list[Cluster],
    cwe: str | None = None,
    confidence: str | None = None,
    min_agreement: int | None = None,
) -> list[Cluster]:
    out = []
    for c in clusters:
        if cwe and c.cwe_id != cwe:
            continue
        if confidence and not any(
            (f.get("confidence") or "").lower() == confidence.lower()
            for f in c.findings
        ):
            continue
        if min_agreement is not None and c.agreement < min_agreement:
            continue
        out.append(c)
    return out


def sort_clusters(clusters: list[Cluster]) -> list[Cluster]:
    """Highest-agreement clusters first, then by file/line."""

    def key(c: Cluster):
        lo, _hi = c.line_range
        return (-c.agreement, c.file_path, c.cwe_id, lo if lo is not None else 10**9)

    return sorted(clusters, key=key)


def to_json(
    clusters: list[Cluster],
    models: list[str],
    scan_ids: list[int],
) -> str:
    payload: dict[str, Any] = {
        "scan_ids": scan_ids,
        "models": models,
        "clusters": [
            {
                "file": c.file_path,
                "cwe": c.cwe_id,
                "line_range": list(c.line_range),
                "agreement": c.agreement,
                "eligible": c.eligible,
                "models_flagged": c.models_flagged,
                "models_missed": c.models_missed,
                "findings": [
                    {
                        "model": f["model_id"],
                        "scan_id": f.get("scan_id"),
                        "line": f["line_number"],
                        "confidence": f["confidence"],
                        "code": f["code_snippet"],
                        "explanation": f["explanation"],
                        "verdict": f.get("verified_status"),
                        "reviewed_by": f.get("verified_by_model"),
                    }
                    for f in c.findings
                ],
            }
            for c in clusters
        ],
    }
    return json.dumps(payload, indent=2)


def render_text(
    clusters: list[Cluster],
    models: list[str],
    scan_ids: list[int],
) -> str:
    """Colorized terminal output. Uses click.style for ANSI."""
    out: list[str] = []
    label = (
        f"Scan {scan_ids[0]}"
        if len(scan_ids) == 1
        else f"Scans {', '.join(str(i) for i in scan_ids)}"
    )
    out.append(click.style(f"=== Comparison: {label} ===", bold=True))
    out.append(f"Models ({len(models)}): {', '.join(models)}")
    out.append("")

    # Agreement histogram
    by_agreement_eligible: dict[tuple[int, int], int] = defaultdict(int)
    for c in clusters:
        by_agreement_eligible[(c.agreement, c.eligible)] += 1
    out.append("Agreement summary:")
    for agreement, eligible in sorted(
        by_agreement_eligible.keys(), reverse=True
    ):
        n = by_agreement_eligible[(agreement, eligible)]
        marker = (
            click.style("strong signal", fg="red", bold=True)
            if eligible >= 2 and agreement == eligible
            else (
                click.style("partial", fg="yellow")
                if agreement >= 2
                else click.style("singleton", fg="white")
            )
        )
        out.append(
            f"  {agreement}/{eligible} eligible models  {n:>4} clusters  ({marker})"
        )
    out.append("")

    # Per-model totals (flagged in any cluster + only-this-model totals)
    only_one_by_model: dict[str, int] = defaultdict(int)
    flagged_by_model: dict[str, int] = defaultdict(int)
    for c in clusters:
        for m in c.models_flagged:
            flagged_by_model[m] += 1
        if c.agreement == 1:
            only_one_by_model[c.models_flagged[0]] += 1
    out.append("Per-model totals:")
    width = max(len(m) for m in models) if models else 0
    for m in models:
        out.append(
            f"  {m:<{width}}  {flagged_by_model.get(m, 0):>4} clusters flagged"
            f"  ({only_one_by_model.get(m, 0)} alone)"
        )
    out.append("")

    out.append(click.style("Clusters (highest agreement first):", bold=True))
    out.append("")
    for c in clusters:
        lo, hi = c.line_range
        if lo is None:
            line_str = "?"
        elif lo == hi:
            line_str = str(lo)
        else:
            line_str = f"{lo}-{hi}"
        header = click.style(
            f"[{c.agreement}/{c.eligible}] {c.file_path}:{line_str}  {c.cwe_id}",
            fg=("red" if c.agreement >= 2 else "white"),
            bold=True,
        )
        out.append(header)
        for f in c.findings:
            conf = (f.get("confidence") or "?").upper()
            conf_color = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "white"}.get(
                conf, "white"
            )
            verdict = f.get("verified_status")
            verdict_str = (
                f"  {click.style(verdict.upper().replace('_', ' '), fg='cyan')}"
                if verdict
                else ""
            )
            ln = f["line_number"] if f["line_number"] is not None else "?"
            expl = (f["explanation"] or "").splitlines()
            expl_first = expl[0] if expl else ""
            out.append(
                f"    {click.style('+', fg='green')} {f['model_id']:<24}"
                f" line {ln:<5} [{click.style(conf, fg=conf_color)}]"
                f"{verdict_str}"
            )
            if expl_first:
                out.append(f"      {expl_first}")
        for m in c.models_missed:
            out.append(f"    {click.style('-', fg='red')} {m:<24} did not flag")
        out.append("")

    return "\n".join(out)
