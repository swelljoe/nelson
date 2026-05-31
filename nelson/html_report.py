"""Generate static HTML reports from scan data."""

from __future__ import annotations

import json
from datetime import datetime
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

from .db import Database

if TYPE_CHECKING:
    from .score import CaseScore, LeaderboardEntry

# Shared CSS used by both report types
_CSS = """\
:root {
  --bg: #1a1a2e;
  --surface: #16213e;
  --surface2: #0f3460;
  --text: #e0e0e0;
  --text-muted: #8899aa;
  --accent: #e94560;
  --green: #4ecca3;
  --yellow: #f0c040;
  --orange: #e07020;
  --red: #e94560;
  --blue: #4ea8de;
  --cyan: #56cfe1;
  --border: #2a3a5e;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
  background: var(--bg); color: var(--text);
  line-height: 1.6; padding: 2rem;
  max-width: 1200px; margin: 0 auto;
}
h1 { color: var(--accent); margin-bottom: 0.5rem; font-size: 1.8rem; }
h2 { color: var(--blue); margin: 1.5rem 0 0.75rem; font-size: 1.3rem; }
h3 { color: var(--cyan); margin: 1rem 0 0.5rem; font-size: 1.1rem; }
.subtitle { color: var(--text-muted); margin-bottom: 1.5rem; }
table {
  width: 100%; border-collapse: collapse;
  margin: 0.75rem 0; background: var(--surface);
  border-radius: 6px; overflow: hidden;
}
th, td { padding: 0.5rem 0.75rem; text-align: left; border-bottom: 1px solid var(--border); }
th { background: var(--surface2); color: var(--blue); font-weight: 600; }
tr:last-child td { border-bottom: none; }
tr:hover { background: rgba(78, 168, 222, 0.05); }
.badge {
  display: inline-block; padding: 0.15rem 0.5rem;
  border-radius: 3px; font-size: 0.8rem; font-weight: 600;
}
.badge-high, .badge-confirmed { background: rgba(233, 69, 96, 0.2); color: var(--red); }
.badge-medium, .badge-needs_review { background: rgba(240, 192, 64, 0.2); color: var(--yellow); }
.badge-low, .badge-unreviewed { background: rgba(136, 153, 170, 0.2); color: var(--text-muted); }
.badge-false_positive, .badge-resolved { background: rgba(78, 204, 163, 0.2); color: var(--green); }
.badge-complete { background: rgba(78, 204, 163, 0.2); color: var(--green); }
.badge-error { background: rgba(233, 69, 96, 0.2); color: var(--red); }
.badge-running, .badge-pending { background: rgba(240, 192, 64, 0.2); color: var(--yellow); }
.card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 6px; padding: 1rem; margin: 0.75rem 0;
}
.finding { border-left: 3px solid var(--border); }
.finding-high { border-left-color: var(--red); }
.finding-medium { border-left-color: var(--yellow); }
.finding-low { border-left-color: var(--text-muted); }
.finding-header { display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; }
.code-block {
  background: var(--bg); padding: 0.5rem 0.75rem; margin: 0.5rem 0;
  border-radius: 4px; font-family: 'Fira Code', monospace;
  font-size: 0.85rem; overflow-x: auto; white-space: pre-wrap;
}
.review-box {
  margin-top: 0.5rem; padding: 0.5rem 0.75rem;
  background: var(--bg); border-radius: 4px;
  font-size: 0.9rem;
}
.stats-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 0.75rem; margin: 0.75rem 0;
}
.stat-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 6px; padding: 0.75rem; text-align: center;
}
.stat-value { font-size: 1.8rem; font-weight: 700; }
.stat-label { font-size: 0.8rem; color: var(--text-muted); }
.file-group { margin: 1rem 0; }
.file-name {
  font-family: monospace; font-size: 0.95rem;
  color: var(--cyan); padding: 0.5rem; cursor: pointer;
  background: var(--surface2); border-radius: 4px 4px 0 0;
}
footer {
  margin-top: 2rem; padding-top: 1rem;
  border-top: 1px solid var(--border);
  color: var(--text-muted); font-size: 0.8rem;
}
a { color: var(--blue); text-decoration: none; }
a:hover { text-decoration: underline; }
"""


def _badge(value: str | None) -> str:
    """Render a colored badge span."""
    if not value:
        return ""
    css_class = f"badge-{value}"
    return f'<span class="badge {css_class}">{escape(value.upper())}</span>'


def _format_tokens(n: int | None) -> str:
    if n is None:
        return "—"
    return f"{n:,}"


def _format_cost(c: float | None) -> str:
    if c is None:
        return "—"
    return f"${c:.4f}"


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def generate_scan_report(db: Database, scan_id: int) -> str:
    """Generate a detailed HTML report for a single scan."""
    s = db.get_scan(scan_id)
    if s is None:
        raise ValueError(f"Scan {scan_id} not found")

    config = json.loads(s["config"]) if s["config"] else {}
    mode = config.get("mode", "focused")
    models = ", ".join(config.get("models", []))
    findings = db.findings_for_scan(scan_id)
    counts = db.job_counts(scan_id)
    total_jobs = sum(counts.values())
    review_sum = db.review_summary(scan_id)
    usage = db.usage_by_model(scan_id)

    # Group findings by file
    by_file: dict[str, list] = {}
    for f in findings:
        by_file.setdefault(f["file_path"], []).append(f)

    # Count by confidence
    conf_counts: dict[str, int] = {}
    for f in findings:
        c = f["confidence"] or "unknown"
        conf_counts[c] = conf_counts.get(c, 0) + 1

    # Build HTML
    parts: list[str] = []
    parts.append(f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nelson Scan Report — Scan {scan_id}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Nelson Scan Report</h1>
<p class="subtitle">Scan {scan_id} — {escape(mode)} mode — {escape(s["target_dir"])}</p>
""")

    # Overview stats
    confirmed = review_sum.get("confirmed", 0)
    false_pos = review_sum.get("false_positive", 0)
    needs = review_sum.get("needs_review", 0)
    unreviewed = review_sum.get("unreviewed", 0)

    parts.append('<div class="stats-grid">')
    parts.append(_stat_card(str(len(findings)), "Total Findings", "var(--accent)"))
    parts.append(_stat_card(str(confirmed), "Confirmed", "var(--red)"))
    parts.append(_stat_card(str(false_pos), "False Positive", "var(--green)"))
    parts.append(_stat_card(str(needs), "Needs Review", "var(--yellow)"))
    parts.append(_stat_card(str(unreviewed), "Unreviewed", "var(--text-muted)"))
    parts.append(_stat_card(str(total_jobs), "Jobs Run", "var(--blue)"))
    parts.append("</div>")

    # Scan metadata
    parts.append("<h2>Scan Details</h2>")
    parts.append("<table>")
    parts.append(f"<tr><th>Target</th><td>{escape(s['target_dir'])}</td></tr>")
    parts.append(f"<tr><th>Commit</th><td>{escape(s['commit_sha'] or 'N/A')}</td></tr>")
    parts.append(f"<tr><th>Mode</th><td>{escape(mode)}</td></tr>")
    parts.append(f"<tr><th>Models</th><td>{escape(models)}</td></tr>")
    parts.append(f"<tr><th>Started</th><td>{escape(s['started_at'] or '')}</td></tr>")
    parts.append(
        "<tr><th>Completed</th>"
        f"<td>{escape(s['completed_at'] or 'in progress')}</td></tr>"
    )
    parts.append("</table>")

    # Confidence breakdown
    if conf_counts:
        parts.append("<h2>Findings by Confidence</h2>")
        parts.append("<table><tr><th>Confidence</th><th>Count</th></tr>")
        for c in ["high", "medium", "low", "unknown"]:
            if c in conf_counts:
                parts.append(f"<tr><td>{_badge(c)}</td><td>{conf_counts[c]}</td></tr>")
        parts.append("</table>")

    # Token usage
    if usage:
        parts.append("<h2>Token Usage</h2>")
        parts.append(
            "<table><tr><th>Model</th><th>Jobs</th>"
            "<th>Tokens In</th><th>Tokens Out</th>"
            "<th>Cost</th></tr>"
        )
        for row in usage:
            parts.append(
                f"<tr><td>{escape(row['model_id'])}</td>"
                f"<td>{row['jobs']}</td>"
                f"<td>{_format_tokens(row['total_tokens_in'])}</td>"
                f"<td>{_format_tokens(row['total_tokens_out'])}</td>"
                f"<td>{_format_cost(row['total_cost_usd'])}</td></tr>"
            )
        parts.append("</table>")

    # Findings by file
    parts.append(f"<h2>Findings ({len(findings)})</h2>")
    if not findings:
        parts.append('<p class="card">No findings.</p>')
    for filepath in sorted(by_file):
        file_findings = by_file[filepath]
        parts.append('<div class="file-group">')
        parts.append(
            f'<div class="file-name">{escape(filepath)}'
            f" ({len(file_findings)} finding"
            f"{'s' if len(file_findings) != 1 else ''})</div>"
        )
        for f in file_findings:
            conf = f["confidence"] or "unknown"
            parts.append(f'<div class="card finding finding-{conf}">')
            parts.append('<div class="finding-header">')
            parts.append(_badge(conf))
            parts.append(
                f'<span style="color: var(--orange)">{escape(f["cwe_id"])}</span>'
            )
            if f["line_number"]:
                parts.append(
                    f'<span style="color: var(--text-muted)">'
                    f"line {f['line_number']}</span>"
                )
            parts.append(
                f'<span style="color: var(--text-muted); font-size: 0.8rem">'
                f"by {escape(f['model_id'])}</span>"
            )
            if f["verified_status"]:
                parts.append(_badge(f["verified_status"]))
            parts.append("</div>")  # finding-header

            if f["explanation"]:
                parts.append(f"<p>{escape(f['explanation'])}</p>")
            if f["code_snippet"]:
                parts.append(
                    f'<div class="code-block">{escape(f["code_snippet"])}</div>'
                )

            if f["verified_status"] and f["suggested_fix"]:
                parts.append('<div class="review-box">')
                parts.append(
                    f"<strong>Review ({escape(f['verified_by_model'] or '')}):"
                    "</strong> "
                    f"{escape(f['suggested_fix'])}"
                )
                parts.append("</div>")

            parts.append("</div>")  # card
        parts.append("</div>")  # file-group

    parts.append(f"""\
<footer>
Generated by <a href="https://github.com/swelljoe/nelson">Nelson</a>
on {_timestamp()}
</footer>
</body>
</html>""")

    return "\n".join(parts)


def generate_summary_report(db: Database) -> str:
    """Generate an executive summary HTML report across all scans."""
    scans = db.list_scans()

    # Aggregate stats
    total_findings = 0
    total_confirmed = 0
    total_false_pos = 0
    total_needs_review = 0
    scan_rows: list[dict] = []

    for s in scans:
        scan_id = s["id"]
        config = json.loads(s["config"]) if s["config"] else {}
        mode = config.get("mode", "focused")
        models = ", ".join(config.get("models", []))
        findings = db.findings_for_scan(scan_id)
        counts = db.job_counts(scan_id)
        total_jobs = sum(counts.values())
        review_sum = db.review_summary(scan_id)
        usage = db.usage_by_model(scan_id)

        confirmed = review_sum.get("confirmed", 0)
        false_pos = review_sum.get("false_positive", 0)
        needs = review_sum.get("needs_review", 0)
        unreviewed = review_sum.get("unreviewed", 0)

        total_findings += len(findings)
        total_confirmed += confirmed
        total_false_pos += false_pos
        total_needs_review += needs

        # Token totals for this scan
        tokens_in = 0
        tokens_out = 0
        cost = 0.0
        for row in usage:
            tokens_in += row["total_tokens_in"] or 0
            tokens_out += row["total_tokens_out"] or 0
            cost += row["total_cost_usd"] or 0.0

        status = "complete" if s["completed_at"] else "in progress"

        scan_rows.append(
            {
                "id": scan_id,
                "target": s["target_dir"],
                "commit": s["commit_sha"] or "",
                "mode": mode,
                "models": models,
                "status": status,
                "started": s["started_at"] or "",
                "total_jobs": total_jobs,
                "findings": len(findings),
                "confirmed": confirmed,
                "false_pos": false_pos,
                "needs": needs,
                "unreviewed": unreviewed,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost": cost,
                "errors": counts.get("error", 0),
            }
        )

    parts: list[str] = []
    parts.append(f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nelson Executive Summary</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Nelson Executive Summary</h1>
<p class="subtitle">{len(scans)} scan{"s" if len(scans) != 1 else ""} — generated {_timestamp()}</p>
""")

    # Top-level stats
    parts.append('<div class="stats-grid">')
    parts.append(_stat_card(str(total_findings), "Total Findings", "var(--accent)"))
    parts.append(_stat_card(str(total_confirmed), "Confirmed", "var(--red)"))
    parts.append(_stat_card(str(total_false_pos), "False Positive", "var(--green)"))
    parts.append(_stat_card(str(total_needs_review), "Needs Review", "var(--yellow)"))
    parts.append(_stat_card(str(len(scans)), "Scans", "var(--blue)"))
    parts.append("</div>")

    if not scans:
        parts.append('<p class="card">No scans found.</p>')
    else:
        # Scan overview table
        parts.append("<h2>All Scans</h2>")
        parts.append(
            "<table><tr>"
            "<th>ID</th><th>Target</th><th>Mode</th>"
            "<th>Models</th><th>Status</th><th>Jobs</th>"
            "<th>Findings</th><th>Confirmed</th>"
            "<th>False Pos</th><th>Needs Review</th>"
            "<th>Tokens</th><th>Cost</th>"
            "</tr>"
        )
        for r in scan_rows:
            total_tokens = r["tokens_in"] + r["tokens_out"]
            parts.append(
                f"<tr>"
                f"<td>{r['id']}</td>"
                f"<td>{escape(_short_path(r['target']))}</td>"
                f"<td>{escape(r['mode'])}</td>"
                f"<td>{escape(r['models'])}</td>"
                f"<td>{_badge(r['status'])}</td>"
                f"<td>{r['total_jobs']}</td>"
                f"<td><strong>{r['findings']}</strong></td>"
                f"<td>{_confirmed_cell(r['confirmed'])}</td>"
                f"<td>{r['false_pos']}</td>"
                f"<td>{_needs_cell(r['needs'])}</td>"
                f"<td>{_format_tokens(total_tokens)}</td>"
                f"<td>{_format_cost(r['cost'] if r['cost'] else None)}</td>"
                f"</tr>"
            )
        parts.append("</table>")

        # Per-scan detail cards
        parts.append("<h2>Scan Details</h2>")
        for r in scan_rows:
            parts.append('<div class="card">')
            parts.append(
                f"<h3>Scan {r['id']} — {escape(r['mode'])} — "
                f"{escape(_short_path(r['target']))}</h3>"
            )
            parts.append(
                f"<p>Models: {escape(r['models'])} · "
                f"Jobs: {r['total_jobs']} · "
                f"Errors: {r['errors']}</p>"
            )
            if r["commit"]:
                parts.append(
                    f'<p style="color: var(--text-muted)">'
                    f"Commit: {escape(r['commit'][:12])}</p>"
                )

            parts.append('<div class="stats-grid">')
            parts.append(_stat_card(str(r["findings"]), "Findings", "var(--accent)"))
            parts.append(_stat_card(str(r["confirmed"]), "Confirmed", "var(--red)"))
            parts.append(
                _stat_card(str(r["false_pos"]), "False Positive", "var(--green)")
            )
            parts.append(_stat_card(str(r["needs"]), "Needs Review", "var(--yellow)"))
            parts.append(
                _stat_card(str(r["unreviewed"]), "Unreviewed", "var(--text-muted)")
            )
            parts.append("</div>")

            # Top findings for this scan (confirmed ones)
            findings = db.findings_for_scan(r["id"])
            confirmed_findings = [
                f for f in findings if f["verified_status"] == "confirmed"
            ]
            if confirmed_findings:
                parts.append(
                    "<h3 style='margin-top: 0.75rem'>"
                    f"Confirmed Findings ({len(confirmed_findings)})</h3>"
                )
                for f in confirmed_findings[:10]:
                    conf = f["confidence"] or "unknown"
                    parts.append(f'<div class="card finding finding-{conf}">')
                    parts.append(
                        f"{_badge(conf)} "
                        f'<span style="color: var(--orange)">'
                        f"{escape(f['cwe_id'])}</span> "
                        f"in <code>{escape(f['file_path'])}</code>"
                    )
                    if f["line_number"]:
                        parts.append(f" line {f['line_number']}")
                    if f["explanation"]:
                        parts.append(f"<p>{escape(f['explanation'])}</p>")
                    parts.append("</div>")
                if len(confirmed_findings) > 10:
                    parts.append(
                        f'<p style="color: var(--text-muted)">'
                        f"… and {len(confirmed_findings) - 10} more."
                        " See the detailed report.</p>"
                    )

            parts.append("</div>")  # card

    parts.append(f"""\
<footer>
Generated by <a href="https://github.com/swelljoe/nelson">Nelson</a>
on {_timestamp()}
</footer>
</body>
</html>""")

    return "\n".join(parts)


def _stat_card(value: str, label: str, color: str) -> str:
    return (
        f'<div class="stat-card">'
        f'<div class="stat-value" style="color: {color}">{value}</div>'
        f'<div class="stat-label">{label}</div></div>'
    )


def _short_path(path: str) -> str:
    """Shorten a path for display, keeping last 2-3 components."""
    p = Path(path)
    parts = p.parts
    if len(parts) <= 3:
        return str(p)
    return str(Path(*parts[-3:]))


def _confirmed_cell(n: int) -> str:
    if n > 0:
        return f'<strong style="color: var(--red)">{n}</strong>'
    return str(n)


def _needs_cell(n: int) -> str:
    if n > 0:
        return f'<span style="color: var(--yellow)">{n}</span>'
    return str(n)


_COMPARE_EXTRA_CSS = """\
.cluster {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 6px; padding: 0.75rem 1rem; margin: 0.75rem 0;
  border-left: 4px solid var(--text-muted);
}
.cluster-strong { border-left-color: var(--red); }
.cluster-partial { border-left-color: var(--yellow); }
.cluster-singleton { border-left-color: var(--text-muted); }
.cluster-header {
  display: flex; gap: 0.75rem; align-items: center;
  flex-wrap: wrap; margin-bottom: 0.5rem;
}
.cluster-loc {
  font-family: monospace; font-size: 0.95rem; color: var(--cyan);
}
.agreement-pill {
  display: inline-block; padding: 0.15rem 0.55rem;
  border-radius: 999px; font-weight: 700; font-size: 0.85rem;
  background: var(--surface2); color: var(--blue);
}
.agreement-strong { background: rgba(233, 69, 96, 0.2); color: var(--red); }
.agreement-partial { background: rgba(240, 192, 64, 0.2); color: var(--yellow); }
.flag-row {
  display: grid;
  grid-template-columns: 1.2rem 14rem 4rem auto auto;
  gap: 0.5rem; padding: 0.25rem 0;
  align-items: baseline; font-size: 0.9rem;
}
.flag-row + .flag-row { border-top: 1px dashed var(--border); }
.flag-row .marker { font-weight: 700; }
.flag-row .marker.flagged { color: var(--green); }
.flag-row .marker.missed { color: var(--red); }
.flag-row .model { font-family: monospace; color: var(--cyan); }
.flag-row .line { color: var(--text-muted); font-family: monospace; }
.flag-row .expl { grid-column: 2 / -1; color: var(--text-muted); font-size: 0.85rem; padding-left: 0.5rem; }
"""


def _agreement_class(agreement: int, eligible: int) -> str:
    if eligible >= 2 and agreement == eligible:
        return "strong"
    if agreement >= 2:
        return "partial"
    return "singleton"


def generate_compare_report(scan_ids, models, clusters) -> str:
    """Generate an HTML model-comparison report.

    Args:
        scan_ids: list[int] of scans being compared
        models: sorted list of distinct model_ids that voted
        clusters: list[compare.Cluster] already filtered/sorted
    """
    n_models = len(models)
    label = (
        f"Scan {scan_ids[0]}"
        if len(scan_ids) == 1
        else f"Scans {', '.join(str(i) for i in scan_ids)}"
    )

    # Aggregates
    agreement_hist: dict[int, int] = {}
    flagged_by_model: dict[str, int] = {m: 0 for m in models}
    only_one_by_model: dict[str, int] = {m: 0 for m in models}
    for c in clusters:
        agreement_hist[c.agreement] = agreement_hist.get(c.agreement, 0) + 1
        for m in c.models_flagged:
            flagged_by_model[m] = flagged_by_model.get(m, 0) + 1
        if c.agreement == 1 and c.models_flagged:
            only_one_by_model[c.models_flagged[0]] = (
                only_one_by_model.get(c.models_flagged[0], 0) + 1
            )

    strong = sum(1 for c in clusters if c.eligible >= 2 and c.agreement == c.eligible)
    partial = sum(1 for c in clusters if c.agreement >= 2 and c.agreement < c.eligible)
    singletons = sum(1 for c in clusters if c.agreement == 1)

    parts: list[str] = []
    parts.append(f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nelson Model Comparison — {escape(label)}</title>
<style>{_CSS}{_COMPARE_EXTRA_CSS}</style>
</head>
<body>
<h1>Model Comparison</h1>
<p class="subtitle">{escape(label)} — {n_models} models — {len(clusters)} clusters</p>
""")

    parts.append('<div class="stats-grid">')
    parts.append(_stat_card(str(strong), "All models agreed", "var(--red)"))
    parts.append(_stat_card(str(partial), "Partial agreement", "var(--yellow)"))
    parts.append(_stat_card(str(singletons), "Only one model", "var(--text-muted)"))
    parts.append(_stat_card(str(len(clusters)), "Total clusters", "var(--blue)"))
    parts.append("</div>")

    # Per-model totals
    parts.append("<h2>Per-model totals</h2>")
    parts.append("<table>")
    parts.append(
        "<tr><th>Model</th><th>Clusters flagged</th><th>Flagged alone</th></tr>"
    )
    for m in models:
        parts.append(
            f"<tr><td>{escape(m)}</td>"
            f"<td>{flagged_by_model.get(m, 0)}</td>"
            f"<td>{_needs_cell(only_one_by_model.get(m, 0))}</td></tr>"
        )
    parts.append("</table>")

    # Clusters
    parts.append("<h2>Clusters (highest agreement first)</h2>")
    if not clusters:
        parts.append("<p>No findings match the filters.</p>")
    for c in clusters:
        klass = _agreement_class(c.agreement, c.eligible)
        lo, hi = c.line_range
        if lo is None:
            line_str = "?"
        elif lo == hi:
            line_str = str(lo)
        else:
            line_str = f"{lo}-{hi}"

        pill_class = (
            "agreement-strong"
            if klass == "strong"
            else ("agreement-partial" if klass == "partial" else "")
        )
        parts.append(f'<div class="cluster cluster-{klass}">')
        parts.append('<div class="cluster-header">')
        parts.append(
            f'<span class="agreement-pill {pill_class}">'
            f"{c.agreement} / {c.eligible}</span>"
        )
        parts.append(
            f'<span class="cluster-loc">{escape(c.file_path)}:{escape(line_str)}</span>'
        )
        parts.append(f"<span>{escape(c.cwe_id)}</span>")
        parts.append("</div>")

        for f in c.findings:
            conf = (f.get("confidence") or "").lower()
            verdict = f.get("verified_status")
            ln = str(f["line_number"]) if f.get("line_number") is not None else "?"
            expl = f.get("explanation") or ""
            parts.append('<div class="flag-row">')
            parts.append('<span class="marker flagged">+</span>')
            parts.append(f'<span class="model">{escape(f["model_id"])}</span>')
            parts.append(f'<span class="line">L{escape(ln)}</span>')
            parts.append(_badge(conf) if conf else "")
            parts.append(_badge(verdict) if verdict else "")
            if expl:
                parts.append(
                    f'<div class="expl">{escape(expl[:600])}'
                    f"{'…' if len(expl) > 600 else ''}</div>"
                )
            parts.append("</div>")
        for m in c.models_missed:
            parts.append('<div class="flag-row">')
            parts.append('<span class="marker missed">-</span>')
            parts.append(f'<span class="model">{escape(m)}</span>')
            parts.append('<span class="line">—</span>')
            parts.append('<span style="color: var(--text-muted)">did not flag</span>')
            parts.append("<span></span>")
            parts.append("</div>")
        parts.append("</div>")

    parts.append(
        f"<footer>Generated by Nelson at {_timestamp()}</footer></body></html>"
    )
    return "\n".join(parts)


# -- Benchmark leaderboard + Pareto (P5) -------------------------------------

_LEADERBOARD_EXTRA_CSS = """\
.lead-rank { color: var(--text-muted); text-align: right; width: 2rem; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.frontier-star { color: var(--green); font-weight: 700; }
.muted { color: var(--text-muted); }
.pareto-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
  gap: 1rem; margin: 0.75rem 0;
}
.pareto-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 6px; padding: 0.75rem;
}
.pareto-card h3 { margin-top: 0; }
.scatter-pt { fill: var(--blue); }
.scatter-pt-frontier { fill: var(--green); }
.scatter-label { fill: var(--text-muted); font-size: 11px; font-family: monospace; }
.scatter-axis { stroke: var(--border); stroke-width: 1; }
.scatter-tick { fill: var(--text-muted); font-size: 10px; }
.scatter-frontier-line { stroke: var(--green); stroke-width: 1.5; fill: none;
  stroke-dasharray: 4 3; opacity: 0.7; }
.matrix td, .matrix th { text-align: center; }
.matrix th.comp, .matrix td.comp { text-align: left; font-family: monospace; }
.cell-hit { color: var(--green); font-weight: 700; }
.cell-miss { color: var(--text-muted); }
.cell-jerr { color: var(--yellow); }
.cell-refu { color: var(--cyan); font-style: italic; }
.cell-excl { color: var(--orange); }
.cell-none { color: var(--border); }
.token-bar { fill: var(--blue); opacity: 0.85; }
"""


def _pct(v: float | None) -> str:
    return f"{v:.0%}" if v is not None else "—"


def _money2(v: float | None) -> str:
    return f"${v:.2f}" if v is not None else "—"


def _secs(v: float | None) -> str:
    return f"{v:.0f}s" if v is not None else "—"


def _tok(v: float | None) -> str:
    """Compact token count: 1.2M / 340k / 980."""
    if v is None:
        return "—"
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if v >= 1_000:
        return f"{v / 1_000:.0f}k"
    return f"{v:.0f}"


def _token_bars_svg(entries: list[LeaderboardEntry]) -> str:
    """Horizontal bar chart of tokens/case per competitor (descending), each bar
    annotated with its latency/case. Linear scale, so the heavy brute-forcers
    dominate visually and the light models read as short bars — the point is to
    expose the order-of-magnitude spread. Pure string building, no JS/assets.
    """
    pts = [
        (_display_name(e.competitor_name), e.tokens_per_case, e.latency_per_case)
        for e in entries
        if e.tokens_per_case is not None
    ]
    if not pts:
        return '<p class="muted">No token usage recorded.</p>'
    pts.sort(key=lambda p: p[1], reverse=True)

    row_h, mt, mb, ml = 26, 10, 26, 200
    width = 720
    height = mt + mb + row_h * len(pts)
    bar_x = ml + 6
    bar_w_max = width - bar_x - 70  # leave room for the value label
    vmax = max(p[1] for p in pts) or 1.0

    s: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        'aria-label="tokens per case per competitor">'
    ]
    for i, (name, tpc, lat) in enumerate(pts):
        y = mt + i * row_h
        cy = y + row_h / 2
        bw = max(1.0, (tpc / vmax) * bar_w_max)
        label = name if len(name) <= 30 else name[:29] + "…"
        s.append(
            f'<text class="scatter-label" x="{ml - 6}" y="{cy + 3:.1f}" '
            f'text-anchor="end">{escape(label)}</text>'
        )
        s.append(
            f'<rect class="token-bar" x="{bar_x}" y="{y + 4:.1f}" '
            f'width="{bw:.1f}" height="{row_h - 10}" rx="2"/>'
        )
        lat_str = f" · {lat:.0f}s" if lat is not None else ""
        s.append(
            f'<text class="scatter-tick" x="{bar_x + bw + 6:.1f}" '
            f'y="{cy + 3:.1f}">{escape(_tok(tpc))}{escape(lat_str)}</text>'
        )
    s.append("</svg>")
    return "".join(s)


def _scatter_svg(
    entries: list[LeaderboardEntry],
    *,
    x_fn,
    y_fn,
    x_label: str,
    x_fmt,
    frontier: set[str],
) -> str:
    """Inline-SVG scatter of quality (y, 0..1) vs an economic axis (x, minimized).

    Frontier competitors are drawn green and joined by a dashed line; the rest are
    muted blue. Pure string building — no JS, no external assets. Points with a
    missing coordinate (no audited cases) are skipped.
    """
    pts: list[tuple[str, float, float, bool]] = []
    for e in entries:
        xv, yv = x_fn(e), y_fn(e)
        if xv is None or yv is None:
            continue
        pts.append(
            (
                _display_name(e.competitor_name),
                float(xv),
                float(yv),
                e.competitor_name in frontier,
            )
        )
    if not pts:
        return (
            f'<p class="muted">No plottable data for quality vs {escape(x_label)}.</p>'
        )

    width, height = 540, 340
    ml, mr, mt, mb = 58, 156, 16, 46  # wide right margin for labels
    pw, ph = width - ml - mr, height - mt - mb
    xs = [p[1] for p in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = 0.0, 1.0

    def sx(v: float) -> float:
        if xmax == xmin:
            return ml + pw / 2
        return ml + (v - xmin) / (xmax - xmin) * pw

    def sy(v: float) -> float:
        return mt + (1 - (v - ymin) / (ymax - ymin)) * ph

    s: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" '
        f'role="img" aria-label="quality vs {escape(x_label)}">'
    ]
    # Axes.
    s.append(
        f'<line class="scatter-axis" x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}"/>'
    )
    s.append(
        f'<line class="scatter-axis" x1="{ml}" y1="{mt + ph}" '
        f'x2="{ml + pw}" y2="{mt + ph}"/>'
    )
    # Y ticks (quality 0 / 0.5 / 1).
    for q in (0.0, 0.5, 1.0):
        y = sy(q)
        s.append(
            f'<line class="scatter-axis" x1="{ml - 4}" y1="{y:.1f}" '
            f'x2="{ml}" y2="{y:.1f}"/>'
        )
        s.append(
            f'<text class="scatter-tick" x="{ml - 8}" y="{y + 3:.1f}" '
            f'text-anchor="end">{q:.1f}</text>'
        )
    s.append(
        f'<text class="scatter-tick" x="14" y="{mt + ph / 2:.1f}" '
        f'transform="rotate(-90 14 {mt + ph / 2:.1f})" '
        'text-anchor="middle">quality (det x prec)</text>'
    )
    # X ticks (min / max of the economic axis).
    for v in sorted({xmin, xmax}):
        x = sx(v)
        s.append(
            f'<line class="scatter-axis" x1="{x:.1f}" y1="{mt + ph}" '
            f'x2="{x:.1f}" y2="{mt + ph + 4}"/>'
        )
        s.append(
            f'<text class="scatter-tick" x="{x:.1f}" y="{mt + ph + 16}" '
            f'text-anchor="middle">{escape(x_fmt(v))}</text>'
        )
    s.append(
        f'<text class="scatter-tick" x="{ml + pw / 2:.1f}" y="{height - 6}" '
        f'text-anchor="middle">{escape(x_label)} (lower is better →)</text>'
    )
    # Frontier line (frontier points sorted by x).
    fpts = sorted([p for p in pts if p[3]], key=lambda p: p[1])
    if len(fpts) >= 2:
        poly = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for _, x, y, _ in fpts)
        s.append(f'<polyline class="scatter-frontier-line" points="{poly}"/>')
    # Points + labels.
    for name, x, y, on_front in pts:
        cx, cy = sx(x), sy(y)
        cls = "scatter-pt-frontier" if on_front else "scatter-pt"
        r = 5 if on_front else 4
        s.append(f'<circle class="{cls}" cx="{cx:.1f}" cy="{cy:.1f}" r="{r}"/>')
        label = name if len(name) <= 22 else name[:21] + "…"
        s.append(
            f'<text class="scatter-label" x="{cx + 8:.1f}" y="{cy + 3:.1f}">'
            f"{escape(label)}</text>"
        )
    s.append("</svg>")
    return "".join(s)


_OUTCOME_CELL = {
    "hit": ("cell-hit", "HIT"),
    "miss": ("cell-miss", "miss"),
    "judge_error": ("cell-jerr", "jerr"),
    "refused": ("cell-refu", "refu"),
    "excluded": ("cell-excl", "excl"),
}


# Friendly leaderboard labels. The runtime prefix (raw-api-loop/, claude-code/) is
# noise in the report — every non-Claude model runs in the API loop, so its prefix
# carries no information — so we drop it. A few competitors are registered under a
# bare model nickname; map those to their versioned product name so every row reads
# at the same level of detail. Display-only: the stored competitor_name (and every
# DB/frontier/matrix lookup keyed on it) is unchanged.
_DISPLAY_OVERRIDES = {
    "claude-code/haiku": "haiku-4.5",
    "claude-code/sonnet": "sonnet-4.6",
    "claude-code/opus": "opus-4.8",
    "raw-api-loop/deepseek": "deepseek-v4-pro",
    "raw-api-loop/mimo": "mimo-v2.5-pro",
}


def _display_name(name: str) -> str:
    """Render a competitor's leaderboard label: explicit override if one exists,
    otherwise drop the leading ``runtime/`` prefix."""
    if name in _DISPLAY_OVERRIDES:
        return _DISPLAY_OVERRIDES[name]
    return name.split("/", 1)[1] if "/" in name else name


# Competitors that audited far fewer than the full corpus are omitted from the
# Pareto charts (frontier + scatter): their quality is measured over a smaller,
# self-selected case subset (so it is not commensurable with full-corpus rows),
# and a cost-capped probe would otherwise stretch the cost axis until every
# full-corpus competitor collapses into one indistinguishable cluster. The
# threshold is a fraction of the fullest run's case count, so a model that is
# only a case or two short (e.g. one uncompletable case) is still plotted; a
# deliberately partial probe like a 4/9 cost-capped run is dropped. Excluded
# competitors keep their asterisked row in the leaderboard table above.
_FRONTIER_MIN_COVERAGE = 0.75


def generate_leaderboard_report(
    entries: list[LeaderboardEntry],
    case_scores: list[CaseScore],
    *,
    title: str = "Nelson Benchmark Leaderboard",
) -> str:
    """Render the benchmark leaderboard: ranked table, Pareto scatter plots, and a
    per-case competitor x case drilldown matrix.

    ``entries`` is :func:`nelson.score.leaderboard` output (already ranked);
    ``case_scores`` is :func:`nelson.score.case_scores` output (for the matrix).
    Cost/latency reflect competitor spend only — judge spend is shown apart so it
    never colors the Pareto trade-off.
    """
    from .score import pareto_frontier

    # The Pareto frontier and scatter plots compare only competitors audited on
    # enough of the corpus to be commensurable — substantially-partial probes are
    # excluded so they neither claim a frontier ★ nor distort the chart axes.
    full_n = max((e.cases for e in entries), default=0)
    plotted = [
        e for e in entries if not full_n or e.cases >= full_n * _FRONTIER_MIN_COVERAGE
    ]
    excluded_from_plots = [e for e in entries if e not in plotted]

    cost_front = {
        e.competitor_name
        for e in pareto_frontier(
            plotted, x=lambda e: e.cost_per_case, y=lambda e: e.quality
        )
    }
    lat_front = {
        e.competitor_name
        for e in pareto_frontier(
            plotted, x=lambda e: e.latency_per_case, y=lambda e: e.quality
        )
    }

    total_cases = sum(e.cases for e in entries)
    total_hits = sum(e.hits for e in entries)
    total_comp_cost = sum(e.competitor_cost for e in entries)
    total_judge_cost = sum(e.judge_cost + e.fp_cost for e in entries)

    parts: list[str] = []
    parts.append(f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>{_CSS}{_LEADERBOARD_EXTRA_CSS}</style>
</head>
<body>
<h1>{escape(title)}</h1>
<p class="subtitle">{len(entries)} competitors — {total_hits} case hits across \
{total_cases} audited cases</p>
""")

    parts.append('<div class="stats-grid">')
    parts.append(_stat_card(str(len(entries)), "Competitors", "var(--blue)"))
    parts.append(_stat_card(str(total_cases), "Cases audited", "var(--cyan)"))
    parts.append(_stat_card(str(total_hits), "Case hits", "var(--green)"))
    parts.append(
        _stat_card(_money2(total_comp_cost), "Competitor spend", "var(--yellow)")
    )
    parts.append(
        _stat_card(_money2(total_judge_cost), "Judge spend", "var(--text-muted)")
    )
    parts.append("</div>")

    # Leaderboard table.
    parts.append("<h2>Leaderboard</h2>")
    parts.append(
        "<table><tr>"
        '<th class="lead-rank">#</th><th>Competitor</th><th>Size</th>'
        '<th class="num">Detect</th><th class="num">Hits/Elig</th>'
        '<th class="num">Precision</th><th class="num">FP/case</th>'
        '<th class="num" title="Confirmed real bugs the model found that are not '
        "the planted target CVE — credited capability, but does not count toward "
        'detection">Other real</th>'
        '<th class="num">Cost/case</th><th class="num">Latency</th>'
        '<th class="num" title="Mean total tokens (prompt + completion) reported '
        "per audited case — only as honest as the provider's usage metering "
        '(see note below the chart)">Tokens/case</th>'
        '<th class="num">Cases</th>'
        "</tr>"
    )
    # A competitor that completed fewer than the fullest run's case count has a
    # detection rate over a partial denominator — flag it so its rate is not read
    # as rank-comparable with full-corpus competitors (see the footnote below).
    # (full_n is computed once near the top of this function.)
    for i, e in enumerate(entries, 1):
        star = (
            ' <span class="frontier-star" title="On a Pareto frontier">★</span>'
            if e.competitor_name in cost_front or e.competitor_name in lat_front
            else ""
        )
        partial = (
            f'<sup style="color: var(--yellow)" title="Partial coverage: audited '
            f'{e.cases} of {full_n} cases — see note below">*</sup>'
            if full_n and e.cases < full_n
            else ""
        )
        parts.append(
            f'<tr><td class="lead-rank">{i}</td>'
            f"<td>{escape(_display_name(e.competitor_name))}{partial}{star}</td>"
            f"<td>{escape(e.size_class or '—')}</td>"
            f'<td class="num">{_pct(e.detection_rate) if e.eligible else "—"}{partial}</td>'
            f'<td class="num">{e.hits}/{e.eligible}</td>'
            f'<td class="num">{_pct(e.precision)}</td>'
            f'<td class="num">{f"{e.fp_per_case:.2f}" if e.fp_per_case is not None else "—"}</td>'
            f'<td class="num">{e.real_others if e.real_others else "—"}</td>'
            f'<td class="num">{_money2(e.cost_per_case)}</td>'
            f'<td class="num">{_secs(e.latency_per_case)}</td>'
            f'<td class="num">{_tok(e.tokens_per_case)}</td>'
            f'<td class="num">{e.cases}</td></tr>'
        )
    parts.append("</table>")
    parts.append(
        '<p class="muted">Detect = case hits / eligible (hits + genuine misses); '
        "undetermined, refused, and auth/infra-excluded cases are not in the "
        "denominator. "
        "Precision = true findings / (true + false positives). Other real = "
        "confirmed real bugs the model found that are not the planted target CVE "
        "(extra capability, but not counted as detection). Cost/latency are "
        "the competitor's own spend per audited case. ★ = on a Pareto "
        "frontier below.</p>"
    )
    if any(full_n and e.cases < full_n for e in entries):
        parts.append(
            '<p class="muted">* Partial coverage: this competitor completed fewer '
            f"than the full {full_n} cases (see the Cases column). Its detection "
            "rate is therefore based on fewer audited cases and is not directly "
            "rank-comparable with full-corpus competitors — read it alongside the "
            "Cases count, not the rank.</p>"
        )

    # Pareto scatter plots.
    parts.append("<h2>Pareto frontier</h2>")
    parts.append('<div class="pareto-grid">')
    parts.append('<div class="pareto-card"><h3>Quality vs cost / case</h3>')
    parts.append(
        _scatter_svg(
            plotted,
            x_fn=lambda e: e.cost_per_case,
            y_fn=lambda e: e.quality,
            x_label="cost / case",
            x_fmt=lambda v: f"${v:.2f}",
            frontier=cost_front,
        )
    )
    parts.append("</div>")
    parts.append('<div class="pareto-card"><h3>Quality vs latency / case</h3>')
    parts.append(
        _scatter_svg(
            plotted,
            x_fn=lambda e: e.latency_per_case,
            y_fn=lambda e: e.quality,
            x_label="latency / case",
            x_fmt=lambda v: f"{v:.0f}s",
            frontier=lat_front,
        )
    )
    parts.append("</div></div>")
    parts.append(
        '<p class="muted">Quality = detection rate x precision (precision treated '
        "as 1.0 when a competitor reported no scorable findings). Green points are "
        "non-dominated — no other competitor is at least as good on quality while "
        "also cheaper/faster. Size is shown in the table; it is categorical, so it "
        "is not used as a numeric Pareto axis.</p>"
    )
    if excluded_from_plots:
        names = ", ".join(
            f"{escape(_display_name(e.competitor_name))} ({e.cases}/{full_n})"
            for e in excluded_from_plots
        )
        parts.append(
            f'<p class="muted">These charts omit {names}: a competitor that '
            f"audited fewer than {round(_FRONTIER_MIN_COVERAGE * 100)}% of the "
            f"{full_n} cases measures its quality over a smaller, self-selected "
            "subset, so the point is not comparable with the full-corpus "
            "competitors — and a cost-capped probe sits so far out on the cost "
            "axis that every other competitor collapses into one indistinguishable "
            "cluster, making the trade-off unreadable. Its position would also "
            "<em>imply</em> a quality ranking the partial run does not establish. "
            "It remains in the leaderboard table above (asterisked).</p>"
        )

    # Tokens (+ latency) per case — exposes the order-of-magnitude usage spread.
    parts.append("<h2>Tokens &amp; time per case</h2>")
    parts.append(_token_bars_svg(entries))
    parts.append(
        '<p class="muted">Mean total tokens (prompt + completion, with the '
        "ReAct loop's resent context counted each turn) per audited case; the "
        "trailing number is mean latency/case. Bars are linear, so brute-force "
        "models dwarf frugal ones. <strong>Data-quality caveat:</strong> these are "
        "the tokens the provider's API <em>reported</em> — some OpenAI-compatible "
        "endpoints under-report usage (a near-zero bar with input ≈ output is the "
        "tell), so a suspiciously short bar may mean broken metering rather than a "
        "frugal model, and that competitor's cost/case is then an underestimate.</p>"
    )

    # Per-case drilldown matrix.
    by_cell: dict[tuple[str, str], str] = {}
    cases: set[str] = set()
    for cs in case_scores:
        by_cell[(cs.competitor_name, cs.case_ext_id)] = cs.outcome
        cases.add(cs.case_ext_id)
    if by_cell:
        ordered_comps = [e.competitor_name for e in entries]
        # include any competitor present only in case_scores (defensive)
        for cs in case_scores:
            if cs.competitor_name not in ordered_comps:
                ordered_comps.append(cs.competitor_name)
        ordered_cases = sorted(cases)
        parts.append("<h2>Per-case results</h2>")
        parts.append('<div style="overflow-x:auto"><table class="matrix"><tr>')
        parts.append('<th class="comp">Competitor</th>')
        for c in ordered_cases:
            parts.append(f"<th>{escape(c)}</th>")
        parts.append("</tr>")
        for comp in ordered_comps:
            parts.append(f'<tr><td class="comp">{escape(_display_name(comp))}</td>')
            for c in ordered_cases:
                outcome = by_cell.get((comp, c))
                if outcome is None:
                    parts.append('<td class="cell-none">·</td>')
                else:
                    cls, txt = _OUTCOME_CELL.get(outcome, ("cell-none", outcome))
                    parts.append(f'<td class="{cls}">{escape(txt)}</td>')
            parts.append("</tr>")
        parts.append("</table></div>")
        parts.append(
            '<p class="muted">HIT = detected; miss = looked, found nothing; '
            "jerr = judge undetermined (out of denominator); refu = model refused "
            "the task (out of denominator, never a miss); excl = auth/infra "
            "failure (never a miss); · = not run.</p>"
        )

    parts.append(
        f"<footer>Generated by Nelson at {_timestamp()}</footer></body></html>"
    )
    return "\n".join(parts)
