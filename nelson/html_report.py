"""Generate static HTML reports from scan data."""

import json
from datetime import datetime
from html import escape
from pathlib import Path

from .db import Database

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


def _agreement_class(agreement: int, total_models: int) -> str:
    if total_models >= 2 and agreement >= total_models:
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

    strong = sum(
        1 for c in clusters if c.eligible >= 2 and c.agreement == c.eligible
    )
    partial = sum(
        1 for c in clusters if c.agreement >= 2 and c.agreement < c.eligible
    )
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
        klass = _agreement_class(c.agreement, n_models)
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
