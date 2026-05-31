"""Nelson CLI: AI-driven vulnerability scanner orchestrator."""

import json
import logging
import sys
from pathlib import Path

import click

from . import compare as compare_mod
from .db import Database
from .html_report import (
    generate_compare_report,
    generate_scan_report,
    generate_summary_report,
)
from .inventory import discover_files, files_from_paths
from .review import run_review
from .scanner import create_scan, run_scan
from .tooling import assess_tooling, format_tooling_report


def _db_path() -> Path:
    return Path("nelson.db")


def _resolve_paths(paths: tuple[str, ...]):
    """Resolve a variadic PATHS argument into (target_dir, files).

    A single directory is walked with the usual filters; one or more
    files (e.g. from a shell glob) go through ``files_from_paths``.
    Exits with an error message if anything is wrong.
    """
    if not paths:
        click.echo("Error: provide one or more files or a directory.", err=True)
        sys.exit(1)

    if len(paths) == 1 and Path(paths[0]).is_dir():
        target_dir = paths[0]
        files = discover_files(target_dir)
    else:
        try:
            files, root = files_from_paths(list(paths))
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        target_dir = str(root)

    return target_dir, files


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
def main(verbose: bool):
    """Nelson: finding vulnerabilities through dumb brute force."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


@main.command()
@click.argument("paths", type=click.Path(exists=True), nargs=-1)
def inventory(paths: tuple[str, ...]):
    """List source files that would be scanned.

    PATHS can be a single directory, a single file, or multiple files
    (e.g. from a shell glob like ``src/*.py``).
    """
    target_dir, files = _resolve_paths(paths)
    if not files:
        click.echo("No source files found.")
        return

    by_lang: dict[str, list] = {}
    total_size = 0
    for f in files:
        by_lang.setdefault(f.language, []).append(f)
        total_size += f.size

    click.echo(f"Found {len(files)} source files ({total_size / 1024:.0f} KB total):\n")
    for lang in sorted(by_lang):
        lang_files = by_lang[lang]
        click.echo(f"  {lang}: {len(lang_files)} files")

    click.echo("\nFiles:")
    for f in files:
        click.echo(f"  [{f.language:>12}] {f.path} ({f.size / 1024:.1f} KB)")

    # Security tooling assessment
    languages = {f.language for f in files}
    statuses = assess_tooling(target_dir, languages)
    if statuses:
        click.echo(f"\n{format_tooling_report(statuses)}")


@main.command()
@click.argument("paths", type=click.Path(exists=True), nargs=-1)
@click.option(
    "-m",
    "--model",
    "models",
    multiple=True,
    default=["claude:haiku"],
    help="Model spec, e.g. 'claude:haiku', 'openai:qwen3.5'. Repeatable.",
)
@click.option(
    "--cwe",
    "cwe_ids",
    multiple=True,
    default=None,
    help="Limit to specific CWE IDs (e.g. --cwe CWE-89). Default: all.",
)
@click.option(
    "--delay",
    default=2.0,
    type=float,
    help="Seconds between jobs (pacing). Default: 2.0",
)
@click.option(
    "--resume",
    type=int,
    default=None,
    help="Resume a previous scan by ID instead of creating a new one.",
)
@click.option(
    "--mode",
    type=click.Choice(["focused", "open"]),
    default="focused",
    help="'focused': per-CWE (default). 'open': find any vuln per file.",
)
@click.option(
    "--parallel/--no-parallel",
    default=True,
    help=(
        "With multiple -m models, run one worker per model concurrently"
        " (default). Single-model scans are unaffected."
    ),
)
@click.option(
    "--db",
    "db_path",
    default="nelson.db",
    help="Path to SQLite database. Default: nelson.db",
)
def scan(
    paths: tuple[str, ...],
    models: tuple[str],
    cwe_ids: tuple[str],
    delay: float,
    resume: int | None,
    mode: str,
    parallel: bool,
    db_path: str,
):
    """Scan PATHS for vulnerabilities.

    PATHS can be a single directory (the default — every source file
    under it is scanned, with the usual filters), a single file, or
    multiple files (e.g. from a shell glob like ``src/*.py``).
    """
    db = Database(db_path)
    cwe_list = list(cwe_ids) if cwe_ids else None

    if resume:
        scan_id = resume
        s = db.get_scan(scan_id)
        if s is None:
            click.echo(f"Scan {scan_id} not found.", err=True)
            sys.exit(1)
        target_dir = s["target_dir"]
        config = json.loads(s["config"]) if s["config"] else {}
        models = tuple(config.get("models", models))
        mode = config.get("mode", mode)
        click.echo(f"Resuming scan {scan_id} ({mode} mode) on {target_dir}")
    else:
        if not paths:
            click.echo(
                "Error: provide one or more files/directories, or --resume.",
                err=True,
            )
            sys.exit(1)

        if mode == "open" and cwe_ids:
            click.echo("Warning: --cwe flags are ignored in open mode.", err=True)
            cwe_list = None

        target_dir, files = _resolve_paths(paths)

        if not files:
            click.echo("No source files found.", err=True)
            sys.exit(1)

        scan_id, files = create_scan(
            db,
            target_dir,
            list(models),
            cwe_ids=cwe_list,
            mode=mode,
            files=files,
        )
        counts = db.job_counts(scan_id)
        total = sum(counts.values())
        click.echo(f"Scan {scan_id} ({mode} mode): {len(files)} files, {total} jobs")

    def on_progress(counts):
        done = counts.get("complete", 0) + counts.get("error", 0)
        total = sum(counts.values())
        pending = counts.get("pending", 0)
        errors = counts.get("error", 0)
        click.echo(
            f"  Progress: {done}/{total} done, {pending} pending, {errors} errors",
            err=True,
        )

    if parallel and len(models) > 1:
        click.echo(f"Starting scan ({len(models)} models in parallel)...")
    else:
        click.echo("Starting scan...")
    run_scan(
        db,
        scan_id,
        list(models),
        target_dir,
        delay=delay,
        on_progress=on_progress,
        parallel=parallel,
        db_path=db_path,
    )

    counts = db.job_counts(scan_id)
    click.echo(f"\nScan {scan_id} complete: {counts}")

    # Quick summary of findings
    findings = db.findings_for_scan(scan_id)
    if findings:
        click.echo(
            f"\n{len(findings)} potential vulnerabilities found."
            f" Run 'nelson report {scan_id}' for details."
        )
    else:
        click.echo("\nNo vulnerabilities found.")


@main.command()
@click.argument("scan_id", type=int, required=False)
@click.option(
    "-m",
    "--model",
    "model_spec",
    default="claude:sonnet",
    help="Model to use for review. Default: claude:sonnet",
)
@click.option(
    "--delay",
    default=2.0,
    type=float,
    help="Seconds between reviews (for CLI pacing). Default: 2.0",
)
@click.option("--db", "db_path", default="nelson.db", help="Path to SQLite database.")
def review(scan_id: int | None, model_spec: str, delay: float, db_path: str):
    """Review findings from a scan with a (usually smarter) model."""
    db = Database(db_path)
    if scan_id is None:
        s = db.latest_scan()
        if s is None:
            click.echo("No scans found.")
            return
        scan_id = s["id"]
    else:
        s = db.get_scan(scan_id)
        if s is None:
            click.echo(f"Scan {scan_id} not found.", err=True)
            sys.exit(1)

    target_dir = s["target_dir"]

    unreviewed = db.unreviewed_findings(scan_id)
    if not unreviewed:
        click.echo(f"Scan {scan_id}: no unreviewed findings.")
        return

    click.echo(
        f"Reviewing {len(unreviewed)} findings from scan {scan_id} with {model_spec}..."
    )

    def on_progress(reviewed, total):
        click.echo(f"  Reviewed {reviewed}/{total}", err=True)

    run_review(
        db,
        scan_id,
        model_spec,
        target_dir,
        delay=delay,
        on_progress=on_progress,
    )

    summary = db.review_summary(scan_id)
    click.echo("\nReview complete:")
    for status, count in sorted(summary.items()):
        click.echo(f"  {status}: {count}")


@main.command(name="list")
@click.option("--db", "db_path", default="nelson.db", help="Path to SQLite database.")
def list_scans(db_path: str):
    """List all scans."""
    db = Database(db_path)
    scans = db.list_scans()
    if not scans:
        click.echo("No scans found.")
        return

    hdr = (
        f"{'ID':>4}  {'Status':<11}  {'Mode':<8}"
        f"  {'Findings':>8}  {'Jobs':>6}  {'Model':<35}  {'Target'}"
    )
    sep = (
        f"{'─' * 4}  {'─' * 11}  {'─' * 8}"
        f"  {'─' * 8}  {'─' * 6}  {'─' * 35}  {'─' * 30}"
    )
    click.echo(hdr)
    click.echo(sep)
    for s in scans:
        scan_id = s["id"]
        status = "complete" if s["completed_at"] else "in progress"
        counts = db.job_counts(scan_id)
        total_jobs = sum(counts.values())
        findings = db.findings_for_scan(scan_id)
        target = s["target_dir"]
        config = json.loads(s["config"]) if s["config"] else {}
        models = ", ".join(config.get("models", []))
        mode = config.get("mode", "focused")
        click.echo(
            f"{scan_id:>4}  {status:<11}  {mode:<8}"
            f"  {len(findings):>8}  {total_jobs:>6}"
            f"  {models:<35}  {target}"
        )


@main.command()
@click.argument("scan_id", type=int, required=False)
@click.option("--db", "db_path", default="nelson.db", help="Path to SQLite database.")
def status(scan_id: int | None, db_path: str):
    """Show status of a scan (default: latest)."""
    db = Database(db_path)
    if scan_id is None:
        s = db.latest_scan()
        if s is None:
            click.echo("No scans found.")
            return
        scan_id = s["id"]
    else:
        s = db.get_scan(scan_id)
        if s is None:
            click.echo(f"Scan {scan_id} not found.", err=True)
            sys.exit(1)

    click.echo(f"Scan {scan_id}")
    click.echo(f"  Target:    {s['target_dir']}")
    click.echo(f"  Commit:    {s['commit_sha'] or 'N/A'}")
    click.echo(f"  Started:   {s['started_at']}")
    click.echo(f"  Completed: {s['completed_at'] or 'in progress'}")

    counts = db.job_counts(scan_id)
    total = sum(counts.values())
    click.echo(f"  Jobs:      {total} total")
    for status_name, count in sorted(counts.items()):
        click.echo(f"    {status_name}: {count}")

    findings = db.findings_for_scan(scan_id)
    click.echo(f"  Findings:  {len(findings)}")

    review_sum = db.review_summary(scan_id)
    if any(k != "unreviewed" for k in review_sum):
        click.echo("  Reviews:")
        for rs, count in sorted(review_sum.items()):
            click.echo(f"    {rs}: {count}")

    usage = db.usage_by_model(scan_id)
    if usage:
        click.echo("\n  Token usage by model:")
        for row in usage:
            tin = row["total_tokens_in"] or 0
            tout = row["total_tokens_out"] or 0
            total = tin + tout
            line = (
                f"    {row['model_id']}: {total:,} tokens"
                f" ({tin:,} in, {tout:,} out)"
                f" across {row['jobs']} jobs"
            )
            if row["total_cost_usd"]:
                line += f" — ${row['total_cost_usd']:.4f}"
            click.echo(line)


@main.command()
@click.argument("scan_id", type=int, required=False)
@click.option("--db", "db_path", default="nelson.db", help="Path to SQLite database.")
@click.option(
    "--confidence",
    type=click.Choice(["high", "medium", "low"]),
    default=None,
    help="Filter by confidence.",
)
@click.option("--cwe", "cwe_filter", default=None, help="Filter by CWE ID.")
@click.option(
    "--verdict",
    type=click.Choice(
        ["confirmed", "false_positive", "needs_review", "resolved", "unreviewed"]
    ),
    default=None,
    help="Filter by review verdict.",
)
@click.option("--json-output", "as_json", is_flag=True, help="Output as JSON.")
def report(
    scan_id: int | None,
    db_path: str,
    confidence: str | None,
    cwe_filter: str | None,
    verdict: str | None,
    as_json: bool,
):
    """Show findings for a scan (default: latest)."""
    db = Database(db_path)
    if scan_id is None:
        s = db.latest_scan()
        if s is None:
            click.echo("No scans found.")
            return
        scan_id = s["id"]

    findings = db.findings_for_scan(scan_id)

    if confidence:
        findings = [f for f in findings if f["confidence"] == confidence]
    if cwe_filter:
        findings = [f for f in findings if f["cwe_id"] == cwe_filter]
    if verdict:
        if verdict == "unreviewed":
            findings = [f for f in findings if f["verified_status"] is None]
        else:
            findings = [f for f in findings if f["verified_status"] == verdict]

    if as_json:
        data = [
            {
                "file": f["file_path"],
                "cwe": f["cwe_id"],
                "line": f["line_number"],
                "confidence": f["confidence"],
                "explanation": f["explanation"],
                "code": f["code_snippet"],
                "model": f["model_id"],
                "verdict": f["verified_status"],
                "reviewed_by": f["verified_by_model"],
                "review_detail": f["suggested_fix"],
            }
            for f in findings
        ]
        click.echo(json.dumps(data, indent=2))
        return

    # Tooling assessment (if we can resolve the target dir)
    s = db.get_scan(scan_id)
    if s and s["target_dir"] and Path(s["target_dir"]).is_dir():
        scan_files = discover_files(s["target_dir"])
        languages = {f.language for f in scan_files}
        statuses = assess_tooling(s["target_dir"], languages)
        if statuses:
            click.echo(format_tooling_report(statuses))
            click.echo("")

    if not findings:
        click.echo("No findings match the filters.")
        return

    click.echo(f"Scan {scan_id}: {len(findings)} findings\n")

    current_file = None
    for f in findings:
        if f["file_path"] != current_file:
            current_file = f["file_path"]
            click.echo(click.style(f"\n{'=' * 60}", fg="blue"))
            click.echo(click.style(f"  {current_file}", fg="blue", bold=True))
            click.echo(click.style(f"{'=' * 60}", fg="blue"))

        conf_colors = {"high": "red", "medium": "yellow", "low": "white"}
        conf = f["confidence"] or "unknown"
        conf_style = click.style(
            conf.upper(),
            fg=conf_colors.get(conf, "white"),
            bold=True,
        )

        click.echo(f"\n  [{conf_style}] {f['cwe_id']} (found by {f['model_id']})")
        if f["line_number"]:
            click.echo(f"  Line: {f['line_number']}")
        if f["code_snippet"]:
            click.echo(f"  Code: {f['code_snippet']}")
        if f["explanation"]:
            click.echo(f"  Why:  {f['explanation']}")

        # Show review status if reviewed
        vs = f["verified_status"]
        if vs:
            verdict_colors = {
                "confirmed": "red",
                "false_positive": "green",
                "needs_review": "yellow",
                "resolved": "cyan",
            }
            verdict_style = click.style(
                vs.upper().replace("_", " "),
                fg=verdict_colors.get(vs, "white"),
                bold=True,
            )
            click.echo(f"  Review: {verdict_style} (by {f['verified_by_model']})")
            if f["suggested_fix"]:
                for line in f["suggested_fix"].splitlines():
                    click.echo(f"    {line}")

    # Summary
    click.echo(f"\n{'─' * 60}")
    summary = db.findings_summary(scan_id)
    if summary:
        click.echo("\nSummary by CWE:")
        for row in summary:
            click.echo(
                f"  {row['cwe_id']} ({row['model_id']}):"
                f" {row['count']} ({row['confidence']})"
            )


@main.command(name="html-report")
@click.argument("scan_id", type=int, required=False)
@click.option("--db", "db_path", default="nelson.db", help="Path to SQLite database.")
@click.option(
    "-o",
    "--output",
    "output_path",
    default=None,
    help="Output file path. Default: nelson-report-<scan_id>.html",
)
def html_report(scan_id: int | None, db_path: str, output_path: str | None):
    """Generate a detailed HTML report for a scan (default: latest)."""
    db = Database(db_path)
    if scan_id is None:
        s = db.latest_scan()
        if s is None:
            click.echo("No scans found.")
            return
        scan_id = s["id"]

    html = generate_scan_report(db, scan_id)
    out = Path(output_path or f"nelson-report-{scan_id}.html")
    out.write_text(html)
    click.echo(f"Wrote {out} ({len(html):,} bytes)")


@main.command(name="html-summary")
@click.option("--db", "db_path", default="nelson.db", help="Path to SQLite database.")
@click.option(
    "-o",
    "--output",
    "output_path",
    default=None,
    help="Output file path. Default: nelson-summary.html",
)
def html_summary(db_path: str, output_path: str | None):
    """Generate an executive summary HTML report across all scans."""
    db = Database(db_path)
    html = generate_summary_report(db)
    out = Path(output_path or "nelson-summary.html")
    out.write_text(html)
    click.echo(f"Wrote {out} ({len(html):,} bytes)")


def _resolve_compare_scans(
    db: Database,
    scan_id: int | None,
    scans: str | None,
) -> list[int]:
    """Validate and return the scan_ids to compare.

    Cross-scan compare requires the scans share target_dir + commit_sha
    (within reason — a missing commit_sha doesn't disqualify, but two
    different SHAs do).
    """
    if scans:
        try:
            ids = [int(s.strip()) for s in scans.split(",") if s.strip()]
        except ValueError:
            click.echo(
                "Error: --scans must be a comma-separated list of integers.",
                err=True,
            )
            sys.exit(1)
        if not ids:
            click.echo("Error: --scans requires at least one scan id.", err=True)
            sys.exit(1)
        rows = []
        for i in ids:
            r = db.get_scan(i)
            if r is None:
                click.echo(f"Scan {i} not found.", err=True)
                sys.exit(1)
            rows.append(r)
        targets = {r["target_dir"] for r in rows}
        if len(targets) > 1:
            click.echo(
                "Error: --scans must reference scans of the same target_dir."
                f" Got: {sorted(targets)}",
                err=True,
            )
            sys.exit(1)
        commits = {r["commit_sha"] for r in rows if r["commit_sha"]}
        if len(commits) > 1:
            click.echo(
                "Warning: scans were taken at different commits"
                f" ({sorted(commits)}); line numbers may not align.",
                err=True,
            )
        return ids

    if scan_id is None:
        s = db.latest_scan()
        if s is None:
            click.echo("No scans found.", err=True)
            sys.exit(1)
        scan_id = s["id"]
    elif db.get_scan(scan_id) is None:
        click.echo(f"Scan {scan_id} not found.", err=True)
        sys.exit(1)
    return [scan_id]


def _build_comparison(
    db: Database,
    scan_ids: list[int],
    line_tolerance: int,
    cwe: str | None,
    confidence: str | None,
    min_agreement: int | None,
):
    findings = db.findings_for_scans(scan_ids)
    focused_cov, open_cov = db.coverage_for_scans(scan_ids)
    clusters = compare_mod.cluster_findings(findings, line_tolerance=line_tolerance)
    compare_mod.annotate_clusters(clusters, focused_cov, open_cov)
    clusters = compare_mod.filter_clusters(
        clusters, cwe=cwe, confidence=confidence, min_agreement=min_agreement
    )
    clusters = compare_mod.sort_clusters(clusters)
    models = compare_mod.all_models(focused_cov, open_cov)
    return clusters, models


@main.command()
@click.argument("scan_id", type=int, required=False)
@click.option(
    "--scans",
    default=None,
    help=(
        "Comma-separated scan ids to compare across (e.g. --scans 3,5,7)."
        " Scans must share target_dir."
    ),
)
@click.option(
    "--line-tolerance",
    default=2,
    type=int,
    help="Line-number window for treating findings as the same issue. Default: 2",
)
@click.option(
    "--cwe",
    "cwe_filter",
    default=None,
    help="Restrict to a single CWE (e.g. --cwe CWE-89).",
)
@click.option(
    "--confidence",
    type=click.Choice(["high", "medium", "low"]),
    default=None,
    help="Restrict to clusters where at least one finding has this confidence.",
)
@click.option(
    "--min-agreement",
    default=None,
    type=int,
    help="Only show clusters where at least N models agreed.",
)
@click.option("--json-output", "as_json", is_flag=True, help="Output as JSON.")
@click.option("--db", "db_path", default="nelson.db", help="Path to SQLite database.")
def compare(
    scan_id: int | None,
    scans: str | None,
    line_tolerance: int,
    cwe_filter: str | None,
    confidence: str | None,
    min_agreement: int | None,
    as_json: bool,
    db_path: str,
):
    """Compare findings across models to see where they agree.

    With a single SCAN_ID, compares the models used in that one scan.
    With --scans 3,5,7, compares across separate scans on the same target.
    """
    db = Database(db_path)
    scan_ids = _resolve_compare_scans(db, scan_id, scans)
    clusters, models = _build_comparison(
        db, scan_ids, line_tolerance, cwe_filter, confidence, min_agreement
    )
    if as_json:
        click.echo(compare_mod.to_json(clusters, models, scan_ids))
        return
    if not clusters:
        click.echo("No findings match the filters.")
        return
    click.echo(compare_mod.render_text(clusters, models, scan_ids))


@main.command(name="html-compare")
@click.argument("scan_id", type=int, required=False)
@click.option(
    "--scans",
    default=None,
    help="Comma-separated scan ids (must share target_dir).",
)
@click.option("--line-tolerance", default=2, type=int)
@click.option("--cwe", "cwe_filter", default=None)
@click.option(
    "--confidence", type=click.Choice(["high", "medium", "low"]), default=None
)
@click.option("--min-agreement", default=None, type=int)
@click.option(
    "-o",
    "--output",
    "output_path",
    default=None,
    help="Output file path. Default: nelson-compare-<id>.html",
)
@click.option("--db", "db_path", default="nelson.db", help="Path to SQLite database.")
def html_compare(
    scan_id: int | None,
    scans: str | None,
    line_tolerance: int,
    cwe_filter: str | None,
    confidence: str | None,
    min_agreement: int | None,
    output_path: str | None,
    db_path: str,
):
    """Generate an HTML model-comparison report."""
    db = Database(db_path)
    scan_ids = _resolve_compare_scans(db, scan_id, scans)
    clusters, models = _build_comparison(
        db, scan_ids, line_tolerance, cwe_filter, confidence, min_agreement
    )
    html = generate_compare_report(scan_ids, models, clusters)
    if output_path:
        out = Path(output_path)
    elif len(scan_ids) == 1:
        out = Path(f"nelson-compare-{scan_ids[0]}.html")
    else:
        out = Path("nelson-compare-" + "-".join(str(i) for i in scan_ids) + ".html")
    out.write_text(html)
    click.echo(f"Wrote {out} ({len(html):,} bytes)")


@main.command()
@click.argument("paths", type=click.Path(exists=True), nargs=-1)
@click.option(
    "--focused-model",
    default="claude:haiku",
    help="Model for focused CWE scan. Default: claude:haiku",
)
@click.option(
    "--open-model",
    default="claude:sonnet",
    help="Model for open scan. Default: claude:sonnet",
)
@click.option(
    "--review-model",
    default="claude:sonnet",
    help="Model for review pass. Default: claude:sonnet",
)
@click.option(
    "--delay",
    default=2.0,
    type=float,
    help="Seconds between jobs (pacing). Default: 2.0",
)
@click.option(
    "--db",
    "db_path",
    default="nelson.db",
    help="Path to SQLite database. Default: nelson.db",
)
def haha(
    paths: tuple[str, ...],
    focused_model: str,
    open_model: str,
    review_model: str,
    delay: float,
    db_path: str,
):
    """Run the full pipeline: focused scan, open scan, review, report.

    PATHS can be a single directory, a single file, or multiple files
    (e.g. from a shell glob like ``src/*.py``).

    This runs both scan modes, reviews all findings, and prints a
    final report. It's convenient but token-intensive on large projects.
    """
    db = Database(db_path)

    target_dir, files = _resolve_paths(paths)
    if not files:
        click.echo("No source files found.", err=True)
        sys.exit(1)

    click.echo(f"Found {len(files)} source files.\n")

    # -- Step 1: Focused scan --
    click.echo(click.style("=== Focused scan ===", bold=True))
    focused_id, _ = create_scan(
        db,
        target_dir,
        [focused_model],
        mode="focused",
        files=files,
    )
    focused_counts = db.job_counts(focused_id)
    focused_total = sum(focused_counts.values())
    click.echo(
        f"Scan {focused_id}: {len(files)} files,"
        f" {focused_total} jobs with {focused_model}"
    )

    def on_scan_progress(counts):
        done = counts.get("complete", 0) + counts.get("error", 0)
        total = sum(counts.values())
        click.echo(f"  {done}/{total}", err=True)

    run_scan(
        db,
        focused_id,
        [focused_model],
        target_dir,
        delay=delay,
        on_progress=on_scan_progress,
        db_path=db_path,
    )
    focused_findings = db.findings_for_scan(focused_id)
    click.echo(f"  {len(focused_findings)} findings\n")

    # -- Step 2: Open scan --
    click.echo(click.style("=== Open scan ===", bold=True))
    open_id, _ = create_scan(
        db,
        target_dir,
        [open_model],
        mode="open",
        files=files,
    )
    open_counts = db.job_counts(open_id)
    open_total = sum(open_counts.values())
    click.echo(
        f"Scan {open_id}: {len(files)} files, {open_total} jobs with {open_model}"
    )

    run_scan(
        db,
        open_id,
        [open_model],
        target_dir,
        delay=delay,
        on_progress=on_scan_progress,
        db_path=db_path,
    )
    open_findings = db.findings_for_scan(open_id)
    click.echo(f"  {len(open_findings)} findings\n")

    # -- Step 3: Review both scans --
    click.echo(click.style("=== Review ===", bold=True))
    for sid, label in [(focused_id, "focused"), (open_id, "open")]:
        unreviewed = db.unreviewed_findings(sid)
        if not unreviewed:
            click.echo(f"  {label} scan {sid}: nothing to review")
            continue
        click.echo(
            f"  Reviewing {len(unreviewed)} findings"
            f" from {label} scan {sid} with {review_model}..."
        )

        def on_review_progress(reviewed, total):
            click.echo(f"    {reviewed}/{total}", err=True)

        run_review(
            db,
            sid,
            review_model,
            target_dir,
            delay=delay,
            on_progress=on_review_progress,
        )

    click.echo()

    # -- Step 4: Summary --
    click.echo(click.style("=== Results ===", bold=True))
    for sid, label in [(focused_id, "focused"), (open_id, "open")]:
        review_sum = db.review_summary(sid)
        confirmed = review_sum.get("confirmed", 0)
        false_pos = review_sum.get("false_positive", 0)
        needs = review_sum.get("needs_review", 0)
        total_findings = sum(review_sum.values())
        click.echo(
            f"  {label} scan {sid}: {total_findings} findings"
            f" — {confirmed} confirmed,"
            f" {false_pos} false positive,"
            f" {needs} needs review"
        )

    click.echo(
        f"\nRun 'nelson report {focused_id} --verdict confirmed'"
        f" or 'nelson report {open_id} --verdict confirmed'"
        " for details."
    )


# -- Benchmark corpus --------------------------------------------------------

CVD_PAYLOAD_URL = "https://red.anthropic.com/2026/cvd/data/payload.json"


@main.group()
def corpus():
    """Build and inspect the benchmark vulnerability corpus."""


@corpus.command(name="import")
@click.option(
    "--url", default=CVD_PAYLOAD_URL, help="CVD payload.json URL to seed from."
)
@click.option("--db", "db_path", default="nelson.db", help="Path to SQLite database.")
def corpus_import(url: str, db_path: str):
    """Seed candidate cases from an Anthropic CVD payload."""
    from .corpus import CVDSeedSource, import_seeds
    from .enrich import HttpxClient, fetch_cvd_payload

    db = Database(db_path)
    try:
        payload = fetch_cvd_payload(url, HttpxClient())
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    n = import_seeds(CVDSeedSource(payload), db)
    click.echo(f"Seeded {n} cases from {url}")
    _echo_status_counts(db)


@corpus.command()
@click.option("--db", "db_path", default="nelson.db", help="Path to SQLite database.")
@click.option("--cache-dir", default="corpus-cache", help="Where repos are fetched.")
@click.option("--judge-model", default="opus", help="Model for the pre-vet judge.")
@click.option("--vet-threshold", default=0.5, type=float, help="Min confidence to vet.")
@click.option("--limit", default=0, type=int, help="Max candidates per pass (0=all).")
@click.option("--no-enrich", is_flag=True, help="Skip OSV/NVD enrichment.")
@click.option("--no-derive", is_flag=True, help="Skip git pre-patch derivation.")
@click.option("--no-vet", is_flag=True, help="Skip the Opus pre-vet judge.")
def build(
    db_path: str,
    cache_dir: str,
    judge_model: str,
    vet_threshold: float,
    limit: int,
    no_enrich: bool,
    no_derive: bool,
    no_vet: bool,
):
    """Advance candidate cases: enrich -> derive -> pre-vet (all idempotent)."""
    from .corpus import CorpusPipeline
    from .derive import SubprocessGitRunner
    from .enrich import HttpxClient, NVDEnricher, OSVEnricher
    from .prevet import ClaudeCLIJudge

    http = HttpxClient()
    enrichers = [] if no_enrich else [OSVEnricher(http), NVDEnricher(http)]
    git = None if no_derive else SubprocessGitRunner()
    judge = None if no_vet else ClaudeCLIJudge(model=judge_model)

    pipe = CorpusPipeline(
        Database(db_path),
        enrichers=enrichers,
        git=git,
        judge=judge,
        cache_dir=cache_dir,
        vet_threshold=vet_threshold,
    )
    summary = pipe.build(limit=limit or None)
    click.echo("Build pass complete:")
    for k, v in summary.as_dict().items():
        click.echo(f"  {k}: {v}")


@corpus.command(name="list")
@click.option("--status", default=None, help="Filter by status.")
@click.option("--source", default=None, help="Filter by source (cvd/osv/manual).")
@click.option("--db", "db_path", default="nelson.db", help="Path to SQLite database.")
def corpus_list(status: str | None, source: str | None, db_path: str):
    """List corpus cases."""
    db = Database(db_path)
    rows = db.list_cases(status=status, source=source)
    if not rows:
        click.echo("No cases.")
        return
    click.echo(f"{'EXT_ID':<22} {'STATUS':<10} {'PROJECT':<16} {'CWE':<10} FIX")
    for r in rows:
        fix = (r["fix_commit"] or "")[:12]
        click.echo(
            f"{r['ext_id']:<22} {r['status']:<10} "
            f"{(r['project'] or '-'):<16} {(r['cwe'] or '-'):<10} {fix or '-'}"
        )
    _echo_status_counts(db)


@corpus.command()
@click.argument("ext_id")
@click.option("--db", "db_path", default="nelson.db", help="Path to SQLite database.")
def show(ext_id: str, db_path: str):
    """Show a single case in full, including derived ground truth."""
    db = Database(db_path)
    row = db.get_case(ext_id)
    if row is None:
        click.echo(f"Case {ext_id} not found.", err=True)
        sys.exit(1)
    for key in row.keys():  # noqa: SIM118 — sqlite3.Row needs .keys() for columns
        value = row[key]
        if key in ("gt_files", "gt_hunks") and value:
            value = json.loads(value)
        click.echo(f"{key:>16}: {value}")


@corpus.command()
@click.option("--cases-dir", default="cases", help="Directory to write manifests into.")
@click.option("--status", default="vetted", help="Which cases to export.")
@click.option("--db", "db_path", default="nelson.db", help="Path to SQLite database.")
def export(cases_dir: str, status: str, db_path: str):
    """Pin cases to cases/<ext_id>.yaml (default: the vetted set)."""
    from .corpus import CorpusPipeline

    pipe = CorpusPipeline(Database(db_path))
    n = pipe.export_manifest(cases_dir, status=status)
    click.echo(f"Exported {n} {status} case(s) to {cases_dir}/")


def _echo_status_counts(db: Database):
    counts = db.count_cases_by_status()
    if counts:
        summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        click.echo(f"Corpus: {summary}")


# -- Benchmark runs ----------------------------------------------------------


def _parse_competitor(spec: str):
    """Build a Competitor from a ``runtime:model`` (or bare ``model``) spec.

    ``claude-code:sonnet`` -> runtime=claude-code, model=sonnet; a bare
    ``sonnet`` defaults the runtime to claude-code (the only P2 runtime).
    """
    from .runner import Competitor

    spec = spec.strip()
    if ":" in spec:
        runtime, model = spec.split(":", 1)
        runtime, model = runtime.strip(), model.strip()
        if not runtime:
            raise click.ClickException(f"competitor spec '{spec}' has no runtime")
    else:
        runtime, model = "claude-code", spec
    if not model:
        raise click.ClickException(f"competitor spec '{spec}' has no model")
    return Competitor(name=f"{runtime}/{model}", model=model, runtime=runtime)


def _resolve_case(ext_id: str, db: Database, manifest_dir: str | None):
    """Find a case by ext_id in the DB, or in a manifest dir if given."""
    from .corpus import Case, load_manifest_dir

    row = db.get_case(ext_id)
    if row is not None:
        return Case.from_row(row)
    if manifest_dir:
        for case in load_manifest_dir(manifest_dir):
            if case.ext_id == ext_id:
                return case
    raise click.ClickException(
        f"case {ext_id} not found in {db.path}"
        + (f" or manifests at {manifest_dir}" if manifest_dir else "")
    )


@main.group()
def bench():
    """Run competitors against corpus cases in isolated containers."""


@bench.command(name="run")
@click.argument("ext_id")
@click.option(
    "--competitor",
    "competitor_spec",
    default="claude-code:sonnet",
    help="Competitor as runtime:model, e.g. claude-code:sonnet.",
)
@click.option("--db", "db_path", default="nelson.db", help="Path to SQLite database.")
@click.option("--cache-dir", default="bench-cache", help="Where source is checked out.")
@click.option("--runs-dir", default="bench-runs", help="Where transcripts are written.")
@click.option(
    "--from-manifest",
    default=None,
    help="Also resolve the case from this cases/ manifest dir if not in the DB.",
)
@click.option(
    "--file",
    "target_file",
    default=None,
    help="Audit only this file (default: every vulnerable file in the case).",
)
@click.option(
    "--no-network", is_flag=True, help="Run the container with no network (offline)."
)
@click.option(
    "--timeout", default=1800.0, type=float, help="Per-run wall-clock cap (s)."
)
@click.option(
    "--max-budget-usd",
    default=0.50,
    type=float,
    help="Per-run cost backstop passed to claude -p (0 to disable).",
)
def bench_run(
    ext_id: str,
    competitor_spec: str,
    db_path: str,
    cache_dir: str,
    runs_dir: str,
    from_manifest: str | None,
    target_file: str | None,
    no_network: bool,
    timeout: float,
    max_budget_usd: float,
):
    """Point one competitor at case EXT_ID's vulnerable file(s), one run each.

    A run audits a single file (the model is told which file, never what bug). By
    default every vulnerable (non-test) file in the case is run; --file scopes to
    one. The case counts as detected (P3) if any of its file-runs is a hit.
    """
    from .runner import BenchRunner, vulnerable_files

    db = Database(db_path)
    competitor = _parse_competitor(competitor_spec)
    case = _resolve_case(ext_id, db, from_manifest)
    files = [target_file] if target_file else vulnerable_files(case)
    if not files:
        raise click.ClickException(
            f"case {case.ext_id} has no vulnerable files to audit (not derived?)"
        )

    runner = BenchRunner(
        db,
        cache_dir=cache_dir,
        runs_dir=runs_dir,
        network=not no_network,
        run_timeout=timeout,
        max_budget_usd=max_budget_usd or None,
    )
    click.echo(
        f"Running {competitor.name} against {case.ext_id} ({case.project}) — "
        f"{len(files)} file(s)…"
    )
    worst = 0
    for f in files:
        click.echo(f"\n• {f}")
        result = runner.run_case(case, competitor, f)
        click.echo(f"  status:   {result.status}")
        if result.error:
            click.echo(f"  error:    {result.error}")
        click.echo(f"  findings: {len(result.findings)}")
        if result.wall_clock_s is not None:
            click.echo(f"  wall:     {result.wall_clock_s:.1f}s")
        if result.tokens_in is not None:
            click.echo(f"  tokens:   in={result.tokens_in} out={result.tokens_out}")
        if result.cost_usd is not None:
            click.echo(f"  cost:     ${result.cost_usd:.4f}")
        if result.status != "complete":
            worst = 1
    if worst:
        sys.exit(1)


@bench.command(name="runs")
@click.option("--db", "db_path", default="nelson.db", help="Path to SQLite database.")
def bench_runs(db_path: str):
    """List benchmark runs (newest first)."""
    db = Database(db_path)
    rows = db.list_runs()
    if not rows:
        click.echo("No runs.")
        return
    click.echo(
        f"{'ID':>4} {'STATUS':<12} {'CASE':<22} {'COMPETITOR':<22} {'COST':>8} WALL"
    )
    for r in rows:
        cost = f"${r['cost_usd']:.3f}" if r["cost_usd"] is not None else "-"
        wall = f"{r['wall_clock_s']:.0f}s" if r["wall_clock_s"] is not None else "-"
        click.echo(
            f"{r['id']:>4} {r['status']:<12} {r['case_ext_id']:<22} "
            f"{r['competitor_name']:<22} {cost:>8} {wall}"
        )


@bench.command(name="show-run")
@click.argument("run_id", type=int)
@click.option("--db", "db_path", default="nelson.db", help="Path to SQLite database.")
def bench_show_run(run_id: int, db_path: str):
    """Show one run's metadata and reported findings."""
    db = Database(db_path)
    row = db.get_run(run_id)
    if row is None:
        click.echo(f"Run {run_id} not found.", err=True)
        sys.exit(1)
    for key in row.keys():  # noqa: SIM118 — sqlite3.Row needs .keys() for columns
        if key == "raw_output":
            continue
        click.echo(f"{key:>16}: {row[key]}")
    findings = db.run_findings(run_id)
    click.echo(f"\nReported findings ({len(findings)}):")
    for f in findings:
        loc = f"{f['file']}:{f['line_start']}" if f["file"] else "(no file)"
        click.echo(f"  [{f['confidence'] or '?'}] {loc} {f['cwe'] or ''}")
        if f["description"]:
            click.echo(f"      {f['description']}")


def _emit_run_score(rs) -> None:
    """Verbose per-run scoring output (one run)."""
    click.echo(
        f"Run {rs.run_id}: {rs.case_ext_id} | {rs.competitor_name} | "
        f"{rs.status} -> {rs.outcome.upper()}"
    )
    for f in rs.findings:
        loc = f"{f.file}:{f.line}" if f.file else "(no file)"
        tag = "localized" if f.localized else "off-target"
        click.echo(f"  [{tag}] {loc}")
        if f.truth is not None:
            click.echo(f"      truth: {f.truth.label} — {f.truth.reasoning}")
        if f.fp is not None:
            click.echo(f"      fp: {f.fp.label} — {f.fp.reasoning}")
    if rs.judge_cost:
        click.echo(f"  truth-judge cost: ${rs.judge_cost:.4f}")
    if rs.fp_cost:
        click.echo(f"  fp-judge cost: ${rs.fp_cost:.4f}")


def _emit_detection_report(reports) -> None:
    """Per-competitor detection table."""
    click.echo(
        f"\n{'COMPETITOR':<26} {'DET':>5} {'HIT':>4} {'MISS':>4} "
        f"{'JERR':>4} {'REFU':>4} {'EXCL':>4} {'JUDGE$':>8}"
    )
    for d in reports:
        rate = f"{d.detection_rate:.0%}" if d.eligible else "-"
        click.echo(
            f"{d.competitor_name:<26} {rate:>5} {d.hits:>4} {d.misses:>4} "
            f"{d.judge_error:>4} {d.refused:>4} {d.excluded:>4} ${d.judge_cost:>7.3f}"
        )
    click.echo(
        "\nDET = hits / eligible (hits + genuine misses). JERR (undetermined), "
        "REFU (model refused) and EXCL (auth/infra) are never counted as misses."
    )


def _emit_precision_report(reports) -> None:
    """Per-competitor precision table (FP judge over non-target findings)."""
    click.echo(
        f"\n{'COMPETITOR':<26} {'PREC':>5} {'FP/CASE':>7} {'TGT':>4} {'OTH':>4} "
        f"{'FP':>4} {'UND':>4} {'FP$':>8}"
    )
    for p in reports:
        prec = f"{p.precision:.0%}" if p.precision is not None else "-"
        fpc = f"{p.fp_per_case:.2f}" if p.fp_per_case is not None else "-"
        click.echo(
            f"{p.competitor_name:<26} {prec:>5} {fpc:>7} {p.target_hits:>4} "
            f"{p.real_others:>4} {p.false_positives:>4} {p.undetermined:>4} "
            f"${p.fp_cost:>7.3f}"
        )
    click.echo(
        "\nPREC = true findings / (true + false positives). TGT target-bug hits, "
        "OTH other real bugs, FP false positives, UND undetermined "
        "(needs_review / judge error — never counted against precision)."
    )


@bench.command(name="score")
@click.argument("run_id", required=False, type=int)
@click.option("--db", "db_path", default="nelson.db", help="Path to SQLite database.")
@click.option("--judge-model", default="opus", help="Truth judge model (claude -p).")
@click.option(
    "--tolerance",
    default=10,
    type=int,
    help="Lines a finding may drift from a patched hunk and still localize.",
)
@click.option(
    "--rescore",
    is_flag=True,
    help="Re-judge complete runs already scored (re-spends judge budget).",
)
@click.option(
    "--precision/--no-precision",
    default=True,
    help="Also FP-judge non-target findings for precision (P4). On by default.",
)
@click.option(
    "--fp-judge-model",
    default="opus",
    help="FP judge model (claude -p, code-grounded).",
)
@click.option(
    "--cache-dir",
    default=None,
    help="Where the FP judge checks out source (default .nelson_cache/fpjudge).",
)
def bench_score(
    run_id: int | None,
    db_path: str,
    judge_model: str,
    tolerance: int,
    rescore: bool,
    precision: bool,
    fp_judge_model: str,
    cache_dir: str | None,
):
    """Score benchmark runs: localization + truth judge (detection) and, by
    default, a code-grounded FP judge over non-target findings (precision).

    With RUN_ID, score that one run verbosely. Without, score every run and print
    the per-competitor detection report (and, with precision on, a precision
    report); complete runs already scored are read back from the DB (no judge
    spend) unless --rescore. Runs that never reached `complete` (auth_failed /
    infra_error) are excluded, never counted as misses. The FP judge never sees
    the advisory (precision must not be circular).
    """
    from .score import (
        ClaudeFPJudge,
        ClaudeRefusalJudge,
        ClaudeTruthJudge,
        GitCodeProvider,
        Scorer,
        detection_report,
        precision_report,
    )

    db = Database(db_path)
    fp_judge = ClaudeFPJudge(model=fp_judge_model) if precision else None
    code = GitCodeProvider(root=cache_dir) if precision else None
    scorer = Scorer(
        db,
        ClaudeTruthJudge(model=judge_model),
        tolerance=tolerance,
        fp_judge=fp_judge,
        code=code,
        # Cheap (fires only on zero-finding, contract-ignoring runs) and
        # detection-integral: a policy refusal is marked refused, not a miss.
        refusal_judge=ClaudeRefusalJudge(model=judge_model),
    )

    if run_id is not None:
        _emit_run_score(scorer.score_run(run_id))
        return

    rows = db.list_runs()
    if not rows:
        click.echo("No runs to score.")
        return
    run_scores = []
    for r in rows:
        rid = r["id"]
        if r["status"] == "complete" and (rescore or scorer.needs_scoring(rid)):
            rs = scorer.score_run(rid)
        else:
            rs = scorer.load_run_score(rid)
        run_scores.append(rs)
        click.echo(
            f"  run {rs.run_id:>3} {rs.case_ext_id:<22} "
            f"{rs.competitor_name:<22} -> {rs.outcome}"
        )
    _emit_detection_report(detection_report(run_scores))
    if precision:
        _emit_precision_report(precision_report(run_scores))


def _emit_leaderboard(entries) -> None:
    """Per-competitor leaderboard table (detection + precision + economics)."""
    click.echo(
        f"\n{'#':>2} {'COMPETITOR':<26} {'SIZE':<8} {'DET':>5} {'PREC':>5} "
        f"{'FP/C':>5} {'OTHER':>5} {'COST/C':>8} {'LAT':>6} {'TOK/C':>7} {'CASES':>5}"
    )
    for i, e in enumerate(entries, 1):
        det = f"{e.detection_rate:.0%}" if e.eligible else "-"
        prec = f"{e.precision:.0%}" if e.precision is not None else "-"
        fpc = f"{e.fp_per_case:.2f}" if e.fp_per_case is not None else "-"
        other = str(e.real_others) if e.real_others else "-"
        cpc = f"${e.cost_per_case:.3f}" if e.cost_per_case is not None else "-"
        lat = f"{e.latency_per_case:.0f}s" if e.latency_per_case is not None else "-"
        tpc = e.tokens_per_case
        tok = _fmt_tokens(tpc) if tpc is not None else "-"
        click.echo(
            f"{i:>2} {e.competitor_name:<26} {(e.size_class or '-'):<8} "
            f"{det:>5} {prec:>5} {fpc:>5} {other:>5} {cpc:>8} {lat:>6} "
            f"{tok:>7} {e.cases:>5}"
        )
    click.echo(
        "\nRanked by detection, then precision, then cost/case. OTHER = confirmed "
        "real bugs found that are not the known target CVE. COST/C, LAT and TOK/C "
        "are the competitor's own spend/time/tokens per audited case (judge spend "
        "excluded). TOK/C reflects provider-reported usage — under-reporting compat "
        "endpoints understate it."
    )


def _fmt_tokens(v: float) -> str:
    """Compact token count for the CLI table: 1.2M / 340k / 980."""
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if v >= 1_000:
        return f"{v / 1_000:.0f}k"
    return f"{v:.0f}"


@bench.command(name="leaderboard")
@click.option("--db", "db_path", default="nelson.db", help="Path to SQLite database.")
@click.option(
    "--html",
    "html_path",
    default=None,
    help="Also write the leaderboard + Pareto report to this HTML file.",
)
@click.option(
    "--tolerance",
    default=10,
    type=int,
    help="Localization tolerance used when reading back persisted scores.",
)
def bench_leaderboard(db_path: str, html_path: str | None, tolerance: int):
    """Rank scored competitors: detection, precision, cost/case, latency, size.

    Read-only — it reloads already-scored runs from the DB (no judge spend; run
    `bench score` first to score new runs). Prints the leaderboard table, and with
    --html writes a report with Pareto scatter plots and a per-case drilldown.
    """
    from .score import (
        ClaudeTruthJudge,
        Scorer,
        case_scores,
        drop_competitors,
        leaderboard,
    )

    db = Database(db_path)
    rows = db.list_runs()
    if not rows:
        click.echo("No runs. Run `nelson bench run` then `nelson bench score`.")
        return

    # A Scorer is needed only for load_run_score / needs_scoring; its judge is
    # never invoked here (leaderboard reflects already-persisted scores).
    scorer = Scorer(db, ClaudeTruthJudge(), tolerance=tolerance)
    retired = {c["name"] for c in db.list_competitors() if c["status"] == "retired"}
    run_scores = drop_competitors(
        [scorer.load_run_score(r["id"]) for r in rows], retired
    )
    unscored = sum(
        1
        for r in rows
        if r["status"] == "complete"
        and scorer.needs_scoring(r["id"], include_precision=True)
    )
    if unscored:
        click.echo(
            f"warning: {unscored} complete run(s) not yet scored — run "
            "`nelson bench score` for full results.",
            err=True,
        )

    entries = leaderboard(run_scores)
    _emit_leaderboard(entries)

    if html_path:
        from .html_report import generate_leaderboard_report

        html = generate_leaderboard_report(entries, case_scores(run_scores))
        Path(html_path).write_text(html)
        click.echo(f"\nWrote leaderboard report to {html_path}")


@bench.command(name="noise")
@click.option("--db", "db_path", default="nelson.db", help="Path to SQLite database.")
@click.option("--tolerance", default=10, type=int, help="Localization tolerance.")
def bench_noise(db_path: str, tolerance: int):
    """Run-to-run variance across repeated (--repeat) trials.

    Read-only — reloads persisted scores (no judge spend). For each competitor it
    shows per-trial detection (so you can see how much one run's number moves) and
    each case's hit-frequency across trials, flagging *flaky* cases (hit in some
    trials, missed in others) — the noise the repeat is there to expose.
    """
    from .score import (
        ClaudeTruthJudge,
        Scorer,
        drop_competitors,
        noise_report,
    )

    db = Database(db_path)
    rows = db.list_runs()
    if not rows:
        click.echo("No runs. Run `nelson bench loop --repeat N` first.")
        return
    scorer = Scorer(db, ClaudeTruthJudge(), tolerance=tolerance)
    retired = {c["name"] for c in db.list_competitors() if c["status"] == "retired"}
    run_scores = drop_competitors(
        [scorer.load_run_score(r["id"]) for r in rows], retired
    )
    reports = noise_report(run_scores)
    max_trials = max((r.n_trials for r in reports), default=0)
    if max_trials <= 1:
        click.echo(
            "Only one trial present — no variance to report. Re-run with "
            "`bench loop --repeat N` (N>1)."
        )
        return

    for r in sorted(reports, key=lambda x: -x.mean_rate):
        rates = ", ".join(
            f"t{t}={h}/{e}" for t, (h, e) in sorted(r.per_trial.items())
        )
        click.echo(
            f"\n{r.competitor_name}  "
            f"mean={r.mean_rate:.0%}  range={r.min_rate:.0%}-{r.max_rate:.0%}  "
            f"spread={r.spread:.0%}  ({r.n_trials} trials)"
        )
        click.echo(f"  per-trial detection: {rates}")
        if r.flaky_cases:
            flaky = ", ".join(
                f"{c} ({r.per_case[c][0]}/{r.per_case[c][1]})" for c in r.flaky_cases
            )
            click.echo(f"  flaky (hit some / missed some): {flaky}")
        else:
            click.echo("  flaky cases: none — every case was all-hit or all-miss")


# -- Automation loop ---------------------------------------------------------


def _emit_loop_report(report) -> None:
    """Print one pass's LoopReport as a compact, cron-log-friendly summary."""
    if report.corpus:
        corpus = ", ".join(f"{k}={v}" for k, v in report.corpus.items() if v)
        click.echo(f"  corpus:   {corpus or 'no change'}")
    if report.competitors_synced or report.competitors_retired:
        line = f"  roster:   {report.competitors_synced} synced"
        if report.competitors_retired:
            line += f", retired {', '.join(report.competitors_retired)}"
        click.echo(line)
    if report.aged_out:
        click.echo(f"  aged out: {', '.join(report.aged_out)}")
    click.echo(
        f"  matrix:   {report.planned} planned, {report.ran} ran "
        f"({report.completed} complete, {report.auth_failed} auth_failed, "
        f"{report.infra_error} infra_error), {report.skipped_caps} capped"
    )
    if report.spend_usd:
        click.echo(f"  spend:    ${report.spend_usd:.4f} (competitor)")
    if report.scored:
        click.echo(f"  scored:   {report.scored} ({report.judge_errors} judge_error)")
    if report.report_path:
        click.echo(f"  report:   {report.report_path}")
    if report.circuit_broken:
        click.echo(
            click.style(
                "  ALERT:    auth circuit breaker tripped — check the host "
                "`claude` login (runs aborted).",
                fg="red",
                bold=True,
            )
        )
    elif report.auth_failed:
        click.echo(
            click.style(
                f"  ALERT:    {report.auth_failed} auth failure(s) — credentials "
                "may be expired.",
                fg="yellow",
            )
        )


@bench.command(name="loop")
@click.option("--db", "db_path", default="nelson.db", help="Path to SQLite database.")
@click.option(
    "--competitors",
    "competitors_path",
    default=None,
    help="YAML competitor roster to sync (adds/retires are config). Optional.",
)
@click.option(
    "--cases-dir",
    default=None,
    help="Import cases/*.yaml manifests into the DB before running. Optional.",
)
@click.option(
    "--refresh-corpus",
    is_flag=True,
    help="Also refresh the corpus first (network + Opus pre-vet spend). Default off.",
)
@click.option("--corpus-limit", default=0, type=int, help="Max candidates per refresh.")
@click.option(
    "--age-out/--no-age-out",
    default=True,
    help="Retire vetted cases no active competitor would find 0-day. On by default.",
)
@click.option(
    "--recency/--no-recency",
    "recency_filter",
    default=True,
    help="Skip a cell when the case predates that competitor's cutoff. On by default.",
)
@click.option(
    "--retry-failed",
    is_flag=True,
    help="Re-run cells whose only runs were auth_failed / infra_error.",
)
@click.option(
    "--repeat",
    default=1,
    type=int,
    help="Run each cell this many independent trials (for noise measurement). "
    "Resumes per-trial; see `bench noise` for the variance report.",
)
@click.option("--max-runs", default=0, type=int, help="Cap runs launched per pass.")
@click.option(
    "--concurrency",
    default=1,
    type=int,
    help="Worker threads for the run matrix (1 = sequential). One model runs at "
    "a time per worker, so distinct models run in parallel without 429s.",
)
@click.option(
    "--max-spend-usd",
    default=0.0,
    type=float,
    help="Stop launching runs once competitor spend reaches this (0 = no cap).",
)
@click.option(
    "--auth-fail-abort",
    default=3,
    type=int,
    help="Consecutive auth failures that trip the circuit breaker. Default: 3.",
)
@click.option(
    "--score/--no-score",
    "do_score",
    default=True,
    help="Score new completions (truth + FP judge). On by default.",
)
@click.option("--tolerance", default=10, type=int, help="Localization tolerance.")
@click.option("--judge-model", default="opus", help="Truth judge model.")
@click.option("--fp-judge-model", default="opus", help="FP judge model.")
@click.option("--cache-dir", default="bench-cache", help="Where source is checked out.")
@click.option("--runs-dir", default="bench-runs", help="Where transcripts are written.")
@click.option(
    "--fp-cache-dir", default=None, help="Where the FP judge checks out source."
)
@click.option("--no-network", is_flag=True, help="Run containers with no network.")
@click.option("--timeout", default=1800.0, type=float, help="Per-run wall-clock cap.")
@click.option(
    "--max-budget-usd", default=0.50, type=float, help="Per-run cost backstop."
)
@click.option(
    "--html", "html_path", default=None, help="Leaderboard report output path."
)
@click.option(
    "--interval",
    default=0.0,
    type=float,
    help="Repeat every N seconds (0 = run one pass and exit; use cron instead).",
)
def bench_loop(
    db_path: str,
    competitors_path: str | None,
    cases_dir: str | None,
    refresh_corpus: bool,
    corpus_limit: int,
    age_out: bool,
    recency_filter: bool,
    retry_failed: bool,
    repeat: int,
    max_runs: int,
    concurrency: int,
    max_spend_usd: float,
    auth_fail_abort: int,
    do_score: bool,
    tolerance: int,
    judge_model: str,
    fp_judge_model: str,
    cache_dir: str,
    runs_dir: str,
    fp_cache_dir: str | None,
    no_network: bool,
    timeout: float,
    max_budget_usd: float,
    html_path: str | None,
    interval: float,
):
    """Run one unattended benchmark pass: refresh -> age out -> run -> score -> rank.

    Idempotent and resumable — it only fills the missing matrix cells and scores
    new completions, so a cron entry (or --interval) can fire it repeatedly. Bound
    an unattended pass with --max-runs / --max-spend-usd; an auth circuit breaker
    aborts the run stage on repeated auth failures (likely an expired host login).
    Exits non-zero when a pass needs attention (auth failure / breaker tripped).
    """
    import time

    from .automate import load_competitors, run_once
    from .corpus import CorpusPipeline, CVDSeedSource, load_manifest_dir
    from .runner import BenchRunner
    from .score import (
        ClaudeFPJudge,
        ClaudeRefusalJudge,
        ClaudeTruthJudge,
        GitCodeProvider,
        Scorer,
    )

    db = Database(db_path)

    declared = load_competitors(competitors_path) if competitors_path else None

    if cases_dir:
        for case in load_manifest_dir(cases_dir):
            db.upsert_case(case.to_db_fields())

    runner = BenchRunner(
        db,
        cache_dir=cache_dir,
        runs_dir=runs_dir,
        network=not no_network,
        run_timeout=timeout,
        max_budget_usd=max_budget_usd or None,
    )

    scorer = None
    if do_score:
        scorer = Scorer(
            db,
            ClaudeTruthJudge(model=judge_model),
            tolerance=tolerance,
            fp_judge=ClaudeFPJudge(model=fp_judge_model),
            code=GitCodeProvider(root=fp_cache_dir),
            refusal_judge=ClaudeRefusalJudge(model=judge_model),
        )

    pipeline = None
    seed_source = None
    if refresh_corpus:
        from .derive import SubprocessGitRunner
        from .enrich import HttpxClient, NVDEnricher, OSVEnricher, fetch_cvd_payload
        from .prevet import ClaudeCLIJudge

        http = HttpxClient()
        pipeline = CorpusPipeline(
            db,
            enrichers=[OSVEnricher(http), NVDEnricher(http)],
            git=SubprocessGitRunner(),
            judge=ClaudeCLIJudge(model=judge_model),
            cache_dir="corpus-cache",
        )
        try:
            payload = fetch_cvd_payload(CVD_PAYLOAD_URL, http)
            seed_source = CVDSeedSource(payload)
        except RuntimeError as e:
            click.echo(f"warning: corpus seed fetch failed: {e}", err=True)

    def once() -> bool:
        report = run_once(
            db,
            runner=runner,
            scorer=scorer,
            pipeline=pipeline,
            seed_source=seed_source,
            corpus_limit=corpus_limit or None,
            declared_competitors=declared,
            age_out=age_out,
            recency_filter=recency_filter,
            retry_failed=retry_failed,
            repeat=repeat,
            max_runs=max_runs or None,
            max_spend_usd=max_spend_usd or None,
            auth_fail_abort=auth_fail_abort,
            concurrency=concurrency,
            report_html_path=html_path,
            on_event=lambda m: click.echo(f"  · {m}", err=True),
        )
        click.echo(f"Pass complete ({db_path}):")
        _emit_loop_report(report)
        return report.needs_attention

    if interval <= 0:
        sys.exit(1 if once() else 0)

    click.echo(f"Looping every {interval:.0f}s (Ctrl-C to stop)…")
    try:
        while True:
            once()
            time.sleep(interval)
    except KeyboardInterrupt:
        click.echo("\nStopped.")
