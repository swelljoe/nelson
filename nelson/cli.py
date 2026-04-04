"""Nelson CLI: AI-driven vulnerability scanner orchestrator."""

import json
import logging
import sys
from pathlib import Path

import click

from .db import Database
from .inventory import discover_files
from .scanner import create_scan, run_scan
from .tooling import assess_tooling, format_tooling_report


def _db_path() -> Path:
    return Path("nelson.db")


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
@click.argument("target_dir", type=click.Path(exists=True, file_okay=False))
def inventory(target_dir: str):
    """List source files that would be scanned in TARGET_DIR."""
    files = discover_files(target_dir)
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

    click.echo(f"\nFiles:")
    for f in files:
        click.echo(f"  [{f.language:>12}] {f.path} ({f.size / 1024:.1f} KB)")

    # Security tooling assessment
    languages = {f.language for f in files}
    statuses = assess_tooling(target_dir, languages)
    if statuses:
        click.echo(f"\n{format_tooling_report(statuses)}")


@main.command()
@click.argument("target_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "-m", "--model", "models", multiple=True, default=["claude:haiku"],
    help="Model spec, e.g. 'claude:haiku', 'openai:qwen3.5'. Repeatable.",
)
@click.option(
    "--cwe", "cwe_ids", multiple=True, default=None,
    help="Limit to specific CWE IDs (e.g. --cwe CWE-89 --cwe CWE-78). Default: all applicable.",
)
@click.option(
    "--delay", default=2.0, type=float,
    help="Seconds between jobs (pacing). Default: 2.0",
)
@click.option(
    "--resume", type=int, default=None,
    help="Resume a previous scan by ID instead of creating a new one.",
)
@click.option(
    "--db", "db_path", default="nelson.db",
    help="Path to SQLite database. Default: nelson.db",
)
def scan(target_dir: str, models: tuple[str], cwe_ids: tuple[str], delay: float, resume: int | None, db_path: str):
    """Scan TARGET_DIR for vulnerabilities."""
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
        click.echo(f"Resuming scan {scan_id} on {target_dir}")
    else:
        from .cwe import applicable_cwes
        from .inventory import SourceFile

        files = discover_files(target_dir)
        if not files:
            click.echo("No source files found.", err=True)
            sys.exit(1)

        scan_id, files = create_scan(db, target_dir, list(models), cwe_ids=cwe_list)
        counts = db.job_counts(scan_id)
        total = sum(counts.values())
        click.echo(f"Scan {scan_id}: {len(files)} files, {total} jobs")

    def on_progress(counts):
        done = counts.get("complete", 0) + counts.get("error", 0)
        total = sum(counts.values())
        pending = counts.get("pending", 0)
        errors = counts.get("error", 0)
        click.echo(f"  Progress: {done}/{total} done, {pending} pending, {errors} errors", err=True)

    click.echo(f"Starting scan with delay={delay}s between jobs...")
    run_scan(db, scan_id, list(models), target_dir, delay=delay, on_progress=on_progress)

    counts = db.job_counts(scan_id)
    click.echo(f"\nScan {scan_id} complete: {counts}")

    # Quick summary of findings
    findings = db.findings_for_scan(scan_id)
    if findings:
        click.echo(f"\n{len(findings)} potential vulnerabilities found. Run 'nelson report {scan_id}' for details.")
    else:
        click.echo("\nNo vulnerabilities found.")


@main.command(name="list")
@click.option("--db", "db_path", default="nelson.db", help="Path to SQLite database.")
def list_scans(db_path: str):
    """List all scans."""
    db = Database(db_path)
    scans = db.list_scans()
    if not scans:
        click.echo("No scans found.")
        return

    click.echo(f"{'ID':>4}  {'Status':<11}  {'Findings':>8}  {'Jobs':>6}  {'Model':<35}  {'Target'}")
    click.echo(f"{'─' * 4}  {'─' * 11}  {'─' * 8}  {'─' * 6}  {'─' * 35}  {'─' * 30}")
    for s in scans:
        scan_id = s["id"]
        status = "complete" if s["completed_at"] else "in progress"
        counts = db.job_counts(scan_id)
        total_jobs = sum(counts.values())
        findings = db.findings_for_scan(scan_id)
        target = s["target_dir"]
        config = json.loads(s["config"]) if s["config"] else {}
        models = ", ".join(config.get("models", []))
        click.echo(f"{scan_id:>4}  {status:<11}  {len(findings):>8}  {total_jobs:>6}  {models:<35}  {target}")


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

    usage = db.usage_by_model(scan_id)
    if usage:
        click.echo(f"\n  Token usage by model:")
        for row in usage:
            tin = row["total_tokens_in"] or 0
            tout = row["total_tokens_out"] or 0
            total = tin + tout
            line = f"    {row['model_id']}: {total:,} tokens ({tin:,} in, {tout:,} out) across {row['jobs']} jobs"
            if row["total_cost_usd"]:
                line += f" — ${row['total_cost_usd']:.4f}"
            click.echo(line)


@main.command()
@click.argument("scan_id", type=int, required=False)
@click.option("--db", "db_path", default="nelson.db", help="Path to SQLite database.")
@click.option("--confidence", type=click.Choice(["high", "medium", "low"]), default=None, help="Filter by confidence.")
@click.option("--cwe", "cwe_filter", default=None, help="Filter by CWE ID.")
@click.option("--json-output", "as_json", is_flag=True, help="Output as JSON.")
def report(scan_id: int | None, db_path: str, confidence: str | None, cwe_filter: str | None, as_json: bool):
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
        conf_style = click.style(conf.upper(), fg=conf_colors.get(conf, "white"), bold=True)

        click.echo(f"\n  [{conf_style}] {f['cwe_id']} (found by {f['model_id']})")
        if f["line_number"]:
            click.echo(f"  Line: {f['line_number']}")
        if f["code_snippet"]:
            click.echo(f"  Code: {f['code_snippet']}")
        if f["explanation"]:
            click.echo(f"  Why:  {f['explanation']}")

    # Summary
    click.echo(f"\n{'─' * 60}")
    summary = db.findings_summary(scan_id)
    if summary:
        click.echo("\nSummary by CWE:")
        for row in summary:
            click.echo(f"  {row['cwe_id']} ({row['model_id']}): {row['count']} ({row['confidence']})")
