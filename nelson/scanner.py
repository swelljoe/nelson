"""Scan orchestrator: builds job matrix, processes queue, handles pacing."""

import logging
import subprocess
import time
from pathlib import Path

from .agents import AgentAdapter, create_adapter
from .cwe import CWE_OPEN, CWE_TOP_25, applicable_cwes, build_open_prompt, build_prompt
from .db import Database
from .inventory import LANGUAGE_MAP, SourceFile, discover_files

log = logging.getLogger(__name__)


def get_commit_sha(target_dir: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=target_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def build_job_matrix(
    files: list[SourceFile],
    models: list[str],
    cwe_ids: list[str] | None = None,
    mode: str = "focused",
) -> list[tuple[str, str, str]]:
    """Build (file_path, cwe_id, model_id) tuples.

    mode="focused": one job per (file, applicable CWE, model)
    mode="open": one job per (file, model) with cwe_id="OPEN"
    """
    jobs = []
    for f in files:
        for model_spec in models:
            adapter = create_adapter(model_spec)
            if mode == "open":
                jobs.append((f.path, "OPEN", adapter.name))
            else:
                cwes = applicable_cwes(f.language)
                if cwe_ids:
                    cwes = [c for c in cwes if c.id in cwe_ids]
                for cwe in cwes:
                    jobs.append((f.path, cwe.id, adapter.name))
    return jobs


def create_scan(
    db: Database,
    target_dir: str,
    models: list[str],
    cwe_ids: list[str] | None = None,
    mode: str = "focused",
    files: list[SourceFile] | None = None,
) -> tuple[int, list[SourceFile]]:
    """Discover files, build matrix, create scan and jobs in DB.

    If ``files`` is provided, it is used as-is (e.g. when the user named
    individual files on the command line); otherwise the target directory
    is walked with ``discover_files``.
    """
    target = str(Path(target_dir).resolve())
    if files is None:
        files = discover_files(target)

    if not files:
        raise ValueError(f"No source files found in {target}")

    commit_sha = get_commit_sha(target)
    config = {"models": models, "cwe_ids": cwe_ids, "mode": mode}
    scan_id = db.create_scan(target, commit_sha=commit_sha, config=config)

    matrix = build_job_matrix(files, models, cwe_ids=cwe_ids, mode=mode)
    db.create_jobs_batch(scan_id, matrix)

    log.info(
        "Scan %d (%s): %d files, %d jobs across %d models",
        scan_id,
        mode,
        len(files),
        len(matrix),
        len(models),
    )
    return scan_id, files


# Map from model_id -> adapter instance, cached
_adapter_cache: dict[str, AgentAdapter] = {}


def _get_adapter(model_spec: str, models: list[str]) -> AgentAdapter | None:
    """Look up or create the adapter for a given model_id."""
    if model_spec not in _adapter_cache:
        # Find the original spec string that produces this model_id
        for spec in models:
            adapter = create_adapter(spec)
            _adapter_cache[adapter.name] = adapter
    return _adapter_cache.get(model_spec)


def _release_job(db: Database, job_id: int):
    """Put a running job back to pending so it can be retried on resume."""
    db.conn.execute(
        "UPDATE jobs SET status = 'pending', started_at = NULL WHERE id = ?",
        (job_id,),
    )
    db.conn.commit()


def run_scan(
    db: Database,
    scan_id: int,
    models: list[str],
    target_dir: str,
    delay: float = 2.0,
    max_backoff: float = 300.0,
    on_progress=None,
):
    """Process pending jobs for a scan.

    Args:
        db: Database instance
        scan_id: Scan to process
        models: Model spec strings (e.g., ["claude:haiku"])
        target_dir: Root directory of the code being scanned
        delay: Seconds to wait between jobs
        max_backoff: Maximum backoff delay on rate limiting
        on_progress: Optional callback(job_counts_dict) called after each job
    """
    target = Path(target_dir).resolve()

    # Build CWE lookup (include the OPEN sentinel)
    cwe_lookup = {c.id: c for c in CWE_TOP_25}
    cwe_lookup["OPEN"] = CWE_OPEN

    # Pre-create all adapters
    adapters: dict[str, AgentAdapter] = {}
    for spec in models:
        a = create_adapter(spec)
        adapters[a.name] = a

    # Derive language for each job's file from its extension. We don't
    # re-run discover_files here because the scan may have been created
    # from an explicit file list that wouldn't be re-discovered.
    def language_for(file_path: str) -> str | None:
        return LANGUAGE_MAP.get(Path(file_path).suffix.lower())

    backoff = 0.0

    current_job_id: int | None = None

    try:
        while True:
            job = db.next_pending_job(scan_id)
            if job is None:
                break

            if not db.claim_job(job["id"]):
                continue  # Someone else claimed it (future: multi-worker)

            current_job_id = job["id"]
            model_id = job["model_id"]
            adapter = adapters.get(model_id)
            if adapter is None:
                db.fail_job(job["id"], f"No adapter for model '{model_id}'")
                current_job_id = None
                continue

            cwe = cwe_lookup.get(job["cwe_id"])
            if cwe is None:
                db.fail_job(job["id"], f"Unknown CWE '{job['cwe_id']}'")
                current_job_id = None
                continue

            language = language_for(job["file_path"])
            if language is None:
                db.fail_job(
                    job["id"],
                    f"Unknown file type: {job['file_path']}",
                )
                current_job_id = None
                continue

            # Read the file
            file_path = target / job["file_path"]
            try:
                content = file_path.read_text(errors="replace")
            except OSError as e:
                db.fail_job(job["id"], f"Cannot read file: {e}")
                current_job_id = None
                continue

            if job["cwe_id"] == "OPEN":
                prompt = build_open_prompt(
                    job["file_path"],
                    content,
                    language,
                )
            else:
                prompt = build_prompt(
                    job["file_path"],
                    content,
                    language,
                    cwe,
                )

            log.info(
                "Job %d: %s x %s x %s",
                job["id"],
                job["file_path"],
                job["cwe_id"],
                model_id,
            )

            result = adapter.run(prompt)

            if result.rate_limited:
                # Put the job back to pending and back off
                _release_job(db, job["id"])
                current_job_id = None
                backoff = min(max(backoff * 2, 30.0), max_backoff)
                log.warning(
                    "Rate limited by %s, backing off %.0fs",
                    model_id,
                    backoff,
                )
                time.sleep(backoff)
                continue

            # Reset backoff on success
            backoff = 0.0

            if result.error:
                db.fail_job(job["id"], result.error)
            else:
                db.complete_job(
                    job["id"],
                    tokens_in=result.tokens_in,
                    tokens_out=result.tokens_out,
                    cost_usd=result.cost_usd,
                )
                for f in result.findings:
                    explanation = f.explanation
                    # In open mode, prepend the model-identified CWE
                    if job["cwe_id"] == "OPEN" and f.cwe_id:
                        explanation = f"[{f.cwe_id}] {explanation}"
                    db.add_finding(
                        job["id"],
                        line_number=f.line_number,
                        code_snippet=f.code_snippet,
                        explanation=explanation,
                        confidence=f.confidence,
                    )

            current_job_id = None

            if on_progress:
                on_progress(db.job_counts(scan_id))

            if delay > 0 and adapter.needs_pacing:
                time.sleep(delay)

    except KeyboardInterrupt:
        log.warning("Scan interrupted by user")
        if current_job_id is not None:
            _release_job(db, current_job_id)
        raise

    db.complete_scan(scan_id)
