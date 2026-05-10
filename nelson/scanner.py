"""Scan orchestrator: builds job matrix, processes queue, handles pacing."""

import logging
import signal
import subprocess
import threading
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


def _release_job(db: Database, job_id: int):
    """Put a running job back to pending so it can be retried on resume."""
    db.conn.execute(
        "UPDATE jobs SET status = 'pending', started_at = NULL WHERE id = ?",
        (job_id,),
    )
    db.conn.commit()


def _language_for(file_path: str) -> str | None:
    """Derive language from extension. We don't re-run discover_files here
    because the scan may have been created from an explicit file list that
    wouldn't be re-discovered."""
    return LANGUAGE_MAP.get(Path(file_path).suffix.lower())


def _process_one_job(
    db: Database,
    job,
    adapter: AgentAdapter,
    target: Path,
    cwe_lookup: dict,
    cancel_event: threading.Event,
) -> tuple[bool, bool]:
    """Run a single claimed job to completion.

    Returns (rate_limited, ok). On rate_limited, the caller should release
    the job and back off. On ok=False, the job has already been failed.
    """
    cwe = cwe_lookup.get(job["cwe_id"])
    if cwe is None:
        db.fail_job(job["id"], f"Unknown CWE '{job['cwe_id']}'")
        return False, False

    language = _language_for(job["file_path"])
    if language is None:
        db.fail_job(job["id"], f"Unknown file type: {job['file_path']}")
        return False, False

    file_path = target / job["file_path"]
    try:
        content = file_path.read_text(errors="replace")
    except OSError as e:
        db.fail_job(job["id"], f"Cannot read file: {e}")
        return False, False

    if job["cwe_id"] == "OPEN":
        prompt = build_open_prompt(job["file_path"], content, language)
    else:
        prompt = build_prompt(job["file_path"], content, language, cwe)

    log.info(
        "Job %d: %s x %s x %s",
        job["id"],
        job["file_path"],
        job["cwe_id"],
        adapter.name,
    )

    result = adapter.run(prompt, cancel_event=cancel_event)

    if result.rate_limited:
        return True, False

    if result.error:
        db.fail_job(job["id"], result.error)
        return False, False

    db.complete_job(
        job["id"],
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        cost_usd=result.cost_usd,
    )
    for f in result.findings:
        explanation = f.explanation
        if job["cwe_id"] == "OPEN" and f.cwe_id:
            explanation = f"[{f.cwe_id}] {explanation}"
        db.add_finding(
            job["id"],
            line_number=f.line_number,
            code_snippet=f.code_snippet,
            explanation=explanation,
            confidence=f.confidence,
        )
    return False, True


def _model_worker(
    db_path: str,
    scan_id: int,
    model_spec: str,
    target_dir: str,
    delay: float,
    max_backoff: float,
    cancel_event: threading.Event,
    progress_event: threading.Event,
):
    """Drain pending jobs for a single model. Owns its own DB connection
    so it can run concurrently with other workers under SQLite WAL."""
    target = Path(target_dir).resolve()
    cwe_lookup = {c.id: c for c in CWE_TOP_25}
    cwe_lookup["OPEN"] = CWE_OPEN

    adapter = create_adapter(model_spec)
    db = Database(db_path)
    backoff = 0.0
    current_job_id: int | None = None

    try:
        while not cancel_event.is_set():
            job = db.next_pending_job(scan_id, model_id=adapter.name)
            if job is None:
                break

            if not db.claim_job(job["id"]):
                continue  # Another worker grabbed it; try again

            current_job_id = job["id"]
            try:
                rate_limited, _ok = _process_one_job(
                    db, job, adapter, target, cwe_lookup, cancel_event
                )
            except KeyboardInterrupt:
                _release_job(db, job["id"])
                current_job_id = None
                break
            current_job_id = None

            if rate_limited:
                _release_job(db, job["id"])
                backoff = min(max(backoff * 2, 30.0), max_backoff)
                log.warning(
                    "Rate limited by %s, backing off %.0fs",
                    adapter.name,
                    backoff,
                )
                # Interruptible sleep
                if cancel_event.wait(backoff):
                    break
                continue

            backoff = 0.0
            progress_event.set()

            if delay > 0 and adapter.needs_pacing and cancel_event.wait(delay):
                break
    finally:
        if current_job_id is not None:
            _release_job(db, current_job_id)
        db.close()


def run_scan(
    db: Database,
    scan_id: int,
    models: list[str],
    target_dir: str,
    delay: float = 2.0,
    max_backoff: float = 300.0,
    on_progress=None,
    parallel: bool = True,
    db_path: str | None = None,
):
    """Process pending jobs for a scan.

    With multiple models, one worker thread per model drains that model's
    pending jobs concurrently. Each worker owns its own SQLite connection;
    job claiming is already atomic, so there's no extra locking. Single-
    model scans take the same path with one worker.

    Args:
        db: Database instance (used by the main thread for progress polling
            and to mark the scan complete; workers each open their own).
        scan_id: Scan to process
        models: Model spec strings (e.g., ["claude:haiku"])
        target_dir: Root directory of the code being scanned
        delay: Per-worker pacing between jobs (CLI adapters only)
        max_backoff: Maximum backoff delay on rate limiting
        on_progress: Optional callback(job_counts_dict) called after each job
        parallel: When False, run workers one at a time even if multiple
            models are passed (preserves the old serial behavior).
        db_path: Path to the SQLite db, used by workers to open their own
            connections. Defaults to ``str(db.path)``.
    """
    if db_path is None:
        db_path = str(db.path)

    cancel_event = threading.Event()
    progress_event = threading.Event()
    worker_exceptions: dict[str, Exception] = {}
    worker_exceptions_lock = threading.Lock()

    original_handler = signal.getsignal(signal.SIGINT)

    def _interrupt(signum, frame):
        cancel_event.set()

    signal.signal(signal.SIGINT, _interrupt)

    def _worker_wrapper(spec: str):
        """Wrapper that catches and stores worker exceptions."""
        try:
            _model_worker(
                db_path,
                scan_id,
                spec,
                target_dir,
                delay,
                max_backoff,
                cancel_event,
                progress_event,
            )
        except Exception as e:
            log.error("Worker for %s failed: %s", spec, e, exc_info=True)
            with worker_exceptions_lock:
                worker_exceptions[spec] = e

    def _spawn(spec: str) -> threading.Thread:
        t = threading.Thread(
            target=_worker_wrapper,
            args=(spec,),
            name=f"nelson-worker-{spec}",
            daemon=True,
        )
        t.start()
        return t

    def _drain(threads: list[threading.Thread]):
        while any(t.is_alive() for t in threads):
            if progress_event.wait(timeout=0.5):
                progress_event.clear()
                if on_progress:
                    on_progress(db.job_counts(scan_id))
        for t in threads:
            t.join()

    try:
        if parallel:
            _drain([_spawn(spec) for spec in models])
        else:
            # One model at a time, but still on a worker thread so the
            # main thread can poll progress and handle SIGINT.
            for spec in models:
                if cancel_event.is_set():
                    break
                _drain([_spawn(spec)])
    finally:
        signal.signal(signal.SIGINT, original_handler)

    if cancel_event.is_set():
        log.warning("Scan interrupted by user")
        raise KeyboardInterrupt

    # Check for worker failures
    if worker_exceptions:
        log.error("One or more workers failed:")
        for spec, exc in worker_exceptions.items():
            log.error("  %s: %s", spec, exc)
        raise RuntimeError(f"Worker failures: {', '.join(worker_exceptions.keys())}")

    # Verify all jobs reached a terminal state before marking scan complete
    job_counts = db.job_counts(scan_id)
    pending = job_counts.get("pending", 0)
    running = job_counts.get("running", 0)
    if pending > 0 or running > 0:
        log.error(
            "Scan has %d pending and %d running jobs, not marking complete",
            pending,
            running,
        )
        raise RuntimeError(
            f"Scan incomplete: {pending} pending, {running} running jobs remain"
        )

    db.complete_scan(scan_id)
