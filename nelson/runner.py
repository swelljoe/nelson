"""Container runner: execute one competitor against one case in isolation.

A *competitor* (model x runtime x tool-profile) is run against a *case* inside a
throwaway, resource-capped, rootless container. The repository is checked out at
the case's pre-patch (`vuln_commit`) and mounted **read-only** — the competitor
audits code it has never been told anything about (no advisory text leaks in;
that is reserved for the truth judge in P3). What the model reports, plus its
token/cost/wall-clock and the full transcript, is captured for scoring.

Isolation is podman (rootless on Fedora). The image is minimal; the host's own
`claude` binary is bind-mounted in rather than baked into a 200 MB+ image, so the
container tracks the host CLI version automatically. Auth is pluggable
(:class:`ContainerAuth`): P2 mounts the host subscription credentials; a
long-lived OAuth token for the parallel matrix is a drop-in later.

Integrity rule (non-negotiable): an auth or infra failure yields a distinct
status and is **never** a "miss" — the model never got a fair look at the code.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .agents import FailureKind, _run_cli, classify_failure

if TYPE_CHECKING:
    from .corpus import Case

# Minimal Fedora base + the tools the agent's own grep/file tools shell out to.
# The claude binary itself is bind-mounted at run time (see ClaudeCodeRuntime).
IMAGE_TAG = "nelson-bench:fedora"
CONTAINERFILE = """\
FROM registry.fedoraproject.org/fedora-minimal:41
RUN microdnf install -y git ripgrep ca-certificates findutils shadow-utils \\
    && microdnf clean all
# A real passwd entry so the agent's getpwuid() works and HOME is writable.
RUN useradd -u 1000 -m -s /bin/bash agent
USER agent
WORKDIR /home/agent
"""

# The competitor sees only the code. No CVE id, description, or bug class — that
# would defeat the benchmark (and the FP-judge must never see it either).
COMPETITOR_PROMPT = """\
You are an expert security auditor reviewing an unfamiliar codebase.

The complete source tree is mounted read-only at /src. Audit it for *exploitable*
security vulnerabilities — memory-safety bugs, injection (SQL / command / path),
authentication or authorization flaws, unsafe deserialization, SSRF, and the like.
Read and grep through the code under /src as needed. Concentrate on real,
exploitable issues in the project's own code, not style nits or theoretical concerns.

When you are done, output ONLY a JSON array as your final message, with no prose
around it. Each element must be:
{"file": "<path relative to /src>", "line": <int>, "code": "<the vulnerable line>",
 "explanation": "<why it is exploitable>", "confidence": "high|medium|low",
 "cwe": "<CWE-id if known, else null>"}

If you find no vulnerabilities, output [].
"""


class RunnerError(RuntimeError):
    """A setup failure (checkout, image build) that prevents a run from starting."""


# -- Value types -------------------------------------------------------------


@dataclass
class Competitor:
    """A benchmark competitor. Mirrors the ``competitors`` table columns."""

    name: str
    model: str
    runtime: str = "claude-code"
    tool_profile: str = "read-grep"
    auth_profile: str | None = None
    cost_model: str | None = "subscription"
    size_class: str | None = None
    knowledge_cutoff: str | None = None
    status: str = "active"

    def to_db_fields(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "runtime": self.runtime,
            "tool_profile": self.tool_profile,
            "auth_profile": self.auth_profile,
            "cost_model": self.cost_model,
            "size_class": self.size_class,
            "knowledge_cutoff": self.knowledge_cutoff,
            "status": self.status,
        }


@dataclass
class RunResult:
    """Outcome of one (competitor, case) run.

    ``status`` is ``complete`` only when the competitor got a fair look and
    returned; ``auth_failed`` / ``infra_error`` are integrity statuses that must
    never be scored as a miss. ``findings`` are raw dicts as the model reported
    them (file/line/...), persisted to ``run_findings`` for P3 to score.
    """

    status: str
    findings: list[dict[str, Any]] = field(default_factory=list)
    raw_output: str = ""
    transcript: str = ""
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    wall_clock_s: float | None = None
    container_id: str | None = None
    error: str | None = None


# -- Source checkout ---------------------------------------------------------


def _git(args: list[str], cwd: Path, timeout: float) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=timeout
    )
    if proc.returncode != 0:
        raise RunnerError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def prepare_checkout(
    repo_url: str, commit: str, dest: Path, timeout: float = 600.0
) -> Path:
    """Check out a working tree at ``commit`` under ``dest`` (idempotent).

    A depth-1 fetch of the single SHA brings down just that commit's tree, no
    history. If ``dest`` is already checked out at ``commit`` it is reused, so
    repeated runs of the same case don't re-fetch.
    """
    dest = Path(dest)
    if (dest / ".git").is_dir():
        try:
            if _git(["rev-parse", "HEAD"], dest, timeout) == commit:
                return dest
        except RunnerError:
            pass  # broken checkout — rebuild it below
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    _git(["init", "-q"], dest, timeout)
    _git(["remote", "add", "origin", repo_url], dest, timeout)
    _git(["fetch", "--depth=1", "-q", "origin", commit], dest, timeout)
    _git(["checkout", "-q", commit], dest, timeout)
    return dest


# -- Auth (pluggable) --------------------------------------------------------


@dataclass
class AuthMaterial:
    """Env vars and mounts that inject a competitor's credentials into a run."""

    env: dict[str, str] = field(default_factory=dict)
    mounts: list[tuple[str, str, str]] = field(default_factory=list)


@runtime_checkable
class ContainerAuth(Protocol):
    def prepare(self, staging: Path) -> AuthMaterial:
        """Stage credentials under ``staging`` and return how to wire them in."""
        ...


class CredentialMountAuth:
    """Inject Claude subscription auth by mounting the host credentials file.

    The host's ``~/.claude/.credentials.json`` is *copied* into a per-run config
    dir (never bind-mounted from the original) so the container can refresh the
    token without touching the host file, and the host's real credentials are
    never exposed read-write. The ``:U`` mount option hands ownership to the
    in-container user under the rootless userns. Swapping to a long-lived
    ``CLAUDE_CODE_OAUTH_TOKEN`` later is a different ContainerAuth, no runner change.
    """

    def __init__(self, credentials_path: Path | None = None):
        self.credentials_path = (
            Path(credentials_path)
            if credentials_path is not None
            else Path.home() / ".claude" / ".credentials.json"
        )

    def prepare(self, staging: Path) -> AuthMaterial:
        if not self.credentials_path.is_file():
            raise RunnerError(
                f"no claude credentials at {self.credentials_path}; "
                "sign in with `claude` on the host first"
            )
        cfg = staging / "cfg"
        cfg.mkdir(parents=True, exist_ok=True)
        dest = cfg / ".credentials.json"
        shutil.copyfile(self.credentials_path, dest)
        dest.chmod(0o600)
        return AuthMaterial(
            env={"CLAUDE_CONFIG_DIR": "/cfg"},
            mounts=[(str(cfg), "/cfg", "rw,U")],
        )


# -- Container backend (pluggable) -------------------------------------------


@dataclass
class ContainerSpec:
    image: str
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)
    # (host_src, container_dst, options) — options e.g. "ro", "rw", "rw,U".
    mounts: list[tuple[str, str, str]] = field(default_factory=list)
    workdir: str = "/home/agent"
    user: str = "agent"
    network: bool = True
    memory: str = "4g"
    cpus: float = 2.0
    pids_limit: int = 512
    name: str | None = None
    stdin: str | None = None


@dataclass
class ContainerExecResult:
    returncode: int
    stdout: str
    stderr: str
    container_id: str | None = None
    timed_out: bool = False


@runtime_checkable
class ContainerBackend(Protocol):
    def ensure_image(self) -> None: ...

    def run(
        self,
        spec: ContainerSpec,
        timeout: float,
        cancel_event: threading.Event | None = None,
    ) -> ContainerExecResult: ...


class PodmanBackend:
    """podman backend (rootless on Fedora).

    SELinux confinement is disabled per-container (``label=disable``) so the
    host claude binary and source tree mount without relabeling shared host
    files; isolation still rests on the rootless userns, dropped capabilities,
    ``no-new-privileges``, and a read-only source mount.
    """

    def __init__(self, image_tag: str = IMAGE_TAG, build_timeout: float = 900.0):
        self.image_tag = image_tag
        self.build_timeout = build_timeout

    def _podman(self, args: list[str], timeout: float) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["podman", *args], capture_output=True, text=True, timeout=timeout
        )

    def ensure_image(self) -> None:
        exists = self._podman(["image", "exists", self.image_tag], timeout=30)
        if exists.returncode == 0:
            return
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "Containerfile").write_text(CONTAINERFILE)
            build = self._podman(
                ["build", "-t", self.image_tag, "-f", "Containerfile", td],
                timeout=self.build_timeout,
            )
        if build.returncode != 0:
            raise RunnerError(
                f"podman build failed: {(build.stdout + build.stderr)[-800:]}"
            )

    def run(
        self,
        spec: ContainerSpec,
        timeout: float,
        cancel_event: threading.Event | None = None,
    ) -> ContainerExecResult:
        name = spec.name or "nelson-run"
        cmd = [
            "podman",
            "run",
            "--rm",
            "-i",
            "--name",
            name,
            "--user",
            spec.user,
            "--workdir",
            spec.workdir,
            "--memory",
            spec.memory,
            "--cpus",
            str(spec.cpus),
            "--pids-limit",
            str(spec.pids_limit),
            "--security-opt",
            "no-new-privileges",
            "--security-opt",
            "label=disable",
            "--cap-drop",
            "ALL",
        ]
        if not spec.network:
            cmd += ["--network", "none"]
        for key, value in spec.env.items():
            cmd += ["-e", f"{key}={value}"]
        for src, dst, opts in spec.mounts:
            cmd += ["-v", f"{src}:{dst}:{opts}"]
        cmd += [spec.image, *spec.argv]

        try:
            result = _run_cli(
                cmd, int(timeout), input_text=spec.stdin, cancel_event=cancel_event
            )
        except subprocess.TimeoutExpired:
            # Best-effort cleanup; --rm leaves nothing once killed.
            self._podman(["rm", "-f", name], timeout=30)
            return ContainerExecResult(
                124, "", "timeout", container_id=name, timed_out=True
            )
        return ContainerExecResult(
            result.returncode, result.stdout, result.stderr, container_id=name
        )


# -- claude-code runtime -----------------------------------------------------


def build_competitor_prompt(case: Case) -> str:
    """The audit prompt. Takes ``case`` for future per-profile scoping, but
    deliberately reveals nothing about the planted vulnerability today."""
    return COMPETITOR_PROMPT


def claude_code_spec(
    competitor: Competitor,
    prompt: str,
    src_dir: Path,
    claude_bin: Path,
    auth: AuthMaterial,
    name: str,
    network: bool = True,
) -> ContainerSpec:
    """Build the ContainerSpec that runs ``claude -p`` over /src as a competitor.

    stream-json (+ --verbose) emits the full turn-by-turn transcript on stdout,
    which we keep verbatim; the trailing ``result`` event carries the final text,
    token usage, and cost. ``--add-dir /src`` grants read access to the mounted
    (read-only) source while the agent's writable cwd stays in its home.
    """
    argv = [
        "claude",
        "-p",
        "--model",
        competitor.model,
        "--output-format",
        "stream-json",
        "--verbose",
        "--dangerously-skip-permissions",
        "--add-dir",
        "/src",
    ]
    mounts = [
        (str(claude_bin), "/usr/local/bin/claude", "ro"),
        (str(src_dir), "/src", "ro"),
        *auth.mounts,
    ]
    return ContainerSpec(
        image=IMAGE_TAG,
        argv=argv,
        env={"HOME": "/home/agent", **auth.env},
        mounts=mounts,
        name=name,
        network=network,
        stdin=prompt,
    )


# -- Output parsing ----------------------------------------------------------


def extract_result(stdout: str) -> tuple[str, int | None, int | None, float | None]:
    """Pull (final_text, tokens_in, tokens_out, cost) from claude's output.

    Handles both ``--output-format json`` (one object) and ``stream-json`` (one
    JSON object per line, the last ``type==result`` carrying the totals). Falls
    back to the raw text if nothing parses.
    """
    text = stdout.strip()
    if not text:
        return "", None, None, None

    # stream-json: scan jsonl for the result event (and tolerate a plain object).
    result_obj: dict[str, Any] | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("type") == "result":
            result_obj = obj
    if result_obj is None:
        # Maybe it was a single (non-stream) JSON object.
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "result" in obj:
                result_obj = obj
        except json.JSONDecodeError:
            pass
    if result_obj is None:
        return stdout, None, None, None

    usage = result_obj.get("usage") or {}
    cost = result_obj.get("total_cost_usd")
    if cost is None:
        cost = result_obj.get("cost_usd")
    # ``input_tokens`` alone is only the fresh (uncached) input of the final turn;
    # an agentic run's real input is dominated by cache reads/writes across turns.
    # Sum them so tokens_in reflects total input processed (what cost is based on).
    tokens_in = _sum_input_tokens(usage)
    return (
        str(result_obj.get("result", "")),
        tokens_in,
        usage.get("output_tokens"),
        cost,
    )


def _sum_input_tokens(usage: dict[str, Any]) -> int | None:
    """Total input tokens = fresh + cache-creation + cache-read, or None if absent."""
    keys = (
        "input_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    )
    present = [usage.get(k) for k in keys if isinstance(usage.get(k), int)]
    return sum(present) if present else None


def parse_competitor_findings(text: str) -> list[dict[str, Any]]:
    """Extract the reported findings (JSON array of dicts) from the final text.

    Tolerant of markdown fences / preamble, like ``parse_findings``, but keeps
    the ``file`` field the run layer needs (which the Finding dataclass drops).
    """
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
    except json.JSONDecodeError:
        pass
    start = text.find("[")
    if start == -1:
        return []
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                try:
                    data = json.loads(text[start : i + 1])
                    if isinstance(data, list):
                        return [d for d in data if isinstance(d, dict)]
                except json.JSONDecodeError:
                    pass
                break
    return []


# -- Orchestrator ------------------------------------------------------------


class BenchRunner:
    """Runs competitors against cases and persists the outcome to the DB."""

    def __init__(
        self,
        db,  # nelson.db.Database
        *,
        backend: ContainerBackend | None = None,
        auth: ContainerAuth | None = None,
        claude_bin: Path | None = None,
        cache_dir: str | Path = "bench-cache",
        runs_dir: str | Path = "bench-runs",
        network: bool = True,
        run_timeout: float = 1800.0,
    ):
        self.db = db
        self.backend = backend or PodmanBackend()
        self.auth = auth or CredentialMountAuth()
        self.claude_bin = Path(claude_bin) if claude_bin else _resolve_claude_bin()
        self.cache_dir = Path(cache_dir)
        self.runs_dir = Path(runs_dir)
        self.network = network
        self.run_timeout = run_timeout

    def run_case(self, case: Case, competitor: Competitor) -> RunResult:
        if not case.repo_url or not case.vuln_commit:
            raise RunnerError(
                f"case {case.ext_id} lacks repo_url/vuln_commit; not derived yet"
            )
        comp_id = self.db.upsert_competitor(competitor.to_db_fields())
        case_id = case.id if case.id is not None else self._case_id(case)
        run_id = self.db.create_run(case_id, comp_id)

        try:
            safe_ext = "".join(
                ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in case.ext_id
            )
            checkout_root = self.cache_dir.resolve()
            checkout_dir = (checkout_root / safe_ext).resolve()
            if checkout_root != checkout_dir and checkout_root not in checkout_dir.parents:
                raise RunnerError(f"unsafe case ext_id for cache dir: {case.ext_id!r}")
            checkout = prepare_checkout(
                case.repo_url,
                case.vuln_commit,
                checkout_dir,
            )
            self.backend.ensure_image()
        except (RunnerError, subprocess.TimeoutExpired) as e:
            self.db.mark_run_infra_error(run_id, str(e))
            return RunResult(status="infra_error", error=str(e))

        name = f"nelson-run-{run_id}"
        # Managed manually (not TemporaryDirectory): the container writes into the
        # mounted config dir as a userns-mapped subuid, leaving files the host
        # user can't unlink — so cleanup must go through `podman unshare`.
        staging = Path(tempfile.mkdtemp(prefix="nelson-auth-"))
        try:
            try:
                material = self.auth.prepare(staging)
            except RunnerError as e:
                self.db.mark_run_auth_failed(run_id, str(e))
                return RunResult(status="auth_failed", error=str(e))

            spec = claude_code_spec(
                competitor,
                build_competitor_prompt(case),
                checkout,
                self.claude_bin,
                material,
                name=name,
                network=self.network,
            )
            self.db.start_run(run_id, container_id=name)
            started = time.monotonic()
            exec_result = self.backend.run(spec, self.run_timeout)
            wall = time.monotonic() - started
        finally:
            _safe_rmtree(staging)

        return self._finalize(run_id, exec_result, wall)

    def _finalize(
        self, run_id: int, exec_result: ContainerExecResult, wall: float
    ) -> RunResult:
        combined = exec_result.stdout + exec_result.stderr
        if exec_result.timed_out:
            self.db.mark_run_infra_error(run_id, "run timed out")
            return RunResult(
                status="infra_error",
                error="timeout",
                wall_clock_s=wall,
                container_id=exec_result.container_id,
                transcript=exec_result.stdout,
            )
        if exec_result.returncode != 0:
            kind = classify_failure(combined, failed=True)
            err = f"exit {exec_result.returncode}: {combined[-500:]}"
            if kind is FailureKind.AUTH:
                self.db.mark_run_auth_failed(run_id, err)
                status = "auth_failed"
            else:
                self.db.mark_run_infra_error(run_id, err)
                status = "infra_error"
            return RunResult(
                status=status,
                error=err,
                wall_clock_s=wall,
                container_id=exec_result.container_id,
                transcript=exec_result.stdout,
            )

        text, tin, tout, cost = extract_result(exec_result.stdout)
        findings = parse_competitor_findings(text)
        transcript_path = self._write_transcript(run_id, exec_result.stdout)
        self.db.complete_run(
            run_id,
            tokens_in=tin,
            tokens_out=tout,
            cost_usd=cost,
            wall_clock_s=wall,
            transcript_path=str(transcript_path),
            raw_output=text[:10000],
        )
        for f in findings:
            line = _as_int(f.get("line"))
            self.db.add_run_finding(
                run_id,
                file=_as_str(f.get("file")),
                line_start=line,
                line_end=line,
                description=_as_str(f.get("explanation")),
                confidence=_as_str(f.get("confidence")),
                cwe=_as_str(f.get("cwe")),
            )
        return RunResult(
            status="complete",
            findings=findings,
            raw_output=text,
            transcript=exec_result.stdout,
            tokens_in=tin,
            tokens_out=tout,
            cost_usd=cost,
            wall_clock_s=wall,
            container_id=exec_result.container_id,
        )

    def _write_transcript(self, run_id: int, transcript: str) -> Path:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        path = self.runs_dir / f"run-{run_id}.jsonl"
        path.write_text(transcript)
        return path

    def _case_id(self, case: Case) -> int:
        row = self.db.get_case(case.ext_id)
        if row is None:
            # The case isn't in this DB (e.g. loaded from a manifest); register it.
            return self.db.upsert_case(case.to_db_fields())
        return row["id"]


def _resolve_claude_bin() -> Path:
    """Resolve the host claude binary to a real path for bind-mounting."""
    found = shutil.which("claude")
    if not found:
        raise RunnerError("claude CLI not found on PATH")
    return Path(found).resolve()


def _safe_rmtree(path: Path) -> None:
    """Remove ``path``, falling back to ``podman unshare`` for subuid-owned files.

    A rootless container writes into bind-mounted dirs as a userns-mapped subuid;
    those files are unremovable by the unprivileged host user. ``podman unshare``
    re-enters that namespace (where the subuid maps to root) so the delete
    succeeds. Best-effort: a stray temp dir is harmless if even that fails.
    """
    try:
        shutil.rmtree(path)
        return
    except OSError:
        pass
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        subprocess.run(
            ["podman", "unshare", "rm", "-rf", str(path)],
            capture_output=True,
            timeout=60,
        )


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_str(value: Any) -> str | None:
    return None if value is None else str(value)
