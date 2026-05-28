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
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .agents import FailureKind, _run_cli, classify_failure
from .prevet import _sanitize_escapes

if TYPE_CHECKING:
    from .corpus import Case

# Minimal Fedora base + the tools the agent's own grep/file tools shell out to.
# The claude binary itself is bind-mounted at run time (see ClaudeCodeRuntime).
# python3 backs the raw-api-loop runtime's in-container ReAct agent (stdlib only,
# no pip). The tag carries a `-py` suffix so `ensure_image` rebuilds rather than
# reusing an older python-less image cached under the bare `:fedora` tag.
IMAGE_TAG = "nelson-bench:fedora-py"
CONTAINERFILE = """\
FROM registry.fedoraproject.org/fedora-minimal:41
RUN microdnf install -y git ripgrep ca-certificates findutils shadow-utils python3 \\
    && microdnf clean all
# A real passwd entry so the agent's getpwuid() works and HOME is writable.
RUN useradd -u 1000 -m -s /bin/bash agent
USER agent
WORKDIR /home/agent
"""

# The competitor is pointed at one file but told NOTHING about the planted vuln
# (no CVE id, description, or bug class) — that would defeat the benchmark (and
# the FP-judge must never see it either). The framing is deliberately *neutral*
# ("review this file"), not "find the bug here": a presumptive prompt would invite
# guessing and wreck both detection and false-positive measurement. Pointing at the
# file mirrors a real file-by-file audit; it scopes the work without leaking what
# to look for. The whole repo is still mounted so the model can follow a definition
# or caller for context, but it reports only on the file under review.
_AUDIT_INTRO = """\
You are an expert security auditor performing a file-by-file review of an
unfamiliar codebase. The complete source tree is mounted read-only at /src.

Review this one file for *exploitable* security vulnerabilities:

    {target}

Read it closely. You may open other files under /src ONLY to understand the file
under review (the definitions, callers, and helpers it depends on) — but report
only vulnerabilities in {target} itself. Look for memory-safety bugs, injection
(SQL / command / path), authentication or authorization flaws, unsafe
deserialization, SSRF, and the like. Report only real, exploitable issues — not
style nits or theoretical concerns. Many files in a codebase are perfectly clean:
if {target} has no exploitable vulnerability, that is a valid and expected answer."""

_OUTPUT_SPEC = """\
When you are done, output ONLY a JSON array as your final message, with no prose
around it. Each element must be:
{"file": "<path relative to /src>", "line": <int>, "code": "<the vulnerable line>",
 "explanation": "<why it is exploitable>", "confidence": "high|medium|low",
 "cwe": "<CWE-id if known, else null>"}

If you find no vulnerabilities in the file, output []."""


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

    @classmethod
    def from_row(cls, row: Any) -> Competitor:
        """Rebuild a Competitor from a ``competitors`` row (drops the DB id)."""
        return cls(
            name=row["name"],
            model=row["model"],
            runtime=row["runtime"],
            tool_profile=row["tool_profile"],
            auth_profile=row["auth_profile"],
            cost_model=row["cost_model"],
            size_class=row["size_class"],
            knowledge_cutoff=row["knowledge_cutoff"],
            status=row["status"],
        )


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
    """Materialize a pristine working tree at ``<dest>/src`` for ``commit``.

    The competitor must not be able to identify the project or look up the fix:
    ``.git/config`` would name the upstream repo, and with the model's network
    on, one ``git fetch origin <fix-sha>`` (or a fresh clone into the writable
    home) would put the upstream fix one ``git diff`` away. So the *mount* —
    ``<dest>/src``, what bind-mounts at ``/src:ro`` — is a pristine tree with
    no ``.git`` at all: we fetch into a scratch bare repo under
    ``<dest>/.gitcache`` (a sibling, **not** under the mount) and
    ``git archive | tar -x`` the commit's tree into ``<dest>/src``. The
    ``<dest>/.commit`` sentinel lets repeat runs of the same case skip the
    fetch.

    Returns ``<dest>/src``, the path to bind-mount as ``/src:ro``.
    """
    dest = Path(dest)
    tree = dest / "src"
    sentinel = dest / ".commit"
    gitdir = dest / ".gitcache"
    if sentinel.is_file() and sentinel.read_text().strip() == commit and tree.is_dir():
        return tree
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    tree.mkdir()
    gitdir.mkdir()
    _git(["init", "--bare", "-q"], gitdir, timeout)
    _git(["remote", "add", "origin", repo_url], gitdir, timeout)
    _git(["fetch", "--depth=1", "-q", "origin", commit], gitdir, timeout)
    _archive_tree(gitdir, commit, tree, timeout)
    sentinel.write_text(commit)
    return tree


def _archive_tree(gitdir: Path, commit: str, tree: Path, timeout: float) -> None:
    """Extract ``commit``'s tree from the bare ``gitdir`` into ``tree``.

    ``git archive`` writes a tarball of just the tree at ``commit`` — no
    history, no refs, no ``.git`` metadata — which ``tar -x`` unpacks into the
    mount. Routed through a tarball file (rather than a pipe) so a failure on
    either side surfaces with its own returncode and stderr.
    """
    tarball = gitdir / "archive.tar"
    arch = subprocess.run(
        [
            "git",
            "--git-dir",
            str(gitdir),
            "archive",
            "--format=tar",
            "-o",
            str(tarball),
            commit,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if arch.returncode != 0:
        raise RunnerError(f"git archive failed: {arch.stderr.strip()}")
    ext = subprocess.run(
        ["tar", "-xf", str(tarball), "-C", str(tree)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if ext.returncode != 0:
        raise RunnerError(f"tar extract failed: {ext.stderr.strip()}")
    tarball.unlink()


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


# Path fragments that mark a file as a test rather than vulnerable product code.
# Test files routinely appear in a fix diff (the regression test for the bug), but
# we audit the code that *had* the bug, not the test that demonstrates the fix.
_TEST_PATH_MARKERS = ("/test/", "/tests/", "/testing/", "__tests__")
_TEST_NAME_RE = re.compile(
    r"(^|[._-])test([._-]|$)|test\.(java|py|js|ts|go|rb)$|"
    r"(test|spec)s?\.(jsx?|tsx?)$|_(test|spec)\.|\.(test|spec)\.",
    re.IGNORECASE,
)


def _is_test_path(path: str) -> bool:
    lower = path.lower()
    if any(marker in lower for marker in _TEST_PATH_MARKERS):
        return True
    name = path.rsplit("/", 1)[-1]
    return bool(_TEST_NAME_RE.search(name))


def vulnerable_files(case: Case) -> list[str]:
    """The product files to point a competitor at: distinct, non-test files that
    carry a real pre-patch hunk.

    Derived ground truth lists every file the fix touched, including the added
    regression test; we audit the files that actually *contained* the bug. Falls
    back to all hunk files (then ``gt_files``) if filtering would leave nothing,
    so a case is never silently un-runnable.
    """
    hunk_files = list(dict.fromkeys(h["file"] for h in case.gt_hunks if h.get("file")))
    product = [f for f in hunk_files if not _is_test_path(f)]
    if product:
        return product
    if hunk_files:
        return hunk_files  # all hunks were in test files; audit them anyway
    return [f for f in case.gt_files if not _is_test_path(f)] or list(case.gt_files)


def build_competitor_prompt(case: Case, target_file: str) -> str:
    """The neutral, file-scoped audit prompt for ``target_file``.

    Takes ``case`` for parity with other builders, but deliberately reveals
    nothing about the planted vulnerability — only which file to review."""
    safe_target = target_file.replace("{", "{{").replace("}", "}}")
    return f"{_AUDIT_INTRO.format(target=safe_target)}\n\n{_OUTPUT_SPEC}"


def claude_code_spec(
    competitor: Competitor,
    prompt: str,
    src_dir: Path,
    claude_bin: Path,
    auth: AuthMaterial,
    name: str,
    network: bool = True,
    max_budget_usd: float | None = None,
) -> ContainerSpec:
    """Build the ContainerSpec that runs ``claude -p`` over /src as a competitor.

    stream-json (+ --verbose) emits the full turn-by-turn transcript on stdout,
    which we keep verbatim; the trailing ``result`` event carries the final text,
    token usage, and cost. ``--add-dir /src`` grants read access to the mounted
    (read-only) source while the agent's writable cwd stays in its home.
    ``max_budget_usd`` is a backstop against a runaway wander (the prompt scoping
    is the primary cost control); ``--max-budget-usd`` only takes effect with -p.
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
    if max_budget_usd is not None:
        argv += ["--max-budget-usd", str(max_budget_usd)]
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


_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


def _load_findings_list(span: str) -> list[dict[str, Any]] | None:
    """json.loads ``span`` into a list of finding dicts, or None if it isn't one.

    Retries once with invalid backslash escapes repaired — models routinely emit
    them when quoting vulnerable code into the ``code`` / ``explanation`` fields.
    Only arrays carrying at least one object count: a bare ``[]`` (no findings) or
    a prose array of strings (e.g. ``["..","evil.txt"]``) is not a findings list.
    """
    for candidate in (span, _sanitize_escapes(span)):
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, list):
            dicts = [d for d in data if isinstance(d, dict)]
            if dicts:
                return dicts
            if data == []:
                return []  # explicit "no findings"
    return None


def _balanced_arrays(text: str) -> list[str]:
    """Every brace-balanced ``[...]`` span in ``text``, outermost, in order."""
    spans: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] == "[":
            depth = 0
            for j in range(i, n):
                if text[j] == "[":
                    depth += 1
                elif text[j] == "]":
                    depth -= 1
                    if depth == 0:
                        spans.append(text[i : j + 1])
                        i = j
                        break
        i += 1
    return spans


def parse_competitor_findings(text: str) -> list[dict[str, Any]]:
    """Extract the reported findings (JSON array of dicts) from the final text.

    The model is asked for a bare JSON array, but in practice often wraps it in a
    reasoning summary and a ```json fence — and that prose can itself contain
    bracketed text (``["..","evil.txt"]``) that looks like, but isn't, the answer.
    So: try the whole text; then any fenced block; then every balanced ``[...]``
    span — and accept only an array that actually carries finding objects,
    preferring the *last* such (a model's final answer comes last).
    """
    text = text.strip()
    whole = _load_findings_list(text)
    if whole is not None:
        return whole

    best: list[dict[str, Any]] | None = None
    for fenced in _FENCE_RE.findall(text):
        parsed = _load_findings_list(fenced.strip())
        if parsed:  # non-empty findings in a fence win outright
            best = parsed
    if best is not None:
        return best

    for span in _balanced_arrays(text):
        parsed = _load_findings_list(span)
        if parsed:
            best = parsed  # keep the last array that carries findings
    return best or []


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
        max_budget_usd: float | None = 0.50,
        preflight: bool = False,
    ):
        self.db = db
        self.backend = backend or PodmanBackend()
        # An explicitly-injected auth wins for every runtime (tests' FakeAuth, a
        # future global OAuth). Left None, each runtime supplies its own default
        # (claude -> credential mount, API-key runtimes -> env injection) in
        # _select_auth, so the legacy claude-code default is preserved.
        self._explicit_auth = auth
        # Resolved lazily per-run (only the claude-code runtime needs it), so a
        # host without `claude` can still run API-key/other runtimes; a missing
        # claude binary then surfaces as infra_error on a claude run, not a
        # construction crash.
        self.claude_bin = Path(claude_bin) if claude_bin else None
        self.cache_dir = Path(cache_dir)
        self.runs_dir = Path(runs_dir)
        self.network = network
        self.run_timeout = run_timeout
        self.max_budget_usd = max_budget_usd
        self.preflight = preflight

    def run_case(
        self, case: Case, competitor: Competitor, target_file: str
    ) -> RunResult:
        """Point ``competitor`` at ``target_file`` of ``case`` in a container."""
        if not case.repo_url or not case.vuln_commit:
            raise RunnerError(
                f"case {case.ext_id} lacks repo_url/vuln_commit; not derived yet"
            )
        comp_id = self.db.upsert_competitor(competitor.to_db_fields())
        case_id = case.id if case.id is not None else self._case_id(case)
        run_id = self.db.create_run(case_id, comp_id, target_file)

        # Function-body imports keep runtimes.py's module-level dependency on this
        # module one-directional (no import cycle), matching the codebase idiom.
        from .runtimes import RuntimeContext, get_runtime

        try:
            # An unknown runtime is a config error: the model never got a fair
            # look, so it is infra_error (handled alongside checkout/build), never
            # a miss.
            runtime = get_runtime(competitor.runtime)
            safe_ext = "".join(
                ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_"
                for ch in case.ext_id
            )
            checkout_root = self.cache_dir.resolve()
            checkout_dir = (checkout_root / safe_ext).resolve()
            if (
                checkout_root != checkout_dir
                and checkout_root not in checkout_dir.parents
            ):
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

        auth = self._select_auth(competitor, runtime)

        # Cheap host-side auth probe (default off) so a dead API key fails before
        # we spend on a container. A missing key is already auth_failed at
        # auth.prepare(); this only avoids the wasted run.
        if self.preflight:
            pf = self._preflight(competitor)
            if pf is not None and not pf.ok:
                status = "auth_failed" if pf.status == "auth_failed" else "infra_error"
                if status == "auth_failed":
                    self.db.mark_run_auth_failed(run_id, pf.detail)
                else:
                    self.db.mark_run_infra_error(run_id, pf.detail)
                return RunResult(status=status, error=pf.detail)

        name = f"nelson-run-{run_id}"
        # Managed manually (not TemporaryDirectory): the container writes into the
        # mounted config dir as a userns-mapped subuid, leaving files the host
        # user can't unlink — so cleanup must go through `podman unshare`.
        staging = Path(tempfile.mkdtemp(prefix="nelson-auth-"))
        try:
            try:
                material = auth.prepare(staging)
            except RunnerError as e:
                self.db.mark_run_auth_failed(run_id, str(e))
                return RunResult(status="auth_failed", error=str(e))

            try:
                ctx = RuntimeContext(
                    competitor=competitor,
                    prompt=build_competitor_prompt(case, target_file),
                    src_dir=checkout,
                    auth=material,
                    name=name,
                    claude_bin=self.claude_bin,
                    network=self.network,
                    max_budget_usd=self.max_budget_usd,
                )
                spec = runtime.build_spec(ctx)
            except RunnerError as e:
                # e.g. a native CLI runtime whose binary isn't installed: a setup
                # failure, never a miss.
                self.db.mark_run_infra_error(run_id, str(e))
                return RunResult(status="infra_error", error=str(e))

            self.db.start_run(run_id, container_id=name)
            started = time.monotonic()
            exec_result = self.backend.run(spec, self.run_timeout)
            wall = time.monotonic() - started
        finally:
            _safe_rmtree(staging)

        return self._finalize(run_id, exec_result, wall, runtime, competitor)

    def _select_auth(self, competitor: Competitor, runtime) -> ContainerAuth:
        """Pick the ContainerAuth for this run.

        An explicitly-injected auth always wins. Otherwise a named ``auth_profile``
        means API-key env injection; absent that, the runtime's own default
        (claude credential mount, gemini credential mount, ...) is used. Resolution
        of secrets is deferred to ``prepare()`` so selection itself never fails.
        """
        if self._explicit_auth is not None:
            return self._explicit_auth
        from .runtimes import auth_for_competitor

        return auth_for_competitor(competitor, runtime)

    def _preflight(self, competitor: Competitor):
        """Host-side reachability/auth probe for OpenAI-compatible API runtimes.

        Returns a PreflightResult, or None when preflight does not apply
        (subscription/credential-mount runtimes, or no auth_profile). Reuses the
        host-side OpenAIAPIAdapter purely as a cheap probe (it maps 401/403 ->
        auth_failed, 429 -> rate_limited, connect error -> infra_error).
        """
        from .agents import OpenAIAPIAdapter
        from .auth import STANDARD_AUTH_PROFILES
        from .runtimes import (
            API_KEY_RUNTIMES,
            DEFAULT_OPENAI_BASE_URL,
            parse_cost_model,
        )

        if not competitor.auth_profile or competitor.runtime not in API_KEY_RUNTIMES:
            return None
        profile = STANDARD_AUTH_PROFILES.get(competitor.auth_profile)
        if profile is None:
            return None
        base_url = parse_cost_model(competitor.cost_model).get(
            "base_url", DEFAULT_OPENAI_BASE_URL
        )
        adapter = OpenAIAPIAdapter(
            model=competitor.model, base_url=str(base_url), auth_profile=profile
        )
        return adapter.preflight()

    def _finalize(
        self,
        run_id: int,
        exec_result: ContainerExecResult,
        wall: float,
        runtime,
        competitor: Competitor,
    ) -> RunResult:
        combined = exec_result.stdout + exec_result.stderr
        if exec_result.timed_out:
            transcript_path = self._write_transcript(run_id, combined)
            self.db.mark_run_infra_error(
                run_id,
                "run timed out",
                wall_clock_s=wall,
                transcript_path=str(transcript_path),
                raw_output=combined[-10000:],
            )
            return RunResult(
                status="infra_error",
                error="timeout",
                wall_clock_s=wall,
                container_id=exec_result.container_id,
                transcript=combined,
            )
        if exec_result.returncode != 0:
            transcript_path = self._write_transcript(run_id, combined)
            kind = classify_failure(combined, failed=True)
            err = f"exit {exec_result.returncode}: {combined[-500:]}"
            if kind is FailureKind.AUTH:
                self.db.mark_run_auth_failed(
                    run_id,
                    err,
                    wall_clock_s=wall,
                    transcript_path=str(transcript_path),
                    raw_output=combined[-10000:],
                )
                status = "auth_failed"
            else:
                self.db.mark_run_infra_error(
                    run_id,
                    err,
                    wall_clock_s=wall,
                    transcript_path=str(transcript_path),
                    raw_output=combined[-10000:],
                )
                status = "infra_error"
            return RunResult(
                status=status,
                error=err,
                wall_clock_s=wall,
                container_id=exec_result.container_id,
                transcript=combined,
            )

        # Each runtime owns its stdout contract: claude-code/raw-api-loop emit a
        # claude-shaped result object (extract_result), gemini-cli emits its own
        # envelope. parse_output normalizes them to text + usage + findings.
        parsed = runtime.parse_output(exec_result, competitor)
        text = parsed.final_text
        findings = parsed.findings
        transcript_path = self._write_transcript(run_id, exec_result.stdout)
        self.db.complete_run(
            run_id,
            tokens_in=parsed.tokens_in,
            tokens_out=parsed.tokens_out,
            cost_usd=parsed.cost,
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
            tokens_in=parsed.tokens_in,
            tokens_out=parsed.tokens_out,
            cost_usd=parsed.cost,
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
