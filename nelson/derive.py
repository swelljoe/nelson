"""Pre-patch derivation: turn a fix commit into vulnerable-code ground truth.

The benchmark runs a competitor against the repository *as it was before* the
fix — so the ground truth is the pre-patch (vulnerable) commit and the lines the
fix touched, expressed in pre-patch coordinates. Given ``repo_url`` + ``fix_commit``:

  vuln_commit = first parent of the fix commit
  gt_files    = files the fix changed
  gt_hunks    = the OLD-side line ranges of each hunk (where the bug lived)

We deliberately take the ``-`` (old) side of each ``@@`` header: a model looking
at the pre-patch checkout sees those lines, and the localization gate scores a
finding by its distance to them.

Git access is behind :class:`GitRunner` so derivation is testable without network
(a fake runner, or a tiny real local repo). Anything that makes a clean
derivation impossible — no fix commit, an unfetchable SHA, a root commit with no
parent, an empty/binary-only diff — returns ``ok=False`` with a reason rather
than raising, so the pipeline can flag the case for the pre-vet step to retire.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .corpus import Case

# @@ -old_start[,old_len] +new_start[,new_len] @@   (lengths default to 1)
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class GitError(Exception):
    """A git operation failed (non-zero exit or unreachable commit)."""


@runtime_checkable
class GitRunner(Protocol):
    def prepare(self, repo_url: str, commit: str, dest: Path) -> None:
        """Ensure ``commit`` and its parent exist locally at ``dest``."""
        ...

    def rev_parse(self, dest: Path, rev: str) -> str | None:
        """Resolve ``rev`` to a full SHA, or None if it doesn't resolve."""
        ...

    def diff(self, dest: Path, base: str, head: str) -> str:
        """Unified diff of ``base``..``head``."""
        ...


class SubprocessGitRunner:
    """GitRunner backed by the ``git`` CLI, fetching only what derivation needs.

    A shallow (depth=2) fetch of the fix commit brings down the commit and its
    parent without cloning history. ``repo_url`` may be a remote URL or a local
    path (used by tests). Reachable-SHA fetch works on GitHub for commits
    reachable from a ref, which covers published fixes.
    """

    def __init__(self, timeout: float = 180.0):
        self.timeout = timeout

    def _run(self, args: list[str], cwd: Path) -> str:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.timeout,
        )
        if proc.returncode != 0:
            raise GitError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
        return proc.stdout

    def prepare(self, repo_url: str, commit: str, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        if not (dest / ".git").exists():
            self._run(["init", "-q"], dest)
        # Idempotent remote setup.
        try:
            self._run(["remote", "add", "origin", repo_url], dest)
        except GitError:
            self._run(["remote", "set-url", "origin", repo_url], dest)
        self._run(["fetch", "--depth=2", "-q", "origin", commit], dest)

    def rev_parse(self, dest: Path, rev: str) -> str | None:
        try:
            return self._run(["rev-parse", "--verify", "-q", rev], dest).strip() or None
        except GitError:
            return None

    def diff(self, dest: Path, base: str, head: str) -> str:
        return self._run(["diff", base, head], dest)

    def show(self, dest: Path, rev: str, path: str) -> str:
        """Contents of ``path`` at ``rev`` (``git show rev:path``).

        ``path`` must be repo-root-relative with forward slashes. Used by the P4
        FP judge to read the pre-patch source it grounds its verdict on; raises
        GitError if the file does not exist at that revision."""
        return self._run(["show", f"{rev}:{path}"], dest)


@dataclass
class Derivation:
    """Outcome of deriving ground truth for one case."""

    ok: bool
    vuln_commit: str | None = None
    gt_files: list[str] = field(default_factory=list)
    gt_hunks: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""
    # Raw fix diff, kept in memory as evidence for the pre-vet judge. Not a DB
    # column (see updates()), since the repo can reproduce it on demand.
    diff: str = ""

    def updates(self) -> dict[str, Any]:
        """Persistable case fields (only meaningful when ``ok``)."""
        return {
            "vuln_commit": self.vuln_commit,
            "gt_files": self.gt_files,
            "gt_hunks": self.gt_hunks,
        }


def _strip_prefix(path: str) -> str:
    """Drop git's ``a/``/``b/`` diff prefix; keep ``/dev/null`` as a sentinel."""
    if path in ("a/", "b/"):
        return ""
    for prefix in ("a/", "b/"):
        if path.startswith(prefix):
            return path[len(prefix) :]
    return path


def parse_diff_hunks(diff_text: str) -> tuple[list[str], list[dict[str, Any]]]:
    """Parse a unified diff into (changed files, pre-patch hunk ranges).

    Each hunk is ``{"file", "start", "end"}`` in OLD-file line numbers. A hunk
    whose old length is 0 (pure addition) contributes no range but its file is
    still recorded. Files are keyed by their pre-patch path (the old side), or
    the new path when the file was added.
    """
    files: list[str] = []
    hunks: list[dict[str, Any]] = []
    old_path = new_path = ""
    for line in diff_text.splitlines():
        if line.startswith("--- "):
            old_path = _strip_prefix(line[4:].strip())
        elif line.startswith("+++ "):
            new_path = _strip_prefix(line[4:].strip())
        elif line.startswith("@@"):
            m = _HUNK_RE.match(line)
            if not m:
                continue
            old_start = int(m.group(1))
            old_len = int(m.group(2)) if m.group(2) is not None else 1
            # File added in the fix -> no pre-patch location; attribute to the
            # new path so the file is still counted.
            target = new_path if old_path in ("", "/dev/null") else old_path
            if target and target not in files:
                files.append(target)
            if old_len > 0 and old_path not in ("", "/dev/null"):
                hunks.append(
                    {
                        "file": old_path,
                        "start": old_start,
                        "end": old_start + old_len - 1,
                    }
                )
    return files, hunks


def derive_ground_truth(
    case: Case, git: GitRunner, cache_dir: str | Path
) -> Derivation:
    """Derive pre-patch ground truth for ``case`` using ``git`` under ``cache_dir``.

    ``cache_dir`` holds one working tree per repo (reused across cases from the
    same project). Returns a :class:`Derivation`; on failure ``ok`` is False and
    ``reason`` explains why, leaving the case un-derived for the pre-vet step.
    """
    if not case.fix_commit or not case.repo_url:
        return Derivation(ok=False, reason="missing repo_url or fix_commit")

    dest = Path(cache_dir) / _repo_slug(case.repo_url)
    try:
        git.prepare(case.repo_url, case.fix_commit, dest)
    except GitError as e:
        return Derivation(ok=False, reason=f"fetch failed: {e}")

    parent = git.rev_parse(dest, f"{case.fix_commit}^1")
    if not parent:
        return Derivation(ok=False, reason="fix commit has no parent (root commit?)")

    try:
        diff_text = git.diff(dest, parent, case.fix_commit)
    except GitError as e:
        return Derivation(ok=False, reason=f"diff failed: {e}")

    files, hunks = parse_diff_hunks(diff_text)
    if not files:
        return Derivation(ok=False, reason="empty or binary-only diff", diff=diff_text)
    return Derivation(
        ok=True, vuln_commit=parent, gt_files=files, gt_hunks=hunks, diff=diff_text
    )


def _repo_slug(repo_url: str) -> str:
    """Filesystem-safe cache directory name for a repo URL."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", repo_url.rstrip("/"))
    return slug.removesuffix(".git").strip("_") or "repo"
