"""Pre-patch derivation: diff parsing, orchestration, and a real local repo."""

import os
import subprocess

from nelson.corpus import Case
from nelson.derive import (
    Derivation,
    GitError,
    GitRunner,
    SubprocessGitRunner,
    derive_ground_truth,
    parse_diff_hunks,
)

_DIFF = """\
diff --git a/src/foo.c b/src/foo.c
index 1111111..2222222 100644
--- a/src/foo.c
+++ b/src/foo.c
@@ -10,3 +10,4 @@ void f() {
 a
-b
+b2
+c
@@ -40 +41,2 @@
-x
+y
+z
diff --git a/newfile.c b/newfile.c
new file mode 100644
--- /dev/null
+++ b/newfile.c
@@ -0,0 +1,2 @@
+hello
+world
"""

_RENAME_DIFF = """\
diff --git a/old/path.c b/new/path.c
similarity index 90%
rename from old/path.c
rename to new/path.c
--- a/old/path.c
+++ b/new/path.c
@@ -5,2 +5,2 @@
-old
+new
"""


def test_parse_diff_uses_old_side_line_ranges():
    files, hunks = parse_diff_hunks(_DIFF)
    assert files == ["src/foo.c", "newfile.c"]
    # old_len 3 from line 10 -> 10..12; omitted length defaults to 1 -> 40..40.
    assert {"file": "src/foo.c", "start": 10, "end": 12} in hunks
    assert {"file": "src/foo.c", "start": 40, "end": 40} in hunks
    # Added file (old side /dev/null) is counted as a file but yields no hunk.
    assert all(h["file"] != "newfile.c" for h in hunks)


def test_parse_diff_rename_keys_pre_patch_path():
    files, hunks = parse_diff_hunks(_RENAME_DIFF)
    assert files == ["old/path.c"]  # pre-patch (old) path
    assert hunks == [{"file": "old/path.c", "start": 5, "end": 6}]


def test_parse_empty_diff():
    assert parse_diff_hunks("") == ([], [])


class FakeGit:
    """Canned GitRunner: configurable parent SHA, diff text, and fetch failure."""

    def __init__(self, parent=None, diff="", prepare_error=None):
        self._parent = parent
        self._diff = diff
        self._prepare_error = prepare_error
        self.prepared = False

    def prepare(self, repo_url, commit, dest):
        if self._prepare_error:
            raise GitError(self._prepare_error)
        self.prepared = True

    def rev_parse(self, dest, rev):
        return self._parent

    def diff(self, dest, base, head):
        return self._diff


def test_fakegit_satisfies_runner_protocol():
    assert isinstance(FakeGit(), GitRunner)


def _case(**kw):
    return Case(
        source="cvd",
        ext_id="GHSA-x",
        repo_url="https://github.com/o/r",
        fix_commit="f" * 40,
        **kw,
    )


def test_derive_success_sets_vuln_commit_and_ground_truth(tmp_path):
    git = FakeGit(parent="p" * 40, diff=_DIFF)
    der = derive_ground_truth(_case(), git, tmp_path)
    assert der.ok
    assert der.vuln_commit == "p" * 40
    assert "src/foo.c" in der.gt_files
    assert der.gt_hunks
    assert der.updates()["vuln_commit"] == "p" * 40


def test_derive_requires_repo_and_fix_commit(tmp_path):
    case = Case(source="cvd", ext_id="CVE-2026-27654")  # un-enriched 2026 seed
    der = derive_ground_truth(case, FakeGit(), tmp_path)
    assert der == Derivation(ok=False, reason="missing repo_url or fix_commit")


def test_derive_handles_root_commit_no_parent(tmp_path):
    der = derive_ground_truth(_case(), FakeGit(parent=None, diff=_DIFF), tmp_path)
    assert not der.ok
    assert "no parent" in der.reason


def test_derive_handles_fetch_failure(tmp_path):
    der = derive_ground_truth(_case(), FakeGit(prepare_error="boom"), tmp_path)
    assert not der.ok
    assert "fetch failed" in der.reason


def test_derive_flags_empty_diff(tmp_path):
    der = derive_ground_truth(_case(), FakeGit(parent="p" * 40, diff=""), tmp_path)
    assert not der.ok
    assert "empty" in der.reason


# Deterministic identity so commits work without the user's global git config.
_GIT_ENV = os.environ | {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
}


def _git(args, cwd):
    subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
        env=_GIT_ENV,
    )


def test_subprocess_runner_derives_from_local_repo(tmp_path):
    # Build a real two-commit repo: a vulnerable version, then the fix.
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git(["init", "-q", "-b", "main"], upstream)
    (upstream / "vuln.c").write_text("int main(){\n  gets(buf);\n  return 0;\n}\n")
    _git(["add", "."], upstream)
    _git(["commit", "-q", "-m", "vulnerable"], upstream)
    (upstream / "vuln.c").write_text(
        "int main(){\n  fgets(buf,n,stdin);\n  return 0;\n}\n"
    )
    _git(["add", "."], upstream)
    _git(["commit", "-q", "-m", "fix the bug"], upstream)

    def _rev(rev):
        return subprocess.run(
            ["git", "rev-parse", rev],
            cwd=str(upstream),
            capture_output=True,
            text=True,
        ).stdout.strip()

    fix, parent = _rev("HEAD"), _rev("HEAD~1")

    case = Case(
        source="manual", ext_id="LOCAL-1", repo_url=str(upstream), fix_commit=fix
    )
    der = derive_ground_truth(case, SubprocessGitRunner(), tmp_path / "cache")

    assert der.ok, der.reason
    assert der.vuln_commit == parent
    assert "vuln.c" in der.gt_files
    assert any(h["file"] == "vuln.c" for h in der.gt_hunks)
