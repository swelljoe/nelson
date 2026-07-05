from pathlib import Path

from nelson.corpus import Case, case_from_manifest, case_to_manifest
from nelson.verify import (
    CandidatePatchVerifier,
    Command,
    CommandResult,
    DifferentialVerifier,
    GitPatchApplier,
    PatchApplicationResult,
    PodmanCommandRunner,
    VerificationConfigError,
    VerificationSpec,
)


def _verification():
    return {
        "invariant": "attacker input cannot cross the boundary",
        "build": [["build"]],
        "witnesses": [
            {
                "name": "primary",
                "command": ["security-test"],
                "vulnerable_exit_codes": [7],
                "fixed_exit_codes": [0],
            }
        ],
        "controls": [["control-test"]],
    }


def _case():
    return Case(
        source="manual",
        ext_id="CVE-test",
        repo_url="repo",
        vuln_commit="bad",
        fix_commit="good",
        verification=_verification(),
    )


class FakeRunner:
    def __init__(self, outcomes):
        self.outcomes = outcomes

    def run(self, command, cwd: Path):
        revision = cwd.parent.name
        code = self.outcomes[(revision, command.argv[0])]
        return CommandResult(command.argv, code)


class FakeBackend:
    def __init__(self):
        self.ready = 0
        self.specs = []

    def ensure_image(self):
        self.ready += 1

    def run(self, spec, timeout, cancel_event=None):
        from nelson.runner import ContainerExecResult

        self.specs.append((spec, timeout))
        return ContainerExecResult(0, "ok", "")


class FakePatchApplier:
    def __init__(self, result=None):
        self.result = result or PatchApplicationResult(True)
        self.calls = []

    def apply(self, patch_path, cwd):
        self.calls.append((patch_path, cwd))
        return self.result


def test_spec_rejects_shell_string_and_requires_witness():
    try:
        VerificationSpec.parse({"invariant": "x", "witnesses": []})
    except VerificationConfigError as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError("invalid spec accepted")


def test_manifest_round_trips_verification():
    case = _case()
    loaded = case_from_manifest(case_to_manifest(case))
    assert loaded.verification == case.verification


def test_differential_verifier_accepts_red_green_and_controls(monkeypatch, tmp_path):
    def checkout(_repo, _commit, dest):
        tree = dest / "src"
        tree.mkdir(parents=True)
        return tree

    monkeypatch.setattr("nelson.verify.prepare_checkout", checkout)
    runner = FakeRunner(
        {
            ("vulnerable", "build"): 0,
            ("vulnerable", "security-test"): 7,
            ("vulnerable", "control-test"): 0,
            ("fixed", "build"): 0,
            ("fixed", "security-test"): 0,
            ("fixed", "control-test"): 0,
        }
    )
    result = DifferentialVerifier(runner).verify(_case(), tmp_path)
    assert result.verified
    assert len(result.checks) == 6


def test_differential_verifier_rejects_witness_that_fails_on_fixed(
    monkeypatch, tmp_path
):
    def checkout(_repo, _commit, dest):
        tree = dest / "src"
        tree.mkdir(parents=True)
        return tree

    monkeypatch.setattr("nelson.verify.prepare_checkout", checkout)
    runner = FakeRunner(
        {
            ("vulnerable", "build"): 0,
            ("vulnerable", "security-test"): 7,
            ("vulnerable", "control-test"): 0,
            ("fixed", "build"): 0,
            ("fixed", "security-test"): 7,
            ("fixed", "control-test"): 0,
        }
    )
    result = DifferentialVerifier(runner).verify(_case(), tmp_path)
    assert not result.verified
    failed = [check for check in result.checks if not check.passed]
    assert [(check.revision, check.name) for check in failed] == [("fixed", "primary")]


def test_missing_harness_is_not_reported_as_failed_proof(tmp_path):
    case = _case()
    case.verification = None
    result = DifferentialVerifier(FakeRunner({})).verify(case, tmp_path)
    assert not result.verified
    assert result.error == "case has no verification mapping"


def test_podman_runner_uses_rw_workspace_without_network(tmp_path):
    backend = FakeBackend()
    harness = tmp_path / "harness"
    harness.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner = PodmanCommandRunner(backend, harness_dir=harness)
    result = runner.run(Command(["test", "-f", "x"]), workspace)
    assert result.exit_code == 0
    spec, timeout = backend.specs[0]
    assert spec.argv == ["test", "-f", "x"]
    assert spec.mounts == [
        (str(workspace.resolve()), "/workspace", "rw,U"),
        (str(harness.resolve()), "/harness", "ro"),
    ]
    assert spec.workdir == "/workspace"
    assert spec.network is False
    assert timeout == 600
    assert backend.ready == 1


def test_git_patch_applier_applies_plain_unified_diff(tmp_path):
    source = tmp_path / "message.txt"
    source.write_text("vulnerable\n")
    patch = tmp_path / "candidate.diff"
    patch.write_text(
        "--- a/message.txt\n"
        "+++ b/message.txt\n"
        "@@ -1 +1 @@\n"
        "-vulnerable\n"
        "+fixed\n"
    )

    result = GitPatchApplier().apply(patch, tmp_path)

    assert result.applied
    assert result.error is None
    assert source.read_text() == "fixed\n"


def test_git_patch_applier_rejects_empty_and_invalid_patches(tmp_path):
    empty = tmp_path / "empty.diff"
    empty.write_text("")
    invalid = tmp_path / "invalid.diff"
    invalid.write_text("not a unified diff\n")

    empty_result = GitPatchApplier().apply(empty, tmp_path)
    invalid_result = GitPatchApplier().apply(invalid, tmp_path)

    assert not empty_result.applied
    assert empty_result.error == "patch is empty"
    assert not invalid_result.applied
    assert invalid_result.error


def test_candidate_verifier_uses_fixed_expectations(monkeypatch, tmp_path):
    def checkout(_repo, _commit, dest):
        tree = dest / "src"
        tree.mkdir(parents=True)
        (tree / "source.txt").write_text("pristine")
        return tree

    monkeypatch.setattr("nelson.verify.prepare_checkout", checkout)
    runner = FakeRunner(
        {
            ("candidate", "build"): 0,
            ("candidate", "security-test"): 0,
            ("candidate", "control-test"): 0,
        }
    )
    applier = FakePatchApplier()
    patch = tmp_path / "candidate.diff"
    patch.write_text("ignored by fake")

    result = CandidatePatchVerifier(runner, applier).verify(
        _case(), patch, tmp_path / "work"
    )

    assert result.verified
    assert result.patch == PatchApplicationResult(True)
    assert [(check.kind, check.expected_exit_codes) for check in result.checks] == [
        ("build", frozenset({0})),
        ("witness", frozenset({0})),
        ("control", frozenset({0})),
    ]
    assert applier.calls[0][1].parent.name == "candidate"


def test_candidate_verifier_reports_patch_failure_without_running_checks(
    monkeypatch, tmp_path
):
    def checkout(_repo, _commit, dest):
        tree = dest / "src"
        tree.mkdir(parents=True)
        return tree

    monkeypatch.setattr("nelson.verify.prepare_checkout", checkout)
    failure = PatchApplicationResult(False, "patch does not apply")
    patch = tmp_path / "candidate.diff"
    patch.write_text("invalid")

    result = CandidatePatchVerifier(
        FakeRunner({}), FakePatchApplier(failure)
    ).verify(_case(), patch, tmp_path / "work")

    assert not result.verified
    assert result.error is None
    assert result.patch == failure
    assert result.checks == []
