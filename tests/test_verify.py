from pathlib import Path

from nelson.corpus import Case, case_from_manifest, case_to_manifest
from nelson.verify import (
    Command,
    CommandResult,
    DifferentialVerifier,
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
