"""Executable differential verification for benchmark corpus cases.

A case harness is trusted benchmark metadata, not model output.  It is run on
both the vulnerable and upstream-fixed commits.  A harness is verified only
when every build and compatibility control succeeds on both revisions and each
security witness produces its declared vulnerable/fixed outcomes.

Commands are argv arrays and are executed without a shell.  This both makes
manifests reproducible and avoids surprising shell interpolation.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .runner import (
    IMAGE_TAG,
    ContainerBackend,
    ContainerSpec,
    PodmanBackend,
    prepare_checkout,
)

if TYPE_CHECKING:
    from .corpus import Case


class VerificationConfigError(ValueError):
    """The case's verification metadata is absent or malformed."""


@dataclass(frozen=True)
class Command:
    argv: list[str]
    timeout_s: float = 600.0
    env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def parse(cls, value: Any, *, field_name: str) -> Command:
        if isinstance(value, list):
            value = {"argv": value}
        if not isinstance(value, dict):
            raise VerificationConfigError(
                f"{field_name} must be a mapping or argv list"
            )
        argv = value.get("argv")
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(part, str) and part for part in argv)
        ):
            raise VerificationConfigError(
                f"{field_name}.argv must be a non-empty string list"
            )
        timeout = value.get("timeout_s", 600.0)
        if not isinstance(timeout, int | float) or timeout <= 0:
            raise VerificationConfigError(f"{field_name}.timeout_s must be positive")
        env = value.get("env", {})
        if not isinstance(env, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in env.items()
        ):
            raise VerificationConfigError(
                f"{field_name}.env must map strings to strings"
            )
        return cls(list(argv), float(timeout), dict(env))


@dataclass(frozen=True)
class Witness:
    name: str
    command: Command
    vulnerable_exit_codes: frozenset[int] = frozenset({1})
    fixed_exit_codes: frozenset[int] = frozenset({0})


@dataclass(frozen=True)
class VerificationSpec:
    build: list[Command]
    witnesses: list[Witness]
    controls: list[Command]
    invariant: str
    image: str = IMAGE_TAG
    network: bool = False

    @classmethod
    def parse(cls, value: Any) -> VerificationSpec:
        if not isinstance(value, dict):
            raise VerificationConfigError("case has no verification mapping")
        invariant = value.get("invariant")
        if not isinstance(invariant, str) or not invariant.strip():
            raise VerificationConfigError("verification.invariant is required")

        def commands(name: str) -> list[Command]:
            raw = value.get(name, [])
            if not isinstance(raw, list):
                raise VerificationConfigError(f"verification.{name} must be a list")
            return [
                Command.parse(item, field_name=f"verification.{name}[{i}]")
                for i, item in enumerate(raw)
            ]

        raw_witnesses = value.get("witnesses")
        if not isinstance(raw_witnesses, list) or not raw_witnesses:
            raise VerificationConfigError("verification.witnesses must be non-empty")
        witnesses: list[Witness] = []
        for i, raw in enumerate(raw_witnesses):
            if not isinstance(raw, dict):
                raise VerificationConfigError(
                    f"verification.witnesses[{i}] must be a mapping"
                )
            name = raw.get("name", f"witness-{i + 1}")
            if not isinstance(name, str) or not name:
                raise VerificationConfigError(
                    f"verification.witnesses[{i}].name is invalid"
                )
            command = Command.parse(
                raw.get("command"), field_name=f"verification.witnesses[{i}].command"
            )
            vulnerable = _exit_codes(
                raw.get("vulnerable_exit_codes", [1]),
                f"verification.witnesses[{i}].vulnerable_exit_codes",
            )
            fixed = _exit_codes(
                raw.get("fixed_exit_codes", [0]),
                f"verification.witnesses[{i}].fixed_exit_codes",
            )
            witnesses.append(Witness(name, command, vulnerable, fixed))
        image = value.get("image", IMAGE_TAG)
        if not isinstance(image, str) or not image:
            raise VerificationConfigError(
                "verification.image must be a non-empty string"
            )
        network = value.get("network", False)
        if not isinstance(network, bool):
            raise VerificationConfigError("verification.network must be boolean")
        return cls(
            commands("build"),
            witnesses,
            commands("controls"),
            invariant.strip(),
            image,
            network,
        )


def _exit_codes(value: Any, field_name: str) -> frozenset[int]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(v, int) for v in value)
    ):
        raise VerificationConfigError(f"{field_name} must be a non-empty integer list")
    return frozenset(value)


@dataclass
class CommandResult:
    argv: list[str]
    exit_code: int | None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None


@runtime_checkable
class CommandRunner(Protocol):
    def run(self, command: Command, cwd: Path) -> CommandResult: ...


class SubprocessCommandRunner:
    def run(self, command: Command, cwd: Path) -> CommandResult:
        env = {**os.environ, **command.env}
        try:
            proc = subprocess.run(
                command.argv,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=command.timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                command.argv,
                None,
                error=f"timeout after {command.timeout_s:g}s",
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
            )
        except OSError as exc:
            return CommandResult(command.argv, None, error=str(exc))
        return CommandResult(command.argv, proc.returncode, proc.stdout, proc.stderr)


class PodmanCommandRunner:
    """Run harness commands in the existing rootless, resource-capped sandbox."""

    def __init__(
        self,
        backend: ContainerBackend | None = None,
        *,
        image: str = IMAGE_TAG,
        network: bool = False,
        harness_dir: str | Path | None = None,
    ):
        self.backend = backend or PodmanBackend()
        self.image = image
        self.network = network
        self.harness_dir = Path(harness_dir).resolve() if harness_dir else None
        self._ready = False

    def run(self, command: Command, cwd: Path) -> CommandResult:
        if not self._ready:
            self.backend.ensure_image()
            self._ready = True
        mounts = [(str(cwd.resolve()), "/workspace", "rw,U")]
        if self.harness_dir is not None:
            mounts.append((str(self.harness_dir), "/harness", "ro"))
        spec = ContainerSpec(
            image=self.image,
            argv=command.argv,
            env=command.env,
            mounts=mounts,
            workdir="/workspace",
            network=self.network,
            name=f"nelson-verify-{os.getpid()}",
        )
        try:
            result = self.backend.run(spec, command.timeout_s)
        except Exception as exc:
            return CommandResult(command.argv, None, error=str(exc))
        if result.timed_out:
            return CommandResult(
                command.argv,
                None,
                result.stdout,
                result.stderr,
                f"timeout after {command.timeout_s:g}s",
            )
        return CommandResult(
            command.argv, result.returncode, result.stdout, result.stderr
        )


@dataclass
class CheckResult:
    revision: str
    kind: str
    name: str
    expected_exit_codes: frozenset[int]
    command_result: CommandResult

    @property
    def passed(self) -> bool:
        return (
            self.command_result.error is None
            and self.command_result.exit_code in self.expected_exit_codes
        )


@dataclass
class VerificationResult:
    case_id: str
    checks: list[CheckResult] = field(default_factory=list)
    error: str | None = None

    @property
    def verified(self) -> bool:
        return (
            self.error is None
            and bool(self.checks)
            and all(c.passed for c in self.checks)
        )


class DifferentialVerifier:
    def __init__(self, runner: CommandRunner | None = None):
        self.runner = runner

    def verify(
        self,
        case: Case,
        work_dir: str | Path,
        harness_dir: str | Path | None = None,
    ) -> VerificationResult:
        result = VerificationResult(case.ext_id)
        try:
            spec = VerificationSpec.parse(case.verification)
        except VerificationConfigError as exc:
            result.error = str(exc)
            return result
        if not case.repo_url or not case.vuln_commit or not case.fix_commit:
            result.error = "case requires repo_url, vuln_commit, and fix_commit"
            return result
        runner = self.runner or PodmanCommandRunner(
            image=spec.image, network=spec.network, harness_dir=harness_dir
        )

        root = Path(work_dir) / case.ext_id
        try:
            revisions = {
                "vulnerable": prepare_checkout(
                    case.repo_url, case.vuln_commit, root / "vulnerable"
                ),
                "fixed": prepare_checkout(
                    case.repo_url, case.fix_commit, root / "fixed"
                ),
            }
        except Exception as exc:  # checkout is infrastructure, not a failed proof
            result.error = f"checkout failed: {exc}"
            return result

        for revision, tree in revisions.items():
            for i, command in enumerate(spec.build):
                check = self._check(
                    revision,
                    "build",
                    f"build-{i + 1}",
                    command,
                    frozenset({0}),
                    tree,
                    runner,
                )
                result.checks.append(check)
                if not check.passed:
                    break
            else:
                expected_attr = f"{revision}_exit_codes"
                for witness in spec.witnesses:
                    result.checks.append(
                        self._check(
                            revision,
                            "witness",
                            witness.name,
                            witness.command,
                            getattr(witness, expected_attr),
                            tree,
                            runner,
                        )
                    )
                for i, command in enumerate(spec.controls):
                    result.checks.append(
                        self._check(
                            revision,
                            "control",
                            f"control-{i + 1}",
                            command,
                            frozenset({0}),
                            tree,
                            runner,
                        )
                    )
        return result

    def _check(
        self,
        revision: str,
        kind: str,
        name: str,
        command: Command,
        expected: frozenset[int],
        cwd: Path,
        runner: CommandRunner,
    ) -> CheckResult:
        return CheckResult(revision, kind, name, expected, runner.run(command, cwd))
