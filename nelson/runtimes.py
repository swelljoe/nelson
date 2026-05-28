"""Container runtimes: how a competitor's (model x runtime) is actually invoked.

The benchmark runs every competitor inside the same isolated, read-only-``/src``
container; what differs per *runtime* is which binary or script executes, the
argv, the credentials injected, and how the container's stdout is parsed back
into findings + usage. This module is that dispatch layer: a
:class:`ContainerRuntime` registry keyed by the ``competitors.runtime`` value, so
:meth:`BenchRunner.run_case` stays runtime-agnostic.

Runtimes registered here:

- ``claude-code`` — the host ``claude`` binary bind-mounted in (the P2 default),
  refactored from the previously-hardcoded path with no behavior change.
- ``raw-api-loop`` — a stdlib ReAct agent (``raw_api_loop.py``) bind-mounted in
  and run under the container's ``python3``, driving any OpenAI-compatible
  endpoint. This is the shared, apples-to-apples *model* harness (DeepSeek / MiMo
  / Kimi become provider configs in the competitor's ``cost_model`` + auth
  profile).
- ``gemini-cli`` — the host ``gemini`` binary bind-mounted in (native agent).
- ``deepseek-cli`` / ``kimi-cli`` / ``pi-custom`` — generic native-CLI runtimes,
  wired and unit-tested but stubbed: an absent host binary resolves to
  ``infra_error`` until the CLI is installed and its argv/output verified.

Integrity rule (non-negotiable): a missing binary, missing API key, or unknown
runtime is a *setup* failure (``infra_error`` / ``auth_failed``) — never a model
that "looked and found nothing". Every failure path here raises a clean
:class:`RunnerError` that the runner maps to a non-miss status.

Several vendor specifics are deliberately NOT guessed and marked
``VERIFY-AT-WIRING`` (gemini auto-approve flag + creds path; provider base URLs,
model ids, key env-var names; native CLI argv/output shapes).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .agents import parse_gemini_envelope
from .auth import (
    DEFAULT_SECRET_STORE,
    STANDARD_AUTH_PROFILES,
    AuthProfile,
    MissingSecretError,
    SecretStore,
)
from .runner import (
    IMAGE_TAG,
    AuthMaterial,
    Competitor,
    ContainerAuth,
    ContainerExecResult,
    ContainerSpec,
    CredentialMountAuth,
    RunnerError,
    _resolve_claude_bin,
    claude_code_spec,
    extract_result,
    parse_competitor_findings,
)

# Runtimes that talk to an OpenAI-compatible API with an env-var key, eligible for
# the cheap host-side preflight probe (and never a credential-mount default).
API_KEY_RUNTIMES = frozenset({"raw-api-loop", "kimi-cli", "pi-custom"})

# Fallback endpoint for raw-api-loop when the competitor's cost_model carries no
# base_url. VERIFY-AT-WIRING: real provider base URLs live in the competitor
# config, not here.
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"

# Where the in-container ReAct script lives on the host (bind-mounted in) and the
# path it is mounted at inside the container.
_RAW_API_SCRIPT_HOST = Path(__file__).with_name("raw_api_loop.py")
_RAW_API_SCRIPT_CONTAINER = "/opt/nelson/raw_api_loop.py"


# -- Value types -------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeContext:
    """Everything a runtime needs to assemble its ContainerSpec for one run."""

    competitor: Competitor
    prompt: str
    src_dir: Path
    auth: AuthMaterial
    name: str
    claude_bin: Path | None = None
    network: bool = True
    max_budget_usd: float | None = None


@dataclass
class ParsedOutput:
    """Normalized result of a runtime's stdout, ready for ``complete_run``."""

    final_text: str
    tokens_in: int | None
    tokens_out: int | None
    cost: float | None
    findings: list[dict[str, Any]]


@runtime_checkable
class ContainerRuntime(Protocol):
    name: str

    def default_auth(self) -> ContainerAuth: ...

    def build_spec(self, ctx: RuntimeContext) -> ContainerSpec: ...

    def parse_output(
        self, exec_result: ContainerExecResult, competitor: Competitor
    ) -> ParsedOutput: ...


# -- Auth strategies (host -> container credential bridging) -----------------


class _FailingAuth:
    """A ContainerAuth that always fails with a clear reason (-> auth_failed).

    Used when a runtime needs credentials it was not given (e.g. ``raw-api-loop``
    with no ``auth_profile``) or an unknown ``auth_profile`` is named — surfacing
    the problem as ``auth_failed`` at ``prepare()`` time, never as a silent
    unauthenticated run.
    """

    def __init__(self, reason: str):
        self.reason = reason

    def prepare(self, staging: Path) -> AuthMaterial:
        raise RunnerError(self.reason)


class EnvKeyAuth:
    """Inject an API key (and any profile env) as container ``-e`` env vars.

    Resolves an :class:`AuthProfile`'s secret *names* to values from the secret
    store and returns them as env (no mounted file). A missing secret becomes a
    :class:`RunnerError` -> ``auth_failed`` — the integrity hinge for API-key
    runtimes (a dead key is never a model that found nothing).
    """

    def __init__(self, profile: AuthProfile, store: SecretStore | None = None):
        self.profile = profile
        self.store = store or DEFAULT_SECRET_STORE

    def prepare(self, staging: Path) -> AuthMaterial:
        try:
            env = self.profile.resolve_env(self.store)
        except MissingSecretError as e:
            raise RunnerError(str(e)) from e
        return AuthMaterial(env=env, mounts=[])


class GeminiCredentialMountAuth:
    """Mount the host gemini sign-in into the container, like CredentialMountAuth.

    Copies the host's gemini credential file(s) into a per-run staging dir mounted
    ``rw,U`` so the container can refresh tokens without touching the host
    originals. Missing creds -> :class:`RunnerError` -> ``auth_failed``.

    VERIFY-AT-WIRING: the host creds location (``~/.gemini/oauth_creds.json``), the
    in-container destination, and whether gemini reads an env var instead of/in
    addition to the dir are unconfirmed — checked when gemini is first run live.
    """

    # Files we attempt to stage (kept narrow so we never copy unrelated, possibly
    # large, sibling dirs under ~/.gemini).
    _CRED_FILES = ("oauth_creds.json", "settings.json")

    def __init__(self, creds_dir: Path | None = None):
        self.creds_dir = Path(creds_dir) if creds_dir else (Path.home() / ".gemini")

    def prepare(self, staging: Path) -> AuthMaterial:
        present = [f for f in self._CRED_FILES if (self.creds_dir / f).is_file()]
        if not present:
            raise RunnerError(
                f"no gemini credentials under {self.creds_dir}; "
                "sign in with `gemini` on the host first"
            )
        dest = staging / "gemini"
        dest.mkdir(parents=True, exist_ok=True)
        for f in present:
            shutil.copyfile(self.creds_dir / f, dest / f)
            (dest / f).chmod(0o600)
        return AuthMaterial(
            env={},
            mounts=[(str(dest), "/home/agent/.gemini", "rw,U")],
        )


def auth_for_competitor(
    competitor: Competitor, runtime: ContainerRuntime, store: SecretStore | None = None
) -> ContainerAuth:
    """Pick the ContainerAuth for a competitor.

    A named ``auth_profile`` means API-key env injection (:class:`EnvKeyAuth`); an
    unknown profile name fails clearly at prepare-time (``auth_failed``). With no
    profile, the runtime's own default applies (claude/gemini credential mount,
    or a clear failure for runtimes that require a key).
    """
    if competitor.auth_profile:
        profile = STANDARD_AUTH_PROFILES.get(competitor.auth_profile)
        if profile is None:
            return _FailingAuth(
                f"unknown auth profile {competitor.auth_profile!r}; "
                f"known: {sorted(STANDARD_AUTH_PROFILES)}"
            )
        return EnvKeyAuth(profile, store)
    return runtime.default_auth()


# -- cost_model parsing ------------------------------------------------------


def parse_cost_model(cost_model: str | None) -> dict[str, Any]:
    """Best-effort parse of a competitor's ``cost_model`` field into a dict.

    ``cost_model`` is free-form: ``"subscription"``, ``None``, or a JSON object
    carrying ``base_url`` + per-token pricing for raw-api runtimes. Anything that
    is not a JSON object yields ``{}``, so the runtime falls back to defaults and
    emits null cost (already safe downstream — leaderboard treats null cost as 0).
    """
    if not cost_model:
        return {}
    try:
        data = json.loads(cost_model)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


# -- Runtimes ----------------------------------------------------------------


def _claude_shaped_output(
    exec_result: ContainerExecResult, competitor: Competitor
) -> ParsedOutput:
    """Parse a claude-shaped stdout (a ``type==result`` object) into ParsedOutput.

    Shared by claude-code, raw-api-loop, and the stubbed native CLIs, all of which
    emit the same final result object.
    """
    text, tin, tout, cost = extract_result(exec_result.stdout)
    return ParsedOutput(text, tin, tout, cost, parse_competitor_findings(text))


class ClaudeCodeRuntime:
    """The host ``claude`` binary bind-mounted into the container (P2 default).

    Also serves as a shared harness for *non-Anthropic* models via their
    Anthropic-compatible endpoint: when the competitor's ``cost_model`` JSON
    carries an ``anthropic_base_url`` (and an ``env`` block of provider model
    mappings), the same Claude Code harness is pointed at that endpoint with the
    provider key supplied by the auth profile (``EnvKeyAuth`` ->
    ``ANTHROPIC_AUTH_TOKEN``). This is the trusted way to run e.g. DeepSeek/Kimi
    agentically — their official Claude Code integration — rather than an
    unofficial third-party CLI. Absent that config, the runtime is native
    subscription claude exactly as before.
    """

    name = "claude-code"

    def default_auth(self) -> ContainerAuth:
        return CredentialMountAuth()

    def build_spec(self, ctx: RuntimeContext) -> ContainerSpec:
        # Resolve the host claude binary lazily here (not at BenchRunner
        # construction), so a host without claude can still run other runtimes; a
        # missing claude then becomes infra_error on a claude run.
        claude_bin = ctx.claude_bin or _resolve_claude_bin()
        spec = claude_code_spec(
            ctx.competitor,
            ctx.prompt,
            ctx.src_dir,
            claude_bin,
            ctx.auth,
            name=ctx.name,
            network=ctx.network,
            max_budget_usd=ctx.max_budget_usd,
        )
        # Optional Anthropic-compatible passthrough (e.g. DeepSeek via its official
        # Claude Code endpoint). base_url + the model-mapping env come from the
        # competitor's cost_model JSON; the token arrives via the auth profile.
        # VERIFY-AT-WIRING: provider base_url (e.g. https://api.deepseek.com/anthropic),
        # the model-mapping env vars, and that --max-budget-usd is harmless against a
        # third-party endpoint (set max_budget_usd=None for compat runs if not).
        cfg = parse_cost_model(ctx.competitor.cost_model)
        base_url = cfg.get("anthropic_base_url")
        if base_url:
            spec.env["ANTHROPIC_BASE_URL"] = str(base_url)
            extra = cfg.get("env")
            if isinstance(extra, dict):
                for k, v in extra.items():
                    spec.env[str(k)] = str(v)
        return spec

    def parse_output(
        self, exec_result: ContainerExecResult, competitor: Competitor
    ) -> ParsedOutput:
        return _claude_shaped_output(exec_result, competitor)


class RawApiLoopRuntime:
    """Shared agentic harness: a stdlib ReAct agent over an OpenAI-compatible API.

    The script runs inside the container (same isolation + read-only ``/src`` as
    claude-code) and emits a claude-shaped result object, so parsing is uniform.
    base_url + per-token pricing come from the competitor's ``cost_model`` JSON;
    the key is injected by :class:`EnvKeyAuth` as ``NELSON_API_KEY``.
    """

    name = "raw-api-loop"

    def default_auth(self) -> ContainerAuth:
        return _FailingAuth(
            "raw-api-loop requires an auth_profile naming the API key secret "
            "(e.g. deepseek-api-key); none was set on the competitor."
        )

    def build_spec(self, ctx: RuntimeContext) -> ContainerSpec:
        cfg = parse_cost_model(ctx.competitor.cost_model)
        base_url = str(cfg.get("base_url") or DEFAULT_OPENAI_BASE_URL)
        env = {
            "HOME": "/home/agent",
            "NELSON_BASE_URL": base_url,
            "NELSON_MODEL": ctx.competitor.model,
            "NELSON_MAX_STEPS": str(cfg.get("max_steps", 12)),
            "NELSON_TOKEN_BUDGET": str(cfg.get("token_budget", 200000)),
            **ctx.auth.env,
        }
        if cfg.get("input_usd_per_mtok") is not None:
            env["NELSON_INPUT_USD_PER_MTOK"] = str(cfg["input_usd_per_mtok"])
        if cfg.get("output_usd_per_mtok") is not None:
            env["NELSON_OUTPUT_USD_PER_MTOK"] = str(cfg["output_usd_per_mtok"])
        mounts = [
            (str(_RAW_API_SCRIPT_HOST), _RAW_API_SCRIPT_CONTAINER, "ro"),
            (str(ctx.src_dir), "/src", "ro"),
            *ctx.auth.mounts,
        ]
        return ContainerSpec(
            image=IMAGE_TAG,
            argv=["python3", _RAW_API_SCRIPT_CONTAINER],
            env=env,
            mounts=mounts,
            name=ctx.name,
            # Network MUST stay on: the loop calls the provider API.
            network=ctx.network,
            stdin=ctx.prompt,
        )

    def parse_output(
        self, exec_result: ContainerExecResult, competitor: Competitor
    ) -> ParsedOutput:
        return _claude_shaped_output(exec_result, competitor)


class GeminiCliRuntime:
    """The host ``gemini`` binary bind-mounted in (native agent, subscription auth)."""

    name = "gemini-cli"

    def default_auth(self) -> ContainerAuth:
        return GeminiCredentialMountAuth()

    def build_spec(self, ctx: RuntimeContext) -> ContainerSpec:
        gemini_bin = shutil.which("gemini")
        if not gemini_bin:
            raise RunnerError(
                "gemini CLI not found on PATH; install it and sign in to run "
                "gemini-cli competitors"
            )
        # VERIFY-AT-WIRING: the non-interactive auto-approve flag for tool use
        # (e.g. --yolo / --approval-mode=yolo) and whether it alters the JSON
        # shape; and how gemini is pointed at /src (workdir vs an include flag).
        argv = ["gemini", "-p", ctx.prompt, "--output-format", "json"]
        if ctx.competitor.model and ctx.competitor.model != "default":
            argv += ["-m", ctx.competitor.model]
        mounts = [
            (str(Path(gemini_bin).resolve()), "/usr/local/bin/gemini", "ro"),
            (str(ctx.src_dir), "/src", "ro"),
            *ctx.auth.mounts,
        ]
        return ContainerSpec(
            image=IMAGE_TAG,
            argv=argv,
            env={"HOME": "/home/agent", **ctx.auth.env},
            mounts=mounts,
            name=ctx.name,
            network=ctx.network,
            stdin=None,  # gemini takes the prompt as the -p argument
        )

    def parse_output(
        self, exec_result: ContainerExecResult, competitor: Competitor
    ) -> ParsedOutput:
        text, tin, tout = parse_gemini_envelope(exec_result.stdout)
        return ParsedOutput(text, tin, tout, None, parse_competitor_findings(text))


class BindMountedCliRuntime:
    """Generic native-CLI runtime: a host agent binary bind-mounted into the run.

    Registered only for vendors that ship an *official* agent CLI we'd trust but
    cannot verify on this host yet (kimi-cli / pi-custom). An absent binary
    resolves to ``infra_error`` (the wired-but-stubbed state), so these compete
    cleanly the moment the CLI is installed. argv template, stdin-vs-arg prompt
    delivery, output JSON shape, and credential needs are VERIFY-AT-WIRING per
    vendor; the default parser assumes a claude-shaped result object.
    """

    def __init__(self, name: str, binary: str):
        self.name = name
        self.binary = binary

    def default_auth(self) -> ContainerAuth:
        return _FailingAuth(
            f"{self.name} requires an auth_profile naming its API key secret; "
            "none was set on the competitor."
        )

    def build_spec(self, ctx: RuntimeContext) -> ContainerSpec:
        binpath = shutil.which(self.binary)
        if not binpath:
            raise RunnerError(
                f"{self.binary!r} not found on PATH; runtime {self.name!r} is "
                "wired but not installed/verified on this host (VERIFY-AT-WIRING)"
            )
        # VERIFY-AT-WIRING: real argv, stdin-vs-arg prompt, output JSON shape.
        argv = [self.binary, "-p", ctx.prompt]
        mounts = [
            (str(Path(binpath).resolve()), f"/usr/local/bin/{self.binary}", "ro"),
            (str(ctx.src_dir), "/src", "ro"),
            *ctx.auth.mounts,
        ]
        return ContainerSpec(
            image=IMAGE_TAG,
            argv=argv,
            env={"HOME": "/home/agent", **ctx.auth.env},
            mounts=mounts,
            name=ctx.name,
            network=ctx.network,
            stdin=None,
        )

    def parse_output(
        self, exec_result: ContainerExecResult, competitor: Competitor
    ) -> ParsedOutput:
        return _claude_shaped_output(exec_result, competitor)


# -- Registry ----------------------------------------------------------------

_RUNTIMES: dict[str, ContainerRuntime] = {}


def register_runtime(runtime: ContainerRuntime) -> None:
    _RUNTIMES[runtime.name] = runtime


def get_runtime(name: str) -> ContainerRuntime:
    """Look up a registered runtime, or raise RunnerError for an unknown name.

    An unknown runtime is a configuration error; the runner maps the raised
    RunnerError to ``infra_error`` (never a miss).
    """
    try:
        return _RUNTIMES[name]
    except KeyError:
        raise RunnerError(
            f"unknown runtime {name!r}; known: {sorted(_RUNTIMES)}"
        ) from None


def _register_defaults() -> None:
    register_runtime(ClaudeCodeRuntime())
    register_runtime(RawApiLoopRuntime())
    register_runtime(GeminiCliRuntime())
    # No `deepseek-cli`: DeepSeek ships no first-party agent CLI (the ones that
    # exist are untrusted third parties), so DeepSeek runs through the two trusted
    # shared harnesses instead — claude-code over its Anthropic-compatible endpoint
    # and raw-api-loop over its OpenAI-compatible one. The generic native-CLI
    # runtime remains available for vendors that do ship an official agent.
    register_runtime(BindMountedCliRuntime("kimi-cli", "kimi"))
    register_runtime(BindMountedCliRuntime("pi-custom", "pi"))


_register_defaults()
