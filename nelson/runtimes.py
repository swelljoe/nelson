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
- ``agy`` — the host ``agy`` (Antigravity / Gemini) agent CLI bind-mounted in; a
  Claude-Code-like interface (it replaced the old ``gemini`` CLI), plain-text out.
- ``codex`` — the host ``codex`` (OpenAI Codex CLI) static binary bind-mounted in;
  drives a GPT model over the ChatGPT-subscription rolling limits via the mounted
  ``~/.codex`` sign-in (no API key / per-token spend). The OpenAI counterpart to
  ``claude-code``; ``codex exec`` non-interactive, plain-text out.
- ``kimi-cli`` / ``pi-custom`` — generic native-CLI runtimes for vendors that ship
  an *official* agent, wired and unit-tested but stubbed: an absent host binary
  resolves to ``infra_error`` until the CLI is installed and its argv/output
  verified. (No ``deepseek-cli`` — DeepSeek has no trusted first-party agent CLI,
  so it runs through ``claude-code`` (Anthropic-compat) and ``raw-api-loop``.)

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


class AgyCredentialMountAuth:
    """Mount the host ``agy`` (Antigravity / Gemini CLI) sign-in into the container.

    Copies the host's ``~/.gemini`` tree (the ``antigravity-cli`` OAuth token +
    settings, and ``config``) into a per-run staging dir mounted ``rw,U`` at
    ``/home/agent/.gemini`` so the container inherits the signed-in session without
    touching the host originals (the agent may refresh the token). Missing creds ->
    :class:`RunnerError` -> ``auth_failed``. Verified live (2026-05-28): the token
    lives at ``~/.gemini/antigravity-cli/antigravity-oauth-token``; the dir is
    small, so the whole tree is copied.
    """

    _TOKEN_REL = Path("antigravity-cli") / "antigravity-oauth-token"

    def __init__(self, creds_dir: Path | None = None):
        self.creds_dir = Path(creds_dir) if creds_dir else (Path.home() / ".gemini")

    def prepare(self, staging: Path) -> AuthMaterial:
        if not (self.creds_dir / self._TOKEN_REL).is_file():
            raise RunnerError(
                f"no agy credentials under {self.creds_dir} "
                f"(missing {self._TOKEN_REL}); sign in with `agy` on the host first"
            )
        dest = staging / "gemini"
        shutil.copytree(self.creds_dir, dest)
        return AuthMaterial(
            env={},
            mounts=[(str(dest), "/home/agent/.gemini", "rw,U")],
        )


class CodexCredentialMountAuth:
    """Mount the host ``codex`` (OpenAI Codex CLI) sign-in into the container.

    Copies just ``~/.codex/auth.json`` (the ChatGPT-subscription OAuth ``tokens``
    — *not* an API key, so runs draw on the plan's rolling limits) into a per-run
    staging dir mounted ``rw,U`` at ``/home/agent/.codex`` with ``CODEX_HOME``
    pointed at it. Writable so codex can refresh the token mid-run without touching
    the host original; the refreshed copy is discarded when the container exits.
    Only ``auth.json`` (+ ``version.json`` if present, to skip the update nag) is
    copied — the rest of ``~/.codex`` is multi-MB sqlite logs/sessions the run does
    not need, and ``--ignore-user-config`` on the exec side means config.toml is
    not required either. Missing creds -> :class:`RunnerError` -> ``auth_failed``.
    """

    def __init__(self, creds_dir: Path | None = None):
        self.creds_dir = Path(creds_dir) if creds_dir else (Path.home() / ".codex")

    def prepare(self, staging: Path) -> AuthMaterial:
        auth = self.creds_dir / "auth.json"
        if not auth.is_file():
            raise RunnerError(
                f"no codex credentials at {auth}; sign in with `codex login` on "
                "the host first"
            )
        dest = staging / "codex"
        dest.mkdir(parents=True, exist_ok=True, mode=0o700)
        dest.chmod(0o700)
        shutil.copyfile(auth, dest / "auth.json")
        (dest / "auth.json").chmod(0o600)
        version = self.creds_dir / "version.json"
        if version.is_file():
            shutil.copyfile(version, dest / "version.json")
        return AuthMaterial(
            env={"CODEX_HOME": "/home/agent/.codex"},
            mounts=[(str(dest), "/home/agent/.codex", "rw,U")],
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


def _effective_cost(
    competitor: Competitor, tin: int | None, tout: int | None, parsed_cost: float | None
) -> float | None:
    """Cost from declared per-MTok pricing when present, else the runtime's own.

    When a competitor's ``cost_model`` carries ``input_usd_per_mtok`` /
    ``output_usd_per_mtok``, compute cost from the observed token counts. This is
    the source of truth for non-Anthropic models driven through Claude Code over
    an Anthropic-compatible endpoint: ``claude``'s reported ``total_cost_usd``
    applies *Anthropic* pricing to the mapped model name, which is meaningless for
    e.g. DeepSeek/MiMo. Native claude competitors declare no pricing, so their
    real reported cost is left untouched. (Input price is applied to all input
    tokens including cache reads — a slight over-estimate, the best we can do
    without per-tier cache accounting.)
    """
    cfg = parse_cost_model(competitor.cost_model)
    in_price = cfg.get("input_usd_per_mtok")
    out_price = cfg.get("output_usd_per_mtok")
    if in_price is None and out_price is None:
        return parsed_cost
    from .raw_api_loop import compute_cost

    return compute_cost(tin or 0, tout or 0, in_price, out_price)


def _claude_shaped_output(
    exec_result: ContainerExecResult, competitor: Competitor
) -> ParsedOutput:
    """Parse a claude-shaped stdout (a ``type==result`` object) into ParsedOutput.

    Shared by claude-code, raw-api-loop, and the stubbed native CLIs, all of which
    emit the same final result object.
    """
    text, tin, tout, cost = extract_result(exec_result.stdout)
    cost = _effective_cost(competitor, tin, tout, cost)
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
        cfg = parse_cost_model(ctx.competitor.cost_model)
        # Resolve the host claude binary lazily here (not at BenchRunner
        # construction), so a host without claude can still run other runtimes; a
        # missing claude then becomes infra_error on a claude run. A competitor may
        # pin its own ``claude_bin`` (an absolute host path) in cost_model — used to
        # hold a third-party Anthropic-compat competitor on an older client version
        # whose message shape that provider's proxy still accepts, without touching
        # the host's default claude (judges, interactive CLI). The read-only mount
        # means the pinned binary can't self-update inside the container.
        pinned = cfg.get("claude_bin")
        if pinned:
            if not isinstance(pinned, str) or not Path(pinned).is_absolute():
                raise RunnerError(
                    "cost_model.claude_bin must be a non-empty absolute host path"
                )
            claude_bin = Path(pinned)
        else:
            claude_bin = ctx.claude_bin or _resolve_claude_bin()
        # Optional Anthropic-compatible passthrough (e.g. DeepSeek via its official
        # Claude Code endpoint). base_url + the model-mapping env come from the
        # competitor's cost_model JSON; the token arrives via the auth profile.
        # VERIFY-AT-WIRING: provider base_url (e.g. https://api.deepseek.com/anthropic)
        # and the model-mapping env vars.
        base_url = cfg.get("anthropic_base_url")
        # claude's --max-budget-usd enforces ITS OWN cost accounting, which prices a
        # third-party model as the mapped *Anthropic* model (opus/sonnet rates) — wildly
        # inflated, so a real compat run trips ``error_max_budget_usd`` mid-audit and is
        # lost as infra_error. Disable the per-run cap for compat runs; real cost comes
        # from declared pricing, and ``--timeout`` + global ``--max-spend-usd`` still
        # bound spend. Native claude (no base_url) keeps the backstop.
        max_budget = None if base_url else ctx.max_budget_usd
        spec = claude_code_spec(
            ctx.competitor,
            ctx.prompt,
            ctx.src_dir,
            claude_bin,
            ctx.auth,
            name=ctx.name,
            network=ctx.network,
            max_budget_usd=max_budget,
        )
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
            "NELSON_MAX_STEPS": str(cfg.get("max_steps", 40)),
            "NELSON_TOKEN_BUDGET": str(cfg.get("token_budget", 500000)),
            # Drives which tools the in-container agent advertises (e.g. a
            # "read-grep-semgrep" profile adds the static-analysis tool); the base
            # "read-grep" profile is unchanged, so existing competitors are unaffected.
            "NELSON_TOOL_PROFILE": ctx.competitor.tool_profile,
            **ctx.auth.env,
        }
        if cfg.get("input_usd_per_mtok") is not None:
            env["NELSON_INPUT_USD_PER_MTOK"] = str(cfg["input_usd_per_mtok"])
        if cfg.get("output_usd_per_mtok") is not None:
            env["NELSON_OUTPUT_USD_PER_MTOK"] = str(cfg["output_usd_per_mtok"])
        if cfg.get("http_timeout") is not None:
            # Slow self-hosted endpoints raise the per-API-call read timeout so a single
            # big-context turn isn't cut off as an infra_error (see _http_timeout).
            env["NELSON_HTTP_TIMEOUT"] = str(cfg["http_timeout"])
        if cfg.get("temperature") is not None:
            # Sampling temperature for the audit (default 0.1). Raised for repeat-trial
            # experiments where run-to-run diversity is the point, not determinism.
            env["NELSON_TEMPERATURE"] = str(cfg["temperature"])
        # Optional anti-repetition / nucleus knobs, injected into the API request only
        # when set. Used to tame the repetition-spiral timeout some local models (e.g.
        # Gemma 4) fall into. Absent for every competitor that doesn't set them, so
        # strict OpenAI-compat endpoints never receive an unknown field.
        for field, env_key in (
            ("top_p", "NELSON_TOP_P"),
            ("min_p", "NELSON_MIN_P"),
            ("frequency_penalty", "NELSON_FREQUENCY_PENALTY"),
            ("presence_penalty", "NELSON_PRESENCE_PENALTY"),
            ("repeat_penalty", "NELSON_REPEAT_PENALTY"),
        ):
            if cfg.get(field) is not None:
                env[env_key] = str(cfg[field])
        # Arbitrary structured request fields merged verbatim into the API payload
        # (e.g. {"chat_template_kwargs": {"enable_thinking": false}} to stop Gemma 4's
        # reasoning phase running away to a timeout on large files). JSON-encoded.
        if isinstance(cfg.get("extra_body"), dict):
            env["NELSON_EXTRA_BODY"] = json.dumps(cfg["extra_body"])
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


class AgyRuntime:
    """The host ``agy`` (Antigravity / Gemini) agent CLI bind-mounted into the run.

    ``agy`` replaced the old ``gemini`` CLI and has a Claude-Code-like interface
    (verified live 2026-05-28, v1.0.3): ``-p`` runs one prompt non-interactively,
    ``--dangerously-skip-permissions`` auto-approves tool use, ``--add-dir`` grants
    workspace access, and it prints the final response as plain text (no JSON
    envelope, no ``-m`` flag — the model is the signed-in account's default). Auth
    is the mounted ``~/.gemini`` sign-in. The bind-mounted dynamically-linked binary
    runs against fedora-minimal's loader (confirmed).
    """

    name = "agy"

    def default_auth(self) -> ContainerAuth:
        return AgyCredentialMountAuth()

    def build_spec(self, ctx: RuntimeContext) -> ContainerSpec:
        agy_bin = shutil.which("agy")
        if not agy_bin:
            raise RunnerError(
                "agy CLI not found on PATH; install it and sign in to run agy "
                "competitors"
            )
        argv = [
            "agy",
            "-p",
            ctx.prompt,
            "--dangerously-skip-permissions",
            "--add-dir",
            "/src",
        ]
        mounts = [
            (str(Path(agy_bin).resolve()), "/usr/local/bin/agy", "ro"),
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
            stdin=None,  # agy takes the prompt as the -p argument
        )

    def parse_output(
        self, exec_result: ContainerExecResult, competitor: Competitor
    ) -> ParsedOutput:
        # agy prints the final response as plain text; the findings JSON array is
        # extracted from it (no token/cost usage is reported by `agy -p`).
        text = exec_result.stdout
        return ParsedOutput(text, None, None, None, parse_competitor_findings(text))


class CodexRuntime:
    """The host ``codex`` (OpenAI Codex CLI) binary bind-mounted into the run.

    Codex ships as a self-contained static-PIE musl binary, so — like ``agy`` —
    it bind-mounts as a single file and runs against fedora-minimal's loader with
    no node/runtime deps. This is the OpenAI counterpart to :class:`ClaudeCodeRuntime`:
    it drives a GPT model (``-m``) through the ChatGPT-subscription rolling limits
    via the mounted ``~/.codex`` sign-in (:class:`CodexCredentialMountAuth`), so
    there is no per-token API spend and no ``auth_profile``/API key.

    ``codex exec`` is the non-interactive mode: ``-C /src`` roots the agent at the
    read-only source mount, ``-s read-only`` confines any model-run shell command
    to reads (matching the read/grep harness the rest of the field uses),
    ``approval_policy="never"`` keeps it from blocking on an approval prompt with
    no TTY, ``--skip-git-repo-check`` tolerates the git-stripped mount,
    ``--ephemeral`` avoids writing session files, and ``--ignore-user-config``
    skips the host config.toml (auth still resolves from ``CODEX_HOME``). The
    prompt is delivered on stdin. VERIFY-AT-WIRING: exec flag set + output shape
    were confirmed live against codex-cli 0.145.0.
    """

    name = "codex"

    def default_auth(self) -> ContainerAuth:
        return CodexCredentialMountAuth()

    def build_spec(self, ctx: RuntimeContext) -> ContainerSpec:
        codex_bin = shutil.which("codex")
        if not codex_bin:
            raise RunnerError(
                "codex CLI not found on PATH; install it and `codex login` on the "
                "host first to run codex competitors"
            )
        argv = [
            "codex",
            "exec",
            "-m",
            ctx.competitor.model,
            "-C",
            "/src",
            "-s",
            "read-only",
            "-c",
            'approval_policy="never"',
            "--skip-git-repo-check",
            "--ephemeral",
            "--ignore-user-config",
            "--color",
            "never",
        ]
        mounts = [
            (str(Path(codex_bin).resolve()), "/usr/local/bin/codex", "ro"),
            (str(ctx.src_dir), "/src", "ro"),
            *ctx.auth.mounts,
        ]
        return ContainerSpec(
            image=IMAGE_TAG,
            argv=argv,
            env={"HOME": "/home/agent", **ctx.auth.env},
            mounts=mounts,
            name=ctx.name,
            # Network MUST stay on: codex calls the OpenAI backend.
            network=ctx.network,
            stdin=ctx.prompt,
        )

    def parse_output(
        self, exec_result: ContainerExecResult, competitor: Competitor
    ) -> ParsedOutput:
        # `codex exec` prints the agent's turns then its final message as plain
        # text (no token/cost envelope on the subscription path); the findings JSON
        # array is scanned out of it exactly as for agy.
        text = exec_result.stdout
        return ParsedOutput(text, None, None, None, parse_competitor_findings(text))


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
    register_runtime(AgyRuntime())
    register_runtime(CodexRuntime())
    # No `deepseek-cli`: DeepSeek ships no first-party agent CLI (the ones that
    # exist are untrusted third parties), so DeepSeek runs through the two trusted
    # shared harnesses instead — claude-code over its Anthropic-compatible endpoint
    # and raw-api-loop over its OpenAI-compatible one. The generic native-CLI
    # runtime remains available for vendors that do ship an official agent.
    register_runtime(BindMountedCliRuntime("kimi-cli", "kimi"))
    register_runtime(BindMountedCliRuntime("pi-custom", "pi"))


_register_defaults()
