"""Agent adapters for invoking AI models via CLI or API.

The unit of evaluation in the benchmark is a *competitor* = (model x runtime x
tool-profile). An adapter here is a **runtime**: an invocation method (Claude
Code, Gemini CLI, a raw OpenAI-compatible API call) carrying a ``model_id`` and
a ``tool_profile``. The legacy single-shot behavior (full file pasted in, JSON
array out, no tool use) is the ``"single-shot"`` tool profile.

Integrity rule (non-negotiable): auth / rate-cap / infra failures are reported
via :class:`FailureKind` and must never be scored as "the model looked and found
nothing." See :func:`classify_failure`.
"""

import contextlib
import json
import logging
import os
import subprocess
import threading
import urllib.error
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import urlparse

import httpx

from . import raw_api_loop
from .auth import AuthProfile, MissingSecretError

log = logging.getLogger(__name__)


class FailureKind(StrEnum):
    """Why a run did not produce a scorable result.

    The string values match the ``runs``/``jobs`` status vocabulary so they can
    be stored directly. ``RATE_LIMIT`` is recoverable (retry after backoff);
    ``AUTH`` and ``INFRA`` are terminal for the job but, crucially, are *not*
    misses — the model never got a fair look at the code.
    """

    AUTH = "auth_failed"
    INFRA = "infra_error"
    RATE_LIMIT = "rate_limited"


# Substrings (lower-cased) that identify each failure class in CLI/API output.
# Order of checks matters: a 429 mentioning "key" is rate-limited, not auth.
_RATE_MARKERS = (
    "rate limit",
    "rate_limit",
    "ratelimit",
    "429",
    "overloaded",
    "quota",
    "too many requests",
    "resource_exhausted",
)
_AUTH_MARKERS = (
    "not logged in",
    "please run /login",
    "oauth token has expired",
    "token has expired",
    "invalid api key",
    "invalid x-api-key",
    "invalid_api_key",
    "authentication_error",
    "authentication failed",
    "unauthorized",
    "401",
    "403",
    "no api key",
    "missing api key",
    "api key not found",
)


def classify_failure(text: str, *, failed: bool = True) -> FailureKind | None:
    """Classify runtime output into a :class:`FailureKind`, or ``None``.

    ``text`` is the combined stdout/stderr (or response body). ``failed`` is
    whether the invocation reported failure (non-zero exit, non-200 status, a
    raised transport error). Rate-limit markers win over auth markers; an
    unexplained failure with no recognizable marker is ``INFRA`` (never a miss).
    A successful invocation with no markers returns ``None``.
    """
    blob = text.lower()
    if any(m in blob for m in _RATE_MARKERS):
        return FailureKind.RATE_LIMIT
    if any(m in blob for m in _AUTH_MARKERS):
        return FailureKind.AUTH
    if failed:
        return FailureKind.INFRA
    return None


@dataclass
class Finding:
    line_number: int | None
    code_snippet: str | None
    explanation: str | None
    confidence: str | None  # high, medium, low
    cwe_id: str | None = None  # Set by model in open scan mode


@dataclass
class AgentResult:
    findings: list[Finding]
    raw_output: str
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    rate_limited: bool = False
    error: str | None = None
    failure_kind: FailureKind | None = None

    def __post_init__(self):
        # Keep the legacy rate_limited bool and the richer failure_kind in sync
        # so existing callers (scanner, review) and new ones agree.
        if self.failure_kind is FailureKind.RATE_LIMIT:
            self.rate_limited = True
        elif self.failure_kind is not None:
            # Any non-rate-limit failure is not "rate limited".
            self.rate_limited = False
        elif self.rate_limited:
            self.failure_kind = FailureKind.RATE_LIMIT


@dataclass
class PreflightResult:
    """Outcome of a cheap per-competitor auth/reachability check."""

    ok: bool
    status: str  # "ok" | "auth_failed" | "infra_error" | "rate_limited"
    detail: str = ""


# A trivial prompt whose only purpose is to make the runtime contact its backend
# so auth/reachability problems surface before any scored work begins.
PREFLIGHT_PROMPT = "Reply with exactly: ok"


def parse_findings(text: str) -> list[Finding]:
    """Extract a JSON array of findings from model output.

    Models sometimes wrap JSON in markdown fences or add preamble text.
    We try to find and parse the JSON array regardless.
    """
    text = text.strip()

    # Try direct parse first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [_parse_one(item) for item in data if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass

    # Try to find JSON array in the text
    start = text.find("[")
    if start == -1:
        return []

    # Find matching closing bracket
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
                        return [
                            _parse_one(item) for item in data if isinstance(item, dict)
                        ]
                except json.JSONDecodeError:
                    pass
                break

    return []


def _parse_one(item: dict) -> Finding:
    return Finding(
        line_number=item.get("line"),
        code_snippet=item.get("code"),
        explanation=item.get("explanation"),
        confidence=item.get("confidence"),
        cwe_id=item.get("cwe"),
    )


def parse_gemini_envelope(raw: str) -> tuple[str, int | None, int | None]:
    """Pull (final_text, tokens_in, tokens_out) from ``gemini --output-format json``.

    The envelope is ``{"response": <text>, "stats": {"models": {<name>:
    {"tokens": {"input": .., "candidates": ..}}}}}``. Tokens are summed across
    every model the CLI used (router + main). Falls back to the raw text with no
    token counts when the output isn't the expected JSON object. Shared by the
    host-subprocess :class:`GeminiCLIAdapter` and the containerized gemini-cli
    runtime so both parse the envelope identically.
    """
    try:
        envelope = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw, None, None
    if not (isinstance(envelope, dict) and "response" in envelope):
        return raw, None, None
    stats = envelope.get("stats", {})
    models_stats = stats.get("models", {}) if isinstance(stats, dict) else {}
    total_in = 0
    total_out = 0
    for model_stats in models_stats.values():
        tokens = model_stats.get("tokens", {}) if isinstance(model_stats, dict) else {}
        total_in += tokens.get("input", 0) or 0
        total_out += tokens.get("candidates", 0) or 0
    tin = total_in if (total_in or total_out) else None
    tout = total_out if (total_in or total_out) else None
    return str(envelope["response"]), tin, tout


def _run_cli(
    cmd: list[str],
    timeout: int,
    input_text: str | None = None,
    cancel_event: threading.Event | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a CLI subprocess that can be cancelled via a threading.Event.

    subprocess.run() defers cancellation until the child exits, making
    Ctrl-C appear unresponsive. We use Popen plus a watchdog thread that
    terminates the child the moment ``cancel_event`` is set. Callers in
    worker threads must use this rather than installing a signal handler,
    since Python only allows signal.signal() on the main thread.

    ``env`` is passed straight to Popen; ``None`` (the default) means the child
    inherits this process's environment unchanged. Auth profiles supply a merged
    environment (parent env + resolved secrets) here.
    """
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE if input_text is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    watchdog_stop = threading.Event()
    watchdog: threading.Thread | None = None
    if cancel_event is not None:

        def _watch():
            while not watchdog_stop.is_set():
                if cancel_event.is_set():
                    proc.terminate()
                    return
                watchdog_stop.wait(0.2)

        watchdog = threading.Thread(target=_watch, daemon=True)
        watchdog.start()

    try:
        stdout, stderr = proc.communicate(
            input=input_text,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise
    finally:
        watchdog_stop.set()
        if watchdog is not None:
            watchdog.join(timeout=1.0)

    if cancel_event is not None and cancel_event.is_set():
        raise KeyboardInterrupt

    return subprocess.CompletedProcess(
        cmd,
        proc.returncode,
        stdout,
        stderr,
    )


class AgentAdapter(ABC):
    """Base class for runtimes (invocation methods).

    A runtime carries the three competitor dimensions as data:
    ``runtime`` (this invocation method, e.g. ``claude-code``), ``model_id``
    (the bare model the runtime drives), and ``tool_profile`` (what the model
    can do — ``single-shot`` today). ``name`` remains the unique identifier used
    as the DB ``model_id`` so existing scans are unaffected.
    """

    name: str
    runtime: str = "unknown"
    model_id: str = "unknown"
    tool_profile: str = "single-shot"
    auth_profile: AuthProfile | None = None
    needs_pacing: bool = False  # CLI tools with rolling subscription limits need delays

    @abstractmethod
    def run(
        self, prompt: str, cancel_event: threading.Event | None = None
    ) -> AgentResult: ...

    def _resolve_env(self) -> dict[str, str] | None:
        """Build the child environment for this runtime.

        Returns ``None`` when no auth profile is attached, so the subprocess
        inherits this process's environment exactly as before. With a profile,
        returns the parent environment merged with the profile's resolved
        secrets (profile values win). Propagates :class:`MissingSecretError`,
        which callers translate into an ``auth_failed`` result.
        """
        if self.auth_profile is None:
            return None
        resolved = self.auth_profile.resolve_env()
        return {**os.environ, **resolved}

    def preflight(self, cancel_event: threading.Event | None = None) -> PreflightResult:
        """Cheap auth/reachability check run before any scored work.

        Sends a trivial prompt and inspects the failure classification. A clean
        response is ``ok``; an auth failure is reported as such (never as a model
        that found nothing). Runtimes may override for a cheaper probe.
        """
        result = self.run(PREFLIGHT_PROMPT, cancel_event=cancel_event)
        if result.failure_kind is FailureKind.AUTH:
            return PreflightResult(False, "auth_failed", result.error or "auth failed")
        if result.failure_kind is not None:
            return PreflightResult(False, result.failure_kind.value, result.error or "")
        if result.error:
            return PreflightResult(False, "infra_error", result.error)
        return PreflightResult(True, "ok")


class ClaudeCLIAdapter(AgentAdapter):
    """Invoke Claude via the claude CLI tool."""

    def __init__(
        self,
        model: str = "haiku",
        timeout: int = 120,
        auth_profile: AuthProfile | None = None,
    ):
        self.model = model
        self.name = f"claude-{model}"
        self.runtime = "claude-code"
        self.model_id = model
        self.auth_profile = auth_profile
        self.needs_pacing = True
        self.timeout = timeout

    def run(
        self, prompt: str, cancel_event: threading.Event | None = None
    ) -> AgentResult:
        cmd = [
            "claude",
            "-p",
            "--model",
            self.model,
            "--output-format",
            "json",
            "--no-session-persistence",
        ]

        try:
            env = self._resolve_env()
        except MissingSecretError as e:
            return AgentResult(
                findings=[],
                raw_output="",
                failure_kind=FailureKind.AUTH,
                error=str(e),
            )

        try:
            result = _run_cli(
                cmd,
                self.timeout,
                input_text=prompt,
                cancel_event=cancel_event,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return AgentResult(
                findings=[],
                raw_output="",
                failure_kind=FailureKind.INFRA,
                error="timeout",
            )

        raw = result.stdout
        stderr = result.stderr

        if result.returncode != 0:
            kind = classify_failure(raw + stderr, failed=True)
            combined_output = (raw + stderr)[:500]
            error = (
                f"exit code {result.returncode}: {combined_output}"
                if combined_output
                else f"exit code {result.returncode}"
            )
            return AgentResult(
                findings=[],
                raw_output=raw,
                failure_kind=kind,
                error=error,
            )

        # Parse the JSON output from claude --output-format json
        text_content = raw
        tokens_in = None
        tokens_out = None
        cost_usd = None
        try:
            envelope = json.loads(raw)
            # claude --output-format json wraps in
            # {"type":"result","result":...,"usage":...}
            if isinstance(envelope, dict) and "result" in envelope:
                text_content = envelope["result"]
                usage = envelope.get("usage", {})
                tokens_in = usage.get("input_tokens")
                tokens_out = usage.get("output_tokens")
                cost_usd = envelope.get("cost_usd")
        except (json.JSONDecodeError, TypeError):
            text_content = raw

        findings = parse_findings(str(text_content))
        return AgentResult(
            findings=findings,
            raw_output=raw,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
        )


class GeminiCLIAdapter(AgentAdapter):
    """Invoke Gemini via the gemini CLI tool."""

    def __init__(
        self,
        model: str | None = None,
        timeout: int = 120,
        auth_profile: AuthProfile | None = None,
    ):
        self.model = model
        self.name = f"gemini-{model}" if model else "gemini"
        self.runtime = "gemini-cli"
        self.model_id = model or "default"
        self.auth_profile = auth_profile
        self.needs_pacing = True
        self.timeout = timeout

    def run(
        self, prompt: str, cancel_event: threading.Event | None = None
    ) -> AgentResult:
        cmd = ["gemini", "-p", prompt, "--output-format", "json"]
        if self.model:
            cmd.extend(["-m", self.model])

        try:
            env = self._resolve_env()
        except MissingSecretError as e:
            return AgentResult(
                findings=[],
                raw_output="",
                failure_kind=FailureKind.AUTH,
                error=str(e),
            )

        try:
            result = _run_cli(cmd, self.timeout, cancel_event=cancel_event, env=env)
        except subprocess.TimeoutExpired:
            return AgentResult(
                findings=[],
                raw_output="",
                failure_kind=FailureKind.INFRA,
                error="timeout",
            )

        raw = result.stdout
        stderr = result.stderr

        if result.returncode != 0:
            kind = classify_failure(raw + stderr, failed=True)
            combined_output = (raw + stderr)[:500]
            error = (
                f"exit code {result.returncode}: {combined_output}"
                if combined_output
                else f"exit code {result.returncode}"
            )
            return AgentResult(
                findings=[],
                raw_output=raw,
                failure_kind=kind,
                error=error,
            )

        # Parse the JSON output from gemini --output-format json
        text_content, tokens_in, tokens_out = parse_gemini_envelope(raw)

        findings = parse_findings(text_content)
        return AgentResult(
            findings=findings,
            raw_output=raw,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )


# -- Tool-using (ReAct) loop for OpenAI-compatible APIs -----------------------
#
# CLI runtimes (Claude Code, Gemini CLI) are already agents that read files on
# their own. A bare OpenAI-compatible endpoint is not — single-shot, it only ever
# sees the one file pasted into the prompt. With tools enabled, the adapter drives
# the same ReAct loop the benchmark's raw-api-loop runtime uses (read_file / grep
# / list_dir), but rooted at a host directory so the model can follow imports,
# callers, and helpers into sibling files before answering. Used by both the scan
# pass (audit a file) and the review pass (verify a reported finding) — the user's
# prompt sets the role and the exact JSON contract, so this framing stays neutral
# on both. Same tool *names* the benchmark uses (dispatch_tool resolves them);
# only the framing differs (project root rather than the container's /src mount).

_TOOL_LOOP_SYSTEM = (
    "You have read-only access to a project's source tree through tools: "
    "read_file, grep, and list_dir, all taking paths relative to the project "
    "root. The file in question is included in the user's message. Use the tools "
    "to read related code in other files (imports, call sites, helper and type "
    "definitions) whenever that helps you answer accurately. When finished, "
    "output EXACTLY the JSON specified in the user's instructions, with no prose "
    "around it."
)

# OpenAI function-calling schema, root-neutral wording (the benchmark's TOOLS say
# "/src"; we keep that instrument untouched and describe a generic project root).
_TOOL_LOOP_SCHEMA: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a source file by path relative to the project root. "
                "Optionally restrict to a line range."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Search the project for a regex (ripgrep). Optional path relative "
                "to the project root narrows the search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List a directory by path relative to the project root.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
    },
]

# Bounds for the host tool loop. Smaller than the benchmark's (40 / 500k): one
# file with optional excursions into neighbors, not a whole-repo hunt, and we want
# per-file cost bounded for interactive use.
_TOOL_LOOP_MAX_STEPS = 15
_TOOL_LOOP_TOKEN_BUDGET = 256_000


class OpenAIAPIAdapter(AgentAdapter):
    """Invoke a local model via OpenAI-compatible API (e.g., llama.cpp server).

    Single-shot by default (the file is pasted into the prompt). With
    ``tools=True`` and a ``tools_root``, ``run`` instead drives a ReAct tool loop
    rooted at that directory so the model can read sibling files — the same loop
    the benchmark's raw-api-loop runtime uses, confined to ``tools_root``.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8080/v1",
        timeout: int = 300,
        auth_profile: AuthProfile | None = None,
        tools: bool = False,
        tools_root: str | None = None,
        max_tool_steps: int = _TOOL_LOOP_MAX_STEPS,
        tool_token_budget: int = _TOOL_LOOP_TOKEN_BUDGET,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        netloc = urlparse(self.base_url).netloc or self.base_url or "local"
        self.name = f"{netloc}/{model}"
        self.runtime = "openai-api"
        self.model_id = model
        self.auth_profile = auth_profile
        self.timeout = timeout
        self.use_tools = tools
        self.tools_root = tools_root
        self.tool_profile = "react" if tools else "single-shot"
        self.max_tool_steps = max_tool_steps
        self.tool_token_budget = tool_token_budget

    def _bearer_token(self) -> str | None:
        """Resolve a bearer token for the request, or ``None`` to send no header.

        Local servers (llama.cpp, LM Studio, Ollama) need no key, so the default
        is no Authorization header. Two ways a key is supplied:

        * A benchmark competitor attaches an :class:`AuthProfile`; we use the
          resolved ``OPENAI_API_KEY`` (or the sole resolved value), and a
          configured-but-absent secret raises :class:`MissingSecretError` so it
          becomes ``auth_failed`` rather than a silent unauthenticated call.
        * An ad-hoc CLI scan (no profile) reads ``OPENAI_API_KEY`` straight from
          the environment — the universal OpenAI-compatible convention. This is
          how hosted endpoints (DeepSeek, OpenRouter, ...) are authenticated
          from ``nelson scan``/``nelson review``. Unset means no header, which is
          correct for localhost.
        """
        if self.auth_profile is None:
            return os.environ.get("OPENAI_API_KEY")
        resolved = self.auth_profile.resolve_env()
        if "OPENAI_API_KEY" in resolved:
            return resolved["OPENAI_API_KEY"]
        return next(iter(resolved.values()), None)

    def run(
        self, prompt: str, cancel_event: threading.Event | None = None
    ) -> AgentResult:
        if self.use_tools and self.tools_root:
            return self._run_with_tools(prompt, cancel_event)

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,  # Low temp for consistent analysis
        }

        if cancel_event is not None and cancel_event.is_set():
            raise KeyboardInterrupt

        try:
            token = self._bearer_token()
        except MissingSecretError as e:
            return AgentResult(
                findings=[],
                raw_output="",
                failure_kind=FailureKind.AUTH,
                error=str(e),
            )
        headers = {"Authorization": f"Bearer {token}"} if token else None

        try:
            resp = httpx.post(url, json=payload, timeout=self.timeout, headers=headers)
        except httpx.TimeoutException:
            return AgentResult(
                findings=[],
                raw_output="",
                failure_kind=FailureKind.INFRA,
                error="timeout",
            )
        except httpx.ConnectError as e:
            return AgentResult(
                findings=[],
                raw_output="",
                failure_kind=FailureKind.INFRA,
                error=f"connection error: {e}",
            )

        if resp.status_code == 429:
            return AgentResult(
                findings=[],
                raw_output="",
                failure_kind=FailureKind.RATE_LIMIT,
                error="rate limited",
            )
        if resp.status_code in (401, 403):
            return AgentResult(
                findings=[],
                raw_output=resp.text,
                failure_kind=FailureKind.AUTH,
                error=f"HTTP {resp.status_code}: {resp.text[:500]}",
            )
        if resp.status_code != 200:
            return AgentResult(
                findings=[],
                raw_output=resp.text,
                failure_kind=classify_failure(resp.text, failed=True),
                error=f"HTTP {resp.status_code}: {resp.text[:500]}",
            )

        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        findings = parse_findings(text)
        return AgentResult(
            findings=findings,
            raw_output=text,
            tokens_in=usage.get("prompt_tokens"),
            tokens_out=usage.get("completion_tokens"),
        )

    def _run_with_tools(
        self, prompt: str, cancel_event: threading.Event | None
    ) -> AgentResult:
        """Drive the ReAct tool loop rooted at ``self.tools_root``.

        Reuses the benchmark loop (``raw_api_loop.run_loop``) for its retry,
        budget, and usage accounting, supplying scan-mode framing + a root-neutral
        tool schema. The tools resolve every path under ``tools_root`` (realpath
        check), so the model cannot read outside the scanned tree. Integrity holds:
        a persistent auth/transport failure is surfaced as a FailureKind, never as
        a model that looked and found nothing.
        """
        if cancel_event is not None and cancel_event.is_set():
            raise KeyboardInterrupt
        try:
            token = self._bearer_token()
        except MissingSecretError as e:
            return AgentResult(
                findings=[], raw_output="", failure_kind=FailureKind.AUTH, error=str(e)
            )

        def _post(url: str, payload: dict, api_key: str) -> dict:
            # Checked between API calls (mirrors single-shot's pre-call cancel
            # check); an in-flight request still blocks until it returns/times out.
            if cancel_event is not None and cancel_event.is_set():
                raise KeyboardInterrupt
            return raw_api_loop._post_chat(url, payload, api_key)

        try:
            final_text, tin, tout, _steps, cost = raw_api_loop.run_loop(
                prompt,
                base_url=self.base_url,
                model=self.model,
                api_key=token or "",
                max_steps=self.max_tool_steps,
                token_budget=self.tool_token_budget,
                temperature=0.1,
                post=_post,
                src_root=self.tools_root,
                system_prompt=_TOOL_LOOP_SYSTEM,
                tools=_TOOL_LOOP_SCHEMA,
            )
        except urllib.error.HTTPError as e:
            body = ""
            with contextlib.suppress(OSError, AttributeError):
                body = e.read().decode("utf-8", errors="replace")
            if e.code in (401, 403):
                kind = FailureKind.AUTH
            elif e.code == 429:
                kind = FailureKind.RATE_LIMIT
            else:
                kind = classify_failure(body, failed=True)
            return AgentResult(
                findings=[],
                raw_output=body,
                failure_kind=kind,
                error=f"HTTP {e.code}: {body[:500]}",
            )
        except urllib.error.URLError as e:
            return AgentResult(
                findings=[],
                raw_output="",
                failure_kind=FailureKind.INFRA,
                error=f"connection error: {e.reason}",
            )

        return AgentResult(
            findings=parse_findings(final_text),
            raw_output=final_text,
            tokens_in=tin or None,
            tokens_out=tout or None,
            cost_usd=cost,
        )


# Registry of known adapters
ADAPTERS: dict[str, type[AgentAdapter]] = {
    "claude": ClaudeCLIAdapter,
    "gemini": GeminiCLIAdapter,
    "openai": OpenAIAPIAdapter,
}


def create_adapter(
    spec: str,
    auth_profile: AuthProfile | None = None,
    *,
    tools: bool = False,
    tools_root: str | None = None,
) -> AgentAdapter:
    """Create an adapter from a spec string.

    ``auth_profile`` is optional and defaults to ``None``: with no profile the
    runtime inherits this process's environment unchanged (the existing
    behavior). A competitor in the benchmark attaches a profile so its secrets
    are injected into the run; ad-hoc CLI scans leave it unset.

    ``tools`` / ``tools_root`` enable the ReAct tool loop for OpenAI-compatible
    endpoints, rooted at ``tools_root`` (the scan's target directory). They are
    ignored for the CLI runtimes (Claude Code, Gemini CLI), which already read
    files on their own.

    Examples:
        "claude:haiku"              -> Claude CLI with Haiku
        "claude:sonnet"             -> Claude CLI with Sonnet
        "gemini:gemini-2.5-flash"   -> Gemini CLI with specific model
        "gemini:"                   -> Gemini CLI with default model
        "lmstudio:google/gemma-4"   -> LM Studio on localhost:1234
        "ollama:llama3"             -> Ollama on localhost:11434
        "openai:model@http://h:p/v1" -> Custom OpenAI-compatible endpoint
    """
    parts = spec.split(":", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid adapter spec '{spec}'. Expected 'type:model', e.g. 'claude:haiku'"
        )

    adapter_type, model_spec = parts

    if adapter_type == "claude":
        return ClaudeCLIAdapter(model=model_spec, auth_profile=auth_profile)
    elif adapter_type == "gemini":
        return GeminiCLIAdapter(
            model=model_spec if model_spec else None, auth_profile=auth_profile
        )
    elif adapter_type == "openai":
        if "@" in model_spec:
            model, url = model_spec.split("@", 1)
            return OpenAIAPIAdapter(
                model=model,
                base_url=url,
                auth_profile=auth_profile,
                tools=tools,
                tools_root=tools_root,
            )
        return OpenAIAPIAdapter(
            model=model_spec,
            auth_profile=auth_profile,
            tools=tools,
            tools_root=tools_root,
        )
    elif adapter_type == "lmstudio":
        return OpenAIAPIAdapter(
            model=model_spec,
            base_url="http://localhost:1234/v1",
            auth_profile=auth_profile,
            tools=tools,
            tools_root=tools_root,
        )
    elif adapter_type == "ollama":
        return OpenAIAPIAdapter(
            model=model_spec,
            base_url="http://localhost:11434/v1",
            auth_profile=auth_profile,
            tools=tools,
            tools_root=tools_root,
        )
    else:
        raise ValueError(
            f"Unknown adapter type '{adapter_type}'. "
            f"Known: claude, gemini, openai, lmstudio, ollama"
        )
