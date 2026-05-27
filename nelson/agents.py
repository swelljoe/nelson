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

import json
import logging
import os
import subprocess
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import urlparse

import httpx

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
            return AgentResult(
                findings=[],
                raw_output=raw,
                failure_kind=kind,
                error=f"exit code {result.returncode}: {stderr[:500]}",
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
            return AgentResult(
                findings=[],
                raw_output=raw,
                failure_kind=kind,
                error=f"exit code {result.returncode}: {stderr[:500]}",
            )

        # Parse the JSON output from gemini --output-format json
        text_content = raw
        tokens_in = None
        tokens_out = None
        try:
            envelope = json.loads(raw)
            if isinstance(envelope, dict) and "response" in envelope:
                text_content = envelope["response"]
                # Sum tokens across all models used (router + main)
                stats = envelope.get("stats", {})
                models_stats = stats.get("models", {})
                total_in = 0
                total_out = 0
                for model_stats in models_stats.values():
                    tokens = model_stats.get("tokens", {})
                    total_in += tokens.get("input", 0)
                    total_out += tokens.get("candidates", 0)
                if total_in or total_out:
                    tokens_in = total_in
                    tokens_out = total_out
        except (json.JSONDecodeError, TypeError):
            text_content = raw

        findings = parse_findings(str(text_content))
        return AgentResult(
            findings=findings,
            raw_output=raw,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )


class OpenAIAPIAdapter(AgentAdapter):
    """Invoke a local model via OpenAI-compatible API (e.g., llama.cpp server)."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8080/v1",
        timeout: int = 300,
        auth_profile: AuthProfile | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        netloc = urlparse(self.base_url).netloc or self.base_url or "local"
        self.name = f"{netloc}/{model}"
        self.runtime = "openai-api"
        self.model_id = model
        self.auth_profile = auth_profile
        self.timeout = timeout

    def _bearer_token(self) -> str | None:
        """Resolve the profile to a bearer token, or ``None`` if no profile.

        Local servers (llama.cpp, LM Studio, Ollama) need no key, so the default
        is no Authorization header. Hosted OpenAI-compatible endpoints attach a
        profile; we use the resolved ``OPENAI_API_KEY`` (or the sole resolved
        value). Raises :class:`MissingSecretError` for a configured-but-absent
        secret so it becomes ``auth_failed``, not a silent unauthenticated call.
        """
        if self.auth_profile is None:
            return None
        resolved = self.auth_profile.resolve_env()
        if "OPENAI_API_KEY" in resolved:
            return resolved["OPENAI_API_KEY"]
        return next(iter(resolved.values()), None)

    def run(
        self, prompt: str, cancel_event: threading.Event | None = None
    ) -> AgentResult:
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


# Registry of known adapters
ADAPTERS: dict[str, type[AgentAdapter]] = {
    "claude": ClaudeCLIAdapter,
    "gemini": GeminiCLIAdapter,
    "openai": OpenAIAPIAdapter,
}


def create_adapter(spec: str, auth_profile: AuthProfile | None = None) -> AgentAdapter:
    """Create an adapter from a spec string.

    ``auth_profile`` is optional and defaults to ``None``: with no profile the
    runtime inherits this process's environment unchanged (the existing
    behavior). A competitor in the benchmark attaches a profile so its secrets
    are injected into the run; ad-hoc CLI scans leave it unset.

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
                model=model, base_url=url, auth_profile=auth_profile
            )
        return OpenAIAPIAdapter(model=model_spec, auth_profile=auth_profile)
    elif adapter_type == "lmstudio":
        return OpenAIAPIAdapter(
            model=model_spec,
            base_url="http://localhost:1234/v1",
            auth_profile=auth_profile,
        )
    elif adapter_type == "ollama":
        return OpenAIAPIAdapter(
            model=model_spec,
            base_url="http://localhost:11434/v1",
            auth_profile=auth_profile,
        )
    else:
        raise ValueError(
            f"Unknown adapter type '{adapter_type}'. "
            f"Known: claude, gemini, openai, lmstudio, ollama"
        )
