"""Agent adapters for invoking AI models via CLI or API."""

import json
import logging
import subprocess
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)


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
) -> subprocess.CompletedProcess[str]:
    """Run a CLI subprocess that can be cancelled via a threading.Event.

    subprocess.run() defers cancellation until the child exits, making
    Ctrl-C appear unresponsive. We use Popen plus a watchdog thread that
    terminates the child the moment ``cancel_event`` is set. Callers in
    worker threads must use this rather than installing a signal handler,
    since Python only allows signal.signal() on the main thread.
    """
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE if input_text is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
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
    """Base class for agent adapters."""

    name: str
    needs_pacing: bool = False  # CLI tools with rolling subscription limits need delays

    @abstractmethod
    def run(
        self, prompt: str, cancel_event: threading.Event | None = None
    ) -> AgentResult: ...


class ClaudeCLIAdapter(AgentAdapter):
    """Invoke Claude via the claude CLI tool."""

    def __init__(self, model: str = "haiku", timeout: int = 120):
        self.model = model
        self.name = f"claude-{model}"
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
            result = _run_cli(
                cmd, self.timeout, input_text=prompt, cancel_event=cancel_event
            )
        except subprocess.TimeoutExpired:
            return AgentResult(findings=[], raw_output="", error="timeout")

        raw = result.stdout
        stderr = result.stderr

        # Detect rate limiting
        if result.returncode != 0:
            combined = raw + stderr
            if any(s in combined.lower() for s in ["rate limit", "429", "overloaded"]):
                return AgentResult(
                    findings=[],
                    raw_output=raw,
                    rate_limited=True,
                    error="rate limited",
                )
            return AgentResult(
                findings=[],
                raw_output=raw,
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

    def __init__(self, model: str | None = None, timeout: int = 120):
        self.model = model
        self.name = f"gemini-{model}" if model else "gemini"
        self.needs_pacing = True
        self.timeout = timeout

    def run(
        self, prompt: str, cancel_event: threading.Event | None = None
    ) -> AgentResult:
        cmd = ["gemini", "-p", prompt, "--output-format", "json"]
        if self.model:
            cmd.extend(["-m", self.model])

        try:
            result = _run_cli(cmd, self.timeout, cancel_event=cancel_event)
        except subprocess.TimeoutExpired:
            return AgentResult(findings=[], raw_output="", error="timeout")

        raw = result.stdout
        stderr = result.stderr

        if result.returncode != 0:
            combined = raw + stderr
            if any(
                s in combined.lower()
                for s in ["rate limit", "429", "overloaded", "quota"]
            ):
                return AgentResult(
                    findings=[],
                    raw_output=raw,
                    rate_limited=True,
                    error="rate limited",
                )
            return AgentResult(
                findings=[],
                raw_output=raw,
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
    ):
        self.model = model
        self.name = f"local-{model}"
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

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
            resp = httpx.post(url, json=payload, timeout=self.timeout)
        except httpx.TimeoutException:
            return AgentResult(findings=[], raw_output="", error="timeout")
        except httpx.ConnectError as e:
            return AgentResult(
                findings=[],
                raw_output="",
                error=f"connection error: {e}",
            )

        if resp.status_code == 429:
            return AgentResult(
                findings=[], raw_output="", rate_limited=True, error="rate limited"
            )
        if resp.status_code != 200:
            return AgentResult(
                findings=[],
                raw_output=resp.text,
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


def create_adapter(spec: str) -> AgentAdapter:
    """Create an adapter from a spec string.

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
        return ClaudeCLIAdapter(model=model_spec)
    elif adapter_type == "gemini":
        return GeminiCLIAdapter(model=model_spec if model_spec else None)
    elif adapter_type == "openai":
        if "@" in model_spec:
            model, url = model_spec.split("@", 1)
            return OpenAIAPIAdapter(model=model, base_url=url)
        return OpenAIAPIAdapter(model=model_spec)
    elif adapter_type == "lmstudio":
        return OpenAIAPIAdapter(model=model_spec, base_url="http://localhost:1234/v1")
    elif adapter_type == "ollama":
        return OpenAIAPIAdapter(model=model_spec, base_url="http://localhost:11434/v1")
    else:
        raise ValueError(
            f"Unknown adapter type '{adapter_type}'. "
            f"Known: claude, gemini, openai, lmstudio, ollama"
        )
