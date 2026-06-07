"""In-container ReAct agent for the ``raw-api-loop`` runtime (stdlib only).

Runs INSIDE the benchmark container (same isolation + read-only ``/src`` mount as
the claude-code runtime). Drives any OpenAI-compatible ``chat/completions``
endpoint with a small tool registry — ``read_file`` / ``grep`` / ``list_dir``, all
confined to ``/src`` — then prints a single claude-shaped result object so the
runner parses every runtime uniformly.

Why stdlib only (``urllib`` + ``json``, no ``httpx``): the container image must
build offline with no pip, and there is no dependency to drift from the host venv.

Configuration comes from the environment (set by ``RawApiLoopRuntime.build_spec``
and ``EnvKeyAuth``):

- ``NELSON_BASE_URL``     OpenAI-compatible base, e.g. ``https://api.deepseek.com``
- ``NELSON_MODEL``        model id passed to the endpoint
- ``NELSON_API_KEY``      bearer token (injected from a secret name)
- ``NELSON_MAX_STEPS``    tool-use turn cap (default 40)
- ``NELSON_TOKEN_BUDGET`` cumulative token cap (default 500000)
- ``NELSON_INPUT_USD_PER_MTOK`` / ``NELSON_OUTPUT_USD_PER_MTOK`` optional pricing

Integrity: an auth/transport failure exits non-zero and prints the provider error
to stderr, so the runner's ``classify_failure`` maps it to ``auth_failed`` /
``infra_error`` — never a model that "found nothing". The audit prompt is read
from stdin verbatim (identical framing to claude-code); nothing about the planted
vulnerability is known to this script.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

SRC_ROOT = "/src"
MAX_TOOL_OUTPUT = 20_000  # chars; keep a single tool result from blowing context
# Per-API-call read timeout. Reasoning models (Gemini 3.x pro, MiMo) can think for
# minutes on a single turn over a large C file; a tight cap aborts the whole run as
# an infra_error and silently drops the slowest (often strongest) models from the
# matrix. 600s lets a slow reasoner finish a turn rather than penalising it. A slow
# self-hosted box (e.g. a 27B on an APU, where one big-context turn can exceed 600s)
# raises it via NELSON_HTTP_TIMEOUT (set from the competitor's cost_model http_timeout).
HTTP_TIMEOUT = 600  # default seconds per API call


def _http_timeout() -> int:
    """Per-call read timeout: NELSON_HTTP_TIMEOUT if set and valid, else the default."""
    raw = os.environ.get("NELSON_HTTP_TIMEOUT")
    if raw:
        with contextlib.suppress(ValueError):
            return int(raw)
    return HTTP_TIMEOUT


# Transient faults that warrant a retry rather than failing the whole run: provider
# rate limits (429 — Mistral caps tokens/minute, which an agentic burst of large-
# context calls trips even on a paid tier) and the 5xx family, plus transport errors
# (a self-hosted endpoint like LM Studio closing the socket mid-response on a heavy
# request). We pace-and-recover with exponential backoff; a persistent outage still
# surfaces after the cap so the runner records a real infra_error, never a false
# "found nothing".
HTTP_RETRY_STATUS = (429, 500, 502, 503, 504)
MAX_HTTP_RETRIES = 5  # retries after the first attempt (6 tries total)
BACKOFF_BASE_S = 2.0  # exponential: 1, 2, 4, 8, 16 s ...
BACKOFF_CAP_S = 60.0  # never wait longer than this between tries

SYSTEM_PROMPT = (
    "You are a tool-using security auditor. Use the provided tools to read the "
    "code under /src that you need, then give your final answer EXACTLY as the "
    "JSON array specified in the user's instructions, with no prose around it."
)

# OpenAI function-calling schema for the three sandboxed tools.
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file under /src. Optionally a line range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "path under /src"},
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
            "description": "Search for a regex under /src (ripgrep).",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "description": "path under /src"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List a directory under /src.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "path under /src"},
                },
            },
        },
    },
]


# -- Sandboxed tools ---------------------------------------------------------


def _resolve_in_src(path: str, src_root: str | None = None) -> str | None:
    """Resolve a model-supplied path under ``src_root``; None if it escapes.

    Defense in depth on top of the read-only mount: realpath collapses ``..`` and
    symlinks, so a target that resolves outside the source root (``/etc/passwd``,
    ``../`` traversal, a symlink pointing out) is rejected. This keeps the audit
    honest — the model cannot read the mounted credentials, this script, or host
    files — even though the mount is already ``ro``.
    """
    root = src_root or SRC_ROOT
    root_real = os.path.realpath(root)
    candidate = path if os.path.isabs(path) else os.path.join(root, path)
    real = os.path.realpath(candidate)
    if real == root_real or real.startswith(root_real + os.sep):
        return real
    return None


def tool_read_file(args: dict[str, Any], src_root: str | None = None) -> str:
    rel = str(args.get("path", ""))
    real = _resolve_in_src(rel, src_root)
    if real is None:
        return f"error: path {rel!r} is outside the source tree"
    if not os.path.isfile(real):
        return f"error: not a file: {rel}"
    try:
        with open(real, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as e:
        return f"error: {e}"
    start = args.get("start_line")
    end = args.get("end_line")
    s = max(1, int(start) if start is not None else 1)
    e = int(end) if end is not None else len(lines)
    window = lines[s - 1 : e]
    # Prefix each line with its absolute 1-based line number (same `N:` delimiter as 
    # `rg -n`),
    # so numbers are consistent across tools). Without this the model reads raw
    # text and must hand-count to cite a location — which it does badly (observed
    # ~75-line drift), so a genuine find lands tens of lines off and the
    # near-exact localization gate rejects it. Numbering removes the guesswork and
    # lets the gate trust the competitor's reported line.
    numbered = (f"{s + i}:{line}" for i, line in enumerate(window))
    return "".join(numbered)[:MAX_TOOL_OUTPUT]


def tool_grep(args: dict[str, Any], src_root: str | None = None) -> str:
    pattern = str(args.get("pattern", ""))
    rel = str(args.get("path", "."))
    if not pattern:
        return "error: empty pattern"
    real = _resolve_in_src(rel, src_root)
    if real is None:
        return f"error: path {rel!r} is outside the source tree"
    try:
        proc = subprocess.run(
            ["rg", "-n", "--no-heading", "-S", "--", pattern, real],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return f"error: {e}"
    out = proc.stdout or proc.stderr or "(no matches)"
    return out[:MAX_TOOL_OUTPUT]


def tool_list_dir(args: dict[str, Any], src_root: str | None = None) -> str:
    rel = str(args.get("path", "."))
    real = _resolve_in_src(rel, src_root)
    if real is None:
        return f"error: path {rel!r} is outside the source tree"
    if not os.path.isdir(real):
        return f"error: not a directory: {rel}"
    try:
        entries = sorted(os.listdir(real))
    except OSError as e:
        return f"error: {e}"
    return "\n".join(entries)[:MAX_TOOL_OUTPUT]


TOOL_FUNCS: dict[str, Callable[[dict[str, Any], str | None], str]] = {
    "read_file": tool_read_file,
    "grep": tool_grep,
    "list_dir": tool_list_dir,
}


def dispatch_tool(name: str, args: dict[str, Any], src_root: str | None = None) -> str:
    """Run a tool by name; a bad call returns an error string (never crashes)."""
    fn = TOOL_FUNCS.get(name)
    if fn is None:
        return f"error: unknown tool {name!r}"
    try:
        return fn(args, src_root)
    except Exception as e:  # a tool error must not kill the loop
        return f"error: {e}"


# -- HTTP + cost -------------------------------------------------------------


def _retry_after_seconds(err: urllib.error.HTTPError) -> float | None:
    """Parse a Retry-After header (delta-seconds form) from a 429/503, if present."""
    with contextlib.suppress(Exception):
        raw = err.headers.get("Retry-After") if err.headers else None
        if raw is not None:
            return max(0.0, float(raw.strip()))
    return None


def _post_chat(
    url: str,
    payload: dict[str, Any],
    api_key: str,
    *,
    max_retries: int = MAX_HTTP_RETRIES,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """POST a chat/completions request and return the parsed JSON body.

    Transient faults are retried with exponential backoff (honoring Retry-After):
    HTTP 429/5xx and transport errors (connection reset, the provider closing the
    socket mid-response, TLS read errors). This lets a rate-limited provider and a
    flaky self-hosted endpoint pace-and-recover instead of failing the whole run on
    the first hiccup. After ``max_retries`` the last error propagates so main()
    still exits non-zero with the provider error — the runner then classifies a
    persistent failure as auth/infra, never masking it as a model finding nothing.
    Injectable in tests via ``run_loop(post=...)``; ``sleep`` is injectable too.
    """
    data = json.dumps(payload).encode("utf-8")
    for attempt in range(max_retries + 1):
        # operator-configured base_url, not model-controlled, so this S310 audit
        # warning is acceptable here. A fresh Request per attempt keeps retries clean.
        req = urllib.request.Request(url, data=data, method="POST")  # noqa: S310
        req.add_header("Content-Type", "application/json")
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        try:
            with urllib.request.urlopen(req, timeout=_http_timeout()) as resp:  # noqa: S310
                body = resp.read().decode("utf-8", errors="replace")
            parsed = json.loads(body)
            return parsed if isinstance(parsed, dict) else {}
        except urllib.error.HTTPError as e:
            if e.code in HTTP_RETRY_STATUS and attempt < max_retries:
                delay = _retry_after_seconds(e)
                if delay is None:
                    delay = min(BACKOFF_CAP_S, BACKOFF_BASE_S**attempt)
                sleep(delay)
                continue
            raise
        except urllib.error.URLError:
            # Transport-level failure (connection refused/reset, socket closed
            # mid-response, TLS read error). HTTPError is a URLError subclass but is
            # handled above, so this is purely the no-HTTP-response case.
            if attempt < max_retries:
                sleep(min(BACKOFF_CAP_S, BACKOFF_BASE_S**attempt))
                continue
            raise
    # Unreachable: the final attempt either returns or raises. Satisfy type checkers.
    raise RuntimeError("retry loop exited without returning")  # pragma: no cover


def usage_delta(usage: dict[str, Any]) -> tuple[int, int]:
    """(input, output) tokens from one response's ``usage`` block.

    Output is taken as ``total_tokens - prompt_tokens`` when that exceeds the
    reported ``completion_tokens``. Reasoning ("thinking") models bill their
    thinking tokens at the output rate, and some providers — notably Gemini's
    OpenAI-compatible layer — fold those into ``total_tokens`` but omit them from
    ``completion_tokens``, so summing ``completion_tokens`` alone undercounts
    billed output (and the cost derived from it). ``max`` keeps the fallback safe
    when ``total_tokens`` is absent or inconsistent (then output = completion).
    """
    prompt = usage.get("prompt_tokens") or 0
    completion = usage.get("completion_tokens") or 0
    total = usage.get("total_tokens") or 0
    return prompt, max(completion, total - prompt)


def compute_cost(
    tokens_in: int, tokens_out: int, in_price: float | None, out_price: float | None
) -> float | None:
    """USD cost from per-million-token prices, or None if no pricing supplied."""
    if in_price is None and out_price is None:
        return None
    cost = 0.0
    if in_price is not None:
        cost += (tokens_in / 1_000_000) * in_price
    if out_price is not None:
        cost += (tokens_out / 1_000_000) * out_price
    return round(cost, 6)


def _trim(value: Any, limit: int = 2000) -> Any:
    """Trim a string for the transcript so a huge turn doesn't bloat the log."""
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "...(trimmed)"
    return value


# -- The ReAct loop ----------------------------------------------------------


def run_loop(
    prompt: str,
    *,
    base_url: str,
    model: str,
    api_key: str,
    max_steps: int = 40,
    token_budget: int = 500_000,
    temperature: float = 0.1,
    post: Callable[[str, dict[str, Any], str], dict[str, Any]] = _post_chat,
    src_root: str | None = None,
    system_prompt: str | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> tuple[str, int, int, list[dict[str, Any]], float | None]:
    """Drive the tool-use loop; return (final_text, in, out, steps, provider_cost).

    Terminates when the model answers without tool calls, at ``max_steps``, or
    once cumulative tokens reach ``token_budget`` — on a cap the final turn is sent
    with tools withheld so the model must produce its answer. If it still emits no
    parseable array, the last assistant text is returned (the runner's parser then
    yields ``[]`` — a legitimate "found nothing", not an error).

    ``system_prompt`` and ``tools`` default to this module's ``SYSTEM_PROMPT`` /
    ``TOOLS`` (the container audit framing over ``/src``) so the benchmark path is
    unchanged. Host callers (the scan-mode API adapter) pass their own framing and
    tool descriptions while reusing this loop's retry/budget/usage machinery; the
    tool *names* must still be those ``dispatch_tool`` knows.

    ``provider_cost`` is the summed real charge the provider reports per call
    (OpenRouter's ``usage.cost`` when we request ``usage: {include: true}``), or
    None if the provider returns no cost. It is authoritative — it already nets
    out prompt-cache discounts that our per-token ``compute_cost`` cannot see
    (the ReAct loop resends the whole context each turn; most of those input
    tokens are cache reads billed at a fraction). main() prefers it over
    ``compute_cost`` so the recorded cost matches billing.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    # usage.include is an OpenRouter request param; guard on the host so a strict
    # OpenAI-compat server (self-hosted llama-server, a vendor API) never sees an
    # unknown field. Providers that don't return ``cost`` just leave it None and
    # main() falls back to compute_cost.
    want_cost = "openrouter.ai" in base_url
    system_prompt = SYSTEM_PROMPT if system_prompt is None else system_prompt
    tools = TOOLS if tools is None else tools
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    total_in = 0
    total_out = 0
    total_cost: float | None = None
    steps: list[dict[str, Any]] = []
    final_text = ""

    for step in range(max_steps):
        force_final = step == max_steps - 1 or (total_in + total_out) >= token_budget
        if force_final:
            # Withholding tools alone isn't enough — some models (e.g. DeepSeek)
            # will still emit a native-format tool call as plain text. Spell out
            # that the budget is spent and only the JSON array is wanted now.
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You have reached your tool-use budget. Do not call any "
                        "tools. Based only on what you have already read, output "
                        "NOW your final answer as the JSON array specified earlier "
                        "(output [] if you found no exploitable vulnerability)."
                    ),
                }
            )
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if want_cost:
            payload["usage"] = {"include": True}
        if not force_final:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        resp = post(url, payload, api_key)
        usage = resp.get("usage") or {}
        step_in, step_out = usage_delta(usage)
        total_in += step_in
        total_out += step_out
        step_cost = usage.get("cost")
        if isinstance(step_cost, (int, float)):
            total_cost = (total_cost or 0.0) + float(step_cost)
        choices = resp.get("choices") or [{}]
        msg = (choices[0].get("message") or {}) if choices else {}
        tool_calls = msg.get("tool_calls") or []
        steps.append(
            {
                "type": "step",
                "step": step,
                "content": _trim(msg.get("content")),
                "tool_calls": [
                    (tc.get("function") or {}).get("name") for tc in tool_calls
                ],
            }
        )

        if tool_calls and not force_final:
            messages.append(msg)  # assistant turn carrying the tool_calls
            for tc in tool_calls:
                fn = tc.get("function") or {}
                name = str(fn.get("name", ""))
                try:
                    call_args = json.loads(fn.get("arguments") or "{}")
                except (json.JSONDecodeError, TypeError):
                    call_args = {}
                if not isinstance(call_args, dict):
                    call_args = {}
                result = dispatch_tool(name, call_args, src_root)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id"),
                        "content": result[:MAX_TOOL_OUTPUT],
                    }
                )
            continue

        # No tool calls (or the forced-final turn): this is the answer.
        final_text = msg.get("content") or ""
        break

    return final_text, total_in, total_out, steps, total_cost


def _float_env(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def main() -> None:
    prompt = sys.stdin.read()
    base_url = os.environ.get("NELSON_BASE_URL", "")
    model = os.environ.get("NELSON_MODEL", "")
    api_key = os.environ.get("NELSON_API_KEY", "")
    if not base_url or not model:
        print("missing NELSON_BASE_URL/NELSON_MODEL", file=sys.stderr)
        sys.exit(2)

    in_price = _float_env("NELSON_INPUT_USD_PER_MTOK")
    out_price = _float_env("NELSON_OUTPUT_USD_PER_MTOK")
    # 0.0 is a legitimate (greedy) temperature, so test for None rather than falsiness.
    temperature = _float_env("NELSON_TEMPERATURE")
    try:
        final_text, tin, tout, steps, provider_cost = run_loop(
            prompt,
            base_url=base_url,
            model=model,
            api_key=api_key,
            max_steps=_int_env("NELSON_MAX_STEPS", 40),
            token_budget=_int_env("NELSON_TOKEN_BUDGET", 500_000),
            temperature=0.1 if temperature is None else temperature,
        )
    except urllib.error.HTTPError as e:
        body = ""
        with contextlib.suppress(OSError, AttributeError):
            body = e.read().decode("utf-8", errors="replace")
        print(f"HTTP {e.code}: {body}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"connection error: {e.reason}", file=sys.stderr)
        sys.exit(1)

    # Prefer the provider's own charge (cache-aware, matches billing); fall back
    # to per-token pricing only when the provider returns no cost.
    cost = (
        provider_cost
        if provider_cost is not None
        else compute_cost(tin, tout, in_price, out_price)
    )

    # JSONL transcript of the turns, then the single claude-shaped result object
    # the runner ingests via extract_result.
    for ev in steps:
        print(json.dumps(ev))
    print(
        json.dumps(
            {
                "type": "result",
                "result": final_text,
                "usage": {"input_tokens": tin, "output_tokens": tout},
                "total_cost_usd": cost,
            }
        )
    )


if __name__ == "__main__":
    main()
