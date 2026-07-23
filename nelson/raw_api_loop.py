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
- ``NELSON_TEMPERATURE``  sampling temperature (default 0.1)
- ``NELSON_TOP_P`` / ``NELSON_MIN_P`` / ``NELSON_FREQUENCY_PENALTY`` /
  ``NELSON_PRESENCE_PENALTY`` / ``NELSON_REPEAT_PENALTY``  optional sampling knobs,
  injected into the request only when set
- ``NELSON_EXTRA_BODY``   optional JSON dict merged verbatim into each request
  (e.g. ``{"chat_template_kwargs": {"enable_thinking": false}}``)

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
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

SRC_ROOT = "/src"
MAX_TOOL_OUTPUT = 20_000  # chars; keep a single tool result from blowing context
# Date-pinned community ruleset baked into the `-semgrep` image (see runner.py).
# Scanned offline via `--config <dir>` so no newer rule is fetched at run time.
SEMGREP_RULES_DIR = os.environ.get("NELSON_SEMGREP_RULES", "/opt/semgrep-rules")
SEMGREP_WALL_TIMEOUT = 240  # whole-scan subprocess cap (per-rule cap is --timeout)
SHELL_WALL_TIMEOUT = 120  # per-command subprocess cap for the bash tool

# tree-sitter (read-grep-treesitter profile): map a file extension to the grammar
# name (and its baked-in per-language module). Covers the corpus languages.
TS_LANG_BY_EXT = {
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".go": "go",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".py": "python",
    ".rs": "rust",
    ".php": "php",
}
TS_MODULE_BY_LANG = {
    "c": "tree_sitter_c",
    "cpp": "tree_sitter_cpp",
    "go": "tree_sitter_go",
    "java": "tree_sitter_java",
    "javascript": "tree_sitter_javascript",
    "python": "tree_sitter_python",
    "rust": "tree_sitter_rust",
    "php": "tree_sitter_php",
}
# Node types that introduce a named definition, unioned across those grammars.
# Used to extract a file outline and to resolve where a symbol is defined.
TS_DEF_TYPES = frozenset(
    {
        "function_definition",
        "function_declaration",
        "function_item",
        "method_definition",
        "method_declaration",
        "constructor_declaration",
        "class_definition",
        "class_declaration",
        "class_specifier",
        "struct_specifier",
        "struct_item",
        "enum_specifier",
        "enum_item",
        "union_specifier",
        "interface_declaration",
        "trait_item",
        "impl_item",
        "trait_declaration",
        "enum_declaration",
        "type_spec",
        "type_alias_declaration",
        "module",
        "namespace_definition",
    }
)
TS_NAME_NODE_TYPES = frozenset(
    {"identifier", "field_identifier", "type_identifier", "name"}
)
TS_MAX_FILES = 40  # cap files parsed per symbol query
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

# Appended to the system prompt ONLY on a semgrep tool profile. Models otherwise
# default to read_file/grep and never discover the static-analysis tool, so the
# experiment would measure an unused tool. This tells the model the tool exists
# and may be used — it does NOT leak anything about the planted bug (no CVE, class,
# or location) — and stresses that a clean scan is not proof of safety so a model
# does not rubber-stamp a file semgrep happens not to flag. The control profile's
# prompt is unchanged (byte-identical to the baseline), so the pair stays clean.
SEMGREP_SYSTEM_NOTE = (
    " In addition to reading and grepping, you have a `semgrep` tool that runs a "
    "static analyzer over a file or directory under /src and returns rule matches "
    "with source->sink data-flow (taint) traces. Use it where it helps — to "
    "corroborate a suspicion or surface tainted data flows you might miss by "
    "reading alone — but treat its output as advisory: it has limited coverage, a "
    "clean result does NOT prove the file is safe, and you must still judge real "
    "exploitability yourself."
)

# Appended to the system prompt ONLY on a treesitter tool profile (same rationale
# as SEMGREP_SYSTEM_NOTE — models will not use a tool they were not told about). It
# describes structure-navigation tools and leaks nothing about the planted bug.
TREESITTER_SYSTEM_NOTE = (
    " In addition to reading and grepping, you have tree-sitter structure tools: "
    "`outline` gives a file's functions, methods, and type definitions with their "
    "signatures and line numbers; `symbol` resolves a name across /src to where it "
    "is defined (with signature) and the sites that reference it. Use them to "
    "navigate precisely and to see how the pieces fit together — to follow a "
    "callee's definition or find every caller — rather than relying on text search "
    "alone. They describe structure only; they do not judge security."
)

# Appended to the system prompt ONLY on a `shell` tool profile. The bash tool is a
# real shell, so this announces the toolbelt (models will not reach for a tool they
# were not told about) and the read-only/no-network sandbox shape. Like the other
# notes it leaks nothing about the planted bug, and — as with semgrep — it stresses
# that a clean command result is not proof of safety so the model does not
# rubber-stamp a file. The control profile's prompt stays byte-identical.
SHELL_SYSTEM_NOTE = (
    " In addition to reading and grepping, you have a `bash` tool that runs shell "
    "commands in the sandbox. The source tree is at /src (read-only); your home "
    "directory and /tmp are writable for scratch work. You have a full Unix "
    "toolbelt — coreutils, find, awk, sed, diff, file, jq, ctags, binutils "
    "(objdump/nm/strings/readelf), hexdump/od — and python3 (with tree-sitter) for "
    "writing and running small analysis scripts. There is no network access. Use "
    "the shell freely to navigate, search, cross-reference, and analyze the code "
    "however you find most effective; it complements the read and grep tools rather "
    "than replacing them. A clean result from any command does NOT prove the file "
    "is safe — you must still judge real exploitability yourself."
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

# The static-analysis tool, advertised ONLY to competitors on a tool profile whose
# name contains "semgrep" (see select_tools / NELSON_TOOL_PROFILE). It runs Semgrep
# over a path under /src with a fixed offline ruleset and returns taint/dataflow
# findings. The description tells the model what the tool can and cannot establish
# so a clean scan is not mistaken for proof of safety (which would tank recall).
SEMGREP_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "semgrep",
        "description": (
            "Run the Semgrep static analyzer over a file or directory under /src, "
            "using a fixed offline ruleset. Returns matches as: rule id, "
            "file:line, severity, message, and — where the rule is a taint rule — "
            "the source->sink dataflow trace. Useful for data-flow and "
            "known-pattern signal to corroborate or steer your own reading. "
            "IMPORTANT: Semgrep has limited rule coverage (especially for C/C++ "
            "memory safety); the ABSENCE of a finding does NOT mean the code is "
            "safe, and a finding still needs your judgement of real exploitability."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "file or directory under /src to scan",
                },
            },
            "required": ["path"],
        },
    },
}

# read-grep-semgrep profile = the three base tools plus semgrep.
SEMGREP_TOOLS: list[dict[str, Any]] = [*TOOLS, SEMGREP_TOOL]

# The shell tool, advertised ONLY to competitors on a tool profile whose name
# contains "shell" (see select_tools / NELSON_TOOL_PROFILE). Unlike the read tools
# it is NOT path-confined — a real shell is the whole point — so confinement is the
# container: /src is mounted read-only, the network is disabled, all capabilities
# are dropped, and the container is ephemeral. The experiment it serves asks whether
# a general toolbelt (the kind a tool-trained model is built to exploit) lifts
# detection over the read-only baseline.
SHELL_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Run a shell command in the sandbox. The source tree is at /src "
            "(read-only); your home directory and /tmp are writable for scratch "
            "work. A general toolbelt is available — coreutils, grep/ripgrep, find, "
            "awk, sed, diff, file, jq, ctags, binutils (objdump/nm/strings/"
            "readelf), hexdump/od — plus python3 (with tree-sitter). There is NO "
            "network access. Output is truncated, so prefer targeted commands. Use "
            "it to navigate, search, extract, and run small analysis scripts as you "
            "would in a terminal."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "shell command, run via `bash -c` with cwd /src",
                },
            },
            "required": ["command"],
        },
    },
}

# read-grep-shell profile = the three base tools plus a full bash shell.
SHELL_TOOLS: list[dict[str, Any]] = [*TOOLS, SHELL_TOOL]

# The tree-sitter tools, advertised ONLY to competitors on a profile whose name
# contains "treesitter". They give STRUCTURE (definitions, signatures, cross-file
# references) with no opinion about vulnerabilities — the bet is that knowing how
# the pieces fit together helps the model reason, without a scanner doing its job.
TREESITTER_TOOLS: list[dict[str, Any]] = [
    *TOOLS,
    {
        "type": "function",
        "function": {
            "name": "outline",
            "description": (
                "Structural outline of a source file under /src via tree-sitter: "
                "every function, method, class, struct, and type definition with "
                "its line number and signature. Use it to orient quickly in an "
                "unfamiliar file before reading."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "file under /src"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "symbol",
            "description": (
                "Resolve a symbol across /src via tree-sitter: where a function, "
                "method, type, struct, or class NAME is DEFINED (with signature and "
                "location) and the sites that REFERENCE it. Use it to see how the "
                "pieces fit together — to follow a callee's definition or find every "
                "caller of a function — more precisely than a text grep."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "symbol name"},
                },
                "required": ["name"],
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
    # `rg -n`), so numbers are consistent across tools). Without this the model
    # reads raw text and must hand-count to cite a location — which it does badly
    # (observed ~75-line drift), so a genuine find lands tens of lines off and the
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


def tool_bash(args: dict[str, Any], src_root: str | None = None) -> str:
    """Run a shell command in the sandbox (read-grep-shell profile).

    Deliberately NOT confined by ``_resolve_in_src`` — a real shell is the point.
    Confinement is the container itself (see runner.py): /src is mounted read-only,
    ``--network none`` blocks egress, ``--cap-drop ALL`` + ``no-new-privileges``
    strip privilege, memory/cpu/pids are capped, and the container is ephemeral.
    The command runs with cwd at the source root; scratch writes must target $HOME
    or /tmp (the /src mount rejects writes). Output is merged stdout+stderr,
    truncated to keep one result from blowing the context budget.
    """
    command = str(args.get("command", "")).strip()
    if not command:
        return "error: empty command"
    cwd = src_root or SRC_ROOT
    try:
        proc = subprocess.run(
            ["bash", "-c", command],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=SHELL_WALL_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return f"error: command timed out after {SHELL_WALL_TIMEOUT}s"
    except (OSError, subprocess.SubprocessError) as e:
        return f"error: {e}"
    out = proc.stdout or ""
    if proc.stderr:
        out += ("\n" if out else "") + "[stderr] " + proc.stderr
    if proc.returncode != 0:
        out = f"(exit {proc.returncode})\n" + out
    return (out or "(no output)")[:MAX_TOOL_OUTPUT]


def _format_semgrep(data: dict[str, Any], rel: str) -> str:
    """Render semgrep's JSON into a compact, context-budget-safe summary."""
    results = data.get("results") or []
    if not results:
        # Surface scan errors (e.g. an unparseable target) distinctly from a clean
        # scan so the model doesn't read a failed run as "nothing found".
        errs = data.get("errors") or []
        if errs:
            msgs = "; ".join(str(e.get("message", e))[:120] for e in errs[:3])
            return f"(semgrep: no findings; {len(errs)} scan error(s): {msgs})"
        return f"(semgrep: no findings under {rel})"
    out = [f"semgrep: {len(results)} finding(s) under {rel}"]
    for r in results:
        extra = r.get("extra") or {}
        loc = f"{r.get('path', '?')}:{(r.get('start') or {}).get('line', '?')}"
        sev = extra.get("severity", "")
        msg = " ".join(str(extra.get("message", "")).split())
        # Local-path configs prefix the rule id with the rules dir as dotted
        # segments (e.g. "opt.semgrep-rules.python..."); drop it for readability.
        check = str(r.get("check_id", "?")).removeprefix("opt.semgrep-rules.")
        out.append(f"\n- {loc} [{sev}] {check}\n  {msg[:300]}")
        # Taint rules carry a source->sink trace — the data-flow signal we want the
        # model to see. Render source and sink locations compactly when present.
        trace = extra.get("dataflow_trace") or {}
        src = trace.get("taint_source")
        sink = trace.get("taint_sink")

        def _loc(node: Any) -> str | None:
            # A trace node is typically ["Loc", [ {start:{line}, path}, ... ]].
            with contextlib.suppress(Exception):
                cell = node[1][0]
                line = cell.get("start", {}).get("line", "?")
                return f"{cell.get('path', '?')}:{line}"
            return None

        s_loc, k_loc = (_loc(src) if src else None), (_loc(sink) if sink else None)
        if s_loc or k_loc:
            out.append(f"  dataflow: source {s_loc or '?'} -> sink {k_loc or '?'}")
    return "\n".join(out)[:MAX_TOOL_OUTPUT]


def tool_semgrep(args: dict[str, Any], src_root: str | None = None) -> str:
    rel = str(args.get("path", "."))
    real = _resolve_in_src(rel, src_root)
    if real is None:
        return f"error: path {rel!r} is outside the source tree"
    if not os.path.exists(real):
        return f"error: no such path: {rel}"
    try:
        proc = subprocess.run(
            [
                "semgrep",
                "scan",
                "--config",
                SEMGREP_RULES_DIR,
                "--json",
                "--quiet",
                "--metrics",
                "off",
                "--disable-version-check",
                "--dataflow-traces",
                "--timeout",
                "30",
                "--max-target-bytes",
                "2000000",
                real,
            ],
            capture_output=True,
            text=True,
            timeout=SEMGREP_WALL_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return f"error: semgrep timed out after {SEMGREP_WALL_TIMEOUT}s on {rel}"
    except (OSError, subprocess.SubprocessError) as e:
        return f"error: {e}"
    if not proc.stdout.strip():
        return f"error: semgrep produced no output; stderr: {proc.stderr[:500]}"
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return f"error: semgrep output unparseable; stderr: {proc.stderr[:500]}"
    return _format_semgrep(data, rel)


# -- tree-sitter tools (structure/AST) ---------------------------------------


def _ts_lang_for(path: str) -> str | None:
    return TS_LANG_BY_EXT.get(os.path.splitext(path)[1].lower())


def _ts_parser(lang: str) -> Any:
    # Lazy import: the grammars live only in the container image, and only the
    # treesitter profile ever calls these tools — the module stays importable (and
    # the other profiles stdlib-clean) on a host without the packages.
    import importlib

    from tree_sitter import Language, Parser  # ty: ignore[unresolved-import]

    module = importlib.import_module(TS_MODULE_BY_LANG[lang])
    # Most grammar packages expose `language()`; a few (php, typescript) name it
    # per-dialect, e.g. `language_php()`.
    lang_fn = getattr(module, "language", None) or getattr(module, f"language_{lang}")
    language = Language(lang_fn())
    try:
        return Parser(language)
    except TypeError:  # older binding: language set via attribute
        parser = Parser()
        parser.language = language
        return parser


def _ts_walk(node: Any):
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.children)


def _ts_def_name(node: Any) -> str:
    """Best-effort name of a definition node across grammars."""
    named = node.child_by_field_name("name")
    if named is not None:
        return _ts_text(named)
    # C/C++ functions nest the name inside (possibly several) declarators; descend
    # the `declarator` field until we reach the identifier itself.
    decl = node.child_by_field_name("declarator")
    seen = 0
    while decl is not None and seen < 8:
        if decl.type in TS_NAME_NODE_TYPES:
            return _ts_text(decl)
        inner = decl.child_by_field_name("declarator")
        if inner is None:
            for c in decl.children:
                if c.type in TS_NAME_NODE_TYPES:
                    return _ts_text(c)
            break
        decl = inner
        seen += 1
    for c in node.children:  # shallow fallback
        if c.type in TS_NAME_NODE_TYPES:
            return _ts_text(c)
    return "?"


def _ts_has_body(node: Any) -> bool:
    """True if a struct/union/enum specifier actually defines a body (vs a type ref)."""
    return any(
        c.type in ("field_declaration_list", "enumerator_list", "declaration_list")
        for c in node.children
    )


def _ts_text(node: Any) -> str:
    return node.text.decode("utf-8", "replace")


def _ts_signature(node: Any) -> str:
    """The declaration header — everything up to the body — collapsed to one line.

    Cutting at the opening brace yields a full C/Go/Java signature ("int foo(char *p)")
    even when C splits the return type onto its own line; brace-less languages
    (Python) fall back to the first line ("def foo(a):").
    """
    txt = _ts_text(node)
    head = txt.split("{", 1)[0] if "{" in txt else txt.split("\n", 1)[0]
    return " ".join(head.split())[:160]


def tool_outline(args: dict[str, Any], src_root: str | None = None) -> str:
    rel = str(args.get("path", ""))
    real = _resolve_in_src(rel, src_root)
    if real is None:
        return f"error: path {rel!r} is outside the source tree"
    if not os.path.isfile(real):
        return f"error: not a file: {rel}"
    lang = _ts_lang_for(real)
    if lang is None:
        return f"error: unsupported language for {rel}"
    try:
        with open(real, "rb") as f:
            data = f.read()
        tree = _ts_parser(lang).parse(data)
    except Exception as e:  # parser/import failure must not kill the loop
        return f"error: {e}"
    rows = []
    for n in _ts_walk(tree.root_node):
        if n.type not in TS_DEF_TYPES:
            continue
        if n.type.endswith("_specifier") and not _ts_has_body(n):
            continue  # a `struct foo *` type reference, not a definition
        line = n.start_point[0] + 1
        entry = f"L{line} {n.type} {_ts_def_name(n)}: {_ts_signature(n)[:120]}"
        rows.append((line, entry))
    if not rows:
        return f"(no definitions found in {rel})"
    rows.sort()
    return "\n".join(r[1] for r in rows)[:MAX_TOOL_OUTPUT]


def tool_symbol(args: dict[str, Any], src_root: str | None = None) -> str:
    name = str(args.get("name", ""))
    if not name:
        return "error: empty name"
    root = src_root or SRC_ROOT
    try:
        proc = subprocess.run(
            ["rg", "-l", "--no-messages", "-F", "--", name, root],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return f"error: {e}"
    files = [f for f in proc.stdout.splitlines() if f][:TS_MAX_FILES]
    defs: list[str] = []
    refs: list[str] = []
    for path in files:
        lang = _ts_lang_for(path)
        if lang is None:
            continue
        try:
            with open(path, "rb") as f:
                data = f.read()
            tree = _ts_parser(lang).parse(data)
        except Exception:  # noqa: S112 - an unparseable file is skipped, not fatal
            continue
        relp = os.path.relpath(path, root)
        for n in _ts_walk(tree.root_node):
            ln = n.start_point[0] + 1
            if n.type in TS_DEF_TYPES and _ts_def_name(n) == name:
                if n.type.endswith("_specifier") and not _ts_has_body(n):
                    continue
                defs.append(f"{relp}:{ln} {n.type}: {_ts_signature(n)[:140]}")
            elif n.type in TS_NAME_NODE_TYPES and _ts_text(n) == name:
                refs.append(f"{relp}:{ln}")
        if len(defs) >= 50:
            break
    out = []
    if defs:
        out.append(f"definitions of {name!r} ({len(defs)}):")
        out.extend("  " + d for d in defs[:50])
    else:
        out.append(f"no definitions of {name!r} found (may be external or a variable)")
    if refs:
        seen: set[str] = set()
        uniq = [r for r in refs if not (r in seen or seen.add(r))]
        out.append(f"references ({len(uniq)}, up to 60 shown):")
        out.append("  " + ", ".join(uniq[:60]))
    return "\n".join(out)[:MAX_TOOL_OUTPUT]


TOOL_FUNCS: dict[str, Callable[[dict[str, Any], str | None], str]] = {
    "read_file": tool_read_file,
    "grep": tool_grep,
    "list_dir": tool_list_dir,
    "semgrep": tool_semgrep,
    "outline": tool_outline,
    "symbol": tool_symbol,
    "bash": tool_bash,
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


# Tool calls a model emitted as plain TEXT instead of the structured ``tool_calls``
# field. DeepSeek v4 in particular falls back mid-loop to its native XML format —
# an <invoke name="read_file"> block whose tag is prefixed with DeepSeek's DSML
# marker — which the OpenAI-shaped client never sees in ``message.tool_calls``.
# Without this the loop reads that turn as the final answer and stops while the
# model was still trying to navigate. The prefix before ``invoke``/``parameter``
# varies by model (DeepSeek's DSML marker, Anthropic's ``antml:``) so it is matched
# loosely; only the ``invoke name="..."`` / ``parameter name="..."`` shape matters.
_NATIVE_INVOKE_RE = re.compile(
    r'<[^>]*?invoke\s+name="([^"]+)"\s*>(.*?)</[^>]*?invoke\s*>', re.DOTALL
)
_NATIVE_PARAM_RE = re.compile(
    r'<[^>]*?parameter\s+name="([^"]+)"[^>]*?>(.*?)</[^>]*?parameter\s*>', re.DOTALL
)


def _coerce_native_param(val: str) -> Any:
    """Best-effort type for a text-XML parameter (line numbers want ints; paths and
    patterns stay strings). Falls back to the raw string when it is not numeric."""
    val = val.strip()
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        return val


def _assistant_turn(msg: dict[str, Any]) -> dict[str, Any]:
    """A minimal, replay-safe assistant message from a model response.

    Reasoning models (e.g. Laguna, DeepSeek-R1) return a ``reasoning_content``
    (and/or ``reasoning``) field on the assistant message. Echoing that field back
    into the next request's ``messages`` makes some provider backends reject the
    call (OpenRouter returns HTTP 400 ``duplicate field 'reasoning_content'``) —
    reasoning traces are model *output*, not conversation input to be resent. So we
    rebuild the turn with only the fields a chat endpoint expects to receive:
    ``role``, ``content``, and ``tool_calls`` when present. Non-reasoning models are
    unaffected (they carry no reasoning field to drop)."""
turn: dict[str, Any] = {"role": "assistant", "content": msg.get("content")}
tool_calls = msg.get("tool_calls")
if tool_calls:
    turn["tool_calls"] = tool_calls
return turn


def parse_native_tool_calls(content: str) -> list[dict[str, Any]]:
    """Extract ``[{"name", "args"}]`` from tool calls a model wrote as plain-text
    XML (see ``_NATIVE_INVOKE_RE``). Empty list when there are none — so the caller
    only treats a turn as native-tool-use when something actually parses."""
    if not content or "invoke" not in content:
        return []
    calls: list[dict[str, Any]] = []
    for m in _NATIVE_INVOKE_RE.finditer(content):
        args = {
            pm.group(1).strip(): _coerce_native_param(pm.group(2))
            for pm in _NATIVE_PARAM_RE.finditer(m.group(2))
        }
        calls.append({"name": m.group(1).strip(), "args": args})
    return calls


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
    extra_sampling: dict[str, Any] | None = None,
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
        # Optional anti-repetition / nucleus knobs (frequency_penalty, min_p,
        # repeat_penalty, …). Only present when the competitor's cost_model sets
        # them, so strict OpenAI-compat endpoints never see an unknown field.
        if extra_sampling:
            payload.update(extra_sampling)
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
        # Fallback: a model (DeepSeek) that wrote tool calls as plain-text XML rather
        # than in the structured field. Only consulted when there are no structured
        # calls and we still want tool use, so normal turns are untouched.
        native_calls = (
            parse_native_tool_calls(msg.get("content") or "")
            if (not tool_calls and not force_final)
            else []
        )
        steps.append(
            {
                "type": "step",
                "step": step,
                "content": _trim(msg.get("content")),
                "tool_calls": [
                    (tc.get("function") or {}).get("name") for tc in tool_calls
                ]
                or [f"{c['name']}(native)" for c in native_calls],
            }
        )

        if tool_calls and not force_final:
            # Rebuild the turn (drop reasoning_content etc.) so replaying it to the
            # provider is accepted; keep the tool_calls the loop is about to honour.
            messages.append(_assistant_turn(msg))
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

        if native_calls:
            # Echo the assistant turn verbatim, then feed results back as a plain
            # user turn — no tool_call_id to honour (the model never sent one), so
            # this stays valid for any OpenAI-compatible endpoint.
            messages.append({"role": "assistant", "content": msg.get("content") or ""})
            results = [
                f"Tool {c['name']} result:\n"
                + dispatch_tool(c["name"], c["args"], src_root)[:MAX_TOOL_OUTPUT]
                for c in native_calls
            ]
            messages.append({"role": "user", "content": "\n\n".join(results)})
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


def select_tools() -> list[dict[str, Any]]:
    """Pick the advertised toolset from the competitor's tool profile.

    The profile string (set as NELSON_TOOL_PROFILE by the runtime) finally drives
    behaviour rather than being a passive DB label: a name containing "semgrep"
    gets the static-analysis tool on top of the base three; everything else gets
    exactly the original read-grep set, so existing competitors are unchanged.
    """
    profile = os.environ.get("NELSON_TOOL_PROFILE", "")
    if "semgrep" in profile:
        return SEMGREP_TOOLS
    if "treesitter" in profile:
        return TREESITTER_TOOLS
    if "shell" in profile:
        return SHELL_TOOLS
    return TOOLS


def select_system_prompt() -> str:
    """System prompt for the active profile: base, plus a tool note if applicable."""
    profile = os.environ.get("NELSON_TOOL_PROFILE", "")
    if "semgrep" in profile:
        return SYSTEM_PROMPT + SEMGREP_SYSTEM_NOTE
    if "treesitter" in profile:
        return SYSTEM_PROMPT + TREESITTER_SYSTEM_NOTE
    if "shell" in profile:
        return SYSTEM_PROMPT + SHELL_SYSTEM_NOTE
    return SYSTEM_PROMPT


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
    # Optional sampling knobs — each injected into the request only when set on the
    # competitor's cost_model. Used to tame the repetition-spiral timeout some local
    # models (e.g. Gemma 4) fall into: frequency/presence/repeat penalties discourage
    # the degenerate loop, and top_p/min_p tighten the nucleus.
    extra_sampling: dict[str, Any] = {}
    for env_key, field in (
        ("NELSON_TOP_P", "top_p"),
        ("NELSON_MIN_P", "min_p"),
        ("NELSON_FREQUENCY_PENALTY", "frequency_penalty"),
        ("NELSON_PRESENCE_PENALTY", "presence_penalty"),
        ("NELSON_REPEAT_PENALTY", "repeat_penalty"),
    ):
        val = _float_env(env_key)
        if val is not None:
            extra_sampling[field] = val
    # Arbitrary structured request fields (e.g. ``chat_template_kwargs`` to disable a
    # model's runaway "thinking" phase) that don't fit the float knobs above. Merged
    # verbatim into every request payload; set from the competitor's cost_model.
    raw_extra = os.environ.get("NELSON_EXTRA_BODY")
    if raw_extra:
        try:
            parsed = json.loads(raw_extra)
            if isinstance(parsed, dict):
                extra_sampling.update(parsed)
        except (ValueError, TypeError):
            print("ignoring malformed NELSON_EXTRA_BODY", file=sys.stderr)
    try:
        final_text, tin, tout, steps, provider_cost = run_loop(
            prompt,
            base_url=base_url,
            model=model,
            api_key=api_key,
            max_steps=_int_env("NELSON_MAX_STEPS", 40),
            token_budget=_int_env("NELSON_TOKEN_BUDGET", 500_000),
            temperature=0.1 if temperature is None else temperature,
            extra_sampling=extra_sampling or None,
            tools=select_tools(),
            system_prompt=select_system_prompt(),
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
