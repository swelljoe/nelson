"""The in-container ReAct agent: tool sandboxing, the loop, and cost — no live API.

The HTTP call is injected (``run_loop(post=...)``) so the loop is exercised
against scripted provider responses. The sandbox tests are the security-critical
ones: a tool must never read outside the source root, even via ``..`` or a symlink.
"""

import json
import urllib.error
from email.message import Message

import pytest

import nelson.raw_api_loop as ral
from nelson.raw_api_loop import (
    MAX_HTTP_RETRIES,
    SEMGREP_TOOLS,
    SHELL_TOOLS,
    TOOLS,
    TREESITTER_TOOLS,
    _format_semgrep,
    _post_chat,
    _resolve_in_src,
    _ts_lang_for,
    compute_cost,
    dispatch_tool,
    run_loop,
    select_system_prompt,
    select_tools,
    tool_bash,
    tool_grep,
    tool_list_dir,
    tool_outline,
    tool_read_file,
    tool_semgrep,
    tool_symbol,
    usage_delta,
)

# -- Scripted-response helpers ----------------------------------------------


def _post_returning(*responses):
    it = iter(responses)

    def post(url, payload, api_key):
        return next(it)

    return post


def _tool_call(name, args, usage=None):
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }
                    ],
                }
            }
        ],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5},
    }


def _final(content, usage=None):
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": usage or {"prompt_tokens": 8, "completion_tokens": 3},
    }


# -- Loop --------------------------------------------------------------------


def test_loop_executes_tool_call_then_emits_final(tmp_path):
    (tmp_path / "a.c").write_text("int main() { return 0; }\n")
    answer = '[{"file": "a.c", "line": 1, "confidence": "low"}]'
    post = _post_returning(
        _tool_call("read_file", {"path": "a.c"}),
        _final(answer),
    )
    final_text, tin, tout, steps, cost = run_loop(
        "audit a.c",
        base_url="https://x/v1",
        model="m",
        api_key="k",
        post=post,
        src_root=str(tmp_path),
    )
    assert final_text == answer
    assert (tin, tout) == (10 + 8, 5 + 3)  # summed across both turns
    assert len(steps) == 2
    assert cost is None  # no provider cost field on a non-OpenRouter endpoint


def test_loop_max_steps_terminates(tmp_path):
    # A model that never stops calling tools must still terminate and return.
    post = _post_returning(*[_tool_call("list_dir", {"path": "."}) for _ in range(5)])
    final_text, _tin, _tout, steps, _cost = run_loop(
        "audit",
        base_url="https://x/v1",
        model="m",
        api_key="k",
        max_steps=3,
        post=post,
        src_root=str(tmp_path),
    )
    assert len(steps) == 3  # capped, did not exhaust the 5 scripted responses
    assert final_text == ""  # forced-final turn carried no answer text


def test_loop_no_final_array_falls_back_to_last_text(tmp_path):
    post = _post_returning(_final("I could not find anything exploitable."))
    final_text, _tin, _tout, _steps, _cost = run_loop(
        "audit",
        base_url="https://x/v1",
        model="m",
        api_key="k",
        post=post,
        src_root=str(tmp_path),
    )
    assert final_text == "I could not find anything exploitable."


# -- Native (plain-text) tool calls, e.g. DeepSeek DSML ----------------------

_DSML = "｜｜DSML｜｜"  # noqa: RUF001  DeepSeek's native tag prefix (fullwidth bars)


def _native_call(name, args):
    """An assistant turn that wrote a tool call as plain-text XML (DeepSeek DSML),
    with NO structured ``tool_calls`` field — the case the fallback must rescue."""
    params = "".join(
        f'<{_DSML}parameter name="{k}">{v}</{_DSML}parameter>' for k, v in args.items()
    )
    content = f'<{_DSML}invoke name="{name}">{params}</{_DSML}invoke>'
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def test_parse_native_tool_calls_reads_dsml_and_ignores_plain_text():
    from nelson.raw_api_loop import parse_native_tool_calls

    content = (
        f'<{_DSML}invoke name="read_file">'
        f'<{_DSML}parameter name="path">/src/a.c</{_DSML}parameter>'
        f'<{_DSML}parameter name="start_line">5</{_DSML}parameter>'
        f"</{_DSML}invoke>"
    )
    assert parse_native_tool_calls(content) == [
        {"name": "read_file", "args": {"path": "/src/a.c", "start_line": 5}}
    ]
    # A normal final answer (JSON array) is not a tool call.
    assert parse_native_tool_calls('[{"file": "a.c", "line": 1}]') == []
    assert parse_native_tool_calls("") == []


def test_loop_executes_native_text_tool_call_then_continues(tmp_path):
    # DeepSeek falls back to DSML mid-loop: the loop must run the tool and keep
    # going, not mistake the DSML turn for the final answer.
    (tmp_path / "a.c").write_text("int main() { return 0; }\n")
    answer = '[{"file": "a.c", "line": 1, "confidence": "low"}]'
    post = _post_returning(
        _native_call("read_file", {"path": "a.c"}),
        _final(answer),
    )
    final_text, _tin, _tout, steps, _cost = run_loop(
        "audit a.c",
        base_url="https://api.deepseek.com",
        model="m",
        api_key="k",
        post=post,
        src_root=str(tmp_path),
    )
    assert final_text == answer  # reached the real answer, did not stop at the DSML
    assert len(steps) == 2
    assert steps[0]["tool_calls"] == ["read_file(native)"]


def test_loop_sums_provider_cost_and_requests_it_on_openrouter(tmp_path):
    # On an OpenRouter endpoint the loop must send usage.include and sum the
    # real per-call cost (cache-aware), which main() prefers over compute_cost.
    (tmp_path / "a.c").write_text("int x;\n")
    seen_payloads = []

    def post(url, payload, api_key):
        seen_payloads.append(payload)
        if len(seen_payloads) == 1:
            r = _tool_call("read_file", {"path": "a.c"})
            r["usage"]["cost"] = 0.004  # provider's real charge for this call
            return r
        r = _final("[]")
        r["usage"]["cost"] = 0.001
        return r

    _text, _tin, _tout, _steps, cost = run_loop(
        "audit a.c",
        base_url="https://openrouter.ai/api/v1",
        model="m",
        api_key="k",
        post=post,
        src_root=str(tmp_path),
    )
    assert cost == pytest.approx(0.005)  # summed across both turns
    assert all(p.get("usage") == {"include": True} for p in seen_payloads)


def test_loop_does_not_request_usage_on_non_openrouter(tmp_path):
    seen = []

    def post(url, payload, api_key):
        seen.append(payload)
        return _final("[]")

    run_loop(
        "audit",
        base_url="https://api.self-hosted.invalid/v1",  # not OpenRouter
        model="m",
        api_key="k",
        post=post,
        src_root=str(tmp_path),
    )
    assert all("usage" not in p for p in seen)  # no unknown field to strict servers


def test_loop_defaults_to_low_temperature(tmp_path):
    seen = []

    def post(url, payload, api_key):
        seen.append(payload)
        return _final("[]")

    run_loop(
        "audit", base_url="x", model="m", api_key="k", post=post, src_root=str(tmp_path)
    )
    assert all(p["temperature"] == 0.1 for p in seen)  # deterministic default


def test_loop_honours_explicit_temperature(tmp_path):
    seen = []

    def post(url, payload, api_key):
        seen.append(payload)
        return _final("[]")

    run_loop(
        "audit",
        base_url="x",
        model="m",
        api_key="k",
        temperature=0.6,
        post=post,
        src_root=str(tmp_path),
    )
    assert all(p["temperature"] == 0.6 for p in seen)  # repeat-trial diversity


def test_loop_overrides_system_prompt_and_tools(tmp_path):
    # Host callers (scan-mode adapter) supply their own framing + tool schema
    # while reusing the loop; the benchmark defaults must not leak in.
    (tmp_path / "a.c").write_text("int x;\n")
    seen = []

    def post(url, payload, api_key):
        seen.append(payload)
        return _final("[]")

    custom_tools = [
        {"type": "function", "function": {"name": "read_file", "parameters": {}}}
    ]
    run_loop(
        "audit",
        base_url="x",
        model="m",
        api_key="k",
        post=post,
        src_root=str(tmp_path),
        system_prompt="CUSTOM SYSTEM",
        tools=custom_tools,
    )
    assert seen[0]["messages"][0] == {"role": "system", "content": "CUSTOM SYSTEM"}
    assert seen[0]["tools"] == custom_tools


def test_loop_defaults_keep_benchmark_framing(tmp_path):
    # No override → the container audit framing over /src, byte-for-byte.
    seen = []

    def post(url, payload, api_key):
        seen.append(payload)
        return _final("[]")

    run_loop(
        "audit", base_url="x", model="m", api_key="k", post=post, src_root=str(tmp_path)
    )
    assert seen[0]["messages"][0]["content"] == ral.SYSTEM_PROMPT
    assert seen[0]["tools"] == ral.TOOLS


# -- Sandbox (security-critical) --------------------------------------------


def test_resolve_in_src_allows_file_under_root(tmp_path):
    (tmp_path / "a.c").write_text("x")
    resolved = _resolve_in_src("a.c", str(tmp_path))
    assert resolved is not None and resolved.endswith("a.c")


def test_resolve_in_src_rejects_parent_traversal(tmp_path):
    assert _resolve_in_src("../etc/passwd", str(tmp_path)) is None


def test_resolve_in_src_rejects_absolute_escape(tmp_path):
    assert _resolve_in_src("/etc/passwd", str(tmp_path)) is None


def test_resolve_in_src_rejects_symlink_escape(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("s")
    src = tmp_path / "src"
    src.mkdir()
    (src / "link").symlink_to(outside)
    assert _resolve_in_src("link/secret", str(src)) is None


def test_read_file_rejects_escape(tmp_path):
    assert "outside" in tool_read_file({"path": "../x"}, str(tmp_path))


def test_grep_rejects_escape(tmp_path):
    assert "outside" in tool_grep({"pattern": "x", "path": "../"}, str(tmp_path))


def test_list_dir_rejects_escape(tmp_path):
    assert "outside" in tool_list_dir({"path": "/etc"}, str(tmp_path))


def test_read_file_line_range(tmp_path):
    (tmp_path / "f.txt").write_text("l1\nl2\nl3\nl4\n")
    out = tool_read_file(
        {"path": "f.txt", "start_line": 2, "end_line": 3}, str(tmp_path)
    )
    # Absolute line numbers are prefixed so the model can cite exact locations.
    assert out == "2:l2\n3:l3\n"


def test_read_file_numbers_from_one(tmp_path):
    (tmp_path / "f.txt").write_text("a\nb\nc\n")
    assert tool_read_file({"path": "f.txt"}, str(tmp_path)) == "1:a\n2:b\n3:c\n"


def test_list_dir_lists_entries(tmp_path):
    (tmp_path / "a").write_text("x")
    (tmp_path / "b").write_text("y")
    assert tool_list_dir({"path": "."}, str(tmp_path)) == "a\nb"


def test_dispatch_unknown_tool_is_error():
    assert "unknown tool" in dispatch_tool("nope", {}, None)


# -- Semgrep tool + profile selection ---------------------------------------


def test_select_tools_default_is_read_grep(monkeypatch):
    monkeypatch.delenv("NELSON_TOOL_PROFILE", raising=False)
    assert select_tools() == TOOLS
    monkeypatch.setenv("NELSON_TOOL_PROFILE", "read-grep")
    assert select_tools() == TOOLS


def test_select_tools_semgrep_profile_adds_semgrep(monkeypatch):
    monkeypatch.setenv("NELSON_TOOL_PROFILE", "read-grep-semgrep")
    tools = select_tools()
    assert tools == SEMGREP_TOOLS
    names = {t["function"]["name"] for t in tools}
    assert "semgrep" in names and {"read_file", "grep", "list_dir"} <= names


def test_select_system_prompt_control_is_baseline(monkeypatch):
    # Control profile must be byte-identical to the baseline system prompt.
    monkeypatch.setenv("NELSON_TOOL_PROFILE", "read-grep")
    assert select_system_prompt() == ral.SYSTEM_PROMPT
    monkeypatch.delenv("NELSON_TOOL_PROFILE", raising=False)
    assert select_system_prompt() == ral.SYSTEM_PROMPT


def test_select_system_prompt_semgrep_adds_note(monkeypatch):
    monkeypatch.setenv("NELSON_TOOL_PROFILE", "read-grep-semgrep")
    prompt = select_system_prompt()
    assert prompt.startswith(ral.SYSTEM_PROMPT)
    assert "semgrep" in prompt and "does NOT prove the file is safe" in prompt


def test_semgrep_rejects_escape(tmp_path):
    # Same sandbox guarantee as the other tools — never scan outside /src.
    assert "outside" in tool_semgrep({"path": "../etc"}, str(tmp_path))


def test_semgrep_dispatches_and_parses(tmp_path, monkeypatch):
    # No real semgrep binary needed: inject a scripted CompletedProcess so the
    # JSON-parse + render path is what's under test.
    (tmp_path / "q.py").write_text("x = 1\n")
    payload = {
        "results": [
            {
                "check_id": "python.lang.security.sqli",
                "path": "q.py",
                "start": {"line": 12},
                "extra": {
                    "severity": "ERROR",
                    "message": "User input flows into SQL query.",
                    "dataflow_trace": {
                        "taint_source": [
                            "Loc",
                            [{"path": "q.py", "start": {"line": 3}}],
                        ],
                        "taint_sink": [
                            "Loc",
                            [{"path": "q.py", "start": {"line": 12}}],
                        ],
                    },
                },
            }
        ]
    }

    class _Proc:
        stdout = json.dumps(payload)
        stderr = ""

    monkeypatch.setattr(ral.subprocess, "run", lambda *a, **k: _Proc())
    out = tool_semgrep({"path": "q.py"}, str(tmp_path))
    assert "1 finding" in out
    assert "q.py:12" in out and "python.lang.security.sqli" in out
    assert "dataflow: source q.py:3 -> sink q.py:12" in out


# -- shell tool + profile selection -----------------------------------------


def test_select_tools_shell_profile_adds_bash(monkeypatch):
    monkeypatch.setenv("NELSON_TOOL_PROFILE", "read-grep-shell")
    tools = select_tools()
    assert tools == SHELL_TOOLS
    names = {t["function"]["name"] for t in tools}
    assert "bash" in names and {"read_file", "grep", "list_dir"} <= names
    assert "semgrep" not in names  # profiles don't bleed into each other


def test_select_system_prompt_shell_adds_note(monkeypatch):
    monkeypatch.setenv("NELSON_TOOL_PROFILE", "read-grep-shell")
    prompt = select_system_prompt()
    assert prompt.startswith(ral.SYSTEM_PROMPT)
    assert "bash" in prompt and "no network access" in prompt.lower()


def test_bash_runs_command_with_cwd_at_src(tmp_path):
    (tmp_path / "hello.c").write_text("int main(){return 0;}\n")
    out = tool_bash({"command": "ls"}, str(tmp_path))
    assert "hello.c" in out


def test_bash_empty_command_is_error():
    assert "empty command" in tool_bash({"command": "  "}, None)


def test_bash_reports_nonzero_exit_and_stderr(tmp_path):
    out = tool_bash({"command": "echo oops >&2; exit 3"}, str(tmp_path))
    assert "(exit 3)" in out and "oops" in out


def test_bash_truncates_large_output(tmp_path):
    out = tool_bash({"command": "yes x | head -n 100000"}, str(tmp_path))
    assert len(out) <= ral.MAX_TOOL_OUTPUT


def test_bash_timeout_is_caught(tmp_path, monkeypatch):
    def _raise(*a, **k):
        raise ral.subprocess.TimeoutExpired(cmd="bash", timeout=ral.SHELL_WALL_TIMEOUT)

    monkeypatch.setattr(ral.subprocess, "run", _raise)
    out = tool_bash({"command": "sleep 999"}, str(tmp_path))
    assert "timed out" in out


# -- tree-sitter tools + profile selection ----------------------------------


def test_select_tools_treesitter_profile_adds_structure_tools(monkeypatch):
    monkeypatch.setenv("NELSON_TOOL_PROFILE", "read-grep-treesitter")
    tools = select_tools()
    assert tools == TREESITTER_TOOLS
    names = {t["function"]["name"] for t in tools}
    assert {"outline", "symbol"} <= names
    assert {"read_file", "grep", "list_dir"} <= names
    assert "semgrep" not in names  # profiles don't bleed into each other


def test_select_system_prompt_treesitter_adds_note(monkeypatch):
    monkeypatch.setenv("NELSON_TOOL_PROFILE", "read-grep-treesitter")
    prompt = select_system_prompt()
    assert prompt.startswith(ral.SYSTEM_PROMPT)
    assert "tree-sitter" in prompt and "outline" in prompt and "symbol" in prompt


def test_ts_lang_for_maps_extensions():
    assert _ts_lang_for("a/b/foo.c") == "c"
    assert _ts_lang_for("Foo.java") == "java"
    assert _ts_lang_for("x.go") == "go"
    assert _ts_lang_for("y.py") == "python"
    assert _ts_lang_for("README.md") is None


def test_outline_rejects_escape(tmp_path):
    assert "outside" in tool_outline({"path": "../etc/passwd"}, str(tmp_path))


def test_outline_unsupported_language(tmp_path):
    (tmp_path / "notes.txt").write_text("hello")
    assert "unsupported" in tool_outline({"path": "notes.txt"}, str(tmp_path))


def test_symbol_empty_name_is_error(tmp_path):
    assert "empty name" in tool_symbol({"name": ""}, str(tmp_path))


def test_format_semgrep_no_findings_is_distinct_from_error():
    assert "no findings" in _format_semgrep({"results": []}, "f.c")
    # Scan errors must not read as a clean "nothing found".
    scan = {"results": [], "errors": [{"message": "parse fail"}]}
    errd = _format_semgrep(scan, "f.c")
    assert "error" in errd and "parse fail" in errd


# -- Cost --------------------------------------------------------------------


def test_compute_cost_from_prices():
    # 1M input @ $0.14 + 0.5M output @ $0.28 = 0.14 + 0.14 = 0.28
    assert compute_cost(1_000_000, 500_000, 0.14, 0.28) == 0.28


def test_compute_cost_none_without_prices():
    assert compute_cost(1000, 200, None, None) is None


def test_usage_delta_counts_thinking_tokens_in_output():
    # A reasoning model (e.g. Gemini compat): completion_tokens is the visible
    # output only; total_tokens additionally carries billed thinking tokens.
    # Output must be total - prompt (1929 + 189 thinking + 7 visible = 2125).
    assert usage_delta(
        {"prompt_tokens": 1929, "completion_tokens": 7, "total_tokens": 2125}
    ) == (1929, 196)


def test_usage_delta_falls_back_to_completion_without_total():
    # Non-thinking provider, or no total_tokens: output = completion_tokens.
    assert usage_delta({"prompt_tokens": 100, "completion_tokens": 40}) == (100, 40)
    assert usage_delta(
        {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140}
    ) == (100, 40)
    assert usage_delta({}) == (0, 0)


# -- _post_chat retry / backoff ----------------------------------------------
#
# Transient faults (provider 429 tokens/min, a self-hosted endpoint dropping the
# socket mid-response) must be retried with backoff, not fail the whole run. The
# real urlopen is monkeypatched; sleep is injected so no test actually waits.


class _FakeResp:
    def __init__(self, body: str):
        self._body = body.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def read(self):
        return self._body


def _http_error(code: int, retry_after=None) -> urllib.error.HTTPError:
    hdrs = Message()
    if retry_after is not None:
        hdrs["Retry-After"] = str(retry_after)
    return urllib.error.HTTPError(
        "http://x/v1/chat/completions", code, "err", hdrs, None
    )


def _scripted_urlopen(events):
    """Return a fake urlopen yielding each event: raise Exceptions, return responses."""
    it = iter(events)

    def fake(_req, timeout=None):
        ev = next(it)
        if isinstance(ev, Exception):
            raise ev
        return ev

    return fake


def test_post_chat_retries_on_429_then_succeeds(monkeypatch):
    slept = []
    monkeypatch.setattr(
        ral.urllib.request,
        "urlopen",
        _scripted_urlopen(
            [_http_error(429), _http_error(429), _FakeResp('{"ok": true}')]
        ),
    )
    out = _post_chat("http://x", {"m": 1}, "k", sleep=slept.append)
    assert out == {"ok": True}
    assert len(slept) == 2  # backed off before each of the two retries


def test_post_chat_honors_retry_after_header(monkeypatch):
    slept = []
    monkeypatch.setattr(
        ral.urllib.request,
        "urlopen",
        _scripted_urlopen([_http_error(429, retry_after=7), _FakeResp('{"ok": 1}')]),
    )
    _post_chat("http://x", {}, "k", sleep=slept.append)
    assert slept == [7.0]  # honored the server's Retry-After, not the backoff curve


def test_post_chat_retries_on_transport_error(monkeypatch):
    slept = []
    monkeypatch.setattr(
        ral.urllib.request,
        "urlopen",
        _scripted_urlopen(
            [urllib.error.URLError("connection reset"), _FakeResp('{"ok": 1}')]
        ),
    )
    out = _post_chat("http://x", {}, "k", sleep=slept.append)
    assert out == {"ok": 1}
    assert len(slept) == 1


def test_post_chat_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(
        ral.urllib.request,
        "urlopen",
        _scripted_urlopen([_http_error(429)] * (MAX_HTTP_RETRIES + 1)),
    )
    # A persistent rate limit still surfaces (so the runner records infra_error).
    with pytest.raises(urllib.error.HTTPError):
        _post_chat("http://x", {}, "k", sleep=lambda _s: None)


def test_http_timeout_defaults_and_env_override(monkeypatch):
    monkeypatch.delenv("NELSON_HTTP_TIMEOUT", raising=False)
    assert ral._http_timeout() == ral.HTTP_TIMEOUT  # default when unset
    monkeypatch.setenv("NELSON_HTTP_TIMEOUT", "1800")
    assert ral._http_timeout() == 1800  # slow self-hosted box raises it
    monkeypatch.setenv("NELSON_HTTP_TIMEOUT", "not-an-int")
    assert ral._http_timeout() == ral.HTTP_TIMEOUT  # garbage falls back to default


def test_post_chat_does_not_retry_non_retryable_status(monkeypatch):
    calls = []

    def fake(_req, timeout=None):
        calls.append(1)
        raise _http_error(400)

    monkeypatch.setattr(ral.urllib.request, "urlopen", fake)
    with pytest.raises(urllib.error.HTTPError):
        _post_chat("http://x", {}, "k", sleep=lambda _s: None)
    assert len(calls) == 1  # 400 is a client error, not retried
