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
    _post_chat,
    _resolve_in_src,
    compute_cost,
    dispatch_tool,
    run_loop,
    tool_grep,
    tool_list_dir,
    tool_read_file,
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
    final_text, tin, tout, steps = run_loop(
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


def test_loop_max_steps_terminates(tmp_path):
    # A model that never stops calling tools must still terminate and return.
    post = _post_returning(*[_tool_call("list_dir", {"path": "."}) for _ in range(5)])
    final_text, _tin, _tout, steps = run_loop(
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
    final_text, _tin, _tout, _steps = run_loop(
        "audit",
        base_url="https://x/v1",
        model="m",
        api_key="k",
        post=post,
        src_root=str(tmp_path),
    )
    assert final_text == "I could not find anything exploitable."


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
    assert out == "l2\nl3\n"


def test_list_dir_lists_entries(tmp_path):
    (tmp_path / "a").write_text("x")
    (tmp_path / "b").write_text("y")
    assert tool_list_dir({"path": "."}, str(tmp_path)) == "a\nb"


def test_dispatch_unknown_tool_is_error():
    assert "unknown tool" in dispatch_tool("nope", {}, None)


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
