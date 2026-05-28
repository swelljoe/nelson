"""The in-container ReAct agent: tool sandboxing, the loop, and cost — no live API.

The HTTP call is injected (``run_loop(post=...)``) so the loop is exercised
against scripted provider responses. The sandbox tests are the security-critical
ones: a tool must never read outside the source root, even via ``..`` or a symlink.
"""

import json

from nelson.raw_api_loop import (
    _resolve_in_src,
    compute_cost,
    dispatch_tool,
    run_loop,
    tool_grep,
    tool_list_dir,
    tool_read_file,
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
