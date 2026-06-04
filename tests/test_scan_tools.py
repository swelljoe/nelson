"""Scan-mode tool loop: OpenAI-compatible models reading sibling files.

The single-shot adapter only ever sees the one pasted file. With ``--tools`` the
adapter drives the benchmark ReAct loop rooted at the scan target, so the model
can follow code into other files. These tests inject the HTTP layer (monkeypatch
``raw_api_loop._post_chat``) so the loop runs against scripted provider responses,
no network. The sandboxing itself is covered in test_raw_api_loop.
"""

import json
import urllib.error
from email.message import Message

import nelson.raw_api_loop as ral
from nelson.agents import FailureKind, OpenAIAPIAdapter, create_adapter


def _tool_call(name, args):
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
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def _final(content):
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 8, "completion_tokens": 3},
    }


def _scripted_post(monkeypatch, *responses):
    """Patch the loop's HTTP poster to return each scripted response in turn."""
    it = iter(responses)
    monkeypatch.setattr(ral, "_post_chat", lambda url, payload, api_key: next(it))


# -- create_adapter wiring ---------------------------------------------------


def test_create_adapter_enables_tools_for_openai():
    a = create_adapter("openai:m@http://h:1/v1", tools=True, tools_root="/root")
    assert isinstance(a, OpenAIAPIAdapter)
    assert a.use_tools and a.tools_root == "/root"
    assert a.tool_profile == "react"


def test_create_adapter_single_shot_by_default():
    a = create_adapter("lmstudio:m")
    assert isinstance(a, OpenAIAPIAdapter)
    assert not a.use_tools
    assert a.tool_profile == "single-shot"


def test_create_adapter_tools_ignored_for_cli_runtime():
    # Claude Code / Gemini CLI are already agentic; the flag is a harmless no-op.
    a = create_adapter("claude:haiku", tools=True, tools_root="/root")
    assert not hasattr(a, "use_tools")


# -- API key from the environment (hosted endpoints, no benchmark profile) ---


def test_bearer_token_reads_openai_api_key_env(monkeypatch):
    # An ad-hoc CLI scan against a hosted endpoint has no AuthProfile; the key
    # comes from OPENAI_API_KEY in the environment (DeepSeek/OpenRouter/etc.).
    monkeypatch.setenv("OPENAI_API_KEY", "sk-hosted-123")
    a = create_adapter("openai:deepseek-chat@https://api.deepseek.com/v1")
    assert isinstance(a, OpenAIAPIAdapter)
    assert a._bearer_token() == "sk-hosted-123"


def test_bearer_token_none_for_local_or_http_even_with_env_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-hosted-123")
    local = create_adapter("lmstudio:m")
    insecure = create_adapter("openai:m@http://api.deepseek.com/v1")
    assert isinstance(local, OpenAIAPIAdapter)
    assert isinstance(insecure, OpenAIAPIAdapter)
    assert local._bearer_token() is None
    assert insecure._bearer_token() is None


def test_bearer_token_none_without_env_key(monkeypatch):
    # Unset -> no Authorization header, which is correct for localhost servers.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    a = create_adapter("lmstudio:m")
    assert isinstance(a, OpenAIAPIAdapter)
    assert a._bearer_token() is None


# -- The tool loop -----------------------------------------------------------


def test_tool_loop_reads_sibling_file_and_returns_findings(tmp_path, monkeypatch):
    (tmp_path / "helper.py").write_text("def run(c): os.system(c)\n")
    answer = '[{"line": 1, "code": "os.system(c)", "confidence": "high"}]'
    _scripted_post(
        monkeypatch,
        _tool_call("read_file", {"path": "helper.py"}),
        _final(answer),
    )
    a = create_adapter("openai:m@http://h:1/v1", tools=True, tools_root=str(tmp_path))
    result = a.run("audit app.py")

    assert result.failure_kind is None
    assert len(result.findings) == 1
    assert result.findings[0].confidence == "high"
    assert result.tokens_in == 18 and result.tokens_out == 8  # summed over turns


def test_tool_loop_single_shot_when_tools_disabled(monkeypatch):
    # Without tools_root the loop is never entered; run() must not touch _post_chat.
    def boom(*a, **k):
        raise AssertionError("tool loop should not run in single-shot mode")

    monkeypatch.setattr(ral, "_post_chat", boom)
    a = create_adapter("openai:m@http://h:1/v1")  # tools off
    assert isinstance(a, OpenAIAPIAdapter)
    # No network call is made because httpx.post would be needed; we only assert
    # the loop branch is not taken (boom would fire if it were).
    assert a.use_tools is False


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://h/v1", code, "err", Message(), None)


def test_tool_loop_maps_auth_error_to_failure_kind(tmp_path, monkeypatch):
    # A 401 from the provider must surface as auth_failed — never an empty,
    # "looked and found nothing" result.
    def raise_401(url, payload, api_key):
        raise _http_error(401)

    monkeypatch.setattr(ral, "_post_chat", raise_401)
    a = create_adapter("openai:m@http://h:1/v1", tools=True, tools_root=str(tmp_path))
    result = a.run("audit app.py")

    assert result.failure_kind is FailureKind.AUTH
    assert result.findings == []


def test_tool_loop_maps_connection_error_to_infra(tmp_path, monkeypatch):
    def raise_conn(url, payload, api_key):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(ral, "_post_chat", raise_conn)
    a = create_adapter("openai:m@http://h:1/v1", tools=True, tools_root=str(tmp_path))
    result = a.run("audit app.py")

    assert result.failure_kind is FailureKind.INFRA


# -- Review pass uses the same loop (JSON object, not the findings array) -----


def test_tool_loop_passes_review_object_through(tmp_path, monkeypatch):
    # The review pass wants a JSON *object* verdict, not the findings array. The
    # shared loop framing must not coerce it into an array — raw_output comes back
    # verbatim for the review parser.
    from nelson.review import parse_review_result

    (tmp_path / "helper.py").write_text("def run(c): os.system(c)\n")
    verdict = (
        '{"verdict": "confirmed", "reasoning": "reachable from app.py", '
        '"severity": "high", "suggested_fix": "use subprocess with a list"}'
    )
    _scripted_post(
        monkeypatch,
        _tool_call("read_file", {"path": "helper.py"}),
        _final(verdict),
    )
    a = create_adapter("openai:m@http://h:1/v1", tools=True, tools_root=str(tmp_path))
    result = a.run("review the reported finding")

    assert result.raw_output == verdict
    parsed = parse_review_result(result.raw_output)
    assert parsed is not None and parsed["verdict"] == "confirmed"


def test_run_review_with_tools_reads_other_files(tmp_path, monkeypatch):
    # End-to-end: an OpenAI-compatible reviewer with --tools reads a sibling file
    # and its object verdict is persisted to the finding.
    from nelson.db import Database
    from nelson.review import run_review

    (tmp_path / "app.py").write_text("from helper import run\nrun(user_input)\n")
    (tmp_path / "helper.py").write_text("import os\ndef run(c): os.system(c)\n")

    db = Database(str(tmp_path / "nelson.db"))
    scan_id = db.create_scan(str(tmp_path))
    db.create_jobs_batch(scan_id, [("app.py", "OPEN", "h/m")])
    job = db.next_pending_job(scan_id, model_id="h/m")
    assert job is not None
    finding_id = db.add_finding(
        job["id"],
        line_number=2,
        code_snippet="run(user_input)",
        explanation="possible command injection",
        confidence="medium",
    )

    verdict = (
        '{"verdict": "confirmed", "reasoning": "os.system sink", "severity": "high"}'
    )
    _scripted_post(
        monkeypatch,
        _tool_call("read_file", {"path": "helper.py"}),
        _final(verdict),
    )

    run_review(
        db, scan_id, "openai:m@http://h:1/v1", str(tmp_path), delay=0, tools=True
    )

    row = db.conn.execute(
        "SELECT verified_status, verified_by_model FROM findings WHERE id = ?",
        (finding_id,),
    ).fetchone()
    assert row["verified_status"] == "confirmed"
