"""Track-B native-agent runtimes: reasonix / qwen / mimo / kimi.

Each is a vendor CLI bind-mounted into the run, restricted to a read-only,
no-web-search/fetch tool posture. These tests assert — without podman, network,
or real credentials — the two things most likely to regress silently: the
per-CLI web-tool-disable lever baked into each credential-mount auth, and the
per-CLI output parsing (esp. qwen's array-of-events shape). Binary/tree presence
is faked via monkeypatch; the integrity rule (missing binary/creds -> RunnerError
-> non-miss) is asserted directly.
"""

import json
from pathlib import Path

import pytest

from nelson.runner import (
    AuthMaterial,
    Competitor,
    ContainerExecResult,
    RunnerError,
)
from nelson.runtimes import (
    KimiCredentialMountAuth,
    KimiRuntime,
    MimoCredentialMountAuth,
    MimoRuntime,
    QwenCredentialMountAuth,
    QwenRuntime,
    ReasonixCredentialMountAuth,
    ReasonixRuntime,
    RuntimeContext,
    _reasonix_readonly_config,
    get_runtime,
)


def _ctx(tmp_path, competitor, auth=None):
    return RuntimeContext(
        competitor=competitor,
        prompt="audit ngx_http_dav_module.c",
        src_dir=tmp_path / "src",
        auth=auth or AuthMaterial(env={}, mounts=[]),
        name="r1",
    )


# -- registration ------------------------------------------------------------


@pytest.mark.parametrize("name", ["reasonix", "kimi", "qwen", "mimo"])
def test_native_runtimes_registered(name):
    assert get_runtime(name).name == name


# -- reasonix ----------------------------------------------------------------


def test_reasonix_readonly_config_pins_allowlist_and_keeps_providers():
    src = (
        'default_model = "deepseek-pro/deepseek-v4-pro"\n\n'
        "[tools]\nenabled = []   # empty = all built-in tools\n\n"
        '[[providers]]\nname = "deepseek-pro"\nmodels = ["deepseek-v4-pro"]\n'
        'api_key_env = "DEEPSEEK_API_KEY"\n'
    )
    out = _reasonix_readonly_config(src)
    assert 'enabled = ["read_file", "grep", "glob", "ls", "code_index", "bash"]' in out
    assert "web_fetch" not in out  # the web tool is not in the allowlist
    # provider config (and the key it needs) survives the rewrite
    assert "[[providers]]" in out and "deepseek-v4-pro" in out


def test_reasonix_readonly_config_appends_table_when_absent():
    out = _reasonix_readonly_config('default_model = "x"\n')
    assert "[tools]" in out and "code_index" in out


def test_reasonix_auth_stages_config_and_env(tmp_path):
    creds = tmp_path / "dotr"
    creds.mkdir()
    (creds / "config.toml").write_text("[tools]\nenabled = []\n")
    (creds / ".env").write_text("DEEPSEEK_API_KEY=sk-x\n")
    material = ReasonixCredentialMountAuth(creds_dir=creds).prepare(tmp_path / "stg")
    assert material.mounts == [
        (str(tmp_path / "stg" / "reasonix"), "/home/agent/.reasonix", "rw,U")
    ]
    staged = tmp_path / "stg" / "reasonix"
    assert (staged / ".env").read_text() == "DEEPSEEK_API_KEY=sk-x\n"
    # web tool dropped from the staged config's allowlist
    assert "code_index" in (staged / "config.toml").read_text()


def test_reasonix_auth_missing_is_runner_error(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RunnerError):
        ReasonixCredentialMountAuth(creds_dir=empty).prepare(tmp_path / "stg")


def test_reasonix_build_spec_argv_and_mounts(monkeypatch, tmp_path):
    monkeypatch.setattr("nelson.runtimes.shutil.which", lambda _b: "/usr/bin/reasonix")
    comp = Competitor(
        name="reasonix/deepseek-v4-pro", model="deepseek-pro", runtime="reasonix"
    )
    spec = ReasonixRuntime().build_spec(_ctx(tmp_path, comp))
    assert spec.argv[:2] == ["reasonix", "run"]
    assert spec.argv[spec.argv.index("--dir") + 1] == "/src"
    assert spec.argv[spec.argv.index("--model") + 1] == "deepseek-pro"
    assert spec.argv[-1] == "audit ngx_http_dav_module.c"
    modes = {dst: opts for _, dst, opts in spec.mounts}
    assert modes["/src"] == "ro" and modes["/usr/local/bin/reasonix"] == "ro"


def test_reasonix_build_spec_infra_when_binary_absent(monkeypatch, tmp_path):
    monkeypatch.setattr("nelson.runtimes.shutil.which", lambda _b: None)
    comp = Competitor(name="reasonix/x", model="deepseek-pro", runtime="reasonix")
    with pytest.raises(RunnerError):
        ReasonixRuntime().build_spec(_ctx(tmp_path, comp))


def test_reasonix_parse_output_findings_null_usage():
    stdout = 'thinking\n[{"file": "a.c", "line": 5, "cwe": "CWE-59"}]\n · 100 tok\n'
    parsed = ReasonixRuntime().parse_output(
        ContainerExecResult(0, stdout, ""), Competitor(name="reasonix/x", model="d")
    )
    assert (parsed.tokens_in, parsed.tokens_out, parsed.cost) == (None, None, None)
    assert parsed.findings[0]["cwe"] == "CWE-59"


# -- kimi --------------------------------------------------------------------


def test_kimi_auth_stages_creds_and_readonly_profile(tmp_path):
    creds = tmp_path / "dotkimi"
    (creds / "credentials").mkdir(parents=True)
    (creds / "credentials" / "kimi-code.json").write_text("{}")
    (creds / "config.toml").write_text('default_model = "kimi-code/k3"\n')
    material = KimiCredentialMountAuth(creds_dir=creds).prepare(tmp_path / "stg")
    assert material.env == {"KIMI_CODE_EXPERIMENTAL_FLAG": "1"}
    dsts = {dst: opts for _, dst, opts in material.mounts}
    assert dsts["/home/agent/.kimi-code"] == "rw,U"
    assert dsts["/home/agent/agents"] == "ro"
    # the restricted profile must be world-readable (ro mount, no uid remap) and
    # must NOT grant the web tools
    prof = next(
        Path(src) for src, dst, _ in material.mounts if dst == "/home/agent/agents"
    )
    body = (prof / "nelson-readonly.md").read_text()
    assert "WebSearch" not in body and "FetchURL" not in body
    assert (prof / "nelson-readonly.md").stat().st_mode & 0o004  # world-readable


def test_kimi_auth_missing_is_runner_error(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RunnerError):
        KimiCredentialMountAuth(creds_dir=empty).prepare(tmp_path / "stg")


def test_kimi_build_spec_argv_and_agent_file(monkeypatch, tmp_path):
    monkeypatch.setattr("nelson.runtimes.shutil.which", lambda _b: "/usr/bin/kimi")
    comp = Competitor(name="kimi/k3", model="kimi-code/k3", runtime="kimi")
    spec = KimiRuntime().build_spec(_ctx(tmp_path, comp))
    assert spec.argv[0] == "kimi" and "-p" in spec.argv
    assert spec.argv[spec.argv.index("-m") + 1] == "kimi-code/k3"
    assert spec.argv[spec.argv.index("--agent-file") + 1].endswith("nelson-readonly.md")
    assert spec.workdir == "/src"


# -- qwen --------------------------------------------------------------------


def test_qwen_auth_injects_tool_exclusions(tmp_path):
    creds = tmp_path / "dotqwen"
    creds.mkdir()
    (creds / "settings.json").write_text(json.dumps({"modelProviders": {"openai": []}}))
    auth = QwenCredentialMountAuth(creds_dir=creds)
    auth.prepare(tmp_path / "stg")
    staged = json.loads((tmp_path / "stg" / "qwen" / "settings.json").read_text())
    assert "web_fetch" in staged["tools"]["exclude"]
    assert "web_search" in staged["tools"]["exclude"]
    assert "modelProviders" in staged  # provider config preserved


def test_qwen_auth_missing_is_runner_error(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RunnerError):
        QwenCredentialMountAuth(creds_dir=empty).prepare(tmp_path / "stg")


def test_qwen_build_spec_no_safe_mode(monkeypatch, tmp_path):
    tree = tmp_path / "qwen-code"
    (tree / "bin").mkdir(parents=True)
    (tree / "bin" / "qwen").write_text("#!/bin/sh\n")
    monkeypatch.setattr(QwenRuntime, "_TREE", tree)
    comp = Competitor(
        name="qwen/qwen3.8-max-preview", model="qwen3.8-max-preview", runtime="qwen"
    )
    spec = QwenRuntime().build_spec(_ctx(tmp_path, comp))
    # --safe-mode would bypass the tools.exclude filter, so it must NOT be passed
    assert "--safe-mode" not in spec.argv
    assert spec.argv[spec.argv.index("-o") + 1] == "json"
    assert spec.argv[spec.argv.index("-m") + 1] == "qwen3.8-max-preview"


def test_qwen_parse_output_reads_result_object_and_usage():
    arr = [
        {"type": "system", "subtype": "init", "tools": ["read_file"]},
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": '[{"file": "a.c", "line": 9, "cwe": "CWE-122"}]',
            "usage": {"input_tokens": 1234, "output_tokens": 56},
        },
    ]
    parsed = QwenRuntime().parse_output(
        ContainerExecResult(0, json.dumps(arr), ""),
        Competitor(name="qwen/x", model="q", cost_model="subscription"),
    )
    assert parsed.tokens_in == 1234 and parsed.tokens_out == 56
    assert parsed.findings[0]["cwe"] == "CWE-122"


def test_qwen_parse_output_survives_non_json():
    parsed = QwenRuntime().parse_output(
        ContainerExecResult(0, "not json at all", ""),
        Competitor(name="qwen/x", model="q"),
    )
    assert parsed.findings == [] and parsed.tokens_in is None


# -- mimo --------------------------------------------------------------------


def test_mimo_auth_writes_webfetch_deny_config(tmp_path):
    home = tmp_path / "home"
    share = home / ".local" / "share" / "mimocode"
    share.mkdir(parents=True)
    (share / "auth.json").write_text('{"token": "x"}')
    material = MimoCredentialMountAuth(home=home).prepare(tmp_path / "stg")
    assert material.mounts == [
        (str(tmp_path / "stg" / "mimo-home"), "/home/agent", "rw,U")
    ]
    cfg = json.loads(
        (
            tmp_path / "stg" / "mimo-home" / ".config" / "mimocode" / "mimocode.jsonc"
        ).read_text()
    )
    assert cfg["permission"]["webfetch"] == "deny"
    assert (
        tmp_path / "stg" / "mimo-home" / ".local" / "share" / "mimocode" / "auth.json"
    ).is_file()


def test_mimo_auth_missing_is_runner_error(tmp_path):
    home = tmp_path / "home"
    (home / ".local" / "share" / "mimocode").mkdir(parents=True)
    with pytest.raises(RunnerError):
        MimoCredentialMountAuth(home=home).prepare(tmp_path / "stg")


def test_mimo_build_spec_argv(monkeypatch, tmp_path):
    mbin = tmp_path / "mimo"
    mbin.write_text("#!/bin/sh\n")
    monkeypatch.setattr("nelson.runtimes.shutil.which", lambda _b: str(mbin))
    comp = Competitor(
        name="mimo/mimo-v2.5-pro",
        model="xiaomi-token-plan-sgp/mimo-v2.5-pro",
        runtime="mimo",
    )
    spec = MimoRuntime().build_spec(_ctx(tmp_path, comp))
    assert spec.argv[:3] == ["mimo", "run", "--pure"]
    assert spec.argv[spec.argv.index("-m") + 1] == "xiaomi-token-plan-sgp/mimo-v2.5-pro"
    assert spec.argv[-1] == "audit ngx_http_dav_module.c"
