"""Runtime dispatch, the host->container auth bridge, and per-runtime spec/parse.

Everything here runs without podman, a network, or real credentials: a fake
container backend records the spec it was handed, and auth/secret resolution is
driven through fakes. The integrity rule is asserted directly — an unknown
runtime, a missing key, and a missing native CLI all resolve to a non-miss status.
"""

import json
from pathlib import Path

import pytest

from nelson.auth import AuthProfile
from nelson.corpus import Case
from nelson.db import Database
from nelson.runner import (
    AuthMaterial,
    BenchRunner,
    Competitor,
    ContainerExecResult,
    ContainerSpec,
    RunnerError,
)
from nelson.runtimes import (
    BindMountedCliRuntime,
    ClaudeCodeRuntime,
    EnvKeyAuth,
    GeminiCliRuntime,
    GeminiCredentialMountAuth,
    ParsedOutput,
    RawApiLoopRuntime,
    RuntimeContext,
    _FailingAuth,
    auth_for_competitor,
    get_runtime,
    parse_cost_model,
)

# -- Fakes -------------------------------------------------------------------


class FakeBackend:
    def __init__(self, result: ContainerExecResult):
        self._result = result
        self.spec: ContainerSpec | None = None
        self.ensured = False

    def ensure_image(self) -> None:
        self.ensured = True

    def run(self, spec, timeout, cancel_event=None):
        self.spec = spec
        return self._result


class FakeAuth:
    def prepare(self, staging):
        return AuthMaterial(env={"X": "1"}, mounts=[])


class FakeStore:
    def __init__(self, values: dict[str, str]):
        self._values = values

    def get(self, name: str) -> str | None:
        return self._values.get(name)


def _vetted_case() -> Case:
    return Case(
        source="cvd",
        ext_id="GHSA-test",
        project="acme/widget",
        repo_url="https://example.invalid/acme/widget",
        vuln_commit="deadbeef",
        fix_commit="cafef00d",
        status="vetted",
    )


def _runner(db, backend, monkeypatch, tmp_path, *, auth=None, preflight=False):
    def _stub(url, commit, dest, **k):
        tree = Path(dest) / "src"
        tree.mkdir(parents=True, exist_ok=True)
        return tree

    monkeypatch.setattr("nelson.runner.prepare_checkout", _stub)
    return BenchRunner(
        db,
        backend=backend,
        auth=auth,
        claude_bin=Path("/usr/bin/claude"),
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs",
        preflight=preflight,
    )


_RESULT = json.dumps(
    {
        "type": "result",
        "result": '[{"file": "a.c", "line": 3, "explanation": "x", '
        '"confidence": "high", "cwe": "CWE-1"}]',
        "usage": {"input_tokens": 50, "output_tokens": 10},
        "total_cost_usd": 0.001,
    }
)


# -- Registry ----------------------------------------------------------------


def test_get_runtime_unknown_raises():
    with pytest.raises(RunnerError):
        get_runtime("nope")


def test_all_planned_runtimes_registered():
    for name in ("claude-code", "raw-api-loop", "gemini-cli", "kimi-cli", "pi-custom"):
        assert get_runtime(name).name == name


def test_deepseek_cli_not_registered():
    # DeepSeek has no trusted first-party agent CLI; it runs through claude-code
    # (Anthropic-compat) and raw-api-loop (OpenAI-compat) instead.
    with pytest.raises(RunnerError):
        get_runtime("deepseek-cli")


def test_run_case_unknown_runtime_is_infra_error(tmp_path, monkeypatch):
    db = Database(tmp_path / "t.db")
    backend = FakeBackend(ContainerExecResult(0, _RESULT, ""))
    runner = _runner(db, backend, monkeypatch, tmp_path, auth=FakeAuth())

    result = runner.run_case(
        _vetted_case(), Competitor(name="x/y", model="y", runtime="bogus"), "a.c"
    )

    assert result.status == "infra_error"
    assert db.list_runs()[0]["status"] == "infra_error"
    assert backend.spec is None  # never reached the container


# -- Auth bridge -------------------------------------------------------------


def test_env_key_auth_injects_resolved_secret(tmp_path):
    profile = AuthProfile(name="p", env={"NELSON_API_KEY": "SECRET_NAME"})
    auth = EnvKeyAuth(profile, store=FakeStore({"SECRET_NAME": "sk-123"}))
    material = auth.prepare(tmp_path)
    assert material.env == {"NELSON_API_KEY": "sk-123"}
    assert material.mounts == []


def test_env_key_auth_missing_secret_is_runner_error(tmp_path):
    profile = AuthProfile(name="p", env={"NELSON_API_KEY": "SECRET_NAME"})
    auth = EnvKeyAuth(profile, store=FakeStore({}))
    with pytest.raises(RunnerError):
        auth.prepare(tmp_path)


def test_failing_auth_raises(tmp_path):
    with pytest.raises(RunnerError):
        _FailingAuth("nope").prepare(tmp_path)


def test_auth_for_competitor_unknown_profile_fails(tmp_path):
    comp = Competitor(name="x", model="m", runtime="raw-api-loop", auth_profile="ghost")
    auth = auth_for_competitor(comp, RawApiLoopRuntime())
    assert isinstance(auth, _FailingAuth)
    with pytest.raises(RunnerError):
        auth.prepare(tmp_path)


def test_auth_for_competitor_without_profile_uses_runtime_default():
    comp = Competitor(name="x", model="sonnet", runtime="claude-code")
    auth = auth_for_competitor(comp, ClaudeCodeRuntime())
    # claude's default is the credential mount, not an env-key injector.
    assert not isinstance(auth, EnvKeyAuth)


def test_gemini_credential_mount_auth_stages_present_files(tmp_path):
    creds = tmp_path / "gem"
    creds.mkdir()
    (creds / "oauth_creds.json").write_text("{}")
    material = GeminiCredentialMountAuth(creds_dir=creds).prepare(tmp_path / "stg")
    assert material.mounts and material.mounts[0][1] == "/home/agent/.gemini"


def test_gemini_credential_mount_auth_missing_is_runner_error(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RunnerError):
        GeminiCredentialMountAuth(creds_dir=empty).prepare(tmp_path / "stg")


# -- cost_model parsing ------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "subscription", "not json", "[1,2]"])
def test_parse_cost_model_non_object_is_empty(value):
    assert parse_cost_model(value) == {}


def test_parse_cost_model_json_object():
    cfg = parse_cost_model('{"base_url": "https://x", "input_usd_per_mtok": 0.1}')
    assert cfg["base_url"] == "https://x"
    assert cfg["input_usd_per_mtok"] == 0.1


# -- raw-api-loop spec -------------------------------------------------------


def test_raw_api_loop_build_spec_mounts_and_env(tmp_path):
    comp = Competitor(
        name="raw/ds",
        model="deepseek-chat",
        runtime="raw-api-loop",
        cost_model='{"base_url": "https://api.deepseek.com", '
        '"input_usd_per_mtok": 0.14, "output_usd_per_mtok": 0.28}',
    )
    ctx = RuntimeContext(
        competitor=comp,
        prompt="audit a.c",
        src_dir=tmp_path / "src",
        auth=AuthMaterial(env={"NELSON_API_KEY": "sk-1"}, mounts=[]),
        name="r1",
        network=True,
    )
    spec = RawApiLoopRuntime().build_spec(ctx)
    assert spec.argv[0] == "python3"
    assert spec.argv[1].endswith("raw_api_loop.py")
    assert spec.stdin == "audit a.c"
    assert spec.network is True
    modes = {dst: opts for _, dst, opts in spec.mounts}
    assert modes["/src"] == "ro"
    assert any(
        dst.endswith("raw_api_loop.py") and opts == "ro" for _, dst, opts in spec.mounts
    )
    assert spec.env["NELSON_BASE_URL"] == "https://api.deepseek.com"
    assert spec.env["NELSON_MODEL"] == "deepseek-chat"
    assert spec.env["NELSON_API_KEY"] == "sk-1"
    assert spec.env["NELSON_INPUT_USD_PER_MTOK"] == "0.14"
    assert spec.env["NELSON_OUTPUT_USD_PER_MTOK"] == "0.28"


def test_raw_api_loop_default_auth_requires_profile(tmp_path):
    with pytest.raises(RunnerError):
        RawApiLoopRuntime().default_auth().prepare(tmp_path)


# -- claude-code Anthropic-compatible passthrough (e.g. DeepSeek) -------------


def test_claude_code_anthropic_compat_injects_base_url_and_model_env(tmp_path):
    comp = Competitor(
        name="claude-code/deepseek",
        model="deepseek-v4-pro",
        runtime="claude-code",
        cost_model='{"anthropic_base_url": "https://api.deepseek.com/anthropic", '
        '"env": {"ANTHROPIC_MODEL": "deepseek-v4-pro"}}',
    )
    ctx = RuntimeContext(
        competitor=comp,
        prompt="audit a.c",
        src_dir=tmp_path / "src",
        auth=AuthMaterial(env={"ANTHROPIC_AUTH_TOKEN": "sk-ds"}, mounts=[]),
        name="r1",
        claude_bin=Path("/usr/bin/claude"),
    )
    spec = ClaudeCodeRuntime().build_spec(ctx)
    assert spec.argv[:2] == ["claude", "-p"]  # still the claude-code harness
    assert spec.env["ANTHROPIC_BASE_URL"] == "https://api.deepseek.com/anthropic"
    assert spec.env["ANTHROPIC_MODEL"] == "deepseek-v4-pro"
    assert spec.env["ANTHROPIC_AUTH_TOKEN"] == "sk-ds"  # token from the auth profile


def test_claude_code_native_has_no_anthropic_base_url(tmp_path):
    comp = Competitor(name="claude-code/sonnet", model="sonnet", runtime="claude-code")
    ctx = RuntimeContext(
        competitor=comp,
        prompt="audit a.c",
        src_dir=tmp_path / "src",
        auth=AuthMaterial(env={"CLAUDE_CONFIG_DIR": "/cfg"}, mounts=[]),
        name="r1",
        claude_bin=Path("/usr/bin/claude"),
    )
    spec = ClaudeCodeRuntime().build_spec(ctx)
    assert "ANTHROPIC_BASE_URL" not in spec.env  # native subscription, unchanged


def test_raw_api_loop_parse_output_ingests_result_object():
    parsed = RawApiLoopRuntime().parse_output(
        ContainerExecResult(0, _RESULT, ""), Competitor(name="x", model="m")
    )
    assert isinstance(parsed, ParsedOutput)
    assert (parsed.tokens_in, parsed.tokens_out, parsed.cost) == (50, 10, 0.001)
    assert len(parsed.findings) == 1
    assert parsed.findings[0]["file"] == "a.c"


# -- gemini-cli + native CLIs ------------------------------------------------


def test_gemini_build_spec_infra_when_binary_absent(monkeypatch, tmp_path):
    monkeypatch.setattr("nelson.runtimes.shutil.which", lambda _b: None)
    ctx = RuntimeContext(
        competitor=Competitor(name="g", model="gemini-2.5-pro", runtime="gemini-cli"),
        prompt="audit",
        src_dir=tmp_path / "src",
        auth=AuthMaterial(env={}, mounts=[]),
        name="r1",
    )
    with pytest.raises(RunnerError):
        GeminiCliRuntime().build_spec(ctx)


def test_gemini_parse_output_sums_model_tokens():
    envelope = json.dumps(
        {
            "response": '[{"file": "x.c", "line": 1, "confidence": "high"}]',
            "stats": {"models": {"g": {"tokens": {"input": 100, "candidates": 20}}}},
        }
    )
    parsed = GeminiCliRuntime().parse_output(
        ContainerExecResult(0, envelope, ""), Competitor(name="g", model="g")
    )
    assert (parsed.tokens_in, parsed.tokens_out, parsed.cost) == (100, 20, None)
    assert len(parsed.findings) == 1


@pytest.mark.parametrize("runtime", ["kimi-cli", "pi-custom"])
def test_native_cli_stub_is_infra_error_when_binary_absent(
    runtime, tmp_path, monkeypatch
):
    monkeypatch.setattr("nelson.runtimes.shutil.which", lambda _b: None)
    db = Database(tmp_path / "t.db")
    backend = FakeBackend(ContainerExecResult(0, _RESULT, ""))
    runner = _runner(db, backend, monkeypatch, tmp_path)
    comp = Competitor(
        name=f"{runtime}/m", model="m", runtime=runtime, auth_profile="deepseek-api-key"
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")  # so auth.prepare succeeds

    result = runner.run_case(_vetted_case(), comp, "a.c")

    assert result.status == "infra_error"  # missing binary, never a miss
    assert backend.spec is None


def test_bind_mounted_cli_build_spec_when_present(monkeypatch, tmp_path):
    monkeypatch.setattr("nelson.runtimes.shutil.which", lambda _b: "/usr/bin/kimi")
    ctx = RuntimeContext(
        competitor=Competitor(name="k", model="m", runtime="kimi-cli"),
        prompt="audit",
        src_dir=tmp_path / "src",
        auth=AuthMaterial(env={"NELSON_API_KEY": "k"}, mounts=[]),
        name="r1",
    )
    spec = BindMountedCliRuntime("kimi-cli", "kimi").build_spec(ctx)
    assert spec.argv[0] == "kimi"
    modes = {dst: opts for _, dst, opts in spec.mounts}
    assert modes["/src"] == "ro"
    assert modes["/usr/local/bin/kimi"] == "ro"


# -- Full dispatch through run_case (fake backend) ---------------------------


def test_run_case_dispatches_to_raw_api_loop(tmp_path, monkeypatch):
    db = Database(tmp_path / "t.db")
    backend = FakeBackend(ContainerExecResult(0, _RESULT, ""))
    runner = _runner(db, backend, monkeypatch, tmp_path)  # no injected auth
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-live")
    comp = Competitor(
        name="raw-api-loop/deepseek",
        model="deepseek-chat",
        runtime="raw-api-loop",
        auth_profile="deepseek-api-key",
        cost_model='{"base_url": "https://api.deepseek.com"}',
    )

    result = runner.run_case(_vetted_case(), comp, "a.c")

    assert result.status == "complete"
    assert len(result.findings) == 1
    # The dispatch used the raw-api-loop spec, and EnvKeyAuth injected the key.
    assert backend.spec is not None
    assert backend.spec.argv[0] == "python3"
    assert backend.spec.env["NELSON_API_KEY"] == "sk-live"
    assert backend.spec.env["NELSON_BASE_URL"] == "https://api.deepseek.com"


def test_run_case_raw_api_missing_key_is_auth_failed(tmp_path, monkeypatch):
    db = Database(tmp_path / "t.db")
    backend = FakeBackend(ContainerExecResult(0, _RESULT, ""))
    runner = _runner(db, backend, monkeypatch, tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    comp = Competitor(
        name="raw-api-loop/deepseek",
        model="deepseek-chat",
        runtime="raw-api-loop",
        auth_profile="deepseek-api-key",
    )

    result = runner.run_case(_vetted_case(), comp, "a.c")

    assert result.status == "auth_failed"  # dead key is never a miss
    assert backend.spec is None


def test_run_case_raw_api_without_profile_is_auth_failed(tmp_path, monkeypatch):
    db = Database(tmp_path / "t.db")
    backend = FakeBackend(ContainerExecResult(0, _RESULT, ""))
    runner = _runner(db, backend, monkeypatch, tmp_path)
    comp = Competitor(
        name="raw/x", model="m", runtime="raw-api-loop"
    )  # no auth_profile

    result = runner.run_case(_vetted_case(), comp, "a.c")

    assert result.status == "auth_failed"
    assert backend.spec is None


def test_preflight_short_circuits_before_container(tmp_path, monkeypatch):
    from nelson.agents import OpenAIAPIAdapter, PreflightResult

    db = Database(tmp_path / "t.db")
    backend = FakeBackend(ContainerExecResult(0, _RESULT, ""))
    runner = _runner(db, backend, monkeypatch, tmp_path, preflight=True)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-live")
    monkeypatch.setattr(
        OpenAIAPIAdapter,
        "preflight",
        lambda self, cancel_event=None: PreflightResult(
            False, "auth_failed", "bad key"
        ),
    )
    comp = Competitor(
        name="raw-api-loop/deepseek",
        model="deepseek-chat",
        runtime="raw-api-loop",
        auth_profile="deepseek-api-key",
        cost_model='{"base_url": "https://api.deepseek.com"}',
    )

    result = runner.run_case(_vetted_case(), comp, "a.c")

    assert result.status == "auth_failed"
    assert backend.spec is None  # no container spend on a dead key
