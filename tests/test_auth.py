"""Auth profile resolution: secret *names*, never values, in the config."""

import pytest

from nelson.auth import (
    STANDARD_AUTH_PROFILES,
    AuthProfile,
    EnvSecretStore,
    MissingSecretError,
    OAuthSpec,
    bootstrap_oauth,
)


class DictStore:
    """A secret store backed by a plain dict, for hermetic tests."""

    def __init__(self, **secrets: str):
        self._secrets = secrets

    def get(self, name: str) -> str | None:
        return self._secrets.get(name)


def test_resolve_env_maps_target_var_to_resolved_value():
    # The profile names a secret; the store supplies the value. The target env
    # var the runtime reads can differ from the secret's name.
    profile = AuthProfile(name="p", env={"OPENAI_API_KEY": "DEEPSEEK_KEY"})
    store = DictStore(DEEPSEEK_KEY="sk-deepseek-123")

    resolved = profile.resolve_env(store)

    assert resolved == {"OPENAI_API_KEY": "sk-deepseek-123"}


def test_profile_holds_names_not_values():
    # Nothing secret is stored on the profile itself — only the lookup name.
    profile = AuthProfile(name="p", env={"ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY"})
    assert "sk-" not in repr(profile)
    assert profile.env["ANTHROPIC_API_KEY"] == "ANTHROPIC_API_KEY"


def test_missing_secret_raises_and_names_every_gap():
    profile = AuthProfile(
        name="kimi",
        env={"A": "PRESENT", "B": "ABSENT_1", "C": "ABSENT_2"},
    )
    store = DictStore(PRESENT="x")

    with pytest.raises(MissingSecretError) as exc:
        profile.resolve_env(store)

    assert exc.value.profile_name == "kimi"
    assert set(exc.value.missing) == {"ABSENT_1", "ABSENT_2"}


def test_env_secret_store_reads_process_env(monkeypatch):
    monkeypatch.setenv("NELSON_TEST_SECRET", "value-from-env")
    store = EnvSecretStore()
    assert store.get("NELSON_TEST_SECRET") == "value-from-env"
    assert store.get("NELSON_TEST_DEFINITELY_ABSENT") is None


def test_empty_profile_resolves_to_empty_env():
    # No declared secrets -> nothing to inject, no error.
    assert AuthProfile(name="none").resolve_env(DictStore()) == {}


def test_standard_profiles_reference_names_not_values():
    anthropic = STANDARD_AUTH_PROFILES["anthropic-api-key"]
    assert anthropic.env == {"ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY"}
    oauth = STANDARD_AUTH_PROFILES["anthropic-oauth"]
    assert isinstance(oauth.oauth, OAuthSpec)
    assert oauth.oauth.token_env == "CLAUDE_CODE_OAUTH_TOKEN"


def test_oauth_bootstrap_is_deferred():
    spec = OAuthSpec(
        runtime="claude-code",
        token_env="CLAUDE_CODE_OAUTH_TOKEN",
        bootstrap_cmd=("claude", "setup-token"),
    )
    with pytest.raises(NotImplementedError):
        bootstrap_oauth(spec)
