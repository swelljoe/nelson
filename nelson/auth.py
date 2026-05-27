"""Auth profiles and secret resolution for competitors.

A competitor's credentials are described by an :class:`AuthProfile` that
references secret *names*, never secret values. Values are resolved at run
time from a :class:`SecretStore` (env vars by default). This keeps keys out
of git and out of the benchmark config: the config says "use the secret named
``ANTHROPIC_API_KEY``", and the value is supplied by the environment of the
machine actually running the benchmark.

OAuth bootstrap (``claude setup-token`` -> ``CLAUDE_CODE_OAUTH_TOKEN`` and
similar) is a one-time *interactive* step. P0 only defines the interface; the
seed competitor set is entirely API-key auth, so the bootstrap is deferred.
"""

import os
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


class MissingSecretError(Exception):
    """A referenced secret name could not be resolved from the store.

    Raised during env resolution so callers can map it to an ``auth_failed``
    status rather than letting a missing key look like an infra crash or, worse,
    a model that genuinely found nothing.
    """

    def __init__(self, profile_name: str, missing: list[str]):
        self.profile_name = profile_name
        self.missing = missing
        super().__init__(
            f"auth profile '{profile_name}' is missing secret(s): "
            f"{', '.join(missing)}"
        )


@runtime_checkable
class SecretStore(Protocol):
    """Resolves a secret *name* to its value, or ``None`` if absent."""

    def get(self, name: str) -> str | None: ...


class EnvSecretStore:
    """Default store: secrets live in the process environment.

    The benchmark machine holds the keys in its environment (or an exported
    untracked ``.env``); the config only ever names them.
    """

    def get(self, name: str) -> str | None:
        return os.environ.get(name)


# Module-level default so call sites don't each construct one.
DEFAULT_SECRET_STORE: SecretStore = EnvSecretStore()


@dataclass(frozen=True)
class OAuthSpec:
    """Descriptor for a one-time interactive OAuth bootstrap (deferred).

    ``token_env`` is the env var the runtime reads (e.g.
    ``CLAUDE_CODE_OAUTH_TOKEN``); ``bootstrap_cmd`` is the interactive command a
    human runs once (e.g. ``["claude", "setup-token"]``). Long-lived tokens are
    preferred over shared credential files for parallel runs because refresh and
    file locking on the shared credential store are undocumented.
    """

    runtime: str
    token_env: str
    bootstrap_cmd: tuple[str, ...]


def bootstrap_oauth(spec: OAuthSpec) -> str:
    """Run the one-time interactive OAuth flow and return the long-lived token.

    Deferred: the seed competitors are all API-key auth. Implementing this means
    driving an interactive subprocess (TTY) and capturing the emitted token,
    which we don't need until a subscription-auth competitor is added (P7).
    """
    raise NotImplementedError(
        "OAuth bootstrap is not implemented yet (deferred to a subscription-auth "
        f"competitor). Run `{' '.join(spec.bootstrap_cmd)}` manually and export "
        f"{spec.token_env}."
    )


@dataclass(frozen=True)
class AuthProfile:
    """Named credential recipe for a competitor.

    ``env`` maps a *target* environment variable (what the runtime reads) to a
    *secret name* (a key into the secret store). For example
    ``{"ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY"}`` or, to point a runtime at a
    differently-named secret, ``{"OPENAI_API_KEY": "DEEPSEEK_KEY"}``.

    ``credential_files`` (dest path -> secret name) is the staged-file analogue
    for runtimes that read a creds file rather than an env var; staging is a stub
    in P0. ``oauth`` attaches an :class:`OAuthSpec` for subscription auth.
    """

    name: str
    env: dict[str, str] = field(default_factory=dict)
    credential_files: dict[str, str] = field(default_factory=dict)
    oauth: OAuthSpec | None = None

    def resolve_env(self, store: SecretStore | None = None) -> dict[str, str]:
        """Resolve ``env`` to ``{target_var: value}`` using the secret store.

        Raises :class:`MissingSecretError` listing every unresolved secret, so a
        missing key surfaces as one clear ``auth_failed`` rather than a partial
        environment that fails opaquely deeper in the runtime.
        """
        store = store or DEFAULT_SECRET_STORE
        resolved: dict[str, str] = {}
        missing: list[str] = []
        for target_var, secret_name in self.env.items():
            value = store.get(secret_name)
            if value is None:
                missing.append(secret_name)
            else:
                resolved[target_var] = value
        if missing:
            raise MissingSecretError(self.name, missing)
        return resolved


# Standard profiles for known runtimes. These are *data*: a competitor attaches
# one by name. They are not applied by default (see create_adapter), so the
# existing inherit-the-parent-environment behavior is unchanged unless a profile
# is explicitly requested.
STANDARD_AUTH_PROFILES: dict[str, AuthProfile] = {
    "anthropic-api-key": AuthProfile(
        name="anthropic-api-key",
        env={"ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY"},
    ),
    "anthropic-oauth": AuthProfile(
        name="anthropic-oauth",
        env={"CLAUDE_CODE_OAUTH_TOKEN": "CLAUDE_CODE_OAUTH_TOKEN"},
        oauth=OAuthSpec(
            runtime="claude-code",
            token_env="CLAUDE_CODE_OAUTH_TOKEN",  # noqa: S106 — env var name, not a secret
            bootstrap_cmd=("claude", "setup-token"),
        ),
    ),
    "gemini-api-key": AuthProfile(
        name="gemini-api-key",
        env={"GEMINI_API_KEY": "GEMINI_API_KEY"},
    ),
    "openai-api-key": AuthProfile(
        name="openai-api-key",
        env={"OPENAI_API_KEY": "OPENAI_API_KEY"},
    ),
}
