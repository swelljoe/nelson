"""User configuration: per-stage model defaults from a YAML file.

Search order (project overrides home; CLI flags override both):

  ./nelson.yaml    project-local
  ~/.nelson.yaml   home

All keys are optional; an absent file simply contributes nothing, so behavior
with no config matches the built-in defaults.

  scan_models:  list[str]   models for the scan stage (``haha`` needs >= 2)
  review_model: str         model for the review/judge stage (``haha`` needs one)
  repeat:       int         default number of passes
  db:           str         default SQLite database path
  delay:        float       default per-job pacing (seconds)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_CONFIG = Path("nelson.yaml")
HOME_CONFIG = Path.home() / ".nelson.yaml"

_KNOWN_KEYS = frozenset({"scan_models", "review_model", "repeat", "db", "delay"})


class ConfigError(Exception):
    """A nelson.yaml exists but is malformed."""


def _load_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: invalid YAML: {e}") from e
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"{path}: top level must be a mapping, got {type(data).__name__}"
        )
    return data


def _validate(cfg: dict[str, Any]) -> None:
    unknown = set(cfg) - _KNOWN_KEYS
    if unknown:
        raise ConfigError(f"unknown config key(s): {', '.join(sorted(unknown))}")

    sm = cfg.get("scan_models")
    if sm is not None and (
        not isinstance(sm, list) or not all(isinstance(x, str) for x in sm)
    ):
        raise ConfigError("scan_models must be a list of strings")

    rm = cfg.get("review_model")
    if rm is not None and not isinstance(rm, str):
        raise ConfigError("review_model must be a string")

    rp = cfg.get("repeat")
    if rp is not None and (not isinstance(rp, int) or isinstance(rp, bool) or rp < 1):
        raise ConfigError("repeat must be a positive integer")

    db = cfg.get("db")
    if db is not None and not isinstance(db, str):
        raise ConfigError("db must be a string")

    dl = cfg.get("delay")
    if dl is not None and (
        not isinstance(dl, int | float) or isinstance(dl, bool) or dl < 0
    ):
        raise ConfigError("delay must be a non-negative number")


def load_config(
    project: Path | None = None, home: Path | None = None
) -> dict[str, Any]:
    """Load and merge home then project config (project wins). ``{}`` if none."""
    project = PROJECT_CONFIG if project is None else project
    home = HOME_CONFIG if home is None else home
    merged: dict[str, Any] = {}
    merged.update(_load_file(home))
    merged.update(_load_file(project))
    _validate(merged)
    return merged


def resolve(cli_value: Any, key: str, cfg: dict[str, Any], default: Any) -> Any:
    """Precedence: explicit CLI value > config[key] > default.

    ``cli_value`` counts as "not supplied" when it is None or an empty tuple/list
    (Click's sentinel for an unset option / unused ``multiple`` flag).
    """
    if cli_value is not None and cli_value != () and cli_value != []:
        return cli_value
    if cfg.get(key) is not None:
        return cfg[key]
    return default
