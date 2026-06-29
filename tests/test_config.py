"""nelson.yaml loading, precedence, and validation."""

import pytest

from nelson.config import ConfigError, load_config, resolve


def _write(path, text):
    path.write_text(text)
    return path


def test_missing_files_yield_empty_config(tmp_path):
    cfg = load_config(project=tmp_path / "none.yaml", home=tmp_path / "nohome.yaml")
    assert cfg == {}


def test_project_overrides_home(tmp_path):
    home = _write(
        tmp_path / "home.yaml",
        "scan_models: [a, b]\nreview_model: home-judge\nrepeat: 5\n",
    )
    project = _write(
        tmp_path / "nelson.yaml",
        "review_model: project-judge\n",
    )
    cfg = load_config(project=project, home=home)
    # project wins for review_model; home contributes the keys project omits.
    assert cfg["review_model"] == "project-judge"
    assert cfg["scan_models"] == ["a", "b"]
    assert cfg["repeat"] == 5


def test_empty_file_is_empty_config(tmp_path):
    project = _write(tmp_path / "nelson.yaml", "")
    cfg = load_config(project=project, home=tmp_path / "nohome.yaml")
    assert cfg == {}


def test_unknown_key_is_error(tmp_path):
    project = _write(tmp_path / "nelson.yaml", "bogus: 1\n")
    with pytest.raises(ConfigError, match="unknown config key"):
        load_config(project=project, home=tmp_path / "nohome.yaml")


def test_non_mapping_top_level_is_error(tmp_path):
    project = _write(tmp_path / "nelson.yaml", "- just\n- a\n- list\n")
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_config(project=project, home=tmp_path / "nohome.yaml")


@pytest.mark.parametrize(
    "body, msg",
    [
        ("scan_models: not-a-list\n", "scan_models"),
        ("scan_models: [1, 2]\n", "scan_models"),
        ("review_model: [a]\n", "review_model"),
        ("repeat: 0\n", "repeat"),
        ("repeat: -1\n", "repeat"),
        ("repeat: 1.5\n", "repeat"),
        ("delay: -2\n", "delay"),
        ("db: 5\n", "db"),
    ],
)
def test_type_validation(tmp_path, body, msg):
    project = _write(tmp_path / "nelson.yaml", body)
    with pytest.raises(ConfigError, match=msg):
        load_config(project=project, home=tmp_path / "nohome.yaml")


def test_resolve_precedence():
    cfg = {"repeat": 7, "review_model": "cfg-judge"}
    # explicit CLI value wins
    assert resolve(9, "repeat", cfg, 3) == 9
    # config used when CLI is the "unset" sentinel
    assert resolve(None, "repeat", cfg, 3) == 7
    assert resolve((), "scan_models", cfg, ["d"]) == ["d"]  # key absent -> default
    # default when neither CLI nor config supply it
    assert resolve(None, "delay", cfg, 2.0) == 2.0
    # empty tuple counts as unset for multiple= options
    assert resolve((), "review_model", cfg, "x") == "cfg-judge"
