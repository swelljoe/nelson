"""haha command: config resolution + the >=2-scan / >=1-review hard gate."""

import contextlib

from click.testing import CliRunner

from nelson import cli
from nelson import config as config_mod
from nelson.inventory import SourceFile


def _isolate_config(monkeypatch, tmp_path):
    """Point config lookup at empty paths so the test ignores real nelson.yaml."""
    monkeypatch.setattr(config_mod, "PROJECT_CONFIG", tmp_path / "nelson.yaml")
    monkeypatch.setattr(config_mod, "HOME_CONFIG", tmp_path / "home.yaml")


def _combined(result):
    err = ""
    with contextlib.suppress(ValueError):
        err = result.stderr or ""
    return result.output + err


def test_haha_requires_two_scan_models(tmp_path, monkeypatch):
    _isolate_config(monkeypatch, tmp_path)
    (tmp_path / "a.py").write_text("x = 1\n")
    result = CliRunner().invoke(
        cli.main,
        [
            "haha",
            str(tmp_path),
            "--scan-model",
            "claude:haiku",
            "--review-model",
            "claude:sonnet",
            "--db",
            str(tmp_path / "n.db"),
        ],
    )
    assert result.exit_code == 1
    assert "at least 2 scan models" in _combined(result)


def test_haha_requires_review_model(tmp_path, monkeypatch):
    _isolate_config(monkeypatch, tmp_path)
    (tmp_path / "a.py").write_text("x = 1\n")
    result = CliRunner().invoke(
        cli.main,
        [
            "haha",
            str(tmp_path),
            "--scan-model",
            "claude:haiku",
            "--scan-model",
            "claude:sonnet",
            "--db",
            str(tmp_path / "n.db"),
        ],
    )
    assert result.exit_code == 1
    assert "needs a review model" in _combined(result)


def test_haha_resolves_from_config_and_runs(tmp_path, monkeypatch):
    # A config supplying scan_models + review_model satisfies the gate with no
    # CLI model flags; the pipeline internals are stubbed so no model runs.
    cfg = tmp_path / "nelson.yaml"
    cfg.write_text(
        "scan_models: [claude:haiku, claude:sonnet]\nreview_model: claude:opus\n"
    )
    monkeypatch.setattr(config_mod, "PROJECT_CONFIG", cfg)
    monkeypatch.setattr(config_mod, "HOME_CONFIG", tmp_path / "home.yaml")
    (tmp_path / "a.py").write_text("x = 1\n")

    captured = {}

    def fake_create_scan(db, target_dir, models, repeat, files):
        captured["models"] = models
        captured["repeat"] = repeat
        return 1, [SourceFile("a.py", "python", 4)]

    monkeypatch.setattr(cli, "create_scan", fake_create_scan)
    monkeypatch.setattr(cli, "run_scan", lambda *a, **k: None)
    monkeypatch.setattr(cli, "run_review", lambda *a, **k: None)

    result = CliRunner().invoke(
        cli.main, ["haha", str(tmp_path), "--db", str(tmp_path / "n.db")]
    )
    assert result.exit_code == 0, _combined(result)
    assert captured["models"] == ["claude:haiku", "claude:sonnet"]
    assert captured["repeat"] == 3  # default applied
