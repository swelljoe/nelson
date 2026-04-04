"""Detect security-relevant static analysis tooling in a project."""

import configparser
import json
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


@dataclass
class SecurityTool:
    name: str
    description: str
    languages: set[str]
    cwe_coverage: list[str]  # CWE IDs this tool can detect
    install_hint: str


@dataclass
class ToolStatus:
    tool: SecurityTool
    present: bool
    security_rules_enabled: bool | None  # None = not applicable / couldn't determine
    detail: str  # human-readable explanation of what was found


# -- Recommended tools per language, with CWE coverage --

RECOMMENDED_TOOLS: list[SecurityTool] = [
    # Python
    SecurityTool(
        name="ruff (S rules)",
        description="Fast Python linter — S rule set replaces Bandit for security checks",
        languages={"python"},
        cwe_coverage=["CWE-78", "CWE-89", "CWE-94", "CWE-502", "CWE-798", "CWE-276"],
        install_hint='Add to pyproject.toml: [tool.ruff.lint] select = ["S"]',
    ),
    SecurityTool(
        name="semgrep",
        description="Multi-language static analysis with community security rulesets",
        languages={"python", "typescript", "javascript", "go", "java", "ruby"},
        cwe_coverage=[
            "CWE-78", "CWE-79", "CWE-89", "CWE-94", "CWE-22", "CWE-502",
            "CWE-918", "CWE-798", "CWE-77",
        ],
        install_hint="Add .semgrep.yml or run: semgrep --config=auto",
    ),
    # Go
    SecurityTool(
        name="golangci-lint (gosec)",
        description="Go linter aggregator — gosec linter checks for security issues",
        languages={"go"},
        cwe_coverage=["CWE-78", "CWE-89", "CWE-22", "CWE-798", "CWE-276", "CWE-362"],
        install_hint="Add gosec to .golangci.yml: linters.enable: [gosec]",
    ),
    # TypeScript / JavaScript
    SecurityTool(
        name="eslint-plugin-security",
        description="ESLint plugin for Node.js security checks",
        languages={"typescript", "javascript"},
        cwe_coverage=["CWE-78", "CWE-89", "CWE-94", "CWE-22"],
        install_hint="npm install --save-dev eslint-plugin-security",
    ),
    # C / C++
    SecurityTool(
        name="clang-tidy (security checks)",
        description="Clang-based C/C++ linter with security-relevant checks",
        languages={"c", "cpp"},
        cwe_coverage=[
            "CWE-787", "CWE-416", "CWE-125", "CWE-476", "CWE-190", "CWE-119",
        ],
        install_hint="Add .clang-tidy with security checks enabled",
    ),
    SecurityTool(
        name="cppcheck",
        description="Static analysis for C/C++ focusing on bugs and undefined behavior",
        languages={"c", "cpp"},
        cwe_coverage=["CWE-787", "CWE-125", "CWE-476", "CWE-190", "CWE-119"],
        install_hint="Install cppcheck and run: cppcheck --enable=warning,security .",
    ),
    # Rust
    SecurityTool(
        name="clippy",
        description="Rust linter (included with rustup) — catches some security patterns",
        languages={"rust"},
        cwe_coverage=["CWE-190", "CWE-362"],
        install_hint="Run: cargo clippy",
    ),
]


def _read_toml(path: Path) -> dict | None:
    if tomllib is None:
        return None
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception:
        return None


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _file_contains(path: Path, *needles: str) -> bool:
    """Check if a file contains any of the given strings."""
    try:
        text = path.read_text(errors="ignore")
        return any(n in text for n in needles)
    except OSError:
        return False


def _check_ruff(target: Path) -> ToolStatus:
    tool = next(t for t in RECOMMENDED_TOOLS if t.name == "ruff (S rules)")

    # Check pyproject.toml
    pyproject = target / "pyproject.toml"
    ruff_toml = target / "ruff.toml"
    dot_ruff = target / ".ruff.toml"

    ruff_config = None
    config_source = None

    if pyproject.exists():
        data = _read_toml(pyproject)
        if data:
            ruff_config = data.get("tool", {}).get("ruff", {})
            config_source = "pyproject.toml"

    if ruff_config is None and ruff_toml.exists():
        ruff_config = _read_toml(ruff_toml)
        config_source = "ruff.toml"

    if ruff_config is None and dot_ruff.exists():
        ruff_config = _read_toml(dot_ruff)
        config_source = ".ruff.toml"

    if ruff_config is None:
        return ToolStatus(
            tool=tool, present=False, security_rules_enabled=None,
            detail="No Ruff configuration found",
        )

    # Check if S rules are enabled
    lint_config = ruff_config.get("lint", ruff_config)
    select = lint_config.get("select", [])
    extend_select = lint_config.get("extend-select", [])
    all_selected = select + extend_select

    # Check for S rules: "S", "S1", "S100", etc.
    s_enabled = any(
        rule == "S" or rule.startswith("S1") or rule.startswith("S2")
        or rule.startswith("S3") or rule.startswith("S4") or rule.startswith("S5")
        or rule.startswith("S6") or rule.startswith("S7")
        for rule in all_selected
    )

    # "ALL" also enables S rules (unless explicitly ignored)
    if "ALL" in all_selected:
        ignore = lint_config.get("ignore", [])
        extend_ignore = lint_config.get("extend-ignore", [])
        all_ignored = ignore + extend_ignore
        s_ignored = any(r == "S" or r.startswith("S1") for r in all_ignored)
        s_enabled = not s_ignored

    if s_enabled:
        return ToolStatus(
            tool=tool, present=True, security_rules_enabled=True,
            detail=f"Ruff with S (Bandit) rules enabled in {config_source}",
        )
    else:
        return ToolStatus(
            tool=tool, present=True, security_rules_enabled=False,
            detail=f"Ruff found in {config_source} but S (Bandit) rules are NOT enabled",
        )


def _check_semgrep(target: Path) -> ToolStatus:
    tool = next(t for t in RECOMMENDED_TOOLS if t.name == "semgrep")

    candidates = [
        target / ".semgrep.yml",
        target / ".semgrep.yaml",
        target / ".semgrep",
    ]
    # Also check CI workflows
    ci_dir = target / ".github" / "workflows"

    found_config = any(c.exists() for c in candidates)
    found_ci = False
    if ci_dir.is_dir():
        for wf in ci_dir.iterdir():
            if wf.suffix in (".yml", ".yaml") and _file_contains(wf, "semgrep"):
                found_ci = True
                break

    if found_config:
        return ToolStatus(
            tool=tool, present=True, security_rules_enabled=None,
            detail="Semgrep configuration found",
        )
    if found_ci:
        return ToolStatus(
            tool=tool, present=True, security_rules_enabled=None,
            detail="Semgrep found in CI workflows",
        )
    return ToolStatus(
        tool=tool, present=False, security_rules_enabled=None,
        detail="No Semgrep configuration or CI usage found",
    )


def _check_golangci_lint(target: Path) -> ToolStatus:
    tool = next(t for t in RECOMMENDED_TOOLS if t.name == "golangci-lint (gosec)")

    candidates = [
        target / ".golangci.yml",
        target / ".golangci.yaml",
        target / ".golangci.toml",
        target / ".golangci.json",
    ]

    for cfg in candidates:
        if cfg.exists():
            has_gosec = _file_contains(cfg, "gosec")
            return ToolStatus(
                tool=tool, present=True,
                security_rules_enabled=has_gosec if has_gosec else False,
                detail=f"golangci-lint config found ({cfg.name})"
                + (", gosec enabled" if has_gosec else ", gosec NOT detected in config"),
            )

    return ToolStatus(
        tool=tool, present=False, security_rules_enabled=None,
        detail="No golangci-lint configuration found",
    )


def _check_eslint_security(target: Path) -> ToolStatus:
    tool = next(t for t in RECOMMENDED_TOOLS if t.name == "eslint-plugin-security")

    # Check package.json for the plugin
    pkg_json = target / "package.json"
    if pkg_json.exists():
        data = _read_json(pkg_json)
        if data:
            all_deps = {}
            for dep_key in ("dependencies", "devDependencies", "peerDependencies"):
                all_deps.update(data.get(dep_key, {}))
            if "eslint-plugin-security" in all_deps:
                return ToolStatus(
                    tool=tool, present=True, security_rules_enabled=True,
                    detail="eslint-plugin-security found in package.json",
                )

    # Check eslint config files
    eslint_configs = [
        target / ".eslintrc.json",
        target / ".eslintrc.js",
        target / ".eslintrc.yml",
        target / ".eslintrc.yaml",
        target / "eslint.config.js",
        target / "eslint.config.mjs",
        target / "eslint.config.ts",
    ]
    for cfg in eslint_configs:
        if cfg.exists() and _file_contains(cfg, "security"):
            return ToolStatus(
                tool=tool, present=True, security_rules_enabled=True,
                detail=f"ESLint security plugin referenced in {cfg.name}",
            )

    return ToolStatus(
        tool=tool, present=False, security_rules_enabled=None,
        detail="No eslint-plugin-security found",
    )


def _check_clang_tidy(target: Path) -> ToolStatus:
    tool = next(t for t in RECOMMENDED_TOOLS if t.name == "clang-tidy (security checks)")
    cfg = target / ".clang-tidy"
    if cfg.exists():
        return ToolStatus(
            tool=tool, present=True, security_rules_enabled=None,
            detail=".clang-tidy configuration found",
        )
    return ToolStatus(
        tool=tool, present=False, security_rules_enabled=None,
        detail="No .clang-tidy configuration found",
    )


def _check_cppcheck(target: Path) -> ToolStatus:
    tool = next(t for t in RECOMMENDED_TOOLS if t.name == "cppcheck")

    # Check CI, Makefile, CMakeLists for cppcheck usage
    makefile = target / "Makefile"
    cmake = target / "CMakeLists.txt"
    ci_dir = target / ".github" / "workflows"

    for f in [makefile, cmake]:
        if f.exists() and _file_contains(f, "cppcheck"):
            return ToolStatus(
                tool=tool, present=True, security_rules_enabled=None,
                detail=f"cppcheck found in {f.name}",
            )

    if ci_dir.is_dir():
        for wf in ci_dir.iterdir():
            if wf.suffix in (".yml", ".yaml") and _file_contains(wf, "cppcheck"):
                return ToolStatus(
                    tool=tool, present=True, security_rules_enabled=None,
                    detail="cppcheck found in CI workflows",
                )

    return ToolStatus(
        tool=tool, present=False, security_rules_enabled=None,
        detail="No cppcheck usage found",
    )


def _check_clippy(target: Path) -> ToolStatus:
    tool = next(t for t in RECOMMENDED_TOOLS if t.name == "clippy")

    # If there's a Cargo.toml, clippy is almost certainly available
    cargo = target / "Cargo.toml"
    if not cargo.exists():
        return ToolStatus(
            tool=tool, present=False, security_rules_enabled=None,
            detail="No Cargo.toml found",
        )

    # Check if clippy is used in CI or Makefile
    ci_dir = target / ".github" / "workflows"
    makefile = target / "Makefile"

    for f in [makefile]:
        if f.exists() and _file_contains(f, "clippy"):
            return ToolStatus(
                tool=tool, present=True, security_rules_enabled=None,
                detail=f"clippy found in {f.name}",
            )

    if ci_dir.is_dir():
        for wf in ci_dir.iterdir():
            if wf.suffix in (".yml", ".yaml") and _file_contains(wf, "clippy"):
                return ToolStatus(
                    tool=tool, present=True, security_rules_enabled=None,
                    detail="clippy found in CI workflows",
                )

    return ToolStatus(
        tool=tool, present=False, security_rules_enabled=None,
        detail="Cargo.toml exists but no clippy usage found in CI or Makefile",
    )


# Map checker function to tool name
_CHECKERS: dict[str, callable] = {
    "ruff (S rules)": _check_ruff,
    "semgrep": _check_semgrep,
    "golangci-lint (gosec)": _check_golangci_lint,
    "eslint-plugin-security": _check_eslint_security,
    "clang-tidy (security checks)": _check_clang_tidy,
    "cppcheck": _check_cppcheck,
    "clippy": _check_clippy,
}


def assess_tooling(target_dir: str | Path, languages: set[str]) -> list[ToolStatus]:
    """Assess security tooling for the given project and languages.

    Returns a ToolStatus for each relevant tool (tools for languages not
    present in the project are excluded).
    """
    target = Path(target_dir).resolve()
    results = []

    for tool in RECOMMENDED_TOOLS:
        # Only check tools relevant to the project's languages
        if not tool.languages & languages:
            continue

        checker = _CHECKERS.get(tool.name)
        if checker:
            results.append(checker(target))
        else:
            results.append(ToolStatus(
                tool=tool, present=False, security_rules_enabled=None,
                detail="Detection not implemented",
            ))

    return results


def format_tooling_report(statuses: list[ToolStatus]) -> str:
    """Format tooling assessment as a CLI report section."""
    if not statuses:
        return ""

    lines = ["Security Tooling Assessment", "=" * 40]

    present = [s for s in statuses if s.present]
    missing = [s for s in statuses if not s.present]
    warnings = [s for s in present if s.security_rules_enabled is False]

    if present:
        lines.append("")
        lines.append("Detected:")
        for s in present:
            check = "✓" if s.security_rules_enabled is not False else "⚠"
            lines.append(f"  {check} {s.tool.name} — {s.detail}")

    if warnings:
        lines.append("")
        lines.append("Warnings:")
        for s in warnings:
            lines.append(f"  ⚠ {s.tool.name}: {s.detail}")
            lines.append(f"    → {s.tool.install_hint}")

    if missing:
        lines.append("")
        lines.append("Recommended:")
        for s in missing:
            covered = ", ".join(s.tool.cwe_coverage[:4])
            if len(s.tool.cwe_coverage) > 4:
                covered += f", +{len(s.tool.cwe_coverage) - 4} more"
            lines.append(f"  ✗ {s.tool.name} — {s.tool.description}")
            lines.append(f"    Covers: {covered}")
            lines.append(f"    → {s.tool.install_hint}")

    if not missing and not warnings:
        lines.append("")
        lines.append("All recommended security tooling is in place.")

    return "\n".join(lines)
