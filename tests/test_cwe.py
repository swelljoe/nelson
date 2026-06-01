"""CWE applicability + name resolution."""

from nelson.cwe import CWE_TOP_25, applicable_cwes, cwe_name


def test_cwe_top_25_has_25_entries():
    assert len({c.id for c in CWE_TOP_25}) == 25


def test_rust_gets_web_logic_cwes_but_not_pure_memory_safety():
    rust = {c.id for c in applicable_cwes("rust")}
    # Web/logic classes a Rust backend is genuinely subject to (the GHSA-f26g
    # authorization case is why this was fixed); includes 2025-edition authz
    # additions 284/639 that list rust inline.
    for cid in ("CWE-862", "CWE-863", "CWE-639", "CWE-284", "CWE-918", "CWE-306"):
        assert cid in rust
    # Safe Rust rules out the memory-safety classes; they stay c/cpp only.
    for cid in ("CWE-787", "CWE-416", "CWE-125", "CWE-122"):
        assert cid not in rust


def test_cwe_name_resolves_top25_and_supplemental_and_unknown():
    assert cwe_name("CWE-22") == "Path Traversal"  # in the Top 25
    # Supplemental: a corpus class outside the Top 25 still resolves a name.
    assert cwe_name("CWE-349") == (
        "Acceptance of Extraneous Untrusted Data With Trusted Data"
    )
    assert cwe_name("CWE-9999") is None  # unknown
