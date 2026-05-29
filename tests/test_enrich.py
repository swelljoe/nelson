"""Enrichment from OSV/NVD, exercised against captured fixtures (no network)."""

import json
from pathlib import Path

from nelson.corpus import Case
from nelson.enrich import (
    NVD_CVE_URL,
    OSV_VULN_URL,
    HttpClient,
    NVDEnricher,
    OSVEnricher,
    enrich_case,
)

FX = Path(__file__).parent / "fixtures"


def _fx(name):
    return json.loads((FX / f"{name}.json").read_text())


class FakeHttp:
    """Routes URLs to canned (status, json); params are ignored (one NVD URL)."""

    def __init__(self, routes: dict[str, tuple[int, object]]):
        self.routes = routes
        self.calls: list[str] = []

    def get_json(self, url, params=None):
        self.calls.append(url)
        return self.routes.get(url, (404, {"code": 5, "message": "Bug not found."}))


def test_is_http_client_protocol():
    assert isinstance(FakeHttp({}), HttpClient)


def test_osv_git_range_yields_repo_and_fix_commit():
    case = Case(source="cvd", ext_id="OSV-2017-1")
    http = FakeHttp({OSV_VULN_URL + "OSV-2017-1": (200, _fx("osv_git_range"))})
    updates = OSVEnricher(http).enrich(case)
    assert updates["repo_url"] == "https://github.com/curl/curl.git"
    assert updates["fix_commit"] == "544bfdebea2a9e8be1c01fc7954cd49638fe2803"


def test_osv_aliases_and_cwe_without_commit():
    # log4shell: a CVE alias and populated cwe_ids, but the OSV record has no
    # GIT range and no /commit/ reference (Maven ecosystem advisory). So it
    # enriches CVE+CWE yet yields no fix commit -> not derivable, stays candidate.
    case = Case(source="cvd", ext_id="GHSA-jfh8-c2jp-5v3q")
    http = FakeHttp(
        {OSV_VULN_URL + "GHSA-jfh8-c2jp-5v3q": (200, _fx("osv_aliases_cwe"))}
    )
    updates = OSVEnricher(http).enrich(case)
    assert updates["cve_id"] == "CVE-2021-44228"
    assert "CWE-502" in updates["cwe"]
    assert "fix_commit" not in updates
    assert "repo_url" not in updates


def test_osv_reference_commit_path():
    # moby: no GIT range, but a GitHub /commit/ reference -> repo + fix commit.
    case = Case(source="cvd", ext_id="GHSA-jq35-85cj-fj4p")
    http = FakeHttp({OSV_VULN_URL + "GHSA-jq35-85cj-fj4p": (200, _fx("osv_refs_only"))})
    updates = OSVEnricher(http).enrich(case)
    assert updates["fix_commit"]
    assert updates["repo_url"] == "https://github.com/moby/moby"


def test_osv_not_found_yields_no_updates():
    # An advisory too fresh for OSV (the 2026 seed reality) stays a candidate.
    case = Case(source="cvd", ext_id="CVE-2026-27654")
    http = FakeHttp({})  # everything 404s
    assert OSVEnricher(http).enrich(case) == {}


def test_osv_follows_ghsa_alias_from_404_message():
    # The 2026 seed reality: the CVE id 404s, but OSV names the GHSA alias it
    # *does* index in the error body, and that record carries the /commit/ ref.
    case = Case(source="cvd", ext_id="CVE-2026-7474")
    http = FakeHttp(
        {
            OSV_VULN_URL + "CVE-2026-7474": (
                404,
                {
                    "code": 5,
                    "message": (
                        "Bug not found, but the following aliases were: "
                        "GHSA-jq35-85cj-fj4p"
                    ),
                },
            ),
            OSV_VULN_URL + "GHSA-jq35-85cj-fj4p": (200, _fx("osv_refs_only")),
        }
    )
    updates = OSVEnricher(http).enrich(case)
    assert updates["fix_commit"]
    assert updates["repo_url"] == "https://github.com/moby/moby"


def test_osv_follows_known_ghsa_id_when_ext_id_misses():
    # If the case already carries a ghsa_id (from the seed), follow it directly
    # without needing the 404 body to name an alias.
    case = Case(
        source="cvd", ext_id="CVE-2026-7474", ghsa_id="GHSA-jq35-85cj-fj4p"
    )
    http = FakeHttp(
        {OSV_VULN_URL + "GHSA-jq35-85cj-fj4p": (200, _fx("osv_refs_only"))}
    )
    updates = OSVEnricher(http).enrich(case)
    assert updates["repo_url"] == "https://github.com/moby/moby"


def test_osv_alias_miss_still_yields_nothing():
    # CVE 404s with no alias in the body -> no second lookup, no updates.
    case = Case(source="cvd", ext_id="CVE-2026-27654")
    http = FakeHttp({})  # default 404 body has no aliases
    assert OSVEnricher(http).enrich(case) == {}
    assert http.calls == [OSV_VULN_URL + "CVE-2026-27654"]


def test_osv_does_not_overwrite_existing_values():
    case = Case(
        source="manual",
        ext_id="OSV-2017-1",
        repo_url="https://example.test/mine",
        fix_commit="manualsha",
    )
    http = FakeHttp({OSV_VULN_URL + "OSV-2017-1": (200, _fx("osv_git_range"))})
    updates = OSVEnricher(http).enrich(case)
    assert "repo_url" not in updates
    assert "fix_commit" not in updates


def test_nvd_fills_missing_cwe():
    case = Case(source="cvd", ext_id="GHSA-x", cve_id="CVE-2021-44228")
    http = FakeHttp({NVD_CVE_URL: (200, _fx("nvd_cve"))})
    updates = NVDEnricher(http).enrich(case)
    assert "CWE-502" in updates["cwe"]


def test_nvd_skips_when_cwe_present_or_no_cve():
    http = FakeHttp({NVD_CVE_URL: (200, _fx("nvd_cve"))})
    # Already classified: don't call NVD.
    assert (
        NVDEnricher(http).enrich(
            Case(source="x", ext_id="a", cve_id="CVE-2021-44228", cwe="CWE-79")
        )
        == {}
    )
    # No CVE id: NVD is CVE-keyed.
    assert NVDEnricher(http).enrich(Case(source="x", ext_id="GHSA-only")) == {}
    assert http.calls == []


def test_enrich_case_chains_osv_then_nvd():
    # OSV (moby) gives repo+fix but empty cwe; NVD then fills the CWE.
    case = Case(source="cvd", ext_id="GHSA-jq35-85cj-fj4p", cve_id="CVE-2021-44228")
    http = FakeHttp(
        {
            OSV_VULN_URL + "GHSA-jq35-85cj-fj4p": (200, _fx("osv_refs_only")),
            NVD_CVE_URL: (200, _fx("nvd_cve")),
        }
    )
    merged = enrich_case(case, [OSVEnricher(http), NVDEnricher(http)])
    assert merged["fix_commit"]
    assert merged["repo_url"].startswith("https://github.com/")
    assert "CWE-" in merged["cwe"]  # came from the NVD fallback
