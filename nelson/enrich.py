"""Enrichment: resolve a seed advisory to repo + fix commit + CWE.

The CVD seed carries only an identifier and a bug description. To derive ground
truth we need the repository, the fix commit, and (for reporting) the CWE. OSV is
the primary source — its ``affected[].ranges`` of type GIT carry the repo URL and
the *fixed* commit SHA directly; otherwise we recover the fix commit from
GitHub ``/commit/`` reference URLs. NVD is a CWE-only fallback for advisories OSV
hasn't classified.

All network access goes through an injectable :class:`HttpClient`, so the
enrichers are unit-tested against captured fixtures with no live calls. A lookup
that 404s (e.g. an advisory too fresh for OSV) yields no updates and leaves the
case a candidate — enrichment is meant to be re-run as the sources catch up.
"""

from __future__ import annotations

import dataclasses
import re
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import httpx

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from .corpus import Case

OSV_VULN_URL = "https://api.osv.dev/v1/vulns/"
NVD_CVE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"

_GITHUB_COMMIT_RE = re.compile(
    r"github\.com/([^/\s]+)/([^/\s]+?)(?:\.git)?/commit/([0-9a-f]{7,40})"
)

# OSV returns 404 for an id it doesn't index directly but names the aliases it
# *does* index in the error body, e.g.
#   {"code":5,"message":"Bug not found, but the following aliases were: GHSA-..."}
# Many fresh CVEs are only indexed under their GHSA alias (which carries the GIT
# range / commit URL), so we follow the alias on a miss.
_GHSA_ID_RE = re.compile(r"GHSA-[0-9a-z]{4}-[0-9a-z]{4}-[0-9a-z]{4}", re.IGNORECASE)


@runtime_checkable
class HttpClient(Protocol):
    """Minimal JSON GET. Returns (status_code, parsed_body|None).

    Status 0 signals a transport failure (DNS, connection, timeout) so callers
    treat it like 'not found' rather than crashing the corpus build.
    """

    def get_json(
        self, url: str, params: dict[str, str] | None = None
    ) -> tuple[int, Any]: ...


class HttpxClient:
    """Default :class:`HttpClient` backed by httpx."""

    def __init__(self, timeout: float = 20.0, headers: dict[str, str] | None = None):
        self.timeout = timeout
        # A descriptive UA is polite to OSV/NVD and avoids some default blocks.
        self.headers = {"User-Agent": "nelson-benchmark/0.1"} | (headers or {})

    def get_json(
        self, url: str, params: dict[str, str] | None = None
    ) -> tuple[int, Any]:
        try:
            resp = httpx.get(
                url, params=params, timeout=self.timeout, headers=self.headers
            )
        except httpx.HTTPError:
            return (0, None)
        try:
            return (resp.status_code, resp.json())
        except ValueError:
            return (resp.status_code, None)


def fetch_cvd_payload(url: str, client: HttpClient) -> dict[str, Any]:
    """Fetch the Anthropic CVD ``payload.json`` (used by the import CLI)."""
    status, data = client.get_json(url)
    if status != 200 or not isinstance(data, dict):
        raise RuntimeError(f"CVD payload fetch failed: HTTP {status} from {url}")
    return data


# -- Enrichers ---------------------------------------------------------------


@runtime_checkable
class Enricher(Protocol):
    name: str

    def enrich(self, case: Case) -> dict[str, Any]:
        """Return field updates for the case (empty dict if nothing found)."""
        ...


def _git_range_fix(osv: dict[str, Any]) -> tuple[str, str] | None:
    """Repo URL + fixed commit SHA from a GIT-type range, if present.

    This is the most reliable signal: the range's ``repo`` is the upstream git
    URL and the ``fixed`` event is an actual commit SHA (not a version).
    """
    for affected in osv.get("affected", []):
        for rng in affected.get("ranges", []):
            if rng.get("type") != "GIT":
                continue
            repo = rng.get("repo")
            for event in rng.get("events", []):
                if event.get("fixed") and repo:
                    return (repo, event["fixed"])
    return None


def _refs_fix(references: Iterable[dict[str, Any]]) -> tuple[str, str] | None:
    """Repo URL + fix commit recovered from a GitHub /commit/ reference URL."""
    for ref in references:
        m = _GITHUB_COMMIT_RE.search(ref.get("url", ""))
        if m:
            owner, repo, sha = m.groups()
            return (f"https://github.com/{owner}/{repo}", sha)
    return None


class OSVEnricher:
    """Enrich from OSV (api.osv.dev): repo, fix commit, CWE, CVE<->GHSA aliases."""

    name = "osv"

    def __init__(self, client: HttpClient):
        self.client = client

    def _lookup(self, ext_id: str) -> tuple[dict[str, Any] | None, Any]:
        """Fetch one OSV record by id.

        Returns ``(record, body)`` where ``record`` is the parsed vuln dict on a
        200 and ``None`` otherwise; ``body`` is the raw parsed response (the 404
        error body names aliases we can follow).
        """
        status, body = self.client.get_json(OSV_VULN_URL + ext_id)
        if status == 200 and isinstance(body, dict):
            return body, body
        return None, body

    def enrich(self, case: Case) -> dict[str, Any]:
        osv, body = self._lookup(case.ext_id)
        if osv is None:
            # Direct miss: OSV's 404 body may name a GHSA alias it *does* index.
            # Re-query it (a known ghsa_id first, else one parsed from the error
            # message) — that record usually carries the GIT range/commit URL.
            alias = case.ghsa_id
            if not alias and isinstance(body, dict):
                m = _GHSA_ID_RE.search(body.get("message", ""))
                if m:
                    alias = m.group(0)
            if alias and alias != case.ext_id:
                osv, _ = self._lookup(alias)
        if osv is None:
            return {}

        updates: dict[str, Any] = {}

        # Cross-reference CVE <-> GHSA via aliases (only fill what's missing).
        for alias in osv.get("aliases") or []:
            if alias.startswith("CVE-") and not case.cve_id:
                updates.setdefault("cve_id", alias)
            elif alias.startswith("GHSA-") and not case.ghsa_id:
                updates.setdefault("ghsa_id", alias)

        # Repo + fix commit: GIT range first, then /commit/ references.
        fix = _git_range_fix(osv) or _refs_fix(osv.get("references", []))
        if fix:
            repo, sha = fix
            if not case.repo_url:
                updates["repo_url"] = repo
            if not case.fix_commit:
                updates["fix_commit"] = sha

        cwe_ids = osv.get("database_specific", {}).get("cwe_ids") or []
        if cwe_ids and not case.cwe:
            updates["cwe"] = ", ".join(cwe_ids)

        return updates


class NVDEnricher:
    """CWE-only fallback from NVD, for advisories OSV hasn't classified."""

    name = "nvd"

    def __init__(self, client: HttpClient, api_key: str | None = None):
        self.client = client
        self.api_key = api_key

    def enrich(self, case: Case) -> dict[str, Any]:
        # Needs a CVE id (NVD is CVE-keyed) and only fills a missing CWE.
        if not case.cve_id or case.cwe:
            return {}
        status, data = self.client.get_json(NVD_CVE_URL, {"cveId": case.cve_id})
        if status != 200 or not isinstance(data, dict):
            return {}
        cwes: list[str] = []
        for vuln in data.get("vulnerabilities", []):
            for weakness in vuln.get("cve", {}).get("weaknesses", []):
                for desc in weakness.get("description", []):
                    value = desc.get("value", "")
                    if value.startswith("CWE-") and value not in cwes:
                        cwes.append(value)
        return {"cwe": ", ".join(cwes)} if cwes else {}


def enrich_case(case: Case, enrichers: Sequence[Enricher]) -> dict[str, Any]:
    """Run enrichers in order, each seeing prior updates, and merge the result.

    Returns the combined field updates (to persist via ``update_case_fields``).
    Order matters: put the authoritative source first so a later fallback only
    fills genuinely-missing fields.
    """
    working = dataclasses.replace(case)
    merged: dict[str, Any] = {}
    for enricher in enrichers:
        for key, value in enricher.enrich(working).items():
            setattr(working, key, value)
            merged[key] = value
    return merged
