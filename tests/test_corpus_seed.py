"""CVD seed parsing, seed import, and manifest round-trip."""

import json
from pathlib import Path

from nelson.corpus import (
    Case,
    CVDSeedSource,
    SeedSource,
    case_from_manifest,
    case_to_manifest,
    import_seeds,
    load_manifest_dir,
    max_severity,
    write_case_manifest,
)
from nelson.db import Database

FIXTURE = Path(__file__).parent / "fixtures" / "cvd_payload.json"


def _payload():
    return json.loads(FIXTURE.read_text())


def test_cvd_source_satisfies_protocol():
    src = CVDSeedSource(_payload())
    assert isinstance(src, SeedSource)
    assert src.name == "cvd"


def test_max_severity_picks_worst_and_ignores_unknown():
    assert max_severity(["low", "critical", "high"]) == "critical"
    assert max_severity([None, "medium", ""]) == "medium"
    assert max_severity([None, "bogus"]) is None


def test_cvd_record_becomes_seed_case():
    src = CVDSeedSource(_payload())
    cases = {c.ext_id: c for c in src.seeds()}

    nginx = cases["CVE-2026-27654"]
    assert nginx.source == "cvd"
    assert nginx.cve_id == "CVE-2026-27654"
    assert nginx.ghsa_id is None
    assert nginx.project == "nginx"
    assert nginx.bug_class == "heap-buffer-overflow"  # primary (first) finding
    # Multiple findings -> headline severity is the worst across them.
    assert nginx.severity == "critical"
    # Every finding's ant_id is carried.
    assert nginx.ant_id is not None
    assert "ANT-2026-HY56VRSB" in nginx.ant_id
    assert "ANT-2026-VS18SA90" in nginx.ant_id
    assert nginx.disclosure_date == "2026-05-20T07:40:37.681447Z"
    # Freshly seeded: no ground truth yet, stays a candidate.
    assert nginx.status == "candidate"
    assert nginx.repo_url is None and nginx.fix_commit is None


def test_ghsa_record_sets_ghsa_id_not_cve():
    src = CVDSeedSource(_payload())
    ghsa = next(c for c in src.seeds() if c.ext_id.startswith("GHSA-"))
    assert ghsa.ghsa_id == ghsa.ext_id
    assert ghsa.cve_id is None


def test_import_seeds_is_idempotent(tmp_path):
    db = Database(tmp_path / "c.db")
    src = CVDSeedSource(_payload())
    n1 = import_seeds(src, db)
    assert n1 == 3  # 2 cve + 1 ghsa in the fixture
    # Re-import: no duplicates.
    import_seeds(CVDSeedSource(_payload()), db)
    assert len(db.list_cases()) == 3
    db.close()


def test_manifest_round_trip_preserves_ground_truth(tmp_path):
    case = Case(
        source="cvd",
        ext_id="CVE-2026-27654",
        cve_id="CVE-2026-27654",
        project="nginx",
        repo_url="https://github.com/nginx/nginx",
        fix_commit="deadbeef",
        vuln_commit="cafef00d",
        cwe="CWE-122",
        status="vetted",
        vet_confidence=0.9,
        gt_files=["src/http/ngx_http.c"],
        gt_hunks=[{"file": "src/http/ngx_http.c", "start": 10, "end": 20}],
    )
    path = write_case_manifest(case, tmp_path / "cases")
    assert path.name == "CVE-2026-27654.yaml"

    loaded = load_manifest_dir(tmp_path / "cases")
    assert len(loaded) == 1
    rt = loaded[0]
    assert rt.fix_commit == "deadbeef"
    assert rt.vuln_commit == "cafef00d"
    assert rt.gt_files == ["src/http/ngx_http.c"]
    assert rt.gt_hunks[0]["start"] == 10
    assert rt.status == "vetted"
    # The DB id is an artifact of the working store, not the pinned manifest.
    assert "id" not in case_to_manifest(case)


def test_manifest_from_partial_dict_tolerates_missing_keys():
    case = case_from_manifest({"source": "manual", "ext_id": "CVE-1", "project": "x"})
    assert case.ext_id == "CVE-1"
    assert case.gt_files == []
    assert case.status == "candidate"
