"""CorpusPipeline orchestration: stages advance candidates idempotently."""

import json
from pathlib import Path

from nelson.corpus import CorpusPipeline, CVDSeedSource
from nelson.db import Database
from nelson.derive import GitError
from nelson.prevet import VetVerdict

FIXTURE = Path(__file__).parent / "fixtures" / "cvd_payload.json"


def _get(db, ext_id):
    row = db.get_case(ext_id)
    assert row is not None
    return row


class FakeGit:
    def __init__(self, parent="p" * 40, diff="", prepare_error=None):
        self._parent, self._diff, self._err = parent, diff, prepare_error

    def prepare(self, repo_url, commit, dest):
        if self._err:
            raise GitError(self._err)

    def rev_parse(self, dest, rev):
        return self._parent

    def diff(self, dest, base, head):
        return self._diff


class StubJudge:
    name = "stub"

    def __init__(self, confidence=0.9, error=None):
        self._c, self._e = confidence, error
        self.calls = 0

    def vet(self, case, diff):
        self.calls += 1
        return VetVerdict(self._c, notes="stub", error=self._e)


class StubEnricher:
    """Fills repo+fix once, only for a case that lacks them."""

    name = "stub"

    def __init__(self, updates):
        self._u = updates

    def enrich(self, case):
        return dict(self._u) if not (case.fix_commit and case.repo_url) else {}


_GOOD_DIFF = (
    "diff --git a/v.c b/v.c\n--- a/v.c\n+++ b/v.c\n"
    "@@ -2,1 +2,1 @@\n-gets(b)\n+fgets(b)\n"
)


def _candidate(db, ext_id="GHSA-x", **kw):
    db.upsert_case({"source": "cvd", "ext_id": ext_id, "project": "p", **kw})


def test_build_only_seeds_when_nothing_wired(tmp_path):
    db = Database(tmp_path / "c.db")
    summary = CorpusPipeline(db).build(CVDSeedSource(json.loads(FIXTURE.read_text())))
    assert summary.seeded == 3
    # No enrichers/git/judge -> nothing can advance; all stay candidates.
    assert summary.skipped == 3
    assert db.count_cases_by_status() == {"candidate": 3}
    db.close()


def test_full_pipeline_vets_a_derivable_case(tmp_path):
    db = Database(tmp_path / "c.db")
    _candidate(db, repo_url="https://github.com/o/r", fix_commit="f" * 40)
    judge = StubJudge(confidence=0.9)
    pipe = CorpusPipeline(
        db, git=FakeGit(diff=_GOOD_DIFF), judge=judge, cache_dir=tmp_path / "cache"
    )
    summary = pipe.build()

    assert (summary.derived, summary.vetted) == (1, 0) or summary.vetted == 1
    case = _get(db, "GHSA-x")
    assert case["status"] == "vetted"
    assert case["vuln_commit"] == "p" * 40
    assert json.loads(case["gt_files"]) == ["v.c"]
    assert case["vet_confidence"] == 0.9

    # Manifest export pins the vetted case.
    n = pipe.export_manifest(tmp_path / "cases")
    assert n == 1
    assert (tmp_path / "cases" / "GHSA-x.yaml").exists()
    db.close()


def test_low_confidence_retires(tmp_path):
    db = Database(tmp_path / "c.db")
    _candidate(db, repo_url="https://github.com/o/r", fix_commit="f" * 40)
    pipe = CorpusPipeline(
        db,
        git=FakeGit(diff=_GOOD_DIFF),
        judge=StubJudge(confidence=0.1),
        cache_dir=tmp_path / "cache",
    )
    summary = pipe.build()
    assert summary.retired == 1
    assert _get(db, "GHSA-x")["status"] == "retired"
    db.close()


def test_judge_failure_keeps_candidate_never_retires(tmp_path):
    db = Database(tmp_path / "c.db")
    _candidate(db, repo_url="https://github.com/o/r", fix_commit="f" * 40)
    pipe = CorpusPipeline(
        db,
        git=FakeGit(diff=_GOOD_DIFF),
        judge=StubJudge(error="timeout"),
        cache_dir=tmp_path / "cache",
    )
    summary = pipe.build()
    assert summary.skipped == 1
    case = _get(db, "GHSA-x")
    # Integrity: a judge failure is not a "retire" — stays candidate to retry.
    assert case["status"] == "candidate"
    assert "prevet failed: timeout" in case["vet_notes"]
    db.close()


def test_underivable_case_is_skipped_with_reason(tmp_path):
    db = Database(tmp_path / "c.db")
    _candidate(db, repo_url="https://github.com/o/r", fix_commit="f" * 40)
    pipe = CorpusPipeline(
        db,
        git=FakeGit(diff=""),  # empty diff -> not derivable
        judge=StubJudge(),
        cache_dir=tmp_path / "cache",
    )
    summary = pipe.build()
    assert summary.skipped == 1
    case = _get(db, "GHSA-x")
    assert case["status"] == "candidate"
    assert "empty" in case["vet_notes"]
    db.close()


def test_enricher_stage_advances_then_derives(tmp_path):
    db = Database(tmp_path / "c.db")
    _candidate(db)  # no repo/fix yet
    enricher = StubEnricher(
        {"repo_url": "https://github.com/o/r", "fix_commit": "f" * 40}
    )
    pipe = CorpusPipeline(
        db,
        enrichers=[enricher],
        git=FakeGit(diff=_GOOD_DIFF),
        judge=StubJudge(confidence=0.8),
        cache_dir=tmp_path / "cache",
    )
    summary = pipe.build()
    assert summary.enriched == 1
    assert summary.vetted == 1
    assert _get(db, "GHSA-x")["repo_url"] == "https://github.com/o/r"
    db.close()


def test_build_is_idempotent_does_not_revet(tmp_path):
    db = Database(tmp_path / "c.db")
    _candidate(db, repo_url="https://github.com/o/r", fix_commit="f" * 40)
    judge = StubJudge(confidence=0.9)
    pipe = CorpusPipeline(
        db, git=FakeGit(diff=_GOOD_DIFF), judge=judge, cache_dir=tmp_path / "cache"
    )
    pipe.build()
    assert judge.calls == 1
    # Second pass: the case is now vetted, not a candidate -> judge not called again.
    second = pipe.build()
    assert judge.calls == 1
    assert second.vetted == 0
    db.close()
