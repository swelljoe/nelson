"""Benchmark corpus: cases, seed sources, and the vetted manifest.

A *case* is one upstream advisory (a CVE or GHSA) plus the ground truth needed
to score a competitor against it: the repository, the pre-patch (vulnerable)
commit, and the files/hunks the fix touched. Cases are built by a pipeline —
seed -> enrich -> derive -> pre-vet — and the vetted set is pinned to
``cases/*.yaml`` so a benchmark run reproduces against a fixed corpus.

This module holds the data model, the pluggable seed sources (the Anthropic CVD
list is the first), and manifest serialization. Enrichment (`enrich.py`),
pre-patch derivation (`derive.py`), and pre-vet (`prevet.py`) live alongside;
`CorpusPipeline` wires them together.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import yaml

from .db import CASE_COLUMNS, Database
from .derive import Derivation, GitRunner, derive_ground_truth
from .enrich import Enricher, enrich_case
from .prevet import Judge, VetVerdict

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Sequence

# Ordered worst-first so max_severity can pick the headline severity of a
# multi-finding advisory. Unknown/missing sorts last.
_SEVERITY_ORDER = ("critical", "high", "medium", "low")


def max_severity(severities: Iterable[str | None]) -> str | None:
    """Return the most severe label among ``severities`` (None if all empty)."""
    best: str | None = None
    best_rank = len(_SEVERITY_ORDER)
    for s in severities:
        if not s:
            continue
        try:
            rank = _SEVERITY_ORDER.index(s.lower())
        except ValueError:
            continue
        if rank < best_rank:
            best_rank, best = rank, s.lower()
    return best


@dataclass
class Case:
    """A corpus case: an advisory plus its derived ground truth.

    Field names mirror the ``cases`` table columns. ``gt_files`` and ``gt_hunks``
    are the ground truth produced by derivation; everything from ``repo_url``
    onward is filled progressively by the pipeline and may be empty for a freshly
    seeded (``candidate``) case.
    """

    source: str
    ext_id: str
    cve_id: str | None = None
    ghsa_id: str | None = None
    ant_id: str | None = None
    project: str | None = None
    repo_url: str | None = None
    vuln_commit: str | None = None
    fix_commit: str | None = None
    bug_class: str | None = None
    cwe: str | None = None
    disclosure_date: str | None = None
    severity: str | None = None
    description: str | None = None
    gt_files: list[str] = field(default_factory=list)
    gt_hunks: list[dict[str, Any]] = field(default_factory=list)
    build_recipe: str | None = None
    verification: dict[str, Any] | None = None
    status: str = "candidate"
    vet_confidence: float | None = None
    vet_notes: str | None = None
    id: int | None = None

    def to_db_fields(self) -> dict[str, Any]:
        """Project onto the DB column set (lists pass through; db JSON-encodes)."""
        return {c: getattr(self, c) for c in CASE_COLUMNS}

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Case:
        """Build a Case from a DB row, decoding the JSON ground-truth columns."""
        data: dict[str, Any] = {k: row[k] for k in (*CASE_COLUMNS, "id")}
        data["gt_files"] = json.loads(row["gt_files"]) if row["gt_files"] else []
        data["gt_hunks"] = json.loads(row["gt_hunks"]) if row["gt_hunks"] else []
        data["verification"] = (
            json.loads(row["verification"]) if row["verification"] else None
        )
        return cls(**data)


# -- Seed sources ------------------------------------------------------------


@runtime_checkable
class SeedSource(Protocol):
    """Yields seed cases from some upstream list. Source-pluggable by design:
    a new disclosure feed is a new SeedSource, not a change to the pipeline."""

    name: str

    def seeds(self) -> Iterator[Case]: ...


class CVDSeedSource:
    """Seeds from an Anthropic coordinated-disclosure ``payload.json``.

    The payload has ``cve_records`` and ``ghsa_records``; each record has an
    ``identifier``, a ``kind`` (cve/ghsa), a ``revealed_at`` timestamp, and one
    or more ``findings`` (``ant_id``, ``project``, ``bug_class``, ``severity``,
    ``title``). One advisory -> one case: findings are aggregated, since the
    fix-commit set that derivation produces is per-advisory. Repo URL, fix
    commit, version ranges, and CWE are *not* in this payload — enrichment fills
    them from OSV/NVD later.
    """

    name = "cvd"

    def __init__(self, payload: dict[str, Any]):
        self._payload = payload

    def seeds(self) -> Iterator[Case]:
        for key in ("cve_records", "ghsa_records"):
            for record in self._payload.get(key, []):
                case = self._record_to_case(record)
                if case is not None:
                    yield case

    @staticmethod
    def _record_to_case(record: dict[str, Any]) -> Case | None:
        ext_id = record.get("identifier")
        if not ext_id:
            return None
        kind = (record.get("kind") or "").lower()
        findings = record.get("findings") or []
        # Primary = first finding; carry every ant_id and the headline severity.
        primary = findings[0] if findings else {}
        projects = sorted({f.get("project") for f in findings if f.get("project")})
        ant_ids = [f.get("ant_id") for f in findings if f.get("ant_id")]
        title = primary.get("title")
        if len(projects) > 1:
            title = f"{title} (also affects: {', '.join(projects[1:])})"
        return Case(
            source="cvd",
            ext_id=ext_id,
            cve_id=ext_id if kind == "cve" else None,
            ghsa_id=ext_id if kind == "ghsa" else None,
            ant_id=", ".join(ant_ids) or None,
            project=projects[0] if projects else None,
            bug_class=primary.get("bug_class"),
            severity=max_severity(f.get("severity") for f in findings),
            disclosure_date=record.get("revealed_at"),
            description=title,
        )


def import_seeds(source: SeedSource, db: Database) -> int:
    """Upsert every seed from ``source`` into the DB; returns the count.

    Idempotent: re-running refreshes seed metadata without disturbing columns
    the seed doesn't carry (enrichment, ground truth) — see ``upsert_case``.
    """
    count = 0
    for case in source.seeds():
        db.upsert_case(case.to_db_fields())
        count += 1
    return count


# -- Manifest (cases/*.yaml) -------------------------------------------------

# Field order in the YAML file: identity, then advisory metadata, then the
# derived ground truth, then vetting. Chosen for human review, not the machine.
_MANIFEST_FIELD_ORDER = (
    "ext_id",
    "source",
    "cve_id",
    "ghsa_id",
    "ant_id",
    "project",
    "repo_url",
    "fix_commit",
    "vuln_commit",
    "bug_class",
    "cwe",
    "severity",
    "disclosure_date",
    "status",
    "vet_confidence",
    "vet_notes",
    "description",
    "build_recipe",
    "verification",
    "gt_files",
    "gt_hunks",
)


def case_to_manifest(case: Case) -> dict[str, Any]:
    """Ordered, human-readable dict for a case's YAML file (drops the DB id)."""
    out: dict[str, Any] = {}
    for name in _MANIFEST_FIELD_ORDER:
        out[name] = getattr(case, name)
    return out


def case_from_manifest(data: dict[str, Any]) -> Case:
    """Inverse of :func:`case_to_manifest`; tolerant of missing optional keys."""
    known = {f.name for f in fields(Case)}
    return Case(**{k: v for k, v in data.items() if k in known})


def manifest_path(cases_dir: str | Path, ext_id: str) -> Path:
    """Path of a case's manifest file. ext_ids (CVE-…/GHSA-…) are filename-safe."""
    return Path(cases_dir) / f"{ext_id}.yaml"


def write_case_manifest(case: Case, cases_dir: str | Path) -> Path:
    """Write one ``cases/<ext_id>.yaml``; returns the path."""
    path = manifest_path(cases_dir, case.ext_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(case_to_manifest(case), sort_keys=False, allow_unicode=True)
    )
    return path


def load_case_manifest(path: str | Path) -> Case:
    data = yaml.safe_load(Path(path).read_text())
    return case_from_manifest(data)


def load_manifest_dir(cases_dir: str | Path) -> list[Case]:
    """Load every ``*.yaml`` under ``cases_dir`` (sorted by filename)."""
    cases_dir = Path(cases_dir)
    if not cases_dir.is_dir():
        return []
    return [load_case_manifest(p) for p in sorted(cases_dir.glob("*.yaml"))]


# -- Pipeline ----------------------------------------------------------------


@dataclass
class BuildSummary:
    """Counts from one corpus build pass (all stages are idempotent)."""

    seeded: int = 0
    enriched: int = 0
    derived: int = 0
    vetted: int = 0
    retired: int = 0
    skipped: int = 0  # candidates that couldn't progress this pass

    def as_dict(self) -> dict[str, int]:
        return {
            "seeded": self.seeded,
            "enriched": self.enriched,
            "derived": self.derived,
            "vetted": self.vetted,
            "retired": self.retired,
            "skipped": self.skipped,
        }


class CorpusPipeline:
    """Wires seed -> enrich -> derive -> pre-vet over the cases in a DB.

    Every component is optional and injected, so the pipeline runs in whatever
    stages are available: enrichers (HTTP), a :class:`GitRunner` (derivation),
    and a :class:`Judge` (pre-vet). With none wired, ``build`` only seeds. Each
    stage is idempotent and only advances ``candidate`` rows, so a re-run resumes
    where sources or budget last allowed — important because fresh advisories
    aren't in OSV yet and only become derivable later.
    """

    def __init__(
        self,
        db: Database,
        *,
        enrichers: Sequence[Enricher] = (),
        git: GitRunner | None = None,
        judge: Judge | None = None,
        cache_dir: str | Path = "corpus-cache",
        vet_threshold: float = 0.5,
    ):
        self.db = db
        self.enrichers = list(enrichers)
        self.git = git
        self.judge = judge
        self.cache_dir = Path(cache_dir)
        self.vet_threshold = vet_threshold

    def import_seeds(self, source: SeedSource) -> int:
        return import_seeds(source, self.db)

    def _reload(self, ext_id: str) -> Case:
        row = self.db.get_case(ext_id)
        if row is None:  # only ever called for a case we just persisted
            raise RuntimeError(f"case {ext_id} disappeared after write")
        return Case.from_row(row)

    def enrich_one(self, case: Case) -> dict[str, Any]:
        updates = enrich_case(case, self.enrichers)
        if updates:
            self.db.update_case_fields(case.ext_id, updates)
        return updates

    def derive_one(self, case: Case) -> Derivation:
        if self.git is None:
            raise RuntimeError("derive_one requires a GitRunner")
        der = derive_ground_truth(case, self.git, self.cache_dir)
        if der.ok:
            self.db.update_case_fields(case.ext_id, der.updates())
        elif der.reason:
            self.db.update_case_fields(case.ext_id, {"vet_notes": der.reason})
        return der

    def prevet_one(self, case: Case, diff: str) -> VetVerdict:
        if self.judge is None:
            raise RuntimeError("prevet_one requires a Judge")
        verdict = self.judge.vet(case, diff)
        if verdict.error:
            # A judge failure is not a verdict: keep the case a candidate to
            # retry, recording why (mirrors the auth/infra integrity rule). Keep
            # the raw reply too — an unattended run needs it to be diagnosable.
            note = f"prevet failed: {verdict.error}"
            if verdict.notes:
                note += f" | reply: {verdict.notes}"
            self.db.update_case_fields(case.ext_id, {"vet_notes": note})
            return verdict
        status = "vetted" if verdict.confidence >= self.vet_threshold else "retired"
        self.db.update_case_fields(
            case.ext_id,
            {
                "status": status,
                "vet_confidence": verdict.confidence,
                "vet_notes": verdict.notes,
            },
        )
        return verdict

    def build(
        self, source: SeedSource | None = None, *, limit: int | None = None
    ) -> BuildSummary:
        """Run every wired stage over candidate cases; return what changed.

        ``limit`` caps how many candidate rows are processed this pass (useful
        for incremental runs that bound API/network cost); None processes all.
        """
        summary = BuildSummary()
        if source is not None:
            summary.seeded = self.import_seeds(source)

        candidates = self.db.list_cases(status="candidate")
        if limit is not None:
            candidates = candidates[:limit]
        for row in candidates:
            case = Case.from_row(row)

            # 1. Enrich until we have a repo + fix commit to work from.
            if (
                self.enrichers
                and not (case.fix_commit and case.repo_url)
                and self.enrich_one(case)
            ):
                summary.enriched += 1
                case = self._reload(case.ext_id)
            if not (case.fix_commit and case.repo_url):
                summary.skipped += 1  # not resolvable yet (e.g. too fresh for OSV)
                continue

            # 2. Derive ground truth. Re-run is cheap (repo is cached) and yields
            #    the fix diff the judge needs; count it only when newly derived.
            der: Derivation | None = None
            if self.git is not None:
                was_derived = bool(case.vuln_commit)
                der = self.derive_one(case)
                if not der.ok:
                    summary.skipped += 1
                    continue
                if not was_derived:
                    summary.derived += 1
                case = self._reload(case.ext_id)

            # 3. Pre-vet (needs both ground truth and the fix diff as evidence).
            if self.judge is not None and case.vuln_commit and der is not None:
                verdict = self.prevet_one(case, der.diff)
                if verdict.error:
                    summary.skipped += 1
                elif verdict.confidence >= self.vet_threshold:
                    summary.vetted += 1
                else:
                    summary.retired += 1

        return summary

    def export_manifest(self, cases_dir: str | Path, status: str = "vetted") -> int:
        """Pin every case with ``status`` to ``cases/<ext_id>.yaml``; return count."""
        count = 0
        for row in self.db.list_cases(status=status):
            write_case_manifest(Case.from_row(row), cases_dir)
            count += 1
        return count
