# Nelson Benchmark Harness — Implementation Plan

> Status: P1 landed (2026-05-26); P2 next. Written 2026-05-26.
> This document is the durable source of truth for the benchmark effort. Update it as phases land.

## 1. Goal

Extend Nelson (do **not** rewrite) from a single-shot LLM vulnerability scanner into a
**benchmark** that measures which models find recent, post-training-cutoff *known*
vulnerabilities (0-day *to the model*) — and at what cost and false-positive rate.

These vulnerabilities have often survived years in the wild and human audits, so they are
a strong test of model capability and of how much an improved harness (tools, agent
environment) contributes. Deliverables: a **leaderboard** and a **Pareto frontier**
(accuracy vs cost / size / speed) over an evolving pool of competitors and an evolving
corpus of vulnerabilities, runnable **unattended**.

## 2. Locked decisions (2026-05-26)

1. **Realism over fairness.** Each model runs in its best available native agent with
   permissions wide open (`--dangerously-skip-permissions` / yolo). Agent-environment is an
   **axis**, not a fixed choice: the unit of evaluation is a
   **competitor = (model × runtime × tool-profile)**. e.g. "Sonnet in Claude Code" and
   "Sonnet via raw API + minimal loop" are distinct competitors and the delta is a result.
   Models with no native agent fall back to a built-in ReAct loop or a `pi`-style custom agent.
   The only invariant across competitors is "same code, same known vuln."
2. **Scoring = localization gate + Opus 4.7 semantic judge** (hybrid). A **hit** =
   a reported finding lands within N lines of a patched hunk in a ground-truth file
   **AND** the judge rules it semantically the same bug as the CVE.
3. **Corpus = auto-propose + Opus 4.7 pre-vet.** Source-pluggable importer; seed from the
   Anthropic CVD list, enrich via OSV/GHSA/NVD for fix commit + version ranges + CWE,
   derive the pre-patch commit (= fix-commit parent) and changed files as ground truth.
   Recency-filtered per competitor knowledge-cutoff; the vetted set is pinned in a manifest.
4. **Isolation = container per run**, using **podman** (preferred on Fedora; rootless is a
   real safety win for untrusted OSS build scripts). Throwaway container per (competitor ×
   case), repo checked out inside, resource/network-capped, credentials injected.

**Integrity rule (non-negotiable):** auth / rate-cap / infra failures get a distinct status
(`auth_failed` / `infra_error`) and are **never** scored as a missed bug. An unauthenticated
or capped model must be distinguishable from one that genuinely looked and found nothing.

## 3. Seed competitor set

Prioritize low-cost / high-benchmark "open-ish" models (too large to run locally, accessed
via hosted API):

| Model | Native agent? | Runtime | Auth |
|-------|---------------|---------|------|
| MiMo v2.5 | agent-agnostic | raw-API loop or `pi`-based custom agent | API key (env var) |
| DeepSeek | yes | its native agent CLI | API key (env var) |
| Kimi k2.6 | yes | its native agent CLI | API key (env var) |
| Claude (Opus 4.7) | yes (Claude Code) | **judge**, used sparingly as competitor | `ANTHROPIC_API_KEY`, API-billed |

- All three seed competitors are **API-key auth** → no OAuth dance for the initial set, so
  the OAuth-bootstrap machinery is *designed for* but *deferred*.
- **Claude is primarily the judge** (`claude -p`, billed at API rates — rolling subscription
  limits no longer apply to `-p`). Judge token cost is tracked **separately** from competitor
  cost so it never distorts the Pareto picture.
- Exact agent CLI names + auth env vars for MiMo / DeepSeek / Kimi are likely newer than the
  assistant's training data — **verify at wiring time (P7)**, do not guess. Registry treats
  them as data so adding a competitor is config, not code.

## 4. Reuse from existing Nelson

- `agents.py` `AgentAdapter` abstraction → **split** into *model* vs *runtime* (P0).
- SQLite job tracking (`db.py`), resumable scans, token/cost accounting.
- Parallel-per-model worker pool + pacing/backoff (`scanner.py`).
- Eligibility-aware clustering (`compare.py`) → reused to dedupe a run's findings and for
  cross-competitor agreement.
- Escalation/verification pattern (`review.py`) → basis for the FP judge.
- HTML reporter (`html_report.py`) → extended for leaderboard / Pareto.

## 5. Components to build

1. **Corpus importer + manifest** (`corpus.py`, `cases/*.yaml`): seed → enrich → derive →
   Opus pre-vet → vetted manifest. Source-pluggable.
2. **Container runner** (`runner.py`): podman-first, pluggable backend. Per-run throwaway
   container, repo at pre-patch commit, resource/network caps, credential injection,
   transcript + raw-output + wall-clock + token/cost capture.
3. **Runtime layer** (split from `agents.py`): `claude-code`, `gemini-cli`, `deepseek`,
   `kimi`, `raw-api-loop` (minimal ReAct + tool registry), later `pi`-custom. Tool-profile
   declared per runtime.
4. **Auth/secrets layer**: auth profile per competitor (env vars + staged credential files,
   referencing secret *names*, not values); one-time `bootstrap` for OAuth (deferred);
   `auth_failed`/`infra_error` status + preflight ping.
5. **Scoring engine** (`score.py`): dedupe → localization gate → truth judge → hit/miss;
   plus FP judge over non-ground-truth findings → precision.
6. **Reporting**: leaderboard (detection rate, precision, FP/case, cost/case, latency,
   size-class) + Pareto frontier; per-case drilldown.
7. **Automation loop** (`schedule`/cron): refresh corpus, age out vulns, add/remove
   competitors, run matrix, rescore, regenerate reports, alert on auth-expiry/infra failures.

## 6. Data model (additions alongside `scans`/`jobs`/`findings`)

Introduces a real schema-migration path (today: `SCHEMA_VERSION = 1`, no migrations).

- **`cases`** — corpus + ground truth: `id`, `source` (cvd/osv/manual), `cve_id`, `ghsa_id`,
  `ant_id`, `project`, `repo_url`, `vuln_commit` (pre-patch SHA), `fix_commit`, `bug_class`,
  `cwe`, `disclosure_date`, `severity`, `description`, `gt_files` (json), `gt_hunks` (json),
  `build_recipe` (nullable), `status` (candidate/vetted/retired), `vet_confidence`, `vet_notes`.
- **`competitors`** — `id`, `name`, `model`, `runtime`, `tool_profile`, `auth_profile`,
  `cost_model` (per-token in/out, or subscription), `size_class`, `knowledge_cutoff`,
  `status` (active/retired), `added_at`.
- **`runs`** — `id`, `case_id`, `competitor_id`, `container_id`, `status`
  (pending/running/complete/infra_error/auth_failed), `started_at`, `completed_at`,
  `tokens_in`, `tokens_out`, `cost_usd`, `wall_clock_s`, `transcript_path`, `raw_output`.
- **`run_findings`** — `id`, `run_id`, `file`, `line_start`, `line_end`, `description`,
  `confidence`, `matches_ground_truth` (bool), `judge_truth_verdict`, `judge_fp_verdict`,
  `judge_reasoning`.
- **`judgments`** — `id`, `target_kind` (truth/fp/prevet), `target_id`, `judge_model`,
  `verdict`, `reasoning`, `tokens_in`, `tokens_out`, `cost_usd` (auditability).

## 7. Scoring methodology (precise)

Per `(competitor, case)` where the run reached `complete`:
- **hit** iff ∃ finding F: F is in a ground-truth file AND `line(F)` within N lines of a
  patched hunk (localization gate) AND truth-judge(F, CVE description) == same-bug.
- **miss** iff `complete` and no such F. (`infra_error`/`auth_failed` runs are excluded from
  the denominator — never counted as misses.)
- **false positives**: every *other* reported finding (not matching ground truth) → FP-judge
  → real-bug vs false-positive. **The FP-judge must NOT see the CVE description** (avoid
  over-trust / circularity); the truth-judge *does* (that's the ground truth).

Metrics: detection rate (recall over eligible cases), precision / false-positives-per-case,
cost per case, wall-clock latency, model size-class. Pareto frontier over
(detection × precision) vs each of (cost, latency, size).

## 8. Phased roadmap (dependency-ordered; one branch per phase)

- **P0 — Refactor & integrity** ✅ *(branch `bench-p0-runtime-auth`)*. Split
  model/runtime in `agents.py`; add `auth_profile` abstraction + env-var credential
  injection; add `infra_error`/`auth_failed` statuses + a preflight auth ping.
  *Gate met: a missing/invalid key marks `auth_failed`, not a miss (tested).*
  Delivered: `nelson/auth.py` (AuthProfile referencing secret *names*, `EnvSecretStore`,
  `MissingSecretError`, OAuth bootstrap **stub**); `FailureKind` + `classify_failure`
  (rate > auth > infra precedence) and `AgentResult.failure_kind` in `agents.py`;
  runtime dimensions (`runtime`/`model_id`/`tool_profile`) surfaced on every adapter
  with `name` unchanged so existing scans/DB are unaffected; env injection via
  `_run_cli(env=…)` (defaults to inheriting the parent env — no regression); a base
  `preflight()` ping; DB `mark_job_auth_failed` / `mark_job_infra_error` (terminal,
  excluded from `coverage_for_scans` so they can never become a miss); scanner routing
  of the three failure kinds; `tests/` (36 tests, pytest added to dev deps).
- **P1 — Corpus foundation** ✅ *(branch `bench-p1-corpus`)*. Source-pluggable importer +
  OSV/NVD enrichment + pre-patch derivation + Opus pre-vet → vetted manifest.
  *Gate met: 5 cases derived live from the 2026 CVD seed and hand-checked (below).*
  Delivered: schema migration path + `cases` table (`db.py`); `Case` model, `CVDSeedSource`,
  and `cases/*.yaml` manifest I/O (`corpus.py`); OSV/NVD enrichment behind an injectable
  HTTP client — GIT-range fix SHA, `/commit/` reference fallback, CVE↔GHSA aliases, CWE
  (`enrich.py`); pre-patch derivation — `vuln_commit = fix^1`, `gt_files`/`gt_hunks` from the
  OLD side of each hunk, behind an injectable `GitRunner` doing depth=2 SHA fetches
  (`derive.py`); `ClaudeCLIJudge` pre-vet with the integrity rule that a judge *failure* never
  retires a case (`prevet.py`); `CorpusPipeline` (idempotent, resumable, per-stage optional)
  + `nelson corpus import/build/list/show/export` CLI. Tested offline against captured
  OSV/NVD/CVD fixtures + a real local git repo.
  - **Real-world findings.** The 2026 CVD payload lists 26 published advisories (14 CVE +
    12 GHSA). A live build enriched 17 and derived 5 — so OSV/NVD already cover a chunk of
    the fresh set, but ~⅓ aren't resolvable to a fix commit yet (no GIT range / no `/commit/`
    reference, e.g. Maven-style ecosystem advisories like log4shell). Enrichment is therefore
    idempotent and re-runnable as sources catch up; un-resolvable seeds stay `candidate`.
  - **Gate hand-check (all 5: `vuln_commit` verified == `fix^1`).** ImageMagick GHSA-x9h5
    (CWE-122 → `MagickCore/draw.c`), junrar GHSA-j273 (CWE-22 path traversal), Ghost
    GHSA-w52v (CWE-89 SQLi), CraftCMS GHSA-cc7p (CWE-863) all derived coherent source-file
    ground truth. MapServer CVE-2026-33721 mis-derived to a *release* commit ("update for
    8.6.1 release"; files `CITATION.cff/CMakeLists.txt/HISTORY.md`) — OSV pointed at the
    release, not the patch. This is precisely the noise the Opus pre-vet step is designed to
    retire, and validates keeping a human-vettable `vetted/retired` manifest.
- **P2 — One competitor end-to-end**: podman runner + one runtime on a few cases.
  *Gate: a real 0-day found and transcript captured inside a container.*
- **P3 — Scoring**: localization gate + truth judge → detection reporting.
- **P4 — Precision**: FP judge over non-ground-truth findings → precision metrics.
- **P5 — Leaderboard + Pareto** reporting.
- **P6 — Automation loop**: scheduling + corpus/competitor lifecycle.
- **P7 — More runtimes**: deepseek, kimi, raw-api loop (MiMo), gemini-cli, pi-custom — each
  a new competitor (verify CLI/auth per vendor).

## 9. P0 detailed breakdown (next up)

1. **Model/runtime split.** Introduce a `Runtime` concept distinct from the model identity.
   Today `ClaudeCLIAdapter("haiku")` conflates them. Target: a runtime wraps (model_id,
   invocation method, tool_profile). Keep existing single-shot behavior working as one
   runtime so nothing regresses.
2. **Auth profile abstraction.** A named profile declaring env vars to set and/or credential
   files to stage, referencing secret *names* (resolved from an untracked secret store /
   env). Wire env-var injection into the run invocation. OAuth bootstrap: stub the interface,
   don't implement yet.
3. **Integrity statuses.** Add `infra_error` and `auth_failed` to the job/run status enum;
   classify them from exit code + known error strings ("Not logged in", "OAuth token has
   expired", "Invalid API key", rate-limit markers). Ensure scoring/reporting excludes them
   from "miss".
4. **Preflight ping.** Cheap per-competitor "say ok" check that surfaces auth problems before
   scored work and emits the right status.
5. **Tests** for status classification (auth failure ≠ empty result) and profile resolution.

## 10. Risks

- **Pre-patch derivation** breaks on squashed/multi-commit fixes — Opus pre-vet is the safety
  net; expect to retire some cases.
- **Build reproducibility**: old OSS may not build unattended. Start tool-profiles at
  read+grep+language-toolchain; treat "won't build" as a per-case capability flag, not a
  failure. Per-case build recipes can come later.
- **Judge circularity**: truth-judge sees the CVE description (intended); FP-judge must not.
- **Judge cost**: bounded (one call per finding-cluster) but real — tracked like any token cost.

## 11. References

- Anthropic CVD seed: <https://red.anthropic.com/2026/cvd/> and machine-readable
  <https://red.anthropic.com/2026/cvd/data/payload.json>. payload fields per record:
  `identifier` (CVE/GHSA), `kind`, `revealed_at`, `findings[]` = {`ant_id`, `project`,
  `bug_class`, `severity`, `title`}. **Lacks** repo URL / fix commit / version ranges / CWE —
  resolve those via OSV.dev / GitHub Advisory DB / NVD. ~88 published advisories; 1,596
  findings across 281 projects.
- Claude Code headless auth: `ANTHROPIC_API_KEY` (per-token); `claude setup-token` →
  `CLAUDE_CODE_OAUTH_TOKEN` (1-yr, subscription, works in `-p`, NOT in `--bare`); creds at
  `~/.claude/.credentials.json` (0600) or under `CLAUDE_CONFIG_DIR`. Prefer long-lived tokens
  over shared credential files for parallel runs (refresh/locking undocumented).
