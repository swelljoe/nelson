# Nelson Benchmark Harness — Implementation Plan

> Status: P3 landed (2026-05-27); P4 next. Written 2026-05-26.
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
| Gemini | yes (gemini CLI) | `gemini` CLI | subscription (CLI sign-in, **no key**) |
| Claude (Opus 4.7) | yes (Claude Code) | **judge**, used sparingly as competitor | subscription via `claude -p` (CLI sign-in, **no key**) |

- The MiMo / DeepSeek / Kimi seed competitors are **API-key auth** (env var) → no OAuth dance
  for that set, so the OAuth-bootstrap machinery is *designed for* but *deferred*.
- **Claude and Gemini use CLI subscription auth, not an API key.** `claude -p` (and the
  `gemini` CLI) run against the already-signed-in subscription on the host; for Claude the
  per-token usage draws against the plan's **included monthly budget** (e.g. $100/mo) — it is
  *not* billed via `ANTHROPIC_API_KEY`. So a no-profile competitor/runtime simply inherits the
  host's authenticated CLI session (the P0 default).
- **Claude is primarily the judge** (`claude -p`). The judge runs on the **host, not in a
  container**, so it needs no credential injection — the CLI is already authenticated. Judge
  token cost is tracked **separately** from competitor cost so it never distorts the Pareto
  picture. (A Claude/Gemini *competitor* run inside a container is the case that would need a
  long-lived token, e.g. `CLAUDE_CODE_OAUTH_TOKEN` — see §11; deferred until needed.)
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
- **P2 — One competitor end-to-end** ✅ *(branch `bench-p2-runner`)*. Rootless-podman
  runner + the `claude-code` runtime, scored DB run layer.
  *Gate met: Sonnet found a real 0-day-to-it and the transcript was captured in a
  container (below).*
  Delivered: schema migration v3 + `competitors` / `runs` / `run_findings` tables and
  accessors, carrying the `auth_failed`/`infra_error` integrity statuses onto runs
  (`db.py`); `nelson/runner.py` — a pluggable `ContainerBackend` (`PodmanBackend`) and
  `ContainerAuth` (`CredentialMountAuth`), `prepare_checkout` (depth-1 fetch of
  `vuln_commit`), `claude_code_spec`, stream-json output parsing (`extract_result` /
  `parse_competitor_findings`), and the `BenchRunner` orchestrator; `nelson bench
  run/runs/show-run` CLI. Tested with an injected fake backend + fake auth (no podman /
  network / credentials), 107 tests total.
  - **Container design.** Minimal `fedora-minimal:41` image (git + ripgrep); the host's
    229 MB `claude` binary is **bind-mounted read-only** rather than baked in, so the
    image tracks the host CLI version. The case repo is checked out at `vuln_commit` and
    mounted **read-only** at `/src`; the agent runs as a non-root user (so
    `--dangerously-skip-permissions` is accepted) with `--cap-drop ALL`,
    `no-new-privileges`, memory/cpu/pids caps. SELinux is per-container `label=disable`
    (avoids relabeling shared host files); isolation rests on the rootless userns + caps
    + ro source. Network is **on** for P2 — the agent needs its model backend; an egress
    allowlist is a later refinement.
  - **Auth.** Decision (2026-05-27): the containerized Claude competitor authenticates by
    **mounting a copy of the host `~/.claude/.credentials.json`** (zero setup, fine for
    sequential P2). `CredentialMountAuth` copies it into a per-run config dir mounted
    `rw,U` at `/cfg` (`CLAUDE_CONFIG_DIR`). A long-lived `CLAUDE_CODE_OAUTH_TOKEN` for the
    parallel matrix (P6) is a drop-in alternative ContainerAuth, no runner change.
  - **Rootless cleanup gotcha (fixed).** The container writes into the mounted config dir
    as a userns-mapped subuid, leaving files the host user can't `unlink`; `_safe_rmtree`
    falls back to `podman unshare rm -rf`.
  - **Token accounting.** stream-json `usage.input_tokens` is only the final turn's fresh
    input; `extract_result` sums `input_tokens + cache_creation + cache_read` so tokens_in
    reflects total input processed. `total_cost_usd` is the authoritative cost.
  - **Gate run (junrar GHSA-j273, CWE-22).** `claude-code/sonnet`, given only the code (no
    advisory), produced 3 findings — all in `LocalFolderExtractor.java`, the sole
    ground-truth file. The high-confidence one (line 76) is the exact backslash Zip-Slip
    the advisory describes, inside GT hunk 73–79; a second (line 61) is inside hunk 55–61.
    676 s wall, ~825k input tokens (cache-dominated), $1.33, full 149-event agentic
    transcript (Read/Bash/Grep + a sub-agent) captured. This is exactly the
    localization-gate + semantic-match signal P3 will score automatically.
- **P3 — Scoring** ✅ *(branch `bench-p3-scoring`)*. Localization gate + Opus truth judge
  over `run_findings` → per-run hit/miss + detection reporting.
  *Gate met: the junrar fixture run scored a HIT, and the judge discriminated — it confirmed
  the real bug and rejected a same-file/same-CWE finding as a different bug (below).*
  Delivered: schema migration v4 + `judgments` ledger (one row per judge call: verdict +
  token cost, so judge spend is tracked separately from competitor cost) and accessors
  `record_finding_score` / `add_judgment` / `judgments` / `get_case_by_id` /
  `get_competitor_by_id` (`db.py`); `nelson/score.py` — a deterministic `localize` gate
  (path-suffix match + line tolerance over `gt_hunks`), `ClaudeTruthJudge` (mirrors the
  pre-vet judge: host `claude -p`, sees the advisory, a failure is surfaced not guessed),
  the `Scorer` (`score_run` localizes → judges only localized findings → persists; outcome
  ∈ hit/miss/judge_error/excluded), `load_run_score` (rebuilds from the DB with no judge
  spend), `needs_scoring`, and `detection_report`; `nelson bench score [RUN_ID]` CLI
  (per-run verbose, or a per-competitor detection table). 132 tests total.
  - **Scoring is two gates, both required for a hit.** (1) Localization is cheap and
    deterministic: the competitor reads the pre-patch tree, so `gt_hunks` line numbers *are*
    its line numbers — the gate is near-exact and the tolerance (default ±10) only forgives
    reporting drift. Only localized findings reach the judge (the rest are FP candidates for
    P4). (2) The Opus truth judge rules same-bug vs different-bug on the *advisory vs the
    model's own explanation* — no code re-checkout needed.
  - **Integrity carried through scoring.** Only `complete` runs are scored; `auth_failed` /
    `infra_error` runs are `excluded` (never a miss). A judge *failure* (timeout / auth /
    unparseable) makes a localized finding **undetermined**, not "different bug": if no other
    finding is a confirmed hit, the run's outcome is `judge_error` — excluded from the
    detection-rate denominator, never silently demoted to a miss. (Mirrors the pre-vet rule.)
  - **Judge-cost bug fixed (also fixes P1 pre-vet).** `_unwrap_claude_json` read `cost_usd`,
    but the CLI emits `total_cost_usd`, so every pre-vet/judge cost was being recorded as
    `None`; it also under-counted input tokens (final-turn only). Now reads `total_cost_usd`
    (legacy `cost_usd` fallback) and sums `input_tokens + cache_creation + cache_read`.
  - **Gate run (junrar GHSA-j273).** With default tolerance, 2 of Sonnet's 3 findings
    localized (lines 61 ∈ 55–61, 76 ∈ 73–79; line 35 correctly off-target). The truth judge
    confirmed line 76 as the exact backslash Zip-Slip the advisory describes (`same_bug`) and
    rejected line 61 as a *different* root-cause flaw (a `startsWith` prefix check, same file
    and same CWE-22) — proving the hybrid gate isn't rubber-stamping localization. Outcome:
    **HIT**, detection 100% over 1 eligible case, $0.096 judge spend logged to `judgments`.
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
- Claude Code auth (this project): **subscription via `claude -p`, no API key.** The host CLI
  is signed in and per-token usage draws against the plan's included monthly budget (e.g.
  $100/mo); `claude -p --output-format json` reports a per-call cost we capture for separate
  accounting. The judge runs on the host (no container) so it just inherits this session.
  *Only* a containerized Claude competitor would need credential injection — options are
  `claude setup-token` → `CLAUDE_CODE_OAUTH_TOKEN` (1-yr, subscription, works in `-p`, NOT in
  `--bare`) or mounting `~/.claude/.credentials.json` (0600, or under `CLAUDE_CONFIG_DIR`);
  prefer the long-lived token for parallel runs (file refresh/locking undocumented). The
  `ANTHROPIC_API_KEY` path exists but is **not** what we use.
- Gemini likewise uses the `gemini` CLI's subscription sign-in (no key).
