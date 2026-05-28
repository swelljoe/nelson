# Nelson Benchmark Harness — Implementation Plan

> Status: P7 landed + merged (PR #14); post-P7 `refused` outcome + Gemini-via-direct-API on branch
> `bench-refusal-outcome` (2026-05-28; prompt left unchanged). Benchmark feature-complete.
> Written 2026-05-26.
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
   The only invariant across competitors is "same code, same known vuln." Each run is **scoped
   to one known-vulnerable file** (a realistic file-by-file audit; the model is told the file,
   never the bug) — this isolates detection from un-scorable repo triage and bounds cost; see §7.
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

**File-scoped runs (decision 2026-05-27).** A run audits **one file** of a case: the
competitor is pointed at a known-vulnerable file (`runs.target_file`) and asked to review
it for vulnerabilities, told *nothing* about the planted bug. This mirrors a real
file-by-file audit, isolates the detection question (can the model recognize the bug *in
the code*) from the un-scorable repo-triage question (which of N files to look at — we have
ground truth only for the vulnerable ones), and cuts cost/latency ~4–5× while shrinking the
false-positive surface to "other findings in this one file." The whole source tree is still
mounted read-only so the model can follow a definition/caller for context. The unit is thus
`(competitor, case, file)`; one case has one run per vulnerable (non-test) file.

Per `(competitor, case, file)` run that reached `complete`:
- **run hit** iff ∃ finding F: F is in a ground-truth file AND `line(F)` within N lines of a
  patched hunk (localization gate) AND truth-judge(F, CVE description) == same-bug.
- **run miss** iff `complete` and no such F (and no localized-but-undetermined finding).
- A localized finding the judge could not decide (timeout/auth/unparseable) → **judge_error**
  (undetermined), never a miss. (`infra_error`/`auth_failed` runs → **excluded**.)

Rolled up to the **case** (the detection unit): a case is detected (**hit**) iff *any* of its
file-runs hit; precedence for the rest is `judge_error` > `miss` > `excluded` (an undetermined
file keeps the case out of the denominator rather than scoring it a clean miss). `judge_error`
and `excluded` cases are excluded from the detection-rate denominator — never counted as misses.
- **false positives (P4)**: every reported finding that is *not the confirmed target bug* —
  a finding that didn't localize, **or one that localized but the truth judge ruled a
  *different* bug** — is FP-judged. The FP judge is **code-grounded**: it reads the actual
  pre-patch source (`git show vuln_commit:path`) and rules `confirmed` (a real bug) /
  `false_positive` / `needs_review`. **The FP-judge must NOT see the CVE description** (avoid
  over-trust / circularity) — its inputs are *only* the finding + the source, so the advisory
  cannot leak by construction; the truth-judge *does* see the advisory (that's the ground
  truth). A confirmed *different* bug is credited as a real finding (`real_other`), never
  penalized; a `needs_review`, a judge failure, or a missing source is **undetermined** —
  excluded from precision, never scored as a false positive (the integrity rule again).
  - **precision** = true findings / (true findings + false positives), where true findings =
    target-bug hits + confirmed other real bugs. Plus **false-positives-per-case** (FP count
    normalized by the cases a competitor audited).

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
  `vuln_commit`, materialized **without `.git`** — see below), `claude_code_spec`,
  stream-json output parsing (`extract_result` /
  `parse_competitor_findings`), and the `BenchRunner` orchestrator; `nelson bench
  run/runs/show-run` CLI. Tested with an injected fake backend + fake auth (no podman /
  network / credentials), covered by the test suite.
  - **Container design.** Minimal `fedora-minimal:41` image (git + ripgrep); the host's
    229 MB `claude` binary is **bind-mounted read-only** rather than baked in, so the
    image tracks the host CLI version. The case repo is checked out at `vuln_commit` and
    mounted **read-only** at `/src`; the agent runs as a non-root user (so
    `--dangerously-skip-permissions` is accepted) with `--cap-drop ALL`,
    `no-new-privileges`, memory/cpu/pids caps. SELinux is per-container `label=disable`
    (avoids relabeling shared host files); isolation rests on the rootless userns + caps
    + ro source. Network is **on** for P2 — the agent needs its model backend; an egress
    allowlist is a later refinement.
  - **Pristine source mount — no `.git` (2026-05-27).** The earlier checkout `git
    init`/`fetch`/`checkout`ed *into* the mount, so `.git` shipped to the competitor:
    `.git/config` named the upstream repo (identity → recall the advisory), and with
    network on `git fetch origin <fix-sha>` would pull the future fix one `git diff`
    away. `prepare_checkout` now fetches into a **scratch bare repo** at
    `<dest>/.gitcache` (a *sibling*, never mounted) and `git archive | tar -x`es the
    commit's tree into `<dest>/src` — the only thing mounted. The mount is a clean
    source tree: no `.git`, no history, no remotes, no commit SHA. (Egress is the
    separate, harder half of the same threat; still a later refinement.)
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
  - **File-scoped harness (decision 2026-05-27; folded into this branch).** The competitor is
    now pointed at one vulnerable file at a time (neutral prompt, no advisory leak) instead of
    auditing the whole repo — see §7. Delivered alongside scoring: schema v5 `runs.target_file`;
    `vulnerable_files(case)` (distinct non-test files carrying a hunk) and a neutral file-scoped
    prompt in `runner.py`; `run_case(case, competitor, target_file)` + a `--max-budget-usd`
    backstop; `bench run` enumerates a case's files into one run each (`--file` to scope);
    case-level rollup (`CaseScore`/`case_scores`) so detection counts cases, not file-runs.
    - **Live comparison (junrar, same case/model).** whole-repo: HIT, **$1.33 / 677 s**, ~825k
      tokens, 20 turns; file-scoped: HIT (judge again confirms line 76, rejects line 61),
      **$0.31 / 223 s**, ~225k tokens — same detection, ~4× cheaper, ~3× faster, FP surface
      down to 3 findings in the one file. This validates the user's thesis: pointed at a file
      with no hint about the bug, the model still finds it.
    - **Finding-parser bug fixed (`parse_competitor_findings`).** The file-scoped prompt elicits
      a reasoning summary whose prose contained a *valid* JSON string-array
      (`["..","evil.txt"]`) **before** the real ```json findings block. The old parser grabbed
      the first balanced `[...]`, found no objects, and returned `[]` — turning a real HIT into
      a phantom miss. Now: try the whole text, then fenced ```json blocks, then every balanced
      array, accepting only one that carries finding *objects* and preferring the last; invalid
      backslash escapes are repaired (shared with pre-vet). 141 tests.
- **P4 — Precision** ✅ *(branch `bench-p4-precision`)*. Code-grounded FP judge over every
  non-target finding → precision + false-positives-per-case.
  *Gate met: the live FP judge, given junrar's real pre-patch source and **no advisory**,
  confirmed a genuine different bug and rejected a plausible-sounding hallucination (below).*
  Delivered: `Database.record_fp_verdict` (fills `run_findings.judge_fp_verdict`, FP cost
  logged to the `judgments` ledger as `target_kind='fp'`, tracked separately from both
  competitor and truth-judge spend); `SubprocessGitRunner.show` (`git show rev:path`);
  `nelson/score.py` — `FPVerdict`, an `FPJudge` Protocol, a `CodeProvider` /
  `GitCodeProvider` (per-(repo,commit) shallow-fetch + file cache; a path absent at the
  revision → None → undetermined), `build_fp_prompt` (takes **no Case** — the advisory cannot
  reach the FP judge), `parse_fp_verdict`, and `ClaudeFPJudge` (host `claude -p`, code-grounded,
  failures surfaced not guessed); the `Scorer` now optionally FP-judges (`fp_judge` + `code`),
  with `FindingScore.is_target_hit` / `fp_category`, `load_run_score` + `needs_scoring` made
  FP-aware, and a `precision_report` / `CompetitorPrecision`; `bench score` gains
  `--precision/--no-precision`, `--fp-judge-model`, `--cache-dir` and a precision table.
  165 tests total.
  - **Two gates, two judges.** The truth judge (P3) answers "is this *the* bug?" from the
    advisory; the FP judge answers "is this a *real* bug?" from the code, never the advisory.
    Splitting them keeps precision honest: a model is credited for finding a genuine *other*
    bug (`real_other`) and penalized only for noise (`false_positive`), and the judge's
    indecision (`needs_review` / failure / unfetchable source) is *undetermined*, never an FP.
  - **Path resolution.** Findings report paths *relative to the `/src` mount* (the repo root),
    so they are already repo-relative; the FP judge peels only a redundant `/src/` prefix for
    `git show` (it must NOT strip a real top-level `src/`, unlike the localization matcher).
  - **Gate run (junrar GHSA-j273, pre-patch `LocalFolderExtractor.java`, no advisory).** Real
    `GitCodeProvider` fetched the file via `git show c7041fc0:…`; the real Opus FP judge then
    (a) **confirmed** the weak `startsWith` prefix-check at line 61 as a genuine CWE-22 sibling-
    directory traversal — independently deriving the attacker-controlled `getFileName`, the
    `/tmp/out` vs `/tmp/out-evil` bypass, exploitability, and the correct fix — and (b) rejected
    a confident "resource leak" claim at line 51 as a **false_positive**, noting the line is
    inside try-with-resources so `close()` is guaranteed. ~$0.14 for the two calls. The
    code-grounded judge discriminates real bugs from plausible noise without the advisory —
    exactly the precision signal P4 needed.
- **P5 — Leaderboard + Pareto** ✅ *(branch `bench-p5-leaderboard`)*. Per-competitor
  leaderboard fusing detection + precision + economics, a Pareto frontier, and a
  per-case drilldown — all a **pure function of the P3/P4 RunScores**, so no new live
  model run was needed.
  *Gate met: 15 tests + a 4-competitor demo report where the Pareto frontier correctly
  dropped the one dominated competitor (below).*
  Delivered: `RunScore` now also carries the competitor's **own** cost / wall-clock /
  tokens + denormalized `size_class` / `knowledge_cutoff` (plumbed from the run +
  competitor rows in `score_run` / `load_run_score`); `LeaderboardEntry` +
  `leaderboard(run_scores)` (combines `detection_report` cases, `precision_report`
  findings, and per-competitor cost/latency over **complete** runs → detection rate,
  precision, FP/case, **cost/case**, **latency/case**, size; ranked detection ↓ →
  precision ↓ → cost/case ↑); `pareto_frontier(entries, x=, y=)` (non-dominated subset,
  minimise-x / maximise-y, ties kept, missing-coord dropped); `generate_leaderboard_report`
  in `html_report.py` (ranked table + two inline-**SVG** scatter plots, no JS/assets, with
  the frontier drawn + a competitor × case outcome matrix); a read-only `nelson bench
  leaderboard [--html PATH]` CLI (reloads persisted scores, **no judge spend**; warns if
  runs still need scoring). 185 tests total.
  - **Judge spend never enters the ranking.** Cost/case and both Pareto axes use only
    the competitor's own `runs.cost_usd`; truth- + FP-judge spend is summed and shown in a
    separate column, exactly as the data model intended — so how we *score* a model can
    never distort how it *ranks*.
  - **Quality = detection_rate × precision** is the Pareto y-axis (maximise) against
    cost/case or latency/case (minimise). Precision-None (no scorable findings) is treated
    as 1.0, but since any hit is itself a true finding, detection > 0 always implies a
    defined precision, so the fallback only ever zeroes out a no-detection competitor — it
    can't inflate. Cost/latency denominators are **distinct cases with a complete run**
    (matching the FP/case denominator), so all per-case rates share one notion of "audited".
  - **Size is categorical**, so it is shown as a leaderboard column + point annotation
    rather than forced onto a numeric Pareto axis (the cost/latency frontiers are the
    numeric trade-offs); revisit if a size ordinal is defined.
  - **Demo gate (synthetic 4-competitor matrix).** opus (90% det, $1.20/case), sonnet
    (70%, $0.30), haiku (40%, $0.05), and a noisy mini (50% det but 30% precision, $0.08).
    The cost-vs-quality frontier resolved to **{haiku, sonnet, opus}** and correctly dropped
    noisy-mini — haiku is both cheaper *and* higher-quality, so the noisy competitor is
    dominated. Table ranks by detection, ★-marks frontier members, and the matrix shows
    hit/miss/jerr/excl per case. This is the leaderboard + Pareto deliverable end-to-end.
- **P6 — Automation loop** ✅ *(branch `bench-p6-automation`)*. One idempotent,
  unattended pass that ties P1-P5 together: (opt-in) refresh corpus -> sync the
  competitor roster -> age out stale cases -> plan + run the missing matrix cells
  -> score new completions -> regenerate the leaderboard -> surface alerts.
  *Gate met: a 3-scenario demo (no podman/judge) — a healthy multi-competitor pass
  aged out a stale case, ran/scored the fresh matrix and wrote the report; a re-run
  planned nothing (idempotent); an all-auth-failing run tripped the circuit breaker
  and flagged needs_attention (below).*
  Delivered: `nelson/automate.py` — a **pure planning core** (`plan_matrix` dedups
  the (competitor x case x file) matrix against existing runs — a complete/pending/
  running cell is skipped, an only-ever-failed cell is re-runnable with
  `--retry-failed`; `case_is_fresh_for` + `parse_date_prefix` recency gate;
  `select_aged_out`; `sync_competitors` / `load_competitors` for config-driven
  roster) and the `run_once` driver that wires the **injected** P1-P5 engines
  (`CorpusPipeline`, `BenchRunner`, `Scorer`) so the whole pass runs under test with
  fakes — it owns **no scoring/detection logic**; `Competitor.from_row`; a
  cron-friendly `nelson bench loop` CLI (`--interval` for a built-in repeat). 210
  tests total.
  - **Unattended-safety rails.** A pass is bounded by `--max-runs` and
    `--max-spend-usd` (competitor spend only); an **auth circuit breaker** aborts the
    run stage after N *consecutive* auth failures (default 3) — the signature of an
    expired host `claude` session, where continuing would only burn budget failing
    identically. A non-auth outcome resets the consecutive count, so one mis-authed
    competitor among healthy ones doesn't trip it. The CLI exits non-zero when a pass
    `needs_attention` (any auth failure / breaker tripped), so a cron mail fires.
  - **Idempotent + resumable.** `plan_matrix` only schedules missing cells and
    scoring only touches complete-but-unscored runs, so a cron entry (or `--interval`)
    can fire the same command repeatedly and it just fills gaps — no run is repeated.
  - **Integrity carried through.** `auth_failed` / `infra_error` runs are *counted*,
    never scored as misses; auth failures surface via `needs_attention`, while infra
    errors are summarized in the matrix/reporting without triggering the cron alert.
    Age-out only ever flips a case to `retired` (never deletes ground truth) and only
    when the data **proves** staleness.
  - **Decisions.** Corpus refresh is **opt-in** (`--refresh-corpus`) — it needs network
    and spends Opus pre-vet budget, so the default loop is run + score + report only.
    Age-out is **on by default** but conservative and *inert until cutoffs exist*: it
    retires a vetted case only when its disclosure is on/before the **minimum** known
    cutoff among active competitors (every active competitor likely trained after the
    bug) AND every active competitor has a parseable cutoff; a case with no disclosure
    date is never aged out. The recency gate (`--recency`) likewise only excludes a
    cell when it can prove the case predates that competitor's cutoff — missing dates
    default to fresh/included. Competitors are **config** (`--competitors roster.yaml`):
    declared ones are upserted, absent active ones flipped to `retired` (history kept).
- **P7 — More runtimes** ✅ *(branch `bench-p7-runtimes`; gates green: ruff + format + ty +
  257 pytest; image builds with python3; NOT merged — user merges per-phase)*. Closed the
  one gap blocking every non-Claude model: `BenchRunner.run_case` was hardcoded to claude-code.
  Delivered a **runtime-dispatch layer** (`nelson/runtimes.py`): a `ContainerRuntime` registry
  keyed by `competitors.runtime`, with the claude path refactored into `ClaudeCodeRuntime` at
  zero behavior change (existing runner tests untouched-green). Runtimes registered:
  - **`raw-api-loop`** (the shared, apples-to-apples *model* harness — user chose "Both"): a
    **stdlib-only** in-container ReAct agent (`nelson/raw_api_loop.py`) with sandboxed
    `read_file`/`grep`/`list_dir` tools confined to `/src` (realpath-guarded against `..`,
    absolute, and symlink escapes), driving any OpenAI-compatible endpoint and emitting a
    claude-shaped result object so parsing stays uniform. DeepSeek/MiMo/Kimi are pure provider
    config (`base_url` + per-token pricing in `cost_model` JSON; key via auth profile).
  - **`claude-code` Anthropic-compat passthrough** (the *other* trusted shared harness): a
    non-Anthropic model with an official Anthropic-compatible endpoint (e.g. DeepSeek via
    `https://api.deepseek.com/anthropic`) runs through the **real Claude Code harness** —
    `ClaudeCodeRuntime.build_spec` reads `anthropic_base_url` + a model-mapping `env` block from
    the competitor's `cost_model` JSON and injects them, with the provider token supplied by the
    auth profile (`deepseek-anthropic` → `ANTHROPIC_AUTH_TOKEN`). So DeepSeek competes on *two*
    trusted harnesses (claude-code-compat and raw-api-loop), agent-vs-agent on one model. Native
    subscription claude is unaffected (no `anthropic_base_url` → unchanged).
  - **`gemini-cli`** (native, bind-mounted host binary; subscription-auth credential mount).
  - **`kimi-cli` / `pi-custom`** (native vendor agents): wired + unit-tested but stubbed — an
    absent host binary resolves to `infra_error`, so they compete the moment the CLI is
    installed and verified. **No `deepseek-cli`:** DeepSeek ships no first-party agent CLI (only
    untrusted third parties), so it runs via the two shared harnesses above instead.
  - **Auth bridge:** `EnvKeyAuth` (resolves an AuthProfile's secret *names* → container `-e`
    vars; a missing key → `auth_failed`, the integrity hinge), `GeminiCredentialMountAuth`,
    and `auth_for_competitor` (profile → env injection; else the runtime's default). New
    `STANDARD_AUTH_PROFILES`: `deepseek-api-key` (OpenAI-compat, → `NELSON_API_KEY`),
    `deepseek-anthropic` (claude-compat, → `ANTHROPIC_AUTH_TOKEN`), `mimo-api-key`,
    `kimi-api-key` (secret *names* only, never values).
  - **Preflight** (`BenchRunner(preflight=…)`, default-off): a cheap host-side OpenAI-compatible
    probe so a dead key fails before container spend (integrity already guaranteed by EnvKeyAuth).
  - **Image:** CONTAINERFILE adds `python3`; `IMAGE_TAG` bumped to `nelson-bench:fedora-py` so
    `ensure_image` rebuilds. Verified live: python3 3.13.9 + ripgrep present, script compiles.
  - **Integrity carried through:** unknown runtime / missing binary / missing key → never a
    miss. No DB schema change (`runtime`/`auth_profile`/`cost_model` already existed;
    `SCHEMA_VERSION` stays 5).
  - **Live gate met (2026-05-28, DeepSeek on the junrar GHSA-j273 CWE-22 case).** Both shared
    harnesses detected the planted zip-slip end-to-end against the real DeepSeek API:
    `claude-code/deepseek` (deepseek-v4-pro, `/anthropic` + `ANTHROPIC_AUTH_TOKEN`) → 1 finding,
    CWE-22 @ L76, 164 s, $0.48; `raw-api-loop/deepseek` (deepseek-chat→v4-flash, OpenAI-compat)
    → 2 findings, CWE-22 @ L82+L67, 133 s, $0.028. The gate also surfaced a real raw-api-loop
    bug (the ReAct agent hit its 12-step cap mid-analysis and, on the forced-final turn, emitted
    a native-format tool call as text → 0 findings); fixed by an explicit forced-final
    instruction + a larger default step budget (12→20), after which the same model produced a
    clean JSON hit. Endpoint values confirmed from api-docs.deepseek.com.
  - **`agy` (Antigravity / Gemini) live-tested (2026-05-28).** The old `gemini` CLI was renamed
    to `agy`, with a Claude-Code-like interface (`-p`, `--dangerously-skip-permissions`,
    `--add-dir`, plain-text out, no `-m`/JSON), so `GeminiCliRuntime` → `AgyRuntime` (bind-mount
    the host `agy`; `AgyCredentialMountAuth` mounts the `~/.gemini` sign-in). On the junrar case
    the harness ran cleanly end-to-end, but the **Gemini model behind Antigravity refused** the
    neutral audit prompt ("Sorry, I cannot fulfill your request… see the OWASP Top Ten") — a real
    model-behavior result, scored as-is (the uniform prompt is *not* tuned per model). This drove
    the post-P7 follow-up below (prompt reword + a `refused` scoring outcome).
  - **MiMo (Xiaomi) live gate met (2026-05-28, junrar) — both harnesses HIT.** MiMo also exposes
    Anthropic-compat (`/anthropic`, `mimo-anthropic` profile) and OpenAI-compat (`/v1`,
    `mimo-api-key`) endpoints; the host is **region-specific** (the Token-Plan `tp-` key uses
    `token-plan-sgp.xiaomimimo.com`, not the `-cn` host the docs implied). `claude-code/mimo`
    (mimo-v2.5-pro) → 2 findings, CWE-22 @ L61+L35, 400 s, $0.84; `raw-api-loop/mimo` → 2 findings,
    CWE-22 @ L61+L34, 189 s. So across DeepSeek + MiMo, all four API-harness combos ran end-to-end
    and localized CWE-22 path-traversal findings to the vulnerable file (the P3 judge formally
    adjudicates hit vs. a related real finding — several pointed at the prefix-match guard rather
    than the L76 backslash zip-slip).
  - **Still VERIFY-AT-WIRING (not yet live):** native CLI (`kimi-cli`/`pi-custom`) argv/output +
    their binaries (no trusted official CLI installed yet).

- **Post-P7 — `refused` outcome + Gemini via direct API** ✅ *(branch `bench-refusal-outcome`)*.
  Triggered by the agy/Gemini refusal. The prompt was **deliberately left unchanged** (see the
  reword investigation below); two things shipped: a `refused` scoring outcome, and a working
  Gemini competitor via the direct API.
  - **`refused` scoring outcome (P3).** A new `ClaudeRefusalJudge` (Opus, host, mirrors the truth
    judge; sees only the model's *output text*, never the advisory or source). A complete run with
    **zero findings that emitted no JSON array** (ignored the output contract) is a cheap
    deterministic *candidate*; the judge must then **positively confirm** a refusal. Confirmed →
    `outcome="refused"` (excluded from the hit/miss denominator like auth/infra, reported on its
    own column REFU; never a miss). "Attempted" or a judge error → stays a **miss** (the
    conservative direction — a refusal we cannot confirm never inflates the score by hiding a real
    miss). A compliant `[]` run is never judged. Persisted to the `judgments` ledger keyed by run
    (`target_kind="refusal"`), so `load_run_score` rebuilds it free; refusal-judge spend folds into
    `judge_cost` (out of the Pareto ranking, like all judge spend). Threaded through
    `detection_report` / `leaderboard` / the case-rollup precedence (`hit > judge_error > miss >
    refused > excluded`) / the HTML matrix / the automation loop report. **Live-validated**: the
    real Opus judge scored an agy refusal → `refused`, `eligible=False`, $0.07. No schema change
    (SCHEMA_VERSION stays 5). Gates green; +13 tests (271 total).
  - **Gemini via the direct API (`raw-api-loop/gemini`).** `agy` (Gemini behind the Antigravity
    CLI) refuses the audit — its wrapper trips a safety filter — so it scores `refused`. But the
    **direct Gemini API engages the identical honest prompt**: tested on the OpenAI-compatible
    endpoint (`generativelanguage.googleapis.com/v1beta/openai`) at *default* safety, Gemini
    2.5-pro found the real CWE-22. Since that endpoint is OpenAI-shaped, the existing `raw-api-loop`
    harness drives it with **zero code change** (Bearer auth, sandboxed read/grep/list tools) —
    apples-to-apples with DeepSeek/MiMo. Added auth profile `gemini-openai`
    (`NELSON_API_KEY → GEMINI_API_KEY`, a Google AI Studio key) + an example competitor. **Live
    gate met**: `raw-api-loop/gemini` (gemini-2.5-pro) vs junrar → judge-confirmed **HIT**, CWE-22
    @ L78 `same_bug`, 54 s, $0.08. No prompt-disguise and no safety-disabling were needed (default
    safety engaged). `BLOCK_NONE` is reachable only via the *native* `generateContent` API (the
    compat layer can't set it) and is held as a robustness fallback if a run is ever safety-blocked
    mid-loop (which would surface as an integrity status / `refused`, never a miss).
  - **Reword investigation (NOT shipped — prompt unchanged).** First tried dropping "*exploitable*"
    from the shared prompt to placate Gemini. It **did not unblock agy** (still refused, naming
    "Zip Slip" while deflecting — the trigger is the *task class*, not the wording; the model's
    self-diagnosis was a post-hoc rationalization). A judged before/after on junrar (n=1) showed the
    reword *moved* outcomes on the working models — DeepSeek hit→miss on both harnesses, MiMo
    miss→hit — i.e. run-to-run noise on which adjacent CWE-22 (the L61 prefix-guard `different_bug`
    vs the L76/L83 zip-slip `same_bug`) a model surfaces, **not** a measurable benefit. With no
    upside (Gemini refuses regardless; the `refused` outcome + direct API handle it) and a real
    accuracy risk to the models that do the work, the reword was **reverted**. A further
    Gemini-suggested "disguise the task as code-health/linting" reword was rejected outright: it
    would change *what every model is asked*, breaking benchmark validity to accommodate one model.
    Lesson recorded: don't perturb the uniform prompt to accommodate a refusing model.

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
