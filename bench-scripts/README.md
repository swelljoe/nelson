# bench-scripts

Reproducible harnesses for the Nelson security-bug-hunting benchmark: the scripts
we use to **add a model to the leaderboard** and to **run the side experiments**
(prompt-lab, quant sweeps, oracle-CWE, semgrep/tree-sitter tool A/Bs, repo-scope,
shadow-judging, FP precision). They were originally ~50 one-off `run_*.sh` /
`*_report.py` files in the repo root; this directory generalizes them so a result
can be replicated, and so a new model can be measured against the field by
pointing one config at it.

Everything here is driven from the project venv and is meant to be run **from the
repo root** (paths like `cases/` and `bench-scripts/rosters/...` are repo-root
relative; the shell wrappers `cd` there for you).

```
bench-scripts/
  add-model.sh         # add model(s) to a baseline DB without disturbing results
  run-experiment.sh    # run an isolated-DB experiment (semgrep/treesitter/reposcope/oracle/...)
  run-arm-sweep.sh     # launch the trial-sweep runner (prompt arms / CWE sweep), 1+ models
  lib/
    common.sh                 # shared preamble: repo-root cd, secret loading, backup_db
    surgical_upsert.py        # upsert competitors WITHOUT syncing/retiring the roster
    clear_competitor_runs.py  # delete a competitor's runs so the matrix re-plans them
  runners/
    arm_sweep.py              # config-driven per-trial runner (modes: arms | cwe-sweep)
    configs/*.yaml            # one experiment description per file
  judges/
    fpjudge.py                # code-grounded Opus FP-judge over off-target findings
    shadow_judge.py           # can cheap models replicate the Opus truth judge?
    shadow_fp_judge.py        # can cheap models replicate the Opus FP judge?
  reports/*.py           # HTML/console report generators (+ analyze_promptlab data layer)
  analysis/*.py          # console-only analyzers
  rosters/competitors-*.yaml  # competitor roster files, one per cohort/experiment
  case-subsets/          # case-manifest subsets used by some experiments
  archive/               # one-shot scripts that already did their job (provenance only)
```

## Prerequisites

- The project venv at `.venv/` (`python -m nelson ...` must work). Override the
  interpreter with `NELSON_PY=/path/to/python`.
- API keys, one secret file per provider (override the directory with
  `NELSON_SECRETS_DIR`, default `/home/joe/secrets`). `lib/common.sh` exports a
  key only if its file exists, so partial setups work:

  | file        | env var              | used for                         |
  |-------------|----------------------|----------------------------------|
  | `openrouter`| `OPENROUTER_API_KEY` | OpenRouter-hosted models + judges |
  | `deepseek`  | `DEEPSEEK_API_KEY`   | DeepSeek direct API              |
  | `mimo`      | `MIMO_API_KEY`       | MiMo direct API                  |
  | `gemini`    | `GEMINI_API_KEY`     | Gemini direct API                |

  Self-hosted servers (llama-server / LM Studio) ignore the value, so
  `LMSTUDIO_API_KEY=lm-studio` is exported unconditionally.
- **Self-hosted models**: serve on a routable LAN IP (not a loopback tunnel),
  point the roster's `cost_model.base_url` at it, and confirm the `model` field
  matches the server's loaded id exactly (`curl $BASE/v1/models`). Give long
  ReAct trajectories headroom with a large context window and `http_timeout 1800`.

## Core concepts

- **Baseline `nelson.db` is sacrosanct.** Adding a model mutates it, so
  `add-model.sh` always `cp -n` backs it up first. Experiments use their own
  isolated DBs (`nelson-*.db`, all gitignored) and never touch the baseline.
- **Surgical add vs roster sync.** `bench loop --competitors roster.yaml` *syncs*:
  every active model absent from the file is retired and dropped from the report.
  That's wrong when you only want to ADD a model. So `add-model.sh` upserts just
  the roster's competitors (via `lib/surgical_upsert.py`) and runs `bench loop`
  with **no** `--competitors`; the matrix planner then fills only the new
  `(model × case × file)` cells and every prior result stays visible.
- **Scoring** is a localization gate → Opus truth judge (hit/miss) plus a
  code-grounded FP judge (precision). `bench loop` scores automatically; the
  arm-sweep experiments instead read detection straight from `run_findings`
  (localization only, no judge spend) and FP-judge separately if needed.

## Workflows

### 1. Add a model to the baseline leaderboard

Write a one-competitor roster in `rosters/` (copy an existing one), then:

```bash
bench-scripts/add-model.sh bench-scripts/rosters/competitors-qwen-agentworld.yaml \
    --concurrency 1        # 1 for a single local server; 3 for hosted cohorts
```

Re-pointing a model to a new endpoint (clear its stale runs so they re-plan):

```bash
bench-scripts/add-model.sh bench-scripts/rosters/competitors-north-mini-code-openrouter.yaml \
    --clear raw-api-loop/north-mini-code
```

Clearing only a failed batch (e.g. server OOM) so a plain loop re-runs them:

```bash
bench-scripts/add-model.sh bench-scripts/rosters/competitors-local-gemma-minimax.yaml \
    --clear raw-api-loop/gemma4-31b --clear-status infra_error --timeout 3600
```

The report defaults to `bench-report-<roster-stem>.html`; override with `--html`.

### 2. Run an isolated-DB experiment

For A/Bs and probes that should not touch the baseline (tool profiles, repo-scope,
case subsets). The roster IS the source of truth here, so this DOES pass
`--competitors`. Idempotent — re-running resumes; `--fresh` starts clean.

```bash
# semgrep tool A/B, fresh isolated DB, 9 canonical cases
bench-scripts/run-experiment.sh --db nelson-semgrep-exp.db \
    --roster bench-scripts/rosters/competitors-semgrep-ab.yaml --cases-dir cases/ --fresh

# repo-scope A/B over a case subset, 3 trials
bench-scripts/run-experiment.sh --db nelson-reposcope-exp.db \
    --roster bench-scripts/rosters/competitors-reposcope.yaml \
    --cases-dir bench-scripts/case-subsets/cases-reposcope/ --repeat 3 --concurrency 3

# oracle-CWE paired A/B = two passes into two DBs
bench-scripts/run-experiment.sh --db nelson-oracle.db --oracle-cwe \
    --roster bench-scripts/rosters/competitors-cheapfree.yaml --cases-dir cases/ \
    --concurrency 7 --html bench-oracle.html
bench-scripts/run-experiment.sh --db nelson-oracle-control.db \
    --roster bench-scripts/rosters/competitors-cheapfree.yaml --cases-dir cases/ \
    --concurrency 7 --html bench-oracle-control.html
```

`--oracle-cwe` and `--repo-scope` are flags; anything else passes through after `--`.

### 3. Run a prompt-arm or CWE sweep

These vary each cell N ways and encode the variant in the run's `trial` index
(no schema change). The experiment is a YAML in `runners/configs/`:

- **`mode: arms`** — N non-leaking prompting strategies (`open`/`plan`/`checklist`),
  each repeated `repeat` times. Trial = `arm_idx*repeat + r`.
- **`mode: cwe-sweep`** — one oracle-CWE hint per applicable weakness class for the
  file's language, plus a final open baseline. One trial per CWE.

```bash
# one model, foreground
bench-scripts/run-arm-sweep.sh --config bench-scripts/runners/configs/promptlab-qwen.yaml \
    --db nelson-promptlab.db --model raw-api-loop/qwen3.6-27b

# several models (e.g. one per self-hosted box), shared DB, parallel + staggered
bench-scripts/run-arm-sweep.sh --config bench-scripts/runners/configs/promptlab-qwen.yaml \
    --db nelson-promptlab-4bit.db \
    --model raw-api-loop/qwen3.6-27b-q4-k-xl --model raw-api-loop/qwen3.6-35b-A3b-q4-k-xl
```

### 4. FP-judge off-target findings

After a sweep, classify the findings that AREN'T the planted CVE as real secondary
bugs vs hallucinations (code-grounded Opus, reads pre-patch source, never the
advisory). Deduped by distinct site and idempotent.

```bash
# the four-quant Qwen sweep (per-tier breakdown)
bench-scripts/judges/fpjudge.py \
    --db nelson-promptlab-bf16.db --db nelson-promptlab.db \
    --db nelson-promptlab-6bit.db --db nelson-promptlab-4bit.db \
    --label BF16 --label Q8 --label Q6 --label Q4
# a single DB; or one case only
bench-scripts/judges/fpjudge.py --db nelson-gemma-promptlab.db
bench-scripts/judges/fpjudge.py --db nelson-fp-sweep.db --case GHSA-9f49-8x56-jmjc
```

### 5. Reports and analysis

`reports/*.py` write self-contained HTML (shared theme via `nelson.report_style`);
`analysis/*.py` print console summaries. Each takes `--db`/`--out`; run with
`--help`. The promptlab reports share `reports/analyze_promptlab.py` as their data
layer (it lives in `reports/` so the cluster's sibling imports resolve with no
path hacks). Example:

```bash
bench-scripts/reports/promptlab_report.py --db nelson-promptlab.db --out promptlab-report.html
bench-scripts/analysis/analyze_oracle.py   # reads nelson-oracle{,-control}.db
```

## Experiment catalog

Each row is one experiment we ran, with the generalized command that reproduces
it and the memory note holding the result. DBs/reports are gitignored artifacts.

| Experiment | Command (run from repo root) | Memory |
|---|---|---|
| Add models to baseline (newbatch cohorts, gemma+minimax, qwen-agentworld, …) | `add-model.sh bench-scripts/rosters/competitors-<cohort>.yaml [--concurrency N] [--clear …]` | `nelson-newbatch-*`, `nelson-gemma-minimax-baseline`, `nelson-qwen-agentworld`, `nelson-openrouter-cohort` |
| Oracle-CWE A/B (cheap/free) | `run-experiment.sh` twice → `nelson-oracle.db --oracle-cwe` + `nelson-oracle-control.db`; `analysis/analyze_oracle.py` | `nelson-oracle-cwe-experiment` |
| CWE-sweep FP probe | `run-arm-sweep.sh --config …/cwe-fp-sweep.yaml --db nelson-fp-sweep.db --model …`; then `judges/fpjudge.py --db nelson-fp-sweep.db --case …`; `reports/cwe_fp_report.py` | `nelson-cwe-sweep-fp-experiment` |
| Prompt-lab (Qwen, 6 cases) + quant sweep (BF16/Q8/Q6/Q4) | `run-arm-sweep.sh --config …/promptlab-qwen.yaml --db nelson-promptlab[-Nbit].db --model …`; `reports/promptlab_report.py`, `reports/promptlab_compare_report.py`; `judges/fpjudge.py` (4 DBs) | `nelson-promptlab-experiment` |
| Prompt-lab (Gemma, full 9 cases) | `run-arm-sweep.sh --config …/promptlab-gemma.yaml --db nelson-gemma-promptlab.db --model …`; `reports/gemma_promptlab_report.py`; `judges/fpjudge.py --db nelson-gemma-promptlab.db` | `nelson-gemma-qat-experiment` |
| Semgrep tool A/B | `run-experiment.sh --db nelson-semgrep-exp.db --roster …/competitors-semgrep-ab.yaml --cases-dir cases/ --fresh` | `nelson-semgrep-tool-experiment` |
| Tree-sitter tool A/B (+ Gemma follow-ups) | `run-experiment.sh --db nelson-treesitter-*.db --roster …/competitors-treesitter-*.yaml --cases-dir bench-scripts/case-subsets/cases-treesitter-*/ --repeat 3` | `nelson-treesitter-tool-experiment` |
| Repo-scope vs file-scope | `run-experiment.sh --db nelson-reposcope-exp.db --roster …/competitors-reposcope.yaml --cases-dir bench-scripts/case-subsets/cases-reposcope/ --repeat 3 --concurrency 3` | `nelson-reposcope-experiment` |
| Shadow truth-judge | `judges/shadow_judge.py` → `reports/shadow_judge_report.py`, `reports/shadow_disagreement_report.py` | `nelson-shadow-judge-experiment` |
| Shadow FP-judge | `judges/shadow_fp_judge.py` → `reports/shadow_fp_judge_report.py` | `nelson-shadow-fp-judge-experiment` |

## archive/

One-shot scripts that already did their job against the frozen baseline and are
kept only for provenance (the misnumber line-number correction and its verifier,
the gemma trajectory probe, and the post-correction report regen). They are not
part of any reproducible workflow — see each file's docstring.

## A note on consolidation

The run scripts, the three per-experiment trial runners, and the three FP-judges
each had real duplication and were merged into single config/flag-driven tools
(`add-model.sh`/`run-experiment.sh`/`run-arm-sweep.sh`, `runners/arm_sweep.py`,
`judges/fpjudge.py`). The HTML report generators were **not** force-merged: they
already share `nelson.report_style` and (for the promptlab family)
`analyze_promptlab`, and they render structurally different reports for different
experiments, so collapsing them into one renderer would add risk (silently
changing already-published output) for little gain. They were relocated and the
shared data layer kept shared, but each keeps its own entry point.
