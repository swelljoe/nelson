#!/usr/bin/env python3
"""Config-driven trial-sweep runner — the generalized form of the three untracked
experiment runners (run_promptlab_experiment.py, run_gemma_promptlab_experiment.py,
run_cwe_fp_experiment.py).

All three did the same thing: take one competitor, audit a fixed (case -> files)
map, and run several variants of each cell, encoding the variant in the run's
``trial`` index so no schema change is needed. They differed only in WHAT the
variants are:

  * mode ``arms``     — N non-leaking prompting strategies (e.g. open / plan /
                        checklist), each repeated ``repeat`` times. The variant is
                        the prompt_mode; trial = arm_idx * repeat + r, so an
                        analyzer decodes arm = trial // repeat.
  * mode ``cwe-sweep`` — one oracle-CWE hint per applicable weakness class for the
                        file's language, plus a final open baseline. The variant is
                        which CWE is leaked into the prompt; trial i in
                        [0, len(cwes)) hints applicable_cwes(language)[i] and trial
                        len(cwes) is the open baseline.

Everything else (load competitor, upsert cases, BenchRunner, per-cell resume
skip, checkout pre-warm, run_case + logging) is shared and identical to the
originals, so detection still comes straight from run_findings with no judge.

The experiment is described by a YAML config (see runners/configs/*.yaml):

    mode: arms                                  # or: cwe-sweep
    competitors: rosters/competitors-...yaml    # roster the --model must be in
    cases_dir: cases/                           # case manifest dir
    repeat: 2                                    # arms mode only; cwe-sweep is 1/CWE
    arms: [open, plan, checklist]               # arms mode only
    targets:                                     # case ext_id -> [source files]
      GHSA-9f49-8x56-jmjc: [src/parser_common.c]
      GHSA-cc7p-2j3x-x7xf: [src/web/UrlManager.php, src/web/Request.php]

Usage:
    LMSTUDIO_API_KEY=lm-studio python arm_sweep.py \
        --config runners/configs/promptlab-qwen.yaml \
        --model raw-api-loop/qwen3.6-27b --db nelson-promptlab.db [--repeat N]

Idempotent + resumable: a (competitor, case, file, trial) that already has a
settled (complete/pending/running) run is skipped.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

import yaml

from nelson.automate import existing_run_status_map, load_competitors
from nelson.corpus import load_manifest_dir
from nelson.cwe import applicable_cwes
from nelson.db import Database
from nelson.inventory import LANGUAGE_MAP
from nelson.runner import BenchRunner, RunnerError

log = logging.getLogger("arm_sweep")

# Statuses that mean a trial already had a fair look (mirror automate's set).
_SETTLED = frozenset({"complete", "pending", "running"})


def _language(file_path: str) -> str:
    lang = LANGUAGE_MAP.get(Path(file_path).suffix.lower())
    if lang is None:
        raise SystemExit(f"no language mapping for {file_path}")
    return lang


@dataclasses.dataclass
class Config:
    mode: str  # "arms" | "cwe-sweep"
    competitors: str
    cases_dir: str
    targets: dict[str, list[str]]
    arms: list[str]
    repeat: int

    @classmethod
    def load(cls, path: str) -> "Config":
        raw = yaml.safe_load(Path(path).read_text())
        mode = raw.get("mode", "arms")
        if mode not in ("arms", "cwe-sweep"):
            raise SystemExit(f"unknown mode {mode!r} (arms | cwe-sweep)")
        targets = {k: (v if isinstance(v, list) else [v]) for k, v in raw["targets"].items()}
        return cls(
            mode=mode,
            competitors=raw["competitors"],
            cases_dir=raw.get("cases_dir", "cases/"),
            targets=targets,
            arms=raw.get("arms", ["open"]),
            repeat=int(raw.get("repeat", 1)),
        )


def _plan_arms(cfg: Config, repeat: int):
    """Yield (trial, arm_name, prompt_mode, cwe_override) for arms mode."""
    for arm_idx, arm in enumerate(cfg.arms):
        for r in range(repeat):
            yield arm_idx * repeat + r, f"{arm}#{r}", arm, None


def _plan_cwe_sweep(file_path: str):
    """Yield (trial, label, prompt_mode, cwe_override) for cwe-sweep mode."""
    cwes = [c.id for c in applicable_cwes(_language(file_path))]
    for i, cid in enumerate(cwes):
        yield i, cid, None, cid
    yield len(cwes), "OPEN", None, ""  # final open baseline (empty cwe = no hint)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="experiment YAML (see configs/)")
    ap.add_argument("--model", required=True, help="competitor name to run")
    ap.add_argument("--db", required=True, help="experiment SQLite db path")
    ap.add_argument(
        "--repeat",
        type=int,
        default=None,
        help="override config repeat (arms mode only)",
    )
    ap.add_argument("--timeout", type=float, default=1800.0, help="per-run wall cap")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = Config.load(args.config)
    repeat = max(1, args.repeat if args.repeat is not None else cfg.repeat)
    if cfg.mode == "cwe-sweep" and (args.repeat or cfg.repeat) not in (1, None):
        log.warning("cwe-sweep does one trial per CWE; ignoring repeat=%s", repeat)

    db = Database(args.db)

    competitors = {c.name: c for c in load_competitors(cfg.competitors)}
    if args.model not in competitors:
        raise SystemExit(
            f"model {args.model!r} not in {cfg.competitors}: {sorted(competitors)}"
        )
    comp = competitors[args.model]

    cases = {c.ext_id: c for c in load_manifest_dir(cfg.cases_dir)}
    targets = []  # (case, [files])
    for ext_id, files in cfg.targets.items():
        if ext_id not in cases:
            raise SystemExit(f"case {ext_id} not found in {cfg.cases_dir}")
        case = cases[ext_id]
        case.id = db.upsert_case(case.to_db_fields())  # row id links create_run
        targets.append((case, files))

    runner = BenchRunner(
        db,
        run_timeout=args.timeout,
        max_budget_usd=None,  # local/free experiments — no cost backstop
        oracle_cwe=(cfg.mode == "cwe-sweep"),  # the hint reads case.cwe live
    )

    settled = existing_run_status_map(db.list_runs())

    def already_done(case_ext: str, file: str, trial: int) -> bool:
        statuses = settled.get((args.model, case_ext, file, trial), set())
        return bool(statuses & _SETTLED)

    # Pre-warm each checkout once so per-trial runs hit the read-only path.
    for case, _files in targets:
        try:
            runner.prewarm_checkout(case)
        except RunnerError as e:
            raise SystemExit(f"checkout failed for {case.ext_id}: {e}") from e

    n_files = sum(len(f) for _c, f in targets)
    log.info(
        "%s [%s]: %d cases, %d files, repeat=%d",
        args.model,
        cfg.mode,
        len(targets),
        n_files,
        repeat,
    )

    total = ran = skipped = 0
    for case, files in targets:
        for target_file in files:
            short = target_file.rsplit("/", 1)[-1]
            if cfg.mode == "arms":
                plan = list(_plan_arms(cfg, repeat))
            else:
                plan = list(_plan_cwe_sweep(target_file))
            log.info("case %s file %s: %d trials", case.ext_id, short, len(plan))
            for trial, label, prompt_mode, cwe_override in plan:
                total += 1
                if already_done(case.ext_id, target_file, trial):
                    skipped += 1
                    log.info("  skip %s trial %d (%s): settled", short, trial, label)
                    continue
                ran += 1
                # arms mode varies the prompt_mode; cwe-sweep varies case.cwe.
                if prompt_mode is not None:
                    runner.prompt_mode = prompt_mode
                run_case = case
                if cwe_override is not None:
                    run_case = dataclasses.replace(case, cwe=cwe_override)
                    run_case.id = case.id
                log.info("  run %s trial %d (%s) ...", short, trial, label)
                result = runner.run_case(run_case, comp, target_file, trial=trial)
                log.info(
                    "  %s trial %d (%s) -> %s (tokens_out=%s, %.0fs)",
                    short,
                    trial,
                    label,
                    result.status,
                    result.tokens_out,
                    result.wall_clock_s or 0.0,
                )

    log.info("DONE %s: %d planned, %d ran, %d skipped", args.model, total, ran, skipped)


if __name__ == "__main__":
    main()
