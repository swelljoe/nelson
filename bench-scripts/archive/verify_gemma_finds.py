#!/usr/bin/env python3
"""Certainty check: with read_file now emitting line numbers, does gemma4-26b-a4b
report the planted bug at a line that PASSES the real localization gate, and does
the Opus truth judge rule it the SAME bug?

Faithful path: same prompt/tools/temp as the benchmark; parses findings with the
benchmark's own ``parse_competitor_findings``; scores with the benchmark's own
``score.localize`` (tolerance 10) + ``score.ClaudeTruthJudge`` (Opus). Baseline
nelson.db is never touched — nothing is persisted; verdicts print to stdout.

Usage:
  verify_gemma_finds.py --server http://10.20.30.1:8000/v1 \
      --case CVE-2026-5199 --target service/worker/batcher/activities.go \
      --label act --reps 3
"""

from __future__ import annotations

import argparse
import pathlib
import time

from nelson import corpus, score
from nelson import raw_api_loop as ral
from nelson.runner import (
    build_competitor_prompt,
    parse_competitor_findings,
    prepare_checkout,
)

MODEL = "unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q8_K_XL"


def run_one(case, target, src_root, server, label, rep):
    prompt = build_competitor_prompt(case, target)
    t0 = time.monotonic()
    state = {"n": 0}

    def timed_post(url, payload, api_key):
        resp = ral._post_chat(url, payload, api_key)
        msg = (resp.get("choices") or [{}])[0].get("message") or {}
        tcs = msg.get("tool_calls") or []
        names = [((tc.get("function") or {}).get("name")) for tc in tcs]
        print(
            f"  [{label}#{rep}] turn {state['n']:2d} +{time.monotonic() - t0:6.0f}s "
            f"tools={names} final={'Y' if not tcs else '-'}",
            flush=True,
        )
        state["n"] += 1
        return resp

    final_text, tin, tout, steps, _ = ral.run_loop(
        prompt,
        base_url=server,
        model=MODEL,
        api_key="lm-studio",
        max_steps=40,
        token_budget=500_000,
        temperature=0.1,
        post=timed_post,
        src_root=src_root,
    )
    return final_text, round(time.monotonic() - t0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    case = {c.ext_id: c for c in corpus.load_manifest_dir("cases")}[args.case]
    src_root = str(
        prepare_checkout(
            case.repo_url, case.vuln_commit, pathlib.Path(f"bench-cache/{case.ext_id}")
        )
    )
    gt = [h for h in case.gt_hunks if h.get("file") == args.target]
    print(
        f"### {args.case} / {args.target}  GT hunks: "
        f"{[(h['start'], h['end']) for h in gt]}  (tolerance {score.DEFAULT_LINE_TOLERANCE})"
    )

    judge = score.ClaudeTruthJudge(model="opus")
    for rep in range(1, args.reps + 1):
        try:
            final_text, dur = run_one(
                case, args.target, src_root, args.server, args.label, rep
            )
        except Exception as e:
            print(f"  [{args.label}#{rep}] ERROR {type(e).__name__}: {e}", flush=True)
            continue
        findings = parse_competitor_findings(final_text)
        print(f"  [{args.label}#{rep}] {dur}s -> {len(findings)} finding(s)")
        verdict_line = "NO-LOCALIZED-FINDING"
        for f in findings:
            line = f.get("line")
            try:
                line = int(line)
            except (TypeError, ValueError):
                line = None
            loc = score.localize(f.get("file"), line, gt, score.DEFAULT_LINE_TOLERANCE)
            tag = "GATE-PASS" if loc.matched else "gate-fail"
            print(
                f"      L{line} {tag}  cwe={f.get('cwe')}  {str(f.get('explanation'))[:90]}"
            )
            if loc.matched:
                rf = score.ReportedFinding(
                    file=f.get("file"),
                    line=line,
                    description=f.get("explanation"),
                    cwe=f.get("cwe"),
                )
                tv = judge.judge(case, rf)
                if tv.error:
                    verdict_line = f"GATE-PASS @L{line}, judge ERROR: {tv.error}"
                else:
                    verdict_line = (
                        f"GATE-PASS @L{line}, judge: "
                        f"{'SAME_BUG (HIT)' if tv.same_bug else 'different_bug (miss)'}"
                        f" — {tv.reasoning}"
                    )
        print(f"  [{args.label}#{rep}] VERDICT: {verdict_line}\n", flush=True)


if __name__ == "__main__":
    main()
