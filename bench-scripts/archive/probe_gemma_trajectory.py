#!/usr/bin/env python3
"""Diagnostic probe: replay one benchmark cell against a live server and stream
the per-turn trajectory, so we can see WHY gemma4-26b-a4b timed out on the two
cells (activities.go, draw.c) instead of inferring it from an empty transcript.

Faithful to the benchmark path: same SYSTEM_PROMPT/TOOLS, same audit prompt
(build_competitor_prompt), same temperature 0.1 / 500k budget / 40 steps, same
read-grep tools over the pristine checkout. The ONLY addition is a timing wrapper
around the HTTP post that prints, live, for every turn:
  - wall time of that single chat call (isolates per-turn latency)
  - prompt size sent (context growth) + tokens in/out the server reports
  - whether tools were offered (force_final on the last turn)
  - the exact tool_calls the model emitted (name + args), or [] = it answered

A format failure shows as turn 0 returning tool_calls=[] fast. A throughput
failure shows as few turns each taking minutes. A spin shows as many turns
repeating the same read. No nelson/ source is modified.

Usage:
  probe_gemma_trajectory.py --server http://10.20.30.1:8000/v1 \
      --case CVE-2026-5199 --target service/worker/batcher/activities.go \
      --label v620-activities [--max-steps 40]
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from nelson import corpus
from nelson import raw_api_loop as ral
from nelson.runner import build_competitor_prompt, prepare_checkout

MODEL = "unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q8_K_XL"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--server", required=True, help="base_url, e.g. http://10.20.30.1:8000/v1"
    )
    ap.add_argument("--case", required=True, help="case ext_id")
    ap.add_argument("--target", required=True, help="target file path within the tree")
    ap.add_argument("--label", required=True, help="short tag for log lines")
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    cases = {c.ext_id: c for c in corpus.load_manifest_dir("cases")}
    case = cases[args.case]
    src_root = str(
        prepare_checkout(
            case.repo_url,
            case.vuln_commit,
            __import__("pathlib").Path(f"bench-cache/{case.ext_id}"),
        )
    )
    prompt = build_competitor_prompt(case, args.target)

    lbl = args.label
    run_t0 = time.monotonic()
    state = {"n": 0}

    def timed_post(url, payload, api_key):
        n = state["n"]
        msgs = payload.get("messages", [])
        prompt_chars = sum(len(json.dumps(m)) for m in msgs)
        has_tools = "tools" in payload
        t0 = time.monotonic()
        resp = ral._post_chat(url, payload, api_key)
        dt = time.monotonic() - t0
        usage = resp.get("usage") or {}
        msg = (resp.get("choices") or [{}])[0].get("message") or {}
        tcs = msg.get("tool_calls") or []
        summ = []
        for tc in tcs:
            fn = tc.get("function") or {}
            summ.append(f"{fn.get('name')}({str(fn.get('arguments'))[:140]})")
        content = msg.get("content") or ""
        elapsed = time.monotonic() - run_t0
        print(
            f"[{lbl}] turn {n:2d} | +{elapsed:7.1f}s run | dt={dt:6.1f}s "
            f"| ctx_chars={prompt_chars:8d} | in={usage.get('prompt_tokens')} "
            f"out={usage.get('completion_tokens')} | tools_offered={'Y' if has_tools else 'FINAL'} "
            f"| tool_calls={len(tcs)}: {summ} | content[{len(content)}]={content[:200]!r}",
            flush=True,
        )
        state["n"] += 1
        return resp

    print(
        f"[{lbl}] START case={args.case} target={args.target} server={args.server}",
        flush=True,
    )
    print(f"[{lbl}] src_root={src_root}", flush=True)
    print(f"[{lbl}] prompt_len={len(prompt)} chars", flush=True)
    try:
        final_text, tin, tout, steps, cost = ral.run_loop(
            prompt,
            base_url=args.server,
            model=args.model,
            api_key="lm-studio",
            max_steps=args.max_steps,
            token_budget=500_000,
            temperature=0.1,
            post=timed_post,
            src_root=src_root,
        )
    except Exception as e:
        print(
            f"[{lbl}] EXCEPTION after {time.monotonic() - run_t0:.1f}s: {type(e).__name__}: {e}",
            flush=True,
        )
        sys.exit(1)

    elapsed = time.monotonic() - run_t0
    print(
        f"[{lbl}] DONE in {elapsed:.1f}s | turns={len(steps)} | in={tin} out={tout} "
        f"| final[{len(final_text)}]={final_text[:400]!r}",
        flush=True,
    )


if __name__ == "__main__":
    main()
