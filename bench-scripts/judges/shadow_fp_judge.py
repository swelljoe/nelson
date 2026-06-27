#!/usr/bin/env python3
"""Shadow FP-judging experiment: can cheap/local models replicate the Opus FP judge?

Companion to shadow_judge.py (which tested the *truth* judge — "is this the same bug
as the advisory?"). This one tests the *FP / precision* judge — "is this finding a
REAL, exploitable bug or a FALSE POSITIVE?" — which is a different, advisory-blind job:
the production judge (`claude -p`, Opus) is shown ONLY the finding + the pre-patch
source (never the advisory), and rules real_bug / false_positive / needs_review.

The motivation: the truth-judge shadows all leaned *lenient* (over-called same_bug).
A lenient bias is benign-to-helpful for matching, but for FP triage a lenient judge
waves marginal findings *through* as real — the opposite of reducing false positives.
So whether cheap models can do the FP job needs its own measurement.

We replay every banked Opus FP verdict (real_bug / false_positive) by rebuilding the
EXACT production input via the shared `build_fp_prompt(finding, source)` — source
reconstructed from `git show vuln_commit:path` through GitCodeProvider, byte-identical
to what Opus saw — and record each shadow's verdict alongside Opus's.

Idempotent + resumable: every (model, input_hash, trial) cell is written once to
nelson-shadow-fp-judge.db. Source DBs are opened READ-ONLY (mode=ro) — the sacrosanct
nelson.db is never written.

Usage:
    .venv/bin/python shadow_fp_judge.py [--limit N] [--models deepseek,mimo,gemma31b]
                                        [--trials 3] [--out nelson-shadow-fp-judge.db]
"""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from nelson.corpus import Case
from nelson.score import (
    GitCodeProvider,
    ReportedFinding,
    build_fp_prompt,
    parse_fp_verdict,
)

# -- Config ------------------------------------------------------------------

# Source DBs that hold Opus FP verdicts. Read-only; baseline first. Same six DBs as
# the truth-judge experiment so the eval set is drawn from the identical corpus runs.
SOURCE_DBS = [
    "nelson.db",
    "nelson-exp.db",
    "nelson-gemma-promptlab.db",
    "nelson-oracle.db",
    "nelson-oracle-control.db",
    "nelson-repeat.db",
]


def _read_key(path: str) -> str:
    return Path(path).read_text().strip()


# Each shadow judge is an OpenAI-compatible chat endpoint. `key` is resolved lazily
# (so --models can run a subset without needing every secret present). Identical
# roster to shadow_judge.py so the two experiments are directly comparable.
SHADOW_MODELS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-v4-pro",
        "key": lambda: _read_key("/home/joe/secrets/deepseek"),
        "timeout": 180.0,
    },
    "mimo": {
        "base_url": "https://token-plan-sgp.xiaomimimo.com/v1",
        "model": "mimo-v2.5-pro",
        "key": lambda: _read_key("/home/joe/secrets/mimo"),
        "timeout": 180.0,
    },
    "gemma31b": {
        "base_url": "http://10.20.30.1:8000/v1",
        "model": "unsloth/gemma-4-31B-it-qat-GGUF:UD-Q4_K_XL",
        "key": lambda: "lm-studio",
        "timeout": 600.0,  # local APU/GPU, generous
    },
}

TEMPERATURE = 0.0  # ask for the model's single best call; trial spread = its own noise
# The FP prompt carries ~200 lines of source, so reasoning models think at length
# before the verdict JSON. At 2048 they ran out of budget mid-reasoning and never
# emitted it (50-67% unparseable). 8192 gives ample room to finish + answer.
MAX_TOKENS = 8192
MAX_RETRIES = 5

_RESULT_DDL = """
CREATE TABLE IF NOT EXISTS shadow_fp_verdicts (
    model           TEXT NOT NULL,
    trial           INTEGER NOT NULL,
    input_hash      TEXT NOT NULL,
    source_db       TEXT,
    case_ext_id     TEXT,
    finding_file    TEXT,
    finding_line    INTEGER,
    opus_real       INTEGER,        -- reference: 1 real_bug / 0 false_positive
    opus_conflict   INTEGER,        -- 1 if Opus gave both labels on identical input
    shadow_real     INTEGER,        -- 1/0, NULL if needs_review/unparseable/errored
    shadow_label    TEXT,           -- real_bug / false_positive / needs_review / NULL
    raw_reply       TEXT,
    latency_ms      INTEGER,
    tokens_out      INTEGER,
    error           TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (model, input_hash, trial)
);
"""

# -- Eval-set loading --------------------------------------------------------


def _load_eval_set() -> tuple[list[dict], dict[str, int]]:
    """Collect every (finding, source) input Opus FP-judged, deduped by prompt.

    Returns (items, stats). Each item is {input_hash, prompt, opus_real, opus_conflict,
    source_db, case_ext_id, finding_file, finding_line}. The prompt is rebuilt from the
    finding + pre-patch source (GitCodeProvider) so it is byte-identical to Opus's
    input; identical prompts across DBs collapse to one row. If Opus labelled identical
    input both ways, opus_conflict=1.

    A finding whose source can no longer be fetched (None) is SKIPPED, not guessed —
    we cannot reconstruct the exact prompt without it. (In practice source is a
    deterministic `git show` at a fixed commit, so a verdict that existed should
    reconstruct; the skip count is a safety report, expected 0.)
    """
    provider = GitCodeProvider()
    by_hash: dict[str, dict] = {}
    stats = {"rows": 0, "skipped_no_source": 0, "skipped_no_file": 0}
    for db_path in SOURCE_DBS:
        if not Path(db_path).exists():
            continue
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        case_cache: dict[int, Case] = {}
        rows = conn.execute(
            """SELECT rf.*, r.case_id AS _case_id
               FROM run_findings rf JOIN runs r ON rf.run_id = r.id
               WHERE rf.judge_fp_verdict IN ('real_bug', 'false_positive')"""
        ).fetchall()
        for row in rows:
            stats["rows"] += 1
            cid = row["_case_id"]
            if cid not in case_cache:
                crow = conn.execute(
                    "SELECT * FROM cases WHERE id = ?", (cid,)
                ).fetchone()
                if crow is None:
                    continue
                case_cache[cid] = Case.from_row(crow)
            case = case_cache[cid]
            finding = ReportedFinding.from_row(row)
            if not finding.file:
                stats["skipped_no_file"] += 1
                continue
            source = provider.source(case, finding.file)
            if source is None:
                stats["skipped_no_source"] += 1
                continue
            prompt = build_fp_prompt(finding, source)
            h = hashlib.sha256(prompt.encode()).hexdigest()[:16]
            opus_real = 1 if row["judge_fp_verdict"] == "real_bug" else 0
            if h in by_hash:
                if by_hash[h]["opus_real"] != opus_real:
                    by_hash[h]["opus_conflict"] = 1
                continue
            by_hash[h] = {
                "input_hash": h,
                "prompt": prompt,
                "opus_real": opus_real,
                "opus_conflict": 0,
                "source_db": db_path,
                "case_ext_id": case.ext_id,
                "finding_file": finding.file,
                "finding_line": finding.line,
            }
        conn.close()
    return list(by_hash.values()), stats


# -- One judge call ----------------------------------------------------------


def _call(client: httpx.Client, cfg: dict, key: str, prompt: str) -> dict:
    """POST one chat completion; retry transient failures with backoff.

    Returns {shadow_real, shadow_label, raw_reply, latency_ms, tokens_out, error}.
    A clean ``needs_review`` is a valid abstention (shadow_real=None, label set,
    error=None); an unparseable/transport failure also yields shadow_real=None but
    with error set. The verdict is never guessed.
    """
    url = f"{cfg['base_url'].rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }
    last_err = "unknown"
    for attempt in range(MAX_RETRIES):
        t0 = time.monotonic()
        try:
            resp = client.post(
                url, json=payload, headers=headers, timeout=cfg["timeout"]
            )
            latency_ms = int((time.monotonic() - t0) * 1000)
            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = f"http {resp.status_code}"
                time.sleep(min(2**attempt, 30))
                continue
            resp.raise_for_status()
            data = resp.json()
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            content = msg.get("content") or ""
            # Some reasoning backends stash the answer beside the chain of thought.
            if not content.strip():
                content = msg.get("reasoning_content") or content
            tokens_out = (data.get("usage") or {}).get("completion_tokens")
            parsed = parse_fp_verdict(content)
            if parsed is None:
                return {
                    "shadow_real": None,
                    "shadow_label": None,
                    "raw_reply": content[:2000],
                    "latency_ms": latency_ms,
                    "tokens_out": tokens_out,
                    "error": "unparseable",
                }
            is_real, _reasoning = parsed
            if is_real is None:  # explicit needs_review — a valid abstention
                return {
                    "shadow_real": None,
                    "shadow_label": "needs_review",
                    "raw_reply": content[:2000],
                    "latency_ms": latency_ms,
                    "tokens_out": tokens_out,
                    "error": None,
                }
            return {
                "shadow_real": 1 if is_real else 0,
                "shadow_label": "real_bug" if is_real else "false_positive",
                "raw_reply": content[:2000],
                "latency_ms": latency_ms,
                "tokens_out": tokens_out,
                "error": None,
            }
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(min(2**attempt, 30))
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(min(2**attempt, 30))
    return {
        "shadow_real": None,
        "shadow_label": None,
        "raw_reply": None,
        "latency_ms": None,
        "tokens_out": None,
        "error": last_err[:300],
    }


# -- Per-model worker --------------------------------------------------------


def _run_model(
    model_key: str, cfg: dict, eval_set: list[dict], trials: int, out_db: str
) -> None:
    try:
        key = cfg["key"]()
    except Exception as exc:
        print(f"[{model_key}] SKIP — cannot read key: {exc}", flush=True)
        return

    conn = sqlite3.connect(out_db, timeout=60.0)
    conn.execute("PRAGMA busy_timeout=60000")
    done = {
        (r[0], r[1])
        for r in conn.execute(
            "SELECT input_hash, trial FROM shadow_fp_verdicts WHERE model = ?",
            (model_key,),
        )
    }
    todo = [
        (item, t)
        for item in eval_set
        for t in range(trials)
        if (item["input_hash"], t) not in done
    ]
    print(
        f"[{model_key}] {len(eval_set)} inputs x {trials} trials = "
        f"{len(eval_set) * trials}; {len(done)} done, {len(todo)} to do",
        flush=True,
    )

    n = 0
    agree = 0
    scored = 0
    with httpx.Client() as client:
        for item, trial in todo:
            res = _call(client, cfg, key, item["prompt"])
            conn.execute(
                """INSERT OR IGNORE INTO shadow_fp_verdicts
                   (model, trial, input_hash, source_db, case_ext_id, finding_file,
                    finding_line, opus_real, opus_conflict, shadow_real,
                    shadow_label, raw_reply, latency_ms, tokens_out, error)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    model_key,
                    trial,
                    item["input_hash"],
                    item["source_db"],
                    item["case_ext_id"],
                    item["finding_file"],
                    item["finding_line"],
                    item["opus_real"],
                    item["opus_conflict"],
                    res["shadow_real"],
                    res["shadow_label"],
                    res["raw_reply"],
                    res["latency_ms"],
                    res["tokens_out"],
                    res["error"],
                ),
            )
            conn.commit()
            n += 1
            if res["shadow_real"] is not None and not item["opus_conflict"]:
                scored += 1
                if res["shadow_real"] == item["opus_real"]:
                    agree += 1
            if n % 25 == 0 or n == len(todo):
                acc = f"{agree / scored:.0%}" if scored else "n/a"
                print(
                    f"[{model_key}] {n}/{len(todo)} done, "
                    f"agree-w-opus {agree}/{scored} ({acc})",
                    flush=True,
                )
    conn.close()
    print(f"[{model_key}] FINISHED", flush=True)


# -- Main --------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="cap inputs (smoke test)")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--models", default=",".join(SHADOW_MODELS))
    ap.add_argument("--out", default="nelson-shadow-fp-judge.db")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    bad = [m for m in models if m not in SHADOW_MODELS]
    if bad:
        print(f"unknown model(s): {bad}; known: {list(SHADOW_MODELS)}")
        return 2

    eval_set, stats = _load_eval_set()
    if args.limit:
        eval_set = eval_set[: args.limit]
    n_real = sum(i["opus_real"] for i in eval_set)
    n_conf = sum(i["opus_conflict"] for i in eval_set)
    print(
        f"eval set: {len(eval_set)} distinct inputs "
        f"(opus real={n_real} fp={len(eval_set) - n_real}, "
        f"self-conflict={n_conf}); from {stats['rows']} verdict rows "
        f"(skipped {stats['skipped_no_source']} no-source, "
        f"{stats['skipped_no_file']} no-file); models={models}; trials={args.trials}",
        flush=True,
    )
    if not eval_set:
        print("empty eval set — nothing to do", flush=True)
        return 1

    conn = sqlite3.connect(args.out)
    conn.executescript(_RESULT_DDL)
    conn.commit()
    conn.close()

    # One thread per model: ≤1 in-flight request per endpoint (rate-limit safe),
    # and a slow/dead Gemma cannot block DeepSeek or MiMo.
    with ThreadPoolExecutor(max_workers=len(models)) as pool:
        futs = [
            pool.submit(
                _run_model, m, SHADOW_MODELS[m], eval_set, args.trials, args.out
            )
            for m in models
        ]
        for f in futs:
            f.result()

    print("ALL MODELS DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
