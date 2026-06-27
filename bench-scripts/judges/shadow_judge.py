#!/usr/bin/env python3
"""Shadow-judging experiment: can cheap/local models replicate the Opus truth judge?

The production truth judge (`claude -p`, Opus) decides, for each reported finding,
whether it is the SAME bug as the advisory's ground truth (same_bug / different_bug).
We have ~193 of those Opus verdicts banked across our DBs. This script replays the
EXACT same judge inputs (advisory + reported finding, via the shared
`build_truth_prompt`) through three candidate "shadow" judges — DeepSeek, MiMo and a
local Gemma-4-31B — and records each one's verdict alongside Opus's, so we can measure
how well a cheap panel reproduces Opus and whether a 3-judge vote could settle the
borderline HIT/MISS calls.

Idempotent + resumable: every (model, input_hash, trial) cell is written once to
nelson-shadow-judge.db. Re-running skips finished cells, so a crash or a kill -9
loses nothing. Source DBs are opened READ-ONLY (mode=ro) — the sacrosanct nelson.db
is never written.

Usage:
    .venv/bin/python shadow_judge.py [--limit N] [--models deepseek,mimo,gemma31b]
                                     [--trials 3] [--out nelson-shadow-judge.db]
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
from nelson.score import ReportedFinding, build_truth_prompt, parse_truth_verdict

# -- Config ------------------------------------------------------------------

# Source DBs that hold Opus truth verdicts. Read-only; baseline first.
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
# (so --models can run a subset without needing every secret present).
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
MAX_TOKENS = 2048  # headroom for reasoning models before the JSON
MAX_RETRIES = 5

_RESULT_DDL = """
CREATE TABLE IF NOT EXISTS shadow_verdicts (
    model           TEXT NOT NULL,
    trial           INTEGER NOT NULL,
    input_hash      TEXT NOT NULL,
    source_db       TEXT,
    case_ext_id     TEXT,
    finding_file    TEXT,
    finding_line    INTEGER,
    opus_same_bug   INTEGER,        -- reference: 1 same / 0 different
    opus_conflict   INTEGER,        -- 1 if Opus gave both labels on identical input
    shadow_same_bug INTEGER,        -- 1/0, NULL if unparseable or errored
    shadow_label    TEXT,
    raw_reply       TEXT,
    latency_ms      INTEGER,
    tokens_out      INTEGER,
    error           TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (model, input_hash, trial)
);
"""

# -- Eval-set loading --------------------------------------------------------


def _load_eval_set() -> list[dict]:
    """Collect every (advisory, finding) input Opus truth-judged, deduped by prompt.

    Returns a list of {input_hash, prompt, opus_same_bug, opus_conflict, source_db,
    case_ext_id, finding_file, finding_line}. Identical prompts across DBs collapse to
    one row; if Opus labelled identical input both ways, opus_conflict=1 (its own
    nondeterminism — kept, but excluded from strict accuracy by the report).
    """
    by_hash: dict[str, dict] = {}
    for db_path in SOURCE_DBS:
        if not Path(db_path).exists():
            continue
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        case_cache: dict[int, Case] = {}
        rows = conn.execute(
            """SELECT rf.*, r.case_id AS _case_id
               FROM run_findings rf JOIN runs r ON rf.run_id = r.id
               WHERE rf.judge_truth_verdict IN ('same_bug', 'different_bug')"""
        ).fetchall()
        for row in rows:
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
            prompt = build_truth_prompt(case, finding)
            h = hashlib.sha256(prompt.encode()).hexdigest()[:16]
            opus_same = 1 if row["judge_truth_verdict"] == "same_bug" else 0
            if h in by_hash:
                if by_hash[h]["opus_same_bug"] != opus_same:
                    by_hash[h]["opus_conflict"] = 1
                continue
            by_hash[h] = {
                "input_hash": h,
                "prompt": prompt,
                "opus_same_bug": opus_same,
                "opus_conflict": 0,
                "source_db": db_path,
                "case_ext_id": case.ext_id,
                "finding_file": finding.file,
                "finding_line": finding.line,
            }
        conn.close()
    return list(by_hash.values())


# -- One judge call ----------------------------------------------------------


def _call(client: httpx.Client, cfg: dict, key: str, prompt: str) -> dict:
    """POST one chat completion; retry transient failures with backoff.

    Returns {shadow_same_bug, shadow_label, raw_reply, latency_ms, tokens_out, error}.
    A parsed-but-unusable reply has shadow_same_bug=None and error set; the verdict
    is never guessed.
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
            parsed = parse_truth_verdict(content)
            if parsed is None:
                return {
                    "shadow_same_bug": None,
                    "shadow_label": None,
                    "raw_reply": content[:2000],
                    "latency_ms": latency_ms,
                    "tokens_out": tokens_out,
                    "error": "unparseable",
                }
            same, _reasoning = parsed
            return {
                "shadow_same_bug": 1 if same else 0,
                "shadow_label": "same_bug" if same else "different_bug",
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
        "shadow_same_bug": None,
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
            "SELECT input_hash, trial FROM shadow_verdicts WHERE model = ?",
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
                """INSERT OR IGNORE INTO shadow_verdicts
                   (model, trial, input_hash, source_db, case_ext_id, finding_file,
                    finding_line, opus_same_bug, opus_conflict, shadow_same_bug,
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
                    item["opus_same_bug"],
                    item["opus_conflict"],
                    res["shadow_same_bug"],
                    res["shadow_label"],
                    res["raw_reply"],
                    res["latency_ms"],
                    res["tokens_out"],
                    res["error"],
                ),
            )
            conn.commit()
            n += 1
            if res["shadow_same_bug"] is not None and not item["opus_conflict"]:
                scored += 1
                if res["shadow_same_bug"] == item["opus_same_bug"]:
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
    ap.add_argument("--out", default="nelson-shadow-judge.db")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    bad = [m for m in models if m not in SHADOW_MODELS]
    if bad:
        print(f"unknown model(s): {bad}; known: {list(SHADOW_MODELS)}")
        return 2

    eval_set = _load_eval_set()
    if args.limit:
        eval_set = eval_set[: args.limit]
    n_same = sum(i["opus_same_bug"] for i in eval_set)
    n_conf = sum(i["opus_conflict"] for i in eval_set)
    print(
        f"eval set: {len(eval_set)} distinct inputs "
        f"(opus same={n_same} diff={len(eval_set) - n_same}, "
        f"self-conflict={n_conf}); models={models}; trials={args.trials}",
        flush=True,
    )

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
