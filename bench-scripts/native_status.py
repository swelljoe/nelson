#!/usr/bin/env python3
"""Live progress + throughput + ETA for the active bench-loop run.

    python bench-scripts/native_status.py [DB] [REPEAT]

For every ACTIVE competitor it reports, per model: runs completed / running /
failed, mean+median seconds per completed run, throughput (runs/hr), the target
cell count (eligible cases x REPEAT), percent done, and a per-model ETA. Because
each competitor runs its cells one-at-a-time (worker-per-competitor), the overall
ETA is the slowest model's ETA. All numbers come straight from the DB, so it is
safe to run repeatedly while the loop is going.
"""

import json
import sqlite3
import sys
from statistics import median

DB = sys.argv[1] if len(sys.argv) > 1 else "nelson.db"
REPEAT = int(sys.argv[2]) if len(sys.argv) > 2 else 3
TERMINAL_FAIL = ("infra_error", "auth_failed", "error", "refused", "judge_error")


def fmt_dur(secs):
    if secs is None:
        return "  —  "
    secs = int(secs)
    h, m = secs // 3600, (secs % 3600) // 60
    return f"{h}h{m:02d}m" if h else f"{m}m{secs % 60:02d}s"


c = sqlite3.connect(DB)
c.row_factory = sqlite3.Row
active = list(
    c.execute(
        "SELECT id, name, knowledge_cutoff FROM competitors WHERE status='active'"
    )
)

overall_eta = 0.0
rows = []
for comp in active:
    cutoff = (comp["knowledge_cutoff"] or "0000-00")[:7]
    # The harness is FILE-scoped: one cell per (case, ground-truth file, trial).
    # So the target is (sum of gt_files over recency-eligible vetted cases) * REPEAT,
    # NOT (cases * REPEAT) — a multi-file case (e.g. wolfSSL, 5 files) is 5x the work.
    target_files = 0
    for cs in c.execute(
        "SELECT gt_files FROM cases WHERE status='vetted' "
        "AND substr(disclosure_date,1,7) >= ?",
        (cutoff,),
    ):
        target_files += len(json.loads(cs["gt_files"] or "[]"))
    target = target_files * REPEAT

    runs = list(
        c.execute(
            "SELECT status, wall_clock_s, started_at, completed_at "
            "FROM runs WHERE competitor_id=?",
            (comp["id"],),
        )
    )
    done = [r for r in runs if r["status"] == "complete"]
    running = sum(1 for r in runs if r["status"] == "running")
    failed = sum(1 for r in runs if r["status"] in TERMINAL_FAIL)
    times = [r["wall_clock_s"] for r in done if r["wall_clock_s"]]
    mean_s = sum(times) / len(times) if times else None
    med_s = median(times) if times else None

    # ETA: each competitor runs its cells one-at-a-time, so remaining * mean seconds.
    remaining = max(target - len(done), 0)
    eta_s = remaining * mean_s if mean_s else None
    if eta_s:
        overall_eta = max(overall_eta, eta_s)
    pct = 100 * len(done) / target if target else 0
    rows.append(
        (comp["name"], len(done), running, failed, target, pct, mean_s, med_s, eta_s)
    )


print(f"\n  DB={DB}  repeat={REPEAT}   ({len(active)} active competitors)\n")
hdr = f"  {'competitor':32s} {'done':>4}/{'tgt':<4} {'%':>4}  {'run':>3} {'fail':>4}  {'mean':>6} {'med':>6}  {'ETA':>7}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for name, done, running, failed, target, pct, mean_s, med_s, eta_s in rows:
    print(
        f"  {name:32s} {done:>4}/{target:<4} {pct:>3.0f}%  {running:>3} {failed:>4}  "
        f"{fmt_dur(mean_s):>6} {fmt_dur(med_s):>6}  {fmt_dur(eta_s):>7}"
    )
print()
print(f"  overall ETA (slowest model, if it keeps pace): ~{fmt_dur(overall_eta)}")
tot_done = sum(r[1] for r in rows)
tot_tgt = sum(r[4] for r in rows)
print(f"  total: {tot_done}/{tot_tgt} cells done ({100 * tot_done / tot_tgt:.0f}%)\n")
