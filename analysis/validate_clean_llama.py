"""Validate the canonical Llama clean dataset (results/v2/llama31_8b/runs_clean.jsonl).

Independent re-check of the canonicalization output. Prints a PASS/FAIL report. Also confirms
no record was invented (every clean record is byte-identical to a line in the raw file) and
that the Qwen raw dataset is untouched (still 750 records). Exits non-zero on any FAIL.

Run:  python analysis/validate_clean_llama.py
"""
from __future__ import annotations
import os
import io
import sys
import json
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LLAMA_DIR = os.path.join(ROOT, "results", "v2", "llama31_8b")
RAW = os.path.join(LLAMA_DIR, "runs.jsonl")
CLEAN = os.path.join(LLAMA_DIR, "runs_clean.jsonl")
QWEN = os.path.join(ROOT, "results", "v2", "runs.jsonl")

CONDS = ["generalist", "specialized", "specialized_length_matched"]
EXPECTED_SEED = {0: 11, 1: 22, 2: 33, 3: 44, 4: 55}


def read_lines(path):
    with io.open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def main():
    checks = []  # (name, passed, detail)

    raw_lines = read_lines(RAW)
    clean_lines = read_lines(CLEAN)
    clean = [json.loads(s) for s in clean_lines]

    # 1. total == 750
    checks.append(("total == 750", len(clean) == 750, f"got {len(clean)}"))

    # 2. each condition == 250
    cc = Counter(r["condition"] for r in clean)
    checks.append(("each condition == 250",
                   all(cc.get(c, 0) == 250 for c in CONDS),
                   ", ".join(f"{c}={cc.get(c,0)}" for c in CONDS)))

    # 3. each condition x task has exactly 5 records
    ct = Counter((r["condition"], r["task_id"]) for r in clean)
    bad_ct = {k: v for k, v in ct.items() if v != 5}
    checks.append(("each condition x task has 5 records", not bad_ct,
                   "all 150 pairs == 5" if not bad_ct else f"offenders={bad_ct}"))

    # 4. run_index set per (cond,task) is exactly {0,1,2,3,4}
    ri = defaultdict(set)
    for r in clean:
        ri[(r["condition"], r["task_id"])].add(r["run_index"])
    bad_ri = {k: sorted(v) for k, v in ri.items() if v != {0, 1, 2, 3, 4}}
    checks.append(("run_index per (cond,task) == {0,1,2,3,4}", not bad_ri,
                   "ok" if not bad_ri else f"offenders={bad_ri}"))

    # 5. no duplicate (condition, task_id, run_index)
    keys = Counter((r["condition"], r["task_id"], r["run_index"]) for r in clean)
    dups = {k: v for k, v in keys.items() if v > 1}
    checks.append(("no duplicate (condition,task_id,run_index)", not dups,
                   f"{len(keys)} unique keys" if not dups else f"dups={dups}"))

    # 6. all 50 tasks exist in all 3 conditions
    tasks_by_cond = {c: set(r["task_id"] for r in clean if r["condition"] == c) for c in CONDS}
    all_tasks = set(r["task_id"] for r in clean)
    task_ok = len(all_tasks) == 50 and all(tasks_by_cond[c] == all_tasks for c in CONDS)
    checks.append(("all 50 tasks in all 3 conditions", task_ok,
                   f"{len(all_tasks)} tasks; " + ", ".join(f"{c}={len(tasks_by_cond[c])}" for c in CONDS)))

    # 7. seed mapping consistent
    bad_seed = [(r["condition"], r["task_id"], r["run_index"], r["seed"])
                for r in clean if r["seed"] != EXPECTED_SEED[r["run_index"]]]
    checks.append(("seed mapping run_index->seed consistent", not bad_seed,
                   "all match 0->11,1->22,2->33,3->44,4->55" if not bad_seed else f"{len(bad_seed)} mismatches: {bad_seed[:5]}"))

    # 8. no records invented: every clean line is byte-identical to some raw line
    raw_set = set(raw_lines)
    invented = [s for s in clean_lines if s not in raw_set]
    checks.append(("no invented records (clean subset of raw, byte-identical)", not invented,
                   f"all {len(clean_lines)} clean lines present verbatim in raw"
                   if not invented else f"{len(invented)} lines not found in raw"))

    # 8b. clean is a strict subset of raw by count (exactly 27 fewer)
    checks.append(("raw(777) - clean(750) == 27 removed", len(raw_lines) - len(clean_lines) == 27,
                   f"raw={len(raw_lines)} clean={len(clean_lines)} removed={len(raw_lines)-len(clean_lines)}"))

    # 9. Qwen raw data unchanged (still exactly 750 records)
    qwen_lines = read_lines(QWEN)
    checks.append(("Qwen raw data intact (750 records)", len(qwen_lines) == 750,
                   f"qwen runs.jsonl = {len(qwen_lines)}"))

    # ---- report ----
    print("=" * 80)
    print("CLEAN LLAMA DATASET VALIDATION  —  results/v2/llama31_8b/runs_clean.jsonl")
    print("=" * 80)
    allpass = True
    for name, ok, detail in checks:
        allpass = allpass and ok
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name:52s}  {detail}")
    print("-" * 80)
    print(f"  OVERALL: {'PASS' if allpass else 'FAIL'}")
    print("=" * 80)
    sys.exit(0 if allpass else 1)


if __name__ == "__main__":
    main()
