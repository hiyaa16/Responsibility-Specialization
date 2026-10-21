"""Post-run validation of the Mistral-7B canonical dataset (results/v2/mistral_7b/runs.jsonl).

Run this BEFORE any statistics (task section 11). Prints a PASS/FAIL report and exits non-zero
on any failure. Also confirms the Qwen and Llama canonical datasets are untouched. Because the
Mistral runner is duplicate-proof by construction, this dataset should be canonical (exactly 750)
from the start — no canonicalization step is expected. If a retries.jsonl exists it is reported
for transparency (diverted re-executions), but it is NOT part of the canonical dataset.

Run:  python analysis/validate_mistral.py
"""
from __future__ import annotations
import os
import io
import sys
import json
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MISTRAL_DIR = os.path.join(ROOT, "results", "v2", "mistral_7b")
MISTRAL = os.path.join(MISTRAL_DIR, "runs.jsonl")
RETRIES = os.path.join(MISTRAL_DIR, "retries.jsonl")
QWEN = os.path.join(ROOT, "results", "v2", "runs.jsonl")
LLAMA_CLEAN = os.path.join(ROOT, "results", "v2", "llama31_8b", "runs_clean.jsonl")

CONDS = ["generalist", "specialized", "specialized_length_matched"]
EXPECTED_SEED = {0: 11, 1: 22, 2: 33, 3: 44, 4: 55}


def read_lines(path):
    with io.open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def count_lines(path):
    return len(read_lines(path)) if os.path.exists(path) else None


def main():
    if not os.path.exists(MISTRAL):
        print(f"ERROR: {MISTRAL} not found. Run the experiment first: python src/runner_mistral7b.py")
        sys.exit(2)

    recs = [json.loads(s) for s in read_lines(MISTRAL)]
    checks = []  # (name, ok, detail)

    # 1. total == 750
    checks.append(("total == 750", len(recs) == 750, f"got {len(recs)}"))

    # 2. condition counts == 250 each
    cc = Counter(r["condition"] for r in recs)
    checks.append(("each condition == 250",
                   all(cc.get(c, 0) == 250 for c in CONDS),
                   ", ".join(f"{c}={cc.get(c,0)}" for c in CONDS)))

    # 3. each condition x task has exactly 5 records
    ct = Counter((r["condition"], r["task_id"]) for r in recs)
    bad_ct = {k: v for k, v in ct.items() if v != 5}
    checks.append(("each condition x task == 5 records", not bad_ct,
                   "all 150 pairs == 5" if not bad_ct else f"offenders={dict(list(bad_ct.items())[:5])}"))

    # 4. run_index set per (cond,task) == {0,1,2,3,4}
    ri = defaultdict(set)
    for r in recs:
        ri[(r["condition"], r["task_id"])].add(r["run_index"])
    bad_ri = {k: sorted(v) for k, v in ri.items() if v != {0, 1, 2, 3, 4}}
    checks.append(("run_index per (cond,task) == {0,1,2,3,4}", not bad_ri,
                   "ok" if not bad_ri else f"offenders={dict(list(bad_ri.items())[:5])}"))

    # 5. seeds exactly 11,22,33,44,55 by run_index
    bad_seed = [(r["condition"], r["task_id"], r["run_index"], r["seed"])
                for r in recs if r["seed"] != EXPECTED_SEED.get(r["run_index"])]
    checks.append(("seed mapping 0->11..4->55 consistent", not bad_seed,
                   "all match" if not bad_seed else f"{len(bad_seed)} mismatches: {bad_seed[:5]}"))

    # 6. no duplicate (condition, task_id, run_index)
    keys = Counter((r["condition"], r["task_id"], r["run_index"]) for r in recs)
    dups = {k: v for k, v in keys.items() if v > 1}
    checks.append(("no duplicate (condition,task_id,run_index)", not dups,
                   f"{len(keys)} unique keys" if not dups else f"dups={dict(list(dups.items())[:5])}"))

    # 7. all 50 tasks in all 3 conditions / no missing combo
    tasks_by_cond = {c: set(r["task_id"] for r in recs if r["condition"] == c) for c in CONDS}
    all_tasks = set(r["task_id"] for r in recs)
    task_ok = len(all_tasks) == 50 and all(tasks_by_cond[c] == all_tasks for c in CONDS)
    checks.append(("all 50 tasks in all 3 conditions (no missing combo)", task_ok,
                   f"{len(all_tasks)} tasks; " + ", ".join(f"{c}={len(tasks_by_cond[c])}" for c in CONDS)))

    # 8. model is mistral:7b everywhere
    models = set(r.get("model") for r in recs)
    checks.append(("model == mistral:7b for all records", models == {"mistral:7b"}, f"models={models}"))

    # 9. required execution-metadata fields present on every record
    needed = ["timestamp_start", "timestamp_end", "execution_order", "execution_seed",
              "condition", "task_id", "run_index", "seed", "model", "temperature",
              "agent_count", "handoffs", "input_tokens", "output_tokens", "total_tokens",
              "llm_calls", "tool_calls", "latency_seconds", "correct", "corrections",
              "repeated_work", "failure_types", "truncated", "agent_outputs"]
    missing_field = {}
    for r in recs:
        for k in needed:
            if k not in r:
                missing_field.setdefault(k, 0)
                missing_field[k] += 1
    checks.append(("all execution-metadata/RunRecord fields preserved", not missing_field,
                   "all present" if not missing_field else f"missing={missing_field}"))

    # 10. Qwen + Llama canonical data intact
    qn = count_lines(QWEN)
    checks.append(("Qwen canonical intact (750)", qn == 750, f"results/v2/runs.jsonl = {qn}"))
    ln = count_lines(LLAMA_CLEAN)
    checks.append(("Llama canonical intact (750)", ln == 750, f"llama31_8b/runs_clean.jsonl = {ln}"))

    # ---- report ----
    print("=" * 82)
    print("MISTRAL-7B POST-RUN VALIDATION  —  results/v2/mistral_7b/runs.jsonl")
    print("=" * 82)
    allpass = True
    for name, ok, detail in checks:
        allpass = allpass and ok
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name:50s}  {detail}")
    rn = count_lines(RETRIES)
    if rn:
        print(f"  [note]  retries.jsonl present: {rn} diverted re-execution(s) "
              f"(NOT part of canonical dataset — provenance only)")
    print("-" * 82)
    print(f"  OVERALL: {'PASS' if allpass else 'FAIL — DO NOT run final statistics on this dataset'}")
    print("=" * 82)
    sys.exit(0 if allpass else 1)


if __name__ == "__main__":
    main()
