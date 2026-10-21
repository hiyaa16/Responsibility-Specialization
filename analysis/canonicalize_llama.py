"""Canonicalize the Llama-3.1-8B v2 raw run log (remove retry/re-execution duplicates).

The raw file results/v2/llama31_8b/runs.jsonl accumulated 777 records instead of the
intended 750 because some scheduled runs were retried and the retry was appended to the
same JSONL. Each intended run is uniquely identified by (condition, task_id, run_index)
with a fixed seed (0->11,1->22,2->33,3->44,4->55). This script produces exactly one record
per identity group and writes a NEW file runs_clean.jsonl. It NEVER modifies runs.jsonl.

Canonicalization rule (identity / execution-validity only — NEVER an outcome metric):
  1. Group records by (condition, task_id, run_index).
  2. One record  -> keep it.
  3. Duplicates  -> prefer execution VALIDITY, then provenance:
       a. candidates = the non-truncated records if any exist, else all records
          (truncation differs -> keep the valid one; if validity is tied there is no
           validity signal, so every record stays a candidate).
       b. among candidates keep the one with the EARLIEST timestamp_start — the original
          scheduled execution; later re-executions are the retries that get removed.
       c. if candidates tie on timestamp_start -> AMBIGUOUS: do not choose, report it.
  Correctness, latency, token counts and every other outcome are irrelevant to selection.

Groups where validity is tied (all-truncated or all-non-truncated) are resolved purely by
chronological provenance and are additionally surfaced as "flagged_groups" in the audit so
no uncertainty is resolved silently.

Outputs (alongside the raw file, never touching it):
  results/v2/llama31_8b/runs_clean.jsonl
  results/v2/llama31_8b/canonicalization_audit.json
  results/v2/llama31_8b/canonicalization_audit.txt

Run:  python analysis/canonicalize_llama.py
"""
from __future__ import annotations
import os
import io
import json
from collections import defaultdict, Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LLAMA_DIR = os.path.join(ROOT, "results", "v2", "llama31_8b")
RAW = os.path.join(LLAMA_DIR, "runs.jsonl")
CLEAN = os.path.join(LLAMA_DIR, "runs_clean.jsonl")
AUDIT_JSON = os.path.join(LLAMA_DIR, "canonicalization_audit.json")
AUDIT_TXT = os.path.join(LLAMA_DIR, "canonicalization_audit.txt")

CONDS = ["generalist", "specialized", "specialized_length_matched"]
EXPECTED_SEED = {0: 11, 1: 22, 2: 33, 3: 44, 4: 55}
EXPECTED_TOTAL = 750
EXPECTED_PER_CONDITION = 250

RULE_TEXT = (
    "One record per (condition, task_id, run_index). For duplicates: "
    "(a) candidates = non-truncated records if any exist else all records "
    "(prefer a valid/non-truncated execution over a truncated one); "
    "(b) among candidates keep the earliest timestamp_start (the original scheduled run); "
    "later re-executions (retries) are removed. Selection never uses correctness, latency, "
    "tokens or any outcome metric. If candidates tie on timestamp_start the group is reported "
    "as ambiguous and NOT resolved. Validity-tied groups (all-truncated / all-non-truncated) are "
    "resolved by chronological provenance and additionally flagged for transparency."
)


def read_raw(path):
    """Return list of (line_index, raw_line_string, parsed_dict), preserving exact bytes."""
    out = []
    with io.open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            s = line.rstrip("\n")
            if s.strip():
                out.append((i, s, json.loads(s)))
    return out


def canonicalize():
    records = read_raw(RAW)
    original_count = len(records)
    original_condition_counts = dict(Counter(r["condition"] for _, _, r in records))

    groups = defaultdict(list)
    for item in records:
        _, _, r = item
        groups[(r["condition"], r["task_id"], r["run_index"])].append(item)

    kept_lines = set()          # line indices to emit into the clean file
    removed = []                # audit entries for removed records
    ambiguous_groups = []       # genuinely unresolved (timestamp tie) -> halt-worthy
    flagged_groups = []         # resolved-by-provenance but surfaced for review
    n_dup_groups = 0

    for key in sorted(groups):
        cond, task, ri = key
        items = groups[key]
        if len(items) == 1:
            kept_lines.add(items[0][0])
            continue

        n_dup_groups += 1
        nontrunc = [it for it in items if not bool(it[2].get("truncated"))]
        candidates = nontrunc if nontrunc else items
        validity_tied = (len(nontrunc) == 0) or (len(nontrunc) == len(items))

        # order candidates by original execution time
        cand_sorted = sorted(candidates, key=lambda it: (it[2].get("timestamp_start") or "", it[0]))
        earliest_ts = cand_sorted[0][2].get("timestamp_start")
        tied = [it for it in cand_sorted if it[2].get("timestamp_start") == earliest_ts]

        if len(tied) > 1:
            # genuine ambiguity: identical timestamps among candidates -> do NOT choose
            ambiguous_groups.append({
                "condition": cond, "task_id": task, "run_index": ri,
                "reason": "multiple candidate records share the same timestamp_start; "
                          "cannot distinguish original from retry on identity grounds",
                "records": [{"line": it[0], "seed": it[2].get("seed"),
                             "truncated": bool(it[2].get("truncated")),
                             "timestamp_start": it[2].get("timestamp_start"),
                             "timestamp_end": it[2].get("timestamp_end")} for it in items],
            })
            # keep ALL records for this group (do not silently resolve) — flagged downstream
            for it in items:
                kept_lines.add(it[0])
            continue

        keep = cand_sorted[0]
        kept_lines.add(keep[0])

        # everything else in the group is a removed retry/re-execution
        for it in items:
            if it[0] == keep[0]:
                continue
            r = it[2]
            kept_trunc = bool(keep[2].get("truncated"))
            this_trunc = bool(r.get("truncated"))
            if nontrunc and this_trunc and not kept_trunc:
                reason = ("retry removed: this execution is truncated and a non-truncated "
                          "execution exists for the same (condition,task_id,run_index)")
            elif this_trunc and kept_trunc:
                reason = ("retry removed: later re-execution; kept the earliest-timestamp "
                          "original (BOTH executions truncated — validity tied, flagged)")
            else:
                reason = ("retry removed: later re-execution; kept the earliest-timestamp "
                          "original (both executions non-truncated — validity tied)")
            removed.append({
                "condition": cond, "task_id": task, "run_index": ri, "seed": r.get("seed"),
                "timestamp_start": r.get("timestamp_start"), "timestamp_end": r.get("timestamp_end"),
                "truncated": this_trunc, "reason_for_removal": reason,
                "removed_line_index": it[0], "kept_line_index": keep[0],
                "kept_timestamp_start": keep[2].get("timestamp_start"),
            })

        # surface validity-tied groups for human review (resolved by provenance only)
        if validity_tied:
            flagged_groups.append({
                "condition": cond, "task_id": task, "run_index": ri,
                "status": ("all-truncated" if not nontrunc else "all-non-truncated"),
                "resolution": "kept earliest-timestamp original; later re-execution removed "
                              "(decision on provenance only, no outcome metric used)",
                "kept_line_index": keep[0],
                "records": [{"line": it[0], "truncated": bool(it[2].get("truncated")),
                             "timestamp_start": it[2].get("timestamp_start"),
                             "latency_seconds": it[2].get("latency_seconds"),
                             "correct": it[2].get("correct")} for it in cand_sorted],
            })

    # --- write clean file: emit kept raw lines verbatim, in original file order ---
    kept_items = [(i, s) for (i, s, _) in records if i in kept_lines]
    kept_items.sort(key=lambda x: x[0])
    with io.open(CLEAN, "w", encoding="utf-8", newline="\n") as f:
        for _, s in kept_items:
            f.write(s + "\n")

    clean_records = [r for (i, _, r) in records if i in kept_lines]
    clean_condition_counts = dict(Counter(r["condition"] for r in clean_records))

    audit = {
        "canonicalization_rule_used": RULE_TEXT,
        "seed_mapping": {str(k): v for k, v in EXPECTED_SEED.items()},
        "raw_file": os.path.relpath(RAW, ROOT).replace("\\", "/"),
        "clean_file": os.path.relpath(CLEAN, ROOT).replace("\\", "/"),
        "original_record_count": original_count,
        "clean_record_count": len(clean_records),
        "expected_record_count": EXPECTED_TOTAL,
        "original_condition_counts": original_condition_counts,
        "clean_condition_counts": clean_condition_counts,
        "number_of_duplicate_groups": n_dup_groups,
        "number_of_removed_records": len(removed),
        "number_of_ambiguous_groups": len(ambiguous_groups),
        "number_of_flagged_groups": len(flagged_groups),
        "removed_records": removed,
        "ambiguous_groups": ambiguous_groups,
        "flagged_groups": flagged_groups,
    }
    with io.open(AUDIT_JSON, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    _write_text_audit(audit)
    return audit


def _write_text_audit(a):
    L = []
    L.append("=" * 88)
    L.append("LLAMA-3.1-8B v2  —  CANONICALIZATION AUDIT (retry/duplicate removal)")
    L.append("=" * 88)
    L.append(f"raw file   : {a['raw_file']}  (NEVER modified)")
    L.append(f"clean file : {a['clean_file']}")
    L.append("")
    L.append(f"original records : {a['original_record_count']}")
    L.append(f"clean records    : {a['clean_record_count']}   (expected {a['expected_record_count']})")
    L.append(f"removed records  : {a['number_of_removed_records']}")
    L.append(f"duplicate groups : {a['number_of_duplicate_groups']}")
    L.append(f"ambiguous groups : {a['number_of_ambiguous_groups']}  (unresolved — manual review needed)")
    L.append(f"flagged groups   : {a['number_of_flagged_groups']}  (validity-tied, resolved by provenance only)")
    L.append("")
    L.append("condition counts  original -> clean")
    for c in CONDS:
        L.append(f"  {c:28s}: {a['original_condition_counts'].get(c,0):4d} -> {a['clean_condition_counts'].get(c,0):4d}")
    L.append("")
    L.append("RULE USED:")
    L.append("  " + a["canonicalization_rule_used"])
    L.append("")
    L.append("-" * 88)
    L.append("REMOVED RECORDS (retries / re-executions):")
    L.append("-" * 88)
    for r in a["removed_records"]:
        L.append(f"  {r['condition']} / {r['task_id']} / run{r['run_index']} seed={r['seed']} "
                 f"truncated={r['truncated']}")
        L.append(f"      removed start={r['timestamp_start']} end={r['timestamp_end']}")
        L.append(f"      kept    start={r['kept_timestamp_start']}")
        L.append(f"      reason : {r['reason_for_removal']}")
    if a["flagged_groups"]:
        L.append("")
        L.append("-" * 88)
        L.append("FLAGGED GROUPS (validity-tied; resolved on chronological provenance only, NOT outcome):")
        L.append("-" * 88)
        for g in a["flagged_groups"]:
            L.append(f"  {g['condition']} / {g['task_id']} / run{g['run_index']}  [{g['status']}]")
            for rec in g["records"]:
                L.append(f"      line={rec['line']} trunc={rec['truncated']} "
                         f"start={rec['timestamp_start']} lat={rec['latency_seconds']} correct={rec['correct']}")
            L.append(f"      -> {g['resolution']}")
    if a["ambiguous_groups"]:
        L.append("")
        L.append("-" * 88)
        L.append("AMBIGUOUS GROUPS (NOT resolved — require manual review):")
        L.append("-" * 88)
        for g in a["ambiguous_groups"]:
            L.append(f"  {g['condition']} / {g['task_id']} / run{g['run_index']} : {g['reason']}")
    else:
        L.append("")
        L.append("No unresolved ambiguous groups.")
    with io.open(AUDIT_TXT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    a = canonicalize()
    print("\n>>> canonicalization complete:",
          f"{a['original_record_count']} -> {a['clean_record_count']} records,",
          f"{a['number_of_removed_records']} removed,",
          f"{a['number_of_ambiguous_groups']} ambiguous.")
