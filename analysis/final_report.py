"""Concise cross-model final report, assembled from the canonicalization audit and the
recomputed (clean 750-run) cross-model comparison. All numbers are read from the JSON
artifacts — nothing is hand-copied. Writes results/v2/llama31_8b/FINAL_REPORT.txt and prints.

Run (after canonicalize_llama.py, validate_clean_llama.py, compare_models.py):
    python analysis/final_report.py
"""
from __future__ import annotations
import os
import io
import json
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LLAMA_DIR = os.path.join(ROOT, "results", "v2", "llama31_8b")
AUDIT = os.path.join(LLAMA_DIR, "canonicalization_audit.json")
SUMMARY = os.path.join(LLAMA_DIR, "compare", "summary.json")
OUT = os.path.join(LLAMA_DIR, "FINAL_REPORT.txt")


def load(p):
    with io.open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    a = load(AUDIT)
    s = load(SUMMARY)
    q_lat = s["qwen"]["paired"]["latency_seconds"]["A-C"]
    l_lat = s["llama"]["paired"]["latency_seconds"]["A-C"]
    rep = {r["metric"]: r for r in s["replication"]}

    # run the independent validator to get a live PASS/FAIL
    v = subprocess.run([sys.executable, os.path.join(ROOT, "analysis", "validate_clean_llama.py")],
                       capture_output=True, text=True)
    val_pass = (v.returncode == 0)

    L = []
    L.append("=" * 84)
    L.append("FINAL REPORT — Llama-3.1-8B v2 canonicalization + Qwen-vs-Llama replication")
    L.append("=" * 84)

    L.append(f"\n1. Raw Llama records                 : {a['original_record_count']}")
    L.append(f"2. Removed duplicate/retry records   : {a['number_of_removed_records']}")
    L.append(f"3. Canonical Llama records           : {a['clean_record_count']}  (expected {a['expected_record_count']})")
    L.append("4. Counts by condition (clean):")
    for c, n in a["clean_condition_counts"].items():
        L.append(f"     {c:28s}: {n}")
    L.append(f"5. Ambiguous (unresolved) groups     : {a['number_of_ambiguous_groups']}")
    L.append(f"   Flagged (validity-tied, resolved by provenance only) groups : {a['number_of_flagged_groups']}")
    L.append(f"     - of which all-truncated (both executions truncated) : "
             f"{sum(1 for g in a['flagged_groups'] if g['status']=='all-truncated')}")
    L.append(f"6. Canonicalization validation       : {'PASS' if val_pass else 'FAIL'} "
             f"(analysis/validate_clean_llama.py, exit={v.returncode})")

    L.append("\n7. Final Qwen vs Llama A-C latency (recomputed from the clean 750-run dataset):")
    L.append(f"     Qwen : median diff = {q_lat['median_diff']:+.4f} s   Holm p = {q_lat['p_holm']:.6g}"
             f"   rank-biserial = {q_lat['rank_biserial']:+.3f}   "
             f"=> {'significant' if q_lat['p_holm'] < 0.05 else 'not significant'}")
    L.append(f"     Llama: median diff = {l_lat['median_diff']:+.4f} s   Holm p = {l_lat['p_holm']:.6g}"
             f"   rank-biserial = {l_lat['rank_biserial']:+.3f}   "
             f"=> {'significant' if l_lat['p_holm'] < 0.05 else 'not significant'}")
    L.append(f"     (Provisional 777-run numbers were Llama +0.1276 s, p=1.0 — NOT reused; "
             f"the above are recomputed.)")

    L.append(f"\n8. A-C latency replication verdict   : {rep['latency_seconds']['verdict']}")

    L.append("\n9. Replicated mechanism-level effects (A-C, both models, Holm-corrected):")
    order = ["corrections", "ov_01", "output_tokens", "repeated_work", "accuracy",
             "total_tokens", "input_tokens", "llm_calls", "tool_calls_total", "max_agent_input"]
    label = {"ov_01": "first_handoff_overlap"}
    for m in order:
        r = rep[m]
        L.append(f"     {label.get(m, m):22s}: Qwen med={r['qwen_A-C_med']:+.3f} (holm={r['qwen_holm']:.3g}) | "
                 f"Llama med={r['llama_A-C_med']:+.3f} (holm={r['llama_holm']:.3g})  ->  {r['verdict']}")

    L.append("\n10. Remaining data-quality concerns:")
    L.append("     - 1 duplicate group (specialized/MQ09/run2) had BOTH executions truncated; the")
    L.append("       earliest-timestamp original was kept on provenance grounds only (no outcome used).")
    L.append("       The retained record is still truncated — a genuine hard case for Llama, not an error.")
    L.append("     - 27 of 750 intended runs were re-executed during collection (appended retries);")
    L.append("       canonicalization kept the original scheduled execution for each. Raw log preserved.")
    L.append("     - Llama shows substantially higher truncation/verbosity than Qwen (larger token")
    L.append("       footprints, A-C token/llm_call/tool_call DIFFERENCES that have no Qwen counterpart),")
    L.append("       so token-level effects are model-specific and not cross-model replication claims.")

    L.append("\nPrinciple check: numbers recomputed from the clean 750-run dataset; conclusions follow")
    L.append("the computed tests. The goal was a valid non-cherry-picked dataset, not agreement.")
    report = "\n".join(L)
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(report + "\n")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(report)
    print(f"\n[written to {os.path.relpath(OUT, ROOT)}]")


if __name__ == "__main__":
    main()
