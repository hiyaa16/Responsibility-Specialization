"""Final three-model report (task sections 18-19). Assembled ONLY from the computed artifacts of
analysis/compare_three_models.py — no number is hand-copied, no verdict is hand-forced. Answers the
7 required questions from the actual replication verdicts and derives (not asserts) a next-step
recommendation A / B / C from the evidence. Writes results/v2/mistral_7b/FINAL_REPORT.txt + prints.

Prereq:  python analysis/compare_three_models.py   (produces results/v2/compare_3_models/summary.json)
Run:     python analysis/final_report_mistral.py
"""
from __future__ import annotations
import os
import io
import sys
import json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SUMMARY = os.path.join(ROOT, "results", "v2", "compare_3_models", "summary.json")
OUT = os.path.join(ROOT, "results", "v2", "mistral_7b", "FINAL_REPORT.txt")

# human labels for internal metric names
LABEL = {"ov_01": "first_handoff_overlap", "latency_seconds": "latency",
         "tool_calls_total": "tool_calls"}


def repl_is(verdict, kind):
    return verdict.upper().startswith(kind)


def main():
    if not os.path.exists(SUMMARY):
        print(f"ERROR: {SUMMARY} not found. Run: python analysis/compare_three_models.py")
        sys.exit(2)
    with io.open(SUMMARY, "r", encoding="utf-8") as f:
        s = json.load(f)

    rep = {r["metric"]: r for r in s["replication"]}
    models = ("Qwen", "Llama", "Mistral")
    nruns = {m: s["models"][m]["n_runs"] for m in models}

    def line_for(metric):
        r = rep[metric]
        return (f"{LABEL.get(metric, metric):20s} "
                + "  ".join(f"{m}: med={r[f'{m}_A-C_med']:+.3f} holm={r[f'{m}_holm']:.3g}" for m in models)
                + f"\n{'':20s} -> {r['verdict']}")

    # coordination vs system-benefit evidence
    coord_metrics = ["ov_01", "corrections"]
    system_metrics = ["latency_seconds", "output_tokens", "total_tokens", "accuracy",
                      "llm_calls", "tool_calls_total"]
    coord_replicated = all(repl_is(rep[m]["verdict"], "REPLICATED") for m in coord_metrics)
    coord_any = any(repl_is(rep[m]["verdict"], "REPLICATED") for m in coord_metrics)
    system_replicated = [m for m in system_metrics if repl_is(rep[m]["verdict"], "REPLICATED")]
    model_dependent = [m for m in rep if repl_is(rep[m]["verdict"], "MODEL-DEPENDENT")]

    # evidence-derived recommendation
    if coord_replicated and not system_replicated:
        rec = ("A. Proceed to DIRECT MECHANISM ISOLATION / controlled-handoff experiment.\n"
               "   Rationale: the coordination signals (first-handoff overlap + corrections) replicate "
               "across all three model families, but they do NOT translate into a model-robust "
               "system-level benefit. The scientifically sharp next move is to isolate the handoff/"
               "coordination mechanism directly and test whether it is causal.")
    elif coord_replicated and system_replicated:
        rec = ("B. Proceed to TOPOLOGY MODERATION experiment.\n"
               "   Rationale: specialization robustly changes coordination AND yields system-level "
               "benefits across models; mapping how topology moderates the effect is the natural "
               "boundary-condition study.")
    else:
        rec = ("C. STOP experimentation; focus on ANALYSIS / PAPER.\n"
               "   Rationale: the coordination signal itself is not robust across model families "
               "(model-dependent). 'Specialization changes coordination but its consequences are "
               "model-dependent' is the supported, publishable finding; more runs will not fix a "
               "model-dependence result.")

    L = []
    L.append("=" * 90)
    L.append("THREE-MODEL FINAL REPORT — responsibility specialization (A vs C, prompt-length-controlled)")
    L.append("=" * 90)
    L.append(f"Models (each canonical 750 runs): " +
             ", ".join(f"{m}={nruns[m]}" for m in models))
    if any(nruns[m] != 750 for m in models):
        L.append("  !! WARNING: a model does not have exactly 750 runs — validate before trusting verdicts.")
    L.append("Primary comparison: A = generalist vs C = specialized_length_matched (controls prompt length).")
    L.append("All verdicts computed from task-paired Wilcoxon + Holm (identical stats across models).")

    L.append("\n" + "-" * 90)
    L.append("SEVEN RESEARCH QUESTIONS")
    L.append("-" * 90)
    L.append("Q1. Does FIRST-HANDOFF OVERLAP replicate across all 3 models?")
    L.append("    " + line_for("ov_01").replace("\n", "\n    "))
    L.append("Q2. Do CORRECTION differences replicate across all 3?")
    L.append("    " + line_for("corrections").replace("\n", "\n    "))
    L.append("Q3. Does the LATENCY effect replicate?")
    L.append("    " + line_for("latency_seconds").replace("\n", "\n    "))
    L.append("Q4. Does OUTPUT-TOKEN behavior replicate?")
    L.append("    " + line_for("output_tokens").replace("\n", "\n    "))
    L.append("Q5. Is specialization's effect MODEL-DEPENDENT?")
    if model_dependent:
        L.append(f"    YES for: {', '.join(LABEL.get(m, m) for m in model_dependent)}")
    else:
        L.append("    No metric was classified MODEL-DEPENDENT.")
    L.append("Q6. Is there evidence for a MODEL-ROBUST COORDINATION mechanism?")
    L.append(f"    coordination metrics (first_handoff_overlap, corrections): "
             f"{'BOTH REPLICATED -> YES, robust' if coord_replicated else ('one replicated -> PARTIAL' if coord_any else 'neither replicated -> NO')}")
    L.append("Q7. Is there evidence the coordination change yields CONSISTENT SYSTEM-LEVEL BENEFITS?")
    if system_replicated:
        L.append(f"    Replicated system metrics: {', '.join(LABEL.get(m, m) for m in system_replicated)}")
    else:
        L.append("    NO system-level metric (latency / tokens / accuracy / calls) replicated across all 3 "
                 "-> no evidence of a consistent system-level benefit.")

    L.append("\n" + "-" * 90)
    L.append("A-C REPLICATION — ALL METRICS")
    L.append("-" * 90)
    for metric in rep:
        L.append(line_for(metric))

    L.append("\n" + "-" * 90)
    L.append("LATENCY TAIL (per model, A/B/C medians + fraction >100s)")
    L.append("-" * 90)
    for m in models:
        t = s[m.lower()]["tail"]
        L.append(f"  {m:8s}: median A={t['A']['median']:.1f}/B={t['B']['median']:.1f}/C={t['C']['median']:.1f}s | "
                 f">100s A={t['A']['frac_gt_100s']:.1%}/B={t['B']['frac_gt_100s']:.1%}/C={t['C']['frac_gt_100s']:.1%} | "
                 f"A-C shift: {t['shift_AC']['label']}")

    L.append("\n" + "-" * 90)
    L.append("FINAL RESEARCH DECISION (evidence-derived, section 19)")
    L.append("-" * 90)
    L.append(rec)
    L.append("\nPRINCIPLE: the best result is not necessarily a positive one. Verdicts follow the data; "
             "no replication was forced, no outcome cherry-picked, no raw data overwritten.")

    report = "\n".join(L)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
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
