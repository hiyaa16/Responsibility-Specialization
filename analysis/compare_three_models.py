"""Three-model cross-replication: Qwen-2.5-7B vs Llama-3.1-8B vs Mistral-7B (v2 study).

Reuses the EXACT statistical machinery of analysis/mechanism_v2.py via analysis/compare_models.py
(same task-level paired Wilcoxon, same Holm correction across the 3 A/B/C pairs per metric, same
matched-pairs rank-biserial + bootstrap CI, same aggregation). It does NOT re-run or modify any
experiment and does NOT change the methodology. The primary test is A (generalist) vs
C (specialized_length_matched); A-C is reported for every metric.

Canonical inputs (each must be exactly 750 runs):
  Qwen   : results/v2/runs.jsonl
  Llama  : results/v2/llama31_8b/runs_clean.jsonl   (de-duplicated canonical set)
  Mistral: results/v2/mistral_7b/runs.jsonl
Outputs -> results/v2/compare_3_models/ :
  three_model_table.csv, replication.csv, per_model_metrics.csv, summary.json, compare_report.txt

Run:  python analysis/compare_three_models.py
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import mechanism_v2 as mech          # noqa: E402  (same stats as both prior analyses)
import compare_models as cm          # noqa: E402  (reuse model_stats + metric lists, unchanged)

    ROOT = mech.ROOT
    CONDS = mech.CONDS
    SHORT = mech.SHORT
    TABLE_METRICS = cm.TABLE_METRICS
    PAIRED_METRICS = cm.PAIRED_METRICS

    MODELS = [
        ("Qwen",    os.path.join(ROOT, "results", "v2"),                      "runs.jsonl"),
        ("Llama",   os.path.join(ROOT, "results", "v2", "llama31_8b"),        "runs_clean.jsonl"),
        ("Mistral", os.path.join(ROOT, "results", "v2", "mistral_7b"),        "runs.jsonl"),
    ]


    def _sign(x):
        return 0 if x == 0 else (1 if x > 0 else -1)


    def verdict3(ac, alpha=0.05):
        """3-model A-C replication classification from each model's paired A-C result.
        ac = {'Qwen':{median_diff,p_holm}, 'Llama':..., 'Mistral':...}. Qwen is the reference study.
        Definitions follow task section 15 (REPLICATED / PARTIALLY REPLICATED / NOT REPLICATED /
        MODEL-DEPENDENT). Uses ACTUAL stats — direction similarity alone is never enough."""
        q, l, m = ac["Qwen"], ac["Llama"], ac["Mistral"]
        sig = {k: v["p_holm"] < alpha for k, v in ac.items()}
        dirs = {k: _sign(v["median_diff"]) for k, v in ac.items()}

        if not sig["Qwen"]:
            any_sig = [k for k in ac if sig[k]]
            if not any_sig:
                return "N/A (no significant A-C effect in any model)"
            return "MODEL-DEPENDENT (significant in %s but not in the Qwen reference)" % ",".join(any_sig)

        qd = dirs["Qwen"]
        others = ["Llama", "Mistral"]
        all_sig_same = all(sig[k] and dirs[k] == qd and qd != 0 for k in others)
        any_opp_sig = any(sig[k] and dirs[k] != 0 and dirs[k] != qd for k in others)
        all_same_dir = all(dirs[k] == qd for k in others) and qd != 0

        if all_sig_same:
            return "REPLICATED (all 3 models: same direction, Holm-significant)"
        if any_opp_sig:
            return "MODEL-DEPENDENT (significant but direction reverses across models)"
        if all_same_dir:
            return "PARTIALLY REPLICATED (same direction, not Holm-significant in all 3)"
        return "NOT REPLICATED (effect does not reproduce across models)"


def table_val(stats, disp, col, short):
    lat_agg = {"median_latency_s": "median", "mean_latency_s": "mean", "p95_latency_s": "p95"}
    if "latency" in disp:
        return stats["tail"][short][lat_agg[disp]]
    return stats["per_condition"][col][short]["mean"]


def build_three_table(S):
    rows = []
    for disp, col in TABLE_METRICS:
        pk = "latency_seconds" if "latency" in disp else col
        ac = {mdl: S[mdl]["paired"][pk]["A-C"] for mdl in S}
        # Latency inference is based on paired task-level mean latency. Median/mean/p95
        # rows are distinct descriptives; do not duplicate one test across all three rows.
        inferential_latency_row = disp == "median_latency_s"
        show_test = ("latency" not in disp) or inferential_latency_row
        row = {"metric": disp}
        for mdl in ("Qwen", "Llama", "Mistral"):
            st = S[mdl]
            row[f"{mdl}_A"] = round(table_val(st, disp, col, "A"), 4)
            row[f"{mdl}_C"] = round(table_val(st, disp, col, "C"), 4)
            row[f"{mdl}_A-C_med"] = round(ac[mdl]["median_diff"], 4) if show_test else np.nan
            row[f"{mdl}_holm"] = ac[mdl]["p_holm"] if show_test else np.nan
        row["A-C_replication"] = verdict3(ac) if show_test else "DESCRIPTIVE ONLY; no separate test"
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    # existence / count guard
    stats = {}
    dfs = {}
    for label, d, fn in MODELS:
        path = os.path.join(d, fn)
        if not os.path.exists(path):
            print(f"ERROR: {label} canonical runs not found at {path}")
            if label == "Mistral":
                print("       Run the Mistral experiment first: python src/runner_mistral7b.py")
                print("       then validate it: python analysis/validate_mistral.py")
            return
    for label, d, fn in MODELS:
        st, df = cm.model_stats(d, fn)
        stats[label] = st
        dfs[label] = df

    outdir = os.path.join(ROOT, "results", "v2", "compare_3_models")
    os.makedirs(outdir, exist_ok=True)

    # headline table
    table = build_three_table(stats)
    table.to_csv(os.path.join(outdir, "three_model_table.csv"), index=False)

    # replication across ALL paired metrics (A-C)
    rep_rows = []
    for name, col in PAIRED_METRICS:
        ac = {mdl: stats[mdl]["paired"][name]["A-C"] for mdl in stats}
        row = {"metric": name}
        for mdl in ("Qwen", "Llama", "Mistral"):
            row[f"{mdl}_A-C_med"] = round(ac[mdl]["median_diff"], 4)
            row[f"{mdl}_holm"] = ac[mdl]["p_holm"]
            row[f"{mdl}_rb"] = round(ac[mdl]["rank_biserial"], 3)
        row["verdict"] = verdict3(ac)
        rep_rows.append(row)
    rep = pd.DataFrame(rep_rows)
    rep.to_csv(os.path.join(outdir, "replication.csv"), index=False)

    # per-model metric dump (per-condition mean+median + A/B/C paired full stats)
    pm_rows = []
    for mdl in ("Qwen", "Llama", "Mistral"):
        st = stats[mdl]
        for name, col in PAIRED_METRICS:
            for c in CONDS:
                pm_rows.append({"model": mdl, "metric": name, "condition": SHORT[c],
                                "mean": st["per_condition"][name][SHORT[c]]["mean"],
                                "median": st["per_condition"][name][SHORT[c]]["median"]})
            for pair in ["A-B", "B-C", "A-C"]:
                r = st["paired"][name][pair]
                pm_rows.append({"model": mdl, "metric": name, "condition": pair,
                                "mean": r["mean_diff"], "median": r["median_diff"],
                                "rank_biserial": r["rank_biserial"], "ci95_lo": r["ci95_lo"],
                                "ci95_hi": r["ci95_hi"], "p_raw": r["p_raw"], "p_holm": r["p_holm"]})
    pd.DataFrame(pm_rows).to_csv(os.path.join(outdir, "per_model_metrics.csv"), index=False)

    summary = {"models": {m: {"n_runs": stats[m]["n_runs"], "run_dir": stats[m]["run_dir"]} for m in stats},
               "qwen": stats["Qwen"], "llama": stats["Llama"], "mistral": stats["Mistral"],
               "three_model_table": table.to_dict("records"),
               "replication": rep.to_dict("records")}
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)

    # ---- console / text report ----
    L = []
    L.append("=" * 100)
    L.append("THREE-MODEL CROSS-REPLICATION  —  Qwen-2.5-7B  vs  Llama-3.1-8B  vs  Mistral-7B  (v2)")
    L.append("=" * 100)
    for m in ("Qwen", "Llama", "Mistral"):
        L.append(f"{m:8s} runs = {stats[m]['n_runs']}   ({os.path.join(stats[m]['run_dir'])})")
    L.append("Statistics: task-level paired Wilcoxon, Holm across the 3 pairs/metric; matched-pairs "
             "rank-biserial; 95% bootstrap CI (identical to mechanism_v2 / compare_models).")
    L.append("Primary test = A (generalist) vs C (specialized_length_matched).")
    L.append("\nTHREE-MODEL A-C TABLE")
    L.append(table.to_string(index=False))
    L.append("\nA-C REPLICATION VERDICTS (all paired metrics):")
    for r in rep_rows:
        L.append(f"  {r['metric']:18s}  "
                 f"Q med={r['Qwen_A-C_med']:+.3f}/h={r['Qwen_holm']:.3g}  "
                 f"L med={r['Llama_A-C_med']:+.3f}/h={r['Llama_holm']:.3g}  "
                 f"M med={r['Mistral_A-C_med']:+.3f}/h={r['Mistral_holm']:.3g}  ->  {r['verdict']}")
    L.append("\nLATENCY TAIL (per model):")
    for m in ("Qwen", "Llama", "Mistral"):
        t = stats[m]["tail"]
        L.append(f"  {m:8s}: median A={t['A']['median']:.1f}/B={t['B']['median']:.1f}/C={t['C']['median']:.1f}s  "
                 f">100s A={t['A']['frac_gt_100s']:.1%}/B={t['B']['frac_gt_100s']:.1%}/C={t['C']['frac_gt_100s']:.1%}  "
                 f"shift A-C: {t['shift_AC']['label']}")
    L.append("\nINTERPRETATION RULES: verdicts are computed from the tests above (not from directional "
             "resemblance). REPLICATED => Holm-significant and same direction on all 3 models; PARTIALLY "
             "REPLICATED => same direction, not significant everywhere; MODEL-DEPENDENT => direction reverses "
             "or effect present only in some models; NOT REPLICATED => fails to reproduce. Associations only; "
             "no causal reduction claim.")
    L.append(f"\nOutputs -> {outdir}/")
    report = "\n".join(L)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    with open(os.path.join(outdir, "compare_report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
