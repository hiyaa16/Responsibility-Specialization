"""Cross-model replication analysis: Qwen-2.5-7B vs Llama-3.1-8B for the v2 study.

Reuses the EXACT statistical machinery of analysis/mechanism_v2.py (same task-level pairing,
same Wilcoxon signed-rank, same Holm correction across the 3 pairs within a metric, same
rank-biserial effect size + bootstrap CI, same Spearman, same tail/shift definitions) so the
two models are analyzed identically. It does not re-run or modify either experiment.

Per model (Qwen, Llama) it computes the 17 requested quantities per condition and the
A-B / B-C / A-C paired comparisons, then builds the cross-model table and a computed
replication verdict for the key A-vs-C comparison.

Inputs (defaults):
  Qwen : results/v2/runs.jsonl                     (the existing frozen run)
  Llama: results/v2/llama31_8b/runs_clean.jsonl    (canonical 750-run de-duplicated dataset
         from analysis/canonicalize_llama.py; falls back to raw runs.jsonl only if absent)
Outputs -> results/v2/llama31_8b/compare/ : cross_model_table.csv, per_model_metrics.csv,
           replication.csv, summary.json, compare_report.txt

Run:  python analysis/compare_models.py
      python analysis/compare_models.py <qwen_dir> <llama_dir>
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import mechanism_v2 as mech   # noqa: E402  (reuse its exact functions/constants)

ROOT = mech.ROOT
CONDS = mech.CONDS
SHORT = mech.SHORT
PAIRS = mech.PAIRS

# metrics that go in the headline cross-model table (display name -> run-level df column)
TABLE_METRICS = [
    ("accuracy", "accuracy"),
    ("median_latency_s", "latency_seconds"),
    ("mean_latency_s", "latency_seconds"),
    ("p95_latency_s", "latency_seconds"),
    ("total_tokens", "total_tokens"),
    ("input_tokens", "input_tokens"),
    ("output_tokens", "output_tokens"),
    ("llm_calls", "llm_calls"),
    ("tool_calls", "tool_calls_total"),
    ("repeated_work", "repeated_work"),
    ("corrections", "corrections"),
    ("first_handoff_overlap", "ov_01"),
]
# the full set of paired metrics we run A/B/C comparisons on (superset of the table)
PAIRED_METRICS = [
    ("accuracy", "correct"), ("latency_seconds", "latency_seconds"),
    ("total_tokens", "total_tokens"), ("input_tokens", "input_tokens"),
    ("output_tokens", "output_tokens"), ("llm_calls", "llm_calls"),
    ("tool_calls_total", "tool_calls_total"), ("repeated_work", "repeated_work"),
    ("corrections", "corrections"), ("ov_01", "ov_01"),
    ("max_agent_input", "max_agent_input"),
]


def central(df, col, how):
    v = df[col].values
    if how == "mean":
        return float(np.mean(v))
    if how == "median":
        return float(np.median(v))
    if how == "p95":
        return float(np.percentile(v, 95))
    raise ValueError(how)


def paired_block(df, col):
    """A/B/C paired task-level Wilcoxon + Holm(3 pairs) for one metric — identical to mechanism_v2."""
    piv = mech.task_pivot(df, col)
    res = [mech.paired_task_test(piv, a, b) for a, b in PAIRS]
    adj = mech.holm([r["p_raw"] for r in res])
    for r, pa in zip(res, adj):
        r["p_holm"] = float(pa)
    return {r["pair"]: r for r in res}


def model_stats(run_dir, filename="runs.jsonl"):
    runs = mech.load_runs(run_dir, filename)
    df = mech.run_level_df(runs)
    adf = mech.agent_level_df(runs)
    out = {"n_runs": len(runs), "run_dir": run_dir,
           "per_condition": {}, "paired": {}, "tail": {}, "failures": {},
           "latency_corr": {}, "agent_workload": {}}

    # per-condition descriptives for every paired metric (mean + median), plus accuracy
    for name, col in PAIRED_METRICS:
        out["per_condition"][name] = {
            SHORT[c]: {"mean": central(df[df.condition == c], col, "mean"),
                       "median": central(df[df.condition == c], col, "median")}
            for c in CONDS}
        out["paired"][name] = paired_block(df, col)

    # latency tail + shift classification (reuse mechanism_v2 definitions)
    tail = {}
    for c in CONDS:
        v = df[df.condition == c]["latency_seconds"].values
        tail[SHORT[c]] = {"median": float(np.median(v)), "mean": float(np.mean(v)),
                          "p90": float(np.percentile(v, 90)), "p95": float(np.percentile(v, 95)),
                          "p99": float(np.percentile(v, 99)), "max": float(v.max()),
                          "frac_gt_60s": float(np.mean(v > 60)), "frac_gt_100s": float(np.mean(v > 100)),
                          "trimmed10_mean": float(mech.stats.trim_mean(v, proportiontocut=0.10))}
    tail["shift_AB"] = mech.classify_shift(tail["A"], tail["B"])
    tail["shift_AC"] = mech.classify_shift(tail["A"], tail["C"])
    tail["shift_BC"] = mech.classify_shift(tail["B"], tail["C"])
    out["tail"] = tail

    # failure types F1..F9 per condition (count + pct of 250)
    from collections import Counter
    for c in CONDS:
        sub = [r for r in runs if r["condition"] == c]
        cnt = Counter(f for r in sub for f in (r.get("failure_types") or []))
        n = len(sub)
        out["failures"][SHORT[c]] = {f"F{i}": {"count": cnt.get(f"F{i}", 0),
                                               "pct": round(100 * cnt.get(f"F{i}", 0) / n, 2) if n else 0.0}
                                     for i in range(1, 10)}

    # latency <-> token correlations (pooled + within condition), Spearman
    for col in ["total_tokens", "input_tokens", "output_tokens"]:
        rho, p, ci, n = mech.spearman_ci(df[col], df["latency_seconds"])
        within = {}
        for c in CONDS:
            s = df[df.condition == c]
            rr, pp, cc, nn = mech.spearman_ci(s[col], s["latency_seconds"])
            within[SHORT[c]] = rr
        out["latency_corr"][col] = {"pooled_rho": rho, "pooled_p": p, "pooled_ci": list(ci),
                                    "within": within}

    # per-agent accumulated input/output workload (means) + max per-agent input
    out["agent_workload"] = {
        SHORT[c]: {str(i): {"input": float(adf[(adf.condition == c) & (adf.agent_index == i)]["input_tokens"].mean()),
                            "output": float(adf[(adf.condition == c) & (adf.agent_index == i)]["output_tokens"].mean())}
                   for i in range(3)}
        for c in CONDS}
    out["max_agent_input_mean"] = {SHORT[c]: float(df[df.condition == c]["max_agent_input"].mean()) for c in CONDS}
    return out, df


def replication_verdict(q_ac, l_ac, alpha=0.05):
    """Classify replication of the A-vs-C effect from the computed paired results of both models.
    q_ac / l_ac are the 'A-C' entries (median_diff + p_holm). Verdict is DERIVED, not assumed."""
    q_sig = q_ac["p_holm"] < alpha
    l_sig = l_ac["p_holm"] < alpha
    qd, ld = q_ac["median_diff"], l_ac["median_diff"]
    same_dir = (qd == 0 and ld == 0) or (np.sign(qd) == np.sign(ld) and qd != 0 and ld != 0)
    if not q_sig:
        return "N/A (no significant Qwen A-C effect to replicate)"
    if q_sig and l_sig and same_dir:
        return "REPLICATED"
    if q_sig and l_sig and not same_dir:
        return "MODEL-DEPENDENT (significant but direction reversed)"
    if q_sig and not l_sig and same_dir:
        return "PARTIALLY REPLICATED (same direction, not Holm-significant on Llama)"
    return "NOT REPLICATED (Llama n.s. and opposite direction)"


def build_table(q, l):
    # latency rows read from the tail block with a specific aggregator; everything else is a
    # per-condition mean. The paired A-C test always uses the underlying run-level column.
    lat_agg = {"median_latency_s": "median", "mean_latency_s": "mean", "p95_latency_s": "p95"}
    rows = []
    for disp, col in TABLE_METRICS:
        pk = "latency_seconds" if "latency" in disp else col

        def val(stats, short):
            if "latency" in disp:
                return stats["tail"][short][lat_agg[disp]]
            return stats["per_condition"][col][short]["mean"]

        q_ac = q["paired"][pk]["A-C"]; l_ac = l["paired"][pk]["A-C"]
        # The paired latency test is on task-level mean latency, not separate median/mean/p95
        # estimands. Show the inferential result only once to avoid implying three separate tests.
        inferential_latency_row = disp == "median_latency_s"
        show_test = ("latency" not in disp) or inferential_latency_row
        rows.append({
            "metric": disp,
            "Qwen_A": round(val(q, "A"), 4), "Qwen_C": round(val(q, "C"), 4),
            "Qwen_A-C_meddiff": round(q_ac["median_diff"], 4) if show_test else np.nan,
            "Qwen_A-C_holm": q_ac["p_holm"] if show_test else np.nan,
            "Llama_A": round(val(l, "A"), 4), "Llama_C": round(val(l, "C"), 4),
            "Llama_A-C_meddiff": round(l_ac["median_diff"], 4) if show_test else np.nan,
            "Llama_A-C_holm": l_ac["p_holm"] if show_test else np.nan,
            "A-C_replication": replication_verdict(q_ac, l_ac) if show_test else "DESCRIPTIVE ONLY; no separate test",
        })
    return pd.DataFrame(rows)


def main():
    qdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "v2")
    ldir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "results", "v2", "llama31_8b")

    # Qwen uses its frozen runs.jsonl. Llama uses the CANONICAL de-duplicated dataset
    # runs_clean.jsonl (produced by analysis/canonicalize_llama.py) when present, so the
    # comparison is run on exactly 750 non-retry records rather than the raw 777-record log.
    # Raw files are never modified; this only changes which file is READ.
    qfile = "runs.jsonl"
    lfile = "runs_clean.jsonl" if os.path.exists(os.path.join(ldir, "runs_clean.jsonl")) else "runs.jsonl"
    if lfile == "runs.jsonl":
        print("WARNING: results/v2/llama31_8b/runs_clean.jsonl not found; falling back to raw "
              "runs.jsonl (may contain retry duplicates). Run analysis/canonicalize_llama.py first.")

    for d, fn, label in [(qdir, qfile, "Qwen"), (ldir, lfile, "Llama")]:
        if not os.path.exists(os.path.join(d, fn)):
            print(f"ERROR: {label} runs not found at {os.path.join(d, fn)}")
            if label == "Llama":
                print("       Run the Llama experiment first: python src/runner_llama31.py")
            return

    q, qdf = model_stats(qdir, qfile)
    l, ldf = model_stats(ldir, lfile)
    outdir = os.path.join(ldir, "compare")
    os.makedirs(outdir, exist_ok=True)

    table = build_table(q, l)
    table.to_csv(os.path.join(outdir, "cross_model_table.csv"), index=False)

    # replication across ALL paired metrics (A-C), not just the table subset
    rep_rows = []
    for name, col in PAIRED_METRICS:
        q_ac, l_ac = q["paired"][name]["A-C"], l["paired"][name]["A-C"]
        rep_rows.append({"metric": name,
                         "qwen_A-C_med": round(q_ac["median_diff"], 4), "qwen_holm": q_ac["p_holm"],
                         "qwen_rb": round(q_ac["rank_biserial"], 3),
                         "llama_A-C_med": round(l_ac["median_diff"], 4), "llama_holm": l_ac["p_holm"],
                         "llama_rb": round(l_ac["rank_biserial"], 3),
                         "verdict": replication_verdict(q_ac, l_ac)})
    rep = pd.DataFrame(rep_rows)
    rep.to_csv(os.path.join(outdir, "replication.csv"), index=False)

    # per-model metric dump (per-condition mean+median + A/B/C paired full stats)
    pm_rows = []
    for mlabel, st in [("qwen", q), ("llama", l)]:
        for name, col in PAIRED_METRICS:
            for c in CONDS:
                pm_rows.append({"model": mlabel, "metric": name, "condition": SHORT[c],
                                "mean": st["per_condition"][name][SHORT[c]]["mean"],
                                "median": st["per_condition"][name][SHORT[c]]["median"]})
            for pair in ["A-B", "B-C", "A-C"]:
                r = st["paired"][name][pair]
                pm_rows.append({"model": mlabel, "metric": name, "condition": pair,
                                "mean": r["mean_diff"], "median": r["median_diff"],
                                "rank_biserial": r["rank_biserial"], "ci95_lo": r["ci95_lo"],
                                "ci95_hi": r["ci95_hi"], "p_raw": r["p_raw"], "p_holm": r["p_holm"]})
    pd.DataFrame(pm_rows).to_csv(os.path.join(outdir, "per_model_metrics.csv"), index=False)

    summary = {"qwen": q, "llama": l,
               "cross_model_table": table.to_dict("records"),
               "replication": rep.to_dict("records")}
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)

    # ---- console / text report ----
    L = []
    L.append("=" * 90)
    L.append("CROSS-MODEL REPLICATION  —  Qwen-2.5-7B  vs  Llama-3.1-8B  (v2 study)")
    L.append("=" * 90)
    L.append(f"Qwen runs  = {q['n_runs']}   ({os.path.join(qdir, qfile)})")
    L.append(f"Llama runs = {l['n_runs']}   ({os.path.join(ldir, lfile)})")
    L.append("Statistics: task-level paired Wilcoxon, Holm across the 3 pairs per metric; "
             "effect size = matched-pairs rank-biserial; 95% bootstrap CI (all identical to mechanism_v2).")
    L.append("\nCROSS-MODEL TABLE (A = generalist, C = specialized_length_matched; A-C is the key test)")
    L.append(table.to_string(index=False))
    L.append("\nREPLICATION VERDICTS (A-vs-C, all paired metrics):")
    for r in rep_rows:
        L.append(f"  {r['metric']:18s}  Qwen A-C med={r['qwen_A-C_med']:+.3f} holm={r['qwen_holm']:.3g} | "
                 f"Llama A-C med={r['llama_A-C_med']:+.3f} holm={r['llama_holm']:.3g}  ->  {r['verdict']}")
    L.append("\nLATENCY TAIL (per model):")
    for mlabel, st in [("Qwen", q), ("Llama", l)]:
        t = st["tail"]
        L.append(f"  {mlabel}: median A={t['A']['median']:.1f}/B={t['B']['median']:.1f}/C={t['C']['median']:.1f}s | "
                 f"mean A={t['A']['mean']:.1f}/B={t['B']['mean']:.1f}/C={t['C']['mean']:.1f}s | "
                 f">100s A={t['A']['frac_gt_100s']:.1%}/B={t['B']['frac_gt_100s']:.1%}/C={t['C']['frac_gt_100s']:.1%}")
        L.append(f"         shift A-C: {t['shift_AC']['label']}")
    L.append("\nLATENCY<->TOKEN ASSOCIATION (pooled Spearman):")
    for mlabel, st in [("Qwen", q), ("Llama", l)]:
        tt = st["latency_corr"]["total_tokens"]; it = st["latency_corr"]["input_tokens"]
        L.append(f"  {mlabel}: total_tokens rho={tt['pooled_rho']:+.2f}  input_tokens rho={it['pooled_rho']:+.2f}")
    L.append("\nINTERPRETATION RULES ENFORCED: verdicts are computed from the tests above. "
             "A-C 'REPLICATED' => the latency difference persisted after prompt-length matching on BOTH models. "
             "'NOT REPLICATED' => the effect did not reproduce under Llama. 'MODEL-DEPENDENT' => direction reversed. "
             "No causal-reduction or context-size/token/tool/LLM-call reduction claim is made here; associations only.")
    L.append(f"\nOutputs -> {outdir}/  (cross_model_table.csv, replication.csv, per_model_metrics.csv, summary.json)")
    report = "\n".join(L)
    with open(os.path.join(outdir, "compare_report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
