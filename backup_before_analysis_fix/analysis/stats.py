"""Statistical analysis (confound C9: runs are NOT iid; they cluster within task).

PRIMARY analysis is paired at the TASK level: for each task we average the per-run
metric within each condition, then run a paired test ACROSS the 30 tasks (both
conditions see the same tasks). This respects the clustering and matches the design.

SECONDARY (optional): a mixed-effects / GEE model with task as a cluster, if statsmodels
is installed.

Multiple-comparison control (Holm) is applied across the primary dependent variables.

Run:  python analysis/stats.py
"""
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS = os.path.join(ROOT, "results", "runs.jsonl")
OUT = os.path.join(ROOT, "results", "stats_report.txt")

PRIMARY_DVS = {
    "success": "correct",          # reliability
    "total_tokens": "total_tokens",  # cost
    "latency_seconds": "latency_seconds",  # latency
}


def load_df():
    rows = []
    with open(RUNS, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                d["correct"] = 1 if d["correct"] else 0
                d["failure_types"] = d.get("failure_types", [])
                rows.append(d)
    return pd.DataFrame(rows)


def holm(pvals):
    """Holm-Bonferroni. Returns adjusted p-values in original order."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def bootstrap_ci(x, iters=5000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    means = [rng.choice(x, size=len(x), replace=True).mean() for _ in range(iters)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_task_level(df, lines):
    lines.append("=" * 70)
    lines.append("PRIMARY: paired task-level analysis (Wilcoxon signed-rank across tasks)")
    lines.append("=" * 70)
    pvals, labels = [], []
    results = {}
    for name, col in PRIMARY_DVS.items():
        piv = df.groupby(["task_id", "condition"])[col].mean().unstack("condition")
        if not {"generalist", "specialized"}.issubset(piv.columns):
            lines.append(f"[{name}] missing a condition; skipped.")
            continue
        piv = piv.dropna()
        g, s = piv["generalist"].values, piv["specialized"].values
        diff = s - g
        try:
            stat, p = stats.wilcoxon(s, g)
        except ValueError:
            stat, p = np.nan, 1.0  # all-zero differences
        pvals.append(p)
        labels.append(name)
        results[name] = {"g_mean": g.mean(), "s_mean": s.mean(),
                         "median_diff": float(np.median(diff)), "p": p}
        lines.append(f"\n[{name}]  n_tasks={len(piv)}")
        lines.append(f"  generalist  mean = {g.mean():.4f}")
        lines.append(f"  specialized mean = {s.mean():.4f}")
        lines.append(f"  median(spec - gen) = {np.median(diff):+.4f}")
        lines.append(f"  Wilcoxon stat={stat}, raw p={p:.4g}")

    if pvals:
        adj = holm(pvals)
        lines.append("\nHolm-adjusted p-values across primary DVs:")
        for lab, raw, a in zip(labels, pvals, adj):
            sig = "*" if a < 0.05 else " "
            lines.append(f"  {lab:16s} raw={raw:.4g}  adj={a:.4g} {sig}")
    return results


def overall_rates(df, lines):
    lines.append("\n" + "=" * 70)
    lines.append("Overall per-condition summary (runs pooled; CI via task-mean bootstrap)")
    lines.append("=" * 70)
    for cond in sorted(df["condition"].unique()):
        sub = df[df["condition"] == cond]
        task_success = sub.groupby("task_id")["correct"].mean().values
        lo, hi = bootstrap_ci(task_success)
        lines.append(f"\n[{cond}]  runs={len(sub)}")
        lines.append(f"  success rate = {sub['correct'].mean():.3f}  (task-mean 95% CI {lo:.3f}-{hi:.3f})")
        lines.append(f"  mean total tokens = {sub['total_tokens'].mean():.0f}")
        lines.append(f"  mean system-prompt tokens (overhead) = {sub['system_prompt_tokens'].mean():.0f}")
        lines.append(f"  mean latency = {sub['latency_seconds'].mean():.2f}s")
        lines.append(f"  mean llm_calls = {sub['llm_calls'].mean():.2f}")


def by_difficulty(df, lines):
    lines.append("\n" + "=" * 70)
    lines.append("Reliability by difficulty x condition (RQ5: when does it help/hurt?)")
    lines.append("=" * 70)
    tab = df.groupby(["difficulty", "condition"])["correct"].mean().unstack("condition")
    lines.append("\n" + tab.round(3).to_string())


def failure_profile(df, lines):
    lines.append("\n" + "=" * 70)
    lines.append("Failure-type frequency per condition (RQ6: pathways shifted)")
    lines.append("=" * 70)
    from collections import Counter
    for cond in sorted(df["condition"].unique()):
        sub = df[df["condition"] == cond]
        c = Counter()
        for fl in sub["failure_types"]:
            for f in fl:
                c[f] += 1
        n = len(sub)
        lines.append(f"\n[{cond}] (n={n} runs)")
        for f in sorted(c):
            lines.append(f"  {f}: {c[f]} ({100*c[f]/n:.1f}% of runs)")


def optional_mixed_model(df, lines):
    lines.append("\n" + "=" * 70)
    lines.append("SECONDARY: GEE logistic (correct ~ condition), clustered by task")
    lines.append("=" * 70)
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        d = df.copy()
        d["spec"] = (d["condition"] == "specialized").astype(int)
        model = smf.gee("correct ~ spec", groups="task_id", data=d,
                        family=sm.families.Binomial(),
                        cov_struct=sm.cov_struct.Exchangeable())
        res = model.fit()
        lines.append("\n" + str(res.summary()))
    except ImportError:
        lines.append("\n(statsmodels not installed — `pip install statsmodels` to enable.)")
    except Exception as e:
        lines.append(f"\n(GEE model failed: {e})")


def main():
    df = load_df()
    if df.empty:
        print("No runs found. Run src/runner.py first.")
        return
    lines = [f"Analysis of {len(df)} runs across conditions: {sorted(df['condition'].unique())}"]
    paired_task_level(df, lines)
    overall_rates(df, lines)
    by_difficulty(df, lines)
    failure_profile(df, lines)
    optional_mixed_model(df, lines)
    report = "\n".join(lines)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\nWritten to {OUT}")


if __name__ == "__main__":
    main()
