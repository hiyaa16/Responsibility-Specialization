"""v2 statistics: THREE-condition, task-level paired analysis (confound C9: runs cluster in
tasks, so we average the 5 runs per task first, then analyze across tasks).

Design:
  * Omnibus: Friedman test across A/B/C on task-mean values (paired by task).
  * Post-hoc: pairwise Wilcoxon signed-rank for A-B, B-C, A-C, Holm-corrected across the 3
    pairs within each DV. Effect size = matched-pairs rank-biserial; plus median diff + 95% CI.
  * Reliability equivalence: TOST on paired task-mean success diffs, margin 0.05 (pre-registered
    in docs/v2_equivalence_margin.md), 90% CI reported.
  * Breakdowns: success by source and by difficulty; token decomposition (focus B vs C);
    failure-type frequency per condition with denominators; timeout/error accounting.

Run:  python analysis/stats_v2.py                      # reads results/v2/runs.jsonl
      python analysis/stats_v2.py results/v2_pilot     # pilot sanity (3 conditions present)
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONDS = ["generalist", "specialized", "specialized_length_matched"]
SHORT = {"generalist": "A", "specialized": "B", "specialized_length_matched": "C"}
PAIRS = [("generalist", "specialized"), ("specialized", "specialized_length_matched"),
         ("generalist", "specialized_length_matched")]
PRIMARY_DVS = {"success": "correct", "total_tokens": "total_tokens",
               "output_tokens": "output_tokens", "latency_seconds": "latency_seconds"}
EQUIV_MARGIN = 0.05  # pre-registered; see docs/v2_equivalence_margin.md


def load_df(run_dir):
    path = os.path.join(run_dir, "runs.jsonl")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                d["correct"] = 1 if d["correct"] else 0
                d["failure_types"] = d.get("failure_types", [])
                rows.append(d)
    return pd.DataFrame(rows)


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(running, 1.0)
    return adj


def rank_biserial(diff):
    d = diff[diff != 0]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    wp = ranks[d > 0].sum()
    wm = ranks[d < 0].sum()
    return float((wp - wm) / (wp + wm)) if (wp + wm) else 0.0


def boot_ci(x, iters=5000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = [rng.choice(x, len(x), replace=True).mean() for _ in range(iters)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def tost_paired(diff, margin, alpha=0.05):
    """Two one-sided tests on paired differences. Returns (p_tost, equivalent, ci90)."""
    d = np.asarray(diff, float)
    n = len(d)
    mean, sd = d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n) if sd > 0 else 1e-12
    df = n - 1
    t_lower = (mean - (-margin)) / se
    t_upper = (mean - margin) / se
    p_lower = 1 - stats.t.cdf(t_lower, df)   # H0: mean <= -margin
    p_upper = stats.t.cdf(t_upper, df)       # H0: mean >=  margin
    p_tost = max(p_lower, p_upper)
    tcrit = stats.t.ppf(1 - alpha, df)
    ci90 = (mean - tcrit * se, mean + tcrit * se)
    return float(p_tost), bool(p_tost < alpha), ci90


def task_pivot(df, col):
    piv = df.groupby(["task_id", "condition"])[col].mean().unstack("condition")
    present = [c for c in CONDS if c in piv.columns]
    return piv[present].dropna(), present


def primary(df, L):
    L += ["=" * 74, "PRIMARY: 3-condition task-level analysis (Friedman + pairwise Wilcoxon/Holm)", "=" * 74]
    for name, col in PRIMARY_DVS.items():
        piv, present = task_pivot(df, col)
        L.append(f"\n[{name}]  n_tasks={len(piv)}  conditions={[SHORT[c] for c in present]}")
        for c in present:
            L.append(f"  {SHORT[c]} {c:28s} mean={piv[c].mean():.4f}  median={piv[c].median():.4f}")
        if len(present) == 3 and len(piv) >= 3:
            chi, p = stats.friedmanchisquare(*[piv[c].values for c in CONDS])
            L.append(f"  Friedman chi2={chi:.3f}  p={p:.4g}")
        # pairwise
        raw, labels, info = [], [], []
        for a, b in PAIRS:
            if a not in present or b not in present:
                continue
            x, y = piv[a].values, piv[b].values
            diff = y - x
            try:
                _, p = stats.wilcoxon(y, x)
            except ValueError:
                p = 1.0
            raw.append(p); labels.append(f"{SHORT[a]}-{SHORT[b]}")
            lo, hi = boot_ci(diff)
            info.append((f"{SHORT[a]}-{SHORT[b]}", np.median(diff), rank_biserial(diff), lo, hi, p))
        if raw:
            adj = holm(raw)
            for (lab, md, rb, lo, hi, p), a in zip(info, adj):
                sig = "*" if a < 0.05 else " "
                L.append(f"    {lab}: median(2nd-1st)={md:+.4f}  rank-biserial={rb:+.3f}  "
                         f"95%CI[{lo:+.4f},{hi:+.4f}]  p={p:.4g} adj={a:.4g}{sig}")


def equivalence(df, L):
    L += ["\n" + "=" * 74, f"RELIABILITY EQUIVALENCE (TOST, pre-registered margin +/-{EQUIV_MARGIN})", "=" * 74]
    piv, present = task_pivot(df, "correct")
    for a, b in PAIRS:
        if a not in present or b not in present:
            continue
        diff = piv[b].values - piv[a].values
        p, eq, ci = tost_paired(diff, EQUIV_MARGIN)
        verdict = "EQUIVALENT" if eq else "NOT shown equivalent (inconclusive)"
        L.append(f"  {SHORT[a]}-{SHORT[b]}: mean diff={diff.mean():+.4f}  90%CI[{ci[0]:+.4f},{ci[1]:+.4f}]  "
                 f"TOST p={p:.4g}  -> {verdict}")
    L.append("  NOTE: a non-significant difference is not proof of equivalence; only a passing TOST is.")


def breakdowns(df, L):
    L += ["\n" + "=" * 74, "Overall per-condition summary (task-mean bootstrap 95% CI)", "=" * 74]
    for c in [x for x in CONDS if x in df["condition"].unique()]:
        sub = df[df["condition"] == c]
        ts = sub.groupby("task_id")["correct"].mean().values
        lo, hi = boot_ci(ts)
        good = sub[sub["error"] == ""] if "error" in sub else sub
        L.append(f"\n[{SHORT[c]} {c}]  runs={len(sub)}  timeouts={int(sub.get('timeout', pd.Series([0]*len(sub))).sum())}  "
                 f"errors={int((sub.get('error', pd.Series(['']*len(sub))) != '').sum())}")
        L.append(f"  success={sub['correct'].mean():.3f} (task-mean 95%CI {lo:.3f}-{hi:.3f})")
        L.append(f"  mean total_tokens={good['total_tokens'].mean():.0f}  output_tokens={good['output_tokens'].mean():.0f}  "
                 f"system_prompt_tokens={sub['system_prompt_tokens'].mean():.0f}")
        L.append(f"  mean latency={good['latency_seconds'].mean():.1f}s  median={good['latency_seconds'].median():.1f}s  "
                 f"mean llm_calls={good['llm_calls'].mean():.2f}")

    for dim in ["source", "difficulty"]:
        if dim not in df:
            continue
        L += ["\n" + "-" * 50, f"success by {dim} x condition"]
        tab = df.groupby([dim, "condition"])["correct"].mean().unstack("condition")
        tab = tab[[c for c in CONDS if c in tab.columns]]
        L.append(tab.round(3).to_string())

    L += ["\n" + "=" * 74, "TOKEN DECOMPOSITION (critical question: does efficiency survive length control? B vs C)", "=" * 74]
    good = df[df["error"] == ""] if "error" in df else df
    for col in ["system_prompt_tokens", "input_tokens", "output_tokens", "total_tokens"]:
        vals = {SHORT[c]: good[good["condition"] == c][col].mean() for c in CONDS if c in good["condition"].unique()}
        L.append(f"  {col:22s} " + "  ".join(f"{k}={v:.0f}" for k, v in vals.items()))

    L += ["\n" + "=" * 74, "Failure-type frequency per condition (denominator = runs)", "=" * 74]
    from collections import Counter
    for c in [x for x in CONDS if x in df["condition"].unique()]:
        sub = df[df["condition"] == c]
        cnt = Counter(f for fl in sub["failure_types"] for f in fl)
        n = len(sub)
        L.append(f"\n[{SHORT[c]} {c}] (n={n})")
        for f in sorted(cnt):
            L.append(f"  {f}: {cnt[f]} ({100*cnt[f]/n:.1f}%)")
    L.append("\n  NOTE: F1 (wrong-responsibility) & F7 (out-of-role tool use) are structurally only "
             "possible in B and C (generalists own all roles). High F2 with high accuracy => inspect "
             "for heuristic over-trigger (Part 17); do NOT change the classifier here.")


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "v2")
    if not os.path.isabs(run_dir):
        run_dir = os.path.join(ROOT, run_dir)
    df = load_df(run_dir)
    if df.empty:
        print("No runs found in", run_dir)
        return
    L = [f"v2 analysis of {len(df)} runs | conditions={sorted(df['condition'].unique())} | dir={run_dir}"]
    primary(df, L)
    equivalence(df, L)
    breakdowns(df, L)
    report = "\n".join(L)
    out = os.path.join(run_dir, "stats_v2_report.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
