"""v2 figures (3 conditions). Saves PNGs next to the runs.
Run: python analysis/figures_v2.py [results/v2|results/v2_pilot]"""
from __future__ import annotations
import os
import sys
import json
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
COND_ORDER = ["generalist", "specialized", "specialized_length_matched"]
SHORT = {"generalist": "A gen", "specialized": "B spec", "specialized_length_matched": "C spec+pad"}
COLORS = {"generalist": "#4C72B0", "specialized": "#DD8452", "specialized_length_matched": "#55A868"}


def load_df(run_dir):
    rows = []
    with open(os.path.join(run_dir, "runs.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                d["correct"] = 1 if d["correct"] else 0
                rows.append(d)
    return pd.DataFrame(rows)


def _ci(x, iters=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = [rng.choice(x, len(x), replace=True).mean() for _ in range(iters)]
    return np.mean(x), np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "v2")
    if not os.path.isabs(run_dir):
        run_dir = os.path.join(ROOT, run_dir)
    df = load_df(run_dir)
    if df.empty:
        print("No runs in", run_dir); return
    conds = [c for c in COND_ORDER if c in df["condition"].unique()]
    figdir = os.path.join(run_dir, "figures")
    os.makedirs(figdir, exist_ok=True)
    good = df[df["error"] == ""] if "error" in df else df

    # reliability
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, c in enumerate(conds):
        ts = df[df.condition == c].groupby("task_id")["correct"].mean().values
        m, lo, hi = _ci(ts)
        ax.bar(i, m, color=COLORS[c], width=0.6)
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor="black", capsize=5)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels([SHORT[c] for c in conds])
    ax.set_ylim(0, 1); ax.set_ylabel("Task success"); ax.set_title("Reliability (95% CI)")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "reliability.png"), dpi=150); plt.close(fig)

    # token decomposition (stacked-ish grouped) + latency boxplots
    for metric, label in [("total_tokens", "Total tokens"), ("output_tokens", "Output tokens"),
                          ("latency_seconds", "Latency (s)")]:
        fig, ax = plt.subplots(figsize=(6, 4))
        data = [good[good.condition == c][metric].values for c in conds]
        bp = ax.boxplot(data, labels=[SHORT[c] for c in conds], patch_artist=True)
        for patch, c in zip(bp["boxes"], conds):
            patch.set_facecolor(COLORS[c])
        ax.set_ylabel(label); ax.set_title(f"{label} by condition")
        fig.tight_layout(); fig.savefig(os.path.join(figdir, f"{metric}.png"), dpi=150); plt.close(fig)

    # failures
    cats = [f"F{i}" for i in range(1, 10)]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.26
    x = np.arange(len(cats))
    for k, c in enumerate(conds):
        sub = df[df.condition == c]
        n = max(len(sub), 1)
        cnt = Counter(f for fl in sub["failure_types"] for f in fl)
        ax.bar(x + (k - 1) * width, [100 * cnt[cat] / n for cat in cats], width,
               label=SHORT[c], color=COLORS[c])
    ax.set_xticks(x); ax.set_xticklabels(cats); ax.set_ylabel("% of runs")
    ax.set_title("Failure-type frequency by condition"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "failures.png"), dpi=150); plt.close(fig)

    print(f"Figures -> {figdir}")


if __name__ == "__main__":
    main()
