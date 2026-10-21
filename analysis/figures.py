"""Figures. Saves PNGs to results/figures/. Run: python analysis/figures.py"""
from __future__ import annotations
import os
import json
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS = os.path.join(ROOT, "results", "runs.jsonl")
FIGDIR = os.path.join(ROOT, "results", "figures")

COND_ORDER = ["generalist", "specialized"]
COLORS = {"generalist": "#4C72B0", "specialized": "#DD8452"}


def load_df():
    rows = []
    with open(RUNS, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                d["correct"] = 1 if d["correct"] else 0
                rows.append(d)
    return pd.DataFrame(rows)


def _bootstrap_ci(x, iters=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    m = [rng.choice(x, len(x), replace=True).mean() for _ in range(iters)]
    return np.mean(x), np.percentile(m, 2.5), np.percentile(m, 97.5)


def fig_reliability(df):
    fig, ax = plt.subplots(figsize=(5, 4))
    for i, cond in enumerate(COND_ORDER):
        ts = df[df.condition == cond].groupby("task_id")["correct"].mean().values
        mean, lo, hi = _bootstrap_ci(ts)
        ax.bar(i, mean, color=COLORS[cond], width=0.6)
        ax.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]], fmt="none",
                    ecolor="black", capsize=5)
    ax.set_xticks(range(len(COND_ORDER)))
    ax.set_xticklabels(COND_ORDER)
    ax.set_ylabel("Task success rate")
    ax.set_ylim(0, 1)
    ax.set_title("Reliability by condition (95% CI)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "reliability.png"), dpi=150)
    plt.close(fig)


def fig_cost_latency(df):
    for metric, label in [("total_tokens", "Total tokens"), ("latency_seconds", "Latency (s)")]:
        fig, ax = plt.subplots(figsize=(5, 4))
        data = [df[df.condition == c][metric].values for c in COND_ORDER]
        bp = ax.boxplot(data, labels=COND_ORDER, patch_artist=True)
        for patch, c in zip(bp["boxes"], COND_ORDER):
            patch.set_facecolor(COLORS[c])
        ax.set_ylabel(label)
        ax.set_title(f"{label} by condition")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGDIR, f"{metric}.png"), dpi=150)
        plt.close(fig)


def fig_by_difficulty(df):
    tab = df.groupby(["difficulty", "condition"])["correct"].mean().unstack("condition")
    tab = tab.reindex(columns=[c for c in COND_ORDER if c in tab.columns])
    fig, ax = plt.subplots(figsize=(6, 4))
    tab.plot(kind="bar", ax=ax, color=[COLORS[c] for c in tab.columns])
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1)
    ax.set_title("Reliability by difficulty x condition")
    ax.legend(title="condition")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "by_difficulty.png"), dpi=150)
    plt.close(fig)


def fig_failures(df):
    cats = [f"F{i}" for i in range(1, 10)]
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.38
    x = np.arange(len(cats))
    for k, cond in enumerate(COND_ORDER):
        sub = df[df.condition == cond]
        n = max(len(sub), 1)
        c = Counter()
        for fl in sub["failure_types"]:
            for f in fl:
                c[f] += 1
        vals = [100 * c[cat] / n for cat in cats]
        ax.bar(x + (k - 0.5) * width, vals, width, label=cond, color=COLORS[cond])
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("% of runs")
    ax.set_title("Failure-type frequency by condition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "failures.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    df = load_df()
    if df.empty:
        print("No runs found. Run src/runner.py first.")
        return
    fig_reliability(df)
    fig_cost_latency(df)
    fig_by_difficulty(df)
    fig_failures(df)
    print(f"Figures written to {FIGDIR}")


if __name__ == "__main__":
    main()
