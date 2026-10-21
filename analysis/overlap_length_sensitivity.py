import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, t as t_dist

FILES = {
    "Qwen": "results/v2/runs.jsonl",
    "Llama": "results/v2/llama31_8b/runs_clean.jsonl",
    "Mistral": "results/v2/mistral_7b/runs.jsonl",
}
A = "generalist"
C = "specialized_length_matched"
OUT = Path("results/v2/length_sensitivity")
OUT.mkdir(parents=True, exist_ok=True)


def tokens(text):
    return re.findall(r"\b\w+\b", text.lower())


def metrics(run):
    agents = sorted(run["agent_outputs"], key=lambda x: x["agent_index"])
    if len(agents) < 2:
        raise ValueError("Run has fewer than two agent outputs")

    texts = [agents[i].get("output_text") for i in (0, 1)]
    if not all(isinstance(x, str) for x in texts):
        raise ValueError("Missing/non-string output_text")

    sets = [set(tokens(x)) for x in texts]
    union = sets[0] | sets[1]
    overlap = len(sets[0] & sets[1]) / len(union) if union else 0.0
    words = sum(len(tokens(x)) for x in texts)

    return {
        "overlap": overlap,
        "words": words,
        "truncated": bool(run.get("truncated", False)),
    }


def load_runs(path):
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("condition") not in (A, C):
                continue
            key = (r["task_id"], r["run_index"], r["condition"])
            if key in result:
                raise ValueError(f"Duplicate run key: {key} in {path}")
            result[key] = metrics(r)
    return result


def paired_wilcoxon(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or np.allclose(values, 0):
        return np.nan
    return float(wilcoxon(values, alternative="two-sided").pvalue)


def holm(pvalues):
    valid = [(i, p) for i, p in enumerate(pvalues) if np.isfinite(p)]
    adjusted = [np.nan] * len(pvalues)
    ordered = sorted(valid, key=lambda x: x[1])
    running = 0.0
    m = len(ordered)
    for rank, (i, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[i] = running
    return adjusted


def regression_intercept(x, y):
    # OLS: delta_overlap = intercept + slope * delta_words
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    if n <= 2 or rank < 2:
        return beta[0], beta[1], np.nan, np.nan

    residual = y - X @ beta
    dof = n - 2
    covariance = (residual @ residual / dof) * np.linalg.inv(X.T @ X)
    se_intercept = np.sqrt(max(covariance[0, 0], 0))
    if se_intercept == 0:
        p_intercept = 0.0 if beta[0] != 0 else 1.0
    else:
        statistic = beta[0] / se_intercept
        p_intercept = float(2 * t_dist.sf(abs(statistic), dof))
    return float(beta[0]), float(beta[1]), float(se_intercept), p_intercept


all_results = []
pair_frames = {}
for model, filename in FILES.items():
    runs = load_runs(filename)
    pairs = {}
    task_keys = sorted({
        (task, idx)
        for task, idx, condition in runs
    })

    for task, idx in task_keys:
        a = runs.get((task, idx, A))
        c = runs.get((task, idx, C))
        if a is None or c is None:
            continue
        pairs[(task, idx)] = {
            "model": model,
            "task_id": task,
            "run_index": idx,
            "delta_overlap_A_minus_C": a["overlap"] - c["overlap"],
            "delta_words_A_minus_C": a["words"] - c["words"],
            "A_overlap": a["overlap"],
            "C_overlap": c["overlap"],
            "A_words": a["words"],
            "C_words": c["words"],
            "A_truncated": a["truncated"],
            "C_truncated": c["truncated"],
            "either_truncated": a["truncated"] or c["truncated"],
        }

    df = pd.DataFrame(pairs.values())
    if df.empty:
        raise ValueError(f"No paired A/C runs found for {model}")

    pair_frames[model] = df
    full = df["delta_overlap_A_minus_C"].to_numpy()
    clean_df = df.loc[~df["either_truncated"]]
    clean = clean_df["delta_overlap_A_minus_C"].to_numpy()

    intercept, slope, se, reg_p = regression_intercept(
        df["delta_words_A_minus_C"], full
    )
    result = {
        "model": model,
        "paired_n": len(df),
        "truncated_excluded_n": len(clean_df),
        "truncated_pair_n": int(df["either_truncated"].sum()),
        "mean_delta_overlap_A_minus_C": float(np.mean(full)),
        "median_delta_overlap_A_minus_C": float(np.median(full)),
        "primary_wilcoxon_p": paired_wilcoxon(full),
        "clean_median_delta_overlap_A_minus_C":
            float(np.median(clean)) if len(clean) else np.nan,
        "clean_wilcoxon_p": paired_wilcoxon(clean),
        "length_adjusted_intercept": intercept,
        "length_adjusted_slope_per_word": slope,
        "length_adjusted_intercept_se": se,
        "length_adjusted_intercept_p": reg_p,
    }
    all_results.append(result)

    print(f"\n===== {model} =====")
    for key, value in result.items():
        if key != "model":
            print(f"{key}: {value}")

summary = pd.DataFrame(all_results)
summary["primary_wilcoxon_holm_p"] = holm(
    summary["primary_wilcoxon_p"].tolist()
)
summary["clean_wilcoxon_holm_p"] = holm(
    summary["clean_wilcoxon_p"].tolist()
)
summary["length_adjusted_intercept_holm_p"] = holm(
    summary["length_adjusted_intercept_p"].tolist()
)

pairs_out = pd.concat(pair_frames.values(), ignore_index=True)
pairs_out.to_csv(OUT / "paired_output_lengths.csv", index=False)
summary.to_csv(OUT / "overlap_length_sensitivity.csv", index=False)

print("\n===== SUMMARY WITH HOLM-ADJUSTED P-VALUES =====")
print(summary.to_string(index=False))
print(f"\nSaved: {OUT / 'paired_output_lengths.csv'}")
print(f"Saved: {OUT / 'overlap_length_sensitivity.csv'}")
