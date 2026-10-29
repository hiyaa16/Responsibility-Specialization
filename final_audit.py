"""Final audit — regenerates every final-results table from the RAW run records.

Independent of analysis/* : reads the raw JSONL directly, recomputes task-level paired
Wilcoxon + Holm, bootstrap CIs, truncation sensitivity, latency/token audit, and writes
CSVs + a summary to final_results/. No LLM calls, no new experiments, no mutation of
original files. Runs in ~1 second.

Usage:  python final_audit.py
"""
from __future__ import annotations
import json, os, csv
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "final_results")
os.makedirs(OUT, exist_ok=True)

CONDS = ["generalist", "specialized", "specialized_length_matched"]
SHORT = {"generalist": "A", "specialized": "B", "specialized_length_matched": "C"}

# raw sources of truth (each 750 runs except v1=300)
V2_MODELS = {
    "Qwen":    "results/v2/runs.jsonl",
    "Llama":   "results/v2/llama31_8b/runs_clean.jsonl",   # de-duplicated canonical set (777 raw -> 750)
    "Mistral": "results/v2/mistral_7b/runs.jsonl",
}
V1_PATH = "results/runs.jsonl"
LLAMA_RAW = "results/v2/llama31_8b/runs.jsonl"

# ----------------------------------------------------------------- metric helpers
def word_set(t):
    return set("".join(c.lower() if c.isalnum() else " " for c in (t or "")).split())
def jaccard(a, b):
    wa, wb = word_set(a), word_set(b)
    return len(wa & wb) / len(wa | wb) if (wa and wb) else 0.0
def holm(pvals):
    pvals = list(pvals); m = len(pvals); order = np.argsort(pvals)
    adj = np.empty(m); run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * pvals[idx]); adj[idx] = min(run, 1.0)
    return adj
def rank_biserial(diff):
    d = np.asarray(diff, float); d = d[d != 0]
    if len(d) == 0: return 0.0
    r = stats.rankdata(np.abs(d)); wp = r[d > 0].sum(); wm = r[d < 0].sum()
    return float((wp - wm) / (wp + wm)) if (wp + wm) else 0.0
def boot_ci(x, iters=5000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    if len(x) == 0: return (np.nan, np.nan)
    v = [np.mean(rng.choice(x, len(x), replace=True)) for _ in range(iters)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

def load(path):
    return [json.loads(l) for l in open(os.path.join(ROOT, path), encoding="utf-8") if l.strip()]

def rec(r):
    ao = sorted(r["agent_outputs"], key=lambda a: a["agent_index"])
    outs = [a["output_text"] for a in ao]
    return {"task_id": r["task_id"], "condition": r["condition"], "run_index": r["run_index"],
            "correct": 1 if r["correct"] else 0, "latency_seconds": r["latency_seconds"],
            "total_tokens": r["total_tokens"], "input_tokens": r["input_tokens"],
            "output_tokens": r["output_tokens"], "llm_calls": r["llm_calls"],
            "repeated_work": r.get("repeated_work", 0.0), "corrections": r.get("corrections", 0),
            "truncated": bool(r.get("truncated")), "timeout": bool(r.get("timeout")),
            "error": r.get("error", ""), "model": r.get("model"), "temperature": r.get("temperature"),
            "ov_01": jaccard(outs[0], outs[1]) if len(outs) >= 2 else np.nan}

def task_grid(rows, col, conds=CONDS):
    d = {}
    for r in rows:
        d.setdefault((r["task_id"], r["condition"]), []).append(r[col])
    tasks = sorted({t for (t, c) in d})
    kept = [t for t in tasks if all((t, c) in d for c in conds)]
    grid = {c: np.array([np.mean(d[(t, c)]) for t in kept], float) for c in conds}
    return kept, grid

def paired(grid, a, b):
    x, y = grid[a], grid[b]; diff = x - y
    try: _, p = stats.wilcoxon(x, y)
    except ValueError: p = 1.0
    lo, hi = boot_ci(diff)
    return {"n": len(diff), "mean_diff": float(np.mean(diff)), "median_diff": float(np.median(diff)),
            "rb": rank_biserial(diff), "ci_lo": lo, "ci_hi": hi, "p_raw": float(p)}

def write_csv(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"  wrote {name}  ({len(rows)} rows)")

# ----------------------------------------------------------------- load everything
DATA = {m: [rec(r) for r in load(p)] for m, p in V2_MODELS.items()}
RAWCOUNT = {"Qwen_v2": 750, "Llama_v2_clean": 750, "Llama_v2_raw": len(load(LLAMA_RAW)),
            "Mistral_v2": 750, "Qwen_v1": len(load(V1_PATH))}
METRICS = ["correct", "ov_01", "total_tokens", "input_tokens", "output_tokens",
           "latency_seconds", "llm_calls", "repeated_work", "corrections"]
PAIRS = [("generalist", "specialized", "A-B"),
         ("specialized", "specialized_length_matched", "B-C"),
         ("generalist", "specialized_length_matched", "A-C")]

# ================================================================= 1. data_quality_audit
print("Generating final_results/ ...")
dq_rows = []
for lbl, path in [("Qwen_v2", V2_MODELS["Qwen"]), ("Llama_v2_raw", LLAMA_RAW),
                  ("Llama_v2_clean", V2_MODELS["Llama"]), ("Mistral_v2", V2_MODELS["Mistral"]),
                  ("Qwen_v1", V1_PATH)]:
    raw = load(path)
    ids = [r["experiment_id"] for r in raw]
    import collections
    dup = sum(v - 1 for v in collections.Counter(ids).values() if v > 1)
    conds = collections.Counter(r["condition"] for r in raw)
    trunc = sum(1 for r in raw if r.get("truncated"))
    tmo = sum(1 for r in raw if r.get("timeout"))
    err = sum(1 for r in raw if r.get("error"))
    miss_model = sum(1 for r in raw if not r.get("model"))
    dq_rows.append([lbl, len(raw), len(set(ids)), dup, dict(conds).__str__(), trunc, tmo, err, miss_model])
write_csv("data_quality_audit.csv",
          ["dataset", "n_rows", "n_unique_experiment_id", "n_duplicate_rows",
           "condition_counts", "truncated", "timeout", "error", "missing_model_label"], dq_rows)

# ================================================================= 2. experiment_design_summary
ed_rows = []
for mdl, rows in DATA.items():
    import collections
    tasks = sorted({r["task_id"] for r in rows})
    for c in CONDS:
        sub = [r for r in rows if r["condition"] == c]
        ed_rows.append([mdl, sub[0]["model"], SHORT[c], c, len(tasks),
                        max(r["run_index"] for r in rows) + 1, len(sub),
                        sorted({r["temperature"] for r in rows}).__str__()])
# v1 (separate experiment, 2 conditions)
v1 = [rec(r) for r in load(V1_PATH)]
for c in ["generalist", "specialized"]:
    sub = [r for r in v1 if r["condition"] == c]
    ed_rows.append(["Qwen_v1(separate)", sub[0]["model"], SHORT[c], c,
                    len(sorted({r["task_id"] for r in v1})), 5, len(sub), "[0.7]"])
write_csv("experiment_design_summary.csv",
          ["study", "model", "cond_short", "condition", "n_tasks", "runs_per_task",
           "n_runs", "temperatures"], ed_rows)

# ================================================================= 3. final_primary_results
# Primary comparison = A vs C.  Pre-registered primary OUTCOME = reliability (accuracy).
# Post-hoc elevated mechanism metric = first-handoff overlap ov_01 (the primary_overlap_holm family).
# We report BOTH, with both Holm families for ov_01.
ov_ac_praw = {}
acc_ac = {}
for mdl, rows in DATA.items():
    _, g = task_grid(rows, "ov_01"); ov_ac_praw[mdl] = paired(g, "generalist", "specialized_length_matched")
    _, ga = task_grid(rows, "correct"); acc_ac[mdl] = paired(ga, "generalist", "specialized_length_matched")
holm_models = holm([ov_ac_praw[m]["p_raw"] for m in V2_MODELS])
pr_rows = []
for mdl, hm in zip(V2_MODELS, holm_models):
    r = ov_ac_praw[mdl]; _, g = task_grid(DATA[mdl], "ov_01")
    A, C = float(np.mean(g["generalist"])), float(np.mean(g["specialized_length_matched"]))
    # within-model Holm across the 3 pairs for ov_01
    praws = [paired(g, a, b)["p_raw"] for a, b, _ in PAIRS]
    hp = holm(praws)[2]  # A-C is 3rd
    pr_rows.append(["first_handoff_overlap(ov_01)[post-hoc primary]", mdl, r["n"], round(A, 4), round(C, 4),
                    round(r["median_diff"], 4), round(r["rb"], 3), round(r["ci_lo"], 4), round(r["ci_hi"], 4),
                    f"{r['p_raw']:.6g}", f"{hm:.6g}", f"{hp:.6g}", hm < 0.05])
for mdl in V2_MODELS:
    r = acc_ac[mdl]; _, g = task_grid(DATA[mdl], "correct")
    A, C = float(np.mean(g["generalist"])), float(np.mean(g["specialized_length_matched"]))
    praws = [paired(g, a, b)["p_raw"] for a, b, _ in PAIRS]
    hp = holm(praws)[2]
    pr_rows.append(["accuracy[pre-registered primary outcome]", mdl, r["n"], round(A, 4), round(C, 4),
                    round(r["median_diff"], 4), round(r["rb"], 3), round(r["ci_lo"], 4), round(r["ci_hi"], 4),
                    f"{r['p_raw']:.6g}", "n/a", f"{hp:.6g}", hp < 0.05])
write_csv("final_primary_results.csv",
          ["metric", "model", "n_task_pairs", "A_mean", "C_mean", "A-C_median_diff", "rank_biserial",
           "ci95_lo(A-C mean diff)", "ci95_hi", "p_raw", "p_holm_across_models", "p_holm_across_pairs",
           "significant_0.05(primary family)"], pr_rows)

# ================================================================= 4. final_three_condition_comparison
tc_rows = []
for mdl, rows in DATA.items():
    for col in METRICS:
        _, g = task_grid(rows, col)
        means = {SHORT[c]: float(np.mean([r[col] for r in rows if r["condition"] == c])) for c in CONDS}
        praws = [paired(g, a, b)["p_raw"] for a, b, _ in PAIRS]
        adj = holm(praws)
        res = {}
        for (a, b, lbl), pa in zip(PAIRS, adj):
            pp = paired(g, a, b); pp["p_holm"] = float(pa); res[lbl] = pp
        tc_rows.append([mdl, col, round(means["A"], 4), round(means["B"], 4), round(means["C"], 4),
                        round(res["A-B"]["median_diff"], 4), f"{res['A-B']['p_holm']:.4g}",
                        round(res["B-C"]["median_diff"], 4), f"{res['B-C']['p_holm']:.4g}",
                        round(res["A-C"]["median_diff"], 4), f"{res['A-C']['p_holm']:.4g}"])
write_csv("final_three_condition_comparison.csv",
          ["model", "metric", "A_mean", "B_mean", "C_mean",
           "A-B_median_diff", "A-B_p_holm", "B-C_median_diff", "B-C_p_holm",
           "A-C_median_diff", "A-C_p_holm"], tc_rows)

# ================================================================= 5. final_model_comparison (A-C replication)
def verdict(ac):
    sig = {k: v["p_holm_pairs"] < 0.05 for k, v in ac.items()}
    sign = {k: (0 if v["median_diff"] == 0 else (1 if v["median_diff"] > 0 else -1)) for k, v in ac.items()}
    q = "Qwen"
    if not sig[q]:
        anys = [k for k in ac if sig[k]]
        return "N/A (no significant A-C effect in any model)" if not anys else \
               f"MODEL-DEPENDENT (sig in {','.join(anys)} not Qwen ref)"
    qd = sign[q]; others = ["Llama", "Mistral"]
    if all(sig[k] and sign[k] == qd and qd != 0 for k in others):
        return "REPLICATED (all 3: same direction, Holm-sig)"
    if any(sig[k] and sign[k] != 0 and sign[k] != qd for k in others):
        return "MODEL-DEPENDENT (direction reverses across models)"
    if all(sign[k] == qd for k in others) and qd != 0:
        return "PARTIALLY REPLICATED (same direction, not sig in all 3)"
    return "NOT REPLICATED"
mc_rows = []
for col in METRICS:
    ac = {}
    for mdl, rows in DATA.items():
        _, g = task_grid(rows, col)
        praws = [paired(g, a, b)["p_raw"] for a, b, _ in PAIRS]
        pp = paired(g, "generalist", "specialized_length_matched")
        pp["p_holm_pairs"] = float(holm(praws)[2]); ac[mdl] = pp
    mc_rows.append([col,
                    round(ac["Qwen"]["median_diff"], 4), f"{ac['Qwen']['p_holm_pairs']:.4g}", round(ac["Qwen"]["rb"], 3),
                    round(ac["Llama"]["median_diff"], 4), f"{ac['Llama']['p_holm_pairs']:.4g}", round(ac["Llama"]["rb"], 3),
                    round(ac["Mistral"]["median_diff"], 4), f"{ac['Mistral']['p_holm_pairs']:.4g}", round(ac["Mistral"]["rb"], 3),
                    verdict(ac)])
write_csv("final_model_comparison.csv",
          ["metric(A-C)", "Qwen_med", "Qwen_holm", "Qwen_rb", "Llama_med", "Llama_holm", "Llama_rb",
           "Mistral_med", "Mistral_holm", "Mistral_rb", "replication_verdict"], mc_rows)

# ================================================================= 6. final_truncation_sensitivity (primary ov_01 A-C)
ts_rows = []
orig_p = {m: ov_ac_praw[m]["p_raw"] for m in V2_MODELS}
clean_p = {}
for mdl, rows in DATA.items():
    clean = [r for r in rows if not r["truncated"]]
    _, g = task_grid(clean, "ov_01"); r = paired(g, "generalist", "specialized_length_matched")
    clean_p[mdl] = r["p_raw"]
    ts_rows.append([mdl, len(rows), sum(1 for x in rows if x["truncated"]), len(clean),
                    round(ov_ac_praw[mdl]["median_diff"], 4), f"{orig_p[mdl]:.6g}",
                    round(r["median_diff"], 4), f"{r['p_raw']:.6g}"])
h_o = holm([orig_p[m] for m in V2_MODELS]); h_c = holm([clean_p[m] for m in V2_MODELS])
for i, mdl in enumerate(V2_MODELS):
    ts_rows[i].append(f"{h_o[i]:.6g}"); ts_rows[i].append(f"{h_c[i]:.6g}")
write_csv("final_truncation_sensitivity.csv",
          ["model", "n_runs_all", "n_truncated", "n_runs_clean", "orig_A-C_median_diff",
           "orig_A-C_p_raw", "clean_A-C_median_diff", "clean_A-C_p_raw",
           "orig_p_holm_across_models", "clean_p_holm_across_models"], ts_rows)

# ================================================================= 7. final_latency_token_audit
EXEC = {"Qwen": (False, True), "Llama": (True, False), "Mistral": (True, False)}  # (has_exec_order, condition_blocked)
lt_rows = []
for mdl, rows in DATA.items():
    has_eo, blocked = EXEC[mdl]
    for c in CONDS:
        v = np.array([r["latency_seconds"] for r in rows if r["condition"] == c], float)
        tr = sum(1 for r in rows if r["condition"] == c and r["truncated"])
        lt_rows.append([mdl, has_eo, blocked, SHORT[c],
                        round(float(np.median(v)), 2), round(float(np.mean(v)), 2),
                        round(float(np.percentile(v, 95)), 2), round(float(np.mean(v > 100)), 3), tr])
write_csv("final_latency_token_audit.csv",
          ["model", "has_execution_order_field", "file_condition_blocked", "cond_short",
           "latency_median_s", "latency_mean_s", "latency_p95_s", "frac_runs_gt_100s", "n_truncated"], lt_rows)

# ================================================================= 8. final_annotation_audit (N/A)
write_csv("final_annotation_audit.csv", ["status", "detail"],
          [["NO_ANNOTATION_DATA",
            "No human-labelled trace dataset exists in the repository. Only a blank "
            "HUMAN_REVIEW_CHECKLIST.txt template (v3_execution_pack) is present. preregistration_v2.md "
            "defers human validation (2 annotators, Cohen's kappa) to a future 'Part 18'. No "
            "inter-annotator agreement can or should be reported. Failure labels F1-F9 are HEURISTIC "
            "classifier outputs (src/failures.py), not human annotations."]])

print("Done -> final_results/")
