# -*- coding: utf-8 -*-
"""Padding-neutrality audit + latency de-confounding, from existing data only.
No model calls. Writes final_results/final_padding_neutrality.csv and
final_results/final_latency_fixed.csv.
"""
import json, os, re, csv, hashlib
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "final_results")
os.makedirs(OUT, exist_ok=True)
P = os.path.join(ROOT, "config", "prompts")

def rd(n): return open(os.path.join(P, n), encoding="utf-8").read().strip()
def words(t): return set(re.findall(r"[a-z0-9]+", t.lower()))
def jac(a, b):
    wa, wb = words(a), words(b)
    return len(wa & wb) / len(wa | wb) if (wa and wb) else 0.0
def sha(t): return hashlib.sha256(t.encode()).hexdigest()[:16]

pre = rd("preamble.txt")
R, S, V = rd("block_retrieve.txt"), rd("block_solve.txt"), rd("block_verify.txt")
PR, PS, PV = rd("pad_retrieve.txt"), rd("pad_solve.txt"), rd("pad_verify.txt")
blocks_all = R + " " + S + " " + V

# ---- load v2 tasks to test padding vs task content ----
tasks = [json.loads(l) for l in open(os.path.join(ROOT, "data", "tasks_v2.jsonl"), encoding="utf-8") if l.strip()]
task_text = " ".join((t.get("question", "") + " " + t.get("context", "")) for t in tasks)
task_words = words(task_text)
answer_words = set()
for t in tasks:
    answer_words |= words(str(t.get("answer", "")))
    for a in (t.get("answer_aliases") or []):
        answer_words |= words(str(a))

print("=" * 80)
print("PADDING NEUTRALITY AUDIT (padding actually used in the 2,250 runs)")
print("=" * 80)
pad_rows = []
BANNED_ROLE = {"retrieve", "retrieval", "solve", "solving", "verify", "verification",
               "responsibility", "reason", "calculate", "lookup", "calculator", "tool"}
for name, pad, blk in [("pad_retrieve", PR, R), ("pad_solve", PS, S), ("pad_verify", PV, V)]:
    pw = words(pad)
    j_block = jac(pad, blk)
    j_all = jac(pad, blocks_all)
    j_task = jac(pad, task_text)
    # does padding leak any task answer tokens? (content fingerprint)
    ans_leak = sorted((pw & answer_words) - {"a", "an", "the", "of", "to", "in", "no", "not", "and", "is", "are", "any"})
    role_leak = sorted(pw & BANNED_ROLE)
    has_digit = bool(re.search(r"\d", pad))
    has_q = "?" in pad
    pad_rows.append([name, len(pad.split()), sha(pad), round(j_block, 3), round(j_all, 3),
                     round(j_task, 3), has_digit, has_q, ";".join(role_leak) or "none",
                     ";".join(ans_leak) or "none"])
    print(f"  {name}: words={len(pad.split())} Jaccard(vs own block)={j_block:.3f} "
          f"(vs all R/S/V)={j_all:.3f} (vs task text)={j_task:.3f} digits={has_digit} '?'={has_q} "
          f"role_leak={role_leak or 'none'} answer_leak={ans_leak or 'none'}")

# runtime length-match (from logged tokens) per model
V2 = {"Qwen": "results/v2/runs.jsonl", "Llama": "results/v2/llama31_8b/runs_clean.jsonl",
      "Mistral": "results/v2/mistral_7b/runs.jsonl"}
CONDS = ["generalist", "specialized", "specialized_length_matched"]
len_rows = []
print("\nRUNTIME per-agent system-prompt length match (A vs C), from logged tokens:")
for mdl, path in V2.items():
    rows = [json.loads(l) for l in open(os.path.join(ROOT, path), encoding="utf-8") if l.strip()]
    per = {c: [] for c in CONDS}
    for r in rows:
        spa = r.get("system_prompt_tokens_per_agent") or []
        if spa: per[r["condition"]].append(spa)
    A = np.mean(np.array(per["generalist"]), axis=0)
    C = np.mean(np.array(per["specialized_length_matched"]), axis=0)
    B = np.mean(np.array(per["specialized"]), axis=0)
    maxdiff = float(np.max(np.abs(A - C)))
    pct = 100 * maxdiff / float(np.mean(A))
    len_rows.append([mdl, [round(float(x), 1) for x in A], [round(float(x), 1) for x in B],
                     [round(float(x), 1) for x in C], round(maxdiff, 1), round(pct, 2)])
    print(f"  {mdl:8} A={[round(float(x)) for x in A]} C={[round(float(x)) for x in C]} "
          f"max|A-C| per-agent={maxdiff:.1f} tok ({pct:.2f}% of A); B(short)={[round(float(x)) for x in B]}")

with open(os.path.join(OUT, "final_padding_neutrality.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["padding_file", "n_words", "sha256_16", "jaccard_vs_own_block",
                "jaccard_vs_all_RSV_blocks", "jaccard_vs_task_text", "contains_digit",
                "contains_question_mark", "role_keyword_leak", "task_answer_token_leak"])
    w.writerows(pad_rows)
    w.writerow([])
    w.writerow(["model", "A_per_agent_tok", "B_per_agent_tok", "C_per_agent_tok",
                "max_abs_A-C_per_agent_tok", "max_abs_A-C_pct_of_A"])
    w.writerows(len_rows)
print("  wrote final_padding_neutrality.csv")

# =====================================================================
print("\n" + "=" * 80)
print("LATENCY DE-CONFOUNDING (execution-order)")
print("=" * 80)
EXEC = {"Qwen": "results/v2/runs.jsonl", "Llama": "results/v2/llama31_8b/runs_clean.jsonl",
        "Mistral": "results/v2/mistral_7b/runs.jsonl"}
lat_rows = []
for mdl, path in EXEC.items():
    rows = [json.loads(l) for l in open(os.path.join(ROOT, path), encoding="utf-8") if l.strip()]
    has_eo = "execution_order" in rows[0]
    df = pd.DataFrame([{"task": r["task_id"], "cond": r["condition"],
                        "lat": r["latency_seconds"], "file_idx": i,
                        "eo": r.get("execution_order", np.nan)} for i, r in enumerate(rows)])
    # execution-order variable: real field if present else file index (= append/execution order)
    df["order"] = df["eo"] if has_eo else df["file_idx"]
    # global drift: Spearman(latency, order)
    rho, p = stats.spearmanr(df["order"], df["lat"])
    # within-condition-block drift (the Qwen confound test): Spearman within each block
    within = {}
    for c in CONDS:
        sub = df[df.cond == c]
        if sub["file_idx"].max() - sub["file_idx"].min() + 1 == len(sub):  # contiguous block
            rr, pp = stats.spearmanr(sub["file_idx"], sub["lat"])
            within[c] = (round(rr, 3), f"{pp:.2g}")
        else:
            within[c] = ("interleaved", "-")
    # naive A-C task-level paired
    piv = df.groupby(["task", "cond"])["lat"].mean().unstack("cond")[CONDS].dropna()
    dnaive = piv["generalist"] - piv["specialized_length_matched"]
    _, p_naive = stats.wilcoxon(piv["generalist"], piv["specialized_length_matched"])
    # de-confounded: OLS latency ~ C(cond) + order, cluster-robust by task  (only meaningful if interleaved)
    if has_eo:
        import statsmodels.api as sm
        d = df.copy()
        d["is_B"] = (d.cond == "specialized").astype(float)
        d["is_C"] = (d.cond == "specialized_length_matched").astype(float)
        X = sm.add_constant(d[["is_B", "is_C", "order"]])
        m = sm.OLS(d["lat"], X).fit(cov_type="cluster", cov_kwds={"groups": d["task"]})
        coef = m.params["is_C"]; pval = m.pvalues["is_C"]
        drift_coef = m.params["order"]
        adj = f"A-C adj coef={-coef:+.2f}s p={pval:.3g}  (order drift={drift_coef:+.4f}s/step)"
    else:
        adj = "NOT DE-CONFOUNDABLE (no execution_order; condition-blocked)"
    print(f"\n{mdl}: has_execution_order={has_eo}")
    print(f"  global Spearman(order, latency) = {rho:+.3f} (p={p:.2g})")
    print(f"  within-block drift (file_idx vs latency): " + "  ".join(f"{c[:4]}={within[c]}" for c in CONDS))
    print(f"  naive A-C paired: median_diff={float(np.median(dnaive)):+.2f}s  p={p_naive:.3g}")
    print(f"  de-confounded:   {adj}")
    lat_rows.append([mdl, has_eo, round(rho, 3), f"{p:.2g}",
                     str({c: within[c][0] for c in CONDS}),
                     round(float(np.median(dnaive)), 2), f"{p_naive:.3g}", adj])

with open(os.path.join(OUT, "final_latency_fixed.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["model", "has_execution_order", "global_spearman_order_vs_latency", "p",
                "within_block_drift_rho(A;B;C)", "naive_A-C_median_s", "naive_A-C_p",
                "deconfounded_A-C"])
    w.writerows(lat_rows)
print("\n  wrote final_latency_fixed.csv")
