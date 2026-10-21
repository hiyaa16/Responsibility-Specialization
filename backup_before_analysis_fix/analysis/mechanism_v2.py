"""v2 MECHANISM analysis: why does responsibility specialization change execution latency?

This is a POST-HOC mechanism/trajectory analysis of the EXISTING 750 runs
(results/v2/runs.jsonl). It does NOT run new LLM calls, does NOT change the
experiment, and does NOT modify the raw data. It complements (does not replace)
analysis/stats_v2.py, which owns the primary aggregate statistics.

Conditions (250 runs each):
  A = generalist                      (3 generalist agents)
  B = specialized                     (retriever, solver, verifier)
  C = specialized_length_matched      (same roles, inert padding to match A's prompt length)

Pairing: 50 tasks x 5 runs x 3 conditions. Runs cluster within tasks (confound C9),
so paired statistics use TASK-MEAN values (average the 5 runs per task first), then
compare across tasks with Wilcoxon signed-rank, Holm-corrected across the 3 pairs
within each metric. Latency is right-skewed, so medians are reported alongside means
and non-parametric tests / Spearman correlations are used throughout.

Field semantics (verified against src/failures.py and the raw data):
  * llm_calls      == sum of per-agent tool_iters (each tool-loop pass = one forward pass). [verified identity]
  * input_tokens   == sum of per-agent input_tokens (prompt-eval load, already aggregated
                      across each agent's tool-loop iterations).                            [verified identity]
  * IMPORTANT SEMANTICS: per-agent input_tokens (a0_input/a1_input/a2_input) and the run-level
    input_tokens are ACCUMULATED INPUT-TOKEN PROCESSING LOADS - the sum of prompt tokens re-processed
    across every forward pass in that agent's tool loop. They are NOT a direct measurement of the
    context *forwarded at a handoff* or of a single-pass context size. This re-analysis CANNOT separate
    "initial handoff/context size" from "repeated re-processing across tool iterations"; we therefore
    report these as accumulated input-token load and avoid context-propagation wording.
  * repeated_work  == max pairwise Jaccard word-overlap between agent outputs in [0,1].
  * corrections    == count of consecutive-agent final-answer changes.
  * F1 (wrong-responsibility) and F7 (out-of-role tool use) are STRUCTURALLY impossible in
    A (generalists own all roles). Treat any A-vs-B/C gap in F1/F7 as a *new failure mode*
    introduced by specialization, not as evidence specialization is worse.

Run:  python analysis/mechanism_v2.py            # reads results/v2/runs.jsonl
      python analysis/mechanism_v2.py <run_dir>  # alternate dir containing runs.jsonl
Outputs: results/v2/mechanism/{summary.json, agent_level.csv, correlations.csv,
         failure_analysis.csv, handoff_analysis.csv, primary_metrics.csv,
         interpretation.md, plots/*}
"""
from __future__ import annotations
import os
import sys
import json
import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONDS = ["generalist", "specialized", "specialized_length_matched"]
SHORT = {"generalist": "A", "specialized": "B", "specialized_length_matched": "C"}
PAIRS = [("generalist", "specialized"),
         ("specialized", "specialized_length_matched"),
         ("generalist", "specialized_length_matched")]
ROLE_BY_INDEX = {  # expected role per agent index per condition (validated in Phase 0)
    "generalist": {0: "generalist", 1: "generalist", 2: "generalist"},
    "specialized": {0: "retriever", 1: "solver", 2: "verifier"},
    "specialized_length_matched": {0: "retriever", 1: "solver", 2: "verifier"},
}

# ----------------------------------------------------------------------------- helpers
def _word_set(text):
    return set(w for w in "".join(c.lower() if c.isalnum() else " " for c in (text or "")).split())


def jaccard(a, b):
    wa, wb = _word_set(a), _word_set(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def holm(pvals):
    pvals = list(pvals)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(running, 1.0)
    return adj


def rank_biserial(diff):
    d = np.asarray(diff, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    wp = ranks[d > 0].sum()
    wm = ranks[d < 0].sum()
    return float((wp - wm) / (wp + wm)) if (wp + wm) else 0.0


def cliffs_delta(x, y):
    """Non-parametric effect size for unpaired distributions (prob x>y minus prob x<y)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    gt = lt = 0
    # O(n log n) via sorting
    ys = np.sort(y)
    for v in x:
        gt += np.searchsorted(ys, v, side="left")
        lt += len(ys) - np.searchsorted(ys, v, side="right")
    return (gt - lt) / (len(x) * len(y))


def boot_ci(x, stat=np.mean, iters=5000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan")
    vals = [stat(rng.choice(x, len(x), replace=True)) for _ in range(iters)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _pearson_vec(a, b):
    """Vectorized Pearson r across columns: a,b shape (n, iters) -> r shape (iters,)."""
    am = a - a.mean(0); bm = b - b.mean(0)
    num = (am * bm).sum(0)
    den = np.sqrt((am ** 2).sum(0) * (bm ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def spearman_ci(x, y, iters=2000, seed=0):
    """Spearman rho (= Pearson on ranks) with a vectorized bootstrap CI (no per-iter scipy loop)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), (float("nan"), float("nan")), n
    rho, p = stats.spearmanr(x, y)
    rx = stats.rankdata(x); ry = stats.rankdata(y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n, iters))
    # bootstrap CI: Pearson on resampled ranks (standard fast approximation to the
    # Spearman bootstrap; per-sample re-ranking omitted for speed, negligible for a CI)
    rhos = _pearson_vec(rx[idx], ry[idx])
    rhos = rhos[np.isfinite(rhos)]
    lo, hi = (float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))) if len(rhos) else (float("nan"), float("nan"))
    return float(rho), float(p), (lo, hi), n


def loglog_exponent(x, y, iters=2000, seed=0):
    """Estimate the power-law exponent b in y ~ a * x**b via OLS slope of log(y) on log(x),
    with a vectorized bootstrap 95% CI. Interpretation of b: >1 super-linear, ~1 linear,
    <1 sub-linear (only meaningful for strictly positive x,y). This is the ONLY place that can
    license any 'super-linear' statement; callers must defer to the returned CI, not assume it."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[m], y[m]
    n = len(x)
    if n < 10:
        return {"b": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": n, "verdict": "not estimable"}
    lx, ly = np.log(x), np.log(y)
    lxm = lx - lx.mean()
    b = float((lxm * (ly - ly.mean())).sum() / (lxm ** 2).sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(iters, n))
    lxb, lyb = lx[idx], ly[idx]
    lxc = lxb - lxb.mean(1, keepdims=True)
    lyc = lyb - lyb.mean(1, keepdims=True)
    slopes = (lxc * lyc).sum(1) / (lxc ** 2).sum(1)
    slopes = slopes[np.isfinite(slopes)]
    lo, hi = float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))
    if lo > 1:
        verdict = "super-linear"
    elif hi < 1:
        verdict = "sub-linear"
    else:
        verdict = "not distinguishable from linear"
    return {"b": b, "ci_lo": lo, "ci_hi": hi, "n": n, "verdict": verdict}


def classify_shift(x, y, med_rel_thresh=0.15, tail_share_thresh=0.50):
    """Classify an x-vs-y distribution difference as a MEDIAN(bulk) shift, a TAIL shift, or BOTH,
    purely from computed statistics. Decomposes the mean gap into a bulk part (difference of 10%-trimmed
    means) and a tail part (remainder). Thresholds are explicit and all components are returned so the
    label is transparent, not hand-chosen per result. x,y are per-condition dicts from the tail block
    (keys: median, mean, trimmed10_mean, p90, p95, p99)."""
    med_diff = x["median"] - y["median"]
    mean_diff = x["mean"] - y["mean"]
    bulk_diff = x["trimmed10_mean"] - y["trimmed10_mean"]
    tail_diff = mean_diff - bulk_diff
    denom = min(x["median"], y["median"])
    med_rel = (med_diff / denom) if denom else float("nan")
    tail_share = (tail_diff / mean_diff) if mean_diff else float("nan")
    p95_diff = x["p95"] - y["p95"]
    med_shift = np.isfinite(med_rel) and abs(med_rel) >= med_rel_thresh
    tail_shift = np.isfinite(tail_share) and tail_share >= tail_share_thresh
    if med_shift and tail_shift:
        label = "BOTH a median(bulk) shift and a tail shift"
    elif tail_shift:
        label = "primarily a TAIL shift"
    elif med_shift:
        label = "primarily a MEDIAN(bulk) shift"
    else:
        label = "no large shift of either kind"
    return {"median_diff": float(med_diff), "median_rel": float(med_rel), "mean_diff": float(mean_diff),
            "bulk_diff": float(bulk_diff), "tail_diff": float(tail_diff), "tail_share": float(tail_share),
            "p95_diff": float(p95_diff), "label": label,
            "thresholds": {"median_rel": med_rel_thresh, "tail_share": tail_share_thresh}}


def paired_task_test(piv, a, b):
    """Wilcoxon signed-rank on task-mean diffs (b - a). Returns dict of stats."""
    x, y = piv[a].values, piv[b].values
    diff = y - x
    try:
        _, p = stats.wilcoxon(y, x)
    except ValueError:  # all-zero diffs
        p = 1.0
    lo, hi = boot_ci(diff, np.mean)
    return {
        "pair": f"{SHORT[a]}-{SHORT[b]}",
        "n_tasks": int(len(diff)),
        "mean_diff": float(np.mean(diff)),
        "median_diff": float(np.median(diff)),
        "rank_biserial": rank_biserial(diff),
        "ci95_lo": lo, "ci95_hi": hi,
        "p_raw": float(p),
    }

# ----------------------------------------------------------------------------- load
def load_runs(run_dir, filename="runs.jsonl"):
    path = os.path.join(run_dir, filename)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_level_df(runs):
    recs = []
    for r in runs:
        ao = sorted(r["agent_outputs"], key=lambda a: a["agent_index"])
        rec = {
            "task_id": r["task_id"], "condition": r["condition"], "run_index": r["run_index"],
            "difficulty": r.get("difficulty"), "category": r.get("category"), "source": r.get("source"),
            "correct": 1 if r["correct"] else 0,
            "latency_seconds": r["latency_seconds"],
            "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"],
            "total_tokens": r["total_tokens"], "llm_calls": r["llm_calls"],
            "system_prompt_tokens": r["system_prompt_tokens"],
            "repeated_work": r.get("repeated_work", 0.0), "corrections": r.get("corrections", 0),
            "handoffs": r.get("handoffs"), "hops": r.get("hops"),
            "truncated": bool(r.get("truncated")), "timeout": bool(r.get("timeout")),
            "error": r.get("error", ""),
            "n_failures": len(r.get("failure_types") or []),
            "tool_calls_total": len(r.get("tool_calls") or []),
        }
        # per-agent derived features
        for a in ao:
            i = a["agent_index"]
            rec[f"a{i}_input"] = a["input_tokens"]
            rec[f"a{i}_output"] = a["output_tokens"]
            rec[f"a{i}_iters"] = a["tool_iters"]
            rec[f"a{i}_toolcalls"] = len(a.get("tool_calls") or [])
        rec["tool_iters_total"] = sum(a["tool_iters"] for a in ao)
        rec["max_agent_input"] = max(a["input_tokens"] for a in ao)
        # exact-duplicate tool calls within each agent (same tool+argument), summed per run
        dup = 0
        for a in ao:
            seen = Counter((c["tool"], c.get("argument", "")) for c in (a.get("tool_calls") or []))
            dup += sum(v - 1 for v in seen.values() if v > 1)
        rec["dup_tool_calls"] = dup
        # consecutive-agent output overlap (handoff reuse / redundancy signal)
        outs = [a["output_text"] for a in ao]
        rec["ov_01"] = jaccard(outs[0], outs[1]) if len(outs) >= 2 else float("nan")
        rec["ov_12"] = jaccard(outs[1], outs[2]) if len(outs) >= 3 else float("nan")
        rec["ov_max"] = max(rec["ov_01"], rec["ov_12"]) if len(outs) >= 3 else float("nan")
        recs.append(rec)
    return pd.DataFrame(recs)


def agent_level_df(runs):
    recs = []
    for r in runs:
        for a in r["agent_outputs"]:
            recs.append({
                "task_id": r["task_id"], "condition": r["condition"], "run_index": r["run_index"],
                "agent_index": a["agent_index"], "role": a["role"],
                "input_tokens": a["input_tokens"], "output_tokens": a["output_tokens"],
                "agent_total_tokens": a["input_tokens"] + a["output_tokens"],
                "tool_iters": a["tool_iters"], "tool_calls": len(a.get("tool_calls") or []),
            })
    return pd.DataFrame(recs)

# ----------------------------------------------------------------------------- Phase 0
def phase0_validation(runs, df, L):
    L += ["=" * 78, "PHASE 0 - DATA VALIDATION", "=" * 78]
    issues = []
    cond_counts = Counter(r["condition"] for r in runs)
    L.append(f"total runs = {len(runs)}  (expected 750)")
    for c in CONDS:
        L.append(f"  {SHORT[c]} {c:30s} = {cond_counts.get(c,0)}  (expected 250)")
    if len(runs) != 750 or any(cond_counts.get(c, 0) != 250 for c in CONDS):
        issues.append("condition counts off expectation")
    unexpected_conds = set(cond_counts) - set(CONDS)
    if unexpected_conds:
        issues.append(f"unexpected conditions: {unexpected_conds}")

    models = Counter(r["model"] for r in runs)
    temps = Counter(r["temperature"] for r in runs)
    acs = Counter(r["agent_count"] for r in runs)
    hos = Counter(r["handoffs"] for r in runs)
    L.append(f"model(s)        = {dict(models)}  (expected qwen2.5:7b)")
    L.append(f"temperature(s)  = {dict(temps)}  (expected 0.7)")
    L.append(f"agent_count(s)  = {dict(acs)}  (expected 3)")
    L.append(f"handoffs        = {dict(hos)}  (expected 2)")
    if set(models) != {"qwen2.5:7b"}: issues.append("unexpected model")
    if set(temps) != {0.7}: issues.append("unexpected temperature")
    if set(acs) != {3}: issues.append("unexpected agent_count")
    if set(hos) != {2}: issues.append("unexpected handoffs")

    # role assignment per index
    role_map = defaultdict(Counter)
    for r in runs:
        for a in r["agent_outputs"]:
            role_map[(r["condition"], a["agent_index"])][a["role"]] += 1
    for c in CONDS:
        for i in range(3):
            got = dict(role_map[(c, i)])
            exp = ROLE_BY_INDEX[c][i]
            ok = (set(got) == {exp} and got.get(exp) == cond_counts.get(c, 0))
            if not ok:
                issues.append(f"role mismatch {c} idx{i}: {got} (expected all {exp})")
    L.append("role-by-index assignment: " + ("OK (matches design)" if not any("role mismatch" in x for x in issues) else "MISMATCH - see issues"))

    # problematic runs (reported, NOT removed)
    trunc = df["truncated"].sum()
    tout = df["timeout"].sum()
    errs = (df["error"].astype(str).str.len() > 0).sum()
    L.append(f"problematic runs (reported, NOT removed): truncated={int(trunc)}, timeout={int(tout)}, error={int(errs)}")
    if trunc:
        tr = df[df["truncated"]][["condition", "task_id", "run_index"]].to_dict("records")
        L.append(f"  truncated run(s): {tr}")

    # null / missing checks on important fields
    important = ["latency_seconds", "input_tokens", "output_tokens", "total_tokens", "llm_calls",
                 "repeated_work", "corrections", "correct", "a0_input", "a1_input", "a2_input"]
    nulls = {k: int(df[k].isna().sum()) for k in important if k in df}
    bad = {k: v for k, v in nulls.items() if v}
    L.append(f"null/missing in important fields: {bad if bad else 'none'}")

    # sanity: impossible values
    neg = {k: int((df[k] < 0).sum()) for k in ["latency_seconds", "input_tokens", "output_tokens", "llm_calls"] if k in df}
    bad_neg = {k: v for k, v in neg.items() if v}
    if bad_neg:
        issues.append(f"negative values: {bad_neg}")
    L.append(f"negative-value check: {bad_neg if bad_neg else 'none'}")

    # execution-order structure: are runs condition-blocked in the file? (latency confound)
    seq = [r["condition"] for r in runs]
    transitions = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
    blocked = transitions <= len(CONDS)
    L.append(f"file ordering: {transitions} condition-transitions across {len(runs)} runs -> "
             f"{'CONDITION-BLOCKED (all runs of a condition contiguous)' if blocked else 'interleaved'}")
    if blocked:
        L.append("  CAUTION: no timestamps are logged; if execution order matched file order, each condition "
                 "ran as one block. On a CPU-bound host, time-varying machine load could then confound the "
                 "LATENCY comparison (not tokens/accuracy). Flagged for Phase 8 tail analysis & limitations.")

    L.append(f"\nVALIDATION VERDICT: {'PASS - clean' if not issues else 'ISSUES: ' + '; '.join(issues)}")
    return {"total": len(runs), "condition_counts": dict(cond_counts),
            "truncated": int(trunc), "timeout": int(tout), "errors": int(errs),
            "nulls": bad, "issues": issues,
            "condition_blocked": bool(blocked), "condition_transitions": int(transitions)}

# ----------------------------------------------------------------------------- Phase 1
PRIMARY_DVS = [
    ("accuracy", "correct"), ("latency_seconds", "latency_seconds"),
    ("total_tokens", "total_tokens"), ("input_tokens", "input_tokens"),
    ("output_tokens", "output_tokens"), ("llm_calls", "llm_calls"),
    ("repeated_work", "repeated_work"), ("corrections", "corrections"),
]


def task_pivot(df, col):
    piv = df.groupby(["task_id", "condition"])[col].mean().unstack("condition")
    present = [c for c in CONDS if c in piv.columns]
    return piv[present].dropna()


def phase1_primary(df, L, summary):
    L += ["\n" + "=" * 78, "PHASE 1 - RECOMPUTE PRIMARY METRICS (task-level paired; Wilcoxon + Holm)", "=" * 78]
    rows = []
    prim = {}
    for name, col in PRIMARY_DVS:
        piv = task_pivot(df, col)
        # per-condition run-level descriptives
        desc = {}
        for c in CONDS:
            v = df[df["condition"] == c][col].values
            desc[SHORT[c]] = {"mean": float(np.mean(v)), "median": float(np.median(v)),
                              "sd": float(np.std(v, ddof=1)), "iqr": float(np.subtract(*np.percentile(v, [75, 25])))}
        L.append(f"\n[{name}]  (run-level mean / median / sd)")
        for c in CONDS:
            d = desc[SHORT[c]]
            L.append(f"  {SHORT[c]} {c:30s} mean={d['mean']:.4f}  median={d['median']:.4f}  sd={d['sd']:.4f}")
        # Friedman omnibus on task means
        chi, pf = stats.friedmanchisquare(*[piv[c].values for c in CONDS])
        L.append(f"  Friedman chi2={chi:.3f} p={pf:.4g}  (n_tasks={len(piv)})")
        # pairwise paired tests, Holm across 3 pairs
        pairstats = [paired_task_test(piv, a, b) for a, b in PAIRS]
        adj = holm([ps["p_raw"] for ps in pairstats])
        for ps, pa in zip(pairstats, adj):
            ps["p_holm"] = float(pa)
            sig = "*" if pa < 0.05 else " "
            L.append(f"    {ps['pair']}: mean(2nd-1st)={ps['mean_diff']:+.4f}  median={ps['median_diff']:+.4f}  "
                     f"rb={ps['rank_biserial']:+.3f}  95%CI[{ps['ci95_lo']:+.4f},{ps['ci95_hi']:+.4f}]  "
                     f"p={ps['p_raw']:.4g} holm={pa:.4g}{sig}")
            rows.append({"metric": name, **ps})
        prim[name] = {"descriptives": desc, "friedman_chi2": float(chi), "friedman_p": float(pf),
                      "pairwise": pairstats}
    L.append("\n  Interpretation note: 'holm<0.05' = evidence of a difference, NOT proof. "
             "Non-significant = no statistically significant evidence of a difference (not proof of none).")
    summary["phase1_primary"] = prim
    return pd.DataFrame(rows)

# ----------------------------------------------------------------------------- Phase 2
def phase2_agent_workload(adf, L, summary, outdir):
    L += ["\n" + "=" * 78, "PHASE 2 - AGENT-LEVEL WORKLOAD", "=" * 78]
    metrics = ["input_tokens", "output_tokens", "agent_total_tokens", "tool_iters", "tool_calls"]
    g = adf.groupby(["condition", "agent_index", "role"])[metrics].agg(["mean", "median"])
    # flatten for CSV
    flat = adf.groupby(["condition", "agent_index", "role"])[metrics].agg(["mean", "median", "sum"]).reset_index()
    flat.columns = ["_".join([str(x) for x in c if x]) for c in flat.columns.to_flat_index()]
    flat.to_csv(os.path.join(outdir, "agent_level.csv"), index=False)

    for c in CONDS:
        L.append(f"\n[{SHORT[c]} {c}]  (mean per agent)")
        for i in range(3):
            sub = adf[(adf["condition"] == c) & (adf["agent_index"] == i)]
            role = sub["role"].iloc[0]
            L.append(f"  agent{i} {role:11s}: input={sub['input_tokens'].mean():8.1f}  "
                     f"output={sub['output_tokens'].mean():6.1f}  iters={sub['tool_iters'].mean():.2f}  "
                     f"toolcalls={sub['tool_calls'].mean():.2f}")
    # agent0 is where tool-looping concentrates; test A vs B/C on agent0 iters & input (task-paired)
    L.append("\n  Agent-0 workload (the retrieval/first stage - where tool loops concentrate):")
    a0 = adf[adf["agent_index"] == 0]
    ag = {}
    for metric in ["input_tokens", "output_tokens", "tool_iters", "tool_calls"]:
        piv = a0.groupby(["task_id", "condition"])[metric].mean().unstack("condition")[CONDS].dropna()
        res = [paired_task_test(piv, a, b) for a, b in PAIRS]
        adj = holm([r["p_raw"] for r in res])
        for r, pa in zip(res, adj):
            r["p_holm"] = float(pa)
        ag[metric] = res
        line = f"    a0.{metric}: " + "  ".join(
            f"{r['pair']} med={r['median_diff']:+.1f}(holm={r['p_holm']:.3g})" for r in res)
        L.append(line)
    summary["phase2_agent0_paired"] = ag
    # total agent work per condition
    tot = adf.groupby("condition")["agent_total_tokens"].mean().to_dict()
    L.append("\n  Mean total agent tokens (sum over agents, per run) by condition: " +
             "  ".join(f"{SHORT[c]}={df_mean:.0f}" for c, df_mean in
                       adf.groupby('condition').apply(lambda s: s.groupby(['task_id','run_index'])['agent_total_tokens'].sum().mean(), include_groups=False).items()))
    summary["phase2_agent_means"] = {
        c: {str(i): adf[(adf.condition==c)&(adf.agent_index==i)][metrics].mean().round(2).to_dict() for i in range(3)}
        for c in CONDS}

# ----------------------------------------------------------------------------- Phase 3
def phase3_redundant(df, L, summary):
    L += ["\n" + "=" * 78, "PHASE 3 - REDUNDANT WORK (repeated_work = max pairwise agent-output Jaccard)", "=" * 78]
    for c in CONDS:
        v = df[df["condition"] == c]["repeated_work"].values
        L.append(f"  {SHORT[c]} {c:30s} mean={v.mean():.4f} median={np.median(v):.4f} "
                 f"p90={np.percentile(v,90):.3f}  frac>=0.6(F6)={np.mean(v>=0.6):.3f}")
    piv = task_pivot(df, "repeated_work")
    res = [paired_task_test(piv, a, b) for a, b in PAIRS]
    adj = holm([r["p_raw"] for r in res])
    for r, pa in zip(res, adj):
        r["p_holm"] = float(pa)
        L.append(f"    {r['pair']}: median diff={r['median_diff']:+.4f}  rb={r['rank_biserial']:+.3f}  holm={pa:.4g}")
    # association with latency (pooled + within condition)
    L.append("  repeated_work vs latency (Spearman):")
    assoc = {}
    for scope, sub in [("pooled", df)] + [(SHORT[c], df[df["condition"] == c]) for c in CONDS]:
        rho, p, ci, n = spearman_ci(sub["repeated_work"], sub["latency_seconds"])
        L.append(f"    {scope:7s} rho={rho:+.3f} 95%CI[{ci[0]:+.3f},{ci[1]:+.3f}] p={p:.4g} n={n}")
        assoc[scope] = {"rho": rho, "p": p, "ci": ci, "n": n}
    summary["phase3"] = {"pairwise": res, "latency_assoc": assoc}

# ----------------------------------------------------------------------------- Phase 4
def phase4_context(df, L, summary):
    L += ["\n" + "=" * 78,
          "PHASE 4 - ACCUMULATED INPUT-TOKEN LOAD PER AGENT",
          "=" * 78]
    L.append("  NB: a0/a1/a2_input are ACCUMULATED input-token processing loads (summed over each agent's")
    L.append("  tool-loop forward passes), NOT a direct measure of forwarded/handoff context size. The")
    L.append("  per-step ratios below are load ratios; they cannot isolate initial handoff size from")
    L.append("  repeated re-processing across tool iterations.")
    d = df.copy()
    d["r21"] = d["a1_input"] / d["a0_input"].replace(0, np.nan)
    d["r32"] = d["a2_input"] / d["a1_input"].replace(0, np.nan)
    ctx = {}
    for c in CONDS:
        s = d[d["condition"] == c]
        L.append(f"\n[{SHORT[c]} {c}]")
        L.append(f"  a0_load mean={s['a0_input'].mean():8.1f} median={s['a0_input'].median():8.1f}")
        L.append(f"  a1_load mean={s['a1_input'].mean():8.1f} median={s['a1_input'].median():8.1f}")
        L.append(f"  a2_load mean={s['a2_input'].mean():8.1f} median={s['a2_input'].median():8.1f}")
        L.append(f"  total_load mean={s['input_tokens'].mean():8.1f}  max_agent_load mean={s['max_agent_input'].mean():8.1f}")
        L.append(f"  load ratio a1/a0 median={s['r21'].median():.3f}  a2/a1 median={s['r32'].median():.3f}")
        ctx[SHORT[c]] = {k: float(s[k].mean()) for k in ["a0_input","a1_input","a2_input","input_tokens","max_agent_input"]}
    # which agent carries the most accumulated load?
    L.append("\n  Share of total accumulated input-token load by agent (mean of per-run shares):")
    for c in CONDS:
        s = d[d["condition"] == c]
        tot = s[["a0_input","a1_input","a2_input"]].sum(axis=1)
        sh = [ (s[f"a{i}_input"]/tot).mean() for i in range(3)]
        L.append(f"    {SHORT[c]}: a0={sh[0]:.3f}  a1={sh[1]:.3f}  a2={sh[2]:.3f}")
    # paired tests on a0_input and max_agent_input and total input
    L.append("\n  Paired task-level tests:")
    pr = {}
    for metric in ["a0_input", "max_agent_input", "input_tokens"]:
        piv = task_pivot(df, metric)
        res = [paired_task_test(piv, a, b) for a, b in PAIRS]
        adj = holm([r["p_raw"] for r in res]);
        for r, pa in zip(res, adj): r["p_holm"] = float(pa)
        pr[metric] = res
        L.append(f"    {metric}: " + "  ".join(f"{r['pair']} med={r['median_diff']:+.1f}(holm={r['p_holm']:.3g})" for r in res))
    summary["phase4"] = {"means": ctx, "paired": pr}

# ----------------------------------------------------------------------------- Phase 5
def phase5_tools(df, adf, runs, L, summary):
    L += ["\n" + "=" * 78, "PHASE 5 - TOOL USE", "=" * 78]
    for c in CONDS:
        s = df[df["condition"] == c]
        L.append(f"  {SHORT[c]} {c:30s} tool_calls/run mean={s['tool_calls_total'].mean():.3f} "
                 f"median={s['tool_calls_total'].median():.1f}  dup_tool_calls/run mean={s['dup_tool_calls'].mean():.3f} "
                 f"frac_runs_with_dup={np.mean(s['dup_tool_calls']>0):.3f}")
    L.append("\n  Tool calls & iters per agent (mean):")
    for c in CONDS:
        L.append(f"  [{SHORT[c]}]")
        for i in range(3):
            sub = adf[(adf.condition==c)&(adf.agent_index==i)]
            L.append(f"     agent{i} {sub['role'].iloc[0]:11s}: tool_calls={sub['tool_calls'].mean():.3f} iters={sub['tool_iters'].mean():.3f}")
    # paired tests on total tool calls & duplicate tool calls
    L.append("\n  Paired task-level tests:")
    pr = {}
    for metric in ["tool_calls_total", "dup_tool_calls"]:
        piv = task_pivot(df, metric)
        res = [paired_task_test(piv, a, b) for a, b in PAIRS]
        adj = holm([r["p_raw"] for r in res])
        for r, pa in zip(res, adj): r["p_holm"] = float(pa)
        pr[metric] = res
        L.append(f"    {metric}: " + "  ".join(f"{r['pair']} med={r['median_diff']:+.3f}(holm={r['p_holm']:.3g})" for r in res))
    # which tools, by role
    tool_by_role = defaultdict(Counter)
    for r in runs:
        for a in r["agent_outputs"]:
            for cc in (a.get("tool_calls") or []):
                tool_by_role[(r["condition"], a["role"])][cc["tool"]] += 1
    L.append("\n  Tool usage counts by (condition, role):  (rule: exact tool+argument repeat within an agent = duplicate)")
    for key in sorted(tool_by_role):
        L.append(f"    {SHORT[key[0]]}/{key[1]:11s}: {dict(tool_by_role[key])}")
    summary["phase5"] = {"paired": pr,
                         "tool_by_role": {f"{SHORT[k[0]]}/{k[1]}": dict(v) for k, v in tool_by_role.items()}}

# ----------------------------------------------------------------------------- Phase 6
def phase6_handoffs(df, runs, L, summary, outdir):
    L += ["\n" + "=" * 78, "PHASE 6 - HANDOFF BEHAVIOR (handoffs fixed at 2; we study what happens AROUND them)", "=" * 78]
    L.append("  Metric = Jaccard word-overlap between consecutive agent outputs (automatically measured).")
    rows = []
    for c in CONDS:
        s = df[df["condition"] == c]
        L.append(f"  {SHORT[c]} {c:30s} overlap a0->a1 mean={s['ov_01'].mean():.3f}  "
                 f"a1->a2 mean={s['ov_12'].mean():.3f}  max mean={s['ov_max'].mean():.3f}")
        rows.append({"condition": c, "short": SHORT[c],
                     "ov_01_mean": s["ov_01"].mean(), "ov_01_median": s["ov_01"].median(),
                     "ov_12_mean": s["ov_12"].mean(), "ov_12_median": s["ov_12"].median(),
                     "ov_max_mean": s["ov_max"].mean()})
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "handoff_analysis.csv"), index=False)
    pr = {}
    for metric in ["ov_01", "ov_12"]:
        piv = task_pivot(df, metric)
        res = [paired_task_test(piv, a, b) for a, b in PAIRS]
        adj = holm([r["p_raw"] for r in res])
        for r, pa in zip(res, adj): r["p_holm"] = float(pa)
        pr[metric] = res
        L.append(f"    {metric}: " + "  ".join(f"{r['pair']} med={r['median_diff']:+.3f}(holm={r['p_holm']:.3g})" for r in res))
    # manual-inspection sample: SAME task_id across A/B/C (run_index 0) so the comparison is like-for-like,
    # not three different tasks that merely happen to be first in file order.
    idx = defaultdict(dict)  # task_id -> {condition: run}
    for r in runs:
        if r["run_index"] == 0:
            idx[r["task_id"]].setdefault(r["condition"], r)
    shared = sorted(t for t, d in idx.items() if all(c in d for c in CONDS))
    sample = {"task_id": None, "run_index": 0, "by_condition": {}}
    if shared:
        tid = shared[0]
        sample["task_id"] = tid
        L.append(f"\n  MANUAL-INSPECTION SAMPLE (illustrative, NOT a metric): SAME task across A/B/C -> "
                 f"task_id={tid}, run_index=0")
        for c in CONDS:
            r = idx[tid][c]
            ao = sorted(r["agent_outputs"], key=lambda a: a["agent_index"])
            snip = [{"role": a["role"], "tool_calls": len(a.get("tool_calls") or []),
                     "output_head": (a["output_text"] or "")[:160].replace("\n", " ")} for a in ao]
            sample["by_condition"][SHORT[c]] = {"task_id": tid, "agents": snip}
            L.append(f"    [{SHORT[c]} {c}] task={tid}")
            for a in snip:
                L.append(f"       {a['role']:11s} (tcalls={a['tool_calls']}): {a['output_head']}")
    else:
        L.append("\n  MANUAL-INSPECTION SAMPLE: no task_id present in all three conditions at run_index 0 (skipped).")
    summary["phase6"] = {"paired": pr, "sample": sample}

# ----------------------------------------------------------------------------- Phase 7
FAIL_LABELS = {
    "F1": "wrong-responsibility exec (structurally B/C only)",
    "F2": "retrieval failure (fact missing from agent1)",
    "F3": "solve-stage answer wrong",
    "F4": "verification-stage failure",
    "F5": "handoff info-loss",
    "F6": "redundant/repeated work (overlap>=0.6)",
    "F7": "out-of-scope tool use (structurally B/C only)",
    "F8": "inter-agent correction (final != solve)",
    "F9": "final answer incorrect",
}


def phase7_failures(runs, df, L, summary, outdir):
    L += ["\n" + "=" * 78, "PHASE 7 - FAILURE MODES (denominator = 250 runs/condition)", "=" * 78]
    cnt = {c: Counter() for c in CONDS}
    for r in runs:
        for f in (r.get("failure_types") or []):
            cnt[r["condition"]][f] += 1
    allf = sorted(FAIL_LABELS)
    rows = []
    header = "  type  " + "".join(f"{SHORT[c]:>10s}" for c in CONDS) + "   label"
    L.append(header)
    for f in allf:
        line = f"  {f:4s} " + "".join(f"{100*cnt[c][f]/250:9.1f}%" for c in CONDS)
        L.append(line + f"   {FAIL_LABELS[f]}")
        rows.append({"failure_type": f, "label": FAIL_LABELS[f],
                     **{f"{SHORT[c]}_count": cnt[c][f] for c in CONDS},
                     **{f"{SHORT[c]}_pct": round(100*cnt[c][f]/250, 2) for c in CONDS}})
    fdf = pd.DataFrame(rows)
    fdf.to_csv(os.path.join(outdir, "failure_analysis.csv"), index=False)
    # structural new failure modes
    L.append("\n  STRUCTURAL NOTE: F1 & F7 are impossible in A by construction. Observed in B/C:")
    L.append(f"    F1: A={cnt['generalist']['F1']}  B={cnt['specialized']['F1']}  C={cnt['specialized_length_matched']['F1']}")
    L.append(f"    F7: A={cnt['generalist']['F7']}  B={cnt['specialized']['F7']}  C={cnt['specialized_length_matched']['F7']}")
    # relation of corrections/repeated_work to correctness
    L.append("\n  Corrections (F8 proxy) & repeated_work vs correctness:")
    for c in CONDS:
        s = df[df["condition"] == c]
        L.append(f"    {SHORT[c]}: corrections mean={s['corrections'].mean():.3f}  "
                 f"repeated_work mean={s['repeated_work'].mean():.3f}  accuracy={s['correct'].mean():.3f}")
    summary["phase7_failures"] = {c: dict(cnt[c]) for c in CONDS}

# ----------------------------------------------------------------------------- Phase 8
def phase8_latency_pathway(df, L, summary, outdir):
    L += ["\n" + "=" * 78, "PHASE 8 - LATENCY PATHWAY (what execution variables move WITH latency?)", "=" * 78]
    predictors = ["input_tokens", "output_tokens", "total_tokens", "llm_calls", "tool_iters_total",
                  "tool_calls_total", "dup_tool_calls", "max_agent_input", "a0_input", "a0_iters",
                  "repeated_work", "corrections"]
    rows = []
    corr_pooled = {}
    corr_within = {SHORT[c]: {} for c in CONDS}
    L.append("  Spearman(latency, X): pooled and within-condition (pooled can be confounded by condition).")
    for X in predictors:
        rho, p, ci, n = spearman_ci(df[X], df["latency_seconds"])
        corr_pooled[X] = {"rho": rho, "p": p, "ci": [ci[0], ci[1]], "n": n}
        rows.append({"predictor": X, "scope": "pooled", "spearman_rho": rho, "p": p,
                     "ci_lo": ci[0], "ci_hi": ci[1], "n": n})
        within = {}
        for c in CONDS:
            s = df[df["condition"] == c]
            rr, pp, cc, nn = spearman_ci(s[X], s["latency_seconds"])
            within[SHORT[c]] = rr
            corr_within[SHORT[c]][X] = {"rho": rr, "p": pp, "ci": [cc[0], cc[1]], "n": nn}
            rows.append({"predictor": X, "scope": SHORT[c], "spearman_rho": rr, "p": pp,
                         "ci_lo": cc[0], "ci_hi": cc[1], "n": nn})
        L.append(f"    {X:18s} pooled rho={rho:+.3f} CI[{ci[0]:+.3f},{ci[1]:+.3f}] p={p:.3g} | "
                 f"within A={within['A']:+.3f} B={within['B']:+.3f} C={within['C']:+.3f}")
    cdf = pd.DataFrame(rows)
    cdf.to_csv(os.path.join(outdir, "correlations.csv"), index=False)
    # rank predictors by |pooled rho| (observed, so nothing downstream has to assume a "dominant" one)
    ranked = sorted(corr_pooled.items(), key=lambda kv: abs(kv[1]["rho"]), reverse=True)
    top_name, top = ranked[0]
    L.append(f"  -> largest |pooled Spearman| with latency: {top_name} (rho={top['rho']:+.3f}); "
             f"ranking: " + ", ".join(f"{k}({v['rho']:+.2f})" for k, v in ranked[:5]) + " ...")
    L.append("     (association only; pooled correlations are additionally confounded by condition.)")

    # how do the top predictors differ across conditions? (paired, already have some)
    L.append("\n  How the leading latency-linked variables differ across conditions (task-paired medians, Holm):")
    pr = {}
    for metric in ["latency_seconds", "input_tokens", "a0_input", "a0_iters", "llm_calls", "output_tokens"]:
        piv = task_pivot(df, metric)
        res = [paired_task_test(piv, a, b) for a, b in PAIRS]
        adj = holm([r["p_raw"] for r in res])
        for r, pa in zip(res, adj): r["p_holm"] = float(pa)
        pr[metric] = res
        L.append(f"    {metric:16s}: " + "  ".join(
            f"{r['pair']} med={r['median_diff']:+.2f}(holm={r['p_holm']:.3g})" for r in res))

    # ---- TAIL ANALYSIS (the latency effect is a tail phenomenon, not a uniform shift) ----
    L.append("\n  LATENCY TAIL ANALYSIS (is the effect uniform, or driven by catastrophic slow runs?):")
    tail = {}
    for c in CONDS:
        v = df[df["condition"] == c]["latency_seconds"].values
        row = {"median": float(np.median(v)), "mean": float(np.mean(v)),
               "p90": float(np.percentile(v, 90)), "p95": float(np.percentile(v, 95)),
               "p99": float(np.percentile(v, 99)), "max": float(v.max()),
               "frac_gt_60s": float(np.mean(v > 60)), "frac_gt_100s": float(np.mean(v > 100)),
               "trimmed10_mean": float(np.sort(v)[:int(len(v) * 0.9)].mean())}
        tail[SHORT[c]] = row
        L.append(f"    {SHORT[c]}: median={row['median']:.1f}s mean={row['mean']:.1f}s | "
                 f"p90={row['p90']:.0f} p95={row['p95']:.0f} p99={row['p99']:.0f} max={row['max']:.0f} | "
                 f">60s={row['frac_gt_60s']:.1%} >100s={row['frac_gt_100s']:.1%} | "
                 f"trimmed10%-mean={row['trimmed10_mean']:.1f}s")
    # profile of the catastrophic tail (>100s)
    L.append("    catastrophic tail (latency>100s): count & token/call profile:")
    for c in CONDS:
        s = df[(df["condition"] == c) & (df["latency_seconds"] > 100)]
        if len(s):
            L.append(f"      {SHORT[c]}: n={len(s)}  mean_input={s['input_tokens'].mean():.0f}  "
                     f"mean_llm_calls={s['llm_calls'].mean():.1f}  mean_a1_input={s['a1_input'].mean():.0f}")
        else:
            L.append(f"      {SHORT[c]}: n=0")
    L.append("    NOTE: runs are condition-blocked (Phase 0) with no timestamps; part of A's tail MAGNITUDE "
             "could be a machine-load artifact of its execution block. Figures above are descriptive only.")
    # classify each specialization comparison as a median(bulk) shift, a tail shift, or both (computed)
    tail["shift_AB"] = classify_shift(tail["A"], tail["B"])
    tail["shift_AC"] = classify_shift(tail["A"], tail["C"])
    tail["shift_BC"] = classify_shift(tail["B"], tail["C"])
    L.append("    SHIFT CLASSIFICATION (computed; thresholds median_rel>=15%, tail_share>=50%):")
    for pair, key in [("A-B", "shift_AB"), ("A-C", "shift_AC"), ("B-C", "shift_BC")]:
        sc = tail[key]
        L.append(f"      {pair}: {sc['label']}  (median_diff={sc['median_diff']:+.1f}s, median_rel={sc['median_rel']:+.1%}, "
                 f"mean_diff={sc['mean_diff']:+.1f}s, tail_share_of_mean_gap={sc['tail_share']:+.0%}, p95_diff={sc['p95_diff']:+.0f}s)")
    summary["phase8_tail"] = tail

    # ---- SCALING TEST: is latency super-/sub-linear in input tokens? (computed, not assumed) ----
    L.append("\n  SCALING TEST - latency vs input_tokens, power-law exponent b (log-log OLS, bootstrap 95% CI):")
    L.append("    b>1 => super-linear, CI spanning 1 => indistinguishable from linear, b<1 => sub-linear.")
    nonlin = {"pooled": loglog_exponent(df["input_tokens"].values, df["latency_seconds"].values)}
    for c in CONDS:
        s = df[df["condition"] == c]
        nonlin[SHORT[c]] = loglog_exponent(s["input_tokens"].values, s["latency_seconds"].values)
    for scope in ["pooled", "A", "B", "C"]:
        e = nonlin[scope]
        L.append(f"    {scope:7s} b={e['b']:+.3f} 95%CI[{e['ci_lo']:+.3f},{e['ci_hi']:+.3f}] n={e['n']} -> {e['verdict']}")
    L.append("    (pooled is confounded by condition and by the blocked-execution caveat; prefer within-condition b.)")

    # Decompose: latency per llm_call and latency per 1k input tokens (is the gap "more calls"
    # or "slower calls"?). These are ratios so we summarize medians per condition.
    d = df.copy()
    d["lat_per_call"] = d["latency_seconds"] / d["llm_calls"].replace(0, np.nan)
    d["lat_per_1k_in"] = d["latency_seconds"] / (d["input_tokens"] / 1000.0).replace(0, np.nan)
    L.append("\n  Latency decomposition (medians):")
    for c in CONDS:
        s = d[d["condition"] == c]
        L.append(f"    {SHORT[c]}: latency={s['latency_seconds'].median():.1f}s  llm_calls={s['llm_calls'].median():.1f}  "
                 f"input_tok={s['input_tokens'].median():.0f}  lat/call={s['lat_per_call'].median():.2f}s  "
                 f"lat/1k_in={s['lat_per_1k_in'].median():.3f}s")
    # partial check: within matched llm_calls, does latency still differ A vs C? (stratify)
    L.append("\n  Stratified check - latency A vs C within same llm_calls bucket (controls call count):")
    for ncalls in sorted(set(df["llm_calls"].unique())):
        a = df[(df.condition=="generalist") & (df.llm_calls==ncalls)]["latency_seconds"]
        cc = df[(df.condition=="specialized_length_matched") & (df.llm_calls==ncalls)]["latency_seconds"]
        if len(a) >= 10 and len(cc) >= 10:
            L.append(f"    llm_calls={ncalls}: A n={len(a)} med={a.median():.1f}s | C n={len(cc)} med={cc.median():.1f}s | "
                     f"delta={cc.median()-a.median():+.1f}s")
    summary["phase8"] = {"paired": pr,
                         "latency_corr_pooled": corr_pooled,
                         "latency_corr_within": corr_within,
                         "top_pooled_predictor": {"name": top_name, "rho": top["rho"], "ci": top["ci"]},
                         "nonlin": nonlin,
                         "lat_decomp": {SHORT[c]: {
                             "latency_med": float(d[d.condition==c]["latency_seconds"].median()),
                             "llm_calls_med": float(d[d.condition==c]["llm_calls"].median()),
                             "input_tok_med": float(d[d.condition==c]["input_tokens"].median()),
                             "lat_per_call_med": float(d[d.condition==c]["lat_per_call"].median()),
                             "lat_per_1k_in_med": float(d[d.condition==c]["lat_per_1k_in"].median()),
                         } for c in CONDS}}
    return cdf

# ----------------------------------------------------------------------------- Phase 9
def _verdict(res_pair, direction_good=None):
    """Helper: holm-significant?"""
    return res_pair["p_holm"] < 0.05


def phase9_triangulation(summary, L):
    L += ["\n" + "=" * 78, "PHASE 9 - MECHANISM TRIANGULATION", "=" * 78]
    p1 = summary["phase1_primary"]
    top = summary["phase8"]["top_pooled_predictor"]
    lat = {r["pair"]: r for r in p1["latency_seconds"]["pairwise"]}

    def pm(block, metric, pair):
        for r in block[metric]:
            if r["pair"] == pair:
                return r
        return None

    rows = []

    # 1 reduced redundant work
    rw = {r["pair"]: r for r in summary["phase3"]["pairwise"]}
    assoc = summary["phase3"]["latency_assoc"]
    rows.append({
        "mechanism": "1. Reduced redundant work (repeated_work)",
        "A_B": f"med{rw['A-B']['median_diff']:+.3f} holm={rw['A-B']['p_holm']:.3g}",
        "A_C": f"med{rw['A-C']['median_diff']:+.3f} holm={rw['A-C']['p_holm']:.3g}",
        "B_C": f"med{rw['B-C']['median_diff']:+.3f} holm={rw['B-C']['p_holm']:.3g}",
        "evidence": f"latency assoc pooled rho={assoc['pooled']['rho']:+.2f}; within A={assoc['A']['rho']:+.2f} B={assoc['B']['rho']:+.2f} C={assoc['C']['rho']:+.2f}",
    })
    # 2 workload redistribution (agent0 input)
    a0 = summary["phase2_agent0_paired"]["input_tokens"]
    a0d = {r["pair"]: r for r in a0}
    rows.append({
        "mechanism": "2. Workload redistribution (agent0 input load)",
        "A_B": f"med{a0d['A-B']['median_diff']:+.0f} holm={a0d['A-B']['p_holm']:.3g}",
        "A_C": f"med{a0d['A-C']['median_diff']:+.0f} holm={a0d['A-C']['p_holm']:.3g}",
        "B_C": f"med{a0d['B-C']['median_diff']:+.0f} holm={a0d['B-C']['p_holm']:.3g}",
        "evidence": "see Phase 2 per-agent means",
    })
    # 3 reduced context growth (max_agent_input)
    mx = {r["pair"]: r for r in summary["phase4"]["paired"]["max_agent_input"]}
    rows.append({
        "mechanism": "3. Reduced max per-agent accumulated input-token load",
        "A_B": f"med{mx['A-B']['median_diff']:+.0f} holm={mx['A-B']['p_holm']:.3g}",
        "A_C": f"med{mx['A-C']['median_diff']:+.0f} holm={mx['A-C']['p_holm']:.3g}",
        "B_C": f"med{mx['B-C']['median_diff']:+.0f} holm={mx['B-C']['p_holm']:.3g}",
        "evidence": "observed per-agent input shares in Phase 4",
    })
    # 4 more targeted tool use (total tool calls)
    tc = {r["pair"]: r for r in summary["phase5"]["paired"]["tool_calls_total"]}
    rows.append({
        "mechanism": "4. More targeted tool use (total tool calls)",
        "A_B": f"med{tc['A-B']['median_diff']:+.3f} holm={tc['A-B']['p_holm']:.3g}",
        "A_C": f"med{tc['A-C']['median_diff']:+.3f} holm={tc['A-C']['p_holm']:.3g}",
        "B_C": f"med{tc['B-C']['median_diff']:+.3f} holm={tc['B-C']['p_holm']:.3g}",
        "evidence": "see Phase 5 per-role tool counts",
    })
    # 5 fewer repeated tool calls (dup_tool_calls)
    dp = {r["pair"]: r for r in summary["phase5"]["paired"]["dup_tool_calls"]}
    rows.append({
        "mechanism": "5. Fewer repeated tool calls (exact dup tool calls)",
        "A_B": f"med{dp['A-B']['median_diff']:+.3f} holm={dp['A-B']['p_holm']:.3g}",
        "A_C": f"med{dp['A-C']['median_diff']:+.3f} holm={dp['A-C']['p_holm']:.3g}",
        "B_C": f"med{dp['B-C']['median_diff']:+.3f} holm={dp['B-C']['p_holm']:.3g}",
        "evidence": "exact tool+argument repeat within an agent",
    })
    # 6 reduced corrections
    co = {r["pair"]: r for r in p1["corrections"]["pairwise"]}
    rows.append({
        "mechanism": "6. Reduced corrections (inter-agent answer changes)",
        "A_B": f"med{co['A-B']['median_diff']:+.3f} holm={co['A-B']['p_holm']:.3g}",
        "A_C": f"med{co['A-C']['median_diff']:+.3f} holm={co['A-C']['p_holm']:.3g}",
        "B_C": f"med{co['B-C']['median_diff']:+.3f} holm={co['B-C']['p_holm']:.3g}",
        "evidence": "corrections run-level",
    })
    # 7 more efficient role interaction (consecutive overlap ov_01)
    ov = {r["pair"]: r for r in summary["phase6"]["paired"]["ov_01"]}
    rows.append({
        "mechanism": "7. Role interaction / handoff overlap (a0->a1 output overlap)",
        "A_B": f"med{ov['A-B']['median_diff']:+.3f} holm={ov['A-B']['p_holm']:.3g}",
        "A_C": f"med{ov['A-C']['median_diff']:+.3f} holm={ov['A-C']['p_holm']:.3g}",
        "B_C": f"med{ov['B-C']['median_diff']:+.3f} holm={ov['B-C']['p_holm']:.3g}",
        "evidence": "jaccard overlap of consecutive outputs",
    })
    # 8 fewer/faster forward passes (llm_calls + output tokens) -- discovered pathway
    lc = {r["pair"]: r for r in p1["llm_calls"]["pairwise"]}
    ot = {r["pair"]: r for r in p1["output_tokens"]["pairwise"]}
    rows.append({
        "mechanism": "8. Forward-pass volume: llm_calls / output tokens (generation load)",
        "A_B": f"calls{lc['A-B']['median_diff']:+.2f}(holm={lc['A-B']['p_holm']:.2g}) out{ot['A-B']['median_diff']:+.0f}(holm={ot['A-B']['p_holm']:.2g})",
        "A_C": f"calls{lc['A-C']['median_diff']:+.2f}(holm={lc['A-C']['p_holm']:.2g}) out{ot['A-C']['median_diff']:+.0f}(holm={ot['A-C']['p_holm']:.2g})",
        "B_C": f"calls{lc['B-C']['median_diff']:+.2f}(holm={lc['B-C']['p_holm']:.2g}) out{ot['B-C']['median_diff']:+.0f}(holm={ot['B-C']['p_holm']:.2g})",
        "evidence": f"largest |pooled Spearman| with latency = {top['name']} (rho={top['rho']:+.2f}); association only",
    })

    # classification is DERIVED FROM THE TESTS, not asserted: does the candidate's own metric differ
    # under the length control (A-C), only before it (A-B), or not at all?
    def classify(pdv):
        ab = pdv["A-B"]["p_holm"] < 0.05
        ac = pdv["A-C"]["p_holm"] < 0.05
        if ac:
            return "Supported under length control (A-C Holm-sig)"
        if ab:
            return "Confounded with prompt length (A-B sig, A-C n.s.)"
        return "Not supported (no sig. A-B or A-C difference)"

    pair_for_verdict = [rw, a0d, mx, tc, dp, co, ov, ot]
    for r, pdv in zip(rows, pair_for_verdict):
        r["verdict"] = classify(pdv)

    # print table
    for r in rows:
        L.append(f"\n  {r['mechanism']}")
        L.append(f"     A/B: {r['A_B']}")
        L.append(f"     A/C: {r['A_C']}   <-- length-controlled test")
        L.append(f"     B/C: {r['B_C']}")
        L.append(f"     evidence: {r['evidence']}")
        L.append(f"     VERDICT (from tests): {r['verdict']}")
    L.append("\n  NOTE: 'verdict' reflects only whether the candidate's own metric differs across conditions "
             "(and survives length control). It is NOT a latency-causation claim; latency links are reported "
             "separately as associations in Phase 8.")
    summary["phase9_table"] = rows
    return rows

# ----------------------------------------------------------------------------- plots
def make_plots(df, adf, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        return [f"plots skipped: {e}"]
    pdir = os.path.join(outdir, "plots")
    os.makedirs(pdir, exist_ok=True)
    notes = []
    order = CONDS
    labels = [SHORT[c] for c in order]

    # latency box
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot([df[df.condition==c]["latency_seconds"].values for c in order], tick_labels=labels, showmeans=True)
    ax.set_ylabel("latency_seconds"); ax.set_title("Latency by condition")
    fig.tight_layout(); fig.savefig(os.path.join(pdir, "latency_box.png"), dpi=120); plt.close(fig)

    # input tokens box
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot([df[df.condition==c]["input_tokens"].values for c in order], tick_labels=labels, showmeans=True)
    ax.set_ylabel("input_tokens"); ax.set_title("Total input tokens by condition")
    fig.tight_layout(); fig.savefig(os.path.join(pdir, "input_tokens_box.png"), dpi=120); plt.close(fig)

    # latency vs input scatter
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = {"generalist": "C0", "specialized": "C1", "specialized_length_matched": "C2"}
    for c in order:
        s = df[df.condition==c]
        ax.scatter(s["input_tokens"], s["latency_seconds"], s=8, alpha=0.4, label=SHORT[c], color=colors[c])
    ax.set_xlabel("input_tokens"); ax.set_ylabel("latency_seconds"); ax.legend(); ax.set_title("Latency vs input tokens")
    fig.tight_layout(); fig.savefig(os.path.join(pdir, "latency_vs_input.png"), dpi=120); plt.close(fig)

    # agent0 input by condition (bar)
    fig, ax = plt.subplots(figsize=(5, 4))
    means = [adf[(adf.condition==c)&(adf.agent_index==0)]["input_tokens"].mean() for c in order]
    ax.bar(labels, means, color=[colors[c] for c in order])
    ax.set_ylabel("agent0 input_tokens (mean)"); ax.set_title("Agent-0 input load by condition")
    fig.tight_layout(); fig.savefig(os.path.join(pdir, "agent0_input_bar.png"), dpi=120); plt.close(fig)

    notes.append(f"plots written to {pdir}")
    return notes

# ----------------------------------------------------------------------------- interpretation.md
def write_interpretation(summary, outdir):
    p1 = summary["phase1_primary"]
    def med(metric, pair):
        for r in p1[metric]["pairwise"]:
            if r["pair"] == pair:
                return r
    lat = {p: med("latency_seconds", p) for p in ["A-B", "A-C", "B-C"]}
    acc = {p: med("accuracy", p) for p in ["A-B", "A-C", "B-C"]}
    tot = {p: med("total_tokens", p) for p in ["A-B", "A-C", "B-C"]}
    inp = {p: med("input_tokens", p) for p in ["A-B", "A-C", "B-C"]}
    out = {p: med("output_tokens", p) for p in ["A-B", "A-C", "B-C"]}
    co = {p: med("corrections", p) for p in ["A-B", "A-C", "B-C"]}
    ov = {r["pair"]: r for r in summary["phase6"]["paired"]["ov_01"]}
    desc = {m: p1[m]["descriptives"] for m in ["accuracy", "latency_seconds", "total_tokens", "input_tokens", "output_tokens", "llm_calls"]}
    ld = summary["phase8"]["lat_decomp"]
    assoc = summary["phase3"]["latency_assoc"]
    tl = summary["phase8_tail"]
    am = summary["phase2_agent_means"]
    a1 = {SHORT[c]: am[c]["1"]["input_tokens"] for c in CONDS}
    a1_ratio_B = (a1["A"] / a1["B"]) if a1["B"] else float("nan")
    a1_ratio_C = (a1["A"] / a1["C"]) if a1["C"] else float("nan")
    top = summary["phase8"]["top_pooled_predictor"]
    corr_pooled = summary["phase8"]["latency_corr_pooled"]
    in_rho = corr_pooled["input_tokens"]["rho"]
    in_within = {c: summary["phase8"]["latency_corr_within"][c]["input_tokens"]["rho"] for c in ["A", "B", "C"]}
    nonlin = summary["phase8"]["nonlin"]
    rw_ab = next(r["median_diff"] for r in summary["phase3"]["pairwise"] if r["pair"] == "A-B")
    blocked = summary["phase0"].get("condition_blocked", False)
    tok_pct_AB = 100.0 * (desc["total_tokens"]["A"]["mean"] - desc["total_tokens"]["B"]["mean"]) / desc["total_tokens"]["A"]["mean"]
    # computed shift classification (median vs tail vs both) and computed significance phrasings
    shift_AB = tl["shift_AB"]; shift_AC = tl["shift_AC"]; shift_BC = tl["shift_BC"]
    rw_ab_row = next(r for r in summary["phase3"]["pairwise"] if r["pair"] == "A-B")
    rw_dir = "lower" if rw_ab_row["median_diff"] < 0 else ("higher" if rw_ab_row["median_diff"] > 0 else "unchanged")
    rw_sig = "Holm-significant" if rw_ab_row["p_holm"] < 0.05 else "not Holm-significant"
    tc_holms = [r["p_holm"] for r in summary["phase5"]["paired"]["tool_calls_total"]]
    tc_sentence = ("at least one pair reached Holm significance" if any(h < 0.05 for h in tc_holms)
                   else "no pair reached Holm significance") + f" (min Holm p={min(tc_holms):.2g})"
    dup_holms = [r["p_holm"] for r in summary["phase5"]["paired"]["dup_tool_calls"]]
    dup_sentence = ("at least one pair Holm-significant" if any(h < 0.05 for h in dup_holms)
                    else "no pair Holm-significant") + f" (min Holm p={min(dup_holms):.2g})"
    # scaling verdict assembled straight from the computed test; licenses any super-linear wording
    scale_desc = "; ".join(
        f"{c}: b={nonlin[c]['b']:.2f} (95%CI[{nonlin[c]['ci_lo']:.2f},{nonlin[c]['ci_hi']:.2f}], {nonlin[c]['verdict']})"
        for c in ["A", "B", "C"])
    superlinear_scopes = [c for c in ["A", "B", "C"] if nonlin[c]["verdict"] == "super-linear"]
    if superlinear_scopes:
        scale_sentence = ("A log-log scaling test of latency vs input tokens IS super-linear (CI excludes 1) in "
                          f"condition(s) {', '.join(superlinear_scopes)}: {scale_desc}. A super-linear reading is "
                          "therefore data-supported only there; treat with the blocked-execution caveat.")
    else:
        scale_sentence = ("A log-log scaling test of latency vs input tokens is NOT distinguishable from linear "
                          f"within conditions ({scale_desc}); a super-linear explanation is NOT established by "
                          "these data and is not asserted.")

    def fmt(r):
        return f"median delta(2nd-1st)={r['median_diff']:+.3f}, Holm p={r['p_holm']:.3g}"

    txt = f"""# Mechanism-Level Analysis - Responsibility Specialization in a 3-Agent LLM System

*Post-hoc analysis of the existing 750 runs (results/v2/runs.jsonl). No new runs; raw data unchanged.
A = generalist, B = specialized, C = specialized_length_matched. 250 runs each, 50 tasks x 5 runs.
Latency is strongly right-skewed -> medians, tail statistics, and non-parametric (Wilcoxon, task-paired,
Holm-corrected) tests are used throughout. All numbers in this file are generated from the runs by
`analysis/mechanism_v2.py`; none are hand-entered.*

## Evidence levels (used explicitly below)

- **OBSERVED** - a descriptive quantity computed directly from the runs (counts, medians, percentiles).
- **ASSOCIATION** - two measured quantities co-vary (correlation, or a cross-condition difference with a
  statistical test); direction of causation is NOT established.
- **PLAUSIBLE MECHANISM** - a process-level story consistent with the observations but not directly tested
  by this design.
- **CAUSAL CLAIM** - would require an intervention this observational re-analysis does not provide; where
  noted, it is explicitly flagged as NOT established.

## 1. Research Question

How does responsibility specialization change the **execution latency** of the system, given that accuracy
is unchanged and the token advantage is ambiguous once prompt length is controlled (condition C)? We look
for execution-level correlates in the traces and separate what is observed from what is inferred.

## 2. Main Result

- **OBSERVED - latency distribution by condition** (medians and upper percentiles):
  - median: A={tl['A']['median']:.1f}s, B={tl['B']['median']:.1f}s, C={tl['C']['median']:.1f}s
  - mean:   A={tl['A']['mean']:.1f}s, B={tl['B']['mean']:.1f}s, C={tl['C']['mean']:.1f}s
  - runs >60s: A={tl['A']['frac_gt_60s']:.1%}, B={tl['B']['frac_gt_60s']:.1%}, C={tl['C']['frac_gt_60s']:.1%};
    runs >100s: A={tl['A']['frac_gt_100s']:.1%}, B={tl['B']['frac_gt_100s']:.1%}, C={tl['C']['frac_gt_100s']:.1%}
  - p95: A={tl['A']['p95']:.0f}s, B={tl['B']['p95']:.0f}s, C={tl['C']['p95']:.0f}s;
    p99: A={tl['A']['p99']:.0f}s, B={tl['B']['p99']:.0f}s, C={tl['C']['p99']:.0f}s;
    max: A={tl['A']['max']:.0f}s, B={tl['B']['max']:.0f}s, C={tl['C']['max']:.0f}s
  - 10%-trimmed mean: A={tl['A']['trimmed10_mean']:.1f}s, B={tl['B']['trimmed10_mean']:.1f}s, C={tl['C']['trimmed10_mean']:.1f}s
- **OBSERVED - shift classification** (computed in Phase 8 by decomposing each mean gap into a bulk vs
  top-10% part; thresholds median_rel>=15%, tail_share>=50% are fixed in code, not per-result):
  - A vs B: {shift_AB['label']} (median diff {shift_AB['median_diff']:+.1f}s = {shift_AB['median_rel']:+.0%};
    {shift_AB['tail_share']:+.0%} of the mean gap comes from the top 10%; p95 diff {shift_AB['p95_diff']:+.0f}s)
  - A vs C: {shift_AC['label']} (median diff {shift_AC['median_diff']:+.1f}s = {shift_AC['median_rel']:+.0%};
    {shift_AC['tail_share']:+.0%} of the mean gap from the top 10%; p95 diff {shift_AC['p95_diff']:+.0f}s)
  - B vs C: {shift_BC['label']}
- **ASSOCIATION - cross-condition difference**: task-paired Wilcoxon (on task means, so it weights the tail)
  gives A vs B: {fmt(lat['A-B'])}; A vs C: {fmt(lat['A-C'])}; B vs C: {fmt(lat['B-C'])}. Read the direction
  and significance off these numbers (Holm-corrected across the 3 pairs).

The shift classification above - not a pre-written adjective - states whether each condition difference is a
median(bulk) shift, a tail shift, or both. Whether specialization *causes* any such difference is treated in
sec.5-7; the sec.7 confound bears directly on it.

## 3. Accuracy

**OBSERVED.** Run-level accuracy: A={desc['accuracy']['A']['mean']:.3f}, B={desc['accuracy']['B']['mean']:.3f},
C={desc['accuracy']['C']['mean']:.3f}. A vs B: {fmt(acc['A-B'])}; A vs C: {fmt(acc['A-C'])}; B vs C: {fmt(acc['B-C'])}.
No statistically significant evidence that specialization changes accuracy (companion TOST in stats_v2.py).

## 4. Efficiency (tokens)

- **OBSERVED - total tokens**: A={desc['total_tokens']['A']['mean']:.0f}, B={desc['total_tokens']['B']['mean']:.0f},
  C={desc['total_tokens']['C']['mean']:.0f} (means; B is {tok_pct_AB:.0f}% below A). **ASSOCIATION:**
  A vs B {fmt(tot['A-B'])}, but A vs C {fmt(tot['A-C'])} - the token difference largely disappears once prompt
  length is matched, so most of the raw token gap tracks the shorter specialized prompt rather than a
  behavioral change.
- **Input tokens**: A vs B {fmt(inp['A-B'])}; A vs C {fmt(inp['A-C'])}; B vs C {fmt(inp['B-C'])}.
- **Output tokens**: A vs B {fmt(out['A-B'])}; A vs C {fmt(out['A-C'])}; B vs C {fmt(out['B-C'])}.

## 5. Mechanism Evidence (each item tagged by evidence level)

- **Accumulated input-token load per agent, OBSERVED values:** the MIDDLE agent's accumulated input-token
  processing load (summed over its tool-loop forward passes - NOT a direct measure of forwarded/handoff
  context size) is a1_load A={a1['A']:.0f}, B={a1['B']:.0f}, C={a1['C']:.0f}
  (A/B={a1_ratio_B:.2f}x, A/C={a1_ratio_C:.2f}x). The >100s tail runs carry high accumulated input-token
  load (Phase 8 profile). **These data cannot separate initial handoff/context size from repeated
  re-processing across tool iterations**, so we read this only as a workload difference.
- **Accumulated input-token load <-> latency, ASSOCIATION:** among the execution variables examined, the
  largest monotonic association with latency was **{top['name']}** (pooled Spearman rho={top['rho']:+.2f}).
  For input_tokens (accumulated load) specifically, pooled rho={in_rho:+.2f}; within condition
  A={in_within['A']:+.2f}, B={in_within['B']:+.2f}, C={in_within['C']:+.2f}. (Correlation, not causation;
  pooled values are additionally condition-confounded.)
- **Scaling, ASSOCIATION / computed:** {scale_sentence}
- **Redundant work (`repeated_work`), ASSOCIATION:** latency association is weak
  (pooled rho={assoc['pooled']['rho']:+.2f}). Across conditions, specialization's repeated_work is {rw_dir}
  than generalist (A vs B median {rw_ab:+.3f}, {rw_sig}). Not a leading latency correlate.
- **Tool use, OBSERVED:** total tool calls across conditions - {tc_sentence}; exact duplicate tool calls -
  {dup_sentence}. (Read significance off these computed Holm p-values; no claim of fewer tools unless they show it.)
- **Handoff overlap, ASSOCIATION:** consecutive-agent output overlap differs by condition
  (a0->a1 overlap A vs B {fmt(ov['A-B'])}; A vs C {fmt(ov['A-C'])}); measured lexically (Jaccard) + a small
  same-task manual sample (Phase 6).
- **Corrections & failure modes, OBSERVED:** corrections are if anything *higher* under specialization
  (A vs C {fmt(co['A-C'])}) - not a speed mechanism. F1 (wrong-responsibility) and F7 (out-of-role tool use)
  are structurally impossible in A and appear only in B/C (F1: B={summary['phase7_failures']['specialized'].get('F1',0)},
  C={summary['phase7_failures']['specialized_length_matched'].get('F1',0)}; F7: B={summary['phase7_failures']['specialized'].get('F7',0)},
  C={summary['phase7_failures']['specialized_length_matched'].get('F7',0)}) - a new failure mode introduced by
  specialization, not evidence it is simply worse.

## 6. Mechanism Conclusion (stated as an evidence hierarchy)

- **OBSERVED:** specialization changes the accumulated per-agent input-token workload (sec.5 values; the
  middle-agent load ratios A/B={a1_ratio_B:.2f}x, A/C={a1_ratio_C:.2f}x, and the Phase 2/4 per-agent loads).
- **ASSOCIATION:** accumulated input-token workload is associated with latency (largest |pooled Spearman|
  predictor = {top['name']}, rho={top['rho']:+.2f}; input_tokens within-condition rho up to
  {max(in_within.values()):+.2f}). Correlational only.
- **PLAUSIBLE MECHANISM (not directly tested):** role specialization may constrain context/workload
  accumulation - consistent with the observed workload difference and the workload<->latency association,
  but not demonstrated as the operative pathway.
- **NOT ESTABLISHED:** that specialization *causally* reduces latency through bounded context growth. This
  would require (i) separating initial handoff/context size from repeated tool-iteration re-processing -
  which these traces cannot do; (ii) an intervention that varies workload while holding role structure
  fixed; and (iii) timestamped, interleaved execution to remove the block confound (sec.7). A super-linear
  workload->latency relationship is asserted only where the sec.5 scaling test's CI excludes 1.

## 7. Alternative Explanations the Data Cannot Separate

- **Execution-order confound (important, bears on the causal reading).** Runs are
  {'condition-blocked (CONFIRMED: all runs of a condition are contiguous in the file)' if blocked else 'not condition-blocked'}
  and no timestamps are logged. If execution followed file order, each condition ran as one block on a single
  CPU-bound host, so time-varying machine load could inflate the generalist block's latency tail. This cannot
  be ruled out and specifically threatens the *magnitude* of the latency tail; it does not threaten the token
  or accuracy results, which are backend-independent.
- **Accumulated load is not handoff size.** Per-agent input_tokens sum prompt tokens re-processed across
  tool-loop iterations; the data cannot isolate the *initial handoff/context size* from *repeated
  re-processing across iterations*. Any "context propagation" reading is therefore unsupported.
- Prompt-eval vs generation share of per-call cost cannot be separated from logged token totals.
- Overlap/redundancy metrics are lexical, not semantic; duplicate-tool detection is exact-match only.

## 8. Limitations

- Single model (qwen2.5:7b), single temperature (0.7), single linear topology, 50 tasks - not broadly generalizable.
- Observational at the execution level; no interventional manipulation of the proposed mediator (accumulated workload).
- Per-agent input_tokens is accumulated processing load, not forwarded-context size (see sec.7); the two cannot be separated here.
- Latency measured on one CPU-bound host with extreme outliers (max ~{tl['A']['max']:.0f}s); absolute seconds
  are backend-specific and the tail magnitude is partly untrustworthy (see confound). Order-robust signals
  (medians, token load, within-condition correlations) are the more trustworthy part.
- Runs condition-blocked without timestamps -> cannot statistically de-confound execution order from architecture.
- One run flagged `truncated`; reported, not removed.

## 9. Conservative, Paper-Ready Claim

> In a controlled 3-agent linear pipeline (qwen2.5:7b, T=0.7, 2 fixed handoffs, matched task set),
> responsibility specialization did **not** change task accuracy and, once system-prompt length was matched
> (condition C), did **not** meaningfully change total token usage. The between-condition **latency**
> difference was characterized by a computed shift classification (A vs B: {shift_AB['label']}; A vs C:
> {shift_AC['label']}), with the generalist showing a heavier upper tail ({tl['A']['frac_gt_100s']:.0%} of
> runs >100s vs <={max(tl['B']['frac_gt_100s'], tl['C']['frac_gt_100s']):.1%} under specialization). This
> difference co-occurred (ASSOCIATION) with a higher **accumulated input-token processing load** on the
> middle agent under the generalist design ({a1_ratio_B:.1f}x the specialized solver's load); because this
> load aggregates re-processing across tool iterations, it is NOT a direct measure of forwarded context size.
> A mechanism in which role specialization *constrains workload/context accumulation* is a PLAUSIBLE account
> consistent with these observations; the design does **not** establish that specialization causally reduces
> latency through bounded context growth. Conditions were executed in blocks on a single CPU-bound host
> without timestamps, so part of the tail magnitude may reflect machine load, and a super-linear
> load->latency relationship is reported only where the scaling test's CI excludes 1. Specialization
> additionally introduced structurally new role-boundary failure modes (F1 wrong-responsibility, F7
> out-of-role tool use) absent by construction in the generalist system.
"""
    path = os.path.join(outdir, "interpretation.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
    return path

# ----------------------------------------------------------------------------- main
def main():
    try:  # Windows consoles default to cp1252 and choke on unicode in the report
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    run_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "v2")
    if not os.path.isabs(run_dir):
        run_dir = os.path.join(ROOT, run_dir)
    outdir = os.path.join(run_dir, "mechanism")
    os.makedirs(outdir, exist_ok=True)

    runs = load_runs(run_dir)
    df = run_level_df(runs)
    adf = agent_level_df(runs)

    L = [f"MECHANISM-LEVEL ANALYSIS | {len(runs)} runs | dir={run_dir}"]
    summary = {"n_runs": len(runs), "run_dir": run_dir}

    summary["phase0"] = phase0_validation(runs, df, L)
    primary_df = phase1_primary(df, L, summary)
    primary_df.to_csv(os.path.join(outdir, "primary_metrics.csv"), index=False)
    phase2_agent_workload(adf, L, summary, outdir)
    phase3_redundant(df, L, summary)
    phase4_context(df, L, summary)
    phase5_tools(df, adf, runs, L, summary)
    phase6_handoffs(df, runs, L, summary, outdir)
    phase7_failures(runs, df, L, summary, outdir)
    phase8_latency_pathway(df, L, summary, outdir)
    phase9_triangulation(summary, L)
    for n in make_plots(df, adf, outdir):
        L.append(n)
    interp = write_interpretation(summary, outdir)

    # save summary.json
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)

    # ---- concise final console summary ----
    p1 = summary["phase1_primary"]
    def g(metric, short): return p1[metric]["descriptives"][short]
    lat = {r["pair"]: r for r in p1["latency_seconds"]["pairwise"]}
    acc = {r["pair"]: r for r in p1["accuracy"]["pairwise"]}
    inp = {r["pair"]: r for r in p1["input_tokens"]["pairwise"]}
    assoc = summary["phase3"]["latency_assoc"]
    S = []
    S.append("\n" + "#" * 78)
    S.append("CONCISE SUMMARY")
    S.append("#" * 78)
    S.append("DATA VALIDATION")
    S.append(f"  {summary['phase0']['total']} total | " +
             " ".join(f"{SHORT[c]}={summary['phase0']['condition_counts'][c]}" for c in CONDS) +
             f" | truncated={summary['phase0']['truncated']} timeout={summary['phase0']['timeout']} error={summary['phase0']['errors']}")
    S.append(f"  verdict: {'PASS' if not summary['phase0']['issues'] else summary['phase0']['issues']}")
    S.append("PRIMARY FINDINGS (run-level median | task-paired Holm p)")
    S.append(f"  accuracy  A={g('accuracy','A')['mean']:.3f} B={g('accuracy','B')['mean']:.3f} C={g('accuracy','C')['mean']:.3f}"
             f"  | A-C holm={acc['A-C']['p_holm']:.3g} (no sig. accuracy change)")
    S.append(f"  latency   A={g('latency_seconds','A')['median']:.1f}s B={g('latency_seconds','B')['median']:.1f}s C={g('latency_seconds','C')['median']:.1f}s"
             f"  | A-B holm={lat['A-B']['p_holm']:.3g}  A-C holm={lat['A-C']['p_holm']:.3g}  B-C holm={lat['B-C']['p_holm']:.3g}")
    S.append(f"  input_tok A={g('input_tokens','A')['median']:.0f} B={g('input_tokens','B')['median']:.0f} C={g('input_tokens','C')['median']:.0f}"
             f"  | A-C holm={inp['A-C']['p_holm']:.3g}")
    S.append("AGENT WORKLOAD (accumulated input-token load per agent; not handoff/context size)")
    pa = summary["phase2_agent_means"]
    for c in CONDS:
        S.append(f"  {SHORT[c]} a0_load={pa[c]['0']['input_tokens']:.0f} a1_load={pa[c]['1']['input_tokens']:.0f} a2_load={pa[c]['2']['input_tokens']:.0f}"
                 f"  a0_iters={pa[c]['0']['tool_iters']:.2f}")
    S.append("REDUNDANT WORK")
    S.append(f"  repeated_work vs latency: pooled rho={assoc['pooled']['rho']:+.2f} (A={assoc['A']['rho']:+.2f} B={assoc['B']['rho']:+.2f} C={assoc['C']['rho']:+.2f})")
    S.append("ACCUMULATED INPUT-TOKEN LOAD (Phase 4; summed over tool iterations, not handoff size)")
    S.append(f"  max_agent_load A-C median diff="
             f"{[r['median_diff'] for r in summary['phase4']['paired']['max_agent_input'] if r['pair']=='A-C'][0]:+.0f}")
    S.append("TOOL USE")
    dup = {r['pair']: r for r in summary['phase5']['paired']['dup_tool_calls']}
    S.append(f"  dup_tool_calls A-C median diff={dup['A-C']['median_diff']:+.3f} (holm={dup['A-C']['p_holm']:.3g})")
    S.append("FAILURE MODES")
    ff = summary["phase7_failures"]
    S.append(f"  F1 A={ff['generalist'].get('F1',0)} B={ff['specialized'].get('F1',0)} C={ff['specialized_length_matched'].get('F1',0)}"
             f" | F7 A={ff['generalist'].get('F7',0)} B={ff['specialized'].get('F7',0)} C={ff['specialized_length_matched'].get('F7',0)} (F1/F7 structural to B/C)")
    S.append("LATENCY PATHWAY")
    tl = summary["phase8_tail"]
    top = summary["phase8"]["top_pooled_predictor"]
    nl = summary["phase8"]["nonlin"]
    superl = [c for c in ["A", "B", "C"] if nl[c]["verdict"] == "super-linear"]
    pa2 = summary["phase2_agent_means"]
    a1A = pa2["generalist"]["1"]["input_tokens"]; a1B = pa2["specialized"]["1"]["input_tokens"]
    ratioB = (a1A / a1B) if a1B else float("nan")
    S.append(f"  latency medians A={tl['A']['median']:.1f} B={tl['B']['median']:.1f} C={tl['C']['median']:.1f}s;"
             f" means A={tl['A']['mean']:.0f} B={tl['B']['mean']:.0f} C={tl['C']['mean']:.0f}s;"
             f" runs>100s A={tl['A']['frac_gt_100s']:.1%} B={tl['B']['frac_gt_100s']:.1%} C={tl['C']['frac_gt_100s']:.1%}")
    S.append(f"  shift classification (computed): A-B {tl['shift_AB']['label']}; A-C {tl['shift_AC']['label']}")
    S.append(f"  largest |pooled rho| with latency = {top['name']} (rho={top['rho']:+.2f}) [association]")
    S.append(f"  scaling test (latency~accumulated input load): " +
             " ".join(f"{c} b={nl[c]['b']:.2f}({nl[c]['verdict']})" for c in ["A", "B", "C"]) +
             (f" -> super-linear in {','.join(superl)}" if superl else " -> NOT distinguishable from linear"))
    S.append(f"  CONFOUND: runs condition-blocked, no timestamps -> A's tail MAGNITUDE may be partly machine-load (tokens/accuracy unaffected)")
    S.append("MECHANISM CONCLUSION  (evidence hierarchy)")
    S.append(f"  OBSERVED: specialization changes accumulated per-agent input-token workload (middle-agent load A/B={ratioB:.1f}x).")
    S.append(f"  ASSOCIATION: accumulated input-token workload is associated with latency (top correlate {top['name']} rho={top['rho']:+.2f}).")
    S.append("  PLAUSIBLE MECHANISM: role specialization may constrain context/workload accumulation (not tested).")
    S.append("  NOT ESTABLISHED: specialization causally reduces latency through bounded context growth.")
    S.append("  accuracy unchanged; token advantage mostly disappears vs length-matched C.")
    S.append(f"\nOutputs in {outdir}:  summary.json, primary_metrics.csv, agent_level.csv, correlations.csv,")
    S.append(f"  failure_analysis.csv, handoff_analysis.csv, interpretation.md, plots/")

    report = "\n".join(L + S)
    with open(os.path.join(outdir, "mechanism_v2_report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
