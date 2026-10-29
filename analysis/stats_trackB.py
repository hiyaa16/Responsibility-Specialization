"""Track B analysis — the pre-registered depth x condition (and complexity x condition) INTERACTION,
reported across ALL outcome families: accuracy, latency, cost, robustness, recovery.

Primary estimand: does the generalist-vs-specialist (A-vs-C) gap CHANGE as the pipeline deepens?
  (1) GEE logistic  correct ~ C(cond)*depth  clustered by task              [accuracy, model-based]
  (2) Difference-in-differences Wilcoxon on per-task (gap@D_hi - gap@D_lo)   [accuracy, non-parametric]
  (3) complexity x condition via GEE (K is between-task)                     [accuracy vs task difficulty]
  (4) latency / cost: OLS  y ~ condC*depth (+ execution_order for latency, the v2 de-confounder), by task
  (5) robustness: truncation rate by cell + accuracy-conclusion sensitivity to dropping truncated runs
  (6) recovery: corrections by depth x condition + DiD
Plus validity: runtime A~=C per-agent length match per depth, interleave/execution-order drift.

Single model:   python analysis/stats_trackB.py [run_dir]
Cross-model:     python analysis/stats_trackB.py --cross results/trackB results/trackB_llama results/trackB_mistral
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONDS = ["generalist", "specialized_length_matched"]
SH = {"generalist": "A", "specialized_length_matched": "C"}


def load(run_dir):
    path = os.path.join(run_dir, "runs.jsonl")
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    recs = []
    for r in rows:
        ao = sorted(r.get("agent_outputs", []), key=lambda a: a["agent_index"])
        ov01 = np.nan
        if len(ao) >= 2:
            def ws(t): return set("".join(c.lower() if c.isalnum() else " " for c in (t or "")).split())
            a, b = ws(ao[0]["output_text"]), ws(ao[1]["output_text"])
            ov01 = (len(a & b) / len(a | b)) if (a and b) else 0.0
        recs.append({"task": r["task_id"], "cond": r["condition"], "depth": int(r["depth"]),
                     "K": int(r.get("complexity_K", r.get("hops", 0))), "correct": int(bool(r["correct"])),
                     "lat": r.get("latency_seconds", np.nan), "tot": r.get("total_tokens", np.nan),
                     "order": r.get("execution_order", np.nan), "trunc": bool(r.get("truncated")),
                     "corr": int(r.get("corrections", 0)), "model": r.get("model", "?"),
                     "ov01": ov01, "sys_per_agent": r.get("system_prompt_tokens_per_agent", [])})
    return pd.DataFrame(recs)


def holm(ps):
    ps = list(ps); m = len(ps); order = np.argsort(ps); adj = np.empty(m); run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * ps[idx]); adj[idx] = min(run, 1.0)
    return adj


def _did_wilcoxon(df, depths, col, agg="mean"):
    """Per-task difference-in-differences on a continuous/count col: (A-C @D_hi) - (A-C @D_lo)."""
    d_lo, d_hi = depths[0], depths[-1]
    piv = df.groupby(["task", "depth", "cond"])[col].agg(agg).unstack("cond")
    did = []
    for t in df["task"].unique():
        try:
            g_hi = piv.loc[(t, d_hi), "generalist"] - piv.loc[(t, d_hi), "specialized_length_matched"]
            g_lo = piv.loc[(t, d_lo), "generalist"] - piv.loc[(t, d_lo), "specialized_length_matched"]
            did.append(g_hi - g_lo)
        except KeyError:
            pass
    did = np.array([x for x in did if np.isfinite(x)])
    if len(did) >= 5 and np.any(did != 0):
        w, p = stats.wilcoxon(did)
        return len(did), float(did.mean()), float(np.median(did)), float(p)
    return len(did), (float(did.mean()) if len(did) else np.nan), np.nan, np.nan


def _ols_interaction(df, depths, col, add_order=False):
    """OLS y ~ condC*depthc (+ order), cluster-robust by task. Returns (coef, p) of the interaction."""
    d = df.dropna(subset=[col]).copy()
    if d.empty:
        return None
    d["condC"] = (d["cond"] == "specialized_length_matched").astype(int)
    d["depthc"] = d["depth"] - min(depths)
    d["const"] = 1.0
    cols = ["const", "condC", "depthc", "condC:depthc"]
    d["condC:depthc"] = d["condC"] * d["depthc"]
    if add_order and d["order"].notna().any():
        d["order_z"] = (d["order"] - d["order"].mean()) / (d["order"].std() or 1.0)
        cols.append("order_z")
    X = d[cols].astype(float); y = d[col].astype(float)
    try:
        res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": d["task"]})
        return float(res.params["condC:depthc"]), float(res.pvalues["condC:depthc"])
    except Exception as e:
        return ("err", str(e))


def _cell_table(df, depths, col, say, label, fmt="{:.3f}", agg="mean"):
    say(f"\n{label} by depth x condition ({agg}):")
    for d in depths:
        line = f"  D={d}: "
        for c in CONDS:
            v = df[(df.depth == d) & (df.cond == c)][col]
            v = v.dropna()
            val = getattr(v, agg)() if len(v) else np.nan
            line += f"{SH[c]}=" + (fmt.format(val) if np.isfinite(val) else "NA") + f"(n={len(v)})  "
        a = df[(df.depth == d) & (df.cond == 'generalist')][col].dropna()
        c = df[(df.depth == d) & (df.cond == 'specialized_length_matched')][col].dropna()
        if len(a) and len(c):
            diff = getattr(a, agg)() - getattr(c, agg)()
            line += f"=> A-C={('+' if diff>=0 else '')}" + fmt.format(diff)
        say(line)


def analyze(df, run_dir, say):
    say("=" * 88)
    say(f"TRACK B ANALYSIS  (model={df['model'].iloc[0] if len(df) else '?'}, "
        f"n_runs={len(df)}, run_dir={run_dir})")
    say("=" * 88)
    depths = sorted(df["depth"].unique())
    say(f"depths={depths}  conditions={sorted(df['cond'].unique())}  tasks={df['task'].nunique()}  "
        f"reps={df.groupby(['depth','cond','task']).size().max()}")

    # ===================== ACCURACY =====================
    say("\n" + "-" * 40 + " ACCURACY " + "-" * 38)
    _cell_table(df, depths, "correct", say, "Accuracy")

    say("\n[PRIMARY] GEE logistic  correct ~ C(cond)*depth  (cluster=task):")
    try:
        d2 = df.copy()
        d2["condC"] = (d2["cond"] == "specialized_length_matched").astype(int)   # 1=C, 0=A
        d2["depthc"] = d2["depth"] - min(depths)
        m = smf.gee("correct ~ condC * depthc", groups="task", data=d2,
                    family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
        it = "condC:depthc"
        say(f"    interaction {it}: coef(log-OR)={m.params[it]:+.4f}  "
            f"95%CI[{m.conf_int().loc[it,0]:+.4f},{m.conf_int().loc[it,1]:+.4f}]  p={m.pvalues[it]:.4g}")
        say(f"    (positive => C improves relative to A as depth increases)")
    except Exception as e:
        say(f"    GEE not estimable yet: {e}")

    if len(depths) >= 2:
        d_lo, d_hi = depths[0], depths[-1]
        n, mn, md, p = _did_wilcoxon(df, depths, "correct")
        say(f"\n[BACKUP] DiD Wilcoxon (A-C gap @D{d_hi}) - (@D{d_lo}): "
            f"n_tasks={n} mean={mn:+.4f} median={md if np.isnan(md) else f'{md:+.4f}'} "
            f"p={'NA' if np.isnan(p) else f'{p:.4g}'}")
        say(f"    (negative => A loses accuracy vs C more at the deeper pipeline)")

    if df["K"].nunique() >= 2:
        say("\nComplexity x condition GEE  correct ~ C(cond)*K  (cluster=task):")
        try:
            d3 = df.copy(); d3["condC"] = (d3["cond"] == "specialized_length_matched").astype(int)
            m3 = smf.gee("correct ~ condC * K", groups="task", data=d3,
                         family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
            say(f"    interaction condC:K coef={m3.params['condC:K']:+.4f} p={m3.pvalues['condC:K']:.4g}")
        except Exception as e:
            say(f"    not estimable yet: {e}")

    # ===================== LATENCY =====================
    say("\n" + "-" * 40 + " LATENCY " + "-" * 39)
    _cell_table(df, depths, "lat", say, "Latency (s)", fmt="{:.1f}")
    r = _ols_interaction(df, depths, "lat", add_order=True)
    if r and not isinstance(r[0], str):
        say(f"    OLS depth x condition interaction (de-confounded by execution_order): "
            f"coef={r[0]:+.3f}s/depth-step  p={r[1]:.4g}")
        say(f"    (negative => specialization's latency advantage widens with depth)")
    elif isinstance(r, tuple):
        say(f"    OLS not estimable: {r[1]}")
    n, mn, md, p = _did_wilcoxon(df, depths, "lat")
    say(f"    DiD Wilcoxon (A-C latency): n={n} mean={mn:+.2f}s "
        f"p={'NA' if np.isnan(p) else f'{p:.4g}'}")

    # ===================== COST (tokens) =====================
    say("\n" + "-" * 41 + " COST " + "-" * 41)
    _cell_table(df, depths, "tot", say, "Total tokens", fmt="{:.0f}")
    r = _ols_interaction(df, depths, "tot")
    if r and not isinstance(r[0], str):
        say(f"    OLS depth x condition interaction: coef={r[0]:+.1f} tok/depth-step  p={r[1]:.4g}")
    n, mn, md, p = _did_wilcoxon(df, depths, "tot")
    say(f"    DiD Wilcoxon (A-C tokens): n={n} mean={mn:+.0f} "
        f"p={'NA' if np.isnan(p) else f'{p:.4g}'}")

    # ===================== ROBUSTNESS (truncation) =====================
    say("\n" + "-" * 38 + " ROBUSTNESS " + "-" * 38)
    _cell_table(df, depths, "trunc", say, "Truncation rate", fmt="{:.3f}")
    say(f"    total truncated: {int(df['trunc'].sum())} / {len(df)}  ({df['trunc'].mean()*100:.1f}%)")
    # sensitivity: does the accuracy interaction conclusion survive dropping truncated runs?
    clean = df[~df["trunc"]]
    if len(clean) and clean["trunc"].notna().any():
        try:
            dc = clean.copy(); dc["condC"] = (dc["cond"] == "specialized_length_matched").astype(int)
            dc["depthc"] = dc["depth"] - min(depths)
            mc = smf.gee("correct ~ condC * depthc", groups="task", data=dc,
                         family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
            say(f"    sensitivity (truncated dropped, n={len(clean)}): "
                f"accuracy interaction coef={mc.params['condC:depthc']:+.4f} p={mc.pvalues['condC:depthc']:.4g} "
                f"(conclusion stable if sign/size ~ primary)")
        except Exception as e:
            say(f"    sensitivity not estimable: {e}")

    # ===================== RECOVERY (corrections) =====================
    say("\n" + "-" * 39 + " RECOVERY " + "-" * 39)
    _cell_table(df, depths, "corr", say, "Corrections (count)", fmt="{:.3f}")
    n, mn, md, p = _did_wilcoxon(df, depths, "corr")
    say(f"    DiD Wilcoxon (A-C corrections): n={n} mean={mn:+.3f} "
        f"p={'NA' if np.isnan(p) else f'{p:.4g}'}")

    # ===================== MECHANISM + VALIDITY =====================
    say("\n" + "-" * 36 + " MECHANISM + VALIDITY " + "-" * 30)
    say("First-handoff overlap ov_01 (A-C) by depth:")
    for d in depths:
        a = df[(df.depth == d) & (df.cond == 'generalist')]["ov01"].mean()
        c = df[(df.depth == d) & (df.cond == 'specialized_length_matched')]["ov01"].mean()
        say(f"    D={d}: A={a:.3f} C={c:.3f} diff={a-c:+.3f}")
    for d in depths:
        for c in CONDS:
            arr = [x for x in df[(df.depth == d) & (df.cond == c)]["sys_per_agent"] if x]
            if arr:
                mp = np.mean(np.array(arr), axis=0)
                say(f"    D={d} {SH[c]}: mean per-agent sys tokens={[round(float(x),1) for x in mp]} "
                    f"total={round(float(sum(mp)),1)}")
    if df["order"].notna().any():
        rho, p = stats.spearmanr(df["order"], df["lat"])
        say(f"    interleave check — Spearman(execution_order, latency)={rho:+.3f} (p={p:.2g}) "
            f"(near 0 => no global drift)")

    # compact machine-readable summary for cross-model synthesis
    summ = {"model": df["model"].iloc[0] if len(df) else "?", "n": len(df), "depths": depths}
    try:
        d2 = df.copy(); d2["condC"] = (d2["cond"] == "specialized_length_matched").astype(int)
        d2["depthc"] = d2["depth"] - min(depths)
        m = smf.gee("correct ~ condC * depthc", groups="task", data=d2,
                    family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
        summ["acc_interaction_coef"] = float(m.params["condC:depthc"])
        summ["acc_interaction_p"] = float(m.pvalues["condC:depthc"])
    except Exception:
        summ["acc_interaction_coef"] = summ["acc_interaction_p"] = None
    for col, key in [("lat", "lat"), ("tot", "tot"), ("corr", "corr")]:
        _, mn, _, p = _did_wilcoxon(df, depths, col)
        summ[f"{key}_did_mean"] = mn; summ[f"{key}_did_p"] = p
    summ["trunc_rate"] = float(df["trunc"].mean())
    return summ


def cross(dirs):
    L = []
    def say(s): L.append(s); print(s)
    summaries = []
    for d in dirs:
        dd = d if os.path.isabs(d) else os.path.join(ROOT, d)
        if not os.path.exists(os.path.join(dd, "runs.jsonl")):
            say(f"[skip] {d}: no runs.jsonl yet"); continue
        df = load(dd)
        summaries.append(analyze(df, dd, say))

    if summaries:
        say("\n" + "=" * 88)
        say("CROSS-MODEL SYNTHESIS — is the depth x condition pattern consistent across models?")
        say("=" * 88)
        hdr = f"{'model':<16}{'n':>5}  {'acc_int_coef':>12} {'acc_p':>8}  " \
              f"{'lat_DiD':>9} {'lat_p':>8}  {'tok_DiD':>9} {'tok_p':>8}  " \
              f"{'corr_DiD':>9} {'corr_p':>8}  {'trunc%':>7}"
        say(hdr); say("-" * len(hdr))
        def f(v, spec):
            return "NA" if (v is None or (isinstance(v, float) and np.isnan(v))) else format(v, spec)
        for s in summaries:
            say(f"{str(s['model']):<16}{s['n']:>5}  "
                f"{f(s['acc_interaction_coef'],'+12.4f')} {f(s['acc_interaction_p'],'8.4g')}  "
                f"{f(s['lat_did_mean'],'+9.2f')} {f(s['lat_did_p'],'8.4g')}  "
                f"{f(s['tot_did_mean'],'+9.0f')} {f(s['tot_did_p'],'8.4g')}  "
                f"{f(s['corr_did_mean'],'+9.3f')} {f(s['corr_did_p'],'8.4g')}  "
                f"{f(s['trunc_rate']*100,'7.1f')}")
        say("\nReplication is 'consistent' when the SIGN of each DiD/interaction agrees across models,")
        say("even if per-model p-values vary (CPU-budget n is modest). Agreement in sign across 3 models")
        say("is itself strong evidence; disagreement flags a model-specific (not general) effect.")
        out = os.path.join(ROOT, "results", "trackB_cross_model.txt")
        with open(out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(L) + "\n")
        say(f"\n-> {out}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--cross":
        cross(sys.argv[2:] or ["results/trackB", "results/trackB_llama", "results/trackB_mistral"])
        return
    run_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "trackB")
    df = load(run_dir)
    L = []
    def say(s): L.append(s); print(s)
    analyze(df, run_dir, say)
    out = os.path.join(run_dir, "stats_trackB.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
