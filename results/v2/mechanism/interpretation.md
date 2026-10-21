# Mechanism-Level Analysis - Responsibility Specialization in a 3-Agent LLM System

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
  - median: A=24.8s, B=21.2s, C=22.4s
  - mean:   A=113.6s, B=25.2s, C=25.5s
  - runs >60s: A=28.8%, B=4.0%, C=2.8%;
    runs >100s: A=21.2%, B=0.0%, C=0.4%
  - p95: A=634s, B=55s, C=51s;
    p99: A=912s, B=78s, C=69s;
    max: A=1448s, B=88s, C=108s
  - 10%-trimmed mean: A=48.4s, B=21.3s, C=22.1s
- **OBSERVED - shift classification** (computed in Phase 8 by decomposing each mean gap into a bulk vs
  top-10% part; thresholds median_rel>=15%, tail_share>=50% are fixed in code, not per-result):
  - A vs B: BOTH a median(bulk) shift and a tail shift (median diff +3.7s = +17%;
    +69% of the mean gap comes from the top 10%; p95 diff +578s)
  - A vs C: primarily a TAIL shift (median diff +2.4s = +11%;
    +70% of the mean gap from the top 10%; p95 diff +583s)
  - B vs C: no large shift of either kind
- **ASSOCIATION - cross-condition difference**: task-paired Wilcoxon (on task means, so it weights the tail)
  gives A vs B: median delta(2nd-1st)=-2.963, Holm p=0.00176; A vs C: median delta(2nd-1st)=-5.382, Holm p=0.000374; B vs C: median delta(2nd-1st)=+0.192, Holm p=0.616. Read the direction
  and significance off these numbers (Holm-corrected across the 3 pairs).

The shift classification above - not a pre-written adjective - states whether each condition difference is a
median(bulk) shift, a tail shift, or both. Whether specialization *causes* any such difference is treated in
sec.5-7; the sec.7 confound bears directly on it.

## 3. Accuracy

**OBSERVED.** Run-level accuracy: A=0.512, B=0.524,
C=0.500. A vs B: median delta(2nd-1st)=+0.000, Holm p=1; A vs C: median delta(2nd-1st)=+0.000, Holm p=1; B vs C: median delta(2nd-1st)=+0.000, Holm p=1.
No statistically significant evidence that specialization changes accuracy (companion TOST in stats_v2.py).

## 4. Efficiency (tokens)

- **OBSERVED - total tokens**: A=16367, B=13780,
  C=14456 (means; B is 16% below A). **ASSOCIATION:**
  A vs B median delta(2nd-1st)=-2228.400, Holm p=0.00479, but A vs C median delta(2nd-1st)=-53.800, Holm p=0.356 - the token difference largely disappears once prompt
  length is matched, so most of the raw token gap tracks the shorter specialized prompt rather than a
  behavioral change.
- **Input tokens**: A vs B median delta(2nd-1st)=-1800.200, Holm p=0.00593; A vs C median delta(2nd-1st)=-33.800, Holm p=0.357; B vs C median delta(2nd-1st)=+1287.300, Holm p=0.314.
- **Output tokens**: A vs B median delta(2nd-1st)=-35.100, Holm p=0.172; A vs C median delta(2nd-1st)=-83.300, Holm p=0.00173; B vs C median delta(2nd-1st)=-33.500, Holm p=0.0316.

## 5. Mechanism Evidence (each item tagged by evidence level)

- **Accumulated input-token load per agent, OBSERVED values:** the MIDDLE agent's accumulated input-token
  processing load (summed over its tool-loop forward passes - NOT a direct measure of forwarded/handoff
  context size) is a1_load A=3697, B=1694, C=1750
  (A/B=2.18x, A/C=2.11x). The >100s tail runs carry high accumulated input-token
  load (Phase 8 profile). **These data cannot separate initial handoff/context size from repeated
  re-processing across tool iterations**, so we read this only as a workload difference.
- **Accumulated input-token load <-> latency, ASSOCIATION:** among the execution variables examined, the
  largest monotonic association with latency was **total_tokens** (pooled Spearman rho=+0.82).
  For input_tokens (accumulated load) specifically, pooled rho=+0.81; within condition
  A=+0.65, B=+0.92, C=+0.89. (Correlation, not causation;
  pooled values are additionally condition-confounded.)
- **Scaling, ASSOCIATION / computed:** A log-log scaling test of latency vs input tokens is NOT distinguishable from linear within conditions (A: b=0.89 (95%CI[0.73,1.05], not distinguishable from linear); B: b=0.63 (95%CI[0.59,0.67], sub-linear); C: b=0.75 (95%CI[0.70,0.80], sub-linear)); a super-linear explanation is NOT established by these data and is not asserted.
- **Redundant work (`repeated_work`), ASSOCIATION:** latency association is weak
  (pooled rho=-0.14). Across conditions, specialization's repeated_work is lower
  than generalist (A vs B median -0.066, not Holm-significant). Not a leading latency correlate.
- **Tool use, OBSERVED:** total tool calls across conditions - no pair reached Holm significance (min Holm p=1); exact duplicate tool calls -
  no pair Holm-significant (min Holm p=0.096). (Read significance off these computed Holm p-values; no claim of fewer tools unless they show it.)
- **Handoff overlap, ASSOCIATION:** consecutive-agent output overlap differs by condition
  (a0->a1 overlap A vs B median delta(2nd-1st)=-0.123, Holm p=4.33e-05; A vs C median delta(2nd-1st)=-0.106, Holm p=0.00183); measured lexically (Jaccard) + a small
  same-task manual sample (Phase 6).
- **Corrections & failure modes, OBSERVED:** corrections are if anything *higher* under specialization
  (A vs C median delta(2nd-1st)=+0.200, Holm p=0.0181) - not a speed mechanism. F1 (wrong-responsibility) and F7 (out-of-role tool use)
  are structurally impossible in A and appear only in B/C (F1: B=130,
  C=111; F7: B=82,
  C=77) - a new failure mode introduced by
  specialization, not evidence it is simply worse.

## 6. Mechanism Conclusion (stated as an evidence hierarchy)

- **OBSERVED:** specialization changes the accumulated per-agent input-token workload (sec.5 values; the
  middle-agent load ratios A/B=2.18x, A/C=2.11x, and the Phase 2/4 per-agent loads).
- **ASSOCIATION:** accumulated input-token workload is associated with latency (largest |pooled Spearman|
  predictor = total_tokens, rho=+0.82; input_tokens within-condition rho up to
  +0.92). Correlational only.
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
  condition-blocked (CONFIRMED: all runs of a condition are contiguous in the file)
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
- Latency measured on one CPU-bound host with extreme outliers (max ~1448s); absolute seconds
  are backend-specific and the tail magnitude is partly untrustworthy (see confound). Order-robust signals
  (medians, token load, within-condition correlations) are the more trustworthy part.
- Runs condition-blocked without timestamps -> cannot statistically de-confound execution order from architecture.
- One run flagged `truncated`; reported, not removed.

## 9. Conservative, Paper-Ready Claim

> In a controlled 3-agent linear pipeline (qwen2.5:7b, T=0.7, 2 fixed handoffs, matched task set),
> responsibility specialization did **not** change task accuracy and, once system-prompt length was matched
> (condition C), did **not** meaningfully change total token usage. The between-condition **latency**
> difference was characterized by a computed shift classification (A vs B: BOTH a median(bulk) shift and a tail shift; A vs C:
> primarily a TAIL shift), with the generalist showing a heavier upper tail (21% of
> runs >100s vs <=0.4% under specialization). This
> difference co-occurred (ASSOCIATION) with a higher **accumulated input-token processing load** on the
> middle agent under the generalist design (2.2x the specialized solver's load); because this
> load aggregates re-processing across tool iterations, it is NOT a direct measure of forwarded context size.
> A mechanism in which role specialization *constrains workload/context accumulation* is a PLAUSIBLE account
> consistent with these observations; the design does **not** establish that specialization causally reduces
> latency through bounded context growth. Conditions were executed in blocks on a single CPU-bound host
> without timestamps, so part of the tail magnitude may reflect machine load, and a super-linear
> load->latency relationship is reported only where the scaling test's CI excludes 1. Specialization
> additionally introduced structurally new role-boundary failure modes (F1 wrong-responsibility, F7
> out-of-role tool use) absent by construction in the generalist system.
