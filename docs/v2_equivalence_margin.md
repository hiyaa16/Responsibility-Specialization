# v2 reliability equivalence margin (pre-registered)

**Fixed BEFORE examining any v2 results.** This file is part of the v2 pre-registration; the
margin must not be changed after seeing data.

## Primary reliability DV
Task success rate (deterministic correctness; `answer_type`-specific objective check — numeric
tolerance / set subset / SQuAD-normalized QA match with aliases). Analyzed at the **task level**
(the 5 stochastic runs per task are averaged first; they are not independent observations).

## Margin
**Δ = ±0.05 (5 percentage points)** on the task-mean success rate, for every pairwise condition
comparison (A–B, B–C, A–C).

## Why 0.05 (justification, a priori)
1. **Practical significance.** On a 50-task benchmark, 5 pp corresponds to 2.5 tasks. A
   reliability difference smaller than this is operationally negligible for *choosing between
   architectures*: below this threshold the decision is dominated by the cost and latency
   dimensions (where specialization shows large, significant effects), not by accuracy.
2. **Calibrated against v1.** The v1 generalist–specialized reliability gap was 0.0067
   (0.67 pp) — an order of magnitude inside this margin. 5 pp comfortably brackets the class of
   effects we are willing to call "no practical difference," without being so wide as to be
   vacuous.
3. **Field convention.** ±5 pp accuracy is a common practical-equivalence threshold in QA / NLP
   reliability comparisons.

## Test
**TOST (two one-sided tests)** on the paired task-level differences for each condition pair,
α = 0.05. Equivalence is declared only if the **90% CI** of the mean paired difference lies
entirely within (−0.05, +0.05). This is reported alongside, not instead of, the Wilcoxon
signed-rank result.

## Interpretation rules (also fixed a priori)
- A non-significant Wilcoxon result is **not** evidence of equivalence on its own.
- Equivalence is claimed **only** if TOST passes at the pre-set margin.
- If neither the difference test nor the equivalence test is conclusive (CI wider than the
  margin), the result is reported as **inconclusive / underpowered**, never as "equivalent."
- With n = 50 paired tasks we report the observed effect size and its CI regardless of
  significance, so readers can judge precision directly (Part 20).
