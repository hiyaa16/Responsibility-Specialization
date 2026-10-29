                                                                                                                                                                                                                                                                   # Pre-registration — Track B: When does responsibility specialization start to matter?

**Status:** design frozen *before* any Track-B run. Track B is a NEW experiment; the v1 (300) and
v2 (2,250) datasets are not modified, re-run, or pooled. Freeze this file + `config/experiment_trackB.yaml`
+ `data/tasks_trackB.jsonl` + the new prompt blocks + padding before the confirmatory run.

## 1. Why Track B exists (the gap in v2)
v2 isolated specialization cleanly but in a regime where it *cannot* show up: a minimal 3-role linear
pipeline on mid-difficulty tasks, where accuracy was flat across conditions (A≈C in all three models).
The v2 finding is a controlled **null + length-artifact debunk**. Track B tests the *boundary condition*:
**specialization is hypothesized to be a latent benefit that appears only once a generalist's single
context becomes overloaded** — i.e., as the number of responsibilities (pipeline *depth*) and the task's
*decomposition complexity* grow. The headline target is an **interaction**, not a main effect.

## 2. The one mechanism, still isolated
Same principle as v2: at every depth level, the generalist and the specialized conditions have **identical
structure** (same number of agents, same number of handoffs, same union of instruction content, and —
via padding — the same per-agent prompt length). The *only* difference is whether the D responsibilities
are **concentrated** in every agent (generalist) or **distributed** one-per-agent (specialized). This keeps
Track B a clean mechanism test while adding the two moderators that should unlock an effect.

## 3. Design — 2 (condition) × D (depth) × K (complexity), within-task
- **Condition** (2): **A = generalist** (every agent holds all D responsibility blocks);
  **C = specialized_length_matched** (each agent holds exactly one block + semantically-inert padding so
  per-agent prompt length matches A). *Condition B (unpadded specialist) is dropped — v2 already showed it
  only re-introduces the length confound.*
- **Depth D** (primary 2 levels, optional 3): **D=3** {Retrieve, Solve, Verify} and **D=7**
  {Retrieve, Extract, Plan, Solve-A, Solve-B, Aggregate, Verify}. Optional middle **D=5**
  {Retrieve, Extract, Plan, Solve, Verify} for a dose-response curve.
  At depth D both conditions run exactly D agents / D−1 handoffs.
- **Complexity K** (task attribute, 2 levels): **K=2** (2 reasoning sub-steps) and **K=4** (4 sub-steps).
  K is intrinsic to each task, not a crossing that multiplies the task set.

**Primary contrast = A vs C at each (D,K)**, then the **difference-in-differences across D and across K.**

## 4. Tasks (`data/tasks_trackB.jsonl`)
Two sources, both with deterministic ground truth:
1. **Synthetic parametric (primary):** a generator produces *K-step compositional* tasks — a short
   reference context stating K atomic facts/quantities, and a question requiring all K to be chained
   (fact-lookup → arithmetic/logical composition). K directly sets decomposition depth, difficulty is
   controlled, and the answer is exact (numeric/string). A **single-agent solvability screen** is run:
   any task a lone generalist solves ≥4/5 at K=4 with no pipeline is discarded as too easy (ensures the
   tasks actually require decomposition).
2. **Real multi-hop (ecological validity):** the MuSiQue 4-hop subset already vendored in
   `data/sources/musique/` (K≈4), reusing v2's QA evaluator path.
Target: **40 synthetic tasks (20 at K=2, 20 at K=4)** + **20 MuSiQue 4-hop** = 60 tasks. Frozen with SHA.

## 5. Held constant
Model `qwen2.5:7b` (primary; optionally one capable API model as a separate, costed robustness run —
see §9); temperature 0.7, top_p 0.9, num_ctx 16384, num_predict 1024; tools `lookup`+`calculator`
available to all agents in all conditions; seeds `[11,22,33]`; deterministic evaluator; F1–F9 classifier.
Instruction-block union is byte-identical across A and C at each depth; padding is the only length filler.

## 6. Hypotheses (directional; interaction is the contribution)
- **H-B1 (PRIMARY): depth × condition interaction on accuracy.** The A−C accuracy gap becomes more
  favorable to C as depth grows: `(A−C)|D=7  <  (A−C)|D=3` (C improves relative to A at higher depth).
- **H-B2: complexity × condition interaction on accuracy.** `(A−C)|K=4 < (A−C)|K=3`… i.e. gap widens with K.
- **H-B3 (mechanism): first-handoff overlap reduction (the v2 replicated effect) scales with depth**, and
  corrections / F1 / F7 shift with depth (exploratory).
- **H-B4 (cost/latency):** reported with execution-order logged from the start (see §8), so latency is
  analyzable this time; no directional claim pre-committed.
- **Null is informative:** if no interaction appears even at D=7/K=4, that is a strong, surprising
  boundary result ("specialization does not help even on deep, highly-decomposable tasks").

## 7. Analysis plan (`analysis/stats_trackB.py`)
- **Primary:** mixed-effects logistic regression `correct ~ condition * depth + (1|task)` (and a second
  with `* complexity`); the **interaction coefficient** is the pre-registered primary estimand, with 95% CI.
- **Secondary / non-parametric back-up:** per-task **difference-in-differences** — for each task compute
  ΔD=(A−C) accuracy at D=3 and at D=7; Wilcoxon signed-rank on (Δ_{D7} − Δ_{D3}). Same for K.
- Effect sizes: odds-ratio (model) + matched-pairs rank-biserial (DiD) + bootstrap 95% CI.
- Multiple testing: Holm across the pre-registered interaction family {depth×cond accuracy,
  complexity×cond accuracy}. Mechanism metrics reported as exploratory.
- Mandatory validity checks (reuse v2 audit code): runtime per-agent length match A≈C at every depth;
  padding neutrality (overlap-vs-content, no leaks); truncation logged, sensitivity re-run.

## 8. Procedure (bake in the v2 lessons)
1. Generate + screen tasks; freeze (SHA). 2. Pilot: 6 tasks × 2 conditions × 2 depths = 24 runs, validate
   harness + token/length match + that generalist actually degrades at D=7. 3. Freeze prompts/padding/config.
4. **Confirmatory, INTERLEAVED, with `execution_order` logged on every run** (so latency is de-confoundable
   — the exact gap that broke v2 Qwen latency). 5. `analysis/stats_trackB.py` + figures.
Runner is resumable/crash-safe (reuse the append+fsync pattern); re-running `--full` continues.

## 9. Compute budget (primary = 2 depths)
Runs = 60 tasks × 2 conditions × 2 depths {3,7} × 3 reps = **720 runs**
(lighter option: 40 synthetic only × 2 × 2 × 3 = 480 runs; fuller option: add D=5 → 1,080 runs).
On the local CPU box (~26 tok/s; v2 Qwen median ≈ 25 s/run at D=3, scaling ~linearly with depth so
D=7 ≈ 55–60 s), expect roughly **6–11 h for 720 runs**, overnight, resumable. An optional capable-API-model
robustness pass is a **paid** add-on and will be costed and approved before any spend.

## 10. New code (mostly extends existing files — no rewrite)
- `config/prompts/`: add `block_extract.txt, block_plan.txt, block_solveA.txt, block_solveB.txt,
  block_aggregate.txt` (same single-responsibility style as R/S/V) + matching `pad_*.txt`.
- `src/conditions.py`: generalize `build_agents(condition, depth)` — depth→ordered block list;
  generalist = all blocks per agent; specialized_length_matched = one block + padding to A's per-agent budget.
- `data/build_tasks_trackB.py`: parametric K-step task generator + single-agent solvability screen +
  MuSiQue 4-hop import.
- `src/runner_trackB.py`: thin wrapper over `runner_v2` adding the depth loop, **interleaved scheduling**,
  and `execution_order` logging (reuse `runner_rerun.py`'s scheduler).
- `analysis/stats_trackB.py`: the interaction models + DiD + validity checks (reuse `mechanism_v2` helpers).
- `preregistration_trackB.md` (this file).
