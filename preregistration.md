# Pre-registration — Causal effect of Responsibility Specialization in LLM multi-agent systems

**Freeze this file (and `config/`, `data/tasks.jsonl`, all `src/` prompts) before the confirmatory run.**

## Research question
What is the causal effect of *responsibility specialization* — allocating distinct task
responsibilities to separate agents via role-specific instructions and bounded scopes —
in an LLM-agent system, when other major architectural factors are held constant?

## Design
Within-task, 2-condition comparison. Same 3-agent linear topology in both.

- **Generalist:** each of 3 agents holds the full responsibility set {Retrieve, Solve, Verify}.
- **Specialized:** agent1=Retrieve, agent2=Solve, agent3=Verify.

The manipulation is implemented as **allocation of identical instruction blocks** (see
`src/conditions.py`): the union of instruction content is byte-identical across conditions;
only *who holds which block* differs. This isolates specialization from prompt content/quality.

## Held constant
Model (`qwen2.5:7b`, chosen over llama3:8b after the pilot showed floor effects — see
calibration log below), agent count (3), tasks (the 30 in `data/tasks.jsonl`), topology
(strict linear, 2 handoffs, no loopback), tools (`lookup`, `calculator`, available to ALL
agents in BOTH conditions), inference params (temperature 0.7, top_p 0.9, num_ctx 8192,
num_predict 1024), seeds ([11,22,33,44,55] by run index), message/payload format, runs per
task (5), and the evaluation + failure-classification procedure.

## Dependent variables
- **Reliability (primary):** task success rate / final-answer correctness (deterministic check).
- **Cost:** input / output / total tokens (exact from Ollama), llm_calls; system-prompt
  overhead reported separately from task/generated tokens.
- **Latency:** wall-clock seconds per run.
- **Mechanism-level:** handoffs, repeated_work (inter-agent overlap), corrections, and the
  F1–F9 failure taxonomy (`src/failures.py`).

## Hypotheses (directional, but the contribution is mechanistic not just win/lose)
- H1: specialization changes success rate (two-sided).
- H2: specialization changes total token cost (two-sided).
- H3: specialization changes latency (two-sided).
- H4–H6: exploratory — behavioral mechanisms, task-condition interactions, failure-pathway shifts.

## Analysis plan
- **Primary:** paired task-level Wilcoxon signed-rank across the 30 tasks (per DV), because
  the 5 runs per task are NOT independent. Holm correction across the 3 primary DVs.
- **Secondary:** GEE logistic for correctness clustered by task (needs `statsmodels`).
- Reliability reported with task-mean bootstrap 95% CIs. Difficulty × condition breakdown.

## Procedure
1. **Pilot** (`python src/runner.py --pilot`, 3 tasks × 2 runs × 2 conditions) — debug the
   harness and, if needed, extend the failure taxonomy. Pilot data is NOT pooled with confirmatory.
2. **Freeze** prompts, tasks, taxonomy, config.
3. **Confirmatory:** `python src/runner.py` → 30 × 5 × 2 = 300 runs.
4. `python analysis/stats.py` and `python analysis/figures.py`.

## Pilot calibration log (2026-09-26, before freeze; applied identically to both conditions)
1. **Model:** llama3:8b → qwen2.5:7b — pilot showed floor effects (gen 17% / spec 0%).
2. **Tool-loop bug fix (`agent.py`):** agents that emitted a `TOOL_CALL` and a premature
   `FINAL_ANSWER` in the same message had the tool skipped, so `lookup` never ran and answers
   were ungrounded (e.g. a Canberra context answered "Madrid"). Tools now execute before any
   final answer is accepted. This was a correctness-critical harness bug, not prompt tuning.
3. **Evaluator (`evaluator.py`):** numeric checker made duration-/unit-aware so a correct answer
   phrased in mixed units ("5 hours and 30 minutes" == 5.5) is not scored wrong.
4. **Failure classifier (`failures.py`):** F2/F5 required-info matching normalized (numeric
   boundary match; ignore $ and commas) to stop mis-firing on correct answers.
After these, pilot accuracy was ~83% in both conditions and mechanism signals appeared as
predicted (F1/F7 only in specialized). Pilot data is archived under `results/pilot_*_archive/`
and NOT pooled with confirmatory data.

## Known limitations (documented, not fixed, to preserve validity)
- F1 (wrong-responsibility) and F7 (out-of-role tool use) are structurally only possible in
  the specialized condition; reported as a *newly introduced* failure mode, not as "worse".
- Failure classifiers are heuristic; subjective categories may be re-checked by blind manual labeling.
- Ollama seeding reduces but does not eliminate GPU-level nondeterminism.
- System-prompt token overhead differs by design (generalist carries all blocks ×3); this is
  a genuine consequence of specialization, reported transparently, not a confound.
