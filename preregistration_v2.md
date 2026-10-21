# Pre-registration v2 — Prompt-length-controlled replication on harder tasks

**Freeze this file + `config/experiment_v2.yaml` + `data/tasks_v2.jsonl` + `config/prompts/pad_*.txt`
before the confirmatory 750-run experiment.** The existing 300 v1 runs are a separate experiment
and are not modified, re-run, or pooled.

## Motivation (two v1 limitations this addresses)
1. **Prompt-length confound (C1).** In v1 the generalist carries all three responsibility blocks
   per agent while the specialist carries one, so v1's lower specialist token cost is partly just
   shorter prompts. (v1's prereg treated this as "a genuine consequence of specialization";
   v2 deliberately *reframes it as a confound to control* — a transparent change of stance, not a
   silent edit.)
2. **Ceiling (C ~0.86 both conditions in v1).** v2 uses 50 harder multi-hop / compositional tasks.

## Design — within-task, 3 conditions, same 3-agent linear topology
- **A generalist:** every agent holds {Retrieve, Solve, Verify}.
- **B specialized:** agent1=Retrieve, agent2=Solve, agent3=Verify.
- **C specialized_length_matched:** B's allocation **plus** semantically-inert padding so each
  agent's system-prompt token budget matches A's.

## Held constant with v1
Model `qwen2.5:7b`; linear topology (3 agents, 2 handoffs, no loopback); both tools
(`lookup`, `calculator`) available to all agents in all conditions; temperature 0.7, top_p 0.9,
num_predict 1024; seeds [11,22,33,44,55]; payload format; deterministic evaluator; F1–F9
classifier (unchanged).

## Deliberate, documented deviations from v1 (required by the harder tasks)
- `num_ctx` 8192 → **16384**: v2 contexts are ~10× larger; 8192 would truncate (C5). Identical
  across the 3 conditions, so non-confounding.
- `request_timeout_s` 300 → **600**: CPU prompt-eval of large contexts is slow. **Timeouts are
  logged as rows** (`timeout`/`error` fields), never silently dropped.
- Evaluator gains an `answer_type="qa"` path (SQuAD normalization + aliases) for multi-hop QA;
  v1's numeric/set/string logic is byte-identical.
- **MuSiQue capped to 10 paragraphs** (2 gold + 8 distractors) instead of its native 20, to match
  the HotpotQA distractor budget and keep CPU runtime tractable. Applied identically to all
  conditions. This slightly reduces the retrieval haystack vs MuSiQue-standard (documented).

## The manipulation lives only in `src/conditions.py`
A and B branches are byte-identical to v1. C appends frozen padding. The padding is ordinary
neutral English (NOT lorem/OOD, NOT the real R/S/V text) containing no task info, reasoning,
role info, examples, hints, or constraints (see `data/build_padding.py`).

## Frozen artifacts (hashes)
- `data/tasks_v2.jsonl` sha256 `3cf99d728714e12c913eff5d4d84d9fa6a19371fbabdc2372e76aff5259e170d`
- `config/prompts/pad_retrieve.txt` sha256 `be965aa3f3c2e974…`
- `config/prompts/pad_solve.txt`    sha256 `50a0968ac10c0734…`
- `config/prompts/pad_verify.txt`   sha256 `a733b9ca8e2c969c…`
- Prompt-length match: each C agent within +0.21%/0%/0% of A's 486-token per-agent budget
  (`results/v2/prompt_length_match_report.txt`); condition totals A=1353, B=925, C=1354.
- Benchmark provenance + source SHAs: `data/sources/PROVENANCE.md`.

## Dependent variables
- **Reliability (primary):** task success (deterministic).
- **Cost:** system-prompt / input / output / total tokens (exact from Ollama); llm_calls.
- **Latency:** wall-clock seconds (environment-dependent — local CPU; no hardware-independent claims).
- **Mechanism:** handoffs, repeated_work, corrections, F1–F9.

## Hypotheses
- H1 (reliability): A, B, C do not differ practically (pre-registered equivalence margin ±0.05).
- H2 (cost): B < A on total tokens. **Critical test B vs C:** does the token saving survive once
  prompt length is matched? If it vanishes at C, the v1 "efficiency" was partly a length artifact.
- H3 (latency): exploratory, environment-dependent.
- H4–H6: mechanism / interaction / failure-pathway shifts (exploratory).

## Analysis plan (`analysis/stats_v2.py`)
Task-level means first (5 runs/task are not independent). Omnibus Friedman across A/B/C per DV;
post-hoc pairwise Wilcoxon (A–B, B–C, A–C) Holm-corrected across the 3 pairs; effect size =
matched-pairs rank-biserial + median diff + bootstrap 95% CI. Reliability equivalence via **TOST**,
margin ±0.05, 90% CI (see `docs/v2_equivalence_margin.md`). Breakdowns by source and difficulty.
Token decomposition focuses on **B vs C**. Failure frequencies with denominators; F2 over-trigger
to be *investigated, not silently changed* (Part 17).

## Procedure
1. **Pilot** (`python src/runner_v2.py --pilot`): 8 tasks × 2 runs × 3 conditions = 48 runs →
   `results/v2_pilot/`. Validate the 15-point checklist (see `docs/v2_pilot_report.md`).
2. **Freeze** (this file + config + tasks + padding).
3. **Confirmatory** (`python src/runner_v2.py`): 50 × 5 × 3 = 750 runs → `results/v2/` (resumable).
4. `python analysis/stats_v2.py` and `python analysis/figures_v2.py`.
Pilot data is archived and NOT pooled with confirmatory.

## Human validation (later, not now — Part 18)
After 750 runs: ~100 sampled traces, 2 annotators, F1–F9 guide, Cohen's κ vs automatic labels.
