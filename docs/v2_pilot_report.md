# v2 pilot report — harness validation (pre-approval for the 750-run)

**Status:** harness **VALIDATED**. The full 48-run pilot cannot complete in this environment in
one shot (see Finding B), but the 15-point validation is satisfied by **15 real runs** across all
3 conditions and all 3 task sources: 5 confirmatory-pilot runs (`results/v2_pilot/`) + 10 smoke
runs (`results/v2_pilot_smoke/`). All runs use `qwen2.5:7b`, CPU-only. Existing v1 data untouched.

## 15-point checklist (Part 11)
| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | 3 conditions construct correctly | PASS | per-agent token lines + roles printed for all 3 |
| 2 | Generalist gets R+S+V | PASS | roles all `generalist`; per-agent = 451 tok = preamble+R+S+V |
| 3 | Specialized gets R/S/V separately | PASS | roles retriever/solver/verifier; tokens [329,291,305] |
| 4 | Length-matched = R/S/V + inert padding | PASS | roles retriever/solver/verifier; tokens [452,451,451] |
| 5 | Prompt token lengths actually matched | PASS | C within +0.21%/0%/0% of A's 486 full/agent; totals A=1353, B=925, C=1354 |
| 6 | Tools identical across conditions | PASS | all call the same `tools.py`; no tool restriction by role |
| 7 | Model is qwen2.5:7b | PASS | every row `model="qwen2.5:7b"` |
| 8 | Linear topology unchanged | PASS | `handoffs=2`; `pipeline.py` unmodified |
| 9 | Tasks load correctly | PASS | 8 pilot tasks (3 MuSiQue + 3 HotpotQA + 2 hand) loaded |
| 10 | Evaluation works | PASS | QA 5/8 correct (not ceiling); numeric/set evaluated |
| 11 | Tokens logged correctly | PASS | real per-agent + input/output/total per row |
| 12 | Latency logged | PASS | median 205 s, max 1136 s (non-error) |
| 13 | **Timeouts logged, not dropped** | PASS | `generalist::MQ16::run0` → `timeout=True` row; bad-URL → error row |
| 14 | F1–F9 logged | PASS | all categories seen; **F1/F7 only in B & C**, zero in generalist |
| 15 | Stored separately from v1's 300 | PASS | writes only to `results/v2_pilot*/`; `results/runs.jsonl` untouched |

## Mechanism sanity (real data)
- **F1 (wrong-responsibility) and F7 (out-of-role tool use)** occurred only in specialized and
  specialized_length_matched (2 each), never in generalist — the v1 structural asymmetry reproduces.
- **F2** fires on most QA runs (as in v1). Per Part 17 this is flagged for **post-hoc investigation**
  of heuristic over-trigger (multi-word-entity substring matching), **not changed** here.
- **No truncation** at `num_ctx=16384` on any run.
- QA reliability 5/8 and hand-authored mostly failing ⇒ the harder set is **not** at ceiling (the v1 limitation this experiment targets).

## Findings requiring a decision before the 750-run
**A. Per-call timeout too low for generalist on hard/4-hop tasks.** `generalist::MQ16::run0`
(4-hop) exceeded the 600 s per-call cap and was logged as a timeout. If left at 600 s, the full
run would convert many legitimate (slow) generalist-hard runs into timeouts, biasing generalist
reliability **downward** — a confound. **Action taken:** `request_timeout_s` raised 600 → **1800**
in `config/experiment_v2.yaml` (a cutoff only; it does not alter any successful run's output).
The pilot used 600 s deliberately, to surface this quickly.

**B. This environment kills long background jobs (~15–35 min).** The pilot was killed twice after
2 and then 3 more runs; the resumable runner picked up each time (`Already done: N`). A 48-run
pilot (~5 h) — let alone the 750-run (~days) — cannot run to completion here in one background
job. **The 750-run must be executed where long jobs are not killed** (a normal local terminal /
overnight / a GPU box). The runner is crash-safe and resumable, so it can also be run in chunks.

## Recommended commands (resumable; safe to interrupt/resume)
```bash
# finish the 48-run pilot (resumes from wherever it stopped)
python src/runner_v2.py --pilot
python analysis/stats_v2.py results/v2_pilot
python analysis/figures_v2.py results/v2_pilot

# confirmatory 750-run (ONLY after you approve)
python src/runner_v2.py
python analysis/stats_v2.py
python analysis/figures_v2.py
```

## Recommendation
The harness is correct and all design checks pass on real runs. Before the 750-run I recommend:
(1) keep the `request_timeout_s=1800` fix; (2) run the confirmatory experiment on a machine/session
that will not kill a multi-hour job (ideally with a GPU — CPU puts the full run at ~days). I will
not launch the 750-run without your approval.
