# Responsibility Specialization in LLM Multi-Agent Systems

A controlled experiment isolating **one** architectural mechanism — *responsibility
specialization* — while holding model, agent count, topology, tools, inference params,
tasks, and evaluation constant. See `preregistration.md` for the frozen design.

## The one intended difference
Both conditions run the same 3-agent linear pipeline. They differ only in how three
identical instruction blocks (Retrieve / Solve / Verify) are **allocated** — see
`src/conditions.py`. Everything else is shared code both conditions call, so the
conditions cannot silently diverge.

## Layout
```
config/        experiment.yaml + the 4 prompt files (preamble + R/S/V blocks)
data/          build_tasks.py -> tasks.jsonl (30 frozen tasks)
src/           schema, llm (Ollama), tools, agent, conditions, pipeline,
               runner (checkpointed), evaluator (deterministic), failures (F1-F9)
analysis/      stats.py (paired task-level + GEE), figures.py
results/       runs.jsonl, experiments.sqlite, stats_report.txt, figures/
```

## Requirements
- Ollama running locally with the model in `config/experiment.yaml` pulled
  (`ollama pull llama3` — or set `model:` to `llama3.1:8b-instruct` after pulling it).
- Python 3.11 with: `pyyaml pandas scipy matplotlib numpy` (and optionally `statsmodels`).

## Run
```bash
python data/build_tasks.py          # (re)build the task file
python src/runner.py --pilot        # smoke test: 3 tasks x 2 runs x 2 conditions
# freeze config/prompts/tasks, then:
python src/runner.py                # confirmatory: 30 x 5 x 2 = 300 runs (resumable)
python analysis/stats.py
python analysis/figures.py
```
The runner is crash-safe: each completed run is appended+fsync'd to `results/runs.jsonl`
and mirrored to SQLite; re-running skips already-completed `experiment_id`s and resumes.

## What this experiment answers
1. Does specialization change reliability? 2. Cost? 3. Latency? 4. Which behavioral
mechanisms explain the changes? 5. When does it help/hurt (difficulty × condition)?
6. Which failure pathways are introduced / reduced / shifted?
