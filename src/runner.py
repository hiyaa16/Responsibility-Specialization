"""Experiment controller: tasks x runs x conditions, with crash-safe checkpointing.

Run:  python src/runner.py            # full run per config/experiment.yaml
      python src/runner.py --pilot    # pilot: few runs, for shaking out bugs

Crash safety (per user's crash-safe-long-jobs preference): each completed run is
appended as ONE line to results/runs.jsonl and fsync'd, and mirrored into SQLite.
Re-running skips any experiment_id already present, so an abrupt stop never loses or
corrupts completed runs and the batch resumes where it left off.
"""
from __future__ import annotations
import os
import sys
import json
import time
import sqlite3
import argparse
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from schema import RunRecord, load_tasks, AgentTurn  # noqa: E402
from llm import OllamaClient                          # noqa: E402
from conditions import build_agents                   # noqa: E402
from pipeline import run_pipeline                      # noqa: E402
from evaluator import evaluate                         # noqa: E402
from failures import classify                          # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _p(rel):
    return os.path.join(ROOT, rel)


def load_config():
    with open(_p("config/experiment.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def existing_ids(path):
    done = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["experiment_id"])
                    except Exception:
                        pass
    return done


def init_sqlite(path):
    con = sqlite3.connect(path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            experiment_id TEXT PRIMARY KEY, task_id TEXT, condition TEXT,
            run_index INTEGER, seed INTEGER, model TEXT, temperature REAL,
            difficulty TEXT, category TEXT, handoffs INTEGER,
            system_prompt_tokens INTEGER, input_tokens INTEGER, output_tokens INTEGER,
            total_tokens INTEGER, llm_calls INTEGER, latency_seconds REAL,
            truncated INTEGER, final_answer TEXT, correct INTEGER,
            failure_types TEXT, corrections INTEGER, repeated_work REAL
        )""")
    con.commit()
    return con


def append_jsonl(path, record: RunRecord):
    with open(path, "a", encoding="utf-8") as f:
        f.write(record.to_json() + "\n")
        f.flush()
        os.fsync(f.fileno())


def insert_sqlite(con, r: RunRecord):
    con.execute(
        "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (r.experiment_id, r.task_id, r.condition, r.run_index, r.seed, r.model,
         r.temperature, r.difficulty, r.category, r.handoffs, r.system_prompt_tokens,
         r.input_tokens, r.output_tokens, r.total_tokens, r.llm_calls, r.latency_seconds,
         int(r.truncated), r.final_answer, int(r.correct),
         json.dumps(r.failure_types), r.corrections, r.repeated_work))
    con.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    pilot = args.pilot or cfg.get("pilot", False)
    runs_per_task = cfg["pilot_runs_per_task"] if pilot else cfg["runs_per_task"]
    seeds = cfg["seeds"][:runs_per_task]

    client = OllamaClient(
        url=cfg["ollama_url"], model=cfg["model"], temperature=cfg["temperature"],
        top_p=cfg["top_p"], num_ctx=cfg["num_ctx"], num_predict=cfg["num_predict"],
        timeout_s=cfg["request_timeout_s"],
    )

    tasks = load_tasks(_p(cfg["tasks_path"]))
    if pilot:
        pilots = [t for t in tasks if t.difficulty == "pilot"]
        tasks = pilots if pilots else tasks[:3]

    runs_out = _p(cfg["runs_out"])
    os.makedirs(os.path.dirname(runs_out), exist_ok=True)
    done = existing_ids(runs_out)
    con = init_sqlite(_p(cfg["sqlite_out"]))

    total = len(cfg["conditions"]) * len(tasks) * runs_per_task
    print(f"Planned runs: {total} ({len(cfg['conditions'])} conditions x "
          f"{len(tasks)} tasks x {runs_per_task} runs). Already done: {len(done)}")

    n = 0
    for condition in cfg["conditions"]:
        agents = build_agents(condition, client, cfg["max_tool_iters"])
        sys_tok_est = sum(len(a.system_prompt) // 4 for a in agents)  # C1 cost decomposition
        for task in tasks:
            for run_index in range(runs_per_task):
                seed = seeds[run_index]
                exp_id = f"{condition}::{task.task_id}::run{run_index}"
                n += 1
                if exp_id in done:
                    continue
                try:
                    out = run_pipeline(agents, task, seed)
                except Exception as e:
                    print(f"  [{n}/{total}] {exp_id} ERROR: {e}")
                    continue

                final_answer, correct = evaluate(out["final_text"], task)
                fails, corrections, repeated = classify(
                    task, out["turns"], out["final_text"], correct)

                if out["truncated"]:
                    print(f"  WARNING truncation on {exp_id} — raise num_ctx/num_predict (C5)")

                rec = RunRecord(
                    experiment_id=exp_id, task_id=task.task_id, condition=condition,
                    run_index=run_index, seed=seed, model=cfg["model"],
                    temperature=cfg["temperature"], agent_count=cfg["agent_count"],
                    difficulty=task.difficulty, category=task.category,
                    agent_outputs=[a.__dict__ for a in out["turns"]],
                    tool_calls=out["tool_calls"], handoffs=out["handoffs"],
                    system_prompt_tokens=sys_tok_est,
                    input_tokens=out["input_tokens"], output_tokens=out["output_tokens"],
                    total_tokens=out["total_tokens"], llm_calls=out["llm_calls"],
                    latency_seconds=round(out["latency_seconds"], 3),
                    truncated=out["truncated"], final_answer=final_answer, correct=correct,
                    failure_types=fails, corrections=corrections, repeated_work=repeated,
                )
                append_jsonl(runs_out, rec)
                insert_sqlite(con, rec)
                print(f"  [{n}/{total}] {exp_id} correct={correct} "
                      f"tok={rec.total_tokens} {rec.latency_seconds}s fails={fails}")

    con.close()
    print("Done.")


if __name__ == "__main__":
    main()
