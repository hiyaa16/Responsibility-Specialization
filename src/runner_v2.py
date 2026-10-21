"""v2 experiment controller (prompt-length-control study). Separate from v1's runner.py so
the existing 300 runs are never touched.

Differences from v1 runner (all additive):
  * reads config/experiment_v2.yaml; 3 conditions incl. specialized_length_matched
  * writes to a v2 output namespace (results/v2 or results/v2_pilot); never to results/runs.jsonl
  * records REAL per-agent system-prompt token counts (src/tokens.py), not len//4
  * TIMEOUTS / ERRORS ARE LOGGED AS ROWS (timeout/error fields), never silently dropped
  * stamps experiment_version, source, source_id, hops on every run
Crash-safe + resumable exactly like v1 (append+fsync jsonl, INSERT OR REPLACE sqlite, skip
already-present experiment_ids).

Run:  python src/runner_v2.py --pilot      # 8 tasks x 2 runs x 3 conditions -> results/v2_pilot
      python src/runner_v2.py              # 50 tasks x 5 runs x 3 conditions -> results/v2
"""
from __future__ import annotations
import os
import sys
import json
import socket
import sqlite3
import argparse
import urllib.error
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from schema import RunRecord, load_tasks            # noqa: E402
from llm import OllamaClient                         # noqa: E402
from conditions import build_agents                  # noqa: E402
from pipeline import run_pipeline                     # noqa: E402
from evaluator import evaluate                        # noqa: E402
from failures import classify                         # noqa: E402
import tokens as tokmod                               # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _p(rel):
    return os.path.join(ROOT, rel)


def load_config():
    with open(_p("config/experiment_v2.yaml"), "r", encoding="utf-8") as f:
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
            experiment_id TEXT PRIMARY KEY, experiment_version TEXT, task_id TEXT,
            source TEXT, hops INTEGER, condition TEXT, run_index INTEGER, seed INTEGER,
            model TEXT, temperature REAL, difficulty TEXT, category TEXT, handoffs INTEGER,
            system_prompt_tokens INTEGER, input_tokens INTEGER, output_tokens INTEGER,
            total_tokens INTEGER, llm_calls INTEGER, latency_seconds REAL, truncated INTEGER,
            timeout INTEGER, error TEXT, final_answer TEXT, correct INTEGER,
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
        "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (r.experiment_id, r.experiment_version, r.task_id, r.source, r.hops, r.condition,
         r.run_index, r.seed, r.model, r.temperature, r.difficulty, r.category, r.handoffs,
         r.system_prompt_tokens, r.input_tokens, r.output_tokens, r.total_tokens, r.llm_calls,
         r.latency_seconds, int(r.truncated), int(r.timeout), r.error, r.final_answer,
         int(r.correct), json.dumps(r.failure_types), r.corrections, r.repeated_work))
    con.commit()


def _is_timeout(e: Exception) -> bool:
    if isinstance(e, (socket.timeout, TimeoutError)):
        return True
    if isinstance(e, urllib.error.URLError) and isinstance(getattr(e, "reason", None), (socket.timeout, TimeoutError)):
        return True
    return "timed out" in str(e).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    pilot = args.pilot
    runs_per_task = cfg["pilot_runs_per_task"] if pilot else cfg["runs_per_task"]
    seeds = cfg["seeds"][:runs_per_task]
    version = cfg["experiment_version"] + ("_pilot" if pilot else "")

    client = OllamaClient(
        url=cfg["ollama_url"], model=cfg["model"], temperature=cfg["temperature"],
        top_p=cfg["top_p"], num_ctx=cfg["num_ctx"], num_predict=cfg["num_predict"],
        timeout_s=cfg["request_timeout_s"],
    )

    tasks = load_tasks(_p(cfg["tasks_path"]))
    if pilot:
        ids = set(cfg["pilot_task_ids"])
        tasks = [t for t in tasks if t.task_id in ids]

    out_dir = _p(cfg["pilot_dir"] if pilot else cfg["out_dir"])
    os.makedirs(out_dir, exist_ok=True)
    runs_out = os.path.join(out_dir, "runs.jsonl")
    sqlite_out = os.path.join(out_dir, "experiments.sqlite")
    done = existing_ids(runs_out)
    con = init_sqlite(sqlite_out)

    total = len(cfg["conditions"]) * len(tasks) * runs_per_task
    print(f"[{version}] Planned runs: {total} ({len(cfg['conditions'])} conditions x "
          f"{len(tasks)} tasks x {runs_per_task} runs). Already done: {len(done)}")
    print(f"Output -> {runs_out}")

    n = 0
    for condition in cfg["conditions"]:
        agents = build_agents(condition, client, cfg["max_tool_iters"])
        # REAL per-agent system-prompt tokens (same tokenizer the model uses)
        per_agent_tok = [tokmod.system_prompt_tokens(a.system_prompt) for a in agents]
        sys_tok = sum(per_agent_tok)
        print(f"  condition={condition}  system_prompt_tokens/agent={per_agent_tok} (sum={sys_tok})")
        for task in tasks:
            for run_index in range(runs_per_task):
                seed = seeds[run_index]
                exp_id = f"{version}::{condition}::{task.task_id}::run{run_index}"
                n += 1
                if exp_id in done:
                    continue

                timeout = False
                error = ""
                try:
                    out = run_pipeline(agents, task, seed)
                except Exception as e:                       # LOG, do not drop (Part 16)
                    timeout = _is_timeout(e)
                    error = f"{type(e).__name__}: {e}"[:300]
                    out = None

                if out is None:
                    rec = RunRecord(
                        experiment_id=exp_id, task_id=task.task_id, condition=condition,
                        run_index=run_index, seed=seed, model=cfg["model"],
                        temperature=cfg["temperature"], agent_count=cfg["agent_count"],
                        difficulty=task.difficulty, category=task.category,
                        agent_outputs=[], tool_calls=[], handoffs=0,
                        system_prompt_tokens=sys_tok, input_tokens=0, output_tokens=0,
                        total_tokens=0, llm_calls=0, latency_seconds=0.0, truncated=False,
                        final_answer="", correct=False, failure_types=["F9"], corrections=0,
                        repeated_work=0.0, experiment_version=version, source=task.source,
                        source_id=task.source_id, hops=task.hops, timeout=timeout, error=error,
                        system_prompt_tokens_per_agent=per_agent_tok)
                    append_jsonl(runs_out, rec)
                    insert_sqlite(con, rec)
                    print(f"  [{n}/{total}] {exp_id} FAILED timeout={timeout} {error}")
                    continue

                final_answer, correct = evaluate(out["final_text"], task)
                fails, corrections, repeated = classify(task, out["turns"], out["final_text"], correct)
                if out["truncated"]:
                    print(f"  WARNING truncation on {exp_id} — raise num_ctx/num_predict (C5)")

                rec = RunRecord(
                    experiment_id=exp_id, task_id=task.task_id, condition=condition,
                    run_index=run_index, seed=seed, model=cfg["model"],
                    temperature=cfg["temperature"], agent_count=cfg["agent_count"],
                    difficulty=task.difficulty, category=task.category,
                    agent_outputs=[a.__dict__ for a in out["turns"]],
                    tool_calls=out["tool_calls"], handoffs=out["handoffs"],
                    system_prompt_tokens=sys_tok, input_tokens=out["input_tokens"],
                    output_tokens=out["output_tokens"], total_tokens=out["total_tokens"],
                    llm_calls=out["llm_calls"], latency_seconds=round(out["latency_seconds"], 3),
                    truncated=out["truncated"], final_answer=final_answer, correct=correct,
                    failure_types=fails, corrections=corrections, repeated_work=repeated,
                    experiment_version=version, source=task.source, source_id=task.source_id,
                    hops=task.hops, timeout=False, error="",
                    system_prompt_tokens_per_agent=per_agent_tok)
                append_jsonl(runs_out, rec)
                insert_sqlite(con, rec)
                print(f"  [{n}/{total}] {exp_id} correct={correct} tok={rec.total_tokens} "
                      f"{rec.latency_seconds}s fails={fails}")

    con.close()
    print("Done.")


if __name__ == "__main__":
    main()
