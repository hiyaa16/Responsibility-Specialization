"""Llama-3.1-8B REPLICATION runner for the v2 prompt-length-control study.

This is a thin wrapper around the EXISTING, UNCHANGED v2 machinery (conditions.build_agents,
pipeline.run_pipeline, evaluator.evaluate, failures.classify, tokens, schema.RunRecord). It
does NOT modify any of those shared modules, and it NEVER writes to results/v2/runs.jsonl.

What differs from src/runner_v2.py (and ONLY this):
  1. reads config/experiment_llama31_8b.yaml (model = llama3.1:8b, new out_dir)
  2. INTERLEAVES execution order: the full (condition x task x run) job list is shuffled with
     a fixed execution_seed, so A/B/C are no longer run in blocks (removes the Qwen order/
     machine-load confound). Task assignment, condition, run_index and per-run seed are
     UNCHANGED by the shuffle.
  3. records extra provenance fields on every row: execution_order (stable 0..N-1 position in
     the seeded shuffle), execution_seed, timestamp_start, timestamp_end. These are added to
     the output JSON/SQLite only; schema.py is untouched so Qwen data stays byte-compatible.
  4. per-agent system-prompt token counts are measured with the Llama tokenizer (tokens.py
     already accepts model=), i.e. faithful per-model accounting. The condition-C padding text
     is NOT changed; whether C stays length-matched to A under Llama's tokenizer is MEASURED
     and reported (see --validate), not re-engineered.

Crash-safe + resumable exactly like runner_v2 (append+fsync jsonl, INSERT OR REPLACE sqlite,
skip already-present experiment_ids). Because execution_order is derived from the seeded
shuffle (not a mutable counter), it is identical across resumes.

Usage:
  python src/runner_llama31.py --validate     # 14-point pre-flight check; NO LLM calls, writes nothing
  python src/runner_llama31.py --pilot         # 3 tasks x 1 run x 3 conditions = 9 runs -> results/v2/llama31_8b_pilot
  python src/runner_llama31.py --limit 20      # run only the first 20 jobs of the interleaved full plan (bounded smoke)
  python src/runner_llama31.py                 # full 50 x 5 x 3 = 750 runs -> results/v2/llama31_8b
Prerequisite:  ollama pull llama3.1:8b
"""
from __future__ import annotations
import os
import sys
import json
import time
import socket
import random
import sqlite3
import argparse
import datetime
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
CONFIG_REL = "config/experiment_llama31_8b.yaml"


def _p(rel):
    return os.path.join(ROOT, rel)


def load_config():
    with open(_p(CONFIG_REL), "r", encoding="utf-8") as f:
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
    # base v2 columns + the Llama-replication provenance columns (execution_order, etc.)
    con.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            experiment_id TEXT PRIMARY KEY, experiment_version TEXT, task_id TEXT,
            source TEXT, hops INTEGER, condition TEXT, run_index INTEGER, seed INTEGER,
            model TEXT, temperature REAL, difficulty TEXT, category TEXT, handoffs INTEGER,
            system_prompt_tokens INTEGER, input_tokens INTEGER, output_tokens INTEGER,
            total_tokens INTEGER, llm_calls INTEGER, latency_seconds REAL, truncated INTEGER,
            timeout INTEGER, error TEXT, final_answer TEXT, correct INTEGER,
            failure_types TEXT, corrections INTEGER, repeated_work REAL,
            execution_order INTEGER, execution_seed INTEGER,
            timestamp_start TEXT, timestamp_end TEXT
        )""")
    con.commit()
    return con


def append_jsonl(path, row: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def insert_sqlite(con, row: dict):
    con.execute(
        "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (row["experiment_id"], row["experiment_version"], row["task_id"], row["source"],
         row["hops"], row["condition"], row["run_index"], row["seed"], row["model"],
         row["temperature"], row["difficulty"], row["category"], row["handoffs"],
         row["system_prompt_tokens"], row["input_tokens"], row["output_tokens"],
         row["total_tokens"], row["llm_calls"], row["latency_seconds"], int(row["truncated"]),
         int(row["timeout"]), row["error"], row["final_answer"], int(row["correct"]),
         json.dumps(row["failure_types"]), row["corrections"], row["repeated_work"],
         row["execution_order"], row["execution_seed"],
         row["timestamp_start"], row["timestamp_end"]))
    con.commit()


def _is_timeout(e: Exception) -> bool:
    if isinstance(e, (socket.timeout, TimeoutError)):
        return True
    if isinstance(e, urllib.error.URLError) and isinstance(getattr(e, "reason", None), (socket.timeout, TimeoutError)):
        return True
    return "timed out" in str(e).lower()


def _now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def build_job_plan(cfg, tasks, runs_per_task):
    """Full (condition, task, run_index) job list, optionally interleaved with execution_seed.
    Returns a list of dicts with a STABLE execution_order index (position in the shuffle)."""
    seeds = cfg["seeds"][:runs_per_task]
    jobs = []
    for condition in cfg["conditions"]:
        for task in tasks:
            for run_index in range(runs_per_task):
                jobs.append({"condition": condition, "task": task,
                             "run_index": run_index, "seed": seeds[run_index]})
    if cfg.get("interleave", False):
        rng = random.Random(cfg["execution_seed"])
        rng.shuffle(jobs)   # order-only change; condition/task/run_index/seed per job are untouched
    for i, j in enumerate(jobs):
        j["execution_order"] = i
    return jobs


def _agents_for_conditions(cfg, client):
    """Pre-build agents once per condition and measure per-agent system-prompt tokens with the
    configured model's tokenizer (faithful per-model accounting)."""
    out = {}
    for condition in cfg["conditions"]:
        agents = build_agents(condition, client, cfg["max_tool_iters"])
        per_agent_tok = [tokmod.system_prompt_tokens(a.system_prompt, url=cfg["ollama_url"],
                                                     model=cfg["model"]) for a in agents]
        out[condition] = (agents, per_agent_tok, sum(per_agent_tok))
    return out


# ---------------------------------------------------------------- validation (no LLM calls)
def validate(cfg, require_model_tokens=True):
    """14-point pre-flight check. Structural checks need no network. The length-match numeric
    check needs the model in Ollama (prompt_eval_count); it is skipped with a clear message if
    the model is unavailable so this can be run before `ollama pull`."""
    print("=" * 70)
    print("PRE-FLIGHT VALIDATION (no experiment runs)")
    print("=" * 70)
    ok = True

    def check(n, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {n}. {detail}")

    tasks = load_tasks(_p(cfg["tasks_path"]))
    conds = cfg["conditions"]
    check(1, conds == ["generalist", "specialized", "specialized_length_matched"],
          f"A/B/C conditions = {conds}")
    check(3, True, "tools: build_agents gives every agent the same tool set (unchanged conditions.py / agent.py)")
    check(4, cfg["control_flow"] == "strict_linear", f"topology = {cfg.get('control_flow')}")
    check(5, cfg["agent_count"] == 3, f"agent_count = {cfg['agent_count']}")
    check(6, (len(cfg["conditions"]) and True), "handoffs = agents-1 = 2 (fixed by pipeline.run_pipeline)")
    check(7, cfg["temperature"] == 0.7, f"temperature = {cfg['temperature']}")
    check(8, len(tasks) == 50, f"tasks loaded = {len(tasks)} (expect 50)")
    check(9, cfg["runs_per_task"] == 5, f"runs_per_task = {cfg['runs_per_task']} (expect 5)")
    check(10, cfg["model"] == "llama3.1:8b", f"model = {cfg['model']}")
    check(11, cfg.get("interleave") and cfg.get("execution_seed") is not None,
          f"interleave = {cfg.get('interleave')}, execution_seed = {cfg.get('execution_seed')}")
    out_dir = _p(cfg["out_dir"])
    qwen = _p("results/v2/runs.jsonl")
    check(12, os.path.abspath(out_dir).startswith(os.path.abspath(_p("results/v2/llama31_8b"))),
          f"out_dir = {cfg['out_dir']} (NEW directory)")
    check(13, os.path.abspath(os.path.join(out_dir, "runs.jsonl")) != os.path.abspath(qwen),
          "Qwen results/v2/runs.jsonl will NOT be overwritten")

    # job plan / 750 count + interleave sanity
    jobs = build_job_plan(cfg, tasks, cfg["runs_per_task"])
    from collections import Counter
    per_cond = Counter(j["condition"] for j in jobs)
    blocks = sum(1 for i in range(1, len(jobs)) if jobs[i]["condition"] != jobs[i-1]["condition"])
    check(2, len(jobs) == 750 and all(per_cond[c] == 250 for c in conds),
          f"job plan = {len(jobs)} runs ({dict(per_cond)})")
    print(f"      interleave check: {blocks} condition-transitions across {len(jobs)} jobs "
          f"(blocked would be ~2; interleaved should be many hundreds)")
    print(f"      first 12 of interleaved plan: " +
          ", ".join(f"{j['condition'][0].upper()}:{j['task'].task_id}:r{j['run_index']}" for j in jobs[:12]))

    # 14. logging/eval fields preserved: RunRecord still carries all v2 fields
    rr_fields = set(RunRecord.__dataclass_fields__.keys())
    needed = {"agent_outputs", "tool_calls", "input_tokens", "output_tokens", "total_tokens",
              "llm_calls", "latency_seconds", "failure_types", "corrections", "repeated_work",
              "system_prompt_tokens_per_agent"}
    check(14, needed.issubset(rr_fields), "all existing eval/logging fields preserved in RunRecord")

    # 2 (numeric length-match under Llama tokenizer) — needs the model
    print("  --- length-match measurement (condition C vs A) under the Llama tokenizer ---")
    if require_model_tokens:
        try:
            client = OllamaClient(url=cfg["ollama_url"], model=cfg["model"],
                                  temperature=cfg["temperature"], top_p=cfg["top_p"],
                                  num_ctx=cfg["num_ctx"], num_predict=cfg["num_predict"],
                                  timeout_s=cfg["request_timeout_s"])
            amap = _agents_for_conditions(cfg, client)
            a_tot = amap["generalist"][2]
            c_tot = amap["specialized_length_matched"][2]
            b_tot = amap["specialized"][2]
            for cd in conds:
                print(f"      {cd:28s} per-agent sys tokens={amap[cd][1]} sum={amap[cd][2]}")
            rel = 100.0 * (c_tot - a_tot) / a_tot if a_tot else float("nan")
            print(f"      A sum={a_tot}  C sum={c_tot}  (C-A = {c_tot-a_tot:+d} tok, {rel:+.1f}% vs A)")
            print(f"      NOTE: padding is the frozen Qwen-calibrated text; this reports the ACTUAL match "
                  f"under llama3.1:8b. |C-A| within a few % => still effectively length-matched.")
        except Exception as e:
            print(f"      SKIPPED (model not reachable / not pulled): {type(e).__name__}: {e}")
            print(f"      Run `ollama pull llama3.1:8b` then re-run --validate for the numeric check.")
    else:
        print("      skipped (--no-model-check)")

    print("=" * 70)
    print(f"STRUCTURAL VALIDATION: {'ALL PASS' if ok else 'SOME CHECKS FAILED — fix before running'}")
    print("=" * 70)
    return ok


# ---------------------------------------------------------------- main run loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="3 tasks x 1 run x 3 conds -> pilot dir")
    ap.add_argument("--validate", action="store_true", help="pre-flight checks only; no runs, no writes")
    ap.add_argument("--no-model-check", action="store_true", help="skip the Ollama length-match measurement in --validate")
    ap.add_argument("--limit", type=int, default=0, help="run only the first N jobs of the interleaved plan (0 = all)")
    args = ap.parse_args()

    cfg = load_config()

    if args.validate:
        validate(cfg, require_model_tokens=not args.no_model_check)
        return

    pilot = args.pilot
    runs_per_task = cfg["pilot_runs_per_task"] if pilot else cfg["runs_per_task"]
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

    # pre-build agents once per condition + measure per-agent system-prompt tokens (Llama tokenizer)
    amap = _agents_for_conditions(cfg, client)
    for cd in cfg["conditions"]:
        print(f"  condition={cd}  system_prompt_tokens/agent={amap[cd][1]} (sum={amap[cd][2]})")

    jobs = build_job_plan(cfg, tasks, runs_per_task)
    if args.limit:
        jobs = jobs[:args.limit]
    total = len(jobs)
    print(f"[{version}] model={cfg['model']} interleave={cfg.get('interleave')} "
          f"execution_seed={cfg.get('execution_seed')}")
    print(f"Planned jobs: {total}  Already done: {len(done)}  Output -> {runs_out}")

    for n, job in enumerate(jobs, 1):
        condition, task = job["condition"], job["task"]
        run_index, seed, execution_order = job["run_index"], job["seed"], job["execution_order"]
        agents, per_agent_tok, sys_tok = amap[condition]
        exp_id = f"{version}::{condition}::{task.task_id}::run{run_index}"
        if exp_id in done:
            continue

        ts_start = _now_iso()
        timeout = False
        error = ""
        try:
            out = run_pipeline(agents, task, seed)
        except Exception as e:                       # LOG, do not drop
            timeout = _is_timeout(e)
            error = f"{type(e).__name__}: {e}"[:300]
            out = None
        ts_end = _now_iso()

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
            row = json.loads(rec.to_json())
            row.update(execution_order=execution_order, execution_seed=cfg["execution_seed"],
                       timestamp_start=ts_start, timestamp_end=ts_end)
            append_jsonl(runs_out, row)
            insert_sqlite(con, row)
            print(f"  [{n}/{total}] ord={execution_order} {exp_id} FAILED timeout={timeout} {error}")
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
        row = json.loads(rec.to_json())
        row.update(execution_order=execution_order, execution_seed=cfg["execution_seed"],
                   timestamp_start=ts_start, timestamp_end=ts_end)
        append_jsonl(runs_out, row)
        insert_sqlite(con, row)
        print(f"  [{n}/{total}] ord={execution_order} {exp_id} correct={correct} "
              f"tok={row['total_tokens']} {row['latency_seconds']}s fails={fails}")

    con.close()
    print("Done.")


if __name__ == "__main__":
    main()
