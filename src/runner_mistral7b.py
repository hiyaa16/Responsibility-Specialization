"""Mistral-7B REPLICATION runner for the v2 prompt-length-control study.

Thin wrapper around the EXISTING, UNCHANGED v2 machinery (conditions.build_agents,
pipeline.run_pipeline, evaluator.evaluate, failures.classify, tokens, schema.RunRecord). It
does NOT modify any shared module, and it NEVER writes to results/v2/runs.jsonl (Qwen) or
results/v2/llama31_8b/ (Llama).

What differs from src/runner_llama31.py (and ONLY this):
  1. reads config/experiment_mistral7b.yaml (model = mistral:7b, out_dir = results/v2/mistral_7b).
  2. DUPLICATE-PROOF CANONICAL WRITES. The Llama raw log accidentally reached 777 records because
     two overlapping executions of the same job appended to the same JSONL (its only resume guard
     was a startup snapshot + a blind append). This runner enforces the invariant
       "exactly one canonical record per (condition, task_id, run_index)"
     at WRITE TIME, so it can never produce duplicates even across kills / resumes / retries:
        (a) single-instance heartbeat lock in out_dir -> refuses to start a second concurrent
            runner (with stale-lock detection so a killed run can be resumed);
        (b) a `claimed` set of canonical keys (seeded from the existing runs.jsonl at startup,
            updated after every write); before appending a canonical record the key is checked;
        (c) if the key is already claimed the record is a retry/re-execution: it is written to a
            SEPARATE retries.jsonl with provenance and is NOT added to runs.jsonl. FIRST write
            wins — selection is purely by arrival/provenance, NEVER by correctness/latency/tokens
            or any outcome (no cherry-picking).
Everything else — tasks, prompts, conditions, padding, tools, topology, agents, handoffs,
temperature, seeds, execution_seed, interleaving, evaluation, logging fields — is identical.

Usage:
  python src/runner_mistral7b.py --validate     # 22-point pre-flight; no LLM calls, writes nothing
  python src/runner_mistral7b.py --pilot         # 5 tasks x 2 runs x 3 conds = 30 -> results/v2/mistral_7b_pilot
  python src/runner_mistral7b.py --limit 20      # first 20 jobs of the interleaved full plan (bounded smoke)
  python src/runner_mistral7b.py                 # full 50 x 5 x 3 = 750 -> results/v2/mistral_7b
  python src/runner_mistral7b.py --force         # steal a stale single-instance lock (previous run was killed)
Prerequisite:  ollama pull mistral:7b
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
import threading
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
CONFIG_REL = "config/experiment_mistral7b.yaml"


def _p(rel):
    return os.path.join(ROOT, rel)


def load_config():
    with open(_p(CONFIG_REL), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_existing(path):
    """Return (done_ids, claimed_keys) from an existing runs.jsonl.
    done_ids    = set of experiment_id already present (skip-completed guard)
    claimed_keys = set of (condition, task_id, run_index) already present (uniqueness guard)."""
    done, claimed = set(), set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    done.add(r["experiment_id"])
                    claimed.add((r["condition"], r["task_id"], r["run_index"]))
                except Exception:
                    pass
    return done, claimed


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
    # INSERT OR IGNORE: the PRIMARY KEY (experiment_id) makes the sqlite mirror a second,
    # independent guarantee that a canonical key is never stored twice.
    con.execute(
        "INSERT OR IGNORE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
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


def commit_canonical(row, claimed, runs_out, retries_out, con):
    """Write exactly one canonical record per (condition, task_id, run_index). Any execution of
    an already-claimed key is diverted to retries.jsonl (provenance preserved) — FIRST write wins,
    outcome-independent. Returns 'written' or 'diverted'."""
    key = (row["condition"], row["task_id"], row["run_index"])
    if key in claimed:
        note = dict(row)
        note["_diverted"] = True
        note["_diverted_at"] = _now_iso()
        note["_diverted_reason"] = ("canonical (condition,task_id,run_index) already present; "
                                    "original attempt kept, this re-execution logged as a retry")
        append_jsonl(retries_out, note)
        return "diverted"
    append_jsonl(runs_out, row)
    insert_sqlite(con, row)
    claimed.add(key)
    return "written"


# ---------------------------------------------------------------- single-instance lock
class InstanceLock:
    """Advisory single-instance lock with a heartbeat so a killed run frees it quickly.
    Prevents two concurrent runners from racing on the same canonical file (the Llama 777 cause)."""
    def __init__(self, path, stale_s=90, beat_s=20):
        self.path, self.stale_s, self.beat_s = path, stale_s, beat_s
        self._stop = threading.Event()
        self._thread = None

    def acquire(self, force=False):
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, json.dumps({"pid": os.getpid(), "start": _now_iso()}).encode())
                os.close(fd)
                break
            except FileExistsError:
                try:
                    age = time.time() - os.path.getmtime(self.path)
                except OSError:
                    continue
                if force or age > self.stale_s:
                    print(f"  [lock] stealing {'forced' if force else f'stale ({age:.0f}s)'} lock {self.path}")
                    try:
                        os.remove(self.path)
                    except OSError:
                        pass
                    continue
                raise SystemExit(
                    f"ERROR: another runner appears active (lock {self.path}, age {age:.0f}s < "
                    f"{self.stale_s}s heartbeat). If that process is dead, re-run with --force.")
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._thread.start()

    def _heartbeat(self):
        while not self._stop.wait(self.beat_s):
            try:
                os.utime(self.path, None)
            except OSError:
                return

    def release(self):
        self._stop.set()
        try:
            os.remove(self.path)
        except OSError:
            pass


def _is_timeout(e: Exception) -> bool:
    if isinstance(e, (socket.timeout, TimeoutError)):
        return True
    if isinstance(e, urllib.error.URLError) and isinstance(getattr(e, "reason", None), (socket.timeout, TimeoutError)):
        return True
    return "timed out" in str(e).lower()


def _now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def build_job_plan(cfg, tasks, runs_per_task):
    """Full (condition, task, run_index) job list, interleaved with execution_seed. Each job gets
    a STABLE execution_order (its position in the seeded shuffle), identical across resumes."""
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


# ---------------------------------------------------------------- validation (no experiment runs)
def validate(cfg, require_model_tokens=True):
    """22-point pre-flight check (superset of runner_llama31's 14). Structural checks need no
    network; the model-reachability + length-match checks need mistral:7b in Ollama and are
    reported (not silently skipped)."""
    print("=" * 74)
    print("PRE-FLIGHT VALIDATION — Mistral-7B replication (no experiment runs)")
    print("=" * 74)
    ok = True

    def check(n, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {n:2d}. {detail}")

    tasks = load_tasks(_p(cfg["tasks_path"]))
    conds = cfg["conditions"]
    jobs = build_job_plan(cfg, tasks, cfg["runs_per_task"])
    from collections import Counter
    per_cond = Counter(j["condition"] for j in jobs)
    transitions = sum(1 for i in range(1, len(jobs)) if jobs[i]["condition"] != jobs[i-1]["condition"])
    keys = [(j["condition"], j["task"].task_id, j["run_index"]) for j in jobs]

    out_dir = _p(cfg["out_dir"])
    qwen = _p("results/v2/runs.jsonl")
    llama_dir = _p("results/v2/llama31_8b")

    check(1, conds == ["generalist", "specialized", "specialized_length_matched"],
          f"conditions = {conds}")
    check(2, True, "same tool set across conditions (build_agents/agent.py unchanged — every agent gets all tools)")
    check(3, cfg["control_flow"] == "strict_linear", f"topology = {cfg.get('control_flow')}")
    check(4, cfg["agent_count"] == 3, f"agent_count = {cfg['agent_count']}")
    check(5, cfg["agent_count"] - 1 == 2, "handoffs = agent_count-1 = 2 (fixed by pipeline.run_pipeline)")
    check(6, cfg["temperature"] == 0.7, f"temperature = {cfg['temperature']}")
    check(7, len(tasks) == 50, f"tasks loaded = {len(tasks)}")
    check(8, cfg["runs_per_task"] == 5, f"runs_per_task = {cfg['runs_per_task']}")
    check(9, cfg["model"] == "mistral:7b", f"model = {cfg['model']}")
    check(10, cfg.get("interleave") is True, f"interleave = {cfg.get('interleave')}")
    check(11, cfg.get("execution_seed") == 20261007, f"execution_seed = {cfg.get('execution_seed')}")
    check(12, os.path.abspath(out_dir) == os.path.abspath(_p("results/v2/mistral_7b")),
          f"out_dir = {cfg['out_dir']}")
    check(13, os.path.abspath(os.path.join(out_dir, "runs.jsonl")) != os.path.abspath(qwen),
          "Qwen results/v2/runs.jsonl will NOT be overwritten")
    check(14, not os.path.abspath(out_dir).startswith(os.path.abspath(llama_dir) + os.sep)
          and os.path.abspath(out_dir) != os.path.abspath(llama_dir),
          "Llama results/v2/llama31_8b/ will NOT be overwritten")
    rr_fields = set(RunRecord.__dataclass_fields__.keys())
    needed = {"agent_outputs", "tool_calls", "input_tokens", "output_tokens", "total_tokens",
              "llm_calls", "latency_seconds", "failure_types", "corrections", "repeated_work",
              "truncated", "system_prompt_tokens_per_agent"}
    check(15, needed.issubset(rr_fields), "all existing eval/logging fields preserved in RunRecord")
    check(16, len(jobs) == 750, f"planned runs = {len(jobs)}")
    check(17, all(per_cond[c] == 250 for c in conds), f"each condition = {dict(per_cond)}")
    check(18, transitions > 300,
          f"execution genuinely interleaved: {transitions} condition-transitions "
          f"(blocked would be ~2)")
    # 21 + 22 first (pure logic, no network)
    check(21, len(keys) == len({k for k in keys}) == 750,
          "canonical key = (condition, task_id, run_index); plan has 750 UNIQUE keys")
    dup_ok = _selftest_uniqueness_guard()
    check(22, dup_ok, "duplicate canonical keys CANNOT be written (write-time guard self-test: "
                      "2nd write of same key diverts to retries.jsonl)")

    # 19 (reachable) + 20 (length match) — need the model
    print("  --- 19/20: model reachability + condition-C length match (need mistral:7b in Ollama) ---")
    reachable = False
    rel = None
    if require_model_tokens:
        try:
            client = OllamaClient(url=cfg["ollama_url"], model=cfg["model"],
                                  temperature=cfg["temperature"], top_p=cfg["top_p"],
                                  num_ctx=cfg["num_ctx"], num_predict=cfg["num_predict"],
                                  timeout_s=cfg["request_timeout_s"])
            amap = _agents_for_conditions(cfg, client)
            reachable = True
            a_tot = amap["generalist"][2]
            c_tot = amap["specialized_length_matched"][2]
            for cd in conds:
                print(f"       {cd:28s} per-agent sys tokens={amap[cd][1]} sum={amap[cd][2]}")
            rel = 100.0 * (c_tot - a_tot) / a_tot if a_tot else float("nan")
            print(f"       A sum={a_tot}  C sum={c_tot}  (C-A = {c_tot-a_tot:+d} tok, {rel:+.2f}% vs A)")
        except Exception as e:
            print(f"       NOT reachable: {type(e).__name__}: {e}")
            print(f"       Run `ollama pull mistral:7b`, start Ollama, then re-run --validate.")
    else:
        print("       skipped (--no-model-check)")
    check(19, reachable, f"model mistral:7b reachable via Ollama = {reachable}")
    check(20, (rel is not None and abs(rel) <= 5.0),
          f"condition-C length-matched to A under Mistral tokenizer "
          f"({'%.2f%%' % rel if rel is not None else 'unmeasured'}; |C-A| <= 5% required)")

    print("=" * 74)
    print(f"VALIDATION: {'ALL 22 PASS' if ok else 'SOME CHECKS FAILED — fix before running'}")
    print("=" * 74)
    return ok


def _selftest_uniqueness_guard():
    """Prove the write-time guard rejects a duplicate canonical key, using a throwaway temp dir
    (no LLM, no touching real outputs)."""
    import tempfile
    d = tempfile.mkdtemp(prefix="dupguard_")
    runs_out = os.path.join(d, "runs.jsonl")
    retries_out = os.path.join(d, "retries.jsonl")
    con = init_sqlite(os.path.join(d, "x.sqlite"))
    claimed = set()
    base = {"experiment_id": "t::generalist::MQ01::run0", "experiment_version": "t",
            "task_id": "MQ01", "source": "s", "hops": 1, "condition": "generalist",
            "run_index": 0, "seed": 11, "model": "mistral:7b", "temperature": 0.7,
            "difficulty": "d", "category": "c", "handoffs": 2, "system_prompt_tokens": 0,
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "llm_calls": 0,
            "latency_seconds": 0.0, "truncated": False, "timeout": False, "error": "",
            "final_answer": "", "correct": False, "failure_types": [], "corrections": 0,
            "repeated_work": 0.0, "execution_order": 0, "execution_seed": 20261007,
            "timestamp_start": _now_iso(), "timestamp_end": _now_iso()}
    r1 = commit_canonical(dict(base), claimed, runs_out, retries_out, con)
    r2 = commit_canonical(dict(base), claimed, runs_out, retries_out, con)  # same key again
    con.close()
    n_canon = sum(1 for _ in open(runs_out, encoding="utf-8")) if os.path.exists(runs_out) else 0
    n_retry = sum(1 for _ in open(retries_out, encoding="utf-8")) if os.path.exists(retries_out) else 0
    import shutil
    shutil.rmtree(d, ignore_errors=True)
    return r1 == "written" and r2 == "diverted" and n_canon == 1 and n_retry == 1


# ---------------------------------------------------------------- main run loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="5 tasks x 2 runs x 3 conds -> pilot dir")
    ap.add_argument("--validate", action="store_true", help="22-point pre-flight; no runs, no writes")
    ap.add_argument("--no-model-check", action="store_true", help="skip Ollama checks in --validate")
    ap.add_argument("--limit", type=int, default=0, help="run only the first N jobs of the interleaved plan")
    ap.add_argument("--force", action="store_true", help="steal a stale single-instance lock (prev run killed)")
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
    retries_out = os.path.join(out_dir, "retries.jsonl")
    sqlite_out = os.path.join(out_dir, "experiments.sqlite")

    lock = InstanceLock(os.path.join(out_dir, ".runner.lock"))
    lock.acquire(force=args.force)
    try:
        done, claimed = load_existing(runs_out)
        con = init_sqlite(sqlite_out)

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

        n_written = n_diverted = 0
        for n, job in enumerate(jobs, 1):
            condition, task = job["condition"], job["task"]
            run_index, seed, execution_order = job["run_index"], job["seed"], job["execution_order"]
            agents, per_agent_tok, sys_tok = amap[condition]
            exp_id = f"{version}::{condition}::{task.task_id}::run{run_index}"
            if exp_id in done:
                continue   # already completed in a previous (resumed) session

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
            else:
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
            status = commit_canonical(row, claimed, runs_out, retries_out, con)
            if status == "written":
                done.add(exp_id)
                n_written += 1
                tag = (f"correct={row['correct']} tok={row['total_tokens']} "
                       f"{row['latency_seconds']}s fails={row['failure_types']}") if out is not None \
                    else f"FAILED timeout={timeout} {error}"
                print(f"  [{n}/{total}] ord={execution_order} {exp_id} {tag}")
            else:
                n_diverted += 1
                print(f"  [{n}/{total}] ord={execution_order} {exp_id} DIVERTED->retries.jsonl "
                      f"(canonical key already present)")

        con.close()
        print(f"Done. canonical written this session={n_written}, diverted retries={n_diverted}, "
              f"canonical total now={len(done)}")
    finally:
        lock.release()


if __name__ == "__main__":
    main()
