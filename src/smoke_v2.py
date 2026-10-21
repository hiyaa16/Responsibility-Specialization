"""Fast harness validation for v2 (NOT the confirmatory pilot). Exercises all 3 conditions on a
few small-context tasks so correctness can be checked in minutes rather than the ~10h a full
48-run CPU pilot needs. Writes to results/v2_pilot_smoke/. Validates: 3-condition construction,
per-agent token match, QA/numeric/set eval, F1-F9 population, and timeout/error-row logging."""
from __future__ import annotations
import os
import sys
import yaml
sys.path.insert(0, os.path.dirname(__file__))
from schema import load_tasks, RunRecord
from llm import OllamaClient
from conditions import build_agents
from pipeline import run_pipeline
from evaluator import evaluate
from failures import classify
import tokens as tokmod
from runner_v2 import append_jsonl, init_sqlite, insert_sqlite, _is_timeout

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SMOKE_TASKS = ["HA01", "HA05", "HP02"]  # numeric, set, QA (HP02 larger context)


def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config/experiment_v2.yaml"), encoding="utf-8"))
    out_dir = os.path.join(ROOT, "results", "v2_pilot_smoke")
    os.makedirs(out_dir, exist_ok=True)
    runs_out = os.path.join(out_dir, "runs.jsonl")
    con = init_sqlite(os.path.join(out_dir, "experiments.sqlite"))
    client = OllamaClient(url=cfg["ollama_url"], model=cfg["model"], temperature=cfg["temperature"],
                          top_p=cfg["top_p"], num_ctx=cfg["num_ctx"], num_predict=cfg["num_predict"],
                          timeout_s=cfg["request_timeout_s"])
    tasks = {t.task_id: t for t in load_tasks(os.path.join(ROOT, "data/tasks_v2.jsonl"))}

    print("=== per-agent system-prompt token check (A~C per agent; B lower) ===")
    for cond in cfg["conditions"]:
        agents = build_agents(cond, client, cfg["max_tool_iters"])
        toks = [tokmod.system_prompt_tokens(a.system_prompt) for a in agents]
        print(f"  {cond:28s} per-agent={toks} sum={sum(toks)} roles={[a.role for a in agents]}")

    print("\n=== smoke runs (seed 11) ===")
    for cond in cfg["conditions"]:
        agents = build_agents(cond, client, cfg["max_tool_iters"])
        per_agent_tok = [tokmod.system_prompt_tokens(a.system_prompt) for a in agents]
        for tid in SMOKE_TASKS:
            task = tasks[tid]
            try:
                out = run_pipeline(agents, task, seed=11)
                final, correct = evaluate(out["final_text"], task)
                fails, corr, rep = classify(task, out["turns"], out["final_text"], correct)
                rec = RunRecord(experiment_id=f"smoke::{cond}::{tid}::run0", task_id=tid, condition=cond,
                    run_index=0, seed=11, model=cfg["model"], temperature=cfg["temperature"],
                    agent_count=3, difficulty=task.difficulty, category=task.category,
                    agent_outputs=[a.__dict__ for a in out["turns"]], tool_calls=out["tool_calls"],
                    handoffs=out["handoffs"], system_prompt_tokens=sum(per_agent_tok),
                    input_tokens=out["input_tokens"], output_tokens=out["output_tokens"],
                    total_tokens=out["total_tokens"], llm_calls=out["llm_calls"],
                    latency_seconds=round(out["latency_seconds"], 1), truncated=out["truncated"],
                    final_answer=final, correct=correct, failure_types=fails, corrections=corr,
                    repeated_work=rep, experiment_version="v2_smoke", source=task.source,
                    source_id=task.source_id, hops=task.hops, timeout=False, error="",
                    system_prompt_tokens_per_agent=per_agent_tok)
                append_jsonl(runs_out, rec); insert_sqlite(con, rec)
                print(f"  {cond[:12]:12s} {tid}: {out['latency_seconds']:.0f}s correct={correct} "
                      f"ans={final[:30]!r} tok={out['total_tokens']} fails={fails}")
            except Exception as e:
                print(f"  {cond[:12]:12s} {tid}: EXCEPTION {e}")

    print("\n=== timeout/error-row logging check (bad URL -> must LOG, not crash) ===")
    bad = OllamaClient(url="http://localhost:1/api/chat", model=cfg["model"], temperature=0.7,
                       top_p=0.9, num_ctx=1024, num_predict=16, timeout_s=3)
    agents = build_agents("specialized", bad, 1)
    try:
        run_pipeline(agents, tasks["HA01"], seed=11)
        print("  ERROR: expected an exception")
    except Exception as e:
        rec = RunRecord(experiment_id="smoke::err::HA01::run0", task_id="HA01", condition="specialized",
            run_index=0, seed=11, model=cfg["model"], temperature=0.7, agent_count=3,
            difficulty="hard", category="finance", agent_outputs=[], tool_calls=[], handoffs=0,
            system_prompt_tokens=0, input_tokens=0, output_tokens=0, total_tokens=0, llm_calls=0,
            latency_seconds=0.0, truncated=False, final_answer="", correct=False,
            failure_types=["F9"], corrections=0, repeated_work=0.0, experiment_version="v2_smoke",
            source="handauthored", source_id="HA01", hops=0, timeout=_is_timeout(e),
            error=f"{type(e).__name__}: {e}"[:120], system_prompt_tokens_per_agent=[])
        append_jsonl(runs_out, rec); insert_sqlite(con, rec)
        print(f"  logged error row: is_timeout={_is_timeout(e)} error={type(e).__name__} (no crash) OK")
    con.close()
    print(f"\nSmoke outputs -> {runs_out}")


if __name__ == "__main__":
    main()
