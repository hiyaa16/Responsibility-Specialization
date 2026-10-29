"""Track B runner — depth x condition x task x rep, INTERLEAVED, with execution_order logged.

Separate output namespace (results/trackB/); never touches v1/v2 data. Crash-safe + resumable
(append+fsync jsonl, skip already-present experiment_ids). Reuses the shared pipeline/evaluator/
failures; the ONLY new knobs are `depth` (pipeline length) and interleaved scheduling + execution_order
logging (so latency is de-confoundable — the exact gap that broke v2 Qwen latency).

Run (Ollama + qwen2.5:7b up; padding frozen via data/build_padding_trackB.py):
    python src/runner_trackB.py --pilot     # smoke test -> results/trackB_pilot
    python src/runner_trackB.py             # confirmatory -> results/trackB  (resumable)
"""
from __future__ import annotations
import os, sys, json, random, socket, argparse, urllib.error
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from schema import load_tasks                     # noqa: E402
from llm import OllamaClient                      # noqa: E402
from conditions_trackB import build_agents_trackB # noqa: E402
from pipeline import run_pipeline                 # noqa: E402
from evaluator import evaluate                    # noqa: E402
from failures import classify                     # noqa: E402
import tokens as tokmod                           # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
def _p(rel): return os.path.join(ROOT, rel)


def load_config(path="config/experiment_trackB.yaml"):
    with open(_p(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def existing_ids(path):
    done = set()
    if os.path.exists(path):
        for line in open(path, "r", encoding="utf-8"):
            line = line.strip()
            if line:
                try: done.add(json.loads(line)["experiment_id"])
                except Exception: pass
    return done


def append_jsonl(path, record: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush(); os.fsync(f.fileno())


def _is_timeout(e):
    if isinstance(e, (socket.timeout, TimeoutError)): return True
    if isinstance(e, urllib.error.URLError) and isinstance(getattr(e, "reason", None), (socket.timeout, TimeoutError)):
        return True
    return "timed out" in str(e).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--config", default="config/experiment_trackB.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    model_tag = cfg.get("model_tag")       # None for Qwen default (uses existing un-tagged padding)
    pilot = args.pilot
    version = cfg["experiment_version"] + ("_pilot" if pilot else "")
    runs_per_task = cfg["pilot_runs_per_task"] if pilot else cfg["runs_per_task"]
    seeds = cfg["seeds"][:runs_per_task]
    depths = cfg["depths"]

    client = OllamaClient(url=cfg["ollama_url"], model=cfg["model"], temperature=cfg["temperature"],
                          top_p=cfg["top_p"], num_ctx=cfg["num_ctx"], num_predict=cfg["num_predict"],
                          timeout_s=cfg["request_timeout_s"])

    tasks = load_tasks(_p(cfg["tasks_path"]))
    if pilot:
        ids = set(cfg["pilot_task_ids"]); tasks = [t for t in tasks if t.task_id in ids]

    out_dir = _p(cfg["pilot_dir"] if pilot else cfg["out_dir"]); os.makedirs(out_dir, exist_ok=True)
    runs_out = os.path.join(out_dir, "runs.jsonl")
    done = existing_ids(runs_out)

    # cache agents per (condition, depth) and their per-agent system-prompt tokens (real tokenizer)
    agent_cache, systok_cache = {}, {}
    for d in depths:
        for cond in cfg["conditions"]:
            ag = build_agents_trackB(cond, d, client, cfg["max_tool_iters"], model_tag=model_tag)
            agent_cache[(cond, d)] = ag
            per = [tokmod.system_prompt_tokens(a.system_prompt) for a in ag]
            systok_cache[(cond, d)] = per
            print(f"  built {cond} D={d}: per-agent sys tokens={per} sum={sum(per)}")

    # build the full job list, then INTERLEAVE with a fixed seed
    jobs = []
    for d in depths:
        for cond in cfg["conditions"]:
            for t in tasks:
                for ri in range(runs_per_task):
                    jobs.append({"depth": d, "condition": cond, "task": t, "run_index": ri, "seed": seeds[ri]})
    random.Random(cfg["schedule_seed"]).shuffle(jobs)

    total = len(jobs)
    print(f"[{version}] interleaved jobs: {total}  depths={depths} conds={cfg['conditions']} "
          f"tasks={len(tasks)} reps={runs_per_task}.  done={len(done)}  -> {runs_out}")

    for order, job in enumerate(jobs):
        d, cond, task, ri = job["depth"], job["condition"], job["task"], job["run_index"]
        exp_id = f"{version}::D{d}::{cond}::{task.task_id}::run{ri}"
        if exp_id in done:
            continue
        agents = agent_cache[(cond, d)]
        per_tok = systok_cache[(cond, d)]
        timeout, error, out = False, "", None
        try:
            out = run_pipeline(agents, task, job["seed"])
        except Exception as e:
            timeout = _is_timeout(e); error = f"{type(e).__name__}: {e}"[:300]
        base = {"experiment_id": exp_id, "experiment_version": version, "task_id": task.task_id,
                "condition": cond, "depth": d, "run_index": ri, "seed": job["seed"],
                "model": cfg["model"], "temperature": cfg["temperature"], "agent_count": d,
                "difficulty": task.difficulty, "category": task.category, "source": task.source,
                "source_id": task.source_id, "hops": task.hops, "complexity_K": task.hops,
                "execution_order": order, "system_prompt_tokens": sum(per_tok),
                "system_prompt_tokens_per_agent": per_tok}
        if out is None:
            base.update({"agent_outputs": [], "tool_calls": [], "handoffs": 0, "input_tokens": 0,
                         "output_tokens": 0, "total_tokens": 0, "llm_calls": 0, "latency_seconds": 0.0,
                         "truncated": False, "timeout": timeout, "error": error, "final_answer": "",
                         "correct": False, "failure_types": ["F9"], "corrections": 0, "repeated_work": 0.0})
            append_jsonl(runs_out, base)
            print(f"  [{order+1}/{total}] {exp_id} FAILED timeout={timeout} {error}")
            continue
        final_answer, correct = evaluate(out["final_text"], task)
        fails, corrections, repeated = classify(task, out["turns"], out["final_text"], correct)
        base.update({"agent_outputs": [a.__dict__ for a in out["turns"]], "tool_calls": out["tool_calls"],
                     "handoffs": out["handoffs"], "input_tokens": out["input_tokens"],
                     "output_tokens": out["output_tokens"], "total_tokens": out["total_tokens"],
                     "llm_calls": out["llm_calls"], "latency_seconds": round(out["latency_seconds"], 3),
                     "truncated": out["truncated"], "timeout": False, "error": "",
                     "final_answer": final_answer, "correct": correct, "failure_types": fails,
                     "corrections": corrections, "repeated_work": repeated})
        append_jsonl(runs_out, base)
        print(f"  [{order+1}/{total}] {exp_id} correct={correct} tok={base['total_tokens']} "
              f"{base['latency_seconds']}s")
    print("Done.")


if __name__ == "__main__":
    main()
