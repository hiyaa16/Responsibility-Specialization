"""Fixed communication protocol — SHARED by both conditions.

Strict linear: Agent1 -> Agent2 -> Agent3. Exactly 2 handoffs. The payload contract
is identical in both conditions: each agent receives the task + the previous agent's
full output text, and emits its own output text. Agent 3's output is the final answer.
"""
from __future__ import annotations
import time


def run_pipeline(agents, task, seed):
    payload = ""
    turns = []
    all_tool_calls = []
    total_in = total_out = total_llm_calls = 0
    truncated_any = False

    t0 = time.perf_counter()
    for agent in agents:
        turn, calls, truncated, llm_calls = agent.run(task, payload, seed=seed)
        turns.append(turn)
        all_tool_calls.extend(calls)
        total_in += turn.input_tokens
        total_out += turn.output_tokens
        total_llm_calls += llm_calls
        truncated_any = truncated_any or truncated
        payload = turn.output_text  # handoff
    latency = time.perf_counter() - t0

    return {
        "turns": turns,
        "tool_calls": all_tool_calls,
        "handoffs": len(agents) - 1,          # 2 for a 3-agent linear pipeline
        "input_tokens": total_in,
        "output_tokens": total_out,
        "total_tokens": total_in + total_out,
        "llm_calls": total_llm_calls,
        "latency_seconds": latency,
        "truncated": truncated_any,
        "final_text": turns[-1].output_text,  # 3rd agent's output, in BOTH conditions
    }
