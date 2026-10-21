"""Single agent abstraction — SHARED by both conditions.

An agent = shared preamble + a set of responsibility blocks + the uniform tool loop.
The ONLY thing that differs between conditions is which responsibility blocks it holds
(see conditions.py). This file has no knowledge of "generalist" vs "specialized".
"""
from __future__ import annotations
import re
from schema import AgentTurn
from llm import OllamaClient
import tools

_TOOL_RE = re.compile(r"^\s*TOOL_CALL:\s*([^|]+)\|(.*)$", re.MULTILINE)


def parse_tool_call(text: str):
    """Return (tool, argument) for the FIRST tool call in text, else None."""
    m = _TOOL_RE.search(text)
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip()


class Agent:
    def __init__(self, agent_index: int, role: str, system_prompt: str,
                 client: OllamaClient, max_tool_iters: int):
        self.agent_index = agent_index
        self.role = role
        self.system_prompt = system_prompt
        self.client = client
        self.max_tool_iters = max_tool_iters

    def run(self, task, incoming_payload: str, seed: int):
        """Execute one agent turn (with bounded tool loop). Returns (AgentTurn, calls, truncated, llm_calls)."""
        if incoming_payload:
            user = (f"TASK:\n{task.question}\n\n"
                    f"PREVIOUS AGENT OUTPUT:\n{incoming_payload}\n")
        else:
            user = f"TASK:\n{task.question}\n"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user},
        ]

        tool_calls: list[dict] = []
        in_tokens = out_tokens = llm_calls = 0
        system_prompt_tokens = 0
        truncated = False
        content = ""

        for it in range(self.max_tool_iters + 1):
            resp = self.client.chat(messages, seed=seed)
            llm_calls += 1
            in_tokens += resp.prompt_tokens
            out_tokens += resp.output_tokens
            if resp.done_reason == "length":
                truncated = True
            content = resp.content
            messages.append({"role": "assistant", "content": content})

            call = parse_tool_call(content)
            # Execute any requested tool BEFORE accepting a final answer. Models frequently emit
            # a TOOL_CALL and a premature FINAL_ANSWER in the SAME message; if we stopped on the
            # FINAL_ANSWER the tool (e.g. lookup) would never run and the agent would hallucinate
            # ungrounded. So we stop only when no tool is requested, or the budget is spent.
            if call is None or it == self.max_tool_iters:
                break

            tool, arg = call
            result = tools.dispatch(tool, arg, task)
            tool_calls.append({
                "agent_index": self.agent_index, "role": self.role,
                "tool": tool, "argument": arg, "result_preview": result[:200],
            })
            messages.append({"role": "user", "content": f"TOOL_RESULT: {result}"})

        # First chat's prompt tokens ~= system prompt + user (approx system overhead).
        # We record the system prompt token share by re-measuring is not needed; we log
        # total input tokens and estimate system overhead separately in the runner.
        turn = AgentTurn(
            agent_index=self.agent_index, role=self.role,
            system_prompt_tokens=system_prompt_tokens,
            input_tokens=in_tokens, output_tokens=out_tokens,
            tool_iters=llm_calls, output_text=content, tool_calls=tool_calls,
        )
        return turn, tool_calls, truncated, llm_calls
