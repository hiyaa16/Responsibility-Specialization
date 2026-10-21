"""THE experimental manipulation lives here and NOWHERE else.

Both conditions build 3 agents from the SAME shared preamble and the SAME three
canonical responsibility blocks (R, S, V). The only difference is ALLOCATION:

  generalist  : every agent gets [R, S, V]                (broad scope, identical agents)
  specialized : agent1=[R], agent2=[S], agent3=[V]         (bounded scope per agent)

Because the union of instruction content is byte-identical across conditions, any
measured difference is attributable to responsibility ALLOCATION, not to prompt content
or prompt-engineering quality (confound C1).
"""
from __future__ import annotations
import os
from agent import Agent
from llm import OllamaClient

_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "..", "config", "prompts")


def _read(name: str) -> str:
    with open(os.path.join(_PROMPT_DIR, name), "r", encoding="utf-8") as f:
        return f.read().strip()


def _load_blocks():
    return {
        "preamble": _read("preamble.txt"),
        "R": _read("block_retrieve.txt"),
        "S": _read("block_solve.txt"),
        "V": _read("block_verify.txt"),
    }


def _load_padding():
    """v2 only: semantically-inert, length-matching padding (see data/build_padding.py)."""
    return {
        "PAD_R": _read("pad_retrieve.txt"),
        "PAD_S": _read("pad_solve.txt"),
        "PAD_V": _read("pad_verify.txt"),
    }


def build_system_prompt(preamble: str, blocks: list[str]) -> str:
    return preamble + "\n\n" + "\n\n".join(blocks)


def build_agents(condition: str, client: OllamaClient, max_tool_iters: int) -> list[Agent]:
    b = _load_blocks()
    preamble = b["preamble"]

    if condition == "generalist":
        allocation = [
            ("generalist", [b["R"], b["S"], b["V"]]),
            ("generalist", [b["R"], b["S"], b["V"]]),
            ("generalist", [b["R"], b["S"], b["V"]]),
        ]
    elif condition == "specialized":
        allocation = [
            ("retriever", [b["R"]]),
            ("solver",    [b["S"]]),
            ("verifier",  [b["V"]]),
        ]
    elif condition == "specialized_length_matched":
        # v2 condition C: identical responsibility ALLOCATION to `specialized`, plus
        # semantically-inert padding so each agent's system-prompt token budget matches the
        # generalist's (controls the prompt-LENGTH confound C1). The padding changes NO
        # responsibility and carries no task content (see data/build_padding.py).
        p = _load_padding()
        allocation = [
            ("retriever", [b["R"], p["PAD_R"]]),
            ("solver",    [b["S"], p["PAD_S"]]),
            ("verifier",  [b["V"], p["PAD_V"]]),
        ]
    else:
        raise ValueError(f"unknown condition: {condition}")

    agents = []
    for i, (role, blocks) in enumerate(allocation):
        sp = build_system_prompt(preamble, blocks)
        agents.append(Agent(i, role, sp, client, max_tool_iters))
    return agents
