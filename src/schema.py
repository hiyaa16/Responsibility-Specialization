"""Data schemas for tasks and run records. Kept dependency-free (stdlib only)."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
import json


@dataclass
class Task:
    task_id: str
    question: str
    context: str                 # reference passage the retriever must extract from
    required_info: list[str]     # key facts/tokens that MUST reach the solver (for F2/F5)
    answer: Any                  # ground-truth answer
    answer_type: str             # "numeric" | "string" | "set"
    tolerance: float = 0.0       # numeric tolerance (absolute)
    expected_reasoning: str = "" # short note on the intended solution path
    difficulty: str = "medium"   # "easy" | "medium" | "hard" | "pilot"
    category: str = "general"
    # ---- v2 additions (optional; v1 tasks omit them and fall back to defaults) ----
    answer_aliases: list = field(default_factory=list)  # acceptable alternative gold answers (MuSiQue)
    source: str = "handauthored"  # "musique" | "hotpotqa" | "handauthored"
    source_id: str = ""           # id in the originating benchmark (traceability)
    hops: int = 0                 # multi-hop count where defined (0 = n/a)

    @staticmethod
    def from_dict(d: dict) -> "Task":
        # tolerate unknown keys so vendored metadata never breaks loading
        import dataclasses as _dc
        known = {f.name for f in _dc.fields(Task)}
        return Task(**{k: v for k, v in d.items() if k in known})


@dataclass
class ToolCall:
    agent_index: int
    role: str
    tool: str
    argument: str
    result_preview: str


@dataclass
class AgentTurn:
    agent_index: int
    role: str                    # "generalist" or "retriever"/"solver"/"verifier"
    system_prompt_tokens: int
    input_tokens: int
    output_tokens: int
    tool_iters: int
    output_text: str
    tool_calls: list[dict] = field(default_factory=list)


@dataclass
class RunRecord:
    experiment_id: str
    task_id: str
    condition: str               # "generalist" | "specialized"
    run_index: int
    seed: int
    model: str
    temperature: float
    agent_count: int
    difficulty: str
    category: str

    agent_outputs: list[dict]    # serialized AgentTurns
    tool_calls: list[dict]
    handoffs: int

    # cost, decomposed (see confound C1): system-prompt overhead vs task/generated tokens
    system_prompt_tokens: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    llm_calls: int

    latency_seconds: float
    truncated: bool              # True if any turn hit context/output limits (C5)

    final_answer: str
    correct: bool
    failure_types: list[str] = field(default_factory=list)
    corrections: int = 0
    repeated_work: float = 0.0   # max inter-agent textual overlap [0,1]

    # ---- v2 additions (optional; default to v1-compatible values) ----
    experiment_version: str = "v1"
    source: str = "handauthored"
    source_id: str = ""
    hops: int = 0
    timeout: bool = False          # True if the run hit the request timeout (logged, NOT dropped)
    error: str = ""                # non-empty if the run failed; the run is still recorded
    system_prompt_tokens_per_agent: list = field(default_factory=list)  # real-tokenizer, per agent

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def load_tasks(path: str) -> list[Task]:
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(Task.from_dict(json.loads(line)))
    return tasks
