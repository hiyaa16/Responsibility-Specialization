"""Generate docs/v2_task_selection.md: the 50-task suitability audit + per-task table.
Reads data/tasks_v2.jsonl so no metadata is invented. Run after build_tasks_v2.py."""
from __future__ import annotations
import json
import os
import hashlib
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TASKS = os.path.join(ROOT, "data", "tasks_v2.jsonl")

EVAL = {"qa": "SQuAD-normalized match + aliases", "numeric": "numeric match (tolerance)",
        "set": "normalized set-subset", "string": "normalized string match"}

AUDIT = """\
## Task-suitability audit (Part 6/9) — answered per source

For every candidate we required a *natural* Retriever -> Solver -> Verifier decomposition.
Tasks where R/S/V would be artificial were not included (e.g., no single-fact lookups, no
pure computation without distractors).

### MuSiQue (answerable dev) and HotpotQA (reconstructed) — multi-hop QA
1. **Retrieve:** the facts in the 2 gold supporting paragraphs (bridge entity + answer
   evidence) must be located among 10 (HotpotQA) or 20 (MuSiQue) paragraphs including many
   distractors.
2. **Reason/solve:** compose the facts across hops (bridge chaining; or comparison) to derive
   the answer — not recoverable from any single paragraph.
3. **Independently verifiable:** the final answer is a short span checked objectively against
   the gold answer (SQuAD normalization + MuSiQue aliases); intermediate facts are checkable
   against the gold paragraphs (used as `required_info` for F2/F5).
4. **Retrieval failure (F2):** a required gold fact never surfaces (a distractor is used, or the
   wrong paragraph is read).
5. **Solver failure (F3):** correct facts but wrong composition (wrong hop, bad comparison).
6. **Verification failure (F4):** the final stage passes a wrong answer or breaks a correct one.
7. **Exercises separation?** YES — multi-hop retrieval over distractors cleanly stresses the
   separation of "find evidence" / "compose" / "check".

> HotpotQA caveat: official distractor data was unreachable; contexts are reconstructed from a
> SHA-pinned subset (official answers preserved). See `data/sources/PROVENANCE.md`.

### Hand-authored — compositional computation with verification traps
1. **Retrieve:** select the relevant numbers/constraints from a context seeded with distractors.
2. **Reason/solve:** multi-step arithmetic/logic, each task containing a deliberate trap
   (e.g., a parallel-vs-serial time, an egg-limited scaling, an inside path area).
3. **Independently verifiable:** exact numeric / set / string answer.
4. **Retrieval failure (F2):** a distractor value is picked up (e.g., the "scheduled arrival"
   decoy) instead of the operative one.
5. **Solver failure (F3):** arithmetic/logic error or falling for the trap.
6. **Verification failure (F4):** the verifier fails to catch the trap.
7. **Exercises separation?** YES — distractors make retrieval matter, multi-step reasoning makes
   solving matter, and the traps make verification matter.
"""


def main():
    tasks = [json.loads(l) for l in open(TASKS, encoding="utf-8")]
    payload = "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks)
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    L = []
    L.append("# v2 task selection & suitability (50 harder tasks)\n")
    L.append(f"Frozen file: `data/tasks_v2.jsonl` — sha256 `{h}`  (freeze before the full run)\n")
    L.append(f"Totals: {len(tasks)} tasks | "
             f"by source {dict(Counter(t['source'] for t in tasks))} | "
             f"by difficulty {dict(Counter(t['difficulty'] for t in tasks))} | "
             f"by hops {dict(Counter(t['hops'] for t in tasks))}\n")
    L.append(AUDIT)
    L.append("\n## Per-task table\n")
    L.append("| task_id | source | source_id | difficulty | hops | answer_type | eval method | answer |")
    L.append("|---|---|---|---|---|---|---|---|")
    for t in tasks:
        ans = str(t["answer"])
        if len(ans) > 32:
            ans = ans[:29] + "..."
        sid = t["source_id"]
        if len(sid) > 26:
            sid = sid[:23] + "..."
        L.append(f"| {t['task_id']} | {t['source']} | {sid} | {t['difficulty']} | "
                 f"{t['hops']} | {t['answer_type']} | {EVAL.get(t['answer_type'],'?')} | {ans} |")

    out = os.path.join(ROOT, "docs", "v2_task_selection.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("Wrote", out, "with", len(tasks), "tasks; sha256", h)


if __name__ == "__main__":
    main()
