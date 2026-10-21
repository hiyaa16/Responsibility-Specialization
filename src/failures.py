"""Extensible failure taxonomy + automated classifiers (confound C8).

Automated heuristics are applied identically to both conditions and are BLIND to
condition in the sense that they only read the run trace + task, never the label.
Subjective categories (F3/F4 nuance) carry a heuristic here but are flagged for
optional blind manual review. Add a new category by appending to TAXONOMY and
registering a classifier in CLASSIFIERS — nothing else changes.

NOTE on interpretation: F1 (wrong-responsibility execution) and F7 (out-of-role tool
use) are STRUCTURALLY only possible in the specialized condition, because generalist
agents own all responsibilities and therefore cannot act "out of role". This asymmetry
is a finding to report as a NEW failure mode introduced by specialization, not as
evidence that specialization is simply worse.
"""
from __future__ import annotations
import re
from evaluator import extract_final, check_correct


def _norm(s: str) -> str:
    return re.sub(r"[$,]", "", s.lower())


def _present(fact: str, text: str) -> bool:
    """Robust presence check for a required-info token (calibrated during pilot).
    Numbers match on digit boundaries (so '4' does not match '14'/'40'); '$'/','
    are ignored (so '$1,200' matches '1200' and '4' matches '4 dollars')."""
    f, t = _norm(fact), _norm(text)
    if not f:
        return True
    if re.fullmatch(r"-?\d+(\.\d+)?", f):
        return re.search(r"(?<!\d)" + re.escape(f) + r"(?!\d)", t) is not None
    return f in t

TAXONOMY = {
    "F1": "Wrong responsibility execution (agent acts outside its assigned scope)",
    "F2": "Retrieval failure (a required fact never surfaced by the first agent)",
    "F3": "Reasoning/solving failure (solve-stage proposed answer is wrong)",
    "F4": "Verification failure (final/verify stage let an error through or broke a correct answer)",
    "F5": "Handoff/information-loss failure (required fact present early, lost by final)",
    "F6": "Redundant/repeated work (high output overlap between agents)",
    "F7": "Unnecessary/out-of-scope tool use",
    "F8": "Inter-agent correction (final answer changed vs. the solve stage)",
    "F9": "Final answer failure (system output incorrect)",
}

# responsibility scope by role -> which tools are 'in scope'
_ROLE_TOOL_SCOPE = {
    "generalist": {"lookup", "calculator"},
    "retriever": {"lookup"},
    "solver": {"calculator"},
    "verifier": {"calculator", "lookup"},
}

_OVERLAP_THRESHOLD = 0.6


def _word_set(text: str):
    return set(w for w in "".join(c.lower() if c.isalnum() else " " for c in text).split())


def _jaccard(a: str, b: str) -> float:
    wa, wb = _word_set(a), _word_set(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def classify(task, turns, final_text, correct):
    """Return (failure_types, corrections, repeated_work)."""
    fails = set()
    outputs = [t.output_text for t in turns]
    finals = [extract_final(o) for o in outputs]

    # ---- F9: system incorrect ----
    if not correct:
        fails.add("F9")

    # ---- F2: retrieval failure (required fact missing from first agent output) ----
    a1 = outputs[0]
    missing_early = [f for f in task.required_info if not _present(f, a1)]
    if missing_early:
        fails.add("F2")

    # ---- F5: info loss (present in agent1, absent in final agent output) ----
    a_last = outputs[-1]
    for f in task.required_info:
        if _present(f, a1) and not _present(f, a_last):
            fails.add("F5")
            break

    # ---- F3: solve-stage proposed answer wrong (agent2 is the solver stage) ----
    if len(turns) >= 2:
        solve_ans = finals[1] or finals[0]
        if solve_ans and not check_correct(solve_ans, task):
            fails.add("F3")

    # ---- F4: verification stage failed (final wrong) OR broke a correct solve answer ----
    if len(turns) >= 3:
        solve_ans = finals[1]
        solve_ok = bool(solve_ans) and check_correct(solve_ans, task)
        if not correct:
            fails.add("F4")           # verifier failed to fix / produce a correct answer
        elif solve_ans and not solve_ok and correct:
            pass                       # verifier correctly fixed a wrong solve answer (good)

    # ---- F6: redundant work (high overlap between agent outputs) ----
    repeated = 0.0
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            repeated = max(repeated, _jaccard(outputs[i], outputs[j]))
    if repeated >= _OVERLAP_THRESHOLD:
        fails.add("F6")

    # ---- F7: out-of-scope tool use ----
    for t in turns:
        scope = _ROLE_TOOL_SCOPE.get(t.role, {"lookup", "calculator"})
        for c in t.tool_calls:
            if c["tool"].lower() not in scope:
                fails.add("F7")

    # ---- F1: wrong responsibility execution (retriever fully solved the task) ----
    for t in turns:
        if t.role == "retriever":
            r_ans = extract_final(t.output_text)
            if r_ans and check_correct(r_ans, task):
                fails.add("F1")

    # ---- F8: inter-agent correction (final differs from solve-stage answer) ----
    corrections = 0
    for i in range(1, len(finals)):
        if finals[i] and finals[i - 1] and finals[i] != finals[i - 1]:
            corrections += 1
    if len(finals) >= 3 and finals[-1] and finals[1] and finals[-1] != finals[1]:
        fails.add("F8")

    return sorted(fails), corrections, round(repeated, 3)
