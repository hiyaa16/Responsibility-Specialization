"""Deterministic answer extraction and correctness checking (confound C7).
No LLM-judge: correctness is decided by exact/numeric/set matching against ground truth.
Applied identically to both conditions.
"""
from __future__ import annotations
import re
import string

_FINAL_RE = re.compile(r"FINAL_ANSWER:\s*(.+?)\s*$", re.IGNORECASE | re.DOTALL)
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_final(text: str) -> str:
    """Return the content of the LAST FINAL_ANSWER line/block, else ''."""
    matches = list(re.finditer(r"FINAL_ANSWER:\s*(.+)", text, re.IGNORECASE))
    if not matches:
        return ""
    # take everything after the last FINAL_ANSWER marker, first line only
    tail = matches[-1].group(1).strip()
    return tail.splitlines()[0].strip() if tail else ""


def _norm_str(s: str) -> str:
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def _squad_norm(s: str) -> str:
    """SQuAD/HotpotQA-style normalization (v2 QA tasks only): lowercase, drop articles,
    strip punctuation, collapse whitespace. Objective, benchmark-standard; applied identically
    to all conditions."""
    s = _norm_str(str(s))
    return " ".join(w for w in s.split() if w not in ("a", "an", "the"))


def _numeric_candidates(s: str):
    """Collect plausible numeric values from a free-text answer, unit-agnostically.
    Handles bare numbers AND durations phrased as 'H hours (and) M minutes', 'H hours',
    or 'M minutes' by offering both the hours- and minutes-interpretations, so a correct
    answer in mixed units (e.g. '5 hours and 30 minutes' == 5.5) is not scored wrong.
    Calibrated during the pilot; applied identically to both conditions."""
    s2 = s.replace(",", "")
    cands = []
    nums = re.findall(r"-?\d+(?:\.\d+)?", s2)
    if nums:                                   # last bare number = the usual conclusion
        cands.append(float(nums[-1]))
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*h(?:ours?|rs?)?\D{0,6}?(\d+(?:\.\d+)?)\s*m(?:in)",
                         s2, re.I):            # H..M combo -> both interpretations
        h, mi = float(m.group(1)), float(m.group(2))
        cands += [h + mi / 60, h * 60 + mi]
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*h(?:ours?|rs?)\b", s2, re.I):
        h = float(m.group(1)); cands += [h, h * 60]
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*m(?:in|inutes?)\b", s2, re.I):
        mi = float(m.group(1)); cands += [mi, mi / 60]
    return cands


def check_correct(candidate: str, task) -> bool:
    if not candidate:
        return False
    if task.answer_type == "numeric":
        target = float(task.answer)
        cands = _numeric_candidates(candidate)
        if not cands:
            return False
        return any(abs(c - target) <= task.tolerance for c in cands)
    if task.answer_type == "qa":
        # v2 multi-hop QA: accept the gold answer or any provided alias, under SQuAD norm.
        # Exact-normalized match OR gold-as-substring (the final answer often embeds the span).
        golds = [task.answer] + list(getattr(task, "answer_aliases", []) or [])
        cand = _squad_norm(candidate)
        if not cand:
            return False
        for g in golds:
            gn = _squad_norm(g)
            if gn and (gn == cand or gn in cand):
                return True
        return False
    if task.answer_type == "set":
        gt = {_norm_str(x) for x in task.answer}
        cand_items = {_norm_str(x) for x in re.split(r"[,;/]| and ", candidate) if x.strip()}
        return gt.issubset(cand_items)
    # string
    gt = _norm_str(str(task.answer))
    cand = _norm_str(candidate)
    return gt == cand or gt in cand


def evaluate(final_text: str, task):
    ans = extract_final(final_text)
    return ans, check_correct(ans, task)
