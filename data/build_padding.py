"""Generate FROZEN, semantically-inert padding for condition C (specialized_length_matched).

Why this design (defensible, no obvious distribution confound):
  The generalist's *extra* system-prompt tokens are ordinary English instructional prose
  (the Solve and Verify blocks the retriever/solver/verifier do not hold). To control the
  prompt-LENGTH confound (C1) without introducing a *distribution* confound, the padding is
  ordinary, grammatical English of a similar neutral register -- NOT lorem-ipsum / repeated
  tokens (which are obviously out-of-distribution and the thing we must avoid) and NOT the
  real R/S/V instructions (which would re-introduce role/task content, defeating the point).

  The padding is therefore built from a fixed pool of DECLARATIVE, non-operative sentences
  that contain: no task information, no reasoning instructions, no role information, no
  examples, no hidden constraints, no alternative solutions, no evaluation hints, no domain
  facts. Each sentence's only relevant property is its length. The pool is frozen here and
  the generated files are hashed; padding is NEVER regenerated or tuned after seeing results.

Matching rule (fixed BEFORE the run): size each specialized agent's padded system prompt to
the generalist per-agent system-prompt token budget (preamble+R+S+V), measured with the REAL
qwen2.5:7b tokenizer (src/tokens.py), to within +/-2%. Achieved differences are reported.

Run once (Ollama with qwen2.5:7b must be up):  python data/build_padding.py
"""
from __future__ import annotations
import os
import sys
import hashlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tokens import full_prompt_tokens, system_prompt_tokens  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PDIR = os.path.join(ROOT, "config", "prompts")
TOLERANCE = 0.02  # +/-2%

# Non-operative declarative sentences. Deliberately contentless; see module docstring.
_HEADER = "NON-OPERATIVE NOTE (length-standardization padding; carries no task content):"
_POOL = [
    "The following text is included only to standardize the overall length of this message across experimental configurations.",
    "It is non-operative and does not carry information related to any particular task.",
    "The wording here is intentionally plain and was selected without regard to the question at hand.",
    "This passage introduces no facts, no constraints, and no examples of any kind.",
    "It does not describe responsibilities and does not alter the responsibilities stated elsewhere.",
    "The sentences in this section are generic and interchangeable with one another.",
    "They are present so that message lengths remain comparable between the conditions.",
    "No part of this section references a question, a context, or an expected answer.",
    "The paragraph conveys no domain knowledge and offers no guidance of any sort.",
    "Its only property of interest to the experiment is its length.",
    "The content was produced from a fixed, neutral word list prepared well in advance.",
    "It contains nothing that would make a task easier or harder to complete.",
    "These lines are filler whose meaning is deliberately empty.",
    "The text is uniform in tone and unremarkable in substance.",
    "It was frozen before any experimental results were examined.",
    "Nothing stated here constitutes a hint, a clue, or a suggestion toward an outcome.",
    "This section is an artifact of the measurement procedure and nothing more.",
    "Its phrasing carries no preference toward any answer or method.",
    "The material is provided purely to equalize token counts between prompts.",
    "It neither assists nor obstructs the instructions that surround it.",
]


def _read(name):
    with open(os.path.join(PDIR, name), "r", encoding="utf-8") as f:
        return f.read().strip()


def _assemble(n_words: int) -> str:
    """Deterministic neutral padding of ~n_words words: header + cycled pool, word-truncated."""
    words = _HEADER.split()
    i = 0
    while len(words) < n_words:
        words += _POOL[i % len(_POOL)].split()
        i += 1
    return " ".join(words[:n_words])


def _fit(prefix_system: str, target_tokens: int):
    """Binary-search a word count so full_prompt_tokens(prefix + '\\n\\n' + padding) is within
    tolerance of target_tokens; return (padding_text, achieved_full_tokens)."""
    lo, hi = 1, 600
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        pad = _assemble(mid)
        tot = full_prompt_tokens(prefix_system + "\n\n" + pad)
        if best is None or abs(tot - target_tokens) < abs(best[2] - target_tokens):
            best = (mid, pad, tot)
        if tot < target_tokens:
            lo = mid + 1
        elif tot > target_tokens:
            hi = mid - 1
        else:
            return pad, tot
    return best[1], best[2]


def main():
    pre = _read("preamble.txt")
    R, S, V = _read("block_retrieve.txt"), _read("block_solve.txt"), _read("block_verify.txt")

    def sp(blocks):
        return pre + "\n\n" + "\n\n".join(blocks)

    target = full_prompt_tokens(sp([R, S, V]))  # generalist per-agent budget (full, with wrapper)
    iso_target = system_prompt_tokens(sp([R, S, V]))

    plan = [("retrieve", sp([R]), "R"), ("solve", sp([S]), "S"), ("verify", sp([V]), "V")]
    report = []
    report.append("PROMPT-LENGTH MATCHING REPORT (condition C = specialized_length_matched)")
    report.append("Tokenizer: qwen2.5:7b via Ollama prompt_eval_count (src/tokens.py)")
    report.append(f"Tolerance: +/-{TOLERANCE*100:.0f}%")
    report.append(f"Generalist per-agent system prompt (preamble+R+S+V): "
                  f"full={target} tok, isolated={iso_target} tok  <-- TARGET")
    report.append("")
    lo_ok, hi_ok = target * (1 - TOLERANCE), target * (1 + TOLERANCE)
    all_ok = True
    for name, prefix, letter in plan:
        pad, achieved = _fit(prefix, target)
        out = os.path.join(PDIR, f"pad_{name}.txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write(pad + "\n")
        h = hashlib.sha256((pad + "\n").encode("utf-8")).hexdigest()[:16]
        padded_full = full_prompt_tokens(prefix + "\n\n" + pad)
        padded_iso = system_prompt_tokens(prefix + "\n\n" + pad)
        base_iso = system_prompt_tokens(prefix)
        diff_pct = 100.0 * (padded_full - target) / target
        ok = lo_ok <= padded_full <= hi_ok
        all_ok = all_ok and ok
        report.append(f"[{letter} -> pad_{name}.txt]  sha256={h}")
        report.append(f"    specialized (preamble+{letter}) isolated tokens : {base_iso}")
        report.append(f"    + padding isolated tokens                       : {padded_iso - base_iso}")
        report.append(f"    padded system prompt: full={padded_full} (target {target}), "
                      f"isolated={padded_iso} (target {iso_target})")
        report.append(f"    difference vs target: {diff_pct:+.2f}%   {'OK' if ok else 'OUT OF TOLERANCE'}")
        report.append("")

    # condition-level totals (sum over the 3 agents) for the token analysis (Part 15)
    gen_total = 3 * iso_target
    spec_total = sum(system_prompt_tokens(p) for _, p, _ in plan)
    c_total = sum(system_prompt_tokens(
        sp([{"R": R, "S": S, "V": V}[l]]) + "\n\n" + _read(f"pad_{n}.txt")) for n, _, l in plan)
    report.append("Condition totals (sum of 3 agents, isolated system tokens):")
    report.append(f"    A generalist                 : {gen_total}")
    report.append(f"    B specialized                : {spec_total}")
    report.append(f"    C specialized_length_matched : {c_total}  "
                  f"(vs A: {100.0*(c_total-gen_total)/gen_total:+.2f}%)")
    report.append("")
    report.append("RESULT: " + ("ALL AGENTS WITHIN TOLERANCE" if all_ok else "SOME AGENTS OUT OF TOLERANCE — review"))

    rp = os.path.join(ROOT, "results", "v2", "prompt_length_match_report.txt")
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    text = "\n".join(report)
    with open(rp, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nWritten pad_*.txt to {PDIR} and report to {rp}")


if __name__ == "__main__":
    main()
