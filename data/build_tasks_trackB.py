"""Track B task generator — parametric K-step compositional tasks with EXACT ground truth.

Why synthetic + parametric: Track B needs a controllable *decomposition-depth* axis so we can test
the depth/complexity x condition INTERACTION (does specialization start to help once a generalist's
single context is overloaded?). K = number of chained reasoning sub-steps. Each task states K atomic
facts (plus distractors that force a real retrieval step) and asks for a value that requires chaining
all K. Answer is computed deterministically here, so correctness needs no LLM judge.

Emits data/tasks_trackB.jsonl in the SAME schema as tasks_v2.jsonl (schema.Task):
  difficulty = "K2"/"K4"  (the complexity level, also stored in hops)
  answer_type = "numeric", tolerance 0
  source = "synthetic_trackB"

A single-agent SOLVABILITY SCREEN (prereg Track B, step 1) is applied at PILOT time with the real
model (a task a lone generalist already aces at K=4 is too easy); it is NOT run here so generation
stays offline and deterministic. Run:  python data/build_tasks_trackB.py
"""
from __future__ import annotations
import os, json, random

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "data", "tasks_trackB.jsonl")

ENTITIES = ["Blue", "Red", "Green", "Amber", "Violet", "Silver", "Coral", "Teal",
            "Maroon", "Olive", "Indigo", "Crimson"]
ITEMS = ["crates", "pallets", "barrels", "cartons", "drums", "sacks"]
UNITS = ["units", "items", "parts", "pieces", "tokens"]

def _gen_one(rng, K, tid):
    """Build a K-step chain: v0 given; each step applies an op depending on the previous value.
    Context states the needed facts + distractors; question asks for the final value."""
    ents = rng.sample(ENTITIES, k=min(len(ENTITIES), K + 2))
    item = rng.choice(ITEMS); unit = rng.choice(UNITS)
    facts, steps = [], []
    base = rng.randint(3, 12)
    facts.append(f"The {ents[0]} depot holds {base} {item}.")
    cur = base
    prev_ent = ents[0]
    op_log = [f"{ents[0]}={base}"]
    for k in range(1, K):
        op = rng.choice(["mul", "add", "sub", "per"])
        e = ents[k]
        if op == "mul":
            m = rng.randint(2, 5)
            facts.append(f"The {e} depot holds {m} times as many {item} as the {prev_ent} depot.")
            cur = cur * m; op_log.append(f"{e}={prev_ent}*{m}={cur}")
        elif op == "add":
            a = rng.randint(4, 20)
            facts.append(f"The {e} depot holds {a} more {item} than the {prev_ent} depot.")
            cur = cur + a; op_log.append(f"{e}={prev_ent}+{a}={cur}")
        elif op == "sub":
            a = rng.randint(2, min(9, max(2, cur - 1)))
            facts.append(f"The {e} depot holds {a} fewer {item} than the {prev_ent} depot.")
            cur = cur - a; op_log.append(f"{e}={prev_ent}-{a}={cur}")
        else:  # per: convert items->units
            per = rng.randint(5, 15)
            facts.append(f"Each {item[:-1]} in the {e} depot contains {per} {unit}.")
            cur = cur * per; op_log.append(f"{e}_{unit}={prev_ent}*{per}={cur}")
        prev_ent = e
    answer = cur
    target_ent = prev_ent
    # distractors: plausible but unused facts (force genuine retrieval)
    distractors = []
    for de in [e for e in ENTITIES if e not in ents][:K]:
        distractors.append(f"The {de} depot holds {rng.randint(3,40)} {rng.choice(ITEMS)}.")
    ctx_lines = facts + distractors
    rng.shuffle(ctx_lines)
    context = " ".join(ctx_lines)
    question = (f"Working step by step from the depot facts, how many {unit if 'per' in str(op_log) else item} "
                f"are associated with the {target_ent} depot? Give a single number.")
    required = [str(base)] + [ln.split('=')[-1] for ln in op_log]
    return {
        "task_id": tid, "question": question, "context": context,
        "required_info": [facts[0]] + facts[1:],
        "answer": float(answer), "answer_type": "numeric", "tolerance": 0.0,
        "expected_reasoning": " ; ".join(op_log),
        "difficulty": f"K{K}", "category": "compositional",
        "answer_aliases": [], "source": "synthetic_trackB", "source_id": f"synthK{K}-{tid}", "hops": K,
    }

def main(n_per_level=20, levels=(2, 4), seed=20261022):
    rng = random.Random(seed)
    tasks = []
    for K in levels:
        made = 0
        while made < n_per_level:
            tid = f"TB-K{K}-{made:02d}"
            t = _gen_one(rng, K, tid)
            # keep answers in a sane, checkable range
            if 0 < t["answer"] < 100000:
                tasks.append(t); made += 1
    with open(OUT, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"wrote {len(tasks)} tasks -> {OUT}")
    import collections
    print("by difficulty:", dict(collections.Counter(t["difficulty"] for t in tasks)))
    # show one example per level
    for K in levels:
        ex = next(t for t in tasks if t["difficulty"] == f"K{K}")
        print(f"\n--- example K={K} (answer={ex['answer']}) ---")
        print("Q:", ex["question"])
        print("CTX:", ex["context"][:300])
        print("steps:", ex["expected_reasoning"])

if __name__ == "__main__":
    main()
