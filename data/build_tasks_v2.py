"""Build data/tasks_v2.jsonl — the FROZEN 50-task harder benchmark for experiment v2.

Composition (chosen for a natural Retriever -> Solver -> Verifier decomposition and to beat
the v1 ceiling; see docs/v2_task_selection.md for the per-task suitability audit):
  20 MuSiQue (answerable dev)  — native multi-hop, 20 paragraphs (2 gold + 18 distractors)
  20 HotpotQA (reconstructed)  — official answers; context = gold (qrels) + sampled distractors
  10 hand-authored             — harder compositional tasks with verification traps

Determinism: a single frozen SEED drives all sampling/shuffling. Re-running reproduces the
exact 50 tasks. Ground truth is copied verbatim from the sources (never modified).

Run once:  python data/build_tasks_v2.py   (then freeze; hash is printed + written to the doc)
"""
from __future__ import annotations
import json
import os
import random
import hashlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "data", "sources")
SEED = 20261003                 # frozen selection seed
N_MUSIQUE, N_HOTPOT, N_HAND = 20, 20, 10
HOTPOT_DISTRACTORS = 8          # -> 10 paragraphs/question, matching the official distractor size
MUSIQUE_DISTRACTORS = 8         # cap MuSiQue to 2 gold + 8 distractors (10 paras) to match the
                                # HotpotQA distractor budget and keep CPU runtime tractable.
                                # Deliberate, documented deviation from MuSiQue's native 20-para set.


# ----------------------------- MuSiQue (native) -----------------------------
def musique_tasks():
    path = os.path.join(SRC, "musique", "musique_ans_v1.0_dev.jsonl")
    items = [json.loads(l) for l in open(path, encoding="utf-8")]
    rng = random.Random(SEED)
    buckets = {"2hop": [], "3hop": [], "4hop": []}
    for d in items:
        ans = (d.get("answer") or "").strip()
        if not (1 <= len(ans.split()) <= 6) or len(ans) < 2:
            continue                                   # keep objectively checkable short answers
        hc = d["id"].split("__")[0]
        klass = "2hop" if hc.startswith("2") else ("3hop" if hc.startswith("3") else "4hop")
        buckets[klass].append(d)
    plan = {"2hop": 8, "3hop": 7, "4hop": 5}           # spread across hop depth = 20
    chosen = []
    for k, n in plan.items():
        rng.shuffle(buckets[k])
        chosen += buckets[k][:n]
    tasks = []
    for i, d in enumerate(chosen):
        hops = int(d["id"][0])
        # cap to 2 gold (is_supporting) + 8 sampled distractors (10 paras), shuffled deterministically
        gold = [p for p in d["paragraphs"] if p.get("is_supporting")]
        distract = [p for p in d["paragraphs"] if not p.get("is_supporting")]
        prng = random.Random(SEED + 100 + i)
        prng.shuffle(distract)
        paras = gold + distract[:MUSIQUE_DISTRACTORS]
        prng.shuffle(paras)
        context = "\n\n".join(f"[{p['title']}] {p['paragraph_text']}" for p in paras)
        req = []
        for step in d["question_decomposition"]:       # intermediate facts that must surface (F2/F5)
            a = (step.get("answer") or "").strip()
            if a and a not in req:
                req.append(a)
        if d["answer"] not in req:
            req.append(d["answer"])
        tasks.append(dict(
            task_id=f"MQ{i+1:02d}", question=d["question"], context=context,
            required_info=req, answer=d["answer"], answer_type="qa",
            answer_aliases=d.get("answer_aliases", []) or [], tolerance=0.0,
            expected_reasoning=" | ".join(s["question"] for s in d["question_decomposition"]),
            difficulty="medium" if hops == 2 else "hard", category="multihop-qa",
            source="musique", source_id=d["id"], hops=hops))
    return tasks


# ----------------------- HotpotQA (reconstructed) --------------------------
def hotpot_tasks():
    hp = os.path.join(SRC, "hotpotqa")
    sq = json.load(open(os.path.join(hp, "sampled_questions.json"), encoding="utf-8"))
    qrels = json.load(open(os.path.join(hp, "qrels.json"), encoding="utf-8"))
    corpus = {}
    for l in open(os.path.join(hp, "corpus.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        corpus[r.get("doc_id") or r["title"]] = r["text"]
    rng = random.Random(SEED + 1)
    cand = [q for q in sq if q["_id"] in qrels and all(t in corpus for t in qrels[q["_id"]])]
    hard = [q for q in cand if q.get("level") == "hard"] or cand
    bridge = [q for q in hard if q.get("type") == "bridge"]
    comp = [q for q in hard if q.get("type") == "comparison"]
    rng.shuffle(bridge); rng.shuffle(comp)
    chosen = (comp[:6] + bridge)[:N_HOTPOT]            # include some comparison questions
    titles = list(corpus.keys())
    tasks = []
    for i, q in enumerate(chosen):
        gold = qrels[q["_id"]]
        pool = [t for t in titles if t not in gold]
        rng.shuffle(pool)
        paras = [(t, corpus[t]) for t in gold] + [(t, corpus[t]) for t in pool[:HOTPOT_DISTRACTORS]]
        rng.shuffle(paras)
        context = "\n\n".join(f"[{t}] {txt}" for t, txt in paras)
        req = [q["answer"]] + [t for t in gold if t != q["answer"]]
        tasks.append(dict(
            task_id=f"HP{i+1:02d}", question=q["question"], context=context,
            required_info=req, answer=q["answer"], answer_type="qa", answer_aliases=[],
            tolerance=0.0,
            expected_reasoning=f"{q.get('type','?')} question; evidence in {len(gold)} gold docs",
            difficulty="hard", category="multihop-qa",
            source="hotpotqa", source_id=q["_id"], hops=2))
    return tasks


# --------------------------- Hand-authored ----------------------------------
def _h(task_id, question, context, required_info, answer, answer_type, difficulty,
       category, tolerance=0.0, expected_reasoning="", answer_aliases=None):
    return dict(task_id=task_id, question=question, context=context,
                required_info=required_info, answer=answer, answer_type=answer_type,
                answer_aliases=answer_aliases or [], tolerance=tolerance,
                expected_reasoning=expected_reasoning, difficulty=difficulty,
                category=category, source="handauthored", source_id=task_id, hops=0)


def hand_tasks():
    T = []
    T.append(_h("HA01",
        "A store offers 3 nested discounts on a $2,400 item: first 15% off, then $100 off the reduced price, then 8% off the running total. 10% sales tax applies to the final discounted price. What is the amount paid, to the nearest cent?",
        "Item list price $2,400. Promotions stack in this order: (1) 15% off list, (2) a flat $100 coupon off the already-reduced price, (3) a further 8% off the running total. Sales tax of 8% is advertised at the door but the register actually applies 10%. A membership would give an extra 5% but this buyer is NOT a member.",
        ["$2,400", "15%", "$100", "8%", "10%"], 1961.28, "numeric", "hard", "finance",
        tolerance=0.02,
        expected_reasoning="2400*.85=2040; -100=1940; *.92=1784.8; *1.10=1963.28? recompute: 1784.8*1.10=1963.28"))
    # (recomputed below to the exact frozen value)
    T[-1]["answer"] = round(((2400*0.85)-100)*0.92*1.10, 2)
    T.append(_h("HA02",
        "Three machines A, B, C fill an order. A does 1/2 the order in 6 hours; B does 1/3 of the order in 2 hours; C does the remaining 1/6 at 1/6 of the order per 4 hours. If they each work on their own portion simultaneously, how many hours until the whole order is done (the slowest portion governs)?",
        "Order split: A handles 1/2, B handles 1/3, C handles 1/6. A completes its 1/2 in 6 hours. B completes its 1/3 in 2 hours. C completes its 1/6 in 4 hours. They start together; the order is done when the LAST portion finishes. A distractor note claims the total is the sum of the times.",
        ["6 hours", "2 hours", "4 hours", "1/2", "1/3", "1/6"], 6, "numeric", "hard", "work-rate",
        expected_reasoning="parallel; finish = max(6,2,4)=6 (NOT 6+2+4)"))
    T.append(_h("HA03",
        "A tank starts at 20 L. Pump X adds 6 L/min for the first 5 minutes only; drain Y removes 2 L/min the whole time; at minute 5 a second inflow Z of 3 L/min starts. What is the volume at minute 10, in litres?",
        "Initial volume 20 L. X: +6 L/min for minutes 0-5 then OFF. Y: -2 L/min for the entire 10 minutes. Z: +3 L/min starting at minute 5 through minute 10. Tank capacity is 200 L (never exceeded).",
        ["20 L", "6 L/min", "2 L/min", "3 L/min", "5 minutes", "10"], 35, "numeric", "hard", "rates",
        expected_reasoning="0-5: (6-2)*5=+20 ->40; 5-10: (3-2)*5=+5 ->45? recompute: 20+20=40; min5-10 net +1*5=+5 ->45"))
    T[-1]["answer"] = 45
    T[-1]["expected_reasoning"] = "0-5min net +4/min=+20 ->40; 5-10min net (3-2)=+1/min=+5 ->45"
    T.append(_h("HA04",
        "Four friends compare ages. Dan is twice Cal's age. Cal is 3 years older than Bea. Bea is half Ada's age. Ada is 24. How old is Dan?",
        "Ada is 24. Bea is half of Ada. Cal is 3 years older than Bea. Dan is twice Cal. A misleading footnote says Dan is the youngest.",
        ["24", "half", "3 years older", "twice"], 30, "numeric", "hard", "algebra",
        expected_reasoning="Bea=12; Cal=15; Dan=30"))
    T.append(_h("HA05",
        "Which of these can be bought together without exceeding a $50 budget, buying the MOST expensive compatible pair? Options: lamp $38, charger $9, headphones $45, cable $6, speaker $33.",
        "Budget $50, buy exactly TWO distinct items, maximize combined price without exceeding $50. Prices: lamp $38, charger $9, headphones $45, cable $6, speaker $33. Note: headphones+lamp=$83 and headphones+speaker=$78 both exceed budget.",
        ["$50", "lamp $38", "speaker $33", "charger $9"], ["lamp", "charger"], "set", "hard", "constraint",
        expected_reasoning="pairs <=50 maximizing sum: lamp38+charger9=47; lamp+cable=44; speaker33+? 33+9=42; 38+9=47 is max -> lamp+charger"))
    T.append(_h("HA06",
        "A train departs 08:50 and is scheduled for 3 h 40 min, but incurs two delays of 25 min and 15 min, and makes up 10 min en route. What is the arrival time (HH:MM, 24h)?",
        "Departure 08:50. Base duration 3 h 40 min. Delay 1: +25 min. Delay 2: +15 min. Time recovered: -10 min. Timezone unchanged. A note says the scheduled (undelayed) arrival is 12:30.",
        ["08:50", "3 h 40", "25 min", "15 min", "10 min"], "13:00", "string", "hard", "time",
        expected_reasoning="08:50+3:40=12:30; +25+15-10=+30 ->13:00"))
    T.append(_h("HA07",
        "A rectangular garden 18 m by 12 m has a 1 m wide path around the INSIDE edge. What is the area of the path in square metres?",
        "Garden 18 m x 12 m. A path of width 1 m runs around the inside perimeter. Inner (non-path) rectangle is therefore 16 m x 10 m. Distractor: someone computed the path as the full garden area.",
        ["18", "12", "1 m", "16", "10"], 56, "numeric", "hard", "geometry",
        expected_reasoning="18*12 - 16*10 = 216-160 = 56"))
    T.append(_h("HA08",
        "You invest $1,000 at 10% per year for 3 years. Year 1 is simple interest; years 2 and 3 are compounded on the year-1 ending balance. What is the final balance to the nearest cent?",
        "Principal $1,000, rate 10%/yr, 3 years. Year 1: simple interest (adds $100 -> $1,100). Years 2-3: compound annually on the $1,100 base. Distractor: a note gives the fully-compounded figure $1,331.",
        ["$1,000", "10%", "3 years", "$1,100"], 1331.00, "numeric", "hard", "finance",
        tolerance=0.02,
        expected_reasoning="Y1 simple ->1100; 1100*1.1=1210; *1.1=1331.00"))
    T.append(_h("HA09",
        "A recipe for 6 servings uses 450 g flour and 3 eggs. You want 8 servings but only have 2 eggs. Scaling by the EGG limit, how many grams of flour should you use (to the nearest gram)?",
        "Base recipe (6 servings): 450 g flour, 3 eggs. Target 8 servings would need proportional amounts, but eggs are the binding constraint: only 2 eggs available. Scale the whole recipe by the egg ratio 2/3. Distractor: the 8-serving flour amount is 600 g.",
        ["6 servings", "450 g", "3 eggs", "2 eggs"], 300, "numeric", "hard", "proportion",
        expected_reasoning="egg-limited scale = 2/3 of base; flour=450*2/3=300"))
    T.append(_h("HA10",
        "In a class of 30, 18 play football, 15 play basketball, and 6 play neither. How many play BOTH?",
        "Class size 30. Football players 18, basketball players 15, neither 6. So 24 play at least one. Inclusion-exclusion gives both. Distractor: a note claims 3 play both.",
        ["30", "18", "15", "6"], 9, "numeric", "hard", "sets",
        expected_reasoning="at least one = 30-6 = 24; both = 18+15-24 = 9"))
    return T


def main():
    tasks = musique_tasks() + hotpot_tasks() + hand_tasks()
    assert len(tasks) == N_MUSIQUE + N_HOTPOT + N_HAND, len(tasks)
    out = os.path.join(ROOT, "data", "tasks_v2.jsonl")
    payload = "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks)
    with open(out, "w", encoding="utf-8") as f:
        f.write(payload)
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    from collections import Counter
    print(f"Wrote {len(tasks)} tasks -> {out}")
    print("by source:", dict(Counter(t["source"] for t in tasks)))
    print("by difficulty:", dict(Counter(t["difficulty"] for t in tasks)))
    print("by hops:", dict(Counter(t["hops"] for t in tasks)))
    print("answer_type:", dict(Counter(t["answer_type"] for t in tasks)))
    print(f"tasks_v2.jsonl sha256: {h}")


if __name__ == "__main__":
    main()
