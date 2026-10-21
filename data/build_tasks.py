"""Builds data/tasks.jsonl — the FIXED benchmark (~30 tasks). Freeze before running.

Every task ships a `context` passage containing the facts needed PLUS distractors, so
that retrieval (selecting the right facts), reasoning (computing), and verification
(catching errors) all genuinely matter. Answers are deterministically checkable.

`required_info` lists distinctive tokens that MUST reach downstream agents; the failure
classifier uses them as an approximate signal for F2 (retrieval) and F5 (info loss).

Run once:  python data/build_tasks.py
"""
import json
import os

T = []


def add(task_id, question, context, required_info, answer, answer_type,
        difficulty, category, tolerance=0.0, expected_reasoning=""):
    T.append({
        "task_id": task_id, "question": question, "context": context,
        "required_info": required_info, "answer": answer, "answer_type": answer_type,
        "tolerance": tolerance, "expected_reasoning": expected_reasoning,
        "difficulty": difficulty, "category": category,
    })


# ---------------- EASY (single retrieval + one operation) ----------------
add("T01", "What is the total cost of 3 notebooks and 2 pens?",
    "Stationery price list. Notebook: $4 each. Pen: $2 each. Stapler: $7 each. "
    "Eraser: $1 each. Highlighter: $3 each. Delivery is free over $20.",
    ["$4", "$2"], 16, "numeric", "easy", "arithmetic",
    expected_reasoning="3*4 + 2*2 = 16")

add("T02", "How many hours are there between 09:15 and 14:45 on the same day?",
    "Meeting log: standup 09:15, review 14:45, retro 16:00. Lunch is 12:30-13:15. "
    "The office opens at 08:00 and closes at 18:00.",
    ["09:15", "14:45"], 5.5, "numeric", "easy", "time",
    expected_reasoning="14:45 - 09:15 = 5h30m = 5.5h")

add("T03", "What is the capital city named in the report?",
    "The 2019 summit was hosted in Canberra, the capital of Australia. Sydney and "
    "Melbourne are larger cities but are not the capital. Wellington is New Zealand's capital.",
    ["Canberra"], "Canberra", "string", "easy", "fact-lookup")

add("T04", "A car travels 240 km using 16 litres of fuel. What is its fuel economy in km per litre?",
    "Trip data: distance 240 km, fuel used 16 litres, average speed 80 km/h, "
    "trip time 3 hours, 2 passengers. Tank capacity is 50 litres.",
    ["240", "16"], 15, "numeric", "easy", "rates",
    expected_reasoning="240/16 = 15")

add("T05", "What is 15% of 320?",
    "Invoice: subtotal 320 dollars, tax rate 15%, shipping 12 dollars, "
    "discount code gives 5% off shipping only.",
    ["320", "15%"], 48, "numeric", "easy", "percentage",
    expected_reasoning="0.15*320 = 48")

add("T06", "How many students passed if 40 out of 50 students passed the exam?",
    "Class results: 50 students total, 40 passed, 10 failed, 3 were absent (counted as failed), "
    "average score 68. The pass mark is 40.",
    ["40", "50"], 40, "numeric", "easy", "reading",
    expected_reasoning="direct: 40 passed")

add("T07", "What is the perimeter of a rectangle that is 8 m long and 5 m wide?",
    "Garden plot: length 8 m, width 5 m, area to be turfed, fence needed around the "
    "perimeter, gate is 1 m wide. Soil depth 0.3 m.",
    ["8", "5"], 26, "numeric", "easy", "geometry",
    expected_reasoning="2*(8+5) = 26")

add("T08", "Convert 2.5 kilometres to metres.",
    "Route notes: leg 1 is 2.5 km, leg 2 is 800 m, total elevation gain 120 m, "
    "the runner weighs 70 kg.",
    ["2.5 km"], 2500, "numeric", "easy", "unit-conversion",
    expected_reasoning="2.5 * 1000 = 2500")

add("T09", "What is the average of the three test scores?",
    "Scores across subjects: math 90, science 75, history 60, art 100 (not counted). "
    "Only math, science and history count toward this average.",
    ["90", "75", "60"], 75, "numeric", "easy", "statistics",
    expected_reasoning="(90+75+60)/3 = 75")

add("T10", "Which fruit is the cheapest per kilogram?",
    "Market prices per kg: apples $3.20, bananas $1.80, cherries $9.50, "
    "grapes $4.10. Bananas are on a 2-for-1 offer today.",
    ["bananas", "$1.80"], "bananas", "string", "easy", "selection")

# ---------------- MEDIUM (multi-step, needs care) ----------------
add("T11", "If a $1200 laptop is discounted 20% and then 8% sales tax is added, what is the final price?",
    "Checkout: laptop base price $1200, member discount 20%, sales tax 8%, "
    "optional warranty $99 (not selected), gift wrap $5 (not selected).",
    ["$1200", "20%", "8%"], 1036.8, "numeric", "medium", "finance",
    tolerance=0.05, expected_reasoning="1200*0.8=960; 960*1.08=1036.8")

add("T12", "A tank fills at 12 L/min and drains at 4 L/min. Starting empty, how many minutes to reach 96 L?",
    "Pump specs: inflow 12 litres/min, outflow (open drain) 4 litres/min, tank capacity 200 L, "
    "target level 96 L. A second pump (10 L/min) is switched off.",
    ["12", "4", "96"], 12, "numeric", "medium", "rates",
    expected_reasoning="net 8 L/min; 96/8 = 12 min")

add("T13", "What is the total distance of a trip with three legs: 45 km, 3/4 of that, and 20 km more than the second leg?",
    "Itinerary: leg A = 45 km, leg B = three-quarters of leg A, leg C = leg B plus 20 km. "
    "Fuel stops every 100 km. Average speed 60 km/h.",
    ["45", "three-quarters", "20 km"], 132.5, "numeric", "medium", "word-problem",
    tolerance=0.01, expected_reasoning="A=45; B=33.75; C=53.75; total=45+33.75+53.75=132.5")

add("T14", "How many full crates are needed to pack 250 apples if each crate holds 24?",
    "Packing: 250 apples, crate capacity 24 apples, damaged apples discarded (7 damaged, "
    "already removed from the 250), pallets hold 10 crates.",
    ["250", "24"], 11, "numeric", "medium", "word-problem",
    expected_reasoning="ceil(250/24)=ceil(10.42)=11")

add("T15", "Team A scored 3 goals in the first half and twice as many in the second. Team B scored 4 total. What is the goal difference (A minus B)?",
    "Match report: Team A first half 3 goals, second half double the first half, "
    "Team B scored 4 goals total, attendance 20000, kickoff 15:00.",
    ["3 goals", "double", "4 goals"], 5, "numeric", "medium", "word-problem",
    expected_reasoning="A=3+6=9; B=4; diff=5")

add("T16", "A recipe for 4 servings needs 300 g flour. How much flour for 10 servings?",
    "Recipe (4 servings): flour 300 g, sugar 120 g, butter 90 g, eggs 2. "
    "Oven 180 C. You want to make 10 servings.",
    ["300 g", "4 servings", "10 servings"], 750, "numeric", "medium", "proportion",
    expected_reasoning="300/4*10 = 750")

add("T17", "What is the compound value of $500 after 2 years at 10% annual interest, compounded yearly?",
    "Account: principal $500, annual rate 10%, compounded yearly, term 2 years, "
    "monthly fee $0, inflation 3% (ignore for this calculation).",
    ["$500", "10%", "2 years"], 605, "numeric", "medium", "finance",
    tolerance=0.5, expected_reasoning="500*1.1^2 = 605")

add("T18", "Which two people can both attend a meeting held from 13:00 to 14:00?",
    "Availability. Ana: free 09:00-12:00. Ben: free 12:30-15:00. Cara: free 13:00-14:30. "
    "Dan: free 15:00-17:00. The meeting is 13:00-14:00.",
    ["Ben", "Cara", "13:00"], ["Ben", "Cara"], "set", "medium", "scheduling",
    expected_reasoning="Only Ben and Cara cover 13:00-14:00")

add("T19", "A worker earns $18/hour for 40 hours and 1.5x for 6 overtime hours. What is the weekly pay?",
    "Payroll: base rate $18/hour, standard week 40 hours, overtime this week 6 hours at "
    "1.5x rate, tax withheld 0 for this exercise, bonus $0.",
    ["$18", "40 hours", "6", "1.5"], 882, "numeric", "medium", "finance",
    expected_reasoning="40*18=720; 6*27=162; total 882")

add("T20", "If a rectangle's area is 48 m^2 and its width is 6 m, what is its length?",
    "Room: area 48 square metres, width 6 metres, ceiling height 3 m, "
    "one door and two windows. Find the length.",
    ["48", "6"], 8, "numeric", "medium", "geometry",
    expected_reasoning="48/6 = 8")

add("T21", "Out of 3 shirts and 2 pairs of trousers, how many distinct shirt-trouser outfits are possible?",
    "Wardrobe: 3 shirts (red, blue, green), 2 pairs of trousers (black, tan), "
    "4 pairs of socks (not part of an outfit here), 1 belt.",
    ["3 shirts", "2 pairs"], 6, "numeric", "medium", "combinatorics",
    expected_reasoning="3*2 = 6")

add("T22", "A train leaves at 10:20 and the journey takes 2 hours 50 minutes. What is the arrival time (HH:MM, 24h)?",
    "Timetable: departure 10:20, journey duration 2 h 50 min, one stop of 10 min "
    "already included in the duration, time zone unchanged.",
    ["10:20", "2 h 50"], "13:10", "string", "medium", "time",
    expected_reasoning="10:20 + 2:50 = 13:10")

# ---------------- HARD (several steps + verification traps) ----------------
add("T23", "A shop buys widgets at $8, marks them up 25%, then during a sale takes 10% off the marked price. What is the sale price, rounded to the nearest cent?",
    "Pricing chain: cost $8, markup 25% on cost, then a 10% sale discount on the marked "
    "price. Loyalty members get an extra 5% (this customer is NOT a member).",
    ["$8", "25%", "10%"], 9.0, "numeric", "hard", "finance",
    tolerance=0.01, expected_reasoning="8*1.25=10; 10*0.9=9.00 (do NOT apply the 5%)")

add("T24", "Three friends split a $150 bill so that Amy pays twice as much as Ben, and Cy pays the same as Ben. How much does Amy pay?",
    "Dinner bill $150. Amy pays twice Ben's share. Cy pays the same as Ben. "
    "A 15% tip was already included in the $150. Split the $150.",
    ["$150", "twice", "same as Ben"], 75, "numeric", "hard", "algebra",
    expected_reasoning="2b+b+b=150 -> 4b=150 -> b=37.5 -> Amy=75")

add("T25", "A cyclist covers 30 km at 15 km/h, rests 20 minutes, then 10 km at 20 km/h. What is the total trip time in minutes?",
    "Ride: segment 1 = 30 km at 15 km/h, rest = 20 minutes, segment 2 = 10 km at 20 km/h. "
    "Wind was a tailwind (ignore). GPS battery 80%.",
    ["30 km", "15 km/h", "20 minutes", "10 km", "20 km/h"], 170, "numeric", "hard", "rates",
    expected_reasoning="2h + 20min + 0.5h = 120+20+30 = 170 min")

add("T26", "A worker completes a job in 6 hours; another in 12 hours. Working together, how many hours to complete the job?",
    "Rates: worker X finishes alone in 6 hours, worker Y in 12 hours. They take one "
    "15-minute break (ignore for the combined-rate calculation). Tools are shared.",
    ["6 hours", "12 hours"], 4, "numeric", "hard", "work-rate",
    tolerance=0.01, expected_reasoning="1/6+1/12=1/4 -> 4 hours")

add("T27", "If 5 machines make 5 widgets in 5 minutes, how long (in minutes) do 100 machines take to make 100 widgets?",
    "Classic setup: 5 machines produce 5 widgets in 5 minutes. Each machine works at a "
    "constant rate. Now consider 100 machines making 100 widgets.",
    ["5 machines", "5 widgets", "5 minutes"], 5, "numeric", "hard", "rates",
    expected_reasoning="1 machine makes 1 widget in 5 min; 100 machines make 100 in 5 min")

add("T28", "A rectangular pool 10 m x 4 m x 2 m deep is filled at 8 cubic metres per hour. How many hours to fill it completely?",
    "Pool: length 10 m, width 4 m, depth 2 m, fill rate 8 m^3/hour, evaporation negligible, "
    "starts empty. Surrounding deck is 1 m wide (irrelevant).",
    ["10 m", "4 m", "2 m", "8"], 10, "numeric", "hard", "geometry",
    expected_reasoning="volume=80 m^3; 80/8 = 10 hours")

add("T29", "A phone plan costs $30/month plus $0.10 per GB over 5 GB. In a month using 12 GB, what is the total bill?",
    "Plan: base $30/month includes 5 GB, then $0.10 per extra GB. This month used 12 GB. "
    "A one-off $15 handset fee was paid last month (not this month).",
    ["$30", "$0.10", "5 GB", "12 GB"], 30.7, "numeric", "hard", "finance",
    tolerance=0.01, expected_reasoning="over = 7 GB; 7*0.10=0.70; 30+0.70=30.70 (exclude last month's fee)")

add("T30", "Which items fit within a 10 kg baggage limit if you must take the laptop and then add the heaviest possible single extra item?",
    "Baggage limit 10 kg. Laptop 3 kg (must take). Options to add ONE more: camera 8 kg, "
    "books 6 kg, jacket 2 kg, tripod 7.5 kg. Choose laptop plus the heaviest single item that still fits.",
    ["10 kg", "laptop 3 kg", "books 6 kg"], ["laptop", "books"], "set", "hard", "constraint",
    expected_reasoning="laptop 3kg fixed; remaining 7kg; heaviest fitting single item is books 6kg (tripod 7.5 and camera 8 exceed)")


def main():
    out = os.path.join(os.path.dirname(__file__), "tasks.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for t in T:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"Wrote {len(T)} tasks to {out}")
    # sanity: difficulty spread
    from collections import Counter
    print(Counter(t["difficulty"] for t in T))


if __name__ == "__main__":
    main()
