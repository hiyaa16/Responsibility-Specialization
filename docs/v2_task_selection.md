# v2 task selection & suitability (50 harder tasks)

Frozen file: `data/tasks_v2.jsonl` — sha256 `3cf99d728714e12c913eff5d4d84d9fa6a19371fbabdc2372e76aff5259e170d`  (freeze before the full run)

Totals: 50 tasks | by source {'musique': 20, 'hotpotqa': 20, 'handauthored': 10} | by difficulty {'medium': 8, 'hard': 42} | by hops {2: 28, 3: 7, 4: 5, 0: 10}

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


## Per-task table

| task_id | source | source_id | difficulty | hops | answer_type | eval method | answer |
|---|---|---|---|---|---|---|---|
| MQ01 | musique | 2hop__814856_110882 | medium | 2 | qa | SQuAD-normalized match + aliases | 1 January 1986 |
| MQ02 | musique | 2hop__771477_149987 | medium | 2 | qa | SQuAD-normalized match + aliases | Greek mythology |
| MQ03 | musique | 2hop__39063_593388 | medium | 2 | qa | SQuAD-normalized match + aliases | Sir Robert Peel, 1st Baronet |
| MQ04 | musique | 2hop__129499_85379 | medium | 2 | qa | SQuAD-normalized match + aliases | February 14, 1912 |
| MQ05 | musique | 2hop__82378_158266 | medium | 2 | qa | SQuAD-normalized match + aliases | The Australian Ballet |
| MQ06 | musique | 2hop__263728_705527 | medium | 2 | qa | SQuAD-normalized match + aliases | Ahmad Shah Bahadur |
| MQ07 | musique | 2hop__506099_20057 | medium | 2 | qa | SQuAD-normalized match + aliases | Richard Stallman |
| MQ08 | musique | 2hop__59133_512508 | medium | 2 | qa | SQuAD-normalized match + aliases | Manhattan Project |
| MQ09 | musique | 3hop1__354480_834494_34053 | hard | 3 | qa | SQuAD-normalized match + aliases | 1905 |
| MQ10 | musique | 3hop1__158139_19788_15107 | hard | 3 | qa | SQuAD-normalized match + aliases | 48% |
| MQ11 | musique | 3hop2__89048_809962_66294 | hard | 3 | qa | SQuAD-normalized match + aliases | Lawrence Hilton - Jacobs |
| MQ12 | musique | 3hop1__664835_650651_7262 | hard | 3 | qa | SQuAD-normalized match + aliases | Walter Sabo |
| MQ13 | musique | 3hop1__17192_78396_54865 | hard | 3 | qa | SQuAD-normalized match + aliases | 1776 |
| MQ14 | musique | 3hop1__626214_503371_21711 | hard | 3 | qa | SQuAD-normalized match + aliases | built in the 15th century |
| MQ15 | musique | 3hop2__132957_379231_40768 | hard | 3 | qa | SQuAD-normalized match + aliases | 1981 |
| MQ16 | musique | 4hop2__71753_685934_707... | hard | 4 | qa | SQuAD-normalized match + aliases | 1932 |
| MQ17 | musique | 4hop3__862_846_613770_7713 | hard | 4 | qa | SQuAD-normalized match + aliases | about 400 years |
| MQ18 | musique | 4hop1__107309_457883_65... | hard | 4 | qa | SQuAD-normalized match + aliases | Walter Sabo |
| MQ19 | musique | 4hop1__31050_725495_499... | hard | 4 | qa | SQuAD-normalized match + aliases | German Renaissance |
| MQ20 | musique | 4hop2__161602_426860_88... | hard | 4 | qa | SQuAD-normalized match + aliases | Thein Sein |
| HP01 | hotpotqa | 5a78e3b955429974737f78f1 | hard | 2 | qa | SQuAD-normalized match + aliases | Sherwood Stewart |
| HP02 | hotpotqa | 5a8e5f1f5542995a26add4d6 | hard | 2 | qa | SQuAD-normalized match + aliases | Sapsali |
| HP03 | hotpotqa | 5a74f5155542993748c89750 | hard | 2 | qa | SQuAD-normalized match + aliases | The Colomac Mine |
| HP04 | hotpotqa | 5a74feb75542996c70cfae6d | hard | 2 | qa | SQuAD-normalized match + aliases | Ali Qushji |
| HP05 | hotpotqa | 5ac07fff554299294b219006 | hard | 2 | qa | SQuAD-normalized match + aliases | tennis |
| HP06 | hotpotqa | 5a88b3b4554299206df2b336 | hard | 2 | qa | SQuAD-normalized match + aliases | music |
| HP07 | hotpotqa | 5ab6e848554299710c8d1faa | hard | 2 | qa | SQuAD-normalized match + aliases | August 19, 1968 |
| HP08 | hotpotqa | 5ae143ed55429920d5234360 | hard | 2 | qa | SQuAD-normalized match + aliases | 1755 |
| HP09 | hotpotqa | 5a7e05ba5542995f4f402392 | hard | 2 | qa | SQuAD-normalized match + aliases | The Late Late Show |
| HP10 | hotpotqa | 5ae1d0e8554299234fd0430a | hard | 2 | qa | SQuAD-normalized match + aliases | Guthred or Guthfrith |
| HP11 | hotpotqa | 5a78e86f55429974737f78fb | hard | 2 | qa | SQuAD-normalized match + aliases | the long history of Japan |
| HP12 | hotpotqa | 5a8aeadc55429950cd6afbe0 | hard | 2 | qa | SQuAD-normalized match + aliases | January 14, 2010 |
| HP13 | hotpotqa | 5ab2659e554299340b5254b2 | hard | 2 | qa | SQuAD-normalized match + aliases | the Parthian Empire |
| HP14 | hotpotqa | 5ab9b7d555429970cfb8eb7a | hard | 2 | qa | SQuAD-normalized match + aliases | Henry Lau |
| HP15 | hotpotqa | 5a82817555429954d2e2eb5a | hard | 2 | qa | SQuAD-normalized match + aliases | "Naked Killer" |
| HP16 | hotpotqa | 5ae4a1ef55429970de88d9e7 | hard | 2 | qa | SQuAD-normalized match + aliases | 2 March 1972 |
| HP17 | hotpotqa | 5ab4107a554299753aec5a2f | hard | 2 | qa | SQuAD-normalized match + aliases | 1961 |
| HP18 | hotpotqa | 5a9030ef5542995651fb50e4 | hard | 2 | qa | SQuAD-normalized match + aliases | American football |
| HP19 | hotpotqa | 5a7dc2495542997cc2c4749a | hard | 2 | qa | SQuAD-normalized match + aliases | 1984 South Asian Games |
| HP20 | hotpotqa | 5a8d12ca5542994ba4e3dbe2 | hard | 2 | qa | SQuAD-normalized match + aliases | Nairobi, Kenya |
| HA01 | handauthored | HA01 | hard | 0 | numeric | numeric match (tolerance) | 1963.28 |
| HA02 | handauthored | HA02 | hard | 0 | numeric | numeric match (tolerance) | 6 |
| HA03 | handauthored | HA03 | hard | 0 | numeric | numeric match (tolerance) | 45 |
| HA04 | handauthored | HA04 | hard | 0 | numeric | numeric match (tolerance) | 30 |
| HA05 | handauthored | HA05 | hard | 0 | set | normalized set-subset | ['lamp', 'charger'] |
| HA06 | handauthored | HA06 | hard | 0 | string | normalized string match | 13:00 |
| HA07 | handauthored | HA07 | hard | 0 | numeric | numeric match (tolerance) | 56 |
| HA08 | handauthored | HA08 | hard | 0 | numeric | numeric match (tolerance) | 1331.0 |
| HA09 | handauthored | HA09 | hard | 0 | numeric | numeric match (tolerance) | 300 |
| HA10 | handauthored | HA10 | hard | 0 | numeric | numeric match (tolerance) | 9 |
