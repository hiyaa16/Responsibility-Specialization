# v2 benchmark source provenance

All source files below are vendored verbatim; **ground truth is never modified**. The v2
task set (`data/tasks_v2.jsonl`) is built from these by `data/build_tasks_v2.py`.

## MuSiQue (answerable, dev split) — native structure, no reconstruction
- File: `musique/musique_ans_v1.0_dev.jsonl`
- Source: MuSiQue v1.0, StonyBrookNLP — https://github.com/StonyBrookNLP/musique
- Obtained: Google Drive id `1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h` (`musique_v1.0.zip`), via `gdown`, 2026-10-03
- sha256: `15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b`
- Each item: `question`, `answer`, `answer_aliases`, 20 `paragraphs` (2 gold `is_supporting:true`
  + 18 distractors), and `question_decomposition` (intermediate sub-answers). `required_info` =
  intermediate answers; answer + aliases for eval.
- **Deviation:** we keep the 2 gold paragraphs and sample 8 of the 18 distractors (fixed seed) ->
  10 paragraphs/question, matching the HotpotQA distractor budget and keeping CPU runtime
  tractable. This reduces the retrieval haystack vs MuSiQue-standard; documented, applied to all
  conditions identically.

## HotpotQA (dev, distractor setting) — RECONSTRUCTED (official hosts unreachable from this env)
Official hosts were unreachable at build time: CMU `curtis.ml.cmu.edu` timed out; all of
huggingface.co was network-blocked. We therefore use a reachable third-party **subset** and
reconstruct per-question contexts. This is documented as a deliberate, transparent deviation.
- Source repo: `dhanshri0507/hotpotqa-rag-retrieval-eval` @ commit
  `a99ed166f165da4454385e834524f907639a7ed1` (GitHub), obtained 2026-10-03
- The repo's `dataset_manifest.json` records it as a 300-question sample (seed 42) of the
  **official** `hotpot_dev_distractor_v1.json`, source sha256
  `4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea`.
- Files (sha256):
  - `hotpotqa/sampled_questions.json`  `c893539c7a3f9d5d385226e5f78e135d0dcdbe1718a4875e581c39659b6e801e` — {_id, question, answer, type, level}
  - `hotpotqa/qrels.json`              `941052ba184e3a8f22159ed7d234d555b28aa1d3f70b00f53b62d695923ccccc` — {_id: [gold doc titles]}
  - `hotpotqa/corpus.jsonl`           `ace7a4883288ccf19bbbe053a4cb56e6b85110ae7936496301a4e87de7db5670` — {doc_id/title, text}
  - `hotpotqa/dataset_manifest.json`  `cce9aaaf9110d8c393c01075d523ad6635e08ca7b17d85caa15c45b27dddefd7`
- **Reconstruction rule** (in `build_tasks_v2.py`): per question, gold paragraphs = the
  `qrels` titles' corpus text; distractors = a fixed-seed sample of other corpus docs;
  context = gold + distractors, shuffled by a fixed seed. The ANSWER is the official answer
  (unmodified). The distractor *composition* is our reconstruction, not HotpotQA's official
  8 distractors — recorded here so no one mistakes it for the native distractor set.
