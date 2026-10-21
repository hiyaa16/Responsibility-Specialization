V3 EXECUTION PACK — PRE-RUN ARTIFACTS (NOT FROZEN, NOT PREREGISTERED, NOT RUN)

WHAT IS INCLUDED
- Exact prompt source files found in the previous prompt audit.
- Two alternative generic filler families (C1 and C2). They are candidates, NOT neutral by assumption.
- build_and_count_prompts.py: counts full prompts via the Ollama /api/chat prompt_eval_count on the actual runtime, then generates per-role C1/C2 candidate prompts and hashes.
- protocol amendment draft and decision memo.

HOW TO RUN THE TOKEN COUNT SCRIPT
1. Copy this folder to the machine where the experiment's Ollama server runs.
2. Confirm the model tag is exact: `ollama list`.
3. Start Ollama and run: `python build_and_count_prompts.py --url http://localhost:11434 --model qwen2.5:7b --tolerance 0.01`
4. Inspect runtime_token_counts.csv and generated_candidate_prompts/. Script exits with an error if the endpoint fails; it does not guess counts.
5. Repeat with exact model tags for Llama/Mistral only if those replications are approved. Do not assume Qwen token counts transfer.

IMPORTANT DESIGN LIMITATIONS
- C1/C2 natural-language fillers can still change model behavior. Generic wording and matched token counts do not prove neutrality.
- The script greedily chooses whole sentences based on count. Human review is required; if it cannot reach the tolerance, report the achieved mismatch rather than editing sentences into fragments.
- The probe-count method compares full prompt counts under the same runtime and fixed user probe. If production prompt serialization differs from this script, adapt the script to call the exact production prompt builder before freezing.
- Existing V2 runs remain immutable. V3 runs must be kept in a separate directory and experiment_id namespace.
- No confirmatory run is authorized until protocol, prompt hashes, task set, outcome definitions, run order, multiplicity plan, and freeze timestamp are recorded.
