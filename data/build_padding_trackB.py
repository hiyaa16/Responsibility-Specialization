"""Freeze Track B length-matching padding for condition C, per depth x role, PER MODEL.

Same neutral, non-operative pool and matching rule as data/build_padding.py (the padding carries no
task/role/reasoning content; its only property is length). For each depth D and each role, size the
specialist's padded per-agent system prompt (preamble + own block + padding) to the GENERALIST per-agent
budget at that depth (preamble + ALL D blocks), measured with THE TARGET MODEL'S OWN TOKENIZER (via Ollama
prompt_eval_count), to within +/-2%.

Model is read from the config file. Output filenames carry the model tag so Qwen / Llama / Mistral each
get their own frozen, correctly-sized padding:
    model_tag unset (Qwen default)  -> config/prompts/pad_trackB_D{D}_{role}.txt
    model_tag=llama31_8b            -> config/prompts/pad_trackB_llama31_8b_D{D}_{role}.txt

Run once per model, with Ollama + that model pulled, BEFORE its run:
    python data/build_padding_trackB.py                                  # Qwen (default config)
    python data/build_padding_trackB.py --config config/experiment_trackB_llama.yaml
    python data/build_padding_trackB.py --config config/experiment_trackB_mistral.yaml
"""
from __future__ import annotations
import os, sys, hashlib, argparse
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tokens import full_prompt_tokens  # noqa: E402
from build_padding import _assemble  # reuse the exact frozen neutral pool  # noqa: E402
from conditions_trackB import DEPTH_ROLES, _BLOCK_FILE  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PDIR = os.path.join(ROOT, "config", "prompts")
TOL = 0.02


def _read(n):
    with open(os.path.join(PDIR, n), "r", encoding="utf-8") as f:
        return f.read().strip()


def _fit(prefix_system, target_tokens, url, model):
    lo, hi, best = 1, 1200, None
    while lo <= hi:
        mid = (lo + hi) // 2
        pad = _assemble(mid)
        tot = full_prompt_tokens(prefix_system + "\n\n" + pad, url, model)
        if best is None or abs(tot - target_tokens) < abs(best[1] - target_tokens):
            best = (pad, tot)
        if tot < target_tokens:
            lo = mid + 1
        elif tot > target_tokens:
            hi = mid - 1
        else:
            return pad, tot
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment_trackB.yaml")
    args = ap.parse_args()
    with open(os.path.join(ROOT, args.config), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model = cfg["model"]
    model_tag = cfg.get("model_tag")                       # None for Qwen default
    url = cfg.get("tokenizer_url", "http://localhost:11434/api/chat")
    depths = cfg["depths"]
    tag = f"{model_tag}_" if model_tag else ""

    pre = _read("preamble.txt")
    report = [f"TRACK B PROMPT-LENGTH MATCHING REPORT  (model={model}, tag={model_tag or 'qwen-default'})",
              f"Tokenizer: {model} via Ollama prompt_eval_count.  Tolerance +/-{TOL*100:.0f}%", ""]
    all_ok = True
    for D in depths:
        roles = DEPTH_ROLES[D]
        all_blocks = [_read(_BLOCK_FILE[r]) for r in roles]
        gen_sp = pre + "\n\n" + "\n\n".join(all_blocks)      # generalist per-agent prompt at depth D
        target = full_prompt_tokens(gen_sp, url, model)
        report.append(f"=== depth {D}: generalist per-agent budget = {target} tok (full) ===")
        lo_ok, hi_ok = target * (1 - TOL), target * (1 + TOL)
        for role in roles:
            prefix = pre + "\n\n" + _read(_BLOCK_FILE[role])
            pad, achieved = _fit(prefix, target, url, model)
            fname = f"pad_trackB_{tag}D{D}_{role}.txt"
            with open(os.path.join(PDIR, fname), "w", encoding="utf-8") as f:
                f.write(pad + "\n")
            h = hashlib.sha256((pad + "\n").encode()).hexdigest()[:16]
            ok = lo_ok <= achieved <= hi_ok
            all_ok = all_ok and ok
            report.append(f"  {fname}  achieved={achieved} (target {target}) "
                          f"{'OK' if ok else 'OUT OF TOL'}  sha={h}")
        report.append("")
    report.append("RESULT: " + ("ALL WITHIN TOLERANCE" if all_ok else "SOME OUT OF TOLERANCE — review"))
    rp = os.path.join(ROOT, cfg["out_dir"], "prompt_length_match_trackB.txt")
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    text = "\n".join(report)
    with open(rp, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nWrote pad_trackB_{tag}*.txt to {PDIR}\nReport -> {rp}")


if __name__ == "__main__":
    main()
