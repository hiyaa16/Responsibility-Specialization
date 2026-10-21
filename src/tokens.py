"""Exact token counting using the SAME tokenizer the experiment runs on.

v2 note (prompt-length control, confound C1): v1 estimated system-prompt tokens as
`len(chars)//4`, which is only a rough proxy. For the length-matched condition we must
size padding by *real* tokens. The faithful tokenizer here is the one Qwen2.5:7b actually
uses inside Ollama, so we measure token counts via Ollama's `prompt_eval_count` rather
than a separate HuggingFace tokenizer (the HF tokenizer would only approximate the GGUF
tokenizer, and this environment has no network access to the HF hub anyway).

Method: `prompt_eval_count` for a chat request is the number of prompt tokens the model
evaluated. With a FIXED user probe, the only thing that varies between calls is the system
message, so differences in `prompt_eval_count` are exactly the system prompt's token cost
(plus a constant template wrapper that cancels in any comparison). This is deterministic
(verified: identical inputs return identical counts; counts scale with content length).

`system_prompt_tokens(S)` returns the isolated token cost of system content S
(wrapper removed via a no-system baseline), so the numbers are human-readable and
directly comparable across conditions.
"""
from __future__ import annotations
import json
import urllib.request

# Fixed probe used for all measurements. Its exact content is irrelevant to the RESULT
# of a comparison (it is constant across calls) but must stay fixed so cached reports
# reproduce. Chosen to resemble the runner's first user turn.
_PROBE = "TASK:\nPROBE QUESTION\n"

_DEFAULT_URL = "http://localhost:11434/api/chat"
_DEFAULT_MODEL = "qwen2.5:7b"

_cache: dict = {}


def _prompt_eval(messages, url: str, model: str) -> int:
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        # num_predict=1 keeps generation negligible; we only read prompt_eval_count.
        "options": {"num_predict": 1, "temperature": 0, "seed": 0},
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return int(payload["prompt_eval_count"])


def _baseline(url: str, model: str) -> int:
    key = ("baseline", url, model)
    if key not in _cache:
        _cache[key] = _prompt_eval([{"role": "user", "content": _PROBE}], url, model)
    return _cache[key]


def full_prompt_tokens(system: str, url: str = _DEFAULT_URL, model: str = _DEFAULT_MODEL) -> int:
    """prompt_eval_count for (system + fixed probe). Use for matching: the probe/template
    offset is constant, so equal values => equal system-prompt budgets."""
    key = ("full", system, url, model)
    if key not in _cache:
        _cache[key] = _prompt_eval(
            [{"role": "system", "content": system}, {"role": "user", "content": _PROBE}],
            url, model)
    return _cache[key]


def system_prompt_tokens(system: str, url: str = _DEFAULT_URL, model: str = _DEFAULT_MODEL) -> int:
    """Isolated token cost of the system content (template wrapper removed). Human-readable;
    equals full_prompt_tokens(system) - full_prompt_tokens(no system)."""
    if not system:
        return 0
    return full_prompt_tokens(system, url, model) - _baseline(url, model)


if __name__ == "__main__":
    # quick self-check
    print("baseline (no system):", _baseline(_DEFAULT_URL, _DEFAULT_MODEL))
    print("sys tokens for 'hello world':", system_prompt_tokens("hello world"))
