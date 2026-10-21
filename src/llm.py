"""Ollama chat interface. Stdlib-only HTTP. Returns content + exact token counts.

Token counts come straight from Ollama's response (prompt_eval_count / eval_count),
so cost accounting is exact and identical in mechanism for both conditions.
"""
from __future__ import annotations
import json
import urllib.request
from dataclasses import dataclass


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int      # prompt_eval_count
    output_tokens: int      # eval_count
    done_reason: str        # "stop" normally; "length" means output hit num_predict


class OllamaClient:
    def __init__(self, url: str, model: str, temperature: float, top_p: float,
                 num_ctx: int, num_predict: int, timeout_s: int):
        self.url = url
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.timeout_s = timeout_s

    def chat(self, messages: list[dict], seed: int) -> LLMResponse:
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
                "seed": seed,
            },
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return LLMResponse(
            content=payload.get("message", {}).get("content", ""),
            prompt_tokens=int(payload.get("prompt_eval_count", 0)),
            output_tokens=int(payload.get("eval_count", 0)),
            done_reason=payload.get("done_reason", "stop"),
        )
