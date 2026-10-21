"""Uniform tool registry. EVERY agent in BOTH conditions has access to ALL tools
(confound C2): specialization is instructional only, never enforced by tool access.

- lookup(query, task)   -> returns the task's reference context (retrieval = extraction)
- calculator(expr)      -> evaluates a single arithmetic expression, deterministically
"""
from __future__ import annotations
import ast
import operator

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos, ast.FloorDiv: operator.floordiv,
}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("non-numeric constant")
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def calculator(expr: str) -> str:
    try:
        val = _safe_eval(ast.parse(expr, mode="eval"))
        # keep integers looking like integers
        if isinstance(val, float) and val.is_integer():
            val = int(val)
        return str(val)
    except Exception as e:  # deterministic error string, still logged
        return f"ERROR: could not evaluate '{expr}' ({e})"


def lookup(query: str, task) -> str:
    # Deterministic: returns the full reference context. The retriever's job is the
    # cognitive extraction of the relevant facts (makes F2 cleanly detectable).
    return task.context


def dispatch(tool: str, argument: str, task) -> str:
    tool = tool.strip().lower()
    if tool == "lookup":
        return lookup(argument, task)
    if tool == "calculator":
        return calculator(argument)
    return f"ERROR: unknown tool '{tool}'"


AVAILABLE_TOOLS = ["lookup", "calculator"]
