from __future__ import annotations

import ast
import json
import re

THINK_RE = re.compile(r"<think>(.*?)(?:</think>|$)", re.S)
BLOCK_RE = re.compile(r"<(tool|tool_call)>\s*(.*?)\s*</\1>", re.S)
FENCE_RE = re.compile(r"```(?:json|tool_call|tool)?\s*(\{.*?\})\s*```", re.S)

# Positional-argument names for function-call syntax, e.g. write_file("/a", "text") -> {"path": "/a", "content": "text"}.
KNOWN_TOOLS: dict[str, tuple[str, ...]] = {"ls": ("path",), "read_file": ("path",), "write_file": ("path", "content"), "run_tests": (), "submit": ()}

TOOL_INSTRUCTIONS = (
    "To call a tool, emit exactly ONE block per turn, as JSON: <tool>{\"name\": \"<tool_name>\", \"args\": {...}}</tool>. "
    "For example: <tool>{\"name\": \"read_file\", \"args\": {\"path\": \"/data/app.log\"}}</tool>. "
    "Wait for the result in the next message before calling another tool. Call submit when you are done."
)


def split_think(text: str) -> tuple[str, str]:
    """Return (cot, visible). Handles Qwen3/R1 style <think> blocks, including truncated (unclosed) ones."""
    cot = "\n".join(m.strip() for m in THINK_RE.findall(text))
    visible = THINK_RE.sub("", text).replace("</think>", "").strip()
    return cot, visible


def is_truncated(raw_turn: str) -> bool:
    """True if generation stopped inside an unclosed <think> block (hit max_new_tokens before finishing its reasoning)."""
    return "<think>" in raw_turn and "</think>" not in raw_turn.split("<think>")[-1]


def prepend_injected_think(prompt: str, text: str) -> str:
    """R1-Distill chat templates END the prompt with '<think>' so the generation has no opening tag at all (and a truncated one has no
    tags whatsoever). Re-add it so parsing is uniform; otherwise unfinished reasoning would be read as a visible final answer."""
    if prompt.rstrip().endswith("<think>") and not text.lstrip().startswith("<think>"):
        return "<think>" + text
    if "</think>" in text and "<think>" not in text:
        return "<think>" + text
    return text


def _clean_name(name: str) -> str:
    """`submit()` / ` read_file ` -> `submit` / `read_file` (models often write the name in call syntax)."""
    return name.strip().rstrip("()").strip()


def _normalize(d) -> dict | None:
    if not isinstance(d, dict) or not isinstance(d.get("name"), str) or not _clean_name(d["name"]):
        return None
    args = d.get("args", d.get("arguments", d.get("parameters", {})))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return {"name": _clean_name(d["name"]), "args": args if isinstance(args, dict) else {}}


def _from_call_syntax(body: str) -> dict | None:
    """`ls("/etc")`, `write_file("/a", "x")`, `read_file(path="/a")`: first call in the body. Literal arguments only (never evaluated)."""
    for line in body.strip().splitlines():
        try:
            node = ast.parse(line.strip(), mode="eval").body
        except SyntaxError:
            continue
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        try:
            pos = [ast.literal_eval(a) for a in node.args]
            kw = {k.arg: ast.literal_eval(k.value) for k in node.keywords if k.arg}
        except (ValueError, SyntaxError):
            continue
        names = KNOWN_TOOLS.get(node.func.id, ())
        args = {names[i] if i < len(names) else f"arg{i}": v for i, v in enumerate(pos)}
        args.update(kw)
        return {"name": node.func.id, "args": args}
    return None


def _parse_body(body: str) -> dict | None:
    try:
        return _normalize(json.loads(body))
    except json.JSONDecodeError:
        return _from_call_syntax(body)


def parse_tool(visible: str) -> dict | None:
    """First tool call in a turn, or None. Accepted: <tool>/<tool_call> blocks holding JSON ('args' or 'arguments') or call syntax,
    fenced ```json blocks, and a bare JSON object. Only the FIRST call is executed (one call per turn, as instructed); later ones are
    ignored. A python code block with no call is NOT a tool call (it is recorded as non-use via tool_use_rate)."""
    for m in BLOCK_RE.finditer(visible):
        c = _parse_body(m.group(2))
        if c:
            return c
    for m in FENCE_RE.finditer(visible):
        c = _parse_body(m.group(1))
        if c:
            return c
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", visible):
        try:
            obj, _ = dec.raw_decode(visible[m.start():])
        except json.JSONDecodeError:
            continue
        c = _normalize(obj)
        if c:
            return c
    return None


def count_tokens_approx(s: str) -> int:
    return max(0, int(len(s.split()) * 1.3))
