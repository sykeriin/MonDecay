from __future__ import annotations

import json
import re

THINK_RE = re.compile(r"<think>(.*?)(?:</think>|$)", re.S)
TOOL_RE = re.compile(r"<tool>\s*(\{.*?\})\s*</tool>", re.S)

TOOL_INSTRUCTIONS = (
    "To call a tool, emit exactly one block per turn: <tool>{\"name\": \"<tool_name>\", \"args\": {...}}</tool>. "
    "You will receive the result in the next message."
)


def split_think(text: str) -> tuple[str, str]:
    """Return (cot, visible). Handles Qwen3/R1 style <think> blocks, including truncated (unclosed) ones."""
    cot = "\n".join(m.strip() for m in THINK_RE.findall(text))
    visible = THINK_RE.sub("", text).replace("</think>", "").strip()
    return cot, visible


def parse_tool(visible: str) -> dict | None:
    m = TOOL_RE.search(visible)
    if not m:
        return None
    try:
        d = json.loads(m.group(1))
        if isinstance(d, dict) and "name" in d:
            d.setdefault("args", {})
            return d
    except json.JSONDecodeError:
        return None
    return None


def count_tokens_approx(s: str) -> int:
    return max(0, int(len(s.split()) * 1.3))
