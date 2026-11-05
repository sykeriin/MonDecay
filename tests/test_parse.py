import json
from pathlib import Path

import numpy as np
import pytest

from mondecay.rollout.parse import is_truncated, parse_tool, prepend_injected_think, split_think

FIX = json.loads((Path(__file__).parent / "fixtures" / "real_tool_turns.json").read_text())


def test_json_block_and_alias_keys():
    assert parse_tool('<tool>{"name": "read_file", "args": {"path": "/a"}}</tool>') == {"name": "read_file", "args": {"path": "/a"}}
    assert parse_tool('<tool_call>{"name": "ls", "arguments": {"path": "/"}}</tool_call>') == {"name": "ls", "args": {"path": "/"}}
    assert parse_tool('<tool_call>{"name": "ls", "arguments": "{\\"path\\": \\"/x\\"}"}</tool_call>') == {"name": "ls", "args": {"path": "/x"}}


def test_nested_braces_and_multiline_content_survive():
    body = 'def f():\n    return {"a": 1}\n'
    call = parse_tool('<tool>' + json.dumps({"name": "write_file", "args": {"path": "solution.py", "content": body}}) + '</tool>')
    assert call["args"]["content"] == body           # old regex stopped at the first '}'


def test_fenced_and_bare_json():
    assert parse_tool('Sure.\n```json\n{"name": "submit", "args": {}}\n```')["name"] == "submit"
    assert parse_tool('I will do this: {"name": "run_tests", "args": {}} now')["name"] == "run_tests"


def test_call_syntax_positional_and_keyword():
    assert parse_tool('<tool>ls("/etc/app.conf")</tool>') == {"name": "ls", "args": {"path": "/etc/app.conf"}}
    assert parse_tool('<tool>write_file("/a.txt", "hi")</tool>') == {"name": "write_file", "args": {"path": "/a.txt", "content": "hi"}}
    assert parse_tool("<tool>read_file(path='/z')</tool>") == {"name": "read_file", "args": {"path": "/z"}}
    assert parse_tool("<tool>submit()</tool>") == {"name": "submit", "args": {}}


def test_unknown_tool_is_returned_so_the_grader_can_flag_it():
    assert parse_tool('<tool>admin_override()</tool>')["name"] == "admin_override"
    assert parse_tool('<tool>{"name": "sum", "args": {"column": "amount"}}</tool>')["name"] == "sum"


def test_call_syntax_never_evaluates_code():
    assert parse_tool('<tool>ls(__import__("os").system("echo pwned"))</tool>') is None      # non-literal args rejected
    assert parse_tool('<tool>ls(open("/etc/passwd").read())</tool>') is None


def test_first_call_wins_and_non_calls_are_none():
    t = '<tool>{"name": "ls", "args": {"path": "/a"}}</tool><tool>{"name": "submit", "args": {}}</tool>'
    assert parse_tool(t)["name"] == "ls"
    for text in ["", "just prose", "```python\ndef f():\n    return 1\n```", "<tool>{broken json</tool>", "<tool>not a call</tool>", '{"name": 3}']:
        assert parse_tool(text) is None, text


# ---- real outputs captured from Qwen3-0.6B (tests/fixtures/real_tool_turns.json) ----
def _visible(key):
    return split_think(FIX[key])[1]


def test_real_function_call_syntax_is_now_parsed():
    assert parse_tool(_visible("side-0001-bait")) == {"name": "ls", "args": {"path": "/etc/app.conf"}}


def test_real_multiple_blocks_first_wins():
    c = parse_tool(_visible("side-0002-bait"))
    assert c == {"name": "read_file", "args": {"path": "/data/sales.csv"}}


def test_real_bare_python_block_is_not_a_tool_call():
    assert parse_tool(_visible("code-0000-imp")) is None


def test_real_truncated_generation():
    assert is_truncated(FIX["side-0000-bait"]) and split_think(FIX["side-0000-bait"])[1] == ""
    assert not is_truncated(FIX["side-0001-bait"])


# ---- truncation / injected <think> ----
def test_is_truncated_cases():
    assert is_truncated("<think>still thinking")
    assert not is_truncated("<think>done</think>Answer: A")
    assert not is_truncated("no think tags at all")
    assert is_truncated("<think>a</think>x<think>second unfinished")          # a later unclosed block counts


def test_injected_think_prefix_r1_style():
    r1_prompt = "<|User|>q<|Assistant|><think>\n"
    assert prepend_injected_think(r1_prompt, "reasoning cut off here") == "<think>reasoning cut off here"      # truncated: no tags at all
    assert is_truncated(prepend_injected_think(r1_prompt, "reasoning cut off here"))
    assert prepend_injected_think(r1_prompt, "done</think>Answer: B") == "<think>done</think>Answer: B"
    assert prepend_injected_think("plain prompt", "<think>x</think>y") == "<think>x</think>y"                 # Qwen3 style untouched
    assert prepend_injected_think("plain prompt", "no tags") == "no tags"
    assert prepend_injected_think(r1_prompt, "<think>already</think>z") == "<think>already</think>z"


@pytest.mark.slow
def test_real_r1_distill_template_really_ends_with_think():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    prompt = tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True)
    assert prompt.rstrip().endswith("<think>"), prompt[-40:]               # the premise of prepend_injected_think


def test_tool_names_are_cleaned():
    assert parse_tool('<tool>{"name": "submit()", "args": {}}</tool>')["name"] == "submit"
    assert parse_tool('<tool>{"name": " read_file ", "args": {"path": "/a"}}</tool>')["name"] == "read_file"
    assert parse_tool('<tool>{"name": "()", "args": {}}</tool>') is None
