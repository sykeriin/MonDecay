import json

import pytest

from mondecay import pipeline as P
from mondecay.monitors.monitor import Monitor
from mondecay.registry import read_jsonl
from mondecay.rollout.backends import ScriptedBackend
from mondecay.schema import Transcript
from mondecay.suite.build import build_suite


class Boom(Exception):
    pass


def agent(counter, crash_after=None):
    def fn(messages):
        counter["n"] += 1
        if crash_after is not None and counter["n"] > crash_after:
            raise Boom
        return "<think>t</think>Answer: A"
    return fn


@pytest.fixture
def specs(tmp_path):
    build_suite(tmp_path / "s", 4, 2, 3)
    return P.load_suite(tmp_path / "s", ["hint"])


def test_rollout_crash_then_resume_has_no_dupes_or_gaps(tmp_path, specs):
    c1 = {"n": 0}
    with pytest.raises(Boom):
        P.rollout("r", ScriptedBackend(agent(c1, crash_after=5)), specs, k=2, runs_root=tmp_path, ask_followup=False)
    part = read_jsonl(tmp_path / "r" / "transcripts.jsonl", Transcript)
    assert 0 < len(part) < len(specs) * 2
    c2 = {"n": 0}
    P.rollout("r", ScriptedBackend(agent(c2)), specs, k=2, runs_root=tmp_path, ask_followup=False)
    full = read_jsonl(tmp_path / "r" / "transcripts.jsonl", Transcript)
    keys = [(t.episode_id, t.sample_idx) for t in full]
    assert len(keys) == len(set(keys)) == len(specs) * 2
    assert c2["n"] == len(specs) * 2 - len(part)          # only the missing work was redone


def test_torn_last_line_is_repaired(tmp_path, specs):
    P.rollout("r", ScriptedBackend(agent({"n": 0})), specs[:2], k=1, runs_root=tmp_path, ask_followup=False)
    p = tmp_path / "r" / "transcripts.jsonl"
    p.write_text(p.read_text() + '{"episode_id": "torn", "sam')      # process killed mid-write
    P.rollout("r", ScriptedBackend(agent({"n": 0})), specs[:3], k=1, runs_root=tmp_path, ask_followup=False)
    ts = read_jsonl(p, Transcript)
    assert [t.episode_id for t in ts] == [s.episode_id for s in specs[:3]]


def test_monitor_resume_and_stale_prompt_ignored(tmp_path, specs):
    P.rollout("r", ScriptedBackend(agent({"n": 0})), specs, k=1, runs_root=tmp_path, ask_followup=False)
    calls = {"n": 0}

    class B:
        def next_token_probs(self, prompt, cands):
            calls["n"] += 1
            return {"YES": 0.3, "NO": 0.7}
    mon = Monitor(B(), "m")
    P.monitor("r", specs, mon, views=["cot"], runs_root=tmp_path)
    first = calls["n"]
    P.monitor("r", specs, mon, views=["cot"], runs_root=tmp_path)
    assert calls["n"] == first                                       # nothing recomputed
    # inject a stale-prompt row: it must not be returned or counted
    p = tmp_path / "r" / "scores.m.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    rows[0]["prompt_sha"] = "old"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = P.monitor("r", specs, mon, views=["cot"], runs_root=tmp_path)
    assert calls["n"] == first + 1 and all(x.prompt_sha != "old" for x in out)
