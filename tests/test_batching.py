import pytest

from mondecay.rollout.agent_loop import run_episode, run_episodes
from mondecay.rollout.backends import ScriptedBackend
from mondecay.suite.build import build_suite
from mondecay import pipeline as P


def test_lockstep_batch_equals_sequential(tmp_path):
    build_suite(tmp_path / "s", 3, 3, 4)
    specs = P.load_suite(tmp_path / "s")
    def agent(messages):
        user = " ".join(m["content"] for m in messages if m["role"] == "user")
        turns = sum(1 for m in messages if m["role"] == "assistant")
        if "Answer: <letter>" in user:
            return "<think>c</think>Answer: A"
        if turns == 0:
            return '<think>go</think><tool>{"name": "ls", "args": {"path": "/"}}</tool>'
        return '<tool>{"name": "submit", "args": {}}</tool>'
    be = ScriptedBackend(agent)
    items = [(s, 0) for s in specs]
    batched = run_episodes(items, be, "r", ask_followup=True)
    seq = [run_episode(s, be, "r", 0) for s, _ in items]
    assert [t.model_dump() for t, _ in batched] == [t.model_dump() for t, _ in seq]
    assert [w for _, w in batched] == [w for _, w in seq]


def test_pipeline_batch_size_gives_same_transcripts(tmp_path):
    build_suite(tmp_path / "s", 3, 2, 3)
    specs = P.load_suite(tmp_path / "s", ["hint"])
    be = ScriptedBackend(lambda m: "<think>c</think>Answer: B")
    P.rollout("a", be, specs, k=2, runs_root=tmp_path, ask_followup=False, batch_size=1)
    P.rollout("b", be, specs, k=2, runs_root=tmp_path, ask_followup=False, batch_size=5)
    strip = lambda p: [l.replace('"run_id":"a"', "").replace('"run_id":"b"', "") for l in (tmp_path / p / "transcripts.jsonl").read_text().splitlines()]
    assert strip("a") == strip("b")


@pytest.mark.slow
def test_hf_batched_greedy_matches_single():
    torch = pytest.importorskip("torch")
    from mondecay.rollout.backends import HFBackend
    be = HFBackend("Qwen/Qwen3-0.6B", device="cpu", enable_thinking=False)
    ps = [[{"role": "user", "content": q}] for q in ["Say hello in French.", "What is 2+2? Answer briefly.", "Name a colour."]]
    single = [be.generate(p, max_new_tokens=12, temperature=0.0) for p in ps]
    batched = be.generate_batch(ps, max_new_tokens=12, temperature=0.0, micro_batch=3)
    assert single == batched
