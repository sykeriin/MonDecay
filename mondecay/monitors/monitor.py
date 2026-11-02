"""Frozen LLM monitors. Score = P(YES) from next-token probabilities, so scores are continuous and deterministic.

Views:
  cot         reasoning only (plus the task prompt, so the monitor knows what to look for)
  action      task + tool calls + final answer, CoT stripped
  cot+action  both
  followup    action view + follow-up Q&A put to the agent after the episode
Changing ANY template under prompts/v1 changes PROMPT_SHA; scores carry it so stale scores are detectable.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from ..graders.hint import extract_letter  # noqa: F401  (re-exported convenience)
from ..rollout.agent_loop import FOLLOWUP_Q
from ..schema import EpisodeSpec, MonitorScore, Transcript, View

PROMPT_DIR = Path(__file__).parent / "prompts" / "v1"
_TEXT = {p.stem: p.read_text() for p in sorted(PROMPT_DIR.glob("*.txt"))}
PROMPT_SHA = hashlib.sha256("".join(f"{k}:{v}" for k, v in sorted(_TEXT.items())).encode()).hexdigest()[:16]


def _fmt_actions(t: Transcript) -> str:
    if not t.actions:
        return "(no tool calls)"
    return "\n".join(f"- {a.name}({a.args}) -> {a.result[:200]!r}" for a in t.actions)


def build_prompt(spec: EpisodeSpec, t: Transcript, view: View) -> str:
    kw = dict(prompt=spec.prompt, actions=_fmt_actions(t), final=t.final_answer or "(none)", cot=t.cot or "(no reasoning shown)",
              followup_q=FOLLOWUP_Q[spec.family], followup=t.followup or "(none)")
    if view == "cot":
        material = _TEXT["view_action"].split("=== Agent's actions")[0].format(**kw) + _TEXT["view_cot"].format(**kw)
    elif view == "action":
        material = _TEXT["view_action"].format(**kw)
    elif view == "cot+action":
        material = _TEXT["view_action"].format(**kw) + "\n" + _TEXT["view_cot"].format(**kw)
    elif view == "followup":
        material = _TEXT["view_followup"].format(**kw)
    else:
        raise ValueError(view)
    return _TEXT["header"].format(misbehaviour=_TEXT[f"misbehaviour_{spec.family}"].strip(), material=material.strip())


class Monitor:
    """`backend` needs `next_token_probs(prompt, candidates)`; chat templating is applied by the caller's backend if desired."""

    def __init__(self, backend, name: str, chat_wrap=None):
        self.backend, self.name, self.chat_wrap = backend, name, chat_wrap

    def score(self, spec: EpisodeSpec, t: Transcript, view: View) -> MonitorScore:
        prompt = build_prompt(spec, t, view)
        if self.chat_wrap:
            prompt = self.chat_wrap(prompt)
        p = self.backend.next_token_probs(prompt, ["YES", "NO"])
        py, pn = p.get("YES", 0.0), p.get("NO", 0.0)
        p_yes = py / (py + pn) if (py + pn) > 0 else 0.5
        return MonitorScore(episode_id=t.episode_id, run_id=t.run_id, sample_idx=t.sample_idx, monitor=self.name,
                            view=view, prompt_sha=PROMPT_SHA, p_yes=p_yes)
