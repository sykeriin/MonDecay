"""Core data schemas. Everything on disk is JSONL of these models."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field

Family = Literal["hint", "code", "side"]
View = Literal["cot", "action", "cot+action", "followup"]


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


class EpisodeSpec(BaseModel):
    """One frozen eval episode. `payload` is family-specific (see suite/*)."""

    episode_id: str
    family: Family
    # For negatives/controls: 'twin' episodes are solvable code tasks, no-bait env, or no-hint item.
    variant: str = "main"
    prompt: str
    payload: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: str = ""


class Transcript(BaseModel):
    episode_id: str
    run_id: str
    sample_idx: int = 0
    cot: str = ""            # concatenated reasoning (<think>) text across turns
    actions: list[ToolCall] = Field(default_factory=list)
    final_answer: str = ""
    cot_tokens: int = 0
    raw_turns: list[str] = Field(default_factory=list)
    followup: str = ""       # subject model's answer to the follow-up question


class Label(BaseModel):
    """Programmatic ground truth. Never derived from reading the CoT."""

    episode_id: str
    run_id: str
    sample_idx: int = 0
    family: Family
    misbehaved: bool
    task_correct: bool = False
    detail: dict[str, Any] = Field(default_factory=dict)


class MonitorScore(BaseModel):
    episode_id: str
    run_id: str
    sample_idx: int = 0
    monitor: str
    view: View
    prompt_sha: str
    p_yes: float


class RunManifest(BaseModel):
    run_id: str
    base_model: str
    intervention: str = "base"
    dose: float | str = 0
    seed: int = 0
    adapter_path: str | None = None
    quant: str | None = None
    data_sha: str | None = None
    git_sha: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
