"""Run registry: paths + manifests so every artifact is traceable to a checkpoint."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .schema import RunManifest

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
SUITE_DIR = ROOT / "data" / "suite"


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def run_dir(run_id: str, runs_root: Path | None = None) -> Path:
    d = (runs_root or RUNS) / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_manifest(m: RunManifest, runs_root: Path | None = None) -> Path:
    if m.git_sha is None:
        m.git_sha = git_sha()
    p = run_dir(m.run_id, runs_root) / "manifest.json"
    p.write_text(m.model_dump_json(indent=2))
    return p


def load_manifest(run_id: str, runs_root: Path | None = None) -> RunManifest:
    return RunManifest.model_validate_json((run_dir(run_id, runs_root) / "manifest.json").read_text())


def list_runs(runs_root: Path | None = None) -> list[RunManifest]:
    root = runs_root or RUNS
    return [RunManifest.model_validate_json(p.read_text()) for p in sorted(root.glob("*/manifest.json"))]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write((r.model_dump_json() if hasattr(r, "model_dump_json") else json.dumps(r)) + "\n")


def read_jsonl(path: Path, model=None):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(model.model_validate_json(line) if model else json.loads(line))
    return out
