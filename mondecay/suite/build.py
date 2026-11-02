from __future__ import annotations

from pathlib import Path

from ..registry import SUITE_DIR, write_jsonl
from ..schema import EpisodeSpec, sha256_json
from . import code_impossible, hint, side_objective


def build_suite(out_dir: Path | None = None, n_hint=100, n_code_pairs=100, n_side=150, seed=0) -> dict:
    """Defaults give 200 hint + 200 code + 187 side episodes. `--small` in the CLI shrinks these for smoke tests."""
    out_dir = out_dir or SUITE_DIR / "v1"
    fam: dict[str, list[EpisodeSpec]] = {
        "hint": hint.build(n_hint, seed),
        "code": code_impossible.build(n_code_pairs, seed),
        "side": side_objective.build(n_side, seed),
    }
    manifest = {}
    for name, eps in fam.items():
        write_jsonl(out_dir / f"{name}.jsonl", eps)
        manifest[name] = {"n": len(eps), "sha256": sha256_json([e.model_dump() for e in eps])}
    write_jsonl(out_dir / "MANIFEST.jsonl", [manifest])
    return manifest
