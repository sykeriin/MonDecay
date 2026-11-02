"""Sandboxed subprocess execution for model-written code.

Not a security boundary against a determined adversary: run only inside a throwaway container with no secrets.
Mitigations: tmpdir cwd, scrubbed env, CPU/memory/file rlimits, wall-clock timeout, and (when available)
`unshare -n` to drop network access.
"""
from __future__ import annotations

import os
import resource
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


def _limits(cpu_s: int, mem_mb: int):
    def fn():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024, mem_mb * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        os.setsid()
    return fn


_UNSHARE_OK: bool | None = None


def _can_unshare() -> bool:
    global _UNSHARE_OK
    if _UNSHARE_OK is None:
        if not shutil.which("unshare"):
            _UNSHARE_OK = False
        else:
            try:
                r = subprocess.run(["unshare", "-n", "true"], capture_output=True, timeout=5)
                _UNSHARE_OK = r.returncode == 0
            except Exception:
                _UNSHARE_OK = False
    return _UNSHARE_OK


def run_in_dir(files: dict[str, str], argv: list[str], timeout: float = 20.0, mem_mb: int = 1024) -> ExecResult:
    """Materialise `files` in a fresh tmpdir and run argv there. `python` in argv is replaced by sys.executable."""
    with tempfile.TemporaryDirectory(prefix="mdsbx_") as d:
        root = Path(d)
        for rel, content in files.items():
            p = (root / rel).resolve()
            if root.resolve() not in p.parents and p != root.resolve():
                raise ValueError(f"path escapes sandbox: {rel}")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        argv = [sys.executable if a == "python" else a for a in argv]
        if _can_unshare():
            argv = ["unshare", "-n", *argv]
        env = {"PATH": os.environ.get("PATH", ""), "HOME": d, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"}
        try:
            r = subprocess.run(argv, cwd=d, env=env, capture_output=True, text=True, timeout=timeout,
                               preexec_fn=_limits(int(timeout) + 2, mem_mb))
            return ExecResult(r.returncode, r.stdout[-8000:], r.stderr[-8000:])
        except subprocess.TimeoutExpired as e:
            return ExecResult(-9, (e.stdout or b"").decode(errors="ignore")[-4000:] if isinstance(e.stdout, bytes) else (e.stdout or ""), "timeout", True)
