"""Every generated command must parse against the real CLI (catches renamed/missing options before a 9-hour GPU session does)."""
import re
import subprocess
import sys
from pathlib import Path

import pytest
import typer.main

ROOT = Path(__file__).resolve().parent.parent


def commands(profile):
    out = subprocess.check_output([sys.executable, str(ROOT / "scripts/gpu/make_commands.py"), "--profile", profile,
                                   "--grid", str(ROOT / "configs/grid.yaml")], text=True, cwd=ROOT)
    return [c.strip() for c in out.splitlines() if c.strip()]


@pytest.mark.parametrize("profile", ["laptop", "kaggle-t4"])
def test_all_generated_commands_parse(profile):
    from mondecay.cli import app
    root = typer.main.get_command(app)
    cmds = commands(profile)
    assert len(cmds) >= 4
    for line in cmds:
        for part in line.split("&&"):
            toks = part.split()
            assert toks[0] == "mondecay", part
            cmd, args = root, toks[1:]
            while hasattr(cmd, "commands") and args and args[0] in cmd.commands:   # descend into `train <sub>`
                cmd, args = cmd.commands[args[0]], args[1:]
            ctx = cmd.make_context(cmd.name, list(args), resilient_parsing=False)   # raises on unknown/missing options
            assert ctx is not None


def test_priority_order_puts_positive_control_first():
    cmds = commands("kaggle-t4")
    first_train = next(c for c in cmds if "mondecay train" in c and "gen-traces" not in c)
    assert "--mode cotpen" in first_train


def test_queue_dry_run_and_skip(tmp_path):
    f = tmp_path / "c.txt"; f.write_text("echo hi\n# c\necho there\n")
    r = subprocess.run([sys.executable, str(ROOT / "scripts/kaggle/run_queue.py"), str(f), "--persist", str(tmp_path / "p"), "--dry-run"],
                       capture_output=True, text=True, cwd=tmp_path)
    assert r.stdout.count("[run") == 2
