#!/usr/bin/env python
"""Run a commands file (from make_commands.py) inside a time-limited session (Kaggle/Colab), safely across restarts.

  python scripts/kaggle/run_queue.py commands.txt --persist /kaggle/working/persist --budget-hours 8.5

* A command is skipped if its hash is in <persist>/done.json (finished in an earlier session).
* runs/ and data/ are mirrored to <persist> after every command, and restored from it at start, so a killed session loses at most
  the command in flight (and rollout/monitor inside it are themselves resumable, so rerunning it continues rather than restarts).
* Stops cleanly before starting a command if less than --reserve-hours of the budget remain.
* Persisting across sessions is up to the platform: on Kaggle, save the version so /kaggle/working/persist becomes a Dataset and attach
  it as input next session (pass --persist <that input path>/persist to restore); on Colab, point --persist at a mounted Drive folder.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("commands")
ap.add_argument("--persist", required=True)
ap.add_argument("--budget-hours", type=float, default=8.5)
ap.add_argument("--reserve-hours", type=float, default=0.5)
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

persist = Path(a.persist); persist.mkdir(parents=True, exist_ok=True)
done_f = persist / "done.json"
done = set(json.loads(done_f.read_text())) if done_f.exists() else set()
t0 = time.time()
MIRROR = ["runs", "data"]

def mirror(src_to_persist: bool):
    for d in MIRROR:
        src, dst = (Path(d), persist / d) if src_to_persist else (persist / d, Path(d))
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)

if not a.dry_run:
    mirror(False)                       # restore previous sessions' work
for line in Path(a.commands).read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    h = hashlib.sha256(line.encode()).hexdigest()[:16]
    if h in done:
        print(f"[skip done] {line[:100]}"); continue
    left = a.budget_hours - (time.time() - t0) / 3600
    if left < a.reserve_hours:
        print(f"[stop] {left:.2f} h left < reserve {a.reserve_hours} h; rerun next session"); break
    print(f"[run {left:.2f} h left] {line[:140]}", flush=True)
    if a.dry_run:
        continue
    rc = subprocess.call(line, shell=True)
    mirror(True)
    if rc == 0:
        done.add(h); done_f.write_text(json.dumps(sorted(done)))
    else:
        print(f"[fail rc={rc}] left in queue; later lines still run", file=sys.stderr)
