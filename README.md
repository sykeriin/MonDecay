# MonDecay

Open harness that measures how much chain-of-thought (CoT) monitorability a reasoning model loses under *ordinary* post-training
(SFT on CoT-free data, distillation to shorter CoT, length penalties, narrow outcome-RL, quantisation), as dose-response curves.

**Status: harness built and tested on CPU; no real experiment has been run.** See `docs/PLAN.md` (full plan), `docs/DECISIONS.md`
(choices + limitations), `docs/PREREG.md` (pre-registration template).

```bash
uv venv .venv && uv pip install -e ".[dev,cpu]"        # GPU box: ".[dev,gpu]"
.venv/bin/pytest -q
.venv/bin/mondecay smoke-scripted                       # fake-agent plumbing check, writes runs_smoke/
.venv/bin/mondecay build-suite --small
.venv/bin/mondecay rollout run0 Qwen/Qwen3-0.6B --families hint --limit 4 --max-new-tokens 160
.venv/bin/mondecay grade run0 && .venv/bin/mondecay monitor run0 Qwen/Qwen3-0.6B --name qwen06
python scripts/gpu/make_commands.py configs/grid.yaml small   # expands the grid into per-checkpoint shell lines
```

## Running on modest hardware
```bash
python scripts/gpu/make_commands.py --profile laptop     > laptop.txt     # RTX 3050 4 GB / WSL2: Qwen3-0.6B pilot, ~7 steps
python scripts/gpu/make_commands.py --profile kaggle-t4  > kaggle.txt     # free T4: R1-Distill-1.5B, positive control first
python scripts/kaggle/run_queue.py kaggle.txt --persist /kaggle/working/persist --budget-hours 8.5
```
Re-run the same command next session: finished steps are skipped, interrupted ones resume. CUDA/4-bit paths are untested until you run them;
see `docs/DECISIONS.md` ("Untested paths").

Pipeline: `build-suite -> [train ...] -> rollout -> grade -> monitor -> metrics -> report`; each stage reads/writes `runs/<run_id>/`.
Ground truth never reads the CoT (impossible tests, counterfactual hints, tool logs). Monitors are frozen: prompt hash stored in every score.
