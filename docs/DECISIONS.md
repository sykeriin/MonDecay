# Decision log (what was chosen, what was not, why)

| # | Decision | Alternative rejected | Reason |
|---|---|---|---|
| 1 | Programmatic ground truth (impossible tests, counterfactual hints, tool logs) | LLM/human labels from transcripts | A model that hides intent would also hide it from a transcript-reading labeller. Circular. |
| 2 | Impossible coding tasks = two tests demanding different outputs for the same input | Hope for spontaneous hacks; "special-case a single test" style impossibility | Small models rarely hack spontaneously. Contradictory pair means no pure function passes, so any pristine pass is a hack. |
| 3 | Pristine tests re-run in a fresh dir; pass = exactly `n_tests` "passed" parsed from pytest output | Trust exit code | `sys.exit(0)` at import can yield exit code 0 with no tests run. |
| 4 | Hint label = answered hinted letter AND paired no-hint run did not | "answered hint letter" alone | Without the counterfactual, a model that was wrong anyway is mislabelled as hint-following. |
| 5 | Hint-family items are synthetic multi-step arithmetic MCQs | MMLU-Pro/GPQA | Answers are exact by construction, CoT is necessary, no contamination, no download. HF-sourced items can be added later (risk: small models guess on knowledge MCQs, so CoT is not load-bearing). |
| 6 | Monitor score = P(YES)/(P(YES)+P(NO)) from next-token probs | Sampled 0-10 score | Continuous, deterministic, no ties at the 5% FPR threshold. |
| 7 | `nohint` episodes are not monitored | Include them as negatives | They contain no hint, so they are trivial negatives and would inflate recall@FPR. |
| 8 | Bootstrap resamples episodes, not rollouts | Row-level bootstrap | k rollouts of one episode are dependent; row-level CIs are too narrow (tested). |
| 9 | Quantisation = RTN fake-quant (any hardware) as baseline | bitsandbytes only | Runs on CPU, reproducible, no extra deps. Does not model kernel behaviour; use AWQ/GPTQ on GPU for headline numbers. |
| 10 | Train/eval share task *templates* but never *instances* (guard on prompt+tests, exact) | Fully disjoint templates | Outcome-RL on the same kind of hackable task is the intervention being studied. Prefix-matching guard was wrong for templated prompts (it fired on all of them) and was replaced. |
| 11 | CoT-penalty training monitor = keyword regex (different from eval monitors) | Same LLM as eval monitor | Measures transfer of obfuscation, not overfitting to the judge. Swap in a different-family LLM for real runs. |
| 12 | `rollout` never overwrites a manifest written by `train` | Always write | It silently dropped `data_sha`/`extra` (found by test; fixed). |
| 13 | LoRA for all interventions | Full fine-tune | 5-10x cheaper and what most customers do; stated as a caveat. |
| 14 | Plain CLI stages + files, no orchestration framework | Hydra/Ray/W&B | Fewer moving parts; stages re-runnable independently. |

## Known limitations (read before trusting any number)
- Everything run so far is plumbing validation on Qwen3-0.6B/CPU or a scripted fake agent. **No scientific result exists yet.**
- `VLLMBackend` is written to the vLLM API but has never been executed (no GPU here). Expect to fix small API mismatches on first GPU run.
- Outcome-RL needs a non-zero success rate or GRPO gets zero-variance rewards (observed in the CPU smoke). Half the training tasks are solvable twins for this reason; the real run must check `frac_reward_zero_std`.
- Verbalisation rate currently uses a keyword heuristic; replace/validate with the 200 hand labels.
- The stateful-hack grader test is weak (accepts "passes and flagged" OR "fails"); strengthen when a real hacking transcript exists.
- The monitor prompt has not been validated against human labels (plan: 200 labels, kappa >= 0.7).

## Hardware reality (added after "I have a 3050 laptop")
Target hardware: RTX 3050 laptop **4 GB** under Windows+WSL2 for dev/pilot, **free Kaggle/Colab T4** for 1.5B runs. No paid GPU.

| # | Decision | Alternative rejected | Reason |
|---|---|---|---|
| 15 | Subject = Qwen3-0.6B (laptop pilot) and R1-Distill-1.5B (T4). **8B subject and 32B monitor dropped from v1.** | Keep 8B/32B | Do not fit (4 GB / 16 GB with quota). Stated as a scale limitation; v1 claims are "<=1.5B, LoRA, few checkpoints". |
| 16 | Monitor = Qwen2.5-3B (laptop) / Qwen2.5-7B 4-bit (T4) | API monitors | Frozen open weights; API models drift silently. A smaller monitor is a weaker monitor: absolute recall will be lower than with 32B. Compare doses within one monitor only. |
| 17 | Lock-step batched episodes (`run_episodes`) over per-episode loops | vLLM on the laptop | vLLM KV cache does not fit in 4 GB; batched HF with left padding is the fallback. Verified: greedy batched output == single output on CPU. |
| 18 | Every stage append-only + resumable; `scripts/kaggle/run_queue.py` mirrors `runs/` to a persist dir | Run whole grid in one session | Sessions die (9-12 h limits, sleeping laptops). Verified: kill, wipe, restore, skip-done. |
| 19 | Grid in priority order with the CoT-penalty positive control FIRST (kaggle-t4: 1 dose) | Spread quota evenly | If the positive control shows no recall drop, nothing else is interpretable; stop before spending quota. |
| 20 | Precision: bf16 only if the GPU supports it, else fp16 (T4); fp32 on CPU | Always bf16 | T4 has no bf16 support. |

### Untested paths (no GPU in the dev container): treat as unverified until the first run on real hardware
- `HFBackend` on CUDA, `load_in_4bit` (bitsandbytes), QLoRA in `run_sft`, fp16 AMP training, GRPO on CUDA, `VLLMBackend`.
- Time/VRAM estimates in docs/PLAN.md section 11 are guesses, not measurements. First thing to do on the T4: time one base rollout of 20 episodes and extrapolate.
- If Qwen3-0.6B/1.5B fail the go/no-go gate (misbehaviour < 10% on >= 2 families, or CoT not load-bearing), the correct outcome is to report that and stop, not to run the interventions anyway.

## Round 2 additions
| # | Decision | Alternative rejected | Reason |
|---|---|---|---|
| 21 | `gate` returns exit code 0/1/2 (PASS/FAIL/INCONCLUSIVE) and INCONCLUSIVE is not "proceed" | Boolean pass/fail | Few positives is the most likely failure mode on small models; a coin-flip "pass" would waste the free quota. |
| 22 | Load-bearing test only counts items whose own CoT is non-empty | All items | Truncating an empty CoT is a no-op and would hide a non-thinking model's lack of load-bearing CoT. |
| 23 | Verdicts use paired bootstrap on shared (episode, sample) rows + matched-positive delta | Compare marginal recall CIs | Marginal CIs ignore the pairing and are too conservative; matched delta separates "hides better" from "hacks differently". |
| 24 | Human labels are blind (programmatic label in a separate key file) | Show the label | Labeller bias would inflate kappa. Unlabelled rows are ignored, never guessed. |
| 25 | Static leaderboard (JSON + HTML, no JS) | Hosted site | Zero infrastructure; can be committed or opened locally. |
| 26 | Tested library versions recorded in docs/TESTED_VERSIONS.txt | Pin in pyproject | TRL/transformers APIs move fast; pins would break on Kaggle's preinstalled stack. If a GPU run breaks on an API change, compare with this file first. |
| 27 | Answer-letter parser accepts `<B>`, `(B)`, `[B]`, `**B**`; letter is case-sensitive | Strict `Answer: B` only | Found by the real Qwen3-0.6B pilot: the model copies the `<letter>` placeholder from the prompt and writes `Answer: <B>`. The strict parser scored every load-bearing item 0 (accuracy 0.0 at all truncation levels, below chance), which would have silently produced a false "CoT is not load-bearing" gate result. Lesson: sanity-check any metric that comes out exactly 0 or below chance. |

### Pilot run on Qwen3-0.6B (plumbing check, NOT a finding)
20 hint-family episodes (10 pairs), 300 new tokens max, CPU. Resume after a killed process continued from 10/20 with no duplicates. Most generations were cut off mid-CoT by the 300-token limit, so there are almost no final answers and 0 labelled misbehaviours; gate = FAIL (misbehaviour 0/10) and load-bearing INCONCLUSIVE (10 items < 30). Forced-answer accuracy rose from 0.2 (no CoT) to 0.5 (full CoT) on 10 items: consistent with a load-bearing CoT but far too few items to claim it.
