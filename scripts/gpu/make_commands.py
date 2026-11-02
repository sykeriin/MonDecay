#!/usr/bin/env python
"""Expand a grid profile into shell commands, one per line, in priority order.

usage: python scripts/gpu/make_commands.py [--profile laptop|kaggle-t4] [--grid configs/grid.yaml]
Each line is self-contained and resumable: train (if any) -> rollout -> grade -> monitor -> metrics.
Distillation needs data/traces_<profile>.jsonl, so a `gen-traces` line is emitted first when the profile uses it.
"""
import argparse
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--profile", default="kaggle-t4")
ap.add_argument("--grid", default="configs/grid.yaml")
a = ap.parse_args()
P = yaml.safe_load(open(a.grid))["profiles"][a.profile]
model, k, mnt, bs = P["model"], P["rollout"]["k"], P["rollout"]["max_new_tokens"], P["rollout"]["batch_size"]
mon = P["monitor"]
suite = f"data/suite/{a.profile}"
q4 = " --load-in-4bit" if mon.get("load_in_4bit") else ""
tag = lambda *x: "-".join(str(i).replace("/", "_") for i in x)
roll = lambda rid, m, adapter="": (f"mondecay rollout {rid} {m} {adapter} --suite {suite} --k {k} --max-new-tokens {mnt} --batch-size {bs}").replace("  ", " ")
post = lambda rid, base: (f"mondecay grade {rid} --suite {suite} && mondecay monitor {rid} {mon['model']} --name {mon['name']}{q4} --suite {suite} "
                          f"&& mondecay metrics {rid} {mon['name']} --suite {suite}" + (f" --base-run {base}" if base else ""))
s = P["suite"]
print(f"mondecay build-suite --out {suite} --n-hint {s['n_hint']} --n-code-pairs {s['n_code_pairs']} --n-side {s['n_side']}")
base = tag(a.profile, "base")
print(f"{roll(base, model)} && {post(base, '')}")
if any(c["kind"] == "distill" for c in P["interventions"].values()):
    print(f"mondecay train gen-traces {model} data/traces_{a.profile}.jsonl --n 300 --backend hf --max-new-tokens {mnt}")
for seed in P["seeds"]:
    for name, c in P["interventions"].items():
        for d in c["doses"]:
            rid = tag(a.profile, name, d, f"s{seed}")
            if c["kind"] == "sft":
                tr, ad, m = f"mondecay train sft {rid} {model} --steps {d} --seed {seed} {P.get('sft_extra', '')}", f"--adapter runs/{rid}/adapter", model
            elif c["kind"] == "distill":
                tr, ad, m = (f"mondecay train distill {rid} {model} data/traces_{a.profile}.jsonl --ratio {d} --steps {c['steps']} --seed {seed} {P.get('sft_extra', '')}",
                             f"--adapter runs/{rid}/adapter", model)
            elif c["kind"] == "rl":
                tr, ad, m = (f"mondecay train rl {rid} {model} --mode {c['mode']} --dose {d} --steps {c.get('steps', d)} --seed {seed}",
                             f"--adapter runs/{rid}/adapter", model)
            else:
                tr, ad, m = f"mondecay train quant {rid} {model} --bits {d}", "", f"runs/{rid}/model"
            print(f"{tr.strip()} && {roll(rid, m, ad)} && {post(rid, base)}")
