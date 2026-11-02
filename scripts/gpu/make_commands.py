#!/usr/bin/env python
"""Expand configs/grid.yaml into one shell command line per checkpoint (train -> rollout -> grade). Pipe to `parallel`/Slurm array."""
import sys, yaml

g = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "configs/grid.yaml"))
mname = sys.argv[2] if len(sys.argv) > 2 else "small"
model = g["models"][mname]; k = g["rollout"]["k"]; mnt = g["rollout"]["max_new_tokens"]
tag = lambda *a: "-".join(str(x).replace("/", "_") for x in a)
print(f"mondecay rollout {tag(mname,'base')} {model} --backend vllm --k {k} --max-new-tokens {mnt} --device cuda && mondecay grade {tag(mname,'base')}")
for seed in g["seeds"]:
    for name, c in g["interventions"].items():
        for d in c["doses"]:
            rid = tag(mname, name, d, f"s{seed}")
            tr = {"sft": f"mondecay train sft {rid} {model} --steps {d} --seed {seed} --device cuda",
                  "distill": f"mondecay train distill {rid} {model} data/traces_{mname}.jsonl --ratio {d} --steps {c.get('steps',1000)} --seed {seed} --device cuda",
                  "rl": f"mondecay train rl {rid} {model} --mode {c.get('mode')} --dose {d} --steps {c.get('steps', d)} --seed {seed} --device cuda",
                  "quant": f"mondecay train quant {rid} {model} --bits {d}"}[c["kind"]]
            ad = "" if c["kind"] == "quant" else f"--adapter runs/{rid}/adapter"
            m = f"runs/{rid}/model" if c["kind"] == "quant" else model
            print(f"{tr} && mondecay rollout {rid} {m} {ad} --intervention {name} --dose {d} --seed {seed} --backend vllm --k {k} --max-new-tokens {mnt} --device cuda && mondecay grade {rid}")
