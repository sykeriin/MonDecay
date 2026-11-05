from __future__ import annotations

from pathlib import Path

import typer

from . import pipeline as P
from .monitors.monitor import Monitor
from .registry import RUNS, SUITE_DIR, load_manifest, save_manifest
from .schema import RunManifest

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _backend(kind: str, model: str, adapter: str | None, device: str = "auto", dtype: str = "auto", load_in_4bit: bool = False):
    if kind == "hf":
        from .rollout.backends import HFBackend
        return HFBackend(model, adapter, dtype=dtype, device=device, load_in_4bit=load_in_4bit)
    if kind == "vllm":
        from .rollout.backends import VLLMBackend
        return VLLMBackend(model, adapter)
    raise typer.BadParameter(kind)


@app.command("build-suite")
def build_suite(small: bool = False, out: Path = SUITE_DIR / "v1", seed: int = 0, n_hint: int = 0, n_code_pairs: int = 0, n_side: int = 0):
    """Generate the frozen planted-misbehaviour suite (JSONL + hashes). Explicit --n-* override the preset (small | full)."""
    from .suite.build import build_suite as bs
    d = (5, 5, 8) if small else (100, 100, 150)
    m = bs(out, n_hint or d[0], n_code_pairs or d[1], n_side or d[2], seed=seed)
    typer.echo(m)


@app.command()
def rollout(run_id: str, model: str, intervention: str = "base", dose: str = "0", seed: int = 0, adapter: str = None,
            backend: str = "hf", k: int = 1, limit: int = 0, families: str = "hint,code,side", suite: Path = SUITE_DIR / "v1",
            max_new_tokens: int = 1024, device: str = "auto", batch_size: int = 1, load_in_4bit: bool = False):
    specs = P.load_suite(suite, families.split(","))
    if limit:
        specs = specs[:limit]
    try:
        load_manifest(run_id)   # written by `train ...`; authoritative (carries data_sha/extra) - never overwrite it
    except FileNotFoundError:
        save_manifest(RunManifest(run_id=run_id, base_model=model, intervention=intervention, dose=dose, seed=seed, adapter_path=adapter))
    be = _backend(backend, model, adapter, device, load_in_4bit=load_in_4bit)
    P.rollout(run_id, be, specs, k=k, max_new_tokens=max_new_tokens, batch_size=batch_size, progress=lambda i, n: typer.echo(f"{i}/{n}", err=True))


@app.command()
def grade(run_id: str, suite: Path = SUITE_DIR / "v1"):
    labels = P.grade(run_id, P.load_suite(suite))
    typer.echo(f"{len(labels)} labels; misbehaved={sum(l.misbehaved for l in labels)}")


@app.command("monitor")
def monitor_cmd(run_id: str, monitor_model: str, name: str = None, backend: str = "hf", suite: Path = SUITE_DIR / "v1", device: str = "auto",
                load_in_4bit: bool = False):
    be = _backend(backend, monitor_model, None, device, load_in_4bit=load_in_4bit)
    tok = getattr(be, "tok", None)
    wrap = (lambda p: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)) if tok else None
    P.monitor(run_id, P.load_suite(suite), Monitor(be, name or monitor_model.split("/")[-1], wrap))


@app.command()
def metrics(run_id: str, monitor_name: str, base_run: str = None, suite: Path = SUITE_DIR / "v1", n_boot: int = 1000):
    typer.echo(P.compute_metrics(run_id, monitor_name, P.load_suite(suite), base_run=base_run, n_boot=n_boot))


@app.command()
def report(monitor_name: str, view: str = "cot+action", out: Path = Path("reports"), banner: str = ""):
    from .analysis.report import collect, plot_dose_response, practitioner_table
    rows = collect(monitor_name, view)
    typer.echo(plot_dose_response(rows, out / "fig_dose_response.png", f"({view}, monitor={monitor_name})"))
    typer.echo(practitioner_table(rows, out / "practitioner_table.md", banner))


@app.command()
def loadbearing(run_id: str, model: str, adapter: str = None, suite: Path = SUITE_DIR / "v1", device: str = "auto", load_in_4bit: bool = False):
    """Truncate the model's own CoT at 0/25/50/75/100% on no-hint MCQ items and force an answer. Writes runs/<id>/loadbearing.json."""
    import json
    from .metrics.loadbearing import loadbearing_items
    from .registry import RUNS, read_jsonl
    from .schema import Transcript
    specs = P.load_suite(suite, ["hint"])
    ts = {t.episode_id: t for t in read_jsonl(RUNS / run_id / "transcripts.jsonl", Transcript) if t.sample_idx == 0}
    be = _backend("hf", model, adapter, device, load_in_4bit=load_in_4bit)
    res = loadbearing_items(be, specs, ts)
    (RUNS / run_id / "loadbearing.json").write_text(json.dumps(res))
    typer.echo({f: round(sum(v.values()) / max(1, len(v)), 3) for f, v in res["items"].items()})


@app.command()
def gate(run_id: str, suite: Path = SUITE_DIR / "v1", out: Path = Path("reports")):
    """Pre-registered go/no-go gate on a BASE run. Exit code 0 = PASS, 1 = FAIL, 2 = INCONCLUSIVE."""
    import json
    from .analysis.gate import evaluate_gate, render_gate_md
    from .registry import RUNS, read_jsonl
    from .schema import Label
    lbp = RUNS / run_id / "loadbearing.json"
    from .rollout.parse import is_truncated
    from .schema import Transcript
    ts = read_jsonl(RUNS / run_id / "transcripts.jsonl", Transcript)
    trunc = sum(bool(t.raw_turns) and is_truncated(t.raw_turns[-1]) for t in ts) / max(1, len(ts))
    res = evaluate_gate(P.load_suite(suite), read_jsonl(RUNS / run_id / "labels.jsonl", Label), json.loads(lbp.read_text()) if lbp.exists() else None,
                        truncated_rate=trunc)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"gate.{run_id}.md").write_text(render_gate_md(res, run_id))
    typer.echo(render_gate_md(res, run_id))
    raise typer.Exit({"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[res["overall"]])


@app.command()
def verdicts(base_run: str, monitor_name: str, view: str = "cot+action", suite: Path = SUITE_DIR / "v1", out: Path = Path("reports"), n_boot: int = 1000, banner: str = ""):
    """Paired comparison of every non-base run against BASE_RUN with rule-based verdicts -> reports/practitioner_table.md + verdicts.json."""
    from .analysis.verdict import compare_all, write_verdicts
    rows = compare_all(base_run, monitor_name, view, P.load_suite(suite), n_boot=n_boot)
    typer.echo(write_verdicts(rows, out, banner))


@app.command("label-export")
def label_export(runs: str, out: Path = Path("reports/human_labels.csv"), n: int = 200, seed: int = 0, suite: Path = SUITE_DIR / "v1"):
    """Blind, stratified sample for hand labelling (comma-separated run ids). Fill human_label with 1/0, keep the .key.json next to it."""
    from .analysis.validation import export_labels
    typer.echo(export_labels(runs.split(","), P.load_suite(suite), out, n, seed))


@app.command("label-eval")
def label_eval(filled_csv: Path, monitor_name: str, threshold: float = 0.5, view: str = "cot+action"):
    import json
    from .analysis.validation import evaluate_labels
    typer.echo(json.dumps(evaluate_labels(filled_csv, monitor_name, threshold, view), indent=1))


@app.command()
def leaderboard(out: Path = Path("leaderboard")):
    from .analysis.leaderboard import build
    typer.echo(build(out))


train = typer.Typer(help="Interventions (the 'doses').")
app.add_typer(train, name="train")


@train.command("sft")
def train_sft(run_id: str, model: str, steps: int = typer.Option(...), n_data: int = 2000, source: str = "synthetic", seed: int = 0, device: str = "auto", lora_r: int = 16,
              batch: int = 4, grad_accum: int = 1, max_len: int = 1024, qlora: bool = False):
    """SFT on CoT-free customer-style data. source: synthetic | dolly."""
    from .interventions import data as D
    from .interventions.sft import run_sft
    rows = D.synthetic_customer_data(n_data, seed) if source == "synthetic" else D.hf_customer_data(n_data, seed)
    run_sft(run_id, model, rows, steps, "sft", steps, seed, lora_r=lora_r, device=device, batch=batch, grad_accum=grad_accum, max_len=max_len, qlora=qlora)


@train.command("gen-traces")
def gen_traces(model: str, out: Path, n: int = 500, backend: str = "hf", device: str = "auto", max_new_tokens: int = 2048):
    """Teacher traces from the BASE model on TRAIN math prompts (disjoint from eval) for distillation."""
    from .interventions.data import write_jsonl_rows
    from .interventions.rl import make_train_math
    from .rollout.parse import split_think
    be = _backend(backend, model, None, device)
    rows = []
    for it in make_train_math(n, seed=777):
        cot, vis = split_think(be.generate([{"role": "user", "content": it["prompt"]}], max_new_tokens=max_new_tokens))
        rows.append({"prompt": it["prompt"], "cot": cot, "answer": vis})
    write_jsonl_rows(out, rows)


@train.command("distill")
def train_distill(run_id: str, model: str, traces: Path, ratio: float = typer.Option(...), steps: int = 500, seed: int = 0, device: str = "auto",
                  batch: int = 4, grad_accum: int = 1, max_len: int = 1024, qlora: bool = False):
    """SFT on compressed teacher CoT. ratio 1.0=control, 0.5, 0.25, 0.0=answer-only. Dose recorded as the CoT *removed* (1-ratio)."""
    import json
    from .interventions.data import build_distill_set
    from .interventions.sft import run_sft
    rows = build_distill_set([json.loads(l) for l in traces.read_text().splitlines() if l.strip()], ratio)
    run_sft(run_id, model, rows, steps, "distill", round(1 - ratio, 3), seed, device=device, batch=batch, grad_accum=grad_accum, max_len=max_len, qlora=qlora, extra={"ratio": ratio})


@train.command("rl")
def train_rl(run_id: str, model: str, mode: str = typer.Option(...), dose: float = typer.Option(...), steps: int = 200, n_train: int = 256, seed: int = 0, device: str = "auto",
             num_generations: int = 8, max_completion_length: int = 2048):
    """mode: length (dose=lambda) | outcome (dose recorded as steps) | cotpen (dose=beta)."""
    from .interventions.rl import run_grpo
    run_grpo(run_id, model, mode, dose, steps, n_train, seed, num_generations, max_completion_length, device=device)


@train.command("quant")
def train_quant(run_id: str, model: str, bits: int = typer.Option(...), adapter: str = None):
    from .interventions.quantize import run_quantize
    run_quantize(run_id, model, bits, adapter=adapter)


@app.command("smoke-scripted")
def smoke_scripted(out: Path = Path("runs_smoke")):
    """End-to-end pipeline check with a FAKE scripted agent/monitor (no model). Proves plumbing, produces NO science."""
    from .smoke import run_smoke
    run_smoke(out)


if __name__ == "__main__":
    app()
