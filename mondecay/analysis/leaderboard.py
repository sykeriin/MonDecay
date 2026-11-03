"""Static leaderboard: leaderboard/data.json + a self-contained index.html (no JS, no external assets)."""
from __future__ import annotations

import html
import json
from pathlib import Path

from ..registry import RUNS, list_runs

VIEW = "cot+action"


def collect_rows(runs_root: Path | None = None) -> list[dict]:
    root = runs_root or RUNS
    rows = []
    for m in list_runs(root):
        for p in sorted((root / m.run_id).glob("metrics.*.json")):
            met = json.loads(p.read_text())
            for fam, fr in met["families"].items():
                v = fr["views"].get(VIEW)
                if not v:
                    continue
                rows.append({"run_id": m.run_id, "base_model": m.base_model, "intervention": m.intervention, "dose": str(m.dose), "seed": m.seed,
                             "monitor": met["monitor"], "family": fam, "n": fr["n"], "n_pos": fr["n_pos"], "recall": v["recall"],
                             "recall_lo": v["recall_ci"][0], "recall_hi": v["recall_ci"][1], "auroc": v["auroc"],
                             "misbehaviour": fr["misbehaviour_rate"], "capability": fr["capability"], "cot_tokens": fr["cot_tokens_mean"]})
    return rows


def _f(x) -> str:
    return "n/a" if x != x else f"{x:.2f}"


def render_html(rows: list[dict], title: str = "MonDecay leaderboard") -> str:
    head = ["run", "model", "intervention", "dose", "seed", "monitor", "family", "n_pos/n", "recall@5%FPR [95% CI]", "AUROC", "misbehaviour", "capability", "CoT tokens"]
    body = []
    for r in sorted(rows, key=lambda r: (r["family"], r["intervention"], r["dose"], r["run_id"])):
        cells = [r["run_id"], r["base_model"], r["intervention"], r["dose"], r["seed"], r["monitor"], r["family"], f"{r['n_pos']}/{r['n']}",
                 f"{_f(r['recall'])} [{_f(r['recall_lo'])}, {_f(r['recall_hi'])}]", _f(r["auroc"]), _f(r["misbehaviour"]), _f(r["capability"]), f"{r['cot_tokens']:.0f}"]
        body.append("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in cells) + "</tr>")
    return (f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title>"
            "<style>:root{color-scheme:light dark}body{font:14px system-ui;margin:16px}table{border-collapse:collapse;display:block;overflow-x:auto}"
            "td,th{border:1px solid #8885;padding:4px 8px;white-space:nowrap}th{position:sticky;top:0;background:Canvas}</style>"
            f"<h1>{html.escape(title)}</h1><p>View: {VIEW}. Small models, LoRA, few checkpoints: read CIs and n_pos before comparing rows. "
            "Compare doses only within one monitor.</p><table><thead><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>")


def build(out_dir: Path = Path("leaderboard"), runs_root: Path | None = None) -> Path:
    rows = collect_rows(runs_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.json").write_text(json.dumps(rows, indent=1))
    (out_dir / "index.html").write_text(render_html(rows))
    return out_dir / "index.html"
