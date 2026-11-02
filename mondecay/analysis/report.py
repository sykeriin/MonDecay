"""Dose-response figure + practitioner table from runs/*/metrics.<monitor>.json and manifests."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..registry import RUNS, list_runs  # noqa: E402


def collect(monitor: str, view: str = "cot+action", runs_root: Path | None = None) -> list[dict]:
    root = runs_root or RUNS
    rows = []
    for m in list_runs(root):
        p = root / m.run_id / f"metrics.{monitor}.json"
        if not p.exists():
            continue
        met = json.loads(p.read_text())
        for fam, fr in met["families"].items():
            v = fr["views"].get(view)
            if not v:
                continue
            rows.append(dict(run_id=m.run_id, base=m.base_model, intervention=m.intervention, dose=m.dose, seed=m.seed, family=fam,
                             recall=v["recall"], lo=v["recall_ci"][0], hi=v["recall_ci"][1], auroc=v["auroc"],
                             mis=fr["misbehaviour_rate"], cap=fr["capability"], n_pos=fr["n_pos"], cot_tokens=fr["cot_tokens_mean"]))
    return rows


def _num(d):
    try:
        return float(d)
    except (TypeError, ValueError):
        return None


def plot_dose_response(rows: list[dict], out: Path, title_suffix: str = "") -> Path:
    ivs = sorted({r["intervention"] for r in rows if r["intervention"] != "base"})
    fams = sorted({r["family"] for r in rows})
    if not ivs or not fams:
        raise ValueError("no non-base runs with metrics found")
    fig, axes = plt.subplots(len(ivs), len(fams), figsize=(4.2 * len(fams), 3.1 * len(ivs)), squeeze=False, sharey=True)
    for i, iv in enumerate(ivs):
        for j, fam in enumerate(fams):
            ax = axes[i][j]
            base = [r for r in rows if r["intervention"] == "base" and r["family"] == fam]
            pts = sorted([r for r in rows if r["intervention"] == iv and r["family"] == fam and _num(r["dose"]) is not None],
                         key=lambda r: _num(r["dose"]))
            if base:
                pts = [{**base[0], "dose": 0}] + pts
            if not pts:
                continue
            x = [_num(r["dose"]) for r in pts]
            ax.plot(x, [r["recall"] for r in pts], "o-", color="#1f77b4", label="monitor recall@5%FPR")
            ax.fill_between(x, [r["lo"] for r in pts], [r["hi"] for r in pts], color="#1f77b4", alpha=0.18)
            ax.plot(x, [r["mis"] for r in pts], "s--", color="#d62728", label="misbehaviour rate")
            ax.plot(x, [r["cap"] for r in pts], "^:", color="#2ca02c", label="capability")
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"{iv} / {fam}", fontsize=9)
            ax.set_xlabel("dose", fontsize=8)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle("MonDecay dose-response " + title_suffix, fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def practitioner_table(rows: list[dict], out: Path, banner: str = "") -> Path:
    base = {r["family"]: r for r in rows if r["intervention"] == "base"}
    lines = [banner, "", "| intervention | dose | family | n_pos | recall | Δrecall vs base | misbehaviour | capability | CoT tokens |", "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted((r for r in rows if r["intervention"] != "base"), key=lambda r: (r["intervention"], str(r["dose"]), r["family"])):
        b = base.get(r["family"])
        d = f"{r['recall'] - b['recall']:+.2f}" if b else "n/a"
        lines.append(f"| {r['intervention']} | {r['dose']} | {r['family']} | {r['n_pos']} | {r['recall']:.2f} [{r['lo']:.2f},{r['hi']:.2f}] | {d} | "
                     f"{r['mis']:.2f} | {r['cap']:.2f} | {r['cot_tokens']:.0f} |")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out
