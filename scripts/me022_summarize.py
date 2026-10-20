"""Describe the registered ME022 factorial; never choose or refit a checkpoint."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def summarize(root, out):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    r = json.loads((root/"results.json").read_text())
    c = r["contract"]
    rows = {(q["arm"], q["seed"]): q for q in r["models"]}
    assert len(rows) == 12
    arms, seeds = c["arms"], c["seeds"]
    vals = np.array([[rows[a, s]["dev"]["Brier"] for s in seeds] for a in arms])
    d = vals[3]-vals[0]
    summary = {
        "study": r["study"], "fitting_source_commit": r["fitting_source_commit"],
        "source_sha256": c["source_sha256"], "contract_sha256": r["contract_sha256"],
        "mean_brier": dict(zip(arms, vals.mean(1).tolist())),
        "primary_paired_seed_deltas": d.tolist(),
        "primary_mean_delta": float(d.mean()),
        "relative_brier_reduction": float(-d.mean()/vals[0].mean()),
        "engineering_candidate_gate_passed": bool((d < 0).all() and d.mean() <= -.005),
        "factorial_duration_effect_mean": float(((vals[1]-vals[0])+(vals[3]-vals[2])).mean()/2),
        "factorial_head_lr_effect_mean": float(((vals[2]-vals[0])+(vals[3]-vals[1])).mean()/2),
        "factorial_interaction_mean": float((vals[3]-vals[2]-vals[1]+vals[0]).mean()),
        "all_native_training_cpu_seconds": float(sum(x["train_cpu_seconds"] for x in rows.values())),
        "causal_prior32": r["causal_prior32"],
        "unmatched_mlp_mean_brier": float(np.mean([x["dev"]["Brier"] for x in r["unmatched_references"]])),
        "historical_development_only": True, "independent_test_opened": False,
        "architecture_superiority_claim": False,
    }
    table = ["| Native TAC configuration | Seed 14001 | Seed 14002 | Seed 14003 | Mean Brier | Mean log loss | Mean CPU s |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for i, arm in enumerate(arms):
        ll = np.mean([rows[arm, s]["dev"]["log_loss"] for s in seeds])
        cpu = np.mean([rows[arm, s]["train_cpu_seconds"] for s in seeds])
        table.append(f"| {arm} | "+" | ".join(f"{v:.6f}" for v in vals[i])+f" | {vals[i].mean():.6f} | {ll:.6f} | {cpu:.2f} |")
    (out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (out/"table.md").write_text("\n".join(table)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for arm in arms:
        color = None
        for seed in seeds:
            trace = rows[arm, seed]["trace"]
            for ax, xkey in zip(axes, ("step", "train_cpu_seconds")):
                line, = ax.plot([t[xkey] for t in trace], [t["dev"]["Brier"] for t in trace],
                                alpha=.25, color=color, linewidth=1)
                if color is None: color = line.get_color()
        traces = [rows[arm, seed]["trace"] for seed in seeds]
        for ax, xkey in zip(axes, ("step", "train_cpu_seconds")):
            ax.plot(np.mean([[t[xkey] for t in tr] for tr in traces], axis=0),
                    np.mean([[t["dev"]["Brier"] for t in tr] for tr in traces], axis=0),
                    label=arm, color=color, linewidth=2)
    axes[0].set_xlabel("Optimizer updates (same exposure schedule)")
    axes[1].set_xlabel("Measured training CPU seconds (excludes evaluation)")
    for ax in axes:
        ax.set_ylabel("Historical-development Brier ↓")
        ax.axhline(summary["causal_prior32"]["Brier"], color="gray", linestyle=":", label="Fixed causal prior32")
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=8)
    fig.suptitle("Native TAC readout optimization · real KuaiRec · 3 seeds\nDevelopment diagnostics; thin lines are seeds, thick lines are means")
    fig.savefig(out/"learning-curves.svg")
    fig.savefig(out/"learning-curves.png", dpi=160)
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/me022")
    p.add_argument("--out", default="research/mathematical-engineering/me022")
    args = p.parse_args()
    summarize(args.root, args.out)
