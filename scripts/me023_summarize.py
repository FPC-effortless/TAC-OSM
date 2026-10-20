"""Fixed contrast and descriptive learning curves for ME023; never selects models."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def summarize(root, out):
    r = json.loads((Path(root)/"results.json").read_text())
    c = r["contract"]
    out = Path(out); out.mkdir(exist_ok=True, parents=True)
    rows = {(m["arm"],m["seed"]): m for m in r["models"]}
    assert len(rows) == 12
    delta = np.array([rows["action_prefix",s]["dev"]["Brier"]-rows["self_prefix",s]["dev"]["Brier"] for s in c["seeds"]])
    memory = np.array([rows["action_prefix",s]["mean_state_brier_delta"] for s in c["seeds"]])
    summary = {"source_commit": r["source_commit"], "primary_paired_deltas": delta.tolist(),
        "primary_mean_delta": float(delta.mean()), "action_prefix_mean_state_deltas": memory.tolist(),
        "engineering_gate_passed": bool((delta < 0).all() and delta.mean() <= -.0005 and (memory >= .0002).all()),
        "training_process_cpu_seconds": float(sum(m["train_cpu_seconds"] for m in rows.values())),
        "arm_means": {}, "independent_test_opened": False, "architectural_superiority_claim": False}
    lines = ["| Native configuration | Seed 14001 Brier | Seed 14002 | Seed 14003 | Mean Brier | Mean ECE | Mean-state Brier delta | CPU s |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for a in c["arms"]:
        arm = [rows[a,s] for s in c["seeds"]]
        mean = {k: float(np.mean([q["dev"][k] for q in arm])) for k in ("Brier","log_loss","ROC_AUC","ECE10")}
        mean["mean_state_brier_delta"] = float(np.mean([q["mean_state_brier_delta"] for q in arm]))
        mean["train_cpu_seconds"] = float(np.mean([q["train_cpu_seconds"] for q in arm]))
        summary["arm_means"][a] = mean
        lines.append(f"| {a} | "+" | ".join(f'{q["dev"]["Brier"]:.6f}' for q in arm)+
                     f' | {mean["Brier"]:.6f} | {mean["ECE10"]:.6f} | {mean["mean_state_brier_delta"]:+.6f} | {mean["train_cpu_seconds"]:.2f} |')
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (out/"table.md").write_text("\n".join(lines)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    for i,a in enumerate(c["arms"]):
        traces = [rows[a,s]["trace"] for s in c["seeds"]]
        color = f"C{i}"
        for ax, key in zip(axes,("step","train_cpu_seconds")):
            for tr in traces:
                ax.plot([q[key] for q in tr],[q["dev"]["Brier"] for q in tr],color=color,alpha=.25)
            ax.plot(np.mean([[q[key] for q in tr] for tr in traces],axis=0),
                    np.mean([[q["dev"]["Brier"] for q in tr] for tr in traces],axis=0),color=color,label=a,lw=2)
    for ax in axes:
        ax.set_ylabel("Historical-development Brier ↓"); ax.grid(alpha=.2)
        ax.axhline(.17068675597890645,color="gray",ls=":",label="Frozen ME022 mean (reference)")
    axes[0].set_xlabel("Optimizer updates")
    axes[1].set_xlabel("Measured training CPU seconds")
    axes[1].legend(fontsize=8)
    fig.suptitle("Bounded query-conditioned native TAC · causal-prefix study\nReal KuaiRec development only · thin lines are seeds")
    fig.savefig(out/"learning-curves.svg")
    fig.savefig(out/"learning-curves.png",dpi=160)
    print(json.dumps(summary))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--root",default="results/me023")
    p.add_argument("--out",default="research/mathematical-engineering/me023")
    a = p.parse_args(); summarize(a.root,a.out)
