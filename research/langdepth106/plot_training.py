"""Read existing FIT logs only; summarize and plot the six LANGDEPTH106 fits.

This script never opens source data or checkpoints, runs a model, scores DEV,
selects checkpoints, changes hyperparameters, or performs training. It requires
the completed original four/eight-layer logs for all three declared seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np


ARMS = ("four", "eight")
SEEDS = (10601, 10602, 10603)
STEPS = 4096
FINAL_WINDOW = 256
ROLLING_WINDOW = 64
EXPECTED = {
    "four": {"parameter_count": 998024, "n_layers": 4, "d_model": 120, "ffn_dim": 224},
    "eight": {"parameter_count": 999784, "n_layers": 8, "d_model": 72, "ffn_dim": 120},
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def causal_mean(values: np.ndarray, window: int = ROLLING_WINDOW) -> np.ndarray:
    """Mean of steps max(1,t-window+1)..t; never reads future or pads ends."""
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    stop = np.arange(1, len(values) + 1)
    start = np.maximum(0, stop - window)
    return (cumulative[stop] - cumulative[start]) / (stop - start)


def read_fit(root: Path, arm: str, seed: int) -> dict:
    directory = root / f"LANGDEPTH106-fit-{arm}-{seed}-attempt-1"
    receipt_path, log_path = directory / "FIT_RECEIPT.json", directory / "TRAIN_LOG.jsonl"
    receipt = json.loads(receipt_path.read_text())
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    require(receipt["arm"] == arm and receipt["seed"] == seed, f"identity mismatch: {directory.name}")
    require(receipt["status"] == "FIT_COMPLETED_EXPLORATORY", "fit is not complete")
    require(receipt["smoke"] is False and receipt["error"] is None, "smoke or failed fit")
    require(receipt["actual_steps"] == receipt["expected_steps"] == STEPS, "unexpected step count")
    require(len(records) == STEPS and [row["step"] for row in records] == list(range(1, STEPS + 1)),
            "training rows are missing, duplicated, or out of order")
    expected = EXPECTED[arm]
    require(receipt["parameter_count"] == expected["parameter_count"], "parameter count differs")
    for field in ("n_layers", "d_model", "ffn_dim"):
        require(receipt["config"][field] == expected[field], f"unexpected {field}")
    losses = np.asarray([row["loss"] for row in records], dtype=np.float64)
    require(np.isfinite(losses).all() and (losses >= 0).all(), "nonfinite or negative loss")
    require(losses[0] == receipt["first_fit_loss"] and losses[-1] == receipt["last_fit_loss"],
            "receipt endpoints differ from the complete log")
    offsets = [offset for row in records for offset in row["offsets"]]
    offsets_sha = hashlib.sha256(np.asarray(offsets, dtype=np.int64).tobytes()).hexdigest()
    require(offsets_sha == receipt["sample_offset_sha256"], "sample offset replay digest differs")
    require(len(offsets) == receipt["sample_draws"], "offset count differs")
    require(records[-1]["sample_draws"] == receipt["sample_draws"], "final logged draw count differs")
    require(records[-1]["training_byte_presentations"] == receipt["training_byte_presentations"],
            "final logged byte presentations differ")
    require(receipt["training_byte_presentations"] == 4194304, "unexpected exposure")
    require(all(math.isfinite(receipt[key]) and receipt[key] > 0
                for key in ("wall_seconds", "process_cpu_seconds", "peak_rss_bytes")),
            "invalid resource receipt")
    tail = losses[-FINAL_WINDOW:]
    exposure = receipt["training_byte_presentations"]
    public = {
        "arm": arm, "seed": seed, "steps": STEPS,
        "parameter_count": receipt["parameter_count"], "config": receipt["config"],
        "first_step_fit_loss_nats_per_next_byte": float(losses[0]),
        "last_step_fit_loss_nats_per_next_byte": float(losses[-1]),
        "final_256_updates": {
            "first_step": STEPS - FINAL_WINDOW + 1, "last_step": STEPS,
            "mean_loss_nats_per_next_byte": float(tail.mean()),
            "minimum_step_loss_nats_per_next_byte": float(tail.min()),
            "maximum_step_loss_nats_per_next_byte": float(tail.max()),
        },
        "wall_seconds": receipt["wall_seconds"],
        "process_cpu_seconds": receipt["process_cpu_seconds"],
        "peak_rss_bytes": receipt["peak_rss_bytes"],
        "training_byte_presentations": exposure, "sample_draws": receipt["sample_draws"],
        "predicted_byte_presentations_per_wall_second": exposure / receipt["wall_seconds"],
        "predicted_byte_presentations_per_process_cpu_second": exposure / receipt["process_cpu_seconds"],
        "sample_offset_sha256": offsets_sha,
        "source_sha256": receipt["source_sha256"], "train_sha256": receipt["train_sha256"],
        "protocol_sha256": receipt["protocol_sha256"],
        "implementation_sha256": receipt["implementation_sha256"],
        "receipt_sha256": sha256(receipt_path), "training_log_sha256": sha256(log_path),
        "source_directory": directory.name,
        "hardware": receipt["hardware"], "cpu_threads": receipt["cpu_threads"],
        "precision": receipt["precision"], "torch_version": receipt["torch_version"],
        "full_training_flops": receipt["full_training_flops"],
    }
    return {"receipt": receipt, "records": records, "losses": losses, "summary": public}


def summarize(fits: dict) -> dict:
    identity_keys = ("protocol_sha256", "source_sha256", "train_sha256", "source_receipt_sha256",
                     "implementation_sha256", "optimizer_metadata", "scheduler_metadata")
    reference = fits[("four", SEEDS[0])]["receipt"]
    for fit in fits.values():
        for key in identity_keys:
            require(fit["receipt"][key] == reference[key], f"unmatched {key}")
    pairs = []
    for seed in SEEDS:
        four, eight = fits[("four", seed)], fits[("eight", seed)]
        require([row["offsets"] for row in four["records"]] ==
                [row["offsets"] for row in eight["records"]], "paired observed offsets differ")
        require([row["lr"] for row in four["records"]] ==
                [row["lr"] for row in eight["records"]], "paired learning rates differ")
        require(four["receipt"]["sample_offset_sha256"] ==
                eight["receipt"]["sample_offset_sha256"], "paired offset digests differ")
        pairs.append({
            "seed": seed, "identical_all_4096_observed_offset_batches": True,
            "identical_all_4096_learning_rates": True,
            "sample_offset_sha256": four["receipt"]["sample_offset_sha256"],
            "final_256_mean_loss_eight_minus_four":
                eight["summary"]["final_256_updates"]["mean_loss_nats_per_next_byte"]
                - four["summary"]["final_256_updates"]["mean_loss_nats_per_next_byte"],
            "wall_seconds_eight_divided_by_four":
                eight["receipt"]["wall_seconds"] / four["receipt"]["wall_seconds"],
            "process_cpu_seconds_eight_divided_by_four":
                eight["receipt"]["process_cpu_seconds"] / four["receipt"]["process_cpu_seconds"],
        })
    aggregates = {}
    for arm in ARMS:
        rows = [fits[(arm, seed)]["summary"] for seed in SEEDS]
        terminal = [row["final_256_updates"]["mean_loss_nats_per_next_byte"] for row in rows]
        aggregates[arm] = {
            "n_fits": len(rows), "parameter_count": EXPECTED[arm]["parameter_count"],
            "mean_first_step_fit_loss": mean(row["first_step_fit_loss_nats_per_next_byte"] for row in rows),
            "mean_last_step_fit_loss": mean(row["last_step_fit_loss_nats_per_next_byte"] for row in rows),
            "mean_final_256_update_fit_loss": mean(terminal),
            "seed_range_final_256_update_mean_fit_loss": [min(terminal), max(terminal)],
            "mean_wall_seconds_per_fit": mean(row["wall_seconds"] for row in rows),
            "mean_process_cpu_seconds_per_fit": mean(row["process_cpu_seconds"] for row in rows),
            "mean_peak_rss_bytes": mean(row["peak_rss_bytes"] for row in rows),
            "arithmetic_mean_predicted_byte_presentations_per_wall_second":
                mean(row["predicted_byte_presentations_per_wall_second"] for row in rows),
            "pooled_predicted_byte_presentations_per_wall_second":
                sum(row["training_byte_presentations"] for row in rows) / sum(row["wall_seconds"] for row in rows),
        }
    return {
        "schema": "langdepth106.descriptive-training-summary.v1",
        "status": "DESCRIPTIVE_FIT_LOG_SUMMARY_NOT_HELDOUT_EVALUATION",
        "per_fit": [fits[(arm, seed)]["summary"] for arm in ARMS for seed in SEEDS],
        "arm_aggregates": aggregates, "paired_seed_checks": pairs,
        "source_identity": {key: reference[key] for key in identity_keys},
        "parameter_gap_fraction_of_smaller_arm": (999784 - 998024) / 998024,
        "plot": {
            "file": "learning_curves.png", "steps": [1, STEPS],
            "unit": "FIT next-byte cross entropy in natural-log nats",
            "smoothing": "causal arithmetic mean of max(1,t-63)..t; shorter prefix at steps 1..63",
            "main_lines": "arithmetic mean of the three smoothed seed curves in each arm",
            "main_band": "pointwise minimum-to-maximum of the three smoothed seed curves; NOT a confidence interval",
            "individual_curves": "all six fits, each shown through actual update 4096",
        },
        "limitations": [
            "Training minibatch loss is not held-out quality or generalization evidence.",
            "Each fit has 4,194,304 predicted-byte presentations from sampled windows; repeated presentations do not guarantee full-epoch coverage.",
            "The four/eight-layer arms change depth, residual width and FFN width together; they compare parameter allocation, not an isolated depth effect.",
            "Three paired seeds on one corpus are not three independent task families; no uncertainty interval or significance claim is computed here.",
            "Resource values are receipt-based CPU FP32 two-thread fit measurements, including fit-run overhead, not GPU training/inference throughput or complete FLOPs.",
            "Summed per-fit wall time is aggregate run cost and not the elapsed duration of concurrently executed jobs.",
            "Only TRAIN_LOG.jsonl and FIT_RECEIPT.json evidence is read; no dataset bytes, checkpoint tensors, DEV or TEST are accessed by this script.",
        ],
    }


def plot(fits: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                         "axes.labelsize": 10, "svg.fonttype": "none"})
    colors = {"four": "#1966A5", "eight": "#AD640E"}
    labels = {"four": "4 layers · width 120 · FF 224 · 998,024 parameters",
              "eight": "8 layers · width 72 · FF 120 · 999,784 parameters"}
    x = np.arange(1, STEPS + 1)
    fig = plt.figure(figsize=(12.8, 9.3), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=.40, wspace=.12)
    main = fig.add_subplot(grid[0, :])
    axes = [main, fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    styles = ("-", "--", ":")
    for arm, sub in zip(ARMS, axes[1:]):
        smooth = np.stack([causal_mean(fits[(arm, seed)]["losses"]) for seed in SEEDS])
        main.fill_between(x, smooth.min(axis=0), smooth.max(axis=0), color=colors[arm], alpha=.16, linewidth=0)
        main.plot(x, smooth.mean(axis=0), color=colors[arm], linewidth=2.1,
                  linestyle="-" if arm == "four" else "--", label=labels[arm])
        for seed, values, style in zip(SEEDS, smooth, styles):
            sub.plot(x, values, color=colors[arm], linewidth=1.5, linestyle=style, label=f"Seed {seed}")
        sub.set_title(f"{EXPECTED[arm]['n_layers']}-layer arm: all three fits", loc="left", pad=10)
        sub.legend(loc="upper right", frameon=False, fontsize=9)
    main.set_title("Mean of three fits per arm; shaded region is the seed range, not a confidence interval",
                   loc="left", pad=12, fontsize=12)
    main.legend(loc="upper right", frameon=False, fontsize=10)
    for axis in axes:
        axis.set_xlim(1, STEPS)
        axis.set_ylim(1.25, 5.85)
        axis.set_xticks([1, 1024, 2048, 3072, 4096], ["1", "1,024", "2,048", "3,072", "4,096"])
        axis.set_xlabel("Optimizer update")
        axis.grid(axis="y", color="#DFE3E8", linewidth=.65)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#8B939D")
        axis.tick_params(length=3, color="#8B939D")
        axis.set_axisbelow(True)
    axes[0].set_ylabel("FIT loss (nats per next byte)")
    axes[1].set_ylabel("FIT loss (nats per next byte)")
    axes[2].set_yticklabels([])
    fig.suptitle("LANGDEPTH106 · TinyShakespeare training trajectories", x=.08, ha="left", y=.97,
                 fontsize=17, fontweight="bold", color="#202833")
    fig.text(.08, .927, "Matched ~1M parameters · three paired seeds · CPU FP32 · 4,096 updates per fit",
             fontsize=11, color="#485462")
    fig.text(.08, .055,
             "Every line uses a causal 64-update mean (shorter prefix at updates 1–63); no future smoothing or endpoint extension.\n"
             "Each fit presents 4,194,304 next-byte targets from sampled FIT windows; this does not guarantee full-epoch coverage.\n"
             "Training loss does not measure held-out quality. Depth and width change together in this allocation study.",
             fontsize=9.5, color="#485462", linespacing=1.5)
    fig.subplots_adjust(left=.08, right=.975, top=.86, bottom=.17)
    fig.savefig(output / "learning_curves.png", dpi=190, facecolor="white")
    plt.close(fig)
    # Compact repository encoding; the plotted data and pixel geometry remain
    # unchanged. Pillow is also Matplotlib's PNG backend dependency.
    from PIL import Image
    path = output / "learning_curves.png"
    with Image.open(path) as image:
        compact = image.convert("RGB").quantize(colors=128)
    compact.save(path, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    fits = {(arm, seed): read_fit(args.fits_root, arm, seed) for arm in ARMS for seed in SEEDS}
    summary = summarize(fits)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot(fits, args.output_dir)
    summary["plot"]["sha256"] = sha256(args.output_dir / "learning_curves.png")
    (args.output_dir / "TRAINING_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["arm_aggregates"], indent=2))


if __name__ == "__main__":
    main()
