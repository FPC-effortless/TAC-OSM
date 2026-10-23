"""ME062 original-FIT prefit measurement. NO optimizer steps or heldout access.

Read only original source FIT episodes through the checksum-locked ME060 loader,
then produce reproducible parameter and PyTorch-supported operator FLOP
receipts for identical source input, same objective and candidate arms.
Results do not qualify a trained comparison or an independent test.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from scripts.me060_train_trajectory import (
    prepare, batch, schedules, digest, HISTORY, HORIZON, BATCH, UPDATES,
)
from tac_osm.real_robot_me061_reliable import VerifiedState
from tac_osm.real_robot_me062_selective import (
    SelectiveTrajectoryModel, selective_gaussian_forecast_loss,
)

SEEDS = (62001, 62002, 62003, 62004, 62005)
CONFIG = {"native_tac": (32, 64), "gru_world": (44, 82)}
FIT_SHA256 = "b55d0e1f894072c5738ef8e45aa283a3b464c1a1dc8f60c63594666d16b30a49"


def model_for(name, identified):
    return SelectiveTrajectoryModel(identified, name, *CONFIG[name])


def objective(model, data):
    states, actions, human = data
    belief = model.initial(len(states))
    with torch.no_grad():
        for t in range(HISTORY - 16):
            belief = model.observe(
                belief, states[:, t], actions[:, t], states[:, t + 1], human[:, t, None])
    snapshots = []
    for t in range(HISTORY - 16, HISTORY):
        belief = model.observe(
            belief, states[:, t], actions[:, t], states[:, t + 1], human[:, t, None])
        snapshots.append(belief)
    future = model.forecast(belief, states[:, HISTORY],
                            actions[:, HISTORY:HISTORY + HORIZON])
    alive = (~human[:, HISTORY:HISTORY + HORIZON]).cumprod(1).bool().unsqueeze(-1)
    main = selective_gaussian_forecast_loss(
        model, future, states[:, HISTORY + 1:HISTORY + HORIZON + 1], alive)
    stacked = VerifiedState(
        *(torch.stack([s[i] for s in snapshots], 1).flatten(0, 1)
          for i in range(5)))
    start = HISTORY - 15
    short = model.forecast(
        stacked, states[:, start:HISTORY + 1].flatten(0, 1),
        actions[:, start:HISTORY + 1].flatten(0, 1)[:, None])
    correct = states[:, start + 1:HISTORY + 2].flatten(0, 1)[:, None]
    mask = (~human[:, start:HISTORY + 1]).flatten()[:, None, None]
    return main + .25 * selective_gaussian_forecast_loss(model, short, correct, mask)


def profile_once(model, actual_batch):
    model.zero_grad(set_to_none=True)
    with profile(activities=[ProfilerActivity.CPU], with_flops=True) as p:
        loss = objective(model, actual_batch)
        loss.backward()
    estimated_flops = int(sum(event.flops for event in p.key_averages()))
    gradients = {key: (value.grad is not None and
                       bool(torch.isfinite(value.grad).all()))
                 for key, value in model.named_parameters()}
    # An initial zero-output model can have untrained upstream gradients;
    # an entirely absent path is reported as a failure, not omitted.
    return {"profiled_forward_backward_flops": estimated_flops,
            "initial_loss": float(loss.detach()),
            "has_finite_gradients": gradients,
            "parameters": sum(p.numel() for p in model.parameters())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    split, metadata, identified, source, source_hash = prepare(args.source_cache)
    if source_hash != FIT_SHA256:
        raise RuntimeError(
            f"authentic FIT source mismatch: expected {FIT_SHA256}, got {source_hash}")
    schedules_by_seed = {str(s): schedules(source, s) for s in SEEDS}
    # One identical original FIT minibatch serves all arms.
    selected = batch(source, schedules_by_seed[str(SEEDS[0])][0])
    results = {}
    for architecture in CONFIG:
        torch.manual_seed(SEEDS[0])
        model = model_for(architecture, identified)
        results[architecture] = profile_once(model, selected)
    names = list(CONFIG)
    sizes = [results[n]["parameters"] for n in names]
    flop_counts = [results[n]["profiled_forward_backward_flops"] for n in names]
    result = {
        "study": "ME062_AUTHENTIC_FIT_PREFLIGHT_ONLY",
        "training_status": "NOT_RUN",
        "optimizer_steps": 0,
        "cal_trajectory_opened": False,
        "dev_trajectory_opened": False,
        "test_trajectory_opened": False,
        "real_source_fit_sha256": source_hash,
        "fit_episode_count": len(source),
        "fit_partition_digest": digest(split["FIT"]),
        "cal_partition_digest": digest(split["CAL"]),
        "dev_partition_digest": digest(split["DEV"]),
        "seeds": list(SEEDS),
        "schedule_sha256": {seed: digest(s) for seed, s in schedules_by_seed.items()},
        "per_arm": results,
        "parameter_relative_gap": (max(sizes) - min(sizes)) / max(sizes),
        "operator_flop_relative_gap": (max(flop_counts) - min(flop_counts)) / max(flop_counts),
        "budget_if_trained": {"batch_size": BATCH, "updates": UPDATES,
                              "history": HISTORY, "horizon": HORIZON},
        "profile_limitation": "torch.profiler supported operators only; optimizer and uncounted ops excluded",
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "scientific_status": "PREFLIGHT_DIAGNOSTIC_NOT_TRAINED",
        "repo_replay_status": "BLOCKED",
        "independent_test": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("ME062_AUTHENTIC_FIT_PREFLIGHT=" + json.dumps(result, sort_keys=True),
          flush=True)
    if result["parameter_relative_gap"] > .01 or result["operator_flop_relative_gap"] > .01:
        raise SystemExit("ME062 PREFLIGHT RESOURCE_PARITY_FAIL: do not train")
    if not all(all(v["has_finite_gradients"].values()) for v in results.values()):
        raise SystemExit("ME062 PREFLIGHT INACTIVE_PARAMETERS: inspect before training")
    print("ME062 PREFLIGHT PARITY_WITHIN_ONE_PERCENT; TRAINING_NOT_STARTED", flush=True)


if __name__ == "__main__":
    main()
