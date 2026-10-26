#!/usr/bin/env python3
"""S1MOTION001 forensic read-only recovery of ORIGINAL trained checkpoint data.

No training; independently replay recorded DEV scores from GitHub Actions
run 40274339695's immutable first artifact. Original run failure is kept.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.s1motion001_real_aloha_fit import (
    DEV, FIT, MatchedGRU, SEEDS, System1ParallelWorld,
    flop_proxy, prepare, score, sha256, source_episodes,
)

LEDGER = ROOT / "research/parallel-six/S1MOTION001-ORIGINAL-LOG-LEDGER.json"


@torch.no_grad()
def cpu_step_latency(model, ep):
    """Observe latency on exactly one real DEV prefix; excludes hardware SLA."""
    x = ep["x"]
    u = ep["prior_u"]
    times = []
    if isinstance(model, System1ParallelWorld):
        state = model.initial_state(1)
        for i in range(x.shape[0]):
            begin = time.perf_counter_ns()
            _, state = model.step(state, x[i:i+1], u[i:i+1])
            times.append((time.perf_counter_ns() - begin)/1e6)
    else:
        h = None
        for i in range(x.shape[0]):
            inp = torch.cat([x[i:i+1], u[i:i+1]], dim=-1).unsqueeze(1)
            begin = time.perf_counter_ns()
            _, h = model.recurrent(inp, h)
            _ = model.readout(h[-1])
            times.append((time.perf_counter_ns() - begin)/1e6)
    return dict(p50_cpu_single_thread_ms=float(np.percentile(times[16:], 50)),
                p95_cpu_single_thread_ms=float(np.percentile(times[16:], 95)),
                warmup_dropped_steps=16, timed_steps=len(times)-16,
                excludes_input_normalization_and_python_source_io=True,
                real_hardware_SLA_qualified=False)


def replay(src: Path, cache: Path, out: Path):
    ledger = json.loads(LEDGER.read_text())
    first = json.loads((src / "before-dev-score.json").read_text())
    if (first["source"]["revision"] != "6728831ef9674b74f3ce20bf517890e9ee20b0d9"
        or len(first["fit_records"]) != 6 or first["trained_seeds"] != list(SEEDS)
        or first.get("actual_adverse_label_available") is not False):
        raise RuntimeError("original pre-DEV source/freeze manifest mismatch")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    episodes, source = source_episodes(cache)
    if source["full_parquet_sha256"] != first["source"]["full_parquet_sha256"]:
        raise RuntimeError("original publisher source hash mismatch")
    fit, dev, norm = prepare(
        {e:episodes[e] for e in FIT}, {e:episodes[e] for e in DEV})
    for k, expected in first["normalization"].items():
        observed = norm[k]
        if np.shape(observed) == ():
            if observed != expected:
                raise RuntimeError("FIT-only normalization threshold mismatch")
        elif not np.allclose(observed, expected, atol=0, rtol=0):
            raise RuntimeError("FIT-only normalization changed")

    out.mkdir(parents=True, exist_ok=True)
    results = []
    for record in ledger["original_observations"]:
        seed = record["seed"]
        pair = {}
        for arm in ("system1", "gru"):
            binary = (src / f"{arm}-{seed}.pt").read_bytes()
            if sha256(binary) != record[arm]["model_sha256"]:
                raise RuntimeError(f"original frozen model changed {seed}:{arm}")
            if not any(f["arm"] == arm and f["seed"] == seed and
                       f["checkpoint_sha256"] == sha256(binary)
                       for f in first["fit_records"]):
                raise RuntimeError("first manifest checkpoint receipt mismatch")
            # Trusted GitHub-generated artifact with SHA checked before loading.
            checkpoint = torch.load(src / f"{arm}-{seed}.pt",
                                    map_location="cpu", weights_only=False)
            if (checkpoint["architecture"] != arm or
                checkpoint["seed"] != seed or checkpoint["updates"] != 192 or
                "optimizer" not in checkpoint or "torch_rng" not in checkpoint):
                raise RuntimeError("original optimizer/RNG/exposure missing")
            scale = torch.tensor(norm["train_delta_scale_normalized"],
                                 dtype=torch.float32)
            if arm == "system1":
                model = System1ParallelWorld()
                model.set_train_delta_scale(scale)
            else:
                model = MatchedGRU(scale)
            model.load_state_dict(checkpoint["model"], strict=True)
            model.eval()
            observed = score(model, dev, norm)
            original = record[arm]
            for key, seen in (
                    ("MSE", observed["raw_joint_MSE"]["tac"]),
                    ("high_motion_MSE", observed["raw_joint_high_motion_MSE"]["tac"]),
                    ("proxy_AUPRC", observed["proxy_AUPRC"]),
                    ("proxy_AUROC", observed["proxy_AUROC"]),
                    ("proxy_Brier", observed["proxy_Brier"])):
                if not np.isclose(seen, original[key], rtol=1e-5, atol=1e-11):
                    raise RuntimeError(f"original log score mismatch {arm} {seed} {key}: {seen}, {original[key]}")
            pair[arm] = dict(original_checkpoint_sha256=sha256(binary),
                             scores=observed,
                             per_step_cpu_latency=cpu_step_latency(model, dev[DEV[0]]),
                             postfit_flop_operator_subset=flop_proxy(model, fit))
            print("S1MOTION001_ORIGINAL_COLD_REPLAY", json.dumps(
                dict(seed=seed, arm=arm,
                     original_MSE=original["MSE"],
                     replay_MSE=observed["raw_joint_MSE"]["tac"],
                     origin_checkpoint_sha=sha256(binary)),
                sort_keys=True), flush=True)
        results.append(dict(seed=seed, observations=pair,
                            predictive_gate_pass=False))
    report = {
        "source_type":"ORIGINAL_FROZEN_SOURCE_ARTIFACT_REPLAY_NO_MODEL_TRAINING",
        "original_training_workflow_failed_json_packaging":True,
        "original_run_id":ledger["original_run_id"],
        "original_artifact_id":ledger["original_artifact_id"],
        "original_head_sha":ledger["original_run_head_commit"],
        "recovery_does_not_replace_original_logged_scores":True,
        "real_source":source,
        "dev_results":results,
        "passed_seeds":0,
        "full_qualification_gate_pass":False,
        "matched_parameters_only_not_full_training_FLOPs":True,
        "real_adverse_labels_available":False,
        "independent_external_skill_proven":False,
        "device_latency_SLA_proven":False,
        "original_protected_TEST_episodes_used":False,
        "same_publisher_robot_across_fit_and_dev":True,
    }
    output = out / "s1motion001-cold-replay.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print("S1MOTION001_COLD_REPLAY_6_ORIGINAL_MODELS_PASS", flush=True)
    print("S1MOTION001_SCIENTIFIC_PREDICTIVE_GATE=FAIL_0_OF_3", flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--original-dir", required=True)
    p.add_argument("--cache", default="/tmp/s1motion001-recovery-source")
    p.add_argument("--output-dir",
                   default="research/diagnostics/s1motion001-recovery")
    a = p.parse_args()
    replay(Path(a.original_dir), Path(a.cache), ROOT / a.output_dir)


if __name__ == "__main__":
    main()
