"""Frozen DEV-049A recipe: matched training, one-shot certification, paid services."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import torch

from tac_osm.development.frozen_action_heads import predictive_metrics
from tac_osm.development.public_state_training import (
    objective, outcome_tensor, prediction_from_state, public_tensors,
)
from tac_osm.development.reliability_gate import (
    outcome_totals, select_threshold, services_for, zero_event_upper_bound,
)
from tac_osm.development.schema_maintenance import (
    DEADLINES, ROLES, exact_public_targets, execute, make_pair,
)
from tac_osm.development.unified_state_prototype import UnifiedPredictiveStatePrototype
from run_frozen_policy_dev045 import cohort, digest, targets
from run_joint_training_dev046 import predict

PROTOCOL = ROOT / "research/dev049/study-protocol.json"
ARMS = ("forecast", "joint", "failure_penalized")


def build(seed):
    torch.manual_seed(seed)
    return UnifiedPredictiveStatePrototype(entities=8, dim=8, budget=8)


def training_cases(seed, step, smoke=False):
    return make_pair(seed * 100000 + step, 32 if smoke else (64, 128)[step % 2],
                     (32, 128, 512)[(step // 4) % 3], ROLES[step % 4],
                     DEADLINES[(step // 12) % 3], domain="dev049a-fit")


def study_cohort(seed, split, smoke=False):
    repetitions = {"calibration": 1, "certification": 2, "evaluation": 4}[split]
    return cohort(seed, "dev049a-" + split, 1 if smoke else repetitions, smoke)


def loss_for(probabilities, labels, arm):
    loss = objective(probabilities, labels, "forecast" if arm == "forecast" else "joint")
    if arm == "failure_penalized":
        predicted_cost = 4 * probabilities + probabilities.new_tensor([0., 1.])
        soft_policy = torch.softmax(-predicted_cost / .25, dim=-1)
        # Simulator labels supervise a smooth failure surrogate, never the policy input.
        loss = loss + 16 * (soft_policy * labels).sum(-1).mean()
    return loss


def evaluate(cases, conditions, probabilities, threshold):
    rows = []
    for case, condition, p in zip(cases, conditions, probabilities):
        services = services_for(p, threshold)
        physical = execute(case.world, case.horizon, services, 0)
        rows.append(dict(**condition, probabilities=p, services=list(services), physical=physical))
    totals = {k: sum(r["physical"][k] for r in rows)
              for k in ("failures", "total_cost", "unnecessary_services", "inspections")}
    totals["services"] = sum(len(r["services"]) for r in rows)
    totals["failed_worlds"] = sum(r["physical"]["failures"] > 0 for r in rows)
    totals["failed_pairs"] = sum(any(r["physical"]["failures"] for r in rows[i:i+2])
                                 for i in range(0, len(rows), 2))
    totals["cost_per_entity"] = totals["total_cost"] / (8 * len(rows))
    totals["zero_failure_pair_risk_upper_95"] = (
        zero_event_upper_bound(len(rows) // 2) if totals["failures"] == 0 else None)
    return dict(threshold=threshold, rows=rows, physical=totals)


def run(seed, output, smoke=False):
    if seed not in range(4901, 4911):
        raise ValueError("unregistered seed")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("freeze all source before research")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cohorts = {s: study_cohort(seed, s, smoke)
               for s in ("calibration", "certification", "evaluation")}
    fit_cases = [training_cases(seed, step, smoke) for step in range(2 if smoke else 216)]
    fingerprints = {s: [c.fingerprint for c in cs] for s, (cs, _) in cohorts.items()}
    fingerprints["fit"] = [c.fingerprint for pair in fit_cases for c in pair]
    sets = list(map(set, fingerprints.values()))
    assert all(not a & b for i, a in enumerate(sets) for b in sets[i+1:])
    report = dict(study="DEV-049A", status="DEVELOPMENT_NOT_CONFIRMATORY", seed=seed,
                  smoke=smoke, source=subprocess.check_output(["git", "rev-parse", "HEAD"],
                      cwd=ROOT, text=True).strip(), protocol_sha256=digest(PROTOCOL),
                  split_fingerprints=fingerprints, arms={})
    for arm in ARMS:
        model = build(seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
        start = time.monotonic()
        curve = []
        for step, cases in enumerate(fit_cases):
            state = model.encode(*public_tensors([c.observations for c in cases]))
            p = prediction_from_state(model, state,
                    torch.tensor([c.horizon for c in cases], dtype=torch.float32))
            loss = loss_for(p, outcome_tensor(cases), arm)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if step % 36 == 0 or step == len(fit_cases)-1:
                curve.append(dict(step=step+1, loss=float(loss.detach())))
        training_seconds = time.monotonic() - start
        model.eval()
        checkpoint = output / (arm + ".pt")
        torch.save(model.state_dict(), checkpoint)
        predictions, inference_seconds = {}, {}
        for split, (cases, _) in cohorts.items():
            p, seconds = predict(model, cases)
            predictions[split] = p.tolist()
            inference_seconds[split] = seconds
        y_cal = targets(cohorts["calibration"][0]).tolist()
        cost_threshold, cost_candidates = select_threshold(predictions["calibration"], y_cal,
                                                           zero_failures=False)
        safe_threshold, safe_candidates = select_threshold(predictions["calibration"], y_cal,
                                                           zero_failures=True)
        certificate = outcome_totals(predictions["certification"],
                                     targets(cohorts["certification"][0]).tolist(), safe_threshold)
        accepted = certificate["failures"] == 0
        thresholds = dict(cost_tuned_threshold=cost_threshold,
                          zero_calibration_failures=safe_threshold,
                          certified_or_service_all=safe_threshold if accepted else -1.)
        report["arms"][arm] = dict(checkpoint_sha256=digest(checkpoint), curve=curve,
            training_seconds=training_seconds, inference_seconds=inference_seconds,
            trainable_parameters=sum(p.numel() for p in model.parameters()), state_scalars=200,
            predictions=predictions, calibration_candidates=cost_candidates,
            zero_failure_candidates=safe_candidates, certificate=certificate,
            certificate_accepted=accepted, certification_used_threshold=safe_threshold,
            prediction_metrics=predictive_metrics(predictions["evaluation"],
                                                 targets(cohorts["evaluation"][0])),
            policies={name: evaluate(*cohorts["evaluation"], predictions["evaluation"], t)
                      for name, t in thresholds.items()})
        print(json.dumps(dict(seed=seed, arm=arm, certificate_accepted=accepted,
              costs={k: v["physical"]["cost_per_entity"] for k, v in
                     report["arms"][arm]["policies"].items()})), flush=True)
    cases, conditions = cohorts["evaluation"]
    exact = [[[v, 0] for v in exact_public_targets(c.observations, c.tick, c.horizon)]
             for c in cases]
    report["controls"] = dict(exact_public_history=evaluate(cases, conditions, exact, .25),
                               service_all=evaluate(cases, conditions, exact, -1.))
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.seed, args.output, args.smoke)
