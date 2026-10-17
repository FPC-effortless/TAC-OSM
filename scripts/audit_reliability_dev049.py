"""Checkpoint inference and independent simulator replay; not optimization replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import torch

from tac_osm.development.schema_maintenance import execute, exact_public_targets
from run_frozen_policy_dev045 import digest, targets
from run_joint_training_dev046 import predict
from run_reliability_dev049 import ARMS, PROTOCOL, build, study_cohort, training_cases


def decisions(p, threshold):
    return tuple(i for i, pair in enumerate(p)
                 if threshold == -1 or pair[0] - pair[1] > threshold)


def audit(directory):
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    assert report["study"] == "DEV-049A" and not report["smoke"]
    assert report["protocol_sha256"] == digest(PROTOCOL)
    seed = report["seed"]
    assert seed in range(4901, 4911) and seed == int(directory.name)
    source_protocol = subprocess.check_output(
        ["git", "show", report["source"] + ":research/dev049/study-protocol.json"], cwd=ROOT)
    import hashlib
    assert hashlib.sha256(source_protocol).hexdigest() == report["protocol_sha256"]
    assert set(report["arms"]) == set(ARMS)
    assert set(report["controls"]) == {"exact_public_history", "service_all"}
    cohorts = {s: study_cohort(seed, s) for s in ("calibration", "certification", "evaluation")}
    fingerprints = {s: [c.fingerprint for c in cs] for s, (cs, _) in cohorts.items()}
    fingerprints["fit"] = [c.fingerprint for step in range(216)
                           for c in training_cases(seed, step)]
    assert fingerprints == report["split_fingerprints"]
    sets = list(map(set, fingerprints.values()))
    assert all(not a & b for i, a in enumerate(sets) for b in sets[i+1:])
    summary = {}
    for arm in ARMS:
        result = report["arms"][arm]
        ck = directory / (arm + ".pt")
        assert digest(ck) == result["checkpoint_sha256"]
        model = build(seed)
        model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
        model.eval()
        for split, (cases, _) in cohorts.items():
            p, _ = predict(model, cases)
            torch.testing.assert_close(p, torch.tensor(result["predictions"][split]), rtol=0, atol=0)
        cal_p = result["predictions"]["calibration"]
        cal_y = targets(cohorts["calibration"][0]).tolist()
        candidates = []
        for t in [-1.] + [i / 40 for i in range(41)]:
            failures = services = failed_worlds = 0
            for p, y in zip(cal_p, cal_y):
                chosen = decisions(p, t)
                count = sum(v[int(i in chosen)] for i, v in enumerate(y))
                failures += count
                services += len(chosen)
                failed_worlds += count > 0
            candidates.append(dict(threshold=t, failures=failures, failed_worlds=failed_worlds,
                                   services=services, total_cost=4*failures+services))
        assert candidates == result["calibration_candidates"] == result["zero_failure_candidates"]
        cost = min(candidates, key=lambda x: (x["total_cost"], x["threshold"]))["threshold"]
        safe = min((x for x in candidates if x["failures"] == 0),
                   key=lambda x: (x["total_cost"], x["threshold"]))["threshold"]
        cert_counts = dict(failures=0, failed_worlds=0, services=0, total_cost=0)
        for c, p in zip(cohorts["certification"][0], result["predictions"]["certification"]):
            physical = execute(c.world, c.horizon, decisions(p, safe), 0)
            for k in ("failures", "total_cost"):
                cert_counts[k] += physical[k]
            cert_counts["failed_worlds"] += physical["failures"] > 0
            cert_counts["services"] += len(physical["services"])
        assert cert_counts == result["certificate"]
        accepted = cert_counts["failures"] == 0
        assert accepted == result["certificate_accepted"]
        assert safe == result["certification_used_threshold"]
        expected = dict(cost_tuned_threshold=cost, zero_calibration_failures=safe,
                        certified_or_service_all=safe if accepted else -1.)
        assert set(result["policies"]) == set(expected)
        for name, policy in result["policies"].items():
            assert policy["threshold"] == expected[name]
            replay_policy(policy, cohorts["evaluation"], result["predictions"]["evaluation"])
            summary[arm + "/" + name] = policy["physical"]
    cases = cohorts["evaluation"][0]
    exact = [[[v, 0] for v in exact_public_targets(c.observations, c.tick, c.horizon)] for c in cases]
    for name, policy in report["controls"].items():
        assert policy["threshold"] == (.25 if name == "exact_public_history" else -1.)
        replay_policy(policy, cohorts["evaluation"], exact)
        summary[name] = policy["physical"]
    return dict(seed=seed, source=report["source"], report_sha256=digest(directory / "report.json"),
                certificate_accepted={a: report["arms"][a]["certificate_accepted"] for a in ARMS},
                policies=summary)


def replay_policy(policy, cohort_data, probabilities):
    cases, conditions = cohort_data
    assert len(policy["rows"]) == len(cases) == len(probabilities)
    totals = dict(failures=0, total_cost=0., unnecessary_services=0, inspections=0, services=0,
                  failed_worlds=0, failed_pairs=0)
    for c, cond, p, row in zip(cases, conditions, probabilities, policy["rows"]):
        assert all(row[k] == v for k, v in cond.items())
        assert row["probabilities"] == p
        services = decisions(p, policy["threshold"])
        assert list(services) == row["services"]
        replay = execute(c.world, c.horizon, services, 0)
        assert replay == row["physical"]
        for k in ("failures", "total_cost", "unnecessary_services", "inspections"):
            totals[k] += replay[k]
        totals["services"] += len(services)
        totals["failed_worlds"] += replay["failures"] > 0
    totals["failed_pairs"] = sum(any(r["physical"]["failures"] for r in policy["rows"][i:i+2])
                                 for i in range(0, len(cases), 2))
    totals["cost_per_entity"] = totals["total_cost"] / (8 * len(cases))
    totals["zero_failure_pair_risk_upper_95"] = (
        1 - .05 ** (1 / (len(cases) // 2)) if not totals["failures"] else None)
    for k, v in totals.items():
        if isinstance(v, float):
            assert abs(policy["physical"][k] - v) < 1e-12
        else:
            assert policy["physical"][k] == v


def summarize(rows, output):
    assert [r["seed"] for r in rows] == list(range(4901, 4911))
    assert len({r["source"] for r in rows}) == 1
    summary = {}
    for name in rows[0]["policies"]:
        values = [r["policies"][name] for r in rows]
        costs = [v["cost_per_entity"] for v in values]
        summary[name] = dict(mean_cost=statistics.mean(costs), seed_sd=statistics.stdev(costs),
            min_cost=min(costs), max_cost=max(costs), failures=sum(v["failures"] for v in values),
            per_seed_failures=[v["failures"] for v in values],
            unnecessary_services=sum(v["unnecessary_services"] for v in values))
    result = dict(status="AUDITED_DEVELOPMENT_NOT_CONFIRMATORY", seeds=rows, summary=summary,
        limitations=["Replay checks checkpoint inference and physical decisions, not each optimization step.",
                     "Ten seeds do not establish real-world or zero-risk safety.",
                     "No integrated superiority or constant-total entity memory claim."])
    Path(output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


def main(root, output):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    summarize([audit(Path(root) / str(seed)) for seed in range(4901, 4911)], output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.root, args.output)
