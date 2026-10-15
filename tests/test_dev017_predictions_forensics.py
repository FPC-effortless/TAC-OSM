"""Integrity tests for DEV-019; simulated records are not research outcomes."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("forensic_dev017", ROOT / "scripts/diagnose_dev017_predictions.py")
analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)


def synthetic_report(seed=43):
    rows, summaries = {}, {}
    for arm in analyzer.ARMS:
        records = []
        for h in (32, 128, 512):
            for d in (16, 64, 128):
                for deadline in ("before", "at", "after"):
                    for role in ("retention", "acquisition", "pending_cancel", "reissued_cancel"):
                        for rep in range(8):
                            for partner in (0, 1):
                                condition = dict(history=h, due_delay=d, deadline=deadline,
                                                 role=role, repetition=rep, partner=partner)
                                prediction = 0 if arm == "tac_joint" and h == 512 else partner
                                logits = [2., 0.] if prediction == 0 else [0., 2.]
                                fingerprint = hashlib.sha256(json.dumps(condition, sort_keys=True).encode()).hexdigest()
                                controls = {mode: {"total_cost": 8., "cost_per_entity": 1.,
                                                   "failures": 0, "services": list(range(8)), "inspections": 0}
                                            for mode in ("none", "selected", "matched_random")}
                                records.append({"condition": condition, "fingerprint": fingerprint,
                                                "target_entity": 0, "wait_targets": [partner] + [0]*7,
                                                "logits": [[logits, [0., 1.], [0., 1.]]]
                                                          + [[[0., 1.]] * 3 for _ in range(7)],
                                                "predictions": [[prediction, 1, 1]] + [[1, 1, 1]] * 7,
                                                "controls": controls})
        extracted = analyzer.extract(records)
        rows[arm] = records
        summaries[arm] = {"primary_future_accuracy": analyzer.accuracy(
                              [e for e in extracted if analyzer.mature(e)]),
                          "H512_long_pending_accuracy": analyzer.accuracy(
                              [e for e in extracted if analyzer.critical(e)])}
    return {"study": "DEV-017", "status": "DEVELOPMENT_ONLY_NOT_CONFIRMATORY",
            "smoke": False, "seed": seed, "rows": rows, "summaries": summaries}


def test_stable_probability_and_balanced_pair_logic():
    assert analyzer.p1([1000, -1000]) < 1e-100
    assert analyzer.p1([-1000, 1000]) > 1 - 1e-10
    results = analyzer.analyze(synthetic_report())["arms"]
    collapsed, structured = results["tac_joint"], results["schema41"]
    assert collapsed["critical_H512"]["accuracy"] == .5
    assert collapsed["critical_H512"]["actual_positive_rate"] == .5
    assert collapsed["critical_H512"]["predicted_positive_rate"] == 0.
    assert collapsed["pair_metrics"]["predicted_pair_disagreement_rate"] == 0.
    assert structured["critical_H512"]["accuracy"] == 1.
    assert structured["pair_metrics"]["predicted_pair_disagreement_rate"] == 1.


def test_bad_prediction_or_cost_is_rejected():
    for kind in ("prediction", "cost"):
        report = synthetic_report()
        r = report["rows"]["tac_joint"][0]
        if kind == "prediction":
            r["predictions"][0][0] = 1
        else:
            r["controls"]["selected"]["total_cost"] = 0.
        try:
            analyzer.analyze(report)
        except AssertionError:
            continue
        assert False, "tampered evidence should be rejected"


def test_three_seed_provenance_and_export(tmp_path):
    paths = []
    for seed in (43, 59, 71):
        path = tmp_path / f"seed{seed}.json"
        path.write_text(json.dumps(synthetic_report(seed)))
        paths.append(path)
    report = analyzer.run(paths, tmp_path / "output")
    assert report["status"] == "POSTHOC_FORENSIC_NOT_CONFIRMATORY"
    assert len((tmp_path / "output" / "critical-H512.csv").read_text().splitlines()) == 19
    assert len((tmp_path / "output" / "cancellation-by-history.csv").read_text().splitlines()) == 55
    assert len(report["inputs"]) == 3
