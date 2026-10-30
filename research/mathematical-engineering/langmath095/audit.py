"""LANGMATH095 no-fit mathematical diagnostics, never capability evidence.

Run with NumPy alone. PyTorch/checkpoints are optional and must pass a
source/partition receipt before loading. Original model code is not edited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np

from scans import matrix_scan

ROOT = Path(__file__).resolve().parents[3]
BASE_COMMIT = "d08fbba33990c503446c2a7d5073219a767f2f62"
SOURCE_BLOBS = {
    "research/mathematical-engineering/context026/context_transition.py": "cfdfa1ef07fc8af0d011c9727d629200025d6652",
    "research/mathematical-engineering/langlayer038/recurrence_only.py": "1b0fca193ae61370a43dbe9ad7f9bc553569ad98",
    "research/mathematical-engineering/langscale070/scaled_recurrence.py": "fc91f1bcc5dbb226c1c80f261288759be39539a5",
    "research/language-scaling/lang10m/model.py": "6461e3e53c97d8188238cb129229992369943d87",
}


def source_inventory(root=ROOT):
    rows = []
    for relative, expected in SOURCE_BLOBS.items():
        path = Path(root) / relative
        if not path.is_file():
            rows.append({"path": relative, "status": "INCONCLUSIVE", "reason": "source unavailable"})
            continue
        raw = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        rows.append({"path": relative, "git_blob_sha1": blob,
                     "sha256": hashlib.sha256(raw).hexdigest(),
                     "status": "PASS" if blob == expected else "FAIL"})
    return rows


def retention_report():
    cap = .995
    return {
        "status": "PASS", "evidence": "algebraic_direct_carry_bound",
        "alpha_cap": cap, "half_life_updates": math.log(.5) / math.log(cap),
        "e_fold_updates": -1 / math.log(cap),
        "approx_zero_preactivation_alpha": cap * .5,
        "survival": {str(n): cap ** n for n in (1, 8, 32, 128, 512, 1024, 4096)},
        "limitation": "fixed input schedule, one carry path; not whole-model context capacity",
    }


def memory_load_report(seed=9501):
    rng = np.random.default_rng(seed)
    rows = []
    for count in (1, 2, 4, 8, 16, 32, 64):
        keys = rng.normal(size=(count, 8))
        keys /= np.linalg.norm(keys, axis=1, keepdims=True)
        values = np.tanh(rng.normal(size=(count, 8)))
        # Best linear map on this constructed arbitrary mapping, not trained TAC.
        fitted = np.linalg.lstsq(keys, values, rcond=None)[0]
        gram = keys @ keys.T
        offdiag = gram - np.eye(count)
        rows.append({"items": count, "address_rank": int(np.linalg.matrix_rank(keys)),
                     "best_linear_mse": float(np.mean((keys @ fitted - values) ** 2)),
                     "off_diagonal_squared_mean": float(np.sum(offdiag ** 2) / max(1, count * (count-1)))})
    return {"status": "PASS", "evidence": "constructed_linear_address_fixture",
            "key_dimension": 8, "isotropic_squared_overlap_expectation": 1/8,
            "rows": rows, "limitation": "single linear read, not an eight-fact bound on the model"}


def aligned_gain_report(seed=9502, steps=2048):
    alpha, erase, write = .9999, .0001, .5
    denominator = 1-alpha*(1-erase)
    gain = write/denominator
    bounded_write = (1-alpha)*write
    rng = np.random.default_rng(seed)
    state = np.zeros((16, 8))
    peak = 0.
    for _ in range(steps):
        key = rng.normal(size=8); key /= np.linalg.norm(key)
        value = np.tanh(rng.normal(size=16))
        a, e, gate = rng.uniform(size=3)
        w = (1-a)*gate
        state = a*(state-e*np.outer(state @ key, key)) + w*np.outer(value, key)
        peak = max(peak, float(np.linalg.norm(state)))
    return {"status": "PASS" if peak <= 4+1e-12 else "FAIL",
            "evidence": "constructed_gain_counterexample_and_bounded_update_fixture",
            "constant_effective_key": {"alpha": alpha, "erase": erase, "write": write,
                                       "equilibrium_gain": gain,
                                       "bounded_policy_equilibrium_gain": bounded_write/denominator},
            "rotating_key_bounded_policy": {"steps": steps, "peak_frobenius": peak,
                                           "invariant_frobenius_bound": 4.},
            "limitation": "admissible parameter counterexample; not an observed checkpoint instability"}


def concept_report(seed=9503):
    rng = np.random.default_rng(seed)
    evidence = np.tanh(rng.normal(size=(6,16)))
    prior = np.tanh(rng.normal(size=(6,16)))
    scores = rng.normal(size=(6,6))
    probs = np.exp(scores-scores.max(axis=1,keepdims=True))
    probs /= probs.sum(axis=1,keepdims=True)
    adjacency = .5*(probs+probs.T); np.fill_diagonal(adjacency,0)
    laplacian = np.diag(adjacency.sum(axis=1))-adjacency
    h = 1.25*np.eye(6)+.25*laplacian
    rhs = evidence+.25*prior
    posterior = np.linalg.solve(h,rhs)
    uniform_error = float(np.max(np.abs(posterior.mean(axis=0)-.8*rhs.mean(axis=0))))
    eigen = np.linalg.eigvalsh(h)
    # Vary only one undirected graph edge, holding evidence/pooling fixed.
    edge = np.zeros((6,6)); edge[0,0]=edge[1,1]=1; edge[0,1]=edge[1,0]=-1
    contrast = np.array([1.,-1.,0,0,0,0]); contrast /= np.linalg.norm(contrast)
    sensitivity = []
    delta = 1e-5
    changed = np.linalg.solve(h+.25*delta*edge,rhs)
    for departure in (0.,.01,.05):
        pool = np.full(6,1/6)+departure*contrast
        measured = float(np.linalg.norm(pool @ ((changed-posterior)/delta)))
        sensitivity.append({"pool_departure_l2": departure, "edge_directional_derivative_l2": measured})
    ok = uniform_error < 1e-12 and eigen[0] >= 1.25-1e-12 and np.max(np.abs(posterior)) <= 1+1e-12
    return {"status": "PASS" if ok else "FAIL", "evidence": "constructed_solve_fixture_and_laplacian_algebra",
            "min_eigenvalue": float(eigen[0]), "condition_number": float(eigen[-1]/eigen[0]),
            "condition_number_upper_bound": 2.2, "uniform_pool_graph_effect_max_abs": uniform_error,
            "bounded_score_max_slot_weight": math.exp(2)/(math.exp(2)+5),
            "sensitivity": sensitivity,
            "posterior_written_back": False,
            "limitation": "trained graph utility and parameter gradients require actual-source checkpoint diagnostics"}


def position_report():
    rows = []
    for clock in (0,128,1024,16384,1000000):
        log = math.log1p(clock)
        # Explicit illustrative coefficient and two lexical values.
        offset = .5*log
        bounded_offset = .5*log/(1+log)
        rows.append({"clock": clock,
                     "illustrative_lexical_difference": math.tanh(offset+.2)-math.tanh(offset-.2),
                     "bounded_position_lexical_difference": math.tanh(bounded_offset+.2)-math.tanh(bounded_offset-.2)})
    return {"status": "PASS", "evidence": "asymptotic_saturation_algebra_and_constructed_coefficient",
            "coefficient": .5, "rows": rows,
            "limitation": "checkpoint-dependent onset; surviving separate lexical-key path is not removed"}


def scan_report(seed=9504):
    rng = np.random.default_rng(seed)
    p = rng.normal(scale=.04,size=(2,65,8,8)) + .8*np.eye(8)
    q = rng.normal(scale=.03,size=(2,65,16,8))
    initial = rng.normal(size=(2,16,8))
    mask = rng.uniform(size=(2,65)) > .2
    reference = matrix_scan(p,q,initial=initial,write_mask=mask,algorithm="sequential")
    rows = []
    for algorithm in ("sequential","tree","chunked"):
        start = time.perf_counter()
        result = matrix_scan(p,q,initial=initial,write_mask=mask,algorithm=algorithm)
        elapsed = time.perf_counter()-start
        error = float(np.max(np.abs(result-reference)))
        rows.append({"algorithm": algorithm, "seconds_fixture_only": elapsed,
                     "max_abs_error": error, "status": "PASS" if error < 1e-11 else "FAIL"})
    return {"status": "PASS" if all(x["status"] == "PASS" for x in rows) else "FAIL",
            "evidence": "constructed_noncommuting_matrix_initial_state_mask_fixture",
            "input_and_output_array_bytes": p.nbytes+q.nbytes+initial.nbytes+mask.nbytes+reference.nbytes,
            "rows": rows, "limitation": "not model decode latency, process RAM, energy or a speedup claim"}


def build_report(*, checkpoint=None, inputs=None, receipt=None, model_kind="500k"):
    started = time.perf_counter()
    inventory = source_inventory()
    checks = {"retention": retention_report(), "linear_memory_load": memory_load_report(),
              "aligned_gain": aligned_gain_report(), "concept_graph": concept_report(),
              "position": position_report(), "scan": scan_report()}
    if any(x is not None for x in (checkpoint,inputs,receipt)):
        if any(x is None for x in (checkpoint,inputs,receipt)):
            raise ValueError("checkpoint, inputs and receipt must be supplied together")
        from checkpoints import audit_checkpoint
        checkpoint_result = audit_checkpoint(Path(checkpoint),Path(inputs),Path(receipt),model_kind=model_kind)
    else:
        checkpoint_result = {"status": "INCONCLUSIVE", "reason": "no authenticated checkpoint/input receipt supplied"}
    statuses = [x["status"] for x in inventory+list(checks.values())]
    if "FAIL" in statuses or checkpoint_result["status"].startswith(("FAIL", "BLOCKED")):
        status = "FAIL"
    elif any(x != "PASS" for x in statuses):
        status = "INCONCLUSIVE"
    else:
        status = "PASS_CONSTRUCTED_CHECKS"
    return {"schema": "tac-osm.langmath095.v1", "study": "LANGMATH095", "status": status,
            "evidence_level": "ENGINEERING_ONLY_NOT_CAPABILITY_OR_THEORY_VALIDATION",
            "source_commit": BASE_COMMIT, "source_inventory": inventory,
            "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                        "elapsed_seconds": time.perf_counter()-started},
            "checks": checks, "checkpoint_diagnostics": checkpoint_result,
            "open_obligations": ["independent real-source provenance", "trained causal persistence and future utility",
                                 "learned structure discovery, transfer and composition", "full acquisition/retrieval/verification costs",
                                 "CUDA AMP full-gradient parity", "PNDS Lean kernel build and composed graph verification"],
            "optimizer_steps": 0, "protected_test_opened": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--inputs",type=Path)
    parser.add_argument("--receipt",type=Path)
    parser.add_argument("--model-kind",choices=("500k","10m"),default="500k")
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        raise FileExistsError(args.output)
    report = build_report(checkpoint=args.checkpoint,inputs=args.inputs,receipt=args.receipt,model_kind=args.model_kind)
    text = json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("x",encoding="utf-8") as file:
            file.write(text)
        print(json.dumps({"status":report["status"],"output":str(args.output),"optimizer_steps":0}))
    else:
        print(text,end="")
    return 1 if report["status"] == "FAIL" or report["checkpoint_diagnostics"]["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
