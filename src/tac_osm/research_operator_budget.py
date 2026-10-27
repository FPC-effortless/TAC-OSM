"""Conservative complete ATen-ledger parity, not invented exact FLOP totals.

Identical operator signatures cancel. All differing arithmetic is bounded
against common matmul work. Unknown differing operators block. Hardware
instruction/time/energy equality is not claimed.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def numel(shape):
    return math.prod(shape)


def signature_work(signature):
    name, inputs, outputs = json.loads(signature)
    name = name.split(".")[0]
    floating = [s for dtype, s in outputs if "float" in dtype or "complex" in dtype]
    n = sum(numel(s) for s in floating)
    if name == "aten::mm":
        a,b = [s for _,s in inputs[:2]]
        return 2*a[-2]*a[-1]*b[-1]
    if name == "aten::bmm":
        a,b = [s for _,s in inputs[:2]]
        return 2*a[0]*a[-2]*a[-1]*b[-1]
    if name == "aten::addmm":
        a,b = [s for _,s in inputs[-2:]]
        return 2*a[-2]*a[-1]*b[-1]+a[-2]*b[-1]
    if name in ("aten::add","aten::add_","aten::sub","aten::mul","aten::div","aten::neg","aten::abs"):
        return n
    if name in ("aten::sum","aten::mean"):
        return max(numel(inputs[0][1]), n)
    if name in ("aten::view","aten::_unsafe_view","aten::reshape","aten::slice","aten::slice_backward",
                "aten::slice_scatter","aten::squeeze","aten::unsqueeze","aten::expand","aten::permute",
                "aten::transpose","aten::detach","aten::alias","aten::clone","aten::copy_","aten::zeros_like"):
        return 0
    raise ValueError("unbounded differing arithmetic operator: " + name)


def ledger_counter(row):
    entries = row["operators"]
    keys = [(r["stage"], r["signature"]) for r in entries]
    if len(set(keys)) != len(keys) or any(type(r["calls"]) is not int or r["calls"] <= 0 for r in entries):
        raise ValueError("invalid/duplicate operator ledger rows")
    return Counter({k:r["calls"] for k,r in zip(keys,entries)})


def prove_pair(a, b):
    x,y = ledger_counter(a),ledger_counter(b)
    common = x & y
    matmul_floor = 0
    for (_,sig), count in common.items():
        name = json.loads(sig)[0].split(".")[0]
        if name in ("aten::mm","aten::bmm","aten::addmm"):
            matmul_floor += count*signature_work(sig)
    if matmul_floor <= 0:
        raise ValueError("common actual matmul work missing")
    unequal = []
    bound = 0
    for arm, delta in (("assoc",x-y),("fastslow",y-x)):
        for (stage,sig),count in sorted(delta.items()):
            cost = signature_work(sig)
            bound += count*cost
            unequal.append(dict(arm=arm,stage=stage,signature=sig,calls=count,arithmetic_upper_bound=count*cost))
    gap = bound/matmul_floor
    if gap > .01:
        raise ValueError("complete ATen arithmetic parity exceeds 1%")
    return dict(common_matmul_flops_lower_bound=matmul_floor,
        differing_arithmetic_upper_bound=bound,relative_full_math_gap_upper_bound=gap,
        complete_unequal_operators=unequal,
        scope="CPU FP32 ATen mathematical operator signatures; not hardware FLOPs or equal elapsed time")


def verify_profile(profile):
    if profile.get("schema") != "tac-osm.complete-operator-profile.v1":
        raise ValueError("complete original operator profile required")
    rows = profile["profiles"]
    seeds = sorted({r["seed"] for r in rows})
    if len(seeds) < 2 or len(rows) != len(seeds)*4:
        raise ValueError("two arms x two optimizer stages x >=2 seeds required")
    proofs = []
    for seed in seeds:
        for kind in ("first","warm"):
            pair = [r for r in rows if r["seed"]==seed and r["kind"]==kind]
            if len(pair)!=2 or {r["arm"] for r in pair}!={"assoc_historic_query","fast_slow_historic_query"}:
                raise ValueError("incomplete original profile matrix")
            for row in pair:
                if row["weights_before"] != row["weights_after"] or row["parameters"] != 99390 or row["finite_gradient_tensors"] != 43:
                    raise ValueError("prefit profile changed weights or failed live parameter/gradient checks")
                if row["operator_sha256"] != digest(row["operators"]):
                    raise ValueError("operator receipt fingerprint mismatch")
            pair.sort(key=lambda r:r["arm"])
            proofs.append(dict(seed=seed,kind=kind,**prove_pair(*pair)))
    return proofs


def validate_comparison(manifest, phase):
    """Alternate accounting with full per-step ledgers, preserving other gates."""
    root = Path.cwd().resolve()
    path = (root/manifest["operator_profile_path"]).resolve()
    if not path.is_relative_to(root) or path.is_symlink():
        raise ValueError("repository-owned operator profile required")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=manifest["operator_profile_sha256"]:
        raise ValueError("frozen operator profile changed")
    profile = json.loads(raw)
    proofs = verify_profile(profile)
    if profile["recipe_sha256"] != manifest["training_recipe_sha256"] or profile["source_sha256"] != manifest["real_world_source"]["sha256"]:
        raise ValueError("source/recipe operator binding mismatch")
    rows = manifest["arms"]
    seeds = sorted({r["seed"] for r in rows})
    if len(rows)!=len(seeds)*2 or len(seeds)<2:
        raise ValueError("matched two-arm seeded training matrix required")
    contracts = ("dataset_sha256","train_split_sha256","dev_split_sha256","test_split_sha256",
        "training_sample_order_sha256","feature_contract_sha256","target_contract_sha256",
        "optimizer_schedule_sha256","tuning_policy_sha256","selection_policy_sha256","evaluation_contract_sha256")
    budget_fields=("steps","examples_seen","tokens_seen","batch_size","history_events","queries_per_window","tuning_trials")
    for seed in seeds:
        pair=[r for r in rows if r["seed"]==seed]
        if len(pair)!=2 or {r["architecture"] for r in pair}!={"assoc_historic_query","fast_slow_historic_query"}:
            raise ValueError("missing/duplicate seeded architecture")
        for field in contracts:
            if pair[0][field]!=pair[1][field] or not all(isinstance(r[field],str) and len(r[field])==64 and set(r[field])<=set('0123456789abcdef') for r in pair):
                raise ValueError("unmatched original data/budget contract: "+field)
        for row in pair:
            if row["dataset_sha256"] != manifest["real_world_source"]["sha256"]:
                raise ValueError("original source differs")
            for field in ("trainable_parameters","deployed_parameters"):
                if type(row[field]) is not int or not 95000<=row[field]<=105000:
                    raise ValueError("100k model band required")
            if row["nontrainable_state_bytes"]!=6800 or row["hardware_profile"]!="CPU-FP32-single-thread":
                raise ValueError("registered source state/hardware differs")
            budget=row["budget"]
            if any(budget[f]!=pair[0]["budget"][f] or type(budget[f]) is not int for f in budget_fields):
                raise ValueError("unmatched actual training exposure budget")
            if any(budget[f]<=0 for f in budget_fields if f!="tuning_trials") or budget["tuning_trials"]!=0:
                raise ValueError("positive fixed workload and no tuning required")
            if budget["examples_seen"]!=budget["steps"]*budget["batch_size"] or budget["tokens_seen"]!=budget["examples_seen"]*budget["history_events"]:
                raise ValueError("inconsistent source exposure counts")
            if phase=="final":
                m=row["measured"]
                expected=[]
                for step in range(budget["steps"]):
                    expected.append(next(p["operator_sha256"] for p in profile["profiles"] if p["seed"]==seed and p["arm"]==row["architecture"] and p["kind"]==("first" if step==0 else "warm")))
                if m["actual_operator_step_sha256"]!=expected or m["actual_steps"]!=budget["steps"] or m["actual_sample_order_sha256"]!=row["training_sample_order_sha256"]:
                    raise ValueError("actual full operator/data/step receipts deviate from prefit")
                for f in ("trainable_parameters","deployed_parameters"):
                    if m[f]!=row[f]:raise ValueError("postfit model size differs")
                if not isinstance(m.get("checkpoint_sha256"),str) or len(m["checkpoint_sha256"])!=64:
                    raise ValueError("actual replayable checkpoint digest required")
        for f in ("trainable_parameters","deployed_parameters"):
            if abs(pair[0][f]-pair[1][f])/max(pair[0][f],pair[1][f])>.01:
                raise ValueError("trained parameter gap exceeds 1%")
    return dict(study_id=manifest["study_id"],phase=phase,status="PASS_COMPLETE_OPERATOR_LEDGER",
        eligible_for_architecture_superiority=False,
        maximum_relative_math_compute_gap=max(p["relative_full_math_gap_upper_bound"] for p in proofs),
        exact_hardware_training_flops="NOT_CLAIMED",actual_step_ledgers_required=True)
