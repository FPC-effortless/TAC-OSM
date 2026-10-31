"""Exact owner-requested LANGMATH101 two-lane INTERNAL program gate.

This is not a relaxation of the ordinary fully matched architecture validator.
Counted dense arithmetic is a declared, audited metric, never complete FLOPs.
Source metadata and hashes are bindings, not independent source authentication.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

STUDY = "LANGMATH101"
CONTRACT = "owner-2026-10-31-langmath100-two-lane"
QUOTE = ("Compare A0 against A3 at nominal depths of 1, 2, 4 and 8 blocks, "
         "first under matched parameters and then under matched training FLOPs.")
LANES = ("parameter_exposure_matched", "counted_arithmetic_matched")
SHA = re.compile(r"^[0-9a-f]{64}$")
DATA_FIELDS = ("dataset_sha256", "train_split_sha256", "dev_split_sha256",
               "feature_contract_sha256", "target_contract_sha256",
               "optimizer_schedule_sha256", "tuning_policy_sha256",
               "selection_policy_sha256", "evaluation_contract_sha256")
EXPOSURE_FIELDS = ("steps", "examples_seen", "tokens_seen", "dense_compute_tokens")


def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(v):
    return hashlib.sha256(canonical(v)).hexdigest()


def need(v, message):
    if not v:
        raise ValueError(message)


def sha(v, name):
    need(isinstance(v, str) and SHA.fullmatch(v), name + " needs SHA256")


def integer(v, name, *, zero=False):
    need(type(v) is int and v >= (0 if zero else 1), name + " needs integer")


def number(v, name):
    need(not isinstance(v, bool) and isinstance(v, (int, float)) and
         math.isfinite(v) and v > 0, name + " needs positive finite number")


def within_one_percent(values, name):
    need(values and min(values) > 0, name + " absent")
    need((max(values)-min(values))/max(values) <= .01,
         name + " exceeds owner lane's one-percent tolerance")


def counts(row):
    for key in ("trainable_parameters", "deployed_parameters"):
        integer(row.get(key), key)
        need(95000 <= row[key] <= 105000, "prospective100K band required")
    need(row["deployed_parameters"] >= row["trainable_parameters"],
         "deployed learned count below trainable count")


def _repo_path(root, name):
    need(isinstance(name, str) and name and not Path(name).is_absolute(),
         "repository-relative file required")
    p = root / name
    need(not p.is_symlink() and p.resolve().is_relative_to(root),
         "source/profile path escapes repository")
    return p


def counted_dense_units(rows):
    """Charge actual mm/addmm/bmm arithmetic from recorded ATen tensor shapes.

    Units: multiply and add count separately. Elementwise, nonlinear, fused,
    attention, indexing, transfer and optimizer work remains explicitly unknown.
    The full operator ledger (including uncharged rows) is fingerprinted.
    """
    total = 0
    uncharged = set()
    seen = set()
    need(isinstance(rows, list) and rows, "complete actual ATen ledger required")
    for row in rows:
        need(isinstance(row, dict), "operator row must be object")
        sig = row.get("signature")
        integer(row.get("calls"), "operator calls")
        need(isinstance(sig, str) and isinstance(row.get("stage"), str),
             "operator stage/signature required")
        key = row["stage"], sig
        need(key not in seen, "duplicate operator signature")
        seen.add(key)
        name, inputs, outputs = json.loads(sig)
        # AdamW emits these dispatch-visible, tensor-free profiling markers.
        # Retain their signatures in the fingerprint; never admit an arbitrary
        # custom namespace as uncharged computation.
        marker = name in ("profiler::_record_function_enter_new", "profiler::_record_function_exit")
        need(isinstance(name, str) and (name.startswith("aten::") or marker),
             "ATen operator or explicit profiling marker required")
        if marker:
            need(inputs == [] and outputs == [], "profiling marker must be tensor-free")
        for tensors in (inputs, outputs):
            need(isinstance(tensors, list), "tensor signature list required")
            for dtype, shape in tensors:
                need(isinstance(dtype, str) and isinstance(shape, list) and
                     all(type(n) is int and n >= 0 for n in shape),
                     "actual tensor shape signature required")
        op = name.split(".")[0]
        charge = None
        if op in ("aten::mm", "aten::addmm"):
            aa, bb = [s for _, s in (inputs[:2] if op == "aten::mm" else inputs[-2:])]
            need(len(aa) == len(bb) == 2 and aa[-1] == bb[-2], "matmul shape mismatch")
            charge = 2*aa[-2]*aa[-1]*bb[-1]
            if op == "aten::addmm":
                charge += aa[-2]*bb[-1]
        elif op == "aten::bmm":
            aa, bb = [s for _, s in inputs[:2]]
            need(len(aa) == len(bb) == 3 and aa[0] == bb[0] and aa[-1] == bb[-2],
                 "batched matmul shape mismatch")
            charge = 2*aa[0]*aa[-2]*aa[-1]*bb[-1]
        if charge is None:
            uncharged.add(op)
        else:
            total += charge*row["calls"]
    need(total > 0, "actual counted dense arithmetic missing")
    return {"counted_dense_units": total, "operators_sha256": digest(rows),
            "uncharged_operator_names": sorted(uncharged),
            "complete_model_FLOPs": "INCONCLUSIVE_NOT_CLAIMED",
            "hardware_instruction_FLOPs": "NOT_MEASURED"}


def validate_program(manifest, phase="preflight", *, root=None):
    root = Path(root or Path.cwd()).resolve()
    need(manifest.get("study_id") == STUDY and manifest.get("study_type") == "owner_two_lane_program"
         and manifest.get("owner_contract") == CONTRACT,
         "exact LANGMATH101 owner two-lane contract required")
    need(manifest.get("owner_instruction_quote") == QUOTE, "exact owner instruction must be recorded")
    need(manifest.get("execution_level") == "internal_research", "owner lanes are internal FIT/DEV only")
    need(phase in ("preflight", "final"), "unknown phase")
    need(manifest.get("registered_before_first_fit") is True, "registration must precede fitting")
    need(manifest.get("no_architecture_superiority_claim") is True,
         "split lanes do not authorize full matched superiority")
    for flag in ("protected_TEST_access", "independently_qualified", "checkpoint_promotion"):
        need(manifest.get(flag) is False, flag + " is not authorized by this contract")
    source = manifest.get("real_world_source", {})
    sha(source.get("sha256"), "real source")
    need(source.get("observed_real_outcomes") is True, "actual original observations required")
    for key in ("publisher_url", "license", "license_evidence_url", "source_receipt"):
        need(isinstance(source.get(key), str) and source[key].strip(), "source record missing: " + key)
    need(manifest.get("arithmetic_convention") == "actual_ATen_mm_addmm_bmm_units",
         "only declared counted arithmetic convention supported")
    need(manifest.get("full_model_FLOPs_status") == "INCONCLUSIVE_NOT_CLAIMED",
         "partial dense counts must not be called complete FLOPs")
    need(isinstance(manifest.get("comparison_scope"), str) and len(manifest["comparison_scope"]) >= 20,
         "declare unequal information/exposure/compute dimensions")
    sha(manifest.get("operator_profile_sha256"), "operator profile")
    p = _repo_path(root, manifest.get("operator_profile_path"))
    raw = p.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == manifest["operator_profile_sha256"], "frozen profile changed")
    profile = json.loads(raw)
    need(profile.get("schema") == "langmath101.owner-two-lane-profile.v1", "actual prefit profile schema required")
    need(profile.get("optimizer_steps") == 0, "prefit profile must not update optimizer")
    need(profile.get("source_sha256") == source["sha256"], "profile source mismatch")
    sha(manifest.get("training_recipe_sha256"), "registered training recipe")
    recipe=profile.get("training_recipe")
    need(isinstance(recipe, dict) and recipe and digest(recipe)==manifest["training_recipe_sha256"]
         and profile.get("training_recipe_sha256")==manifest["training_recipe_sha256"],
         "actual profiled recipe differs from frozen source program")
    source_hashes = profile.get("source_hashes")
    need(isinstance(source_hashes, dict) and source_hashes, "actual executable source bindings required")
    for name, expected in source_hashes.items():
        sha(expected, "implementation source")
        need(hashlib.sha256(_repo_path(root, name).read_bytes()).hexdigest() == expected,
             "actual executable changed: " + name)
    sha(profile.get("implementation_sha256"), "implementation")
    need(digest(source_hashes) == profile["implementation_sha256"], "implementation bindings mismatch")
    rows = manifest.get("arms")
    need(isinstance(rows, list) and len(rows) >= 8, "both lanes x >=two architectures x >=two seeds required")
    keys = [(r.get("lane"), r.get("architecture"), r.get("seed")) for r in rows]
    need(len(set(keys)) == len(keys), "duplicate lane/model/seed")
    seeds = sorted({r["seed"] for r in rows})
    names = sorted({r["architecture"] for r in rows})
    need(len(seeds) >= 2 and len(names) >= 2, "controlled seed/architecture matrix required")
    tokenizers=profile.get("tokenizer_receipts_sha256")
    need(isinstance(tokenizers,dict) and set(tokenizers)=={str(s) for s in seeds},
         "actual tokenizer receipt byte bindings required for every seed")
    for seed in seeds:
        sha(tokenizers[str(seed)],"actual tokenizer receipt")
    expected = {(l, n, s) for l in LANES for n in names for s in seeds}
    need(set(keys) == expected, "complete two-lane matrix required")
    pp = profile.get("models")
    need(isinstance(pp, list), "actual profiled model rows required")
    profiles = {(r.get("architecture"), r.get("seed")): r for r in pp}
    need(len(profiles) == len(pp) and set(profiles) == {(n, s) for n in names for s in seeds},
         "profile model/seed matrix differs")
    schedule_records = manifest.get("source_schedules")
    need(isinstance(schedule_records, dict) and set(schedule_records) == {str(s) for s in seeds},
         "all frozen common source schedules required")
    schedules = {}
    for seed in seeds:
        spec = schedule_records[str(seed)]
        sha(spec.get("sha256"), "common schedule file")
        schedule_raw = _repo_path(root, spec.get("path")).read_bytes()
        need(hashlib.sha256(schedule_raw).hexdigest() == spec["sha256"], "frozen common schedule changed")
        schedule = json.loads(schedule_raw)
        need(isinstance(schedule, list) and len(schedule) == 4096 and
             all(isinstance(item, dict) and item for item in schedule),
             "exactly4096 source batch-ID entries must be frozen before fitting")
        schedules[seed] = schedule
    reports = []
    for row in rows:
        counts(row)
        integer(row["seed"], "seed")
        need(isinstance(row["architecture"], str) and len(row["architecture"]) >= 2,
             "architecture identity required")
        for key in DATA_FIELDS + ("training_sample_order_sha256",):
            sha(row.get(key), key)
        need(row["dataset_sha256"] == source["sha256"], "actual source differs")
        pr = profiles[row["architecture"], row["seed"]]
        counts(pr)
        need((pr["trainable_parameters"], pr["deployed_parameters"]) ==
             (row["trainable_parameters"], row["deployed_parameters"]), "live profile counts differ")
        need(pr.get("weights_before") == pr.get("weights_after"), "prefit profile changed weights")
        sha(pr.get("weights_before"), "profile initial weights")
        integer(pr.get("parameter_tensor_count"), "parameter tensor count")
        need(pr.get("finite_gradient_tensor_count") == pr["parameter_tensor_count"],
             "all active parameter tensors need finite gradients")
        need(pr.get("input_shape") and all(type(n) is int and n > 0 for n in pr["input_shape"]),
             "actual profiled source shape missing")
        charged = counted_dense_units(pr.get("operators"))
        need(pr.get("counted_dense_units") == charged["counted_dense_units"], "profile arithmetic invented")
        need(pr.get("operator_sha256") == charged["operators_sha256"], "profile operator fingerprint differs")
        b = row.get("budget", {})
        for key in EXPOSURE_FIELDS + ("batch_size", "history_events"):
            integer(b.get(key), "budget." + key)
        need(pr["input_shape"] == [b["batch_size"], b["history_events"]],
             "prefit actual source shape differs from planned training geometry")
        for key in ("planned_counted_units", "max_counted_units"):
            number(b.get(key), "budget." + key)
        need(b["planned_counted_units"] <= b["max_counted_units"], "planned charged work exceeds ceiling")
        integer(b.get("tuning_trials"), "tuning trials", zero=True)
        need(b["tuning_trials"] == 0, "fixed program has no model-specific tuning")
        need(b["examples_seen"] == b["steps"]*b["batch_size"], "planned examples inconsistent")
        need(b["dense_compute_tokens"] == b["examples_seen"]*b["history_events"],
             "planned source geometry inconsistent")
        need(b["tokens_seen"] <= b["dense_compute_tokens"], "observed tokens exceed dense slots")
        sha(b.get("common_schedule_sha256"), "common source schedule")
        schedule=schedules[row["seed"]]
        need(b["common_schedule_sha256"]==digest(schedule), "schedule content binding differs")
        need((b["batch_size"],b["history_events"])==(4,272), "exactB4x272 source geometry required")
        dynamic = row["lane"] == "counted_arithmetic_matched"
        stop = b.get("dynamic_stop_rule")
        if dynamic:
            need(isinstance(stop,dict) and stop.get("kind")=="first_step_reaching_counted_target",
                 "counted lane must use the exact prospective stopping rule")
            number(stop.get("target_counted_units"), "frozen counted target")
            number(stop.get("max_single_step_counted_units"), "frozen single-step charge bound")
            integer(stop.get("min_steps"), "minimum stopping steps")
            need(stop.get("max_steps")==4096 and stop["min_steps"]>=200 and
                 stop["min_steps"]<=4096, "dynamic counted lane bounds must be200..4096")
            need(stop["max_single_step_counted_units"]<=.005*stop["target_counted_units"],
                 "a single actual step may not exceed0.5% of the frozen target")
            need(charged["counted_dense_units"]<=stop["max_single_step_counted_units"],
                 "actual prefit source charge already exceeds the stopping bound")
            need(b["max_counted_units"]>=stop["target_counted_units"]+stop["max_single_step_counted_units"],
                 "predeclared budget needs the bounded first-crossing headroom")
            need(stop["min_steps"]<=b["steps"]<=stop["max_steps"],
                 "prefit exposure estimate exceeds frozen stopping bounds")
            need(b["planned_counted_units"]==stop["target_counted_units"],
                 "counted lane planned metric is its frozen target, not assumed actual consumption")
        else:
            need(stop is None and b["steps"]==512,
                 "parameter/exposure lane is exactly512 steps, without adaptive stopping")
            need(row["training_sample_order_sha256"]==digest(schedule[:512]),
                 "fixed parameter lane must bind its exact512 source batches")
        if phase == "final":
            mm = row.get("measured", {})
            counts(mm)
            need((mm["trainable_parameters"], mm["deployed_parameters"]) ==
                 (row["trainable_parameters"], row["deployed_parameters"]), "postfit counts differ")
            for name in ("checkpoint_sha256", "initial_weights_sha256", "implementation_sha256",
                         "actual_sample_order_sha256", "actual_operator_journal_sha256"):
                sha(mm.get(name), name)
            need(mm["initial_weights_sha256"] == pr["weights_before"], "profile/training initialization differs")
            need(mm["implementation_sha256"] == profile["implementation_sha256"], "trained source changed")
            need(mm.get("training_recipe_sha256") == manifest["training_recipe_sha256"],
                 "actual trained recipe differs from frozen profile")
            for field in EXPOSURE_FIELDS:
                key = "actual_" + field
                integer(mm.get(key), key)
                if not dynamic:
                    need(mm[key] == b[field], "actual fixed-lane exposure differs from preregistration: " + field)
            actual_steps = mm["actual_steps"]
            need(actual_steps<=4096 and mm["actual_examples_seen"]==actual_steps*4 and
                 mm["actual_dense_compute_tokens"]==actual_steps*4*272 and
                 mm["actual_tokens_seen"]<=mm["actual_dense_compute_tokens"],
                 "actual runtime source exposure is inconsistent")
            actual_order=mm.get("sample_order")
            need(actual_order==schedule[:actual_steps] and
                 mm["actual_sample_order_sha256"]==digest(actual_order),
                 "actual training must consume a contiguous frozen schedule prefix")
            if not dynamic:
                need(mm["actual_sample_order_sha256"]==row["training_sample_order_sha256"],
                     "fixed-lane source order changed")
            number(mm.get("charged_operator_units"), "actual charged units")
            need(mm["charged_operator_units"] <= b["max_counted_units"], "actual counted budget exceeded")
            journal = mm.get("operator_steps")
            need(isinstance(journal, list) and len(journal) == actual_steps, "actual every-step ledger required")
            need(digest(journal) == mm["actual_operator_journal_sha256"], "operator journal fingerprint differs")
            ledgers = mm.get("operator_ledgers")
            need(isinstance(ledgers, dict) and ledgers,
                 "actual complete signature ledgers behind every charged step required")
            verified = {}
            for fingerprint, actual_rows in ledgers.items():
                sha(fingerprint, "actual ledger identity")
                item = counted_dense_units(actual_rows)
                need(item["operators_sha256"] == fingerprint, "actual ledger bytes changed")
                verified[fingerprint] = item["counted_dense_units"]
            for step in journal:
                sha(step.get("operator_sha256"), "actual step operator fingerprint")
                number(step.get("counted_dense_units"), "actual step counted units")
                need(verified.get(step["operator_sha256"]) == step["counted_dense_units"],
                     "step charge differs from its actual operator tensor-shape ledger")
                integer(step.get("observed_tokens"), "actual step observed tokens")
                need(step["observed_tokens"]<=4*272, "actual step observes more than source slots")
            need(sum(s["counted_dense_units"] for s in journal) == mm["charged_operator_units"],
                 "allocated budget is not actual consumed work")
            need(sum(s["observed_tokens"] for s in journal)==mm["actual_tokens_seen"],
                 "actual valid token exposure does not match every-step journal")
            if dynamic:
                target=stop["target_counted_units"];bound=stop["max_single_step_counted_units"]
                need(stop["min_steps"]<=actual_steps<=stop["max_steps"],
                     "counted lane failed its frozen runtime stopping bounds")
                need(all(s["counted_dense_units"]<=bound for s in journal),
                     "an actual step exceeded the predeclared0.5% target bound")
                total=mm["charged_operator_units"]
                need(total-journal[-1]["counted_dense_units"]<target<=total and
                     total<=target+bound,
                     "counted lane did not stop at the FIRST target crossing; no rollback or continuation")
            for field in ("optimizer_cpu_seconds", "peak_memory_bytes"):
                number(mm.get(field), field)
            need(isinstance(mm.get("hardware"), dict) and mm["hardware"].get("cpu") and
                 mm["hardware"].get("torch"), "actual hardware/software receipt required")
    for seed in seeds:
        for lane in LANES:
            group = [r for r in rows if r["seed"] == seed and r["lane"] == lane]
            for key in ("trainable_parameters", "deployed_parameters"):
                within_one_percent([r[key] for r in group], key)
            for key in DATA_FIELDS:
                need(len({r[key] for r in group}) == 1, "source/feature/evaluation mismatch: " + key)
            need(len({r["budget"]["common_schedule_sha256"] for r in group}) == 1,
                 "every arm must consume the same deterministic source schedule prefix")
            if lane == "parameter_exposure_matched":
                need(len({r["training_sample_order_sha256"] for r in group}) == 1,
                     "parameter/exposure lane source order differs")
                for field in EXPOSURE_FIELDS + ("batch_size", "history_events"):
                    need(len({r["budget"][field] for r in group}) == 1,
                         "parameter/exposure lane exposure differs: " + field)
            else:
                need(len({r["budget"]["max_counted_units"] for r in group}) == 1,
                     "compute lane common charged ceiling missing")
                need(len({r["budget"]["dynamic_stop_rule"]["target_counted_units"] for r in group})==1,
                     "counted lane target must be frozen and common to every architecture")
                within_one_percent([r["budget"]["planned_counted_units"] for r in group],
                                   "compute lane planned counted arithmetic")
                if phase == "final":
                    within_one_percent([r["measured"]["charged_operator_units"] for r in group],
                                       "compute lane ACTUALLY CONSUMED counted arithmetic")
            reports.append({"seed": seed, "lane": lane,
                            "matched_dimensions": ["parameters", "real_source", "evaluation", "selection"] +
                            (["ordered_exposure"] if lane == LANES[0] else ["counted_dense_arithmetic"]),
                            "unmatched_dimensions": ["actual_compute"] if lane == LANES[0] else ["training_exposure"]})
    return {"study_id": STUDY, "phase": phase, "status": "PASS_OWNER_TWO_LANE_INTERNAL",
            "model_rows": len(rows), "lane_reports": reports,
            "eligible_for_architecture_superiority": False,
            "all_three_way_match": "NOT_ESTABLISHED_SPLIT_LANE_DESIGN",
            "complete_model_FLOPs": "INCONCLUSIVE_NOT_CLAIMED",
            "independently_qualified": False, "protected_TEST_permitted": False,
            "source_authentication": "METADATA_AND_BYTE_BINDINGS_NOT_INDEPENDENT_ATTESTATION"}
