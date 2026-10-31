"""Original-FIT full-graph numerical and state-boundary verification. No fits."""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import torch
from preflight import build, comparator_classes, counts, load_native, parameter_digest
from source import next_byte_batch, verify_package
import math_reference as reference
from probe_adapters import adapter_for
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"src"))
from tac_osm.research_probe import probe_dependencies, probe_entity_isolation, probe_memory

HERE = Path(__file__).resolve().parent


def checked_math():
    record = json.loads((HERE/"MATH_SOURCE_MAP.json").read_text())
    expected = record["mathematical_files_sha256"]["foundation/langcap103/reference.py"]
    if hashlib.sha256((HERE/"math_reference.py").read_bytes()).hexdigest() != expected:
        raise ValueError("Independent equation file changed")
    for path, expected in record["source_files_sha256"].items():
        if hashlib.sha256((HERE.parents[1]/path).read_bytes()).hexdigest() != expected:
            raise ValueError("Actual model source changed: "+path)
    return record


def compare(a, b, tolerance, name):
    if a.shape != b.shape or not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
        raise ValueError(name+": invalid shape or finite boundary")
    if not torch.allclose(a, b, atol=tolerance["atol"], rtol=tolerance["rtol"]):
        delta = float((a-b).abs().max())
        raise ValueError(f"{name}: pointwise mismatch, max abs={delta}")
    return float((a-b).abs().max()) if a.numel() else 0.


def source_forward(model, arm, tokens):
    result = model(tokens)
    if isinstance(result, tuple): return result
    return result, {}


def compare_case(model, config, tokens, dtype, tolerance):
    arm = config["arm"]
    actual, state = source_forward(model, arm, tokens)
    parameters = dict(model.named_parameters())
    predicted, refstate = reference.forward(arm, tokens, parameters, depth=config["depth"])
    forward_error = compare(actual, predicted, tolerance, "logits")
    state_error = 0.
    if arm == "tac_associative_stack":
        if state.tensors.keys() != refstate.keys(): raise ValueError("Native state schema differs")
        state_error = max(compare(state.tensors[k], refstate[k], tolerance, "state."+k) for k in refstate)
        serial, serial_state = model(tokens, parallel=False)
        compare(serial, actual, tolerance, "serial/parallel logits")
        for k in refstate: compare(serial_state.tensors[k], state.tensors[k], tolerance, "serial/parallel state")
        cut = 3
        left, carried = model(tokens[:, :cut]); right, final = model(tokens[:, cut:], state=carried)
        compare(torch.cat((left, right), 1), actual, tolerance, "native chunk carry")
        for k in refstate: compare(final.tensors[k], state.tensors[k], tolerance, "native chunk state")
    elif arm == "stacked_gru":
        state_error = compare(state, refstate, tolerance, "GRU hidden")
        cut = 3
        a, h = model.recurrent(model.embedding(tokens[:, :cut]))
        b, final = model.recurrent(model.embedding(tokens[:, cut:]), h)
        compare(model.head(torch.cat((a,b),1)), actual, tolerance, "GRU chunk carry")
        compare(final, state, tolerance, "GRU chunk state")
    weights = torch.linspace(.5, 1.5, actual.numel(), device=actual.device, dtype=actual.dtype).reshape_as(actual)
    ga = torch.autograd.grad((actual*weights).mean(), tuple(parameters.values()))
    gr = torch.autograd.grad((predicted*weights).mean(), tuple(parameters.values()))
    gradient_error = max(compare(a, b, tolerance, "parameter gradient."+name)
                         for name, a, b in zip(parameters, ga, gr))
    # Deliberately perturb the independent reference readout. An audit that
    # accepts this would be comparing a source path with itself or too loosely.
    readout = "head.out.bias" if arm == "tac_associative_stack" else "head.bias"
    wrong = dict(parameters); wrong[readout] = parameters[readout].detach().clone() + .05
    with torch.no_grad():
        bad, _ = reference.forward(arm, tokens, wrong, depth=config["depth"])
    try: compare(actual.detach(), bad, tolerance, "deliberately wrong readout")
    except ValueError: mutation_rejected = True
    else: raise ValueError("Source-equivalence falsifier failed: wrong output accepted")
    with torch.no_grad():
        changed = tokens.clone(); changed[:, -2:] = (changed[:, -2:]+97)%256
        future, _ = source_forward(model, arm, changed)
        compare(actual[:, :-2], future[:, :-2], tolerance, "actual-source future poison")
    return {"dtype": str(dtype), "length": tokens.shape[1], "batch": tokens.shape[0],
        "forward_max_abs": forward_error, "state_max_abs": state_error,
        "all_parameter_gradient_max_abs": gradient_error, "parameter_tensors_compared": len(parameters),
        "wrong_readout_rejected": mutation_rejected, "future_poison_invariance": "PASS"}


def probe_case(model, arm, tokens):
    sample = {"history": tokens[:, :4], "query": tokens[:, 4:5],
              "owners": tuple("original-fit-stream-"+str(i) for i in range(tokens.shape[0]))}
    adapter = adapter_for(arm, model)
    state = adapter.start(model, sample)
    with torch.no_grad():
        for t in range(4): state = adapter.advance(model, state, sample, t)
        actual = source_forward(model, arm, tokens[:, :5])[0][:, -1:]
        observed = adapter.forecast(model, state, sample)["next_byte_logits"]
        compare(actual, observed, {"atol":1e-8,"rtol":1e-6}, "adapter/source forecast")
    isolation = probe_entity_isolation(model, sample, adapter, tolerance=1e-8)
    if isolation["status"] != "PASS": raise ValueError("Actual-model entity isolation failed")
    # Mutation falsifier: add a real cross-batch term to the adapter forecast;
    # the architecture-independent isolation probe MUST detect the contamination.
    def bad_forecast(m, s, x):
        logits = adapter.forecast(m,s,x)["next_byte_logits"]
        return {"next_byte_logits":logits + x["query"].double().mean()/256}
    poisoned = probe_entity_isolation(model, sample, replace(adapter,forecast=bad_forecast), tolerance=1e-8)
    # First bytes could be equal in some original files; use a distinct query
    # for this negative control without changing any actual-source case above.
    if poisoned["status"] != "FAIL":
        negative = {**sample,"query":torch.arange(tokens.shape[0],device=tokens.device)[:,None]}
        poisoned = probe_entity_isolation(model, negative, replace(adapter,forecast=bad_forecast), tolerance=1e-8)
    if poisoned["status"] != "FAIL": raise ValueError("Cross-entity leakage falsifier was not detected")
    dependencies = probe_dependencies(model, sample, adapter)
    for head in dependencies["heads"].values():
        if any(v["connected"] is not True or v["finite"] is not True for v in head.values()):
            raise ValueError("Declared source dependency disconnected or nonfinite")
    if arm == "causal_transformer":
        memory = {"status":"INCONCLUSIVE_DISCRETE_EXTERNAL_HISTORY_UNSUPPORTED_BY_FLOAT_STATE_PROTOCOL",
            "external_prefix_tensor_bytes": state["prefix"].numel()*state["prefix"].element_size(),
            "observed_source_bytes":state["prefix"].numel(),"stateless_claim":False,
            "reason":"actual prefix is int64; no fabricated floating memory or zero-cache inference"}
    else:
        memory = probe_memory(model, sample, adapter)
        if not memory["query_does_not_write_state"]: raise ValueError("Forecast mutated caller state")
    return {"adapter":adapter.name,"adapter_equals_actual_source":True,"entity_isolation":isolation,
        "cross_entity_contamination_rejected":True,"dependencies":dependencies,"memory":memory,
        "candidate_permutation":"INCONCLUSIVE_NO_LEGAL_CANDIDATE_AXIS_DECLARED"}


def run(receipt_path, corpus_path, out):
    torch.set_num_threads(2)
    mapping = checked_math(); source, data = verify_package(receipt_path, corpus_path)
    x, _, ids = next_byte_batch(source,data,length=9,batch=2)
    tokens = torch.tensor(x,dtype=torch.long)
    original = json.loads((HERE.parents[0]/"langcap103_results/PROFILE_RUN_41219856043.json").read_text())
    native = load_native(); classes = comparator_classes(torch); reports=[]
    for cohort in original["cohorts"]:
        for record in cohort["arms"]:
            config=record["configuration"]; cases=[]
            for dtype_name, dtype in (("float32",torch.float32),("float64",torch.float64)):
                torch.manual_seed(10301)
                model=build(config,record["target"],native,classes).to(dtype=dtype).eval()
                if counts(model)["trainable_parameters"] != record["trainable_parameters"]:
                    raise ValueError("Model geometry changed")
                before=parameter_digest(model)
                for length in mapping["numerical_plan"]["prefix_lengths"]:
                    cases.append(compare_case(model,config,tokens[:,:length],dtype,mapping["tolerances"][dtype_name]))
                if dtype == torch.float64: probes=probe_case(model,config["arm"],tokens)
                if parameter_digest(model) != before: raise ValueError("Verification changed actual weights")
                del model
            reports.append({"configuration":config,"target":record["target"],"cases":cases,"probes":probes})
    report={"schema":"langcap103.actual-source-equation-probe.v1","status":"FINITE_SOURCE_TESTS_PASSED_GLOBAL_PROOFS_OPEN",
        "torch_version":torch.__version__,"source_sha256":source["source_sha256"],"original_fit_rows":ids,
        "numerical_and_probe_reports":reports,"optimizer_updates":0,"dev_scoring":False,
        "protected_test_access":False,"global_ieee_source_equivalence_proved":False,"training_authorized":False,
        "limits":["untrained parameters and finite original-FIT prefixes only; no trained capability claim",
            "shared Torch primitive arithmetic kernels, independent graph equations",
            "Transformer discrete external-memory surgery remains unsupported by floating-state protocol",
            "candidate permutation has no declared legal axis; no fabricated PASS",
            "full FLOPs, trusted independent source/reward/ACL receipt and permitted two-T4 execution remain separate"]}
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({"status":report["status"],"models":len(reports),"full_graph_forward_backward_cases":sum(len(r['cases']) for r in reports),"optimizer_updates":0,"training_authorized":False}))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--receipt',required=True,type=Path)
    p.add_argument('--corpus',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();run(a.receipt,a.corpus,a.out)
