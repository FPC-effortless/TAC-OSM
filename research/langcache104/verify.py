"""Bounded no-fit source/gradient fixtures. Never authorizes scientific fitting."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.langcache104.model import Config104, DeepCacheTAC104
from research.langcache104.probe_adapter import adapter_for, state_views
from tac_osm.research_probe import probe_dependencies, probe_entity_isolation


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def numeric_case(dtype):
    torch.manual_seed(104)
    config = Config104(d_model=16, n_layers=2, n_heads=2, key_dim=4, value_dim=4,
                       write_rank=2, ffn_dim=24, window=5, chunk_size=3)
    model = DeepCacheTAC104(config).to(dtype=dtype).eval()
    before = parameter_digest(model)
    tokens = torch.arange(22, dtype=torch.long).reshape(2, 11)*7 % 256
    owners = ("constructed-stream-A", "constructed-stream-B")
    chunk, chunk_state = model(tokens, owners=owners, mode="chunk")
    serial, serial_state = model(tokens, owners=owners, mode="recurrent")
    tolerance = 3e-5 if dtype == torch.float32 else 3e-10
    torch.testing.assert_close(chunk, serial, atol=tolerance, rtol=tolerance)
    state_error = max(float((x-state_views(serial_state)[name]).abs().max().detach())
                      if x.numel() else 0. for name, x in state_views(chunk_state).items())
    weight = torch.linspace(.7, 1.3, chunk.numel(), dtype=dtype).reshape_as(chunk)
    names, parameters = zip(*model.named_parameters())
    ga = torch.autograd.grad((chunk*weight).mean(), parameters)
    gb = torch.autograd.grad((serial*weight).mean(), parameters)
    gradients = []
    for name, actual, expected in zip(names, ga, gb):
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        if not bool(torch.isfinite(actual).all()):
            raise ValueError("nonfinite gradient at "+name)
        gradients.append({"name": name, "connected": True,
                          "finite": True, "nonzero": bool((actual != 0).any()),
                          "max_absolute_parity_error": float((actual-expected).abs().max())})
    if not all(row["nonzero"] for row in gradients):
        raise ValueError("zero parameter-tensor gradient in declared active source pathway")
    probe = None
    if dtype == torch.float64:
        sample = {"history": tokens[:, :4], "query": tokens[:, 4:5], "owners": owners}
        adapter = adapter_for(model)
        isolation = probe_entity_isolation(model, sample, adapter, tolerance=1e-9)
        if isolation["status"] != "PASS":
            raise ValueError("actual-source entity isolation failed")
        probe = {"adapter": adapter.name, "entity_isolation": isolation,
                 "input_state_dependencies": probe_dependencies(model, sample, adapter),
                 "candidate_permutation": "INCONCLUSIVE_NO_LEGAL_CANDIDATE_AXIS"}
    if before != parameter_digest(model):
        raise ValueError("No-fit verification changed source parameters or buffers")
    return {"dtype": str(dtype), "configuration": asdict(config), "counts": model.counts(),
            "parameter_and_buffer_sha256": before, "unchanged_parameters_and_buffers": True,
            "logit_max_abs_error": float((chunk-serial).abs().max().detach()),
            "state_max_abs_error": state_error, "all_parameter_gradients": gradients,
            "state_tensor_bytes_batch2_after_11_bytes": chunk_state.state_bytes(),
            "probe": probe}


def run(out):
    torch.set_num_threads(2)
    test_paths = ["research/langcache104/test_kernels.py", "research/langcache104/test_model.py",
                  "research/langcache104/test_exact_store.py", "research/langcache104/test_config_state.py",
                  "research/langcache104/test_math_reference.py", "research/langcache104/test_math_correspondence.py",
                  "research/langcache104/test_registry.py",
                  "tests/test_research_probe_agnostic.py"]
    with tempfile.TemporaryDirectory(prefix="langcache104-no-fit-") as directory:
        junit = Path(directory)/"junit.xml"
        result = subprocess.run([sys.executable, "-m", "pytest", "-q", *test_paths,
                                 "--junitxml", str(junit)], cwd=ROOT, text=True,
                                capture_output=True, timeout=300)
        if result.returncode:
            raise RuntimeError("Source fixture failure:\n"+result.stdout+result.stderr)
        xml = ET.parse(junit).getroot()
        suites = list(xml) if xml.tag == "testsuites" else [xml]
        tests = {name: sum(int(suite.get(name, 0)) for suite in suites)
                 for name in ("tests", "failures", "errors", "skipped")}
    files = ["model.py", "kernels.py", "exact_store.py", "probe_adapter.py",
             "test_model.py", "test_kernels.py", "test_exact_store.py", "test_config_state.py",
             "math_reference.py", "test_math_reference.py", "test_math_correspondence.py", "test_registry.py", "verify.py"]
    hashes = {"research/langcache104/"+name: hashlib.sha256(
        (ROOT/"research/langcache104"/name).read_bytes()).hexdigest() for name in files}
    report = {"schema": "tac-osm.langcache104-engineering.v1",
              "status": "PASS_FINITE_NO_FIT_FIXTURES_ONLY", "optimizer_updates": 0,
              "data_kind": "constructed_engineering_inputs_no_capability_scores",
              "dev_scoring": False, "protected_test_access": False,
              "torch_version": torch.__version__, "hardware": "CPU",
              "cuda_available": torch.cuda.is_available(),
              "gpu_speed_or_precision_qualification": "INCONCLUSIVE_NOT_EXECUTED",
              "tests": tests, "source_files_sha256": hashes,
              "cases": [numeric_case(torch.float32), numeric_case(torch.float64)],
              "training_authorized": False,
              "independent_full_graph_reference": {"forward_and_all_state": "PASS_TINY_FP64_COLD_AND_CONTINUED_MASKED_FIXTURES",
                  "finite_difference_gradients": "PASS_60_PARAMETER_TENSOR_COORDINATES_AND_14_STATE_TENSOR_COORDINATES",
                  "scope": "SAMPLED_COORDINATES_NOT_ALL_ELEMENT_OR_ALL_GEOMETRY_PROOF"},
              "open_gates": {"trained_capability": "NOT_TRAINED",
                  "primary_real_source": "OPEN_NO_SOURCE_USED",
                  "full_graph_lean_correspondence": "OPEN",
                  "independent_signed_source_goal_reward_acl_attestation": "OPEN",
                  "matched_neural_competitors_and_compute": "OPEN",
                  "learned_exact_key_routing": "NOT_IMPLEMENTED_EXPLICIT_KEY_API_ONLY"},
              "limits": ["finite constructed source fixtures do not establish learned ability",
                  "shared PyTorch primitives; independently written recurrent operator oracles",
                  "full-model chunk/recurrent parity shares projections; separate plain-list graph checks tiny fixtures and sampled derivatives",
                  "no optimizer, DEV scoring, protected TEST, GPU benchmark or paid compute",
                  "exact store is an explicit observed-record API, separate from neural logits"]}
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"status": report["status"], "tests": tests,
                      "optimizer_updates": 0, "report": str(out)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
