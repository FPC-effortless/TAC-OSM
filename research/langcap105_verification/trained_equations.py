"""Bounded trained-source/PNDS equation checks; no fitting or heldout scoring."""
import argparse
import json
from pathlib import Path
import sys

from checkpoints import SOURCE_HASHES, PROTOCOL_SHA, digest, require

HELPER_COMMIT = "0b21f0feed5d6f89bb49218d8ca320cca6220850"
HELPER_HASHES = {
    "research/langcap103/math_reference.py": "20950b068ffac7450539fe381640fbdf3286af54f9f02001715317eb0b5fdcb4",
    "research/langcap103/verify_graphs.py": "8e8ff91bebb1bf6e69e0686448a2013dc72050c2b0a0bba7c8bd14f35df97a25",
    "research/langcap103/probe_adapters.py": "1a7f841611fd9d45768d8da4a5869fa3b7989654e768ab3392d582e46efa957a",
    "research/langcap103/MATH_SOURCE_MAP.json": "01c6caf146fc9ab037991704622983ae07f707cf16da4480029ec0180abbbc4e",
    "src/tac_osm/research_probe.py": "32e75366b0e2fbdd301f7061381bb355cb7522733678a3743ad9b24e2a1226c8",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--helpers-root", required=True, type=Path)
    parser.add_argument("--fits", required=True, type=Path)
    parser.add_argument("--fit-inputs", required=True, type=Path)
    parser.add_argument("--checkpoint-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "Do not overwrite previous equation audit")
    audit = json.loads(args.checkpoint_audit.read_text())
    accepted = {(m["scale"], m["arm"], m["seed"]): m["checkpoint_sha256"] for m in audit["models"]}
    for p, expected in HELPER_HASHES.items():
        require(digest(args.helpers_root / p) == expected, "Equation helper changed: " + p)
    for p, expected in SOURCE_HASHES.items():
        require(digest(args.source_root / p) == expected, "Original training code changed: " + p)
        if "/langcap103/" in p and not p.endswith("source.py"):
            require(digest(args.helpers_root / p) == expected, "Equation helper model source differs")
    protocol_path = args.source_root / "research/langcap105/OWNER_PROTOCOL.json"
    require(digest(protocol_path) == PROTOCOL_SHA, "Original protocol differs")
    protocol = json.loads(protocol_path.read_text())
    sys.path.insert(0, str(args.source_root / "research/langcap105"))
    import train_owner as owner
    data = owner.verify_source(next(args.fit_inputs.rglob("SOURCE_RECEIPT.json")),
                               next(args.fit_inputs.rglob("FIT_ONLY.tar")), protocol)
    sys.path.insert(0, str(args.helpers_root / "research/langcap103"))
    import verify_graphs as graphs
    import torch
    require(torch.__version__ == "2.15.0+cpu", "Original CPU Torch version required")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    graphs.checked_math()
    names = sorted(data)[:2]
    tokens = torch.tensor([list(data[n][:9]) for n in names], dtype=torch.long)
    reports = []; seen = set()
    for path in sorted(args.fits.rglob("FIT_RECEIPT.json")):
        v = json.loads(path.read_text()); key = (v["scale"], v["arm"], v["seed"])
        require(key in accepted and key not in seen, "Unaudited or duplicate checkpoint")
        require(digest(path.parent / "TRAINED.pt") == accepted[key], "Audited checkpoint changed")
        payload = torch.load(path.parent / "TRAINED.pt", map_location="cpu", weights_only=True)
        spec = next(m for m in protocol["models"][key[0]]["configurations"] if m["arm"] == key[1])
        config = {k: x for k, x in spec.items() if k != "trainable_parameters"}
        cases = []
        for dtype, tolerance in ((torch.float32, {"atol":1e-5, "rtol":1e-5}),
                                 (torch.float64, {"atol":1e-8, "rtol":1e-6})):
            model = graphs.build(config, protocol["models"][key[0]]["target_parameters"],
                                 graphs.load_native(), graphs.comparator_classes(torch)).to(dtype=dtype).eval()
            model.load_state_dict(payload["model"], strict=True)
            before = graphs.parameter_digest(model)
            cases.append(graphs.compare_case(model, config, tokens, dtype, tolerance))
            require(before == graphs.parameter_digest(model), "Equation audit changed trained parameters")
        seen.add(key)
        reports.append({"scale":key[0], "arm":key[1], "seed":key[2], "checkpoint_sha256":accepted[key], "cases":cases})
        print(json.dumps({"trained_graph":"PASS", "key":key}), flush=True)
    require(seen == set(accepted), "Not every tensor-audited checkpoint received equation verification")
    out = {"status":"ALL_AVAILABLE_TRAINED_GRAPHS_PASS_BOUNDED_NUMERICAL_CHECKS",
        "complete_18_fit_cohort":len(seen) == 18, "models":reports, "model_count":len(seen),
        "case_count":sum(len(m["cases"]) for m in reports), "helper_source_commit":HELPER_COMMIT,
        "helper_sha256":HELPER_HASHES, "original_training_run":41229365510, "torch_version":torch.__version__,
        "optimizer_updates":0, "dev_scoring":False, "protected_test_access":False,
        "limits":["Two original FIT prefixes of nine bytes only; no heldout capability evidence",
            "Separately formulated full graphs share Torch arithmetic primitives",
            "FP64 checks cast saved FP32 weights; they are not FP64-trained models",
            "Global IEEE equivalence and full mathematical release remain OPEN"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
