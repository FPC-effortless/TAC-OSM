"""No-fit T4 smoke workload, one isolated frozen seed per physical T4.

Not a neural model training run. Outputs constructed tensor arithmetic only.
"""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available() or args.device != "cuda:0":
        raise RuntimeError("isolated physical T4 must expose cuda:0 inside worker")
    name = torch.cuda.get_device_name(0)
    if "T4" not in name.upper():
        raise RuntimeError("non-T4 GPU forbidden")
    torch.manual_seed(args.seed)
    tensor = torch.arange(1024, device=args.device, dtype=torch.float32)
    scalar = float((tensor * tensor).sum().item())
    torch.cuda.synchronize()
    row = {"schema": "tac-osm.no-fit-t4-smoke.v1", "seed": args.seed,
           "physical_slot": int(__import__("os").environ["TAC_OSM_PHYSICAL_T4_SLOT"]),
           "device": name, "numeric_diagnostic": scalar,
           "source_provenance": "constructed_tensor_only",
           "model_training": False, "capability_claim_authorized": False}
    args.output.write_text(json.dumps(row, indent=2)+"\n")

if __name__ == "__main__":
    main()
