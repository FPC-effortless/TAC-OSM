"""Scalar-only source and no-fit scope check. Never imports Torch."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def run():
    protocol = json.loads((HERE/"PROTOCOL.json").read_bytes())
    require(protocol["schema"] == "flock013.no-fit-diagnosis.v1", "unknown protocol")
    require(protocol["optimizer_updates_allowed"] == 0 and protocol["max_neural_forwards"] == 100, "no-fit budget changed")
    require(protocol["protected_TEST_reads_allowed"] == 0 and protocol["heldout_queries_allowed"] == 0, "data boundary changed")
    require(protocol["seed"] == 18001 and protocol["max_CPU_seconds"] == 290 and protocol["max_wall_seconds"] == 300, "scope changed")
    for name, expected in protocol["code_sha256"].items():
        raw = (HERE/name).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == expected, "changed diagnostic source: "+name)
        if name.endswith(".py"):
            ast.parse(raw, filename=name)
    old = REPO/"research/local_binding_owner_fixed_20261103"
    raw = (old/"SOURCE_MANIFEST.json").read_bytes()
    require(hashlib.sha256(raw).hexdigest() == "5aa2c227e4ee3a58eadd96260fd74d7409a5ced56a1db1f86b99797794589787", "wrong original source manifest")
    manifest = json.loads(raw)
    for name, row in manifest["files"].items():
        require(hashlib.sha256((REPO/name).read_bytes()).hexdigest() == row["sha256"], "changed original source: "+name)
    receipt = {"status": "PASS_SOURCE_ONLY_NO_FIT_SCOPE", "original_source_files": len(manifest["files"]), "new_code_files": len(protocol["code_sha256"]), "protocol_sha256": hashlib.sha256((HERE/"PROTOCOL.json").read_bytes()).hexdigest(), "model_constructors": 0, "optimizer_updates": 0, "review": "root review; no independent reviewer or full graph/AD/IEEE qualification claimed"}
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    run()
