"""Reconstruct frozen source/fixtures and report admission; never imports Torch."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    source = json.loads((HERE/"SOURCE_MANIFEST.json").read_bytes())
    for name, row in source["files"].items():
        raw = (REPO/name).read_bytes()
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("Original/repair source differs: "+name)
        target = HERE/"source"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    for name in ("admission", "core", "fixtures", "runner", "audit_scorer"):
        load(name, HERE/"code"/(name+".py"))
    fixtures = sys.modules["fixtures"]
    data = fixtures.generate()
    data["INPUT_AUDIT.json"] = fixtures.audit(data)
    data["SCORER_UNIT_TESTS.json"] = sys.modules["audit_scorer"].test()
    protocol = json.loads((HERE/"PROTOCOL.json").read_bytes())
    (HERE/"fixtures").mkdir(exist_ok=True)
    for name, value in data.items():
        raw = (json.dumps(value, indent=2)+"\n").encode()
        if hashlib.sha256(raw).hexdigest() != protocol["fixture_files"][name]["sha256"]:
            raise ValueError("Fixture identity differs: "+name)
        (HERE/"fixtures"/name).write_bytes(raw)
    grant = json.loads((HERE/"REQUEST_TEMPLATE.json").read_bytes())
    result = sys.modules["admission"].inspect(HERE, protocol, grant)
    if "torch" in sys.modules:
        raise RuntimeError("Source bootstrap imported Torch")
    receipt = {"source_files": len(source["files"]), "fixtures": len(data), "admission": result, "model_constructors": 0, "optimizer_updates": 0, "torch_imported": False}
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    main()
